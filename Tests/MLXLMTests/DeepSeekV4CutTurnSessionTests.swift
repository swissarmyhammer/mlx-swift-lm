// Copyright © 2026 Apple Inc.
//
// A weight-free `ChatSession` test of the history render of a DeepSeek-V4 turn
// that stops on the token budget inside its reasoning (card `^mf1mwaw`).
//
// The DeepSeek-V4 prompt of a thinking turn ends with `<｜Assistant｜><think>`.
// A turn that stops before `</think>` leaves `<think>` + the partial reasoning
// in the caches. The encoder writes a past turn as `reasoning` + `</think>` +
// `content`, thus a session that records the partial reasoning as the content
// renders `<think></think>partial`, which is not the text the model wrote. The
// session records the partial text as the reasoning of the turn instead, and the
// render is `<think>partial</think><｜end▁of▁sentence｜>`: the reference render of
// `reasoning_content=partial, content=""`.
//
// The tokenizer below renders with the real `DeepSeekV4ChatEncoder`, through
// `DeepSeekV4EncodingTokenizer`, over a vocabulary of the encoder markers and
// single characters. The test compares the tokens the warm caches hold after
// the prefill of turn 2 with the prompt of a cold session from the full
// history, as `scratchpad/dsv4bench` compares the two on the real weights.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The history render of a DeepSeek-V4 turn that a ``ChatSession`` generated.
@Suite(.serialized)
struct DeepSeekV4CutTurnSessionTests {

    /// The markers of a DeepSeek-V4 prompt.
    private typealias Marker = DeepSeekV4ChatEncoder.SpecialToken

    /// The vocabulary: each encoder marker, then each character of the
    /// alphabet.
    private static let vocabulary = PieceTokenizer(
        markers: Marker.allMarkers, endOfTurn: Marker.endOfSentence)

    /// The first user turn.
    private static let firstPrompt = "My color is teal"

    /// The second user turn.
    private static let secondPrompt = "What is my color"

    /// The reasoning the model writes in turn 1.
    private static let reasoning = "The user says teal"

    /// The answer the model writes in turn 1 after it closes its reasoning.
    private static let answer = "Noted"

    /// The text of the generation prompt of a thinking turn.
    private static let thinkingTail = Marker.assistant + Marker.thinkStart

    /// One way turn 1 ends, and how the history then holds it.
    struct TurnEndCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The text the model writes in turn 1, after the prompt.
        let generated: String
        /// Whether the model ends turn 1 with the end-of-sentence token. When
        /// false, turn 1 stops on the token budget.
        let writesEndOfSentence: Bool
        /// The content of the assistant turn of the history of the cold
        /// session.
        let coldContent: String
        /// The reasoning of the assistant turn of the history of the cold
        /// session, or nil for none.
        let coldReasoning: String?
        /// The text that the render of turn 2 writes for turn 1, after the
        /// generation prompt of turn 1.
        let renderedTurn: String

        var testDescription: String { name }
    }

    /// The cases, one row for each way turn 1 ends.
    static let turnEndCases: [TurnEndCase] = [
        TurnEndCase(
            name: "stops inside its reasoning: the reasoning is recorded, the render closes it",
            generated: reasoning, writesEndOfSentence: false,
            coldContent: "", coldReasoning: reasoning,
            renderedTurn: reasoning + Marker.thinkEnd + Marker.endOfSentence),
        TurnEndCase(
            name: "the control: closes its reasoning: the content keeps the whole text",
            generated: reasoning + Marker.thinkEnd + answer, writesEndOfSentence: true,
            coldContent: reasoning + Marker.thinkEnd + answer, coldReasoning: nil,
            renderedTurn: reasoning + Marker.thinkEnd + answer + Marker.endOfSentence),
    ]

    /// The warm caches of turn 2 hold the cold prompt of turn 2, and that
    /// prompt holds turn 1 as the model wrote it.
    @Test(arguments: turnEndCases)
    func theWarmTurnAfterTurnOneHoldsTheColdPrompt(_ row: TurnEndCase) async throws {
        let endOfSentence = try #require(Self.vocabulary.convertTokenToId(Marker.endOfSentence))
        let written =
            Self.vocabulary.encode(text: row.generated, addSpecialTokens: false)
            + (row.writesEndOfSentence ? [endOfSentence] : [])
        let warmModel = Self.model(script: written + [endOfSentence])
        let warm = Self.session(model: warmModel, history: [], tokenBudget: written.count)

        _ = try await warm.completionInfo(respondingTo: Self.firstPrompt)
        let second = try #require(try await warm.completionInfo(respondingTo: Self.secondPrompt))
        let warmPrompt = Array(warmModel.attentionCacheTokens.prefix(second.totalPromptTokenCount))

        let coldModel = Self.model(script: [endOfSentence])
        let cold = Self.session(
            model: coldModel,
            history: [
                .user(Self.firstPrompt), .assistant(row.coldContent, reasoning: row.coldReasoning),
            ],
            tokenBudget: written.count)
        let coldInfo = try #require(try await cold.completionInfo(respondingTo: Self.secondPrompt))
        let coldPrompt = Array(
            coldModel.attentionCacheTokens.prefix(coldInfo.totalPromptTokenCount))

        #expect(
            warmPrompt == coldPrompt,
            """
            warm <<<\(Self.vocabulary.decode(tokenIds: warmPrompt))>>>
            cold <<<\(Self.vocabulary.decode(tokenIds: coldPrompt))>>>
            """)
        #expect(
            Self.vocabulary.decode(tokenIds: warmPrompt)
                == Self.prompt(withFirstTurn: row.renderedTurn))
    }

    /// A history that a caller builds with an answer alone renders that
    /// answer after a closed, empty think block, as before card `^mf1mwaw`.
    @Test func anAnswerOnlyHistoryRendersAsBefore() async throws {
        let endOfSentence = try #require(Self.vocabulary.convertTokenToId(Marker.endOfSentence))
        let model = Self.model(script: [endOfSentence])
        let session = Self.session(
            model: model, history: [.user(Self.firstPrompt), .assistant(Self.answer)],
            tokenBudget: 1)

        let info = try #require(try await session.completionInfo(respondingTo: Self.secondPrompt))

        let prompt = Array(model.attentionCacheTokens.prefix(info.totalPromptTokenCount))
        #expect(
            Self.vocabulary.decode(tokenIds: prompt)
                == Self.prompt(
                    withFirstTurn: Marker.thinkEnd + Self.answer + Marker.endOfSentence))
    }

    // MARK: - Helpers

    /// The text of the prompt of turn 2, with `firstTurn` as the text of turn
    /// 1 after the generation prompt of turn 1.
    private static func prompt(withFirstTurn firstTurn: String) -> String {
        Marker.beginOfSentence + Marker.user + firstPrompt + thinkingTail + firstTurn
            + Marker.user + secondPrompt + thinkingTail
    }

    /// A model that writes `script`, one token for each forward pass.
    private static func model(script: [Int]) -> ScriptedTokenModel {
        ScriptedTokenModel(
            script: script, finalToken: script.last ?? 0,
            vocabularySize: vocabulary.vocabularySize, toolCallFormat: .dsml)
    }

    /// A session of a DeepSeek-V4 model in thinking mode over `model`.
    private static func session(
        model: ScriptedTokenModel, history: [Chat.Message], tokenBudget: Int
    ) -> ChatSession {
        let tokenizer = DeepSeekV4EncodingTokenizer(wrapping: vocabulary)
        let configuration = ModelConfiguration(
            id: "deepseek-v4-cut-turn-test", toolCallFormat: .dsml,
            reasoningConfig: DeepSeekV4EncodingTokenizer.reasoningConfig)
        let context = ModelContext(
            configuration: configuration, model: model,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        let session = ChatSession(
            context, history: history,
            generateParameters: GenerateParameters(maxTokens: tokenBudget, temperature: 0))
        session.thinkingEnabled = true
        return session
    }
}
