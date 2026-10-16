// Copyright © 2026 Apple Inc.
//
// A weight-free `ChatSession` test of the history render of a Qwen 3.8 turn
// with thinking on (card `^z4aax33`).
//
// The Qwen 3.8 chat template ends a prompt with `<think>\n` when thinking is on,
// and writes a past assistant turn as `<think>\n` + `reasoning_content` +
// `\n</think>\n\n` + `content`. A session that records the whole generated text
// as `content` writes `<think>\n\n</think>\n\n` + the reasoning for that turn in
// the next render. The tokenizer writes `\n\n` as one token, thus that render
// parts from the cached tokens inside `<think>\n`, and the cache cannot keep the
// tokens the model generated. Card `^95mp4sw` measured 73 tokens of prefill
// again on each turn at 32k.
//
// The tokenizer below writes that template shape for the messages it gets, and
// the scripted model writes the tokens of a turn with reasoning.
//
// A turn that stops inside its reasoning leaves no `</think>` in the cache, but
// the render of the next turn closes the think block (card `^ef2zynk`). The
// second test runs a hybrid model and examines where the next turn starts to
// feed the caches, and whether the caches then hold the cold render.
//
// The thinking switch of the session (`ChatSession.thinkingEnabled`, card
// `^2fm0eaq`) applies the prompt strategy of the reasoning protocol. For the
// Qwen 3.5 protocol, thinking off keeps the template flag on and ends the prompt
// with the closed think block. The switch tests examine the render that the
// template gets, the prompt that the model gets, and the cache reuse of the
// next turn.

import Foundation
import MLX
import Testing
import os

@testable import MLXLLM
@testable import MLXLMCommon

/// The history render of a turn that a ``ChatSession`` of a Qwen 3.8 model
/// generated.
@Suite(.serialized)
struct ChatSessionReasoningHistoryTests {

    /// How a test tells the session to think: the template flag in the
    /// additional context, the thinking switch of the session, or both.
    struct ThinkingSetting: Sendable {
        /// The template flag in the additional context, or nil for no flag.
        let templateFlag: Bool?
        /// The thinking switch of the session, or nil when the switch is not set.
        let sessionSwitch: Bool?

        /// The template flag `enabled`, with no session switch.
        static func flag(_ enabled: Bool) -> Self {
            ThinkingSetting(templateFlag: enabled, sessionSwitch: nil)
        }

        /// The session switch `enabled`, with no template flag.
        static func sessionSwitch(_ enabled: Bool) -> Self {
            ThinkingSetting(templateFlag: nil, sessionSwitch: enabled)
        }
    }

    /// One way a session can record the turn the model generated.
    struct HistoryCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The reasoning protocol of the model.
        let reasoningConfig: ReasoningConfig
        /// How the session asks for thinking.
        let thinking: ThinkingSetting
        /// The text the model writes in turn 1, after the prompt.
        let generated: String
        /// The `content` the session must record for turn 1.
        let recordedContent: String
        /// The `reasoning_content` the session must record for turn 1, or nil
        /// when the message must carry no reasoning.
        let recordedReasoning: String?
        /// Whether the render of turn 2 keeps the tokens the model wrote in
        /// turn 1, thus the cache serves them.
        let keepsTheGeneratedTurn: Bool

        var testDescription: String { name }
    }

    /// The reasoning the model writes in turn 1 when thinking is on.
    private static let reasoning = "Paris first."

    /// The answer the model writes in turn 1.
    private static let answer = "OK"

    /// The text the model writes in turn 1 when thinking is on: the reasoning,
    /// the close of the think block, and the answer.
    private static let reasonedTurn =
        reasoning + "\n" + QwenHistoryTokenizer.closeThink + "\n\n" + answer

    /// The cases, one row for each way to record a turn.
    static let cases: [HistoryCase] = [
        HistoryCase(
            name: "thinking on: the reasoning goes to reasoning_content",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .flag(true),
            generated: reasonedTurn, recordedContent: answer, recordedReasoning: reasoning,
            keepsTheGeneratedTurn: true),
        HistoryCase(
            name: "thinking off: the answer is the content",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .flag(false),
            generated: answer, recordedContent: answer, recordedReasoning: nil,
            keepsTheGeneratedTurn: true),
        HistoryCase(
            name: "the session switch on: the reasoning goes to reasoning_content",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .sessionSwitch(true),
            generated: reasonedTurn, recordedContent: answer, recordedReasoning: reasoning,
            keepsTheGeneratedTurn: true),
        HistoryCase(
            name: "the switch off: the closed block ends the prompt, the answer is the content",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .sessionSwitch(false),
            generated: answer, recordedContent: answer, recordedReasoning: nil,
            keepsTheGeneratedTurn: true),
        HistoryCase(
            name: "the control: a protocol that does not replay reasoning keeps the raw text",
            reasoningConfig: QwenReasoningProtocol.tagged, thinking: .flag(true),
            generated: reasonedTurn, recordedContent: reasonedTurn, recordedReasoning: nil,
            keepsTheGeneratedTurn: false),
    ]

    /// The first user turn.
    private static let firstPrompt = "My color is teal"

    /// The second user turn.
    private static let secondPrompt = "What is my color"

    /// The text the model writes in turn 2.
    private static let secondAnswer = "teal"

    /// Tokens each turn may generate. More than each scripted turn holds.
    private static let generatedTokenBudget = 64

    /// The history of turn 1 that the render of turn 2 holds, and the cache
    /// reuse of turn 2.
    @Test(arguments: cases)
    func theHistoryRenderOfTurnOne(_ row: HistoryCase) async throws {
        let tokenizer = QwenHistoryTokenizer()
        let endOfTurn = try #require(tokenizer.convertTokenToId(QwenHistoryTokenizer.endOfTurn))
        let script =
            tokenizer.encode(text: row.generated, addSpecialTokens: false) + [endOfTurn]
            + tokenizer.encode(text: Self.secondAnswer, addSpecialTokens: false) + [endOfTurn]
        let model = ScriptedTokenModel(
            script: script, finalToken: endOfTurn,
            vocabularySize: QwenHistoryTokenizer.vocabularySize, toolCallFormat: .qwen35)
        let session = session(
            model: model, reasoningConfig: row.reasoningConfig, thinking: row.thinking,
            tokenizer: tokenizer, tokenBudget: Self.generatedTokenBudget)

        let first = try #require(try await respond(session, to: Self.firstPrompt))
        let firstPrompt = Self.promptOfTurnOne(
            of: model, promptTokenCount: first.totalPromptTokenCount, tokenizer: tokenizer)
        let second = try #require(try await respond(session, to: Self.secondPrompt))

        let renders = tokenizer.renders.withLock { $0 }
        try #require(renders.count == 2)
        let history = try #require(
            renders[1].messages.first { $0["role"] as? String == "assistant" })
        #expect(history["content"] as? String == row.recordedContent)
        #expect(history["reasoning_content"] as? String == row.recordedReasoning)

        // The turn the model wrote, with the end-of-turn token that closed it.
        let committedTurn = row.generated + QwenHistoryTokenizer.endOfTurn
        #expect(
            renders[1].text.hasPrefix(firstPrompt + committedTurn)
                == row.keepsTheGeneratedTurn,
            "turn 2 render <<<\(renders[1].text)>>>")
        let committedTokenCount = tokenizer.encode(text: committedTurn, addSpecialTokens: false)
            .count
        if row.keepsTheGeneratedTurn {
            #expect(
                second.cachedPromptTokenCount == first.totalPromptTokenCount + committedTokenCount)
        } else {
            #expect(second.cachedPromptTokenCount < first.totalPromptTokenCount)
        }
    }

    // MARK: - A turn that ends inside its reasoning

    /// One way turn 1 of a hybrid Qwen 3.8 model ends with thinking on, and
    /// how turn 2 then reuses the caches (card `^ef2zynk`).
    struct TurnEndCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The text the model writes in turn 1, after the prompt.
        let generated: String
        /// Whether the model ends turn 1 with the end-of-turn token. When
        /// false, turn 1 stops on the token budget.
        let writesEndOfTurn: Bool
        /// Whether turn 2 feeds after every token that turn 1 left in the
        /// caches. When false, turn 2 goes back to the checkpoint at the end of
        /// the turn-1 prompt.
        let reusesTheLedger: Bool
        /// Whether the caches hold the cold prompt of turn 2 when the prefill
        /// of turn 2 ends.
        let holdsTheColdRender: Bool
        /// How the session asks for thinking.
        let thinking: ThinkingSetting
        /// Whether the prompt of a turn ends with the open think block. When
        /// false, it ends with the closed think block.
        let promptThinks: Bool

        var testDescription: String { name }
    }

    /// Whitespace that the template trims from the end of the reasoning and
    /// of the answer. A turn that ends with it does not render the tokens the
    /// model wrote.
    private static let trimmedWhitespace = " "

    /// The cases, one row for each way turn 1 ends.
    static let turnEndCases: [TurnEndCase] = [
        TurnEndCase(
            name: "stops inside reasoning that the template trims: the prompt-end checkpoint",
            generated: reasoning + trimmedWhitespace, writesEndOfTurn: false,
            reusesTheLedger: false, holdsTheColdRender: true, thinking: .flag(true),
            promptThinks: true),
        TurnEndCase(
            name: "stops inside reasoning that the render extends: the ledger extends",
            generated: reasoning, writesEndOfTurn: false,
            reusesTheLedger: true, holdsTheColdRender: true, thinking: .flag(true),
            promptThinks: true),
        TurnEndCase(
            name: "the control: closes its reasoning: the splice keeps the tokens the model wrote",
            generated: reasonedTurn + trimmedWhitespace, writesEndOfTurn: true,
            reusesTheLedger: true, holdsTheColdRender: false, thinking: .flag(true),
            promptThinks: true),
        TurnEndCase(
            name: "the session switch off: the closed block ends turn 1, the ledger extends",
            generated: answer, writesEndOfTurn: true,
            reusesTheLedger: true, holdsTheColdRender: true, thinking: .sessionSwitch(false),
            promptThinks: false),
    ]

    /// Where turn 2 starts to feed the caches of a hybrid model, and whether
    /// the caches then hold the cold prompt of turn 2.
    @Test(arguments: turnEndCases)
    func theTurnAfterTurnOneEnds(_ row: TurnEndCase) async throws {
        let tokenizer = QwenHistoryTokenizer()
        let endOfTurn = try #require(tokenizer.convertTokenToId(QwenHistoryTokenizer.endOfTurn))
        let written =
            tokenizer.encode(text: row.generated, addSpecialTokens: false)
            + (row.writesEndOfTurn ? [endOfTurn] : [])
        // The feed to the checkpoint at the end of the last message is one
        // forward pass before the prefill, and nothing samples its token.
        let model = ScriptedTokenModel(
            script: [endOfTurn] + written, finalToken: endOfTurn,
            vocabularySize: QwenHistoryTokenizer.vocabularySize, toolCallFormat: .qwen35,
            hasRecurrentLayer: true)
        let session = session(
            model: model, reasoningConfig: QwenReasoningProtocol.qwen35, thinking: row.thinking,
            tokenizer: tokenizer, tokenBudget: written.count)

        let first = try #require(try await respond(session, to: Self.firstPrompt))
        let firstTurnPassCount = model.passes.count
        _ = try await respond(session, to: Self.secondPrompt)

        let promptEnd = first.totalPromptTokenCount
        let secondTurnStart = try #require(model.passes.dropFirst(firstTurnPassCount).first)
        #expect(
            secondTurnStart.offset == (row.reusesTheLedger ? promptEnd + written.count : promptEnd))

        let coldRender = try #require(
            tokenizer.renders.withLock { $0 }.last {
                $0.messages.last?["content"] as? String == Self.secondPrompt
            })
        // The cold prompt: the messages of the cold render, and the
        // generation prompt that the prompt of a turn ends with.
        let coldPrompt =
            coldRender.messages.map(QwenHistoryTokenizer.render).joined()
            + QwenHistoryTokenizer.generationPrompt(thinking: row.promptThinks)
        let coldTokens = tokenizer.encode(text: coldPrompt, addSpecialTokens: false)
        let heldTokens = Self.attentionCacheTokens(after: model.passes)
        #expect(
            (Array(heldTokens.prefix(coldTokens.count)) == coldTokens) == row.holdsTheColdRender,
            "held <<<\(tokenizer.decode(tokenIds: heldTokens))>>>")
    }

    /// The tokens the attention cache holds after `passes`. Each pass begins
    /// where the cache stood, which drops the tokens a restore trimmed away.
    private static func attentionCacheTokens(after passes: [ScriptedTokenModel.Pass]) -> [Int] {
        passes.reduce(into: []) { held, pass in
            held = Array(held.prefix(pass.offset)) + pass.tokens
        }
    }

    // MARK: - The thinking switch of the session

    /// One way to set thinking, and what the template and the model then get.
    struct SwitchCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The reasoning protocol of the model, or nil when the model declares
        /// none.
        let reasoningConfig: ReasoningConfig?
        /// How the session asks for thinking.
        let thinking: ThinkingSetting
        /// Whether the template gets the flag on, thus writes the open think
        /// block.
        let templateThinks: Bool
        /// Whether the prompt that the model gets ends with the open think
        /// block. When false, it ends with the closed think block.
        let promptThinks: Bool

        var testDescription: String { name }
    }

    /// The cases, one row for each way to set thinking.
    static let switchCases: [SwitchCase] = [
        SwitchCase(
            name: "the control: no switch keeps the template default",
            reasoningConfig: QwenReasoningProtocol.qwen35,
            thinking: ThinkingSetting(templateFlag: nil, sessionSwitch: nil),
            templateThinks: true, promptThinks: true),
        SwitchCase(
            name: "the switch on: the open block ends the prompt",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .sessionSwitch(true),
            templateThinks: true, promptThinks: true),
        SwitchCase(
            name: "the switch off on Qwen 3.5: the flag stays on, the closed block ends the prompt",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .sessionSwitch(false),
            templateThinks: true, promptThinks: false),
        SwitchCase(
            name: "the switch off on a protocol that renders the flag: the flag goes off",
            reasoningConfig: QwenReasoningProtocol.tagged, thinking: .sessionSwitch(false),
            templateThinks: false, promptThinks: false),
        SwitchCase(
            name: "the switch on and the template flag off: the switch wins",
            reasoningConfig: QwenReasoningProtocol.qwen35,
            thinking: ThinkingSetting(templateFlag: false, sessionSwitch: true),
            templateThinks: true, promptThinks: true),
        SwitchCase(
            name: "the template flag off with no switch: the flag of the caller stays",
            reasoningConfig: QwenReasoningProtocol.qwen35, thinking: .flag(false),
            templateThinks: false, promptThinks: false),
        SwitchCase(
            name: "the switch off on a model with no reasoning protocol: the switch does nothing",
            reasoningConfig: nil, thinking: .sessionSwitch(false),
            templateThinks: true, promptThinks: true),
    ]

    /// The render that the template gets, and the prompt that the model gets,
    /// for each way to set thinking.
    @Test(arguments: switchCases)
    func theThinkingSettingOfTheRenderAndThePrompt(_ row: SwitchCase) async throws {
        let tokenizer = QwenHistoryTokenizer()
        let endOfTurn = try #require(tokenizer.convertTokenToId(QwenHistoryTokenizer.endOfTurn))
        let model = ScriptedTokenModel(
            script: tokenizer.encode(text: Self.answer, addSpecialTokens: false) + [endOfTurn],
            finalToken: endOfTurn, vocabularySize: QwenHistoryTokenizer.vocabularySize,
            toolCallFormat: .qwen35)
        let session = session(
            model: model, reasoningConfig: row.reasoningConfig, thinking: row.thinking,
            tokenizer: tokenizer, tokenBudget: Self.generatedTokenBudget)

        let first = try #require(try await respond(session, to: Self.firstPrompt))

        let render = try #require(tokenizer.renders.withLock { $0 }.first)
        let templatePrompt = QwenHistoryTokenizer.generationPrompt(thinking: row.templateThinks)
        #expect(render.text.hasSuffix(templatePrompt), "render <<<\(render.text)>>>")
        let prompt = Self.promptOfTurnOne(
            of: model, promptTokenCount: first.totalPromptTokenCount, tokenizer: tokenizer)
        #expect(
            prompt.hasSuffix(QwenHistoryTokenizer.generationPrompt(thinking: row.promptThinks)),
            "prompt <<<\(prompt)>>>")
    }

    /// A model that always reasons cannot turn thinking off. The switch off
    /// fails the turn before the model runs, with the error of the prompt
    /// strategy.
    @Test func theSwitchOffFailsOnAModelThatAlwaysReasons() async throws {
        let tokenizer = QwenHistoryTokenizer()
        let endOfTurn = try #require(tokenizer.convertTokenToId(QwenHistoryTokenizer.endOfTurn))
        let model = ScriptedTokenModel(
            script: [endOfTurn], finalToken: endOfTurn,
            vocabularySize: QwenHistoryTokenizer.vocabularySize, toolCallFormat: .qwen35)
        let session = session(
            model: model, reasoningConfig: .alwaysOnThinking, thinking: .sessionSwitch(false),
            tokenizer: tokenizer, tokenBudget: Self.generatedTokenBudget)

        await #expect(throws: ReasoningError.cannotDisableReasoning) {
            try await respond(session, to: Self.firstPrompt)
        }
        #expect(model.passes.isEmpty)
    }

    // MARK: - Helpers

    /// The text of the prompt that the model got in turn 1: the first
    /// `promptTokenCount` tokens that the attention cache holds. With the
    /// closed think block, this is not the text that the template wrote.
    private static func promptOfTurnOne(
        of model: ScriptedTokenModel, promptTokenCount: Int, tokenizer: QwenHistoryTokenizer
    ) -> String {
        tokenizer.decode(
            tokenIds: Array(attentionCacheTokens(after: model.passes).prefix(promptTokenCount)))
    }

    /// A session over `model` and `tokenizer` that asks for thinking as
    /// `thinking` tells.
    private func session(
        model: ScriptedTokenModel, reasoningConfig: ReasoningConfig?, thinking: ThinkingSetting,
        tokenizer: QwenHistoryTokenizer, tokenBudget: Int
    ) -> ChatSession {
        let configuration = ModelConfiguration(
            id: "qwen38-history-test", toolCallFormat: .qwen35,
            reasoningConfig: reasoningConfig)
        let context = ModelContext(
            configuration: configuration, model: model,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: tokenBudget, temperature: 0),
            additionalContext: thinking.templateFlag.map {
                [QwenHistoryTokenizer.thinkingFlag: $0] as [String: any Sendable]
            })
        session.thinkingEnabled = thinking.sessionSwitch
        return session
    }

    /// Runs one turn of `session` and returns its completion information.
    private func respond(_ session: ChatSession, to prompt: String) async throws
        -> GenerateCompletionInfo?
    {
        var info: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: prompt) {
            info = item.info ?? info
        }
        return info
    }
}

/// What one render of the chat template got and wrote.
struct QwenHistoryRender: Sendable {
    /// The raw messages the template got.
    let messages: [[String: any Sendable]]
    /// The text the template wrote.
    let text: String
}

/// A tokenizer that writes the Qwen 3.8 chat template shape for its messages,
/// with no system block and no tools, and records each render.
///
/// A special marker and a run of two newlines are one token each, as in the
/// Qwen tokenizer. Every other character is its own token.
struct QwenHistoryTokenizer: MLXLMCommon.Tokenizer {

    /// The chat template flag that turns thinking on and off.
    static let thinkingFlag = "enable_thinking"

    /// The marker that opens a message.
    static let startOfTurn = "<|im_start|>"

    /// The marker that closes a message, and the token that ends a turn.
    static let endOfTurn = "<|im_end|>"

    /// The marker that opens a think block.
    static let openThink = "<think>"

    /// The marker that closes a think block.
    static let closeThink = "</think>"

    /// The pieces of the vocabulary, in the order the encoder tries them: the
    /// markers, two newlines before one, then each character of the alphabet.
    static let pieces: [String] =
        [startOfTurn, endOfTurn, openThink, closeThink, "\n\n", "\n"]
        + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,".map(String.init)

    /// The size of the vocabulary: each piece, and the unknown token.
    static let vocabularySize = pieces.count + 1

    /// Every render, in order.
    let renders = OSAllocatedUnfairLock<[QwenHistoryRender]>(initialState: [])

    var bosToken: String? { nil }
    var eosToken: String? { Self.endOfTurn }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { Self.pieces.count }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let thinking = additionalContext?[Self.thinkingFlag] as? Bool ?? true
        let text = messages.map(Self.render).joined() + Self.generationPrompt(thinking: thinking)
        renders.withLock { $0.append(QwenHistoryRender(messages: messages, text: text)) }
        return encode(text: text, addSpecialTokens: false)
    }

    /// The text of one message, as the Qwen 3.8 template writes it when
    /// `preserve_thinking` is undefined.
    static func render(_ message: [String: any Sendable]) -> String {
        let role = message["role"] as? String ?? ""
        let content = (message["content"] as? String ?? "").trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard role == "assistant" else {
            return startOfTurn + role + "\n" + content + endOfTurn + "\n"
        }
        let reasoning = (message["reasoning_content"] as? String ?? "").trimmingCharacters(
            in: .whitespacesAndNewlines)
        return startOfTurn + "assistant\n" + openThink + "\n" + reasoning + "\n" + closeThink
            + "\n\n" + content + endOfTurn + "\n"
    }

    /// The generation prompt: an open think block with thinking on, and a closed,
    /// empty think block with thinking off.
    static func generationPrompt(thinking: Bool) -> String {
        let block = thinking ? openThink + "\n" : openThink + "\n\n" + closeThink + "\n\n"
        return startOfTurn + "assistant\n" + block
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let index = Self.pieces.firstIndex(where: { rest.hasPrefix($0) }) {
                tokens.append(index)
                rest = rest.dropFirst(Self.pieces[index].count)
            } else {
                tokens.append(Self.pieces.count)
                rest = rest.dropFirst()
            }
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap(convertIdToToken).joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        Self.pieces.firstIndex(of: token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        Self.pieces.indices.contains(id) ? Self.pieces[id] : nil
    }
}
