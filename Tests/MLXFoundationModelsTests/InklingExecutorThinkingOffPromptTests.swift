// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// The prompt of a text call of an Inkling 2 model with thinking off, on a scripted model.
///
/// The Inkling 2 template reads `reasoning_effort: none` only as a hint: the real checkpoint
/// still opens a thinking frame, which can spend the whole token budget and leave the answer
/// empty. A text call with thinking off thus appends the header of the answer frame,
/// `<|message_model|><|content_text|>`, after the prompt, and the model answers at once.
@Suite("The thinking-off text prompt of an Inkling 2 model on the executor")
struct InklingExecutorThinkingOffPromptTests {

    /// The question of the user.
    private static let question = "Is Rome in Italy?"

    /// The answer the scripted model writes.
    private static let answer = "Rome is in Italy."

    /// The tokenizer whose vocabulary holds every text of the pass the test reads.
    private static let tokenizer = InklingFrameTokenizer(pieces: [question, answer])

    /// The frame of the question, in tokens.
    private static var questionFrame: [Int] {
        [
            tokenizer.token(.messageUser), tokenizer.token(.contentText),
            tokenizer.token(question), tokenizer.token(.endMessage),
        ]
    }

    /// The header of the answer frame, in tokens.
    private static var answerHeader: [Int] {
        [tokenizer.token(.messageModel), tokenizer.token(.contentText)]
    }

    /// The tokens the scripted model writes after the header: the answer and the end of its
    /// turn.
    private static var answerBody: [Int] {
        [tokenizer.token(answer), tokenizer.token(.endMessage), tokenizer.token(.endSampling)]
    }

    /// The token of the thinking-effort text at `level`.
    private static func effortToken(_ level: String) -> Int {
        tokenizer.token(InklingFrameTokenizer.effortPrefix + level)
    }

    /// The prompt of the one pass of a text call of a scripted Inkling 2 model, and the text the
    /// call streamed as its response.
    ///
    /// - Parameters:
    ///   - capabilities: the capabilities the model declares.
    ///   - reasoningLevel: the reasoning level of the call.
    ///   - written: the tokens the scripted model writes.
    /// - Returns: the prompt tokens of the pass and the response text.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func textPass(
        capabilities: [LanguageModelCapabilities.Capability],
        reasoningLevel: ContextOptions.ReasoningLevel?, written: [Int]
    ) async throws -> (prompt: [Int], response: String) {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let recorder = ScriptedPromptRecorder()
        let configuration = ModelConfiguration(
            id: "probe/inkling-thinking-off-\(UUID().uuidString)", toolCallFormat: .inkling,
            reasoningConfig: InklingReasoningProtocol.inkling2)
        let model = MLXLanguageModel(
            configuration: configuration, capabilities: capabilities,
            weightsLocation: { _ in weights },
            load: { _, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: ScriptedLanguageModel(rounds: [written], promptRecorder: recorder),
                        processor: ScalarTokenizerInputProcessor(
                            tokenizer: tokenizer, configuration: configuration),
                        tokenizer: tokenizer))
            })
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = reasoningLevel
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: question))]))
            ]),
            contextOptions: contextOptions)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorThinkingOff")
        let result = try await ScriptedExecutorPass.respond(
            to: request, model: model, inside: store)
        return (try #require(recorder.prompts.last), result.responseText)
    }

    @Test("no_think with reasoning declared starts after the header of the answer frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func noThinkStartsAfterTheAnswerHeader() async throws {
        let pass = try await Self.textPass(
            capabilities: [.reasoning], reasoningLevel: .custom("no_think"),
            written: Self.answerBody)

        #expect(
            pass.prompt.suffix(Self.questionFrame.count + Self.answerHeader.count)
                == ArraySlice(Self.questionFrame + Self.answerHeader))
        #expect(pass.prompt.contains(Self.effortToken(InklingReasoningProtocol.thinkingOffEffort)))
        #expect(pass.response == Self.answer)
    }

    @Test("undeclared reasoning starts after the header of the answer frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func undeclaredReasoningStartsAfterTheAnswerHeader() async throws {
        let pass = try await Self.textPass(
            capabilities: [], reasoningLevel: nil, written: Self.answerBody)

        #expect(
            pass.prompt.suffix(Self.questionFrame.count + Self.answerHeader.count)
                == ArraySlice(Self.questionFrame + Self.answerHeader))
        #expect(pass.response == Self.answer)
    }

    @Test("thinking on keeps the prompt of the template")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thinkingOnKeepsThePrompt() async throws {
        let pass = try await Self.textPass(
            capabilities: [.reasoning], reasoningLevel: nil,
            written: Self.answerHeader + Self.answerBody)

        #expect(pass.prompt.suffix(Self.questionFrame.count) == ArraySlice(Self.questionFrame))
        #expect(pass.response == Self.answer)
    }
}

#endif
