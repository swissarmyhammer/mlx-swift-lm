// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// The prompt of a schema pass of an Inkling 2 model on the executor, on a scripted model.
///
/// The Inkling 2 template writes no generation prompt: the model writes the header of its
/// answer frame, `<|message_model|><|content_text|>`, after the prompt. A grammar that starts at
/// the end of the prompt forbids that header, and the real checkpoint then fills every string
/// of the schema with `"}"`. The schema pass thus renders the prompt with thinking off and
/// appends the header, and the grammar starts where the answer starts.
@Suite("The schema prompt of an Inkling 2 model on the executor")
struct InklingExecutorSchemaPromptTests {

    /// The question of the user.
    private static let question = "Is Rome in Italy?"

    /// The answer the scripted model writes under the grammar of a Boolean schema.
    private static let answer = "true"

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

    /// The token of the thinking-effort text at `level`.
    private static func effortToken(_ level: String) -> Int {
        tokenizer.token(InklingFrameTokenizer.effortPrefix + level)
    }

    /// The prompt of the one schema pass of a scripted Inkling 2 model, and its response text.
    ///
    /// - Parameter reasoningConfig: the reasoning protocol of the model, or nil for none.
    /// - Returns: the prompt tokens the pass fed the model, and the text it streamed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaPass(reasoningConfig: ReasoningConfig?) async throws -> (
        prompt: [Int], response: String
    ) {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let configuration = ModelConfiguration(
            id: "probe/inkling-schema-prompt-\(UUID().uuidString)", toolCallFormat: .inkling,
            reasoningConfig: reasoningConfig)
        let recorder = ScriptedPromptRecorder()
        let model = MLXLanguageModel(
            configuration: configuration, capabilities: [.guidedGeneration],
            weightsLocation: { _ in weights },
            load: { _, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: ScriptedLanguageModel(
                            rounds: [[tokenizer.token(answer)]], promptRecorder: recorder),
                        processor: ScalarTokenizerInputProcessor(
                            tokenizer: tokenizer, configuration: configuration),
                        tokenizer: tokenizer))
            })
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: question))]))
            ]),
            schema: Bool.generationSchema)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorSchemaPrompt")
        let result = try await ScriptedExecutorPass.respond(
            to: request, model: model, inside: store)
        return (try #require(recorder.prompts.last), result.responseText)
    }

    @Test("the schema pass starts its grammar after the header of the answer frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theSchemaPassStartsAfterTheAnswerHeader() async throws {
        let pass = try await Self.schemaPass(reasoningConfig: InklingReasoningProtocol.inkling2)

        #expect(
            pass.prompt.suffix(Self.questionFrame.count + Self.answerHeader.count)
                == ArraySlice(Self.questionFrame + Self.answerHeader))
        #expect(pass.response == Self.answer)
    }

    @Test("the schema pass renders thinking off")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theSchemaPassRendersThinkingOff() async throws {
        let pass = try await Self.schemaPass(reasoningConfig: InklingReasoningProtocol.inkling2)

        #expect(pass.prompt.contains(Self.effortToken(InklingReasoningProtocol.thinkingOffEffort)))
        #expect(!pass.prompt.contains(Self.effortToken(InklingFrameTokenizer.defaultEffort)))
    }

    @Test("a model without an answer opening keeps the prompt of the template")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aModelWithoutAnAnswerOpeningKeepsThePrompt() async throws {
        let pass = try await Self.schemaPass(reasoningConfig: nil)

        #expect(pass.prompt.suffix(Self.questionFrame.count) == ArraySlice(Self.questionFrame))
        #expect(pass.prompt.contains(Self.effortToken(InklingFrameTokenizer.defaultEffort)))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
