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

    /// The text answer the scripted model writes on the allowed tool path, which calls no tool.
    private static let textAnswer = "Rome is in Italy."

    /// The tokenizer whose vocabulary holds every text of the pass the test reads.
    private static let tokenizer = InklingFrameTokenizer(pieces: [question, answer, textAnswer])

    /// The one tool of a pass on the tool path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var tool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "lookup", description: "Looks a place up.", parameters: String.generationSchema)
    }

    /// The tokens the scripted model writes on the allowed tool path: a text answer frame and
    /// the end of its turn, with no tool frame.
    private static var textAnswerTurn: [Int] {
        answerHeader + [
            tokenizer.token(textAnswer), tokenizer.token(.endMessage),
            tokenizer.token(.endSampling),
        ]
    }

    /// The Inkling 2 frames with a reasoning protocol that cannot turn thinking off.
    private static var alwaysOnWithAnswerOpening: ReasoningConfig {
        var config = InklingReasoningProtocol.frames
        config.promptStrategy = .alwaysOn
        return config
    }

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

    /// The prompts of a scripted Inkling 2 model for one call under a Boolean schema, and the
    /// response text of the call.
    ///
    /// - Parameters:
    ///   - reasoningConfig: the reasoning protocol of the model, or nil for none.
    ///   - capabilities: the capabilities the model declares.
    ///   - enabledTools: the tools of the call; with a tool the call runs allowed tool mode.
    ///   - rounds: the tokens each pass of the call writes, in the order of the passes.
    /// - Returns: the prompt tokens of each pass, in order, and the text the call streamed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaCall(
        reasoningConfig: ReasoningConfig?,
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration],
        enabledTools: [Transcript.ToolDefinition] = [],
        rounds: [[Int]]? = nil
    ) async throws -> (prompts: [[Int]], response: String) {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let configuration = ModelConfiguration(
            id: "probe/inkling-schema-prompt-\(UUID().uuidString)", toolCallFormat: .inkling,
            reasoningConfig: reasoningConfig)
        let recorder = ScriptedPromptRecorder()
        let model = MLXLanguageModel(
            configuration: configuration, capabilities: capabilities,
            weightsLocation: { _ in weights },
            load: { _, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: ScriptedLanguageModel(
                            rounds: rounds ?? [[tokenizer.token(answer)]],
                            promptRecorder: recorder),
                        processor: ScalarTokenizerInputProcessor(
                            tokenizer: tokenizer, configuration: configuration),
                        tokenizer: tokenizer))
            })
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: question))]))
            ]),
            enabledTools: enabledTools,
            schema: Bool.generationSchema,
            generationOptions: enabledTools.isEmpty
                ? GenerationOptions() : GenerationOptions(toolCallingMode: .allowed))
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorSchemaPrompt")
        let result = try await ScriptedExecutorPass.respond(
            to: request, model: model, inside: store)
        return (recorder.prompts, result.responseText)
    }

    /// The prompt of the one schema pass of a scripted Inkling 2 model, and its response text.
    ///
    /// - Parameter reasoningConfig: the reasoning protocol of the model, or nil for none.
    /// - Returns: the prompt tokens the pass fed the model, and the text it streamed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaPass(reasoningConfig: ReasoningConfig?) async throws -> (
        prompt: [Int], response: String
    ) {
        let call = try await schemaCall(reasoningConfig: reasoningConfig)
        return (try #require(call.prompts.last), call.response)
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

    @Test("a reasoning protocol without an answer opening keeps the prompt of the template")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aProtocolWithoutAnAnswerOpeningKeepsThePrompt() async throws {
        #expect(QwenReasoningProtocol.qwen35.answerOpening == nil)
        let template = try await Self.schemaPass(reasoningConfig: nil)
        let pass = try await Self.schemaPass(reasoningConfig: QwenReasoningProtocol.qwen35)

        #expect(pass.prompt == template.prompt)
        #expect(pass.prompt.contains(Self.effortToken(InklingFrameTokenizer.defaultEffort)))
        #expect(
            !pass.prompt.contains(Self.effortToken(InklingReasoningProtocol.thinkingOffEffort)))
        #expect(pass.response == Self.answer)
    }

    @Test("the schema fallback of allowed tool mode starts after the header of the answer frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theAllowedToolSchemaFallbackStartsAfterTheAnswerHeader() async throws {
        let call = try await Self.schemaCall(
            reasoningConfig: InklingReasoningProtocol.inkling2,
            capabilities: [.guidedGeneration, .toolCalling], enabledTools: [Self.tool],
            rounds: [Self.textAnswerTurn, [Self.tokenizer.token(Self.answer)]])

        try #require(call.prompts.count == 2, "prompts \(call.prompts)")
        let schemaPrompt = call.prompts[1]
        #expect(
            schemaPrompt.suffix(Self.questionFrame.count + Self.answerHeader.count)
                == ArraySlice(Self.questionFrame + Self.answerHeader))
        #expect(
            schemaPrompt.contains(Self.effortToken(InklingReasoningProtocol.thinkingOffEffort)))
        #expect(call.response == Self.answer)
    }

    @Test("an answer opening on a model that always reasons is an unsupported capability")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anAnswerOpeningThatCannotTurnThinkingOffThrows() async throws {
        do {
            _ = try await Self.schemaCall(
                reasoningConfig: Self.alwaysOnWithAnswerOpening,
                capabilities: [.guidedGeneration, .reasoning])
            Issue.record("Expected unsupportedCapability(.reasoning), but respond returned")
        } catch let error as LanguageModelError {
            guard case .unsupportedCapability(let unsupported) = error else {
                Issue.record("Expected unsupportedCapability, got \(error)")
                return
            }
            #expect(unsupported.capability == .reasoning)
            #expect(
                unsupported.debugDescription
                    == MLXLanguageModel.Executor.answerOpeningNeedsThinkingOffMessage)
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
