// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Synchronization
import Testing

@testable import MLXFoundationModels

/// The error of a render whose prompt is not a chat.
private struct UnexpectedDeepSeekV4PromptError: Error {}

/// A processor that renders each prompt with ``DeepSeekV4EncodingTokenizer``
/// over a byte vocabulary, and records the text of each render.
///
/// The byte vocabulary holds no DeepSeek-V4 marker, thus the tokenizer writes
/// the rendered text as its UTF-8 bytes, and a decode gives the text back.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class DeepSeekV4RecordingProcessor: UserInputProcessor, Sendable {

    /// The tokenizer that renders and encodes each prompt.
    private let tokenizer = DeepSeekV4EncodingTokenizer(wrapping: ScriptedByteTokenizer())

    /// The text of each render, in order.
    private let rendered = Mutex<[String]>([])

    /// The text of each render, in order.
    var renders: [String] { rendered.withLock { $0 } }

    /// Renders `input` with the DeepSeek-V4 encoder and records the text.
    ///
    /// - Parameter input: the chat, the tools and the template variables.
    /// - Returns: the tokens of the render.
    /// - Throws: ``UnexpectedDeepSeekV4PromptError`` when the prompt is not a
    ///   chat.
    func prepare(input: UserInput) async throws -> LMInput {
        guard case .chat(let messages) = input.prompt else {
            throw UnexpectedDeepSeekV4PromptError()
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: messages), tools: input.tools,
            additionalContext: input.additionalContext)
        let text = ScriptedByteTokenizer().decode(tokenIds: tokens, skipSpecialTokens: false)
        rendered.withLock { $0.append(text) }
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

/// Proves that the reasoning level of a request selects the generation mode
/// of a DeepSeek-V4 prompt, through the reasoning protocol of DeepSeek-V4
/// (``DeepSeekV4EncodingTokenizer/reasoningConfig``).
///
/// The executor is available from iOS 27, macOS 27 and visionOS 27. On an
/// earlier system each test records an issue, thus it fails and does not pass
/// with no assertion.
@Suite("The reasoning level selects the generation mode of a DeepSeek-V4 prompt")
struct DeepSeekV4ReasoningLevelRenderTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    /// The identifier of the first entry, which names the session.
    private static let sessionID = "deepseek-v4-session"

    /// The level name that turns thinking off for one request.
    private static let thinkingOffLevelName = "no_think"

    /// One reasoning level, and the generation prompt that the render must then
    /// end with.
    struct LevelCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The name of the custom reasoning level of the request, or nil for
        /// no level.
        let customLevel: String?
        /// The text that the render of the request must end with.
        let generationPrompt: String

        var testDescription: String { name }
    }

    /// The generation prompt of the thinking mode: an open reasoning block.
    private static let thinkingPrompt =
        DeepSeekV4ChatEncoder.SpecialToken.assistant + DeepSeekV4ChatEncoder.SpecialToken.thinkStart

    /// The generation prompt of the chat mode: a closed reasoning block.
    private static let chatPrompt =
        DeepSeekV4ChatEncoder.SpecialToken.assistant + DeepSeekV4ChatEncoder.SpecialToken.thinkEnd

    /// The cases, one row for each reasoning level.
    static let cases: [LevelCase] = [
        LevelCase(
            name: "the control: no level keeps the encoder default, thinking",
            customLevel: nil, generationPrompt: thinkingPrompt),
        LevelCase(
            name: "no_think: the prompt closes the reasoning",
            customLevel: thinkingOffLevelName, generationPrompt: chatPrompt),
    ]

    @Test("the reasoning level selects the generation mode", arguments: cases)
    func theReasoningLevelSelectsTheGenerationMode(_ row: LevelCase) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectTheRenderEnds(with: row)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    /// One pass for the level of `row` renders a prompt that ends with the
    /// generation prompt of `row`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectTheRenderEnds(with row: LevelCase) async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let processor = DeepSeekV4RecordingProcessor()
        let model = ScriptedSessionModel.make(
            weights: weights, processor: processor,
            reasoningConfig: DeepSeekV4EncodingTokenizer.reasoningConfig)
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = row.customLevel.map { .custom($0) }
        let request = makeExecutorRequest(
            transcript: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID),
            contextOptions: contextOptions)
        let store = ExecutorPromptCacheStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("DeepSeekV4ReasoningLevelRenderTests-\(UUID().uuidString)"),
            writer: { _, _ in })

        _ = try await ScriptedExecutorPass.respond(to: request, model: model, inside: store)

        #expect(
            processor.renders.contains { $0.hasSuffix(row.generationPrompt) },
            "renders <<<\(processor.renders)>>>")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
