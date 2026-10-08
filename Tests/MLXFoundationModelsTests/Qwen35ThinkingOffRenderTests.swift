// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The error of a contract render whose prompt is not a chat.
private struct UnexpectedPromptError: Error {}

/// A processor that renders a prompt with the parts of the Qwen 3.5 chat
/// template that the thinking flag changes, one token for each UTF-8 byte.
///
/// The package has no Jinja engine, thus this processor states the contract of
/// `chat_template.jinja` of `mlx-community/Qwen3.8-27B-mxfp4`:
///
/// - The reasoning instructions go first in the system block when
///   `enable_thinking` is undefined or true, and they are absent when it is
///   false.
/// - The generation prompt is the assistant header and an open think block when
///   thinking is on, and the header and a closed, empty think block when
///   thinking is off.
///
/// `IntegrationTesting` renders the real template file and holds it to the
/// same contract.
struct Qwen35TemplateContractProcessor: UserInputProcessor {

    /// The chat template flag that turns thinking on and off.
    static let thinkingFlag = "enable_thinking"

    /// The reasoning instructions that the template writes when thinking is on.
    static let reasoningInstructions = "Reasoning effort is set to xhigh."

    /// The header that opens the assistant turn of the generation prompt.
    static let assistantHeader = "<|im_start|>assistant\n"

    /// The open think block of the generation prompt when thinking is on.
    static let openThinkBlock = "<think>\n"

    /// The closed, empty think block of the generation prompt when thinking is off.
    static let closedThinkBlock = "<think>\n\n</think>\n\n"

    /// The separator between two parts of the system block.
    private static let partSeparator = "\n\n"

    /// Renders `input` and encodes the text as one token for each byte.
    ///
    /// - Parameter input: the chat, the tools and the template variables.
    /// - Returns: the tokens of the render.
    /// - Throws: ``UnexpectedPromptError`` when the prompt is not a chat.
    func prepare(input: UserInput) async throws -> LMInput {
        guard case .chat(let messages) = input.prompt else { throw UnexpectedPromptError() }
        let thinking = (input.additionalContext?[Self.thinkingFlag] as? Bool) != false
        let text = Self.render(
            messages: messages, toolCount: input.tools?.count ?? 0, thinking: thinking)
        let tokens = ScriptedByteTokenizer().encode(text: text, addSpecialTokens: false)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    /// The system block of a render.
    ///
    /// - Parameters:
    ///   - instructions: the text of the system message, or nil when there is none.
    ///   - toolCount: the number of tools the render describes.
    ///   - thinking: whether the template flag turns thinking on.
    /// - Returns: the system block, or an empty string when it holds nothing.
    static func systemBlock(instructions: String?, toolCount: Int, thinking: Bool) -> String {
        let parts = [
            thinking ? reasoningInstructions : "",
            toolCount > 0 ? "# Tools: \(toolCount)" : "",
            instructions ?? "",
        ].filter { !$0.isEmpty }
        guard !parts.isEmpty else { return "" }
        return "<|im_start|>system\n" + parts.joined(separator: partSeparator) + "<|im_end|>\n"
    }

    /// Renders the system block, each turn, and the generation prompt.
    private static func render(messages: [Chat.Message], toolCount: Int, thinking: Bool) -> String {
        let instructions = messages.first { $0.role == .system }?.content
        let turns = messages.filter { $0.role != .system }.map {
            "<|im_start|>\($0.role.rawValue)\n\($0.content)<|im_end|>\n"
        }
        return systemBlock(instructions: instructions, toolCount: toolCount, thinking: thinking)
            + turns.joined() + assistantHeader + (thinking ? openThinkBlock : closedThinkBlock)
    }
}

/// Proves that a turn of the Qwen 3.5 protocol with thinking off keeps the
/// system block of a turn with thinking on, thus a session that turns thinking
/// off and on again reuses its prompt cache past the system block.
///
/// Each test runs on the unconstrained path (no tools) and on the tool path.
/// The executor is available from iOS 27, macOS 27 and visionOS 27. On an
/// earlier system each test records an issue, thus it fails and does not pass
/// with no assertion.
@Suite("A Qwen 3.5 turn with thinking off keeps the system block of thinking on")
struct Qwen35ThinkingOffRenderTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    /// The identifier of the first entry, which names the session.
    private static let sessionID = "qwen35-session"

    /// The instructions of the session.
    private static let instructions = "Be terse."

    /// The text a pass with thinking on writes: thinking that the open block of
    /// the prompt started, the end delimiter, and the answer.
    private static let thinkingScript = "plan</think>A"

    /// The text a pass with thinking off writes.
    private static let answerScript = "A"

    /// The level name that turns thinking off for one request.
    private static let thinkingOffLevelName = "no_think"

    /// The one tool of a pass on the tool path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var tool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "lookup", description: "Looks a word up.", parameters: String.generationSchema)
    }

    @Test(
        "a turn with thinking off renders the system block of thinking on",
        arguments: [false, true])
    func aThinkingOffTurnRendersTheSystemBlockOfThinkingOn(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAThinkingOffTurnRendersTheSystemBlockOfThinkingOn(withTools: withTools)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn with thinking off reuses a turn with thinking on", arguments: [false, true])
    func aThinkingOffTurnReusesAThinkingOnTurn(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectTheSecondTurnReusesTheSystemBlock(
                firstThinking: true, withTools: withTools)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn with thinking on reuses a turn with thinking off", arguments: [false, true])
    func aThinkingOnTurnReusesAThinkingOffTurn(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectTheSecondTurnReusesTheSystemBlock(
                firstThinking: false, withTools: withTools)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The checks

    /// One pass with thinking off renders the system block of thinking on, and
    /// its render ends with the generation prompt and a closed, empty think block.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAThinkingOffTurnRendersTheSystemBlockOfThinkingOn(withTools: Bool)
        async throws
    {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeModel(weights: weights, scripts: [Self.answerScript])
        let store = makeStore()

        _ = try await respond(
            turns: 1, thinking: false, withTools: withTools, model: model, inside: store)

        let entry = try #require(await store.peek(key(of: model)))
        let render = ScriptedByteTokenizer().decode(tokenIds: entry.renderTokens)
        let systemBlock = Self.systemBlock(withTools: withTools)
        #expect(
            render.hasPrefix(systemBlock),
            "The render must start with the system block of thinking on. Render: <<<\(render)>>>")
        #expect(
            render.hasSuffix(
                Qwen35TemplateContractProcessor.assistantHeader
                    + Qwen35TemplateContractProcessor.closedThinkBlock),
            "The render must end with the generation prompt and a closed block. Render: <<<\(render)>>>"
        )
    }

    /// Two passes of one session, the first with thinking `firstThinking` and the
    /// second with the other state. The second pass reuses at least the system
    /// block that the first pass rendered.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectTheSecondTurnReusesTheSystemBlock(
        firstThinking: Bool, withTools: Bool
    ) async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let scripts = [firstThinking, !firstThinking].map {
            $0 ? Self.thinkingScript : Self.answerScript
        }
        let model = makeModel(weights: weights, scripts: scripts)
        let store = makeStore()

        _ = try await respond(
            turns: 1, thinking: firstThinking, withTools: withTools, model: model, inside: store)
        let reused = try await respond(
            turns: 2, thinking: !firstThinking, withTools: withTools, model: model, inside: store)

        let systemBlockLength = Self.systemBlock(withTools: withTools).utf8.count
        #expect(
            reused >= systemBlockLength,
            "The second pass reused \(reused) tokens, under the \(systemBlockLength)-token system block."
        )
    }

    // MARK: - Fixtures

    /// The system block of a render with thinking on.
    private static func systemBlock(withTools: Bool) -> String {
        Qwen35TemplateContractProcessor.systemBlock(
            instructions: instructions, toolCount: withTools ? 1 : 0, thinking: true)
    }

    /// A scripted Qwen 3.5 model whose processor states the template contract.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModel(weights: URL, scripts: [String]) -> MLXLanguageModel {
        ScriptedSessionModel.make(
            weights: weights, scripts: scripts, processor: Qwen35TemplateContractProcessor(),
            reasoningConfig: QwenReasoningProtocol.qwen35)
    }

    /// A store in a folder of its own, whose writer writes nothing.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeStore() -> ExecutorPromptCacheStore {
        ExecutorPromptCacheStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("Qwen35ThinkingOffRenderTests-\(UUID().uuidString)"),
            writer: { _, _ in })
    }

    /// The key of the session of `model`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func key(of model: MLXLanguageModel) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: model.modelID, sessionID: Self.sessionID)
    }

    /// Runs one pass over the instructions and `turns` prompts.
    ///
    /// - Parameters:
    ///   - turns: the number of prompts of the transcript.
    ///   - thinking: whether the request leaves thinking on.
    ///   - withTools: whether the request enables ``tool``.
    ///   - model: the model of the pass.
    ///   - store: the prompt cache store the pass binds.
    /// - Returns: the prompt tokens the pass reused.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func respond(
        turns: Int, thinking: Bool, withTools: Bool, model: MLXLanguageModel,
        inside store: ExecutorPromptCacheStore
    ) async throws -> Int {
        var contextOptions = ContextOptions()
        if !thinking {
            contextOptions.reasoningLevel = .custom(Self.thinkingOffLevelName)
        }
        let request = makeExecutorRequest(
            transcript: Self.transcript(turns: turns),
            enabledTools: withTools ? [Self.tool] : [],
            contextOptions: contextOptions)
        return try await ScriptedExecutorPass.respond(to: request, model: model, inside: store)
            .reusedTokenCount
    }

    /// The instructions of the session and `turns` prompts.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func transcript(turns: Int) -> Transcript {
        let instructionsEntry = Transcript.Instructions(
            id: sessionID, segments: [.text(Transcript.TextSegment(content: instructions))],
            toolDefinitions: [])
        let prompts = (0 ..< turns).map { turn in
            Transcript.Entry.prompt(
                Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "turn \(turn)"))]
                ))
        }
        return Transcript(entries: [.instructions(instructionsEntry)] + prompts)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
