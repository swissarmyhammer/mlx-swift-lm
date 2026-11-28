// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Proves that a named reasoning level of a request gets to the template as
/// `reasoning_effort`, on the plain path and on the tool path (decision D218).
///
/// The scripted processor states the contract of the Qwen3.8 template
/// (``Qwen35TemplateContractProcessor``): the reasoning instructions name the level of
/// `reasoning_effort`, and `xhigh` when it is undefined. A request with no level, or with
/// thinking off, renders as before. A protocol with no effort key ignores the level.
///
/// The executor is available from iOS 27, macOS 27 and visionOS 27. On an earlier system
/// each test records an issue, thus it fails and does not pass with no assertion.
@Suite("A named reasoning level gets to the template as reasoning_effort")
struct ReasoningEffortLevelTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    /// The identifier of the first entry, which names the session.
    private static let sessionID = "effort-session"

    /// The instructions of the session.
    private static let instructions = "Be terse."

    /// The text a pass with thinking on writes.
    private static let thinkingScript = "plan</think>A"

    /// The text a pass with thinking off writes.
    private static let answerScript = "A"

    /// The level name that turns thinking off for one request.
    private static let thinkingOffLevelName = "no_think"

    /// The named level of the tests.
    private static let level = "low"

    /// The one tool of a pass on the tool path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var tool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "lookup", description: "Looks a word up.", parameters: String.generationSchema)
    }

    // MARK: - The level of a request

    @Test func `a custom level names the effort`() {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let effort = MLXLanguageModel.Executor.reasoningEffort(for:)
            #expect(effort(.custom("low")) == "low")
            #expect(effort(.custom(" Medium\n")) == "medium")
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test func `no level, the named cases, no_think and an empty name name no effort`() {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let effort = MLXLanguageModel.Executor.reasoningEffort(for:)
            #expect(effort(nil) == nil)
            #expect(effort(.light) == nil)
            #expect(effort(.moderate) == nil)
            #expect(effort(.deep) == nil)
            #expect(effort(.custom("no_think")) == nil)
            #expect(effort(.custom("  ")) == nil)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The render

    @Test("a named level renders its reasoning instructions", arguments: [false, true])
    func aNamedLevelRendersItsInstructions(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let render = try await renderOfOnePass(
                level: .custom(Self.level), withTools: withTools)
            #expect(
                render.hasPrefix(Self.systemBlock(withTools: withTools, effort: Self.level)),
                "Render: <<<\(render)>>>")
            #expect(
                render.hasSuffix(
                    Qwen35TemplateContractProcessor.assistantHeader
                        + Qwen35TemplateContractProcessor.openThinkBlock),
                "Render: <<<\(render)>>>")
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    /// `deep` is false for a request with no level, and true for the level `.deep`.
    @Test(
        "no level and a named case render the default instructions",
        arguments: [false, true], [false, true])
    func noLevelRendersTheDefaultInstructions(deep: Bool, withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let render = try await renderOfOnePass(
                level: deep ? .deep : nil, withTools: withTools)
            #expect(
                render.hasPrefix(
                    Self.systemBlock(
                        withTools: withTools,
                        effort: Qwen35TemplateContractProcessor.defaultEffort)),
                "Render: <<<\(render)>>>")
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("thinking off renders as before", arguments: [false, true])
    func thinkingOffRendersAsBefore(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let render = try await renderOfOnePass(
                level: .custom(Self.thinkingOffLevelName), withTools: withTools)
            #expect(
                !render.contains(
                    Qwen35TemplateContractProcessor.reasoningInstructions(effort: Self.level)))
            #expect(
                render.hasSuffix(
                    Qwen35TemplateContractProcessor.assistantHeader
                        + Qwen35TemplateContractProcessor.closedThinkBlock),
                "Render: <<<\(render)>>>")
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a protocol with no effort key ignores the level", arguments: [false, true])
    func aProtocolWithNoEffortKeyIgnoresTheLevel(withTools: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let config = QwenReasoningProtocol.qwen3
            let withLevel = try await renderOfOnePass(
                level: .custom(Self.level), withTools: withTools, config: config)
            let withoutLevel = try await renderOfOnePass(
                level: nil, withTools: withTools, config: config)
            #expect(withLevel == withoutLevel, "Render: <<<\(withLevel)>>>")
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - Fixtures

    /// The prompt that one pass of a new session renders.
    ///
    /// - Parameters:
    ///   - level: the reasoning level of the request.
    ///   - withTools: whether the request enables ``tool``.
    ///   - config: the reasoning protocol of the model.
    /// - Returns: the text of the render that the store holds after the pass.
    /// - Throws: the error of the executor, or an issue when the store holds no entry.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func renderOfOnePass(
        level: ContextOptions.ReasoningLevel?, withTools: Bool,
        config: ReasoningConfig = QwenReasoningProtocol.qwen35
    ) async throws -> String {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let thinking = MLXLanguageModel.Executor.thinkingEnabled(for: level) != false
        let model = ScriptedSessionModel.make(
            weights: weights, scripts: [thinking ? Self.thinkingScript : Self.answerScript],
            processor: Qwen35TemplateContractProcessor(), reasoningConfig: config)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "ReasoningEffortLevelTests")

        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = level
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                TestTranscript.instructions(id: Self.sessionID, Self.instructions),
                TestTranscript.prompt("turn 0"),
            ]),
            enabledTools: withTools ? [Self.tool] : [],
            contextOptions: contextOptions)
        _ = try await ScriptedExecutorPass.respond(to: request, model: model, inside: store)

        let key = PromptCacheTestKey.make(
            modelID: .init(model.modelID), sessionID: .init(Self.sessionID))
        let entry = try #require(await store.peek(key))
        return ScriptedByteTokenizer().decode(tokenIds: entry.renderTokens)
    }

    /// The system block of a render with thinking on at `effort`.
    private static func systemBlock(withTools: Bool, effort: String) -> String {
        Qwen35TemplateContractProcessor.systemBlock(
            instructions: instructions, toolCount: withTools ? 1 : 0, thinking: true,
            effort: effort)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
