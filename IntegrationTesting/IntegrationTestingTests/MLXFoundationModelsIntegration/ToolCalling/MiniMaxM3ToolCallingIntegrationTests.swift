// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state/country, e.g. 'Tokyo, Japan'.")
    var location: String
}

/// Minimum physical memory required to safely attempt loading
/// `mlx-community/MiniMax-M3-4bit` -- mirrors
/// `MiniMaxM3CoherenceIntegrationTests`'s threshold (chain reconciliation
/// across kanban ^wz8y8qq/^b90razv put the checkpoint's real size in the
/// 120-214 GB range; 220 GB leaves headroom above the higher estimate).
private let minimaxM3ToolCallingRequiredMemoryBytes: UInt64 = 220 * 1_024 * 1_024 * 1_024

/// Default checkpoint source (a Hub repo id) used when
/// `MLX_MINIMAX_M3_CHECKPOINT` is not set -- the same override
/// `MiniMaxM3CoherenceIntegrationTests` honors, so a pre-downloaded copy or
/// an MXFP4 variant already pointed at for the coherence test applies here
/// too.
private let defaultMiniMaxM3ToolCallingCheckpointID = "mlx-community/MiniMax-M3-4bit"

/// Resolves the model id to test against, honoring
/// `MLX_MINIMAX_M3_CHECKPOINT` (a Hub id override; local-directory overrides
/// are handled by the coherence test's `ModelConfiguration`-based loader --
/// this suite loads by id through `makeReasoningTestModel`, which only
/// accepts Hub ids).
private func resolveMiniMaxM3ToolCallingModelID() -> String {
    ProcessInfo.processInfo.environment["MLX_MINIMAX_M3_CHECKPOINT"]
        ?? defaultMiniMaxM3ToolCallingCheckpointID
}

/// Gated real-weights tool round trip for MiniMax-M3 (kanban ^ayw1xee):
/// `toolCalls -> toolOutput -> response` end to end through
/// `MLXLanguageModel.Executor`, verifying the namespaced-XML tool-call
/// format (``MiniMaxM3ToolCallParser``) and the toggleable `<mm:think>`
/// reasoning protocol both work against the real checkpoint, and that
/// thinking output lands in `.reasoning` events -- never leaking into the
/// tool call or the final text reply.
///
/// Mirrors `MiniMaxM3CoherenceIntegrationTests`'s gating convention exactly:
/// this is an `IntegrationTesting`-only suite (an Xcode project run via
/// `xcodebuild`, not part of `swift test`) that skips gracefully (passes
/// trivially) rather than failing when the machine lacks enough memory or
/// the checkpoint fails to download/load. Run explicitly via:
/// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/MiniMaxM3ToolCallingIntegrationTests`
@Suite(.serialized, .timeLimit(.minutes(240)))
struct MiniMaxM3ToolCallingIntegrationTests {

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func weatherTool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather in a given location. "
                + "Use this whenever the user asks about weather, temperature, or conditions.",
            parameters: WeatherArgs.generationSchema)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func weatherPromptEntry() -> Transcript.Entry {
        .prompt(
            Transcript.Prompt(
                segments: [
                    .text(Transcript.TextSegment(content: "What's the weather in Tokyo?"))
                ],
                responseFormat: nil))
    }

    /// Whether `text` leaks M3's raw reasoning delimiters -- the acceptance
    /// bar this test exists to defend: thinking must be consumed into
    /// `.reasoning` events, never echoed into tool-call arguments or the
    /// final response text.
    private func leaksReasoningMarkers(_ text: String) -> Bool {
        text.contains("<mm:think>") || text.contains("</mm:think>")
    }

    /// Drains a response stream into its constituent reasoning text, response
    /// text, and (if the model called it) the first tool call's id/name/args.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Collected {
        var reasoning = ""
        var response = ""
        var toolCallID: String?
        var toolCallName: String?
        var toolCallArguments = ""
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collect(_ stream: TestResponseStream) async throws -> Collected {
        var collected = Collected()
        for try await event in stream {
            if let reasoningEvent = reflectedChannelPayload(
                of: event, caseLabel: "reasoning",
                as: LanguageModelExecutorGenerationChannel.Reasoning.self),
                let fragment = reflectedChannelPayload(
                    of: reasoningEvent.action, caseLabel: "appendText",
                    as: LanguageModelExecutorGenerationChannel.TextFragment.self)
            {
                collected.reasoning += fragment.content
            } else if let toolCallsEvent = reflectedChannelPayload(
                of: event, caseLabel: "toolCalls",
                as: LanguageModelExecutorGenerationChannel.ToolCalls.self),
                let toolCall = reflectedChannelPayload(
                    of: toolCallsEvent.action, caseLabel: "toolCall",
                    as: LanguageModelExecutorGenerationChannel.ToolCalls.ToolCall.self)
            {
                if collected.toolCallID == nil {
                    collected.toolCallID = toolCall.id
                    collected.toolCallName = toolCall.name
                }
                if let argsDelta = reflectedChannelPayload(
                    of: toolCall.action, caseLabel: "appendArguments",
                    as: LanguageModelExecutorGenerationChannel.ToolCalls.ToolCall.ArgumentsFragment
                        .self)
                {
                    collected.toolCallArguments += argsDelta.content
                }
            } else if let responseEvent = reflectedChannelPayload(
                of: event, caseLabel: "response",
                as: LanguageModelExecutorGenerationChannel.Response.self),
                let fragment = reflectedChannelPayload(
                    of: responseEvent.action, caseLabel: "appendText",
                    as: LanguageModelExecutorGenerationChannel.TextFragment.self)
            {
                collected.response += fragment.content
            }
        }
        return collected
    }

    @Test
    func minimaxM3ToolRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        guard ProcessInfo.processInfo.physicalMemory >= minimaxM3ToolCallingRequiredMemoryBytes
        else {
            print(
                "Skipping MiniMax-M3 tool round-trip test: physical memory "
                    + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                    + "\(minimaxM3ToolCallingRequiredMemoryBytes) bytes")
            return
        }

        let modelID = resolveMiniMaxM3ToolCallingModelID()
        let model = makeReasoningTestModel(modelID)
        let executor: MLXLanguageModel.Executor
        do {
            executor = try makeMLXExecutor(for: model)
        } catch {
            print(
                "Skipping MiniMax-M3 tool round-trip test: failed to construct executor: \(error)")
            return
        }

        // Force thinking on so the round trip exercises M3's toggleable
        // `thinking_mode: "enabled"` path deterministically, rather than
        // leaving it to the model's own choice under the "adaptive" default.
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = .deep

        // Round 1: prompt + enabled tool -> expect a get_weather tool call.
        let round1Request = makeExecutorRequest(
            transcript: Transcript(entries: [Self.weatherPromptEntry()]),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 1024),
            contextOptions: contextOptions)

        let round1: Collected
        do {
            round1 = try await collect(
                try await executeResponse(executor, request: round1Request, model: model))
        } catch {
            print(
                "Skipping MiniMax-M3 tool round-trip test: failed to load/run checkpoint "
                    + "\(modelID): \(error)")
            return
        }

        #expect(
            round1.toolCallName == "get_weather",
            "expected a get_weather tool call, got name=\(String(describing: round1.toolCallName)) response=\(round1.response.prefix(200))"
        )
        #expect(!round1.toolCallArguments.isEmpty, "expected non-empty tool-call arguments")
        let parsedArguments =
            try? JSONSerialization.jsonObject(with: Data(round1.toolCallArguments.utf8))
            as? [String: Any]
        #expect(
            parsedArguments?["location"] is String,
            "get_weather arguments should carry a string location")

        #expect(
            !leaksReasoningMarkers(round1.toolCallArguments),
            "no <mm:think> markers may leak into tool-call arguments")
        #expect(
            !leaksReasoningMarkers(round1.response),
            "no <mm:think> markers may leak into the response")
        #expect(
            !leaksReasoningMarkers(round1.reasoning),
            "reasoning markers must be consumed, not echoed, in .reasoning text")

        guard let toolCallID = round1.toolCallID, let toolCallName = round1.toolCallName else {
            Issue.record("no tool call id/name captured; cannot continue the round trip")
            return
        }

        // Round 2: feed the tool's output back as history (toolCalls +
        // toolOutput entries correlated by the real id the model/executor
        // generated) and confirm a final text response comes back.
        let toolCall = Transcript.ToolCall(
            id: toolCallID, toolName: toolCallName,
            arguments: try GeneratedContent(json: round1.toolCallArguments))
        let toolOutput = Transcript.ToolOutput(
            id: toolCallID, toolName: toolCallName,
            segments: [.text(Transcript.TextSegment(content: "Sunny, 22°C in Tokyo."))])
        let round2Request = makeExecutorRequest(
            transcript: Transcript(entries: [
                Self.weatherPromptEntry(),
                .toolCalls(Transcript.ToolCalls(id: "mm3-tool-calls-1", [toolCall])),
                .toolOutput(toolOutput),
            ]),
            enabledTools: [Self.weatherTool()],
            generationOptions: GenerationOptions(maximumResponseTokens: 1024),
            contextOptions: contextOptions)

        let round2 = try await collect(
            try await executeResponse(executor, request: round2Request, model: model))

        #expect(
            !round2.response.isEmpty,
            "expected a final text response after the tool output was fed back")
        #expect(
            !leaksReasoningMarkers(round2.response),
            "no <mm:think> markers may leak into the final response")
        #expect(
            !leaksReasoningMarkers(round2.reasoning),
            "reasoning markers must be consumed, not echoed, in round 2's .reasoning text")
    }
}

#endif  // FoundationModelsIntegration
