// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Tool-argument schema for these tests. Named `EagerFallbackWeatherArgs`
/// (not the more obvious `WeatherArgs`) because `ToolCallingSchemaTests.swift`
/// already declares a private `@Generable WeatherArgs` in this same test
/// target -- `@Generable`'s macro-generated conformances collide across
/// files on the type name, not just within one file.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct EagerFallbackWeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

/// Thrown by `EagerFallbackProbeProcessor.prepare` when it is asked to
/// render a transcript that includes a `.tool`-role message WITHOUT any
/// `tools` in scope. Standing in for a toolCalling-capable model whose
/// *default* (no-`tools`) chat template can't handle a replayed
/// `.tool`-role message (see `TranscriptConverter.mlxMessages`) -- exactly
/// the eager, unconditional `context.processor.prepare` call
/// `prepareRespondSetup` must skip once the tool-calling branch is going to
/// be taken. `runToolCalling` also calls `context.processor.prepare`, but
/// supplies `tools:` (see `runToolCalling`'s doc comment), which is what a
/// real chat template branches on to render/accept `role == "tool"` --
/// mirrored here by only throwing when `input.tools` is empty.
private struct EagerFallbackProbeError: Error {}

/// A `UserInputProcessor` that throws `EagerFallbackProbeError` if handed
/// any message with `.tool` role and no `tools` in scope. `respond()` must
/// never reach a NO-`tools` call on the tool-calling path: the eager
/// fallback render in `prepareRespondSetup` is skipped once tools are
/// enabled, and `runToolCalling`'s own tool-aware retokenization always
/// supplies `tools:` (at minimum the final-answer tool), so a `.tool`-role
/// message is always paired with a populated `tools` array by the time
/// this processor sees it on that path.
private struct EagerFallbackProbeProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        if case .chat(let messages) = input.prompt,
            messages.contains(where: { $0.role == .tool }),
            (input.tools?.isEmpty ?? true)
        {
            throw EagerFallbackProbeError()
        }
        return LMInput(tokens: MLXArray([Int32(0)]))
    }
}

/// Thrown the instant real generation would start, so this test never
/// needs a real model's weights or vocabulary-sized logits to prove the
/// tool-calling branch was reached.
private struct EagerFallbackGenerationProbeError: Error {}

/// Minimal `LanguageModel` stand-in for `context.model`. Throws from
/// `prepare` before any `MLXArray` computation runs, so
/// `GuidedGenerationLoop.run` (used by the tool-calling path's Phase 2)
/// never calls `callAsFunction` and never needs real weights.
private final class EagerFallbackGenerationProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        throw EagerFallbackGenerationProbeError()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

/// A byte-fallback tokenizer using the SentencePiece `<0xNN>` convention
/// (the same vocab shape `ToolCallingSchemaTests.makeByteTokenizer()` uses
/// to prove real `GrammarConstraint` compilation), so `runToolCalling`'s
/// constraint setup (`TokenizerVocabExtractor.extractForGrammar`, xgrammar
/// compile) runs against a known-compatible vocab instead of an
/// unclassified one.
private struct EagerFallbackByteFallbackTokenizer: MLXLMCommon.Tokenizer {
    private static let vocabSize = 256

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16).map(Int.init)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0 && id < Self.vocabSize else { return nil }
        return String(format: "<0x%02X>", id)
    }

    var bosToken: String? { nil }
    var eosToken: String? { String(format: "<0x%02X>", Self.vocabSize - 1) }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Executor.respond eager fallback vs tool-branch dispatch ordering")
struct EagerFallbackPrepareOrderingTests {

    @Test("Tool-calling branch dispatch isn't blocked by the eager default-template render")
    func toolCallingRequestSkipsEagerFallbackRender() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/eager-fallback-\(UUID().uuidString)"
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [.toolCalling],
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: { configuration, _ in
                // Constructed fresh here (no outer capture) so this
                // `@Sendable` closure never has to carry a non-Sendable
                // model/processor/tokenizer instance across the boundary.
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: EagerFallbackGenerationProbeModel(),
                        processor: EagerFallbackProbeProcessor(),
                        tokenizer: EagerFallbackByteFallbackTokenizer()))
            })
        let executor = try makeMLXExecutor(for: model)

        // A continuation round: TranscriptConverter.mlxMessages replays a
        // prior .toolCalls/.toolOutput round as assistant/tool-role
        // messages, so `messages` here carries a `.tool`-role entry ahead
        // of the fresh prompt -- the exact shape that triggers the bug.
        let toolCall = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Boston, MA"}"#))
        let toolCalls = Transcript.ToolCalls(id: "calls-1", [toolCall])
        let toolOutput = Transcript.ToolOutput(
            id: "call-1", toolName: "get_weather",
            segments: [.text(Transcript.TextSegment(content: "Sunny, 72F"))])
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "And tomorrow?"))],
            responseFormat: nil
        )
        let weatherTool = Transcript.ToolDefinition(
            name: "get_weather", description: "Get current weather",
            parameters: EagerFallbackWeatherArgs.generationSchema
        )

        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .toolCalls(toolCalls), .toolOutput(toolOutput), .prompt(prompt),
            ]),
            enabledTools: [weatherTool]
        )

        let channel = LanguageModelExecutorGenerationChannel()
        // respond() sends a metadata event on the channel before reaching
        // either throw point below; with no concurrent reader, that send
        // would suspend forever.
        let drainer = Task<Void, Never> {
            do {
                for try await _ in channel {}
            } catch {
                // Draining only; the probe error surfaces via respond()'s throw below.
            }
        }
        defer { drainer.cancel() }

        do {
            try await executor.respond(to: request, model: model, streamingInto: channel)
            Issue.record(
                "Expected EagerFallbackGenerationProbeError once dispatch reached real generation"
            )
        } catch is EagerFallbackProbeError {
            Issue.record(
                """
                prepareRespondSetup eagerly rendered the default-template prompt even \
                though the tool-calling branch never reads it -- the ordering fix regressed
                """
            )
        } catch is EagerFallbackGenerationProbeError {
            // Expected: dispatch reached the tool-calling branch's real
            // generation call without the eager fallback ever blocking it.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
