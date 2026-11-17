// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Each generation pass of the executor runs its generation through one helper. The helper
/// gives the generation the prefill recorder and the hand-off of the pass, lets the pass read
/// the stream, and commits the cache of the pass once the generation ended.
///
/// These passes run a scripted generation over a slot. No weights are needed.
@Suite("A committed generation keeps the cache its generation left")
struct ExecutorCommittedGenerationTests {

    /// The prompt of the pass, which ends at the transcript boundary.
    private static let promptTokens = [1, 2, 3]

    /// The token the scripted generation feeds into the caches after the prompt.
    private static let generatedToken = 101

    @Test("the prefill the generation reports becomes the checkpoint of the committed cache")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePrefillTheGenerationReportsBecomesTheCheckpointOfTheCommittedCache() async throws {
        let slot = try await commitScriptedGeneration("committed-generation-checkpoint")

        #expect(slot.entry?.checkpoint?.tokens == Self.promptTokens)
    }

    @Test("the tokens the generation returns extend the ledger of the committed cache")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theTokensTheGenerationReturnsExtendTheLedgerOfTheCommittedCache() async throws {
        let slot = try await commitScriptedGeneration("committed-generation-ledger")

        #expect(slot.entry?.tokens == Self.promptTokens + [Self.generatedToken])
    }

    /// Runs one scripted pass through ``withCommittedGeneration`` and returns the slot it
    /// committed into. The generation feeds the prompt, reports the prefill, feeds
    /// ``generatedToken`` and returns it as the generated tokens of the pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func commitScriptedGeneration(_ modelID: String) async throws
        -> ExecutorPromptCacheSlot
    {
        let executor = try makeMLXExecutor(for: makeStubModel(modelID))
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })
        let planned = try #require(
            try slot.plan(
                input: LMInput(tokens: MLXArray(Self.promptTokens)),
                model: ScriptedLanguageModel(
                    rounds: [], cacheLayerCount: 1, recurrentLayerCount: 1),
                parameters: GenerateParameters(),
                transcriptBoundary: Self.promptTokens.count,
                decodeTokens: { _ in "" }))

        try await executor.withCommittedGeneration(plan: planned, promptCache: slot) {
            preparedState, _ in
            feed(planned.caches, tokenCount: Self.promptTokens.count)
            preparedState(nil) { nil }
            feed(planned.caches, tokenCount: 1)
            let stream = AsyncStream<Int> { $0.finish() }
            return (stream, Task { [Self.generatedToken] })
        } consume: { stream, _ in
            for await _ in stream {}
        }
        return slot
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
