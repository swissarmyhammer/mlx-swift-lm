// Copyright © 2026 Apple Inc.
//
// Tests for the PromptCache-level mechanism `Executor.populatePromptCacheChunks
// (model:transcript:)` (kanban `kr3zkap`) relies on: prewarming a transcript is
// exactly resolve→prefill→store with ZERO generated tokens (prewarm never
// samples a reply -- see that method's doc comment), reusing the same
// `resolvePromptCache`/`storePromptCache` entry points every real `respond()`
// round goes through. The resolve/store-level tests below simulate the exact
// shape a prewarm round produces -- `makeChunkableCache(tokenCount:)`
// positioned at exactly the fed token count, with no extra generated token,
// mirroring `GenerateParameters(maxTokens: 0)`'s zero-token prefill -- against
// the `PromptCacheProbeModel` fixture (whose `prepare`/`callAsFunction`
// intentionally trap -- see `PromptCacheTestSupport.swift`), since a real
// prefill forward pass isn't needed to prove resolve/store dedup and reuse
// behavior.
//
// `populatePromptCacheChunks` ITSELF -- specifically its `isTextOnly` gate --
// IS directly exercisable at this unit level, without a real model's weights:
// `MLXLanguageModel(configuration:capabilities:weightsLocation:load:)` lets a
// test hand it a fully fake `ModelContainer` backed by a custom
// `UserInputProcessor`, exactly the pattern
// `ToolCallingImagePreservationTests.swift`/
// `UnsupportedTranscriptContentMappingTests.swift` already establish for
// probing `context.processor.prepare(input:)` without any download or
// tokenization/generation infrastructure. `populatePromptCacheChunksSkips
// MultimodalTranscriptWithoutError` below reuses that convention: a processor
// that always returns a multimodal `LMInput`, and a generation model that
// traps if ever reached (it must not be -- the `isTextOnly` gate returns
// before `resolvePromptCache`/`generate` ever touch `context.model`).
//
// `PromptCache.resolve`/`.store()` are pre-existing, already-tested entry
// points; nothing here requires a PromptCache.swift code change to pass --
// what's new is proving the specific PREWARM framing the acceptance criteria
// name: populate from a prewarm round, prewarm the same transcript twice
// without duplicating chunks, a first real `respond()` round after prewarm
// reporting `cachedTokenCount > 0` (`PromptCacheSlot.cachedTokenCount` ==
// `promptTokens.count - feedInput.text.tokens.size`, computed here as
// `newTokens.count - tokensToFeed.count`), and a multimodal transcript
// genuinely skipping chunk population through `populatePromptCacheChunks`
// itself, not just at the underlying resolve/store level.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// A `UserInputProcessor` that always returns a multimodal `LMInput` (a
/// non-nil `image`), regardless of the transcript content it's handed --
/// isolates `populatePromptCacheChunks`'s `isTextOnly` gate from real
/// tokenization or image-pixel processing, mirroring
/// `UnsupportedTranscriptContentMappingTests.swift`'s `ImageFailingProcessor`
/// but succeeding (with image content) instead of throwing.
private struct AlwaysMultimodalProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray([Int32(0)])),
            image: .init(pixels: MLXArray([Float(0)])))
    }
}

/// Minimal `LanguageModel` stand-in for `context.model`. Never reached by
/// `populatePromptCacheChunks`'s multimodal-skip test: the `isTextOnly` gate
/// returns before `resolvePromptCache`/`generate` ever touch the model, for
/// an `LMInput` carrying image content.
private final class MultimodalSkipProbeGenerationModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?) throws -> PrepareResult {
        fatalError("not reached -- the isTextOnly gate returns before the model is touched")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not reached -- the isTextOnly gate returns before the model is touched")
    }
}

/// Constructs an `MLXLanguageModel` wired to a fully fake `ModelContainer`
/// (no download, no real weights) whose processor always reports multimodal
/// input -- the same `MLXLanguageModel(configuration:capabilities:
/// weightsLocation:load:)` construction `ToolCallingImagePreservationTests.swift`/
/// `UnsupportedTranscriptContentMappingTests.swift` already establish for
/// probing `context.processor.prepare(input:)` directly.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeMultimodalSkipProbeModel(modelID: String) -> MLXLanguageModel {
    MLXLanguageModel(
        configuration: ModelConfiguration(id: modelID),
        capabilities: [.vision],
        weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
        load: { configuration, _ in
            ModelContainer(
                context: ModelContext(
                    configuration: configuration,
                    model: MultimodalSkipProbeGenerationModel(),
                    processor: AlwaysMultimodalProcessor(),
                    tokenizer: ByteTokenizer()))
        })
}

@Suite("PromptCache prewarm scenario (chunk population from a zero-generated-token prefill)")
struct PromptCachePrewarmScenarioTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    /// Simulates one `populatePromptCacheChunks` round: resolves against
    /// `cache`, then stores a cache positioned at exactly `newTokens.count`
    /// with no extra generated token -- exactly what a `GenerateParameters
    /// (maxTokens: 0)` prefill produces (see `populatePromptCacheChunks`'s
    /// doc comment for why prewarm never generates a real token to
    /// reconcile).
    ///
    /// - Returns: The `resolve()` result from BEFORE this round's store, so
    ///   a caller can assert how much of `newTokens` this round actually had
    ///   to feed.
    private func simulatePrewarmRound(
        cache: PromptCache, modelID: String, newTokens: [Int], model: any MLXLMCommon.LanguageModel
    ) async -> (cache: [KVCache], tokensToFeed: [Int]) {
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: newTokens, model: model)
        await cache.store(
            modelID: modelID, tokens: newTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: newTokens.count)]))
        return resolved
    }

    // MARK: - Populate from a token sequence + fresh cache

    @Test("prewarming a transcript for the first time populates the chunk store")
    func prewarmPopulatesChunkStoreFromFreshCache() async {
        let cache = PromptCache()
        let modelID = "prewarm-populate-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)
        let transcriptTokens = Array(0 ..< (chunkSize * 3))

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "nothing stored before the first prewarm")

        let resolved = await simulatePrewarmRound(
            cache: cache, modelID: modelID, newTokens: transcriptTokens, model: model)

        #expect(
            resolved.tokensToFeed == transcriptTokens,
            "nothing was stored yet, so the first prewarm must prefill the whole transcript")
        #expect(
            await cache.chunkCount(modelID: modelID) > 0,
            "the first prewarm must populate the chunk store")
    }

    // MARK: - Prewarming twice: dedup, near-zero second-round work

    @Test(
        "prewarming the identical transcript twice does not duplicate chunks, and the second prewarm feeds only the capped remainder"
    )
    func prewarmingSameTranscriptTwiceDoesNotDuplicateChunks() async {
        let cache = PromptCache()
        let modelID = "prewarm-dedup-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)
        let transcriptTokens = Array(0 ..< (chunkSize * 3))

        _ = await simulatePrewarmRound(
            cache: cache, modelID: modelID, newTokens: transcriptTokens, model: model)
        let chunkCountAfterFirstPrewarm = await cache.chunkCount(modelID: modelID)
        #expect(chunkCountAfterFirstPrewarm > 0)

        // Second prewarm of the IDENTICAL transcript: resolve must now find
        // the stored chain, so only the capped remainder needs a (still
        // simulated) prefill -- proof of near-zero second-round work, not a
        // full re-feed of the whole transcript.
        let secondResolved = await simulatePrewarmRound(
            cache: cache, modelID: modelID, newTokens: transcriptTokens, model: model)

        #expect(
            secondResolved.tokensToFeed.count < transcriptTokens.count,
            "a second prewarm of the identical transcript must feed only the capped remainder, not the whole transcript again"
        )

        let chunkCountAfterSecondPrewarm = await cache.chunkCount(modelID: modelID)
        #expect(
            chunkCountAfterSecondPrewarm == chunkCountAfterFirstPrewarm,
            "prewarming the identical transcript twice must not duplicate chunks")
    }

    // MARK: - respond() after prewarm reuses on its FIRST round

    @Test(
        "a respond() round whose prompt extends a prewarmed transcript reports cachedTokenCount > 0 on its FIRST resolve"
    )
    func respondAfterPrewarmReportsCachedTokenCountOnFirstRound() async {
        let cache = PromptCache()
        let modelID = "prewarm-then-respond-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)
        let transcriptTokens = Array(0 ..< (chunkSize * 3))

        // Simulate prewarm(transcript:) once, ahead of any respond() round.
        _ = await simulatePrewarmRound(
            cache: cache, modelID: modelID, newTokens: transcriptTokens, model: model)

        // The FIRST respond() round ever, for this model, extends the
        // prewarmed transcript with a new user turn -- exactly the scenario
        // the acceptance criteria require to show reuse before any respond()
        // has run.
        let firstRespondTokens = transcriptTokens + Array(90_000 ..< 90_004)
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: firstRespondTokens, model: model)

        #expect(
            resolved.tokensToFeed == Array(firstRespondTokens.suffix(resolved.tokensToFeed.count)))

        // `PromptCacheSlot.cachedTokenCount` is defined as `promptTokens.count
        // - feedInput.text.tokens.size` (`MLXLanguageModel.swift`); mirror
        // that exact computation here against `resolve()`'s own result.
        let cachedTokenCount = firstRespondTokens.count - resolved.tokensToFeed.count
        #expect(
            cachedTokenCount > 0,
            "cachedTokenCount must be positive on the FIRST respond() round after a prewarm")
    }

    // MARK: - Empty transcript is a no-op, not a crash

    @Test("resolving an empty token sequence never crashes and reports nothing cached")
    func resolvingEmptyTokenSequenceIsSafeNoOp() async {
        let cache = PromptCache()
        let modelID = "prewarm-empty-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: [], model: model)

        #expect(resolved.tokensToFeed.isEmpty)
        #expect(await cache.chunkCount(modelID: modelID) == 0)
    }

    // MARK: - Multimodal transcript skips chunk population, through populatePromptCacheChunks itself

    @Test(
        "populatePromptCacheChunks skips chunk population for a multimodal transcript, without throwing"
    )
    func populatePromptCacheChunksSkipsMultimodalTranscriptWithoutError() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/prewarm-multimodal-skip-\(UUID().uuidString)"
        let model = makeMultimodalSkipProbeModel(modelID: modelID)
        let executor = try makeMLXExecutor(for: model)

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())), label: "photo")
        let multimodalTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "What color is this image?")),
                        .attachment(attachment),
                    ],
                    responseFormat: nil))
        ])

        // Must complete without throwing: `populatePromptCacheChunks`'s
        // `isTextOnly` gate skips chunk population for multimodal input
        // rather than surfacing an error to a caller that only wanted a
        // best-effort prewarm (see that method's doc comment). Reaching
        // `MultimodalSkipProbeGenerationModel.prepare`/`callAsFunction`
        // would `fatalError` -- surviving to the assertion below already
        // proves the gate returned before the model was ever touched.
        try await executor.populatePromptCacheChunks(model: model, transcript: multimodalTranscript)

        // The chunk store must remain untouched for this model: resolving
        // an unrelated token sequence under the same `modelID` finds no
        // stored prefix and must feed it in full, proving population was
        // genuinely skipped rather than silently succeeding.
        let unrelatedTokens = Array(0 ..< 8)
        let resolved = await MLXLanguageModel.resolvePromptCache(
            modelID: modelID, newTokens: unrelatedTokens, model: PromptCacheProbeModel(),
            parameters: nil)
        #expect(
            resolved.tokensToFeed == unrelatedTokens,
            """
            the chunk store must still be empty for this model -- \
            populatePromptCacheChunks must not have stored anything for the \
            skipped multimodal transcript
            """
        )
    }
}

#endif
