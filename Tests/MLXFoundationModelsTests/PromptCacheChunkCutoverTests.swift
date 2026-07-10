// Copyright © 2026 Apple Inc.
//
// Minimal chunk-path tests for `PromptCache.resolve`/`store` (kanban cthbfmw,
// the production cutover from the old slot pool to the chunk store built by
// `wx4w28j`/`mej3zgh`/`mwwk2bs`). These are deliberately narrow -- the full
// behavioral suites (reuse/divergence/eviction-scope/concurrency, replacing
// what `PromptCacheResolveTests`/`PromptCacheSlotPoolTests`/
// `PromptCacheMultiSessionTests`/`PromptCacheConcurrencyTests` covered for
// the old slot pool) are a follow-up task (kanban `t71kdmj`); this file only
// proves the cutover itself isn't merged untested:
//
// 1. An extension turn feeds just the suffix past the matched chunk prefix.
// 2. A second `resolve()` of the SAME continuation reuses again -- proving
//    there is no checkout (nothing was removed from the chunk store by the
//    first call).
// 3. A divergent continuation feeds from the last shared chunk boundary,
//    even though more chunks existed for the original (non-diverging)
//    continuation.
// 4. An unchunkable cache (`RotatingKVCache` layer) degrades silently:
//    `store` contributes nothing, and the next `resolve` rebuilds.
//
// Real tensor CONTENT is populated (mirroring `PromptCacheChunkTests`'s
// `makeCache`, not `PromptCacheTestSupport`'s offset-only fixtures) because
// `store()` now routes through `sliceChunks`, which requires a genuine
// `KVCacheSimple.state` (not just `offset`) to slice -- this forces a real
// GPU-device eval under plain `swift test`; see `TestBootstrap.swift`'s
// `MetalLibraryTestBootstrap` (kanban 23ff1zx, memory note
// `swiftpm-test-gpu-metallib-limit`) for why the `init()` bootstrap call
// below is required.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache resolve/store chunk-path cutover")
struct PromptCacheChunkCutoverTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixtures

    /// A fully-packed `KVCacheSimple` layer covering exactly `tokenCount`
    /// sequence positions, mirroring `PromptCacheChunkTests.makeCache`:
    /// setting `state` directly (rather than via `update(keys:values:)`)
    /// leaves `offset == keys.dim(2) == tokenCount` exactly, the "verified"
    /// shape `sliceChunks` (and therefore `store()`) requires.
    private func makeCache(tokenCount: Int, headDim: Int = 4) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let count = tokenCount * headDim
        let keys = MLXArray(Int32(0) ..< Int32(count), [1, 1, tokenCount, headDim])
        let values = MLXArray(
            Int32(10_000) ..< Int32(10_000 + count), [1, 1, tokenCount, headDim])
        cache.state = [keys, values]
        return cache
    }

    // MARK: - Extension turn / no-checkout reuse

    @Test(
        """
        an extension turn feeds only the suffix past the matched chunk prefix, and a second \
        resolve of the SAME continuation reuses again (no checkout occurred)
        """
    )
    func extensionTurnFeedsSuffixAndReusesOnSecondResolve() async {
        let cache = PromptCache()
        let modelID = "chunk-cutover-extend-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = PromptCache.defaultChunkSize
        let firstTurnTokens = Array(0 ..< (chunkSize * 3))

        await cache.store(
            modelID: modelID, tokens: firstTurnTokens,
            cache: SendableBox([makeCache(tokenCount: firstTurnTokens.count)]))

        let extensionTokens = firstTurnTokens + Array(90_000 ..< 90_005)
        let expectedMatchedChunkCount = firstTurnTokens.count / chunkSize
        let expectedFeedCount = extensionTokens.count - expectedMatchedChunkCount * chunkSize

        let firstResolve = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: extensionTokens, model: model)
        #expect(firstResolve.tokensToFeed == Array(extensionTokens.suffix(expectedFeedCount)))
        #expect(firstResolve.cache[0].offset == expectedMatchedChunkCount * chunkSize)

        // Second resolve of the identical continuation: nothing was checked
        // out (removed) by the first call, so the chunk store still serves
        // the same prefix from chunks.
        let secondResolve = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: extensionTokens, model: model)
        #expect(
            secondResolve.tokensToFeed.count == expectedFeedCount,
            "a second resolve of the same continuation must still be served from chunks, proving no checkout occurred"
        )
        #expect(secondResolve.cache[0].offset == expectedMatchedChunkCount * chunkSize)
    }

    // MARK: - Divergence

    @Test("a divergent continuation feeds from the last shared chunk boundary")
    func divergentContinuationFeedsFromLastSharedChunkBoundary() async {
        let cache = PromptCache()
        let modelID = "chunk-cutover-diverge-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = PromptCache.defaultChunkSize
        let firstTurnTokens = Array(0 ..< (chunkSize * 3))

        await cache.store(
            modelID: modelID, tokens: firstTurnTokens,
            cache: SendableBox([makeCache(tokenCount: firstTurnTokens.count)]))

        // Shares the first two chunks verbatim, then diverges for the rest
        // of what would have been the third chunk plus a further tail --
        // proving the walk stops at the LAST shared boundary even though a
        // third chunk exists in the store for the non-diverging continuation.
        let sharedPrefix = Array(firstTurnTokens.prefix(chunkSize * 2))
        let divergentTail = Array(99_000 ..< 99_070)
        let divergentTokens = sharedPrefix + divergentTail

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: divergentTokens, model: model)

        #expect(resolved.tokensToFeed == divergentTail)
        #expect(resolved.cache[0].offset == sharedPrefix.count)
    }

    // MARK: - Unchunkable degradation

    @Test("an unchunkable cache (RotatingKVCache layer) stores nothing, and the next resolve rebuilds")
    func unchunkableCacheStoresNothingAndNextResolveRebuilds() async {
        let cache = PromptCache()
        let modelID = "chunk-cutover-unchunkable-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let tokens = Array(0 ..< (PromptCache.defaultChunkSize * 2))

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeNonTrimmableSlotCache(tokenCount: tokens.count)]))

        #expect(await cache.chunkCount(modelID: modelID) == 0)

        let resolved = await resolveOnce(cache: cache, modelID: modelID, newTokens: tokens, model: model)
        #expect(
            resolved.tokensToFeed == tokens,
            "no chunks stored means the next resolve rebuilds and feeds the full prompt")
    }
}

#endif
