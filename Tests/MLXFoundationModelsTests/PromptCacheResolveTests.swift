// Copyright © 2026 Apple Inc.
//
// Resolve-suite rewrite for kanban `t71kdmj`: full behavioral replacement
// for the deleted `PromptCacheResolveTests.swift` (old slot-pool semantics,
// removed at the `cthbfmw` cutover), re-established over chunk semantics.
//
// This task's description names five resolve-suite behavioral guarantees.
// Three already have a chunk-semantics equivalent elsewhere and are
// deliberately NOT duplicated here (see each guarantee's cross-reference
// below); this file adds the two that were still missing at the `resolve()`
// entry point a real caller uses:
//
// 1. An extension turn feeds only the post-chunk-boundary suffix, and a
//    second resolve of the same continuation reuses again (no checkout) --
//    `PromptCacheChunkCutoverTests.extensionTurnFeedsSuffixAndReusesOnSecondResolve`.
// 2. Divergence mid-chunk feeds from the last shared chunk boundary --
//    `PromptCacheChunkCutoverTests.divergentContinuationFeedsFromLastSharedChunkBoundary`
//    (chunk-store level: `PromptCacheChunkStoreTests.lookupStopsAtDivergence`).
// 3. Identical prompt feeds exactly the capped remainder (>=1 token, never
//    the whole prompt) -- NEW, see
//    `identicalPromptFeedsCappedRemainderNeverWholePrompt` below.
//    (`PromptCacheChunkStoreTests.lookupCapsAtCountMinusOne` proves the same
//    cap at the `lookupLongestPrefix` level; this test proves it holds
//    through the composed `resolve()` call a real caller makes.)
// 4. An unchunkable cache (`RotatingKVCache` layer) stores nothing, and the
//    next resolve rebuilds --
//    `PromptCacheChunkCutoverTests.unchunkableCacheStoresNothingAndNextResolveRebuilds`.
// 5. Assembled-cache offset correctness after a simulated turn -- NEW, see
//    `assembledCacheOffsetCorrectAfterSimulatedTurn` below.
//
// Real tensor CONTENT is populated (`PromptCacheTestSupport.makeChunkableCache`)
// because `store()` routes through `sliceChunks`, which requires a genuine
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

@Suite("PromptCache.resolve end-to-end behavior (chunk semantics)")
struct PromptCacheResolveTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Identical prompt / capped remainder

    @Test(
        """
        an identical prompt (no extension) is fed exactly the capped remainder -- at \
        least one token, and strictly fewer than the full prompt -- even though every \
        token of the prompt was previously stored
        """
    )
    func identicalPromptFeedsCappedRemainderNeverWholePrompt() async {
        let cache = PromptCache()
        let modelID = "resolve-identical-capped-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)
        let tokens = Array(0 ..< (chunkSize * 3))

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: tokens, model: model)

        let expectedMatchedChunkCount = (tokens.count - 1) / chunkSize
        let expectedFeedCount = tokens.count - expectedMatchedChunkCount * chunkSize

        #expect(resolved.tokensToFeed.count == expectedFeedCount)
        #expect(
            resolved.tokensToFeed.count >= 1,
            "generation always needs at least one fresh token to produce new logits")
        #expect(
            resolved.tokensToFeed.count < tokens.count,
            "an identical, fully-stored prompt must never be fed back in its entirety")
        #expect(resolved.tokensToFeed == Array(tokens.suffix(expectedFeedCount)))
        #expect(resolved.cache[0].offset == expectedMatchedChunkCount * chunkSize)
    }

    // MARK: - Assembled-cache offset correctness

    @Test(
        """
        the assembled cache's offset lands exactly on the matched token count, and \
        stays exact through a simulated generation turn and a follow-up resolve
        """
    )
    func assembledCacheOffsetCorrectAfterSimulatedTurn() async {
        let cache = PromptCache()
        let modelID = "resolve-offset-turn-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        let headDim = 4
        await cache.setChunkSize(chunkSize)

        let firstTurnTokens = Array(0 ..< (chunkSize * 2))
        await cache.store(
            modelID: modelID, tokens: firstTurnTokens,
            cache: SendableBox(
                [makeChunkableCache(tokenCount: firstTurnTokens.count, headDim: headDim)]))

        let secondTurnTokens = firstTurnTokens + Array(80_000 ..< 80_010)
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: secondTurnTokens, model: model)

        let matchedTokenCount = firstTurnTokens.count
        #expect(resolved.cache[0].offset == matchedTokenCount)
        #expect(
            resolved.tokensToFeed
                == Array(secondTurnTokens.suffix(secondTurnTokens.count - matchedTokenCount)))

        // Simulate the rest of the turn: feed every fed token plus one
        // freshly generated token through `update()`, exactly as real
        // generation advances the assembled cache.
        let assembledCache = resolved.cache[0] as! KVCacheSimple
        let newTokenCount = resolved.tokensToFeed.count + 1
        let newValueBase = Int32(50_000)
        let newKeys = MLXArray(
            newValueBase ..< (newValueBase + Int32(newTokenCount * headDim)),
            [1, 1, newTokenCount, headDim])
        let newValues = MLXArray(
            (newValueBase + 1_000) ..< (newValueBase + 1_000 + Int32(newTokenCount * headDim)),
            [1, 1, newTokenCount, headDim])
        _ = assembledCache.update(keys: newKeys, values: newValues)

        let generatedTokenID = 70_000
        let grownTokens = secondTurnTokens + [generatedTokenID]
        #expect(
            assembledCache.offset == grownTokens.count,
            "the assembled cache's offset must advance by exactly the fed-plus-generated token count"
        )

        // Store the grown round back, and confirm a follow-up resolve sees
        // an offset consistent with what was actually just stored.
        await cache.store(
            modelID: modelID, tokens: grownTokens, cache: SendableBox([assembledCache]))

        let thirdTurnTokens = grownTokens + Array(81_000 ..< 81_005)
        let resolvedAgain = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: thirdTurnTokens, model: model)

        // `store()` persists both complete, chunk-aligned chunks AND the
        // trailing partial-span TAIL chunk (kanban 5ra1wzm) -- `grownTokens`
        // is NOT an exact multiple of `chunkSize` here (19 tokens at
        // chunkSize 4), so the stored tail lets this resolve reuse ALL of
        // `grownTokens`, not merely its largest chunk-aligned multiple,
        // capped only by the `thirdTurnTokens.count - 1` invariant.
        let expectedStoredTokenCount = min(grownTokens.count, thirdTurnTokens.count - 1)
        #expect(resolvedAgain.cache[0].offset == expectedStoredTokenCount)
        #expect(
            resolvedAgain.tokensToFeed.count == thirdTurnTokens.count - expectedStoredTokenCount)
    }
}

#endif
