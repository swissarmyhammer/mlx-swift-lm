// Copyright © 2026 Apple Inc.
//
// Actor-level tests for `PromptCache.resolve`: the composition of
// `selectSlot` + `decide` + `applyDecision` + `regenerateLastToken` against
// a real stored slot, driven end-to-end through the actor's public surface.
// `PromptCacheTests` unit-tests each pure helper in isolation; these tests
// pin down what a caller actually receives -- which `KVCache` instance, at
// what offset, with which tokens left to feed -- for every decision branch,
// including the fallback-to-rebuild guards that only composition reaches.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Tests for `PromptCache.resolve`'s slot checkout across every `applyDecision` branch.
@Suite("PromptCache.resolve end-to-end decision application")
struct PromptCacheResolveTests {

    private let model = PromptCacheProbeModel()

    @Test("multi-turn continuation reuses the stored cache unchanged and feeds only the new suffix")
    func continuationReusesCacheAndFeedsSuffix() async {
        let cache = PromptCache()
        let modelID = "resolve-suffix-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        let storedSlot = await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: storedTokens)

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens + [5, 6], model: model)

        #expect(
            cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlot),
            "the stored KVCache instance must be reused")
        #expect(resolved.tokensToFeed == [5, 6])
        #expect(
            resolved.cache[0].offset == storedTokens.count,
            "a pure suffix reuse must not move the cache's offset")
    }

    @Test(
        """
        an identical prompt resent (reuseSuffix count == 0) trims one token and feeds \
        exactly that token again -- the n_past-- single-token regeneration, never a \
        full rebuild
        """
    )
    func identicalPromptRegeneratesLastToken() async {
        let cache = PromptCache()
        let modelID = "resolve-regenerate-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        let storedSlot = await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: storedTokens)

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens, model: model)

        #expect(
            cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlot),
            "the stored KVCache instance must be reused")
        #expect(resolved.tokensToFeed == [4], "exactly the final token must be re-fed")
        #expect(
            resolved.cache[0].offset == storedTokens.count - 1,
            "the cache must be trimmed back one token so re-feeding the last token lands at the stored offset"
        )
    }

    @Test("a diverging prompt trims the stored cache to the common prefix and feeds the new suffix")
    func divergenceTrimsAndFeedsNewSuffix() async {
        let cache = PromptCache()
        let modelID = "resolve-trim-\(UUID().uuidString)"
        let storedSlot = await storeWellFormedSlot(
            cache: cache, modelID: modelID, tokens: [1, 2, 3, 4])

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: [1, 2, 9, 10], model: model)

        #expect(
            cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlot),
            "the stored KVCache instance must be reused")
        #expect(
            resolved.tokensToFeed == [9, 10], "only the tokens beyond the common prefix are fed")
        #expect(
            resolved.cache[0].offset == 2, "the cache must be trimmed back to the common prefix")
    }

    @Test(
        """
        a shrunk prompt fully covered by the stored cache (trimTo with an empty new \
        suffix -- an edited-shorter transcript) trims past the prefix by one and \
        re-feeds that one token
        """
    )
    func shrunkPromptRegeneratesLastCommonToken() async {
        let cache = PromptCache()
        let modelID = "resolve-shrunk-\(UUID().uuidString)"
        let storedSlot = await storeWellFormedSlot(
            cache: cache, modelID: modelID, tokens: [1, 2, 3, 4])

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: [1, 2, 3], model: model)

        #expect(
            cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlot),
            "the stored KVCache instance must be reused")
        #expect(resolved.tokensToFeed == [3], "exactly the last still-common token must be re-fed")
        #expect(
            resolved.cache[0].offset == 2,
            "the cache must land one token short of the shrunk prompt's full length")
    }

    @Test("a diverging prompt against a NON-trimmable stored cache rebuilds from scratch")
    func divergenceAgainstNonTrimmableCacheRebuilds() async {
        let cache = PromptCache()
        let modelID = "resolve-rebuild-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        let slotCache = makeNonTrimmableSlotCache(tokenCount: storedTokens.count)
        await cache.store(modelID: modelID, tokens: storedTokens, cache: SendableBox([slotCache]))

        let newTokens = [1, 2, 9, 10]
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: newTokens, model: model)

        #expect(
            cacheIdentity(resolved: resolved) != ObjectIdentifier(slotCache),
            "a non-trimmable diverged cache must never be handed back to the caller")
        #expect(resolved.tokensToFeed == newTokens, "a rebuild feeds every token from scratch")
        #expect(resolved.cache[0].offset == 0, "a rebuilt cache starts empty")
    }

    @Test(
        """
        an identical prompt resent against a NON-trimmable stored cache falls back to \
        a rebuild -- single-token regeneration requires trimming, which this cache \
        cannot do
        """
    )
    func identicalPromptAgainstNonTrimmableCacheRebuilds() async {
        let cache = PromptCache()
        let modelID = "resolve-regenerate-fallback-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        let slotCache = makeNonTrimmableSlotCache(tokenCount: storedTokens.count)
        await cache.store(modelID: modelID, tokens: storedTokens, cache: SendableBox([slotCache]))

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens, model: model)

        #expect(cacheIdentity(resolved: resolved) != ObjectIdentifier(slotCache))
        #expect(resolved.tokensToFeed == storedTokens, "a rebuild feeds every token from scratch")
    }

    @Test(
        """
        a stored cache whose real offset contradicts its slot's token count fails \
        trim verification and falls back to a rebuild instead of feeding tokens \
        against corrupt KV state
        """
    )
    func inconsistentStoredOffsetFailsVerificationAndRebuilds() async {
        let cache = PromptCache()
        let modelID = "resolve-verify-fallback-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        // Deliberately mispositioned: the slot claims 4 tokens but the cache
        // is really at offset 0, so any trim lands somewhere other than the
        // requested target and must not be trusted.
        let slotCache = KVCacheSimple()
        await cache.store(modelID: modelID, tokens: storedTokens, cache: SendableBox([slotCache]))

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: [1, 2, 9], model: model)

        #expect(
            cacheIdentity(resolved: resolved) != ObjectIdentifier(slotCache),
            "an unverifiable trim must surrender the slot and rebuild")
        #expect(resolved.tokensToFeed == [1, 2, 9], "a rebuild feeds every token from scratch")
    }

    @Test(
        """
        an identical prompt resent against a trimmable but mispositioned cache fails \
        single-token-regeneration's trim verification and rebuilds -- never a \
        one-token feed against corrupt KV state
        """
    )
    func identicalPromptWithMispositionedCacheRebuilds() async {
        let cache = PromptCache()
        let modelID = "resolve-regenerate-verify-fallback-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3, 4]
        // Two layers at MISMATCHED offsets: neither matches the slot's
        // claimed 4 tokens, so `trimAndVerify`'s per-layer `allSatisfy`
        // (exercised multi-layer here, single-layer everywhere else) must
        // reject the trim inside `regenerateLastToken` even though every
        // layer is trimmable.
        let layer0 = KVCacheSimple()
        let layer1 = makeSlotCache(tokenCount: storedTokens.count)
        await cache.store(
            modelID: modelID, tokens: storedTokens, cache: SendableBox([layer0, layer1]))

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens, model: model)

        #expect(
            cacheIdentity(resolved: resolved) != ObjectIdentifier(layer0),
            "an unverifiable regeneration trim must surrender the slot and rebuild")
        #expect(resolved.tokensToFeed == storedTokens, "a rebuild feeds every token from scratch")
    }

    @Test("a checked-out slot is consumed: the same continuation resolved twice only reuses once")
    func checkoutRemovesTheSlot() async {
        let cache = PromptCache()
        let modelID = "resolve-consume-\(UUID().uuidString)"
        let storedTokens = [1, 2, 3]
        let storedSlot = await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: storedTokens)

        let first = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens + [4], model: model)
        #expect(cacheIdentity(resolved: first) == ObjectIdentifier(storedSlot))

        // The slot was removed on checkout and never stored back (as if the
        // first round's generation were still in flight), so a second
        // resolve finds nothing and rebuilds.
        let second = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: storedTokens + [5], model: model)
        #expect(
            cacheIdentity(resolved: second) != ObjectIdentifier(storedSlot),
            "a checked-out slot must not be available to a second caller until stored back")
        #expect(second.tokensToFeed == storedTokens + [5])
    }
}

#endif
