// Copyright © 2026 Apple Inc.
//
// Slot-pool bookkeeping tests for `PromptCache`: LRU eviction at the
// `maxSlotsPerModel` cap, the `evictAll`/`remove` invalidation API, and
// per-model isolation of stored slots. Slot storage is private to the
// actor, so every assertion goes through observable `resolve()` behavior:
// a surviving slot hands back the exact stored `KVCache` instance
// (`ObjectIdentifier` equality), an evicted or invalidated one forces a
// rebuild (a fresh instance fed every token).

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Tests for `PromptCache.store`'s LRU eviction at the `maxSlotsPerModel` cap.
@Suite("PromptCache slot-pool LRU eviction")
struct PromptCacheLRUEvictionTests {

    private let model = PromptCacheProbeModel()

    /// Disjoint token prefixes (distinct first token) so no two slots ever
    /// share a common prefix -- each resolve targets exactly one slot.
    private func sessionTokens(_ session: Int) -> [Int] {
        [100 + session, 1, 2]
    }

    @Test("storing one slot past the cap evicts exactly the least-recently-used slot")
    func overflowEvictsLeastRecentlyUsed() async {
        let cache = PromptCache()
        let modelID = "lru-overflow-\(UUID().uuidString)"

        // maxSlotsPerModel + 1 slots, stored oldest-first.
        let sessionCount = PromptCache.defaultMaxSlotsPerModel + 1
        var storedSlots: [KVCacheSimple] = []
        for session in 0 ..< sessionCount {
            storedSlots.append(
                await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(session)))
        }

        // The oldest slot (session 0) must be gone: its continuation rebuilds.
        let evicted = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sessionTokens(0) + [9], model: model)
        #expect(
            cacheIdentity(resolved: evicted) != ObjectIdentifier(storedSlots[0]),
            "the least-recently-used slot must have been evicted at the cap")
        #expect(evicted.tokensToFeed == sessionTokens(0) + [9])

        // Every newer slot survived: each continuation reuses its exact instance.
        for session in 1 ..< sessionCount {
            let resolved = await resolveOnce(
                cache: cache, modelID: modelID, newTokens: sessionTokens(session) + [9], model: model)
            #expect(
                cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlots[session]),
                "slot for session \(session) should have survived LRU eviction")
        }
    }

    @Test("a resolve+store round-trip refreshes a slot's recency, protecting it from eviction")
    func touchRefreshesRecency() async throws {
        let cache = PromptCache()
        let modelID = "lru-touch-\(UUID().uuidString)"

        // Fill to the cap, oldest-first: session 0 is the LRU slot.
        var storedSlots: [KVCacheSimple] = []
        for session in 0 ..< PromptCache.defaultMaxSlotsPerModel {
            storedSlots.append(
                await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(session)))
        }

        // Touch session 0: check its slot out and store it back grown by one
        // turn -- exactly what a completed generation round does. Its
        // recency stamp is now the newest, making session 1 the LRU slot.
        let touchedTokens = sessionTokens(0) + [7]
        let touched = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: touchedTokens, model: model)
        #expect(cacheIdentity(resolved: touched) == ObjectIdentifier(storedSlots[0]))
        // `KVCache.offset` is get-only through the protocol; the reused
        // instance is the `KVCacheSimple` this test stored.
        try #require(touched.cache[0] as? KVCacheSimple).offset = touchedTokens.count
        await cache.store(
            modelID: modelID, tokens: touchedTokens, cache: SendableBox(touched.cache))

        // A fifth prefix overflows the cap.
        await storeWellFormedSlot(
            cache: cache, modelID: modelID, tokens: sessionTokens(PromptCache.defaultMaxSlotsPerModel))

        // Session 1 (now least recently used) was evicted...
        let evicted = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sessionTokens(1) + [9], model: model)
        #expect(
            cacheIdentity(resolved: evicted) != ObjectIdentifier(storedSlots[1]),
            "the untouched oldest slot must be the one evicted")

        // ...while the touched session 0 survived.
        let survivor = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: touchedTokens + [9], model: model)
        #expect(
            cacheIdentity(resolved: survivor) == ObjectIdentifier(storedSlots[0]),
            "the freshly-touched slot must have been protected from eviction")
    }

    @Test("lowering the slot limit evicts at the new cap on the next store")
    func loweredSlotLimitEvictsAtNewCap() async {
        let cache = PromptCache()
        let modelID = "lru-lowered-\(UUID().uuidString)"
        await cache.setMaxSlotsPerModel(2)

        var storedSlots: [KVCacheSimple] = []
        for session in 0 ..< 3 {
            storedSlots.append(
                await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(session)))
        }

        // With the cap at 2, the oldest of the three must be gone...
        let evicted = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sessionTokens(0) + [9], model: model)
        #expect(
            cacheIdentity(resolved: evicted) != ObjectIdentifier(storedSlots[0]),
            "with the limit lowered to 2, storing a third slot must evict the oldest")

        // ...and the two newest must both survive.
        for session in 1 ..< 3 {
            let resolved = await resolveOnce(
                cache: cache, modelID: modelID, newTokens: sessionTokens(session) + [9], model: model)
            #expect(
                cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlots[session]),
                "slot for session \(session) must survive under the lowered limit")
        }
    }

    @Test("raising the slot limit retains more conversations than the default")
    func raisedSlotLimitRetainsMoreThanDefault() async {
        let cache = PromptCache()
        let modelID = "lru-raised-\(UUID().uuidString)"
        let limit = PromptCache.defaultMaxSlotsPerModel + 2
        await cache.setMaxSlotsPerModel(limit)

        var storedSlots: [KVCacheSimple] = []
        for session in 0 ..< limit {
            storedSlots.append(
                await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(session)))
        }

        for session in 0 ..< limit {
            let resolved = await resolveOnce(
                cache: cache, modelID: modelID, newTokens: sessionTokens(session) + [9], model: model)
            #expect(
                cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlots[session]),
                "with the limit raised to \(limit), all \(limit) slots must survive (session \(session))"
            )
        }
    }

    @Test(
        "a zero or negative slot limit clamps to one retained conversation, never zero",
        arguments: [0, -3])
    func nonPositiveSlotLimitClampsToOne(limit: Int) async {
        let cache = PromptCache()
        let modelID = "lru-clamped-\(UUID().uuidString)"
        await cache.setMaxSlotsPerModel(limit)

        let older = await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(0))
        let newer = await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(1))

        // Exactly one slot retained: the newest survives, the older is gone.
        let resolvedNewer = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sessionTokens(1) + [9], model: model)
        #expect(
            cacheIdentity(resolved: resolvedNewer) == ObjectIdentifier(newer),
            "limit \(limit) must clamp to 1 and still retain the newest conversation")
        let resolvedOlder = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sessionTokens(0) + [9], model: model)
        #expect(
            cacheIdentity(resolved: resolvedOlder) != ObjectIdentifier(older),
            "limit \(limit) must clamp to 1 and evict everything but the newest conversation")
    }

    @Test("storing exactly maxSlotsPerModel slots evicts nothing")
    func atCapNothingEvicted() async {
        let cache = PromptCache()
        let modelID = "lru-at-cap-\(UUID().uuidString)"

        var storedSlots: [KVCacheSimple] = []
        for session in 0 ..< PromptCache.defaultMaxSlotsPerModel {
            storedSlots.append(
                await storeWellFormedSlot(cache: cache, modelID: modelID, tokens: sessionTokens(session)))
        }

        for session in 0 ..< PromptCache.defaultMaxSlotsPerModel {
            let resolved = await resolveOnce(
                cache: cache, modelID: modelID, newTokens: sessionTokens(session) + [9], model: model)
            #expect(
                cacheIdentity(resolved: resolved) == ObjectIdentifier(storedSlots[session]),
                "at the cap, no slot should have been evicted (session \(session))")
        }
    }
}

/// Tests for `PromptCache.evictAll`/`remove` and per-model slot isolation.
@Suite("PromptCache eviction API and per-model isolation")
struct PromptCacheEvictionScopeTests {

    private let model = PromptCacheProbeModel()

    @Test("evictAll drops every model's slots")
    func evictAllDropsAllModels() async {
        let cache = PromptCache()
        let modelA = "evict-all-a-\(UUID().uuidString)"
        let modelB = "evict-all-b-\(UUID().uuidString)"
        let tokensA = [1, 2, 3]
        let tokensB = [4, 5, 6]
        let storedSlotA = await storeWellFormedSlot(cache: cache, modelID: modelA, tokens: tokensA)
        let storedSlotB = await storeWellFormedSlot(cache: cache, modelID: modelB, tokens: tokensB)

        await cache.evictAll()

        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: tokensA + [9], model: model)
        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: tokensB + [9], model: model)
        #expect(
            cacheIdentity(resolved: resolvedA) != ObjectIdentifier(storedSlotA),
            "evictAll must drop model A's slot")
        #expect(
            cacheIdentity(resolved: resolvedB) != ObjectIdentifier(storedSlotB),
            "evictAll must drop model B's slot")
        #expect(
            resolvedA.tokensToFeed == tokensA + [9], "a post-eviction round rebuilds from scratch")
        #expect(
            resolvedB.tokensToFeed == tokensB + [9], "a post-eviction round rebuilds from scratch")
    }

    @Test("remove drops only the named model's slots, leaving other models' slots reusable")
    func removeIsPerModel() async {
        let cache = PromptCache()
        let modelA = "remove-a-\(UUID().uuidString)"
        let modelB = "remove-b-\(UUID().uuidString)"
        let tokensA = [1, 2, 3]
        let tokensB = [4, 5, 6]
        let storedSlotA = await storeWellFormedSlot(cache: cache, modelID: modelA, tokens: tokensA)
        let storedSlotB = await storeWellFormedSlot(cache: cache, modelID: modelB, tokens: tokensB)

        await cache.remove(modelID: modelA)

        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: tokensA + [9], model: model)
        #expect(
            cacheIdentity(resolved: resolvedA) != ObjectIdentifier(storedSlotA),
            "remove must drop the named model's slot")

        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: tokensB + [9], model: model)
        #expect(
            cacheIdentity(resolved: resolvedB) == ObjectIdentifier(storedSlotB),
            "remove must NOT disturb a different model's slot")
    }

    @Test(
        """
        slots are keyed strictly per model: an identical token sequence against a \
        different model id never reuses another model's KV state, and leaves it intact
        """
    )
    func identicalTokensNeverCrossModels() async {
        let cache = PromptCache()
        let modelA = "isolation-a-\(UUID().uuidString)"
        let modelB = "isolation-b-\(UUID().uuidString)"
        let tokens = [1, 2, 3, 4]
        let storedSlotA = await storeWellFormedSlot(cache: cache, modelID: modelA, tokens: tokens)

        // Model B asks with the exact same tokens: KV state is conditioned
        // on the MODEL, not just the token sequence, so handing model A's
        // cache to model B would silently corrupt B's attention.
        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: tokens + [5], model: model)
        #expect(
            cacheIdentity(resolved: resolvedB) != ObjectIdentifier(storedSlotA),
            "model A's KVCache instance must never be handed out for model B")
        #expect(resolvedB.tokensToFeed == tokens + [5], "model B starts from scratch")

        // And model A's slot is still there for model A.
        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: tokens + [5], model: model)
        #expect(
            cacheIdentity(resolved: resolvedA) == ObjectIdentifier(storedSlotA),
            "model A's slot must be untouched by model B's resolve")
    }
}

#endif
