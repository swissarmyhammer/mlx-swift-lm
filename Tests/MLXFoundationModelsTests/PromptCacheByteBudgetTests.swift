// Copyright © 2026 Apple Inc.
//
// Byte-budget LRU eviction suite for kanban `2sdt6dj`: closes the "KNOWN
// WINDOW: no capacity bound" limitation documented in `PromptCache`'s own
// doc comment since the `cthbfmw` cutover -- the chunk store previously grew
// unboundedly. Exercises the three core guarantees the task description
// calls out:
//
// 1. Byte-accounting invariant -- `insertPastBudgetNeverExceedsBudget`:
//    inserting past the configured budget always evicts back down to at or
//    under budget by the time `insert` returns.
// 2. LRU eviction order -- `byteBudgetEvictsLeastRecentlyTouchedLineageFirst`:
//    a lineage touched via `resolve()` survives budget pressure that kills
//    an untouched sibling lineage first, exercised through the real
//    `resolve()`/`store()` call surface (not the lower chunk-store
//    primitives) since that's the actual guarantee callers depend on.
// 3. Orphan-lineage byte reclamation --
//    `evictingLineageHeadReclaimsWholeLineageBytesTransitively`: evicting a
//    chain's head (the globally least-recently-used chunk, since chunks in
//    one `insert` call are stamped in chain order) must reclaim the WHOLE
//    lineage's bytes, not just the head's -- proving the transitive
//    orphan-reclamation strategy (`PromptCache.evictChunkAndDescendants`),
//    not merely that orphaned chunks become unreachable.
//
// Tests 2 exercises the higher-level `resolve()`/`store()` surface (mirrors
// `PromptCacheEvictionScopeTests`'s rationale for testing at that level) and
// therefore needs a real GPU-device eval under plain `swift test` -- see
// `TestBootstrap.swift`'s `MetalLibraryTestBootstrap` (kanban 23ff1zx, memory
// note `swiftpm-test-gpu-metallib-limit`). Tests 1 and 3 exercise the lower
// `insert`/`chunkCount`/`totalStoredByteCount` chunk-store primitives
// directly with hand-crafted `StoredChunk` fixtures (mirroring
// `PromptCacheChunkStoreTests`'s style) so their byte math is exact and
// needs no GPU bootstrap of its own.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache byte-budget LRU eviction")
struct PromptCacheByteBudgetTests {

    // MARK: - Fixtures (mirrors `PromptCacheChunkStoreTests`'s style)

    /// A `StoredChunk` with a placeholder (never content-inspected) tensor
    /// pair and an explicit, caller-controlled `byteSize` -- lets these
    /// tests reason about exact byte totals without depending on real
    /// tensor shapes.
    private func makeStoredChunk(
        tokens: [Int], parentKey: Int, byteSize: Int, lastUsed: Int = 0
    ) -> PromptCache.StoredChunk {
        let key = PromptCache.chunkKey(parentKey: parentKey, tokens: tokens)
        let dummy = MLXArray([Int32(0)])
        return PromptCache.StoredChunk(
            tokens: tokens, layers: [(keys: dummy, values: dummy)], parentKey: parentKey,
            chunkKey: key, byteSize: byteSize, lastUsed: lastUsed)
    }

    /// A hash-chained lineage of `chunkCount` chunks, each `byteSizePerChunk`
    /// bytes, mirroring `sliceChunks`'s chain-key derivation
    /// (`PromptCache.chunkKey(parentKey:tokens:)`) so `lookupLongestPrefix`
    /// would walk it exactly like a genuine stored round.
    private func makeChunkChain(
        chunkCount: Int, chunkSize: Int, byteSizePerChunk: Int, tokenBase: Int = 0
    ) -> [PromptCache.StoredChunk] {
        var parentKey = PromptCache.rootChunkKey
        var chunks: [PromptCache.StoredChunk] = []
        for index in 0 ..< chunkCount {
            let start = tokenBase + index * chunkSize
            let tokens = Array(start ..< (start + chunkSize))
            let chunk = makeStoredChunk(
                tokens: tokens, parentKey: parentKey, byteSize: byteSizePerChunk)
            chunks.append(chunk)
            parentKey = chunk.chunkKey
        }
        return chunks
    }

    // MARK: - 1. Byte-accounting invariant

    @Test("inserting past the byte budget always evicts back down to at or under budget")
    func insertPastBudgetNeverExceedsBudget() async {
        let cache = PromptCache()
        let modelID = "byte-budget-invariant-\(UUID().uuidString)"
        let chunkSize = 4
        let budget = 300
        await cache.setByteBudget(budget)

        // Five independent, single-chunk lineages of 100 bytes each (500
        // total) -- well past the 300-byte budget.
        for index in 0 ..< 5 {
            let chunk = makeStoredChunk(
                tokens: Array((index * 1_000) ..< (index * 1_000 + chunkSize)),
                parentKey: PromptCache.rootChunkKey, byteSize: 100)
            await cache.insert(modelID: modelID, chunks: [chunk])
            // The invariant must hold after EVERY insert returns, not just
            // the last one.
            let total = await cache.totalStoredByteCount()
            #expect(total <= budget, "accounted bytes must never exceed the budget after insert")
        }

        let finalTotal = await cache.totalStoredByteCount()
        #expect(finalTotal <= budget)
        #expect(finalTotal == 300, "eviction should land exactly at budget for these uniform sizes")
    }

    // MARK: - 2. LRU eviction order (through resolve()/store())

    @Test(
        """
        byte-budget pressure evicts the least-recently-touched lineage first; a lineage \
        recently touched via resolve() survives
        """
    )
    func byteBudgetEvictsLeastRecentlyTouchedLineageFirst() async {
        let cache = PromptCache()
        let modelID = "byte-budget-lru-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)

        let touchedTokens = Array(0 ..< (chunkSize * 3))
        let untouchedTokens = Array(50_000 ..< (50_000 + chunkSize * 3))

        // Store "touched" FIRST and "untouched" SECOND, so store order ALONE
        // -- absent the resolve() touch below -- would make "untouched" the
        // more-recently-stamped lineage and "touched" the eviction target.
        // This is deliberate: it isolates the resolve()-driven recency bump
        // as the ONLY mechanism that can make "touched" survive and
        // "untouched" die below. A no-op (or broken) `lastUsed` bump in
        // `lookupLongestPrefix` would leave store order in charge, and the
        // assertions below would fail instead of passing vacuously.
        await cache.store(
            modelID: modelID, tokens: touchedTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: touchedTokens.count)]))
        await cache.store(
            modelID: modelID, tokens: untouchedTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: untouchedTokens.count)]))

        let totalAfterBothStored = await cache.totalStoredByteCount()

        // Touch the "touched" lineage via a real resolve() -- this is the
        // production call path that bumps `lastUsed` for matched chunks --
        // overriding its store-order disadvantage by stamping it with the
        // freshest recency of anything in the store so far.
        let touchedExtended = touchedTokens + Array(90_000 ..< 90_005)
        _ = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: touchedExtended, model: model)

        // Clamp the budget to fit exactly one lineage's worth of bytes (both
        // lineages are the same token count/shape, so they're equal-sized),
        // forcing eviction of exactly one lineage.
        let perLineageBudget = totalAfterBothStored / 2 + 1
        await cache.setByteBudget(perLineageBudget)

        let totalAfterEviction = await cache.totalStoredByteCount()
        #expect(totalAfterEviction <= perLineageBudget)
        #expect(
            await cache.chunkCount(modelID: modelID) == 3,
            "exactly one 3-chunk lineage should remain after evicting the other")

        let untouchedExtended = untouchedTokens + Array(91_000 ..< 91_005)
        let resolvedUntouched = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: untouchedExtended, model: model)
        #expect(
            resolvedUntouched.tokensToFeed == untouchedExtended,
            "the untouched lineage must have been evicted first, forcing a full rebuild")

        let resolvedTouched = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: touchedExtended, model: model)
        #expect(
            resolvedTouched.tokensToFeed.count < touchedExtended.count,
            "the recently-touched lineage must survive and still partially reuse")
    }

    // MARK: - 3. Orphan-lineage byte reclamation

    @Test(
        """
        evicting a lineage's head under budget pressure reclaims the WHOLE lineage's bytes, \
        not just the head's
        """
    )
    func evictingLineageHeadReclaimsWholeLineageBytesTransitively() async {
        let cache = PromptCache()
        let modelID = "byte-budget-orphan-\(UUID().uuidString)"
        let chunkSize = 4
        let chunkCount = 4
        let byteSizePerChunk = 100

        // A single 4-chunk lineage, inserted together in one `insert` call --
        // chunks are stamped in chain order, so the head (parentKey ==
        // rootChunkKey) is the globally least-recently-used entry.
        let lineage = makeChunkChain(
            chunkCount: chunkCount, chunkSize: chunkSize, byteSizePerChunk: byteSizePerChunk)
        await cache.insert(modelID: modelID, chunks: lineage)

        let totalAfterInsert = await cache.totalStoredByteCount()
        #expect(totalAfterInsert == chunkCount * byteSizePerChunk)
        #expect(await cache.chunkCount(modelID: modelID) == chunkCount)

        // Budget is exceeded by only ONE chunk's worth of bytes (400 stored,
        // 350 allowed) -- if reclamation only freed enough bytes to satisfy
        // the budget numerically (i.e. dropped just the head, or the head
        // plus one descendant), the store would still hold now-unreachable
        // orphaned chunks. Transitive reclamation must instead free the
        // WHOLE lineage.
        await cache.setByteBudget(chunkCount * byteSizePerChunk - byteSizePerChunk / 2)

        let totalAfterEviction = await cache.totalStoredByteCount()
        #expect(
            totalAfterEviction == 0,
            "evicting the lineage's head must transitively reclaim every descendant's bytes too")
        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "no orphaned descendants should remain in the store")
    }

    // MARK: - 4. Multi-lineage eviction within a single insert() call

    /// `evictToBudget()` (`PromptCache.swift`) re-scans for a fresh
    /// `globalLRUVictim()` victim after EVERY eviction via a `while` loop, not
    /// just once via an `if` -- so a single `insert(modelID:chunks:)` call
    /// that pushes far enough past budget must evict MULTIPLE independent
    /// lineages before returning, not just one. The other byte-budget tests
    /// in this file never exercise this: `insertPastBudgetNeverExceedsBudget`
    /// only ever inserts ONE chunk per `insert()` call (so at most one
    /// eviction is ever needed per call), and
    /// `byteBudgetEvictsLeastRecentlyTouchedLineageFirst` clamps the budget to
    /// fit exactly one surviving lineage (so exactly one eviction suffices).
    /// A regression that swapped the `while` for an `if` would silently
    /// under-evict in exactly this multi-lineage-per-call scenario while
    /// still passing both of those.
    @Test("a single insert() call evicts more than one lineage when the budget demands it")
    func insertEvictsMultipleLineagesInOneCallWhenBudgetForcesIt() async {
        let cache = PromptCache()
        let modelID = "byte-budget-multi-lineage-\(UUID().uuidString)"
        let chunkSize = 4
        let budget = 250
        await cache.setByteBudget(budget)

        // Five independent, single-chunk lineages of 100 bytes each (500
        // bytes total), all passed to ONE insert() call. Landing at/under the
        // 250-byte budget requires evicting THREE of them (500 -> 400 -> 300
        // -> 200), not just one (which would only reach 400).
        let chunks = (0 ..< 5).map { index in
            makeStoredChunk(
                tokens: Array((index * 1_000) ..< (index * 1_000 + chunkSize)),
                parentKey: PromptCache.rootChunkKey, byteSize: 100)
        }
        await cache.insert(modelID: modelID, chunks: chunks)

        let finalTotal = await cache.totalStoredByteCount()
        #expect(finalTotal <= budget, "a single-eviction bug would land at 400, still over budget")
        #expect(
            finalTotal == 200,
            "evicting exactly 3 of the 5 100-byte lineages lands at 200, the exact point at/under the 250 budget")

        let remainingCount = await cache.chunkCount(modelID: modelID)
        #expect(
            remainingCount == 2,
            """
            3 lineages must have been evicted in this ONE insert() call -- more than the 1 lineage \
            a single-eviction bug (an `if` instead of `evictToBudget`'s `while`) could explain
            """)
    }

    // MARK: - 5. Cross-model LRU eviction

    /// `globalLRUVictim()` (`PromptCache.swift`) scans EVERY model's entries in
    /// `chunkStore`, not just the model being inserted into -- the whole
    /// point of the store's single GLOBAL byte budget shared across models
    /// (see `byteBudget`'s doc comment). The other LRU-ordering test in this
    /// file, `byteBudgetEvictsLeastRecentlyTouchedLineageFirst`, only ever
    /// stores/resolves against ONE `modelID`, so it can't distinguish a
    /// correct global scan from a bug that narrowed the scan to just the
    /// model the triggering `insert()`/`resolve()` call happened to touch.
    /// This test stores chunks under TWO distinct model ids, touches the
    /// older one's chunk so it becomes the globally most-recently-used entry,
    /// then triggers eviction via an `insert()` for that SAME (touched)
    /// model -- proving the victim it picks is the untouched OTHER model's
    /// chunk, not merely the oldest chunk within the model being inserted
    /// into.
    @Test("byte-budget eviction picks its victim across ALL models, not just the model being inserted into")
    func evictionPicksVictimAcrossModelsNotJustTheInsertingModel() async {
        let cache = PromptCache()
        let chunkSize = 4
        let modelA = "byte-budget-cross-model-a-\(UUID().uuidString)"
        let modelB = "byte-budget-cross-model-b-\(UUID().uuidString)"
        let byteSizePerChunk = 100

        // Clamp the budget up front (nothing stored yet, so this evicts
        // nothing): 2.5 chunks' worth, so the store tolerates 2 chunks but
        // not 3.
        await cache.setByteBudget(byteSizePerChunk * 2 + byteSizePerChunk / 2)

        let sharedTokens = Array(0 ..< chunkSize)
        let chunkB = makeStoredChunk(
            tokens: sharedTokens, parentKey: PromptCache.rootChunkKey, byteSize: byteSizePerChunk)
        let chunkA = makeStoredChunk(
            tokens: sharedTokens, parentKey: PromptCache.rootChunkKey, byteSize: byteSizePerChunk)

        // Store B first, then A -- store order alone makes B the OLDER
        // entry, ahead of any touch.
        await cache.insert(modelID: modelB, chunks: [chunkB])
        await cache.insert(modelID: modelA, chunks: [chunkA])

        // Touch model A's chunk via a real `lookupLongestPrefix` walk (the
        // same mechanism `resolve()` uses to bump `lastUsed` on a match),
        // stamping it with the freshest recency of anything stored so far --
        // overriding its store-order disadvantage relative to B.
        let touchTokens = sharedTokens + [chunkSize * 1_000]
        _ = await cache.lookupLongestPrefix(
            modelID: modelA, newTokens: touchTokens, chunkSize: chunkSize
        ).consume()

        // Insert a SECOND chunk for model A -- the model that was just
        // touched. This pushes the GLOBAL total to 300 bytes, over the
        // 250-byte budget, forcing exactly one eviction. If `globalLRUVictim()`
        // incorrectly scanned only model A (the model this `insert()` call is
        // for), it would evict A's own just-touched chunk instead of B's
        // untouched one.
        let extraChunkForA = makeStoredChunk(
            tokens: Array(1_000 ..< (1_000 + chunkSize)), parentKey: PromptCache.rootChunkKey,
            byteSize: byteSizePerChunk)
        await cache.insert(modelID: modelA, chunks: [extraChunkForA])

        let finalTotal = await cache.totalStoredByteCount()
        #expect(finalTotal <= byteSizePerChunk * 2 + byteSizePerChunk / 2)
        #expect(
            await cache.chunkCount(modelID: modelB) == 0,
            """
            model B's untouched chunk must be evicted first, even though the triggering insert() \
            was for model A
            """)
        #expect(
            await cache.chunkCount(modelID: modelA) == 2,
            "model A's touched chunk and its newly-inserted chunk must both survive")
    }

    // MARK: - 6. currentByteBudget() round-trip (review finding, kanban 2sdt6dj)

    /// `currentByteBudget()` (`PromptCache.swift`) had no test consumer of its
    /// own -- unlike its siblings `totalStoredByteCount`/`chunkCount`, both
    /// exercised throughout this file -- flagged by review. Asserts
    /// `setByteBudget(_:)`'s effect is directly observable through the getter,
    /// mirroring `PromptCacheChunkTests`'s `setChunkSize`/getter round-trip
    /// coverage for the sibling `chunkSize` property.
    @Test("setByteBudget's effect is observable via currentByteBudget()")
    func setByteBudgetIsObservableViaCurrentByteBudget() async {
        let cache = PromptCache()
        let budget = 12_345

        await cache.setByteBudget(budget)

        #expect(await cache.currentByteBudget() == budget)
    }

    @Test(
        "setByteBudget clamps non-positive requests up to 1, observable via currentByteBudget()",
        arguments: [0, -3]
    )
    func setByteBudgetClampsNonPositiveToOneObservably(requested: Int) async {
        let cache = PromptCache()

        await cache.setByteBudget(requested)

        #expect(
            await cache.currentByteBudget() == 1,
            "requested byte budget \(requested) must clamp to 1")
    }
}

#endif
