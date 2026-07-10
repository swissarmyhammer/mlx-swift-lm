// Copyright © 2026 Apple Inc.
//
// Tests for `PromptCache`'s per-model CHUNK STORE (kanban mej3zgh): `insert`
// (dedup-on-key, refresh `lastUsed`, never replace tensors) and
// `lookupLongestPrefix` (hash-chain walk over chunk-aligned windows, capped
// at `newTokens.count - 1` so generation always has >=1 fresh token to feed,
// collision-safety verified by a token-array compare on every key match).
//
// This chunk store IS the production path: `resolve()`/`store()` call
// straight into `lookupLongestPrefix`/`insert` (the earlier slot-pool
// mechanism -- `entries: [String: [Slot]]`, `selectSlot`, `decide` -- was
// deleted once the cutover to chunks landed, kanban cthbfmw).
// `MLXLanguageModel.swift`'s `Executor` calls `resolve`/`store`/`evictAll`/
// `remove`, which now assemble cache state from these chunks on every call.
// These tests exercise the chunk store in isolation, independent of
// `resolve()`/`store()`'s assemble/slice logic (see
// `PromptCacheChunkCutoverTests` for that wiring).
//
// No fixture here evaluates real tensor CONTENT (unlike `PromptCacheChunkTests`,
// which slices real `KVCacheSimple` stacks and checks tensor values) -- chunks
// are constructed directly with placeholder `MLXArray`s, since chunk-store
// bookkeeping (dedup, chain walk, collision safety, eviction scope) never
// inspects tensor content, only `tokens`/`chunkKey`/`parentKey`/`lastUsed`.
//
// `lookupLongestPrefix`'s result is boxed in a `SendableBox` (its
// `StoredChunk`s carry non-`Sendable` `MLXArray` tensors, mirroring
// `resolve()`'s boxed `[KVCache]` result) -- every call site below
// `.consume()`s it immediately, exactly once.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache chunk store")
struct PromptCacheChunkStoreTests {

    // MARK: - Fixtures

    /// A `StoredChunk` with a placeholder (never content-inspected) tensor pair.
    ///
    /// `forcedKey` lets a test store a chunk under a key that does NOT match
    /// `hash(parentKey, tokens)` -- simulating a genuine 64-bit hash collision
    /// without needing to find a real one (see `forcedCollisionTreatedAsMiss`).
    private func makeStoredChunk(
        tokens: [Int], parentKey: Int, forcedKey: Int? = nil, byteSize: Int = 64,
        lastUsed: Int = 0
    ) -> PromptCache.StoredChunk {
        let key = forcedKey ?? PromptCache.chunkKey(parentKey: parentKey, tokens: tokens)
        let dummy = MLXArray([Int32(0)])
        return PromptCache.StoredChunk(
            tokens: tokens, layers: [(keys: dummy, values: dummy)], parentKey: parentKey,
            chunkKey: key, byteSize: byteSize, lastUsed: lastUsed)
    }

    /// Slices `tokens` into a hash-chained `StoredChunk` sequence the same way
    /// `PromptCache.sliceChunks` would (full `chunkSize`-token spans only, any
    /// trailing partial span dropped) -- but without needing a real
    /// `KVCacheSimple` stack, since these tests never inspect tensor content.
    private func makeChunkChain(tokens: [Int], chunkSize: Int) -> [PromptCache.StoredChunk] {
        var parentKey = PromptCache.rootChunkKey
        var chunks: [PromptCache.StoredChunk] = []
        let chunkCount = tokens.count / chunkSize
        for index in 0 ..< chunkCount {
            let start = index * chunkSize
            let end = start + chunkSize
            let chunk = makeStoredChunk(tokens: Array(tokens[start ..< end]), parentKey: parentKey)
            chunks.append(chunk)
            parentKey = chunk.chunkKey
        }
        return chunks
    }

    // MARK: - Dedup

    @Test("storing two conversations sharing a 3-chunk prefix keeps the shared chunks only once")
    func dedupSharedPrefixStoredOnce() async {
        let cache = PromptCache()
        let modelID = "chunk-dedup-\(UUID().uuidString)"
        let chunkSize = 4
        let sharedTokens = Array(0 ..< (chunkSize * 3))
        let conversationA = sharedTokens + Array(1_000 ..< 1_004)
        let conversationB = sharedTokens + Array(2_000 ..< 2_004)

        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: conversationA, chunkSize: chunkSize))
        let countAfterFirst = await cache.chunkCount(modelID: modelID)

        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: conversationB, chunkSize: chunkSize))
        let countAfterSecond = await cache.chunkCount(modelID: modelID)

        #expect(countAfterFirst == 4, "conversation A alone contributes its own 4 chunks")
        #expect(
            countAfterSecond == 5,
            "only B's one diverging chunk should be new; the 3 shared chunks must dedup, not double"
        )
    }

    @Test("dedup on an already-present key refreshes lastUsed but never replaces the stored tensors")
    func dedupRefreshesRecencyWithoutReplacingTensors() async {
        let cache = PromptCache()
        let modelID = "chunk-dedup-tensors-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        let key = PromptCache.chunkKey(parentKey: PromptCache.rootChunkKey, tokens: tokens)
        let originalArray = MLXArray([Int32(1), 2, 3, 4])
        let original = PromptCache.StoredChunk(
            tokens: tokens, layers: [(keys: originalArray, values: originalArray)],
            parentKey: PromptCache.rootChunkKey, chunkKey: key, byteSize: 64, lastUsed: 0)
        await cache.insert(modelID: modelID, chunks: [original])

        let replacementArray = MLXArray([Int32(9), 9, 9, 9])
        let replacement = PromptCache.StoredChunk(
            tokens: tokens, layers: [(keys: replacementArray, values: replacementArray)],
            parentKey: PromptCache.rootChunkKey, chunkKey: key, byteSize: 64, lastUsed: 0)
        await cache.insert(modelID: modelID, chunks: [replacement])

        #expect(await cache.chunkCount(modelID: modelID) == 1)
        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
        ).consume()
        #expect(matched.count == 1)
        #expect(
            ObjectIdentifier(matched[0].layers[0].keys) == ObjectIdentifier(originalArray),
            "dedup must retain the ORIGINAL tensor instance, never swap in the replacement's tensors"
        )
    }

    @Test("inserting an already-present chunk key refreshes its lastUsed recency stamp")
    func dedupRefreshesLastUsed() async {
        let cache = PromptCache()
        let modelID = "chunk-dedup-recency-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        let chunk = makeChunkChain(tokens: tokens, chunkSize: chunkSize)[0]
        await cache.insert(modelID: modelID, chunks: [chunk])

        let firstLastUsed =
            await cache.lookupLongestPrefix(
                modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
            ).consume()[0].lastUsed

        await cache.insert(modelID: modelID, chunks: [chunk])

        let secondLastUsed =
            await cache.lookupLongestPrefix(
                modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
            ).consume()[0].lastUsed

        #expect(secondLastUsed > firstLastUsed, "re-inserting a present key must refresh lastUsed")
    }

    // MARK: - Chain walk / longest prefix

    @Test("lookup returns the longest matching chain prefix, stopping at the first missing chunk")
    func lookupReturnsLongestPrefixUntilMissing() async {
        let cache = PromptCache()
        let modelID = "chunk-chain-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< (chunkSize * 3))
        let chunks = makeChunkChain(tokens: tokens, chunkSize: chunkSize)
        // Captured (as plain, Sendable `Int`s) BEFORE `chunks` is sent into the
        // actor below -- reusing `chunks` itself afterward would risk a
        // send/reuse data race under strict concurrency checking, since
        // `StoredChunk` carries non-`Sendable` `MLXArray` tensors.
        let expectedKeys = chunks.prefix(2).map(\.chunkKey)
        // Only the first two of three chunks are stored -- the third is "missing".
        await cache.insert(modelID: modelID, chunks: Array(chunks.prefix(2)))

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens + [999], chunkSize: chunkSize
        ).consume()

        #expect(matched.count == 2)
        #expect(matched.map(\.chunkKey) == expectedKeys)
    }

    @Test("diverging tokens stop the walk at the last shared chunk boundary")
    func lookupStopsAtDivergence() async {
        let cache = PromptCache()
        let modelID = "chunk-divergence-\(UUID().uuidString)"
        let chunkSize = 4
        let sharedTokens = Array(0 ..< (chunkSize * 2))
        let storedTokens = sharedTokens + Array(1_000 ..< 1_004)
        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: storedTokens, chunkSize: chunkSize))

        let divergentQuery = sharedTokens + Array(2_000 ..< 2_004) + [1]

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: divergentQuery, chunkSize: chunkSize
        ).consume()

        #expect(matched.count == 2, "only the two shared leading chunks should match")
    }

    @Test("lookupLongestPrefix touches (bumps) lastUsed for every matched chunk")
    func lookupTouchesLastUsedForMatchedChunks() async {
        let cache = PromptCache()
        let modelID = "chunk-touch-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< (chunkSize * 2))
        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        let first = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
        ).consume()
        let second = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
        ).consume()

        #expect(first.count == 2)
        #expect(second.count == 2)
        for index in 0 ..< 2 {
            #expect(
                second[index].lastUsed > first[index].lastUsed,
                "each matched chunk's lastUsed must be bumped again on the second lookup")
        }
    }

    // MARK: - Cap at count - 1

    @Test(
        "lookup for newTokens exactly equal to a fully-stored sequence returns at most floor((count-1)/chunkSize) chunks"
    )
    func lookupCapsAtCountMinusOne() async {
        let cache = PromptCache()
        let modelID = "chunk-cap-\(UUID().uuidString)"
        let chunkSize = 4
        let chunkCount = 5
        let tokens = Array(0 ..< (chunkSize * chunkCount))
        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens, chunkSize: chunkSize
        ).consume()

        let expectedMax = (tokens.count - 1) / chunkSize
        #expect(matched.count == expectedMax)
        #expect(matched.count < chunkCount, "lookup must never cover the whole stored prompt")
    }

    @Test("lookup never returns zero-token feed room even when the prompt is a single chunk long")
    func lookupCapLeavesAtLeastOneTokenForSingleChunkPrompt() async {
        let cache = PromptCache()
        let modelID = "chunk-cap-single-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        await cache.insert(
            modelID: modelID, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens, chunkSize: chunkSize
        ).consume()

        #expect(
            matched.isEmpty,
            "a single chunk's worth of tokens leaves no room for a full chunk match under the cap")
    }

    // MARK: - Collision safety

    @Test(
        "a key match whose stored tokens differ from the chunk-aligned window is treated as a miss, never served as a hit"
    )
    func forcedCollisionTreatedAsMiss() async {
        let cache = PromptCache()
        let modelID = "chunk-collision-\(UUID().uuidString)"
        let chunkSize = 4
        let queryTokens = Array(0 ..< chunkSize)
        let collidingKey = PromptCache.chunkKey(
            parentKey: PromptCache.rootChunkKey, tokens: queryTokens)

        // Forge a chunk stored under the EXACT key the real query window would
        // compute, but with different tokens -- simulating a genuine 64-bit
        // hash collision without needing to find a real one.
        let forgedChunk = makeStoredChunk(
            tokens: [9_999, 9_998, 9_997, 9_996], parentKey: PromptCache.rootChunkKey,
            forcedKey: collidingKey)
        await cache.insert(modelID: modelID, chunks: [forgedChunk])

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: queryTokens + [1], chunkSize: chunkSize
        ).consume()

        #expect(matched.isEmpty, "a colliding key with mismatched tokens must be treated as a miss")
    }

    @Test("a forced collision partway down the chain stops the walk at the last genuine match")
    func forcedCollisionMidChainStopsWalk() async {
        let cache = PromptCache()
        let modelID = "chunk-collision-midchain-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< (chunkSize * 3))
        var chunks = makeChunkChain(tokens: tokens, chunkSize: chunkSize)
        // Corrupt the SECOND chunk's tokens in place, leaving its chunkKey (and
        // the third chunk's parentKey, which chains from it) unchanged -- a
        // forged collision one hop into the chain.
        chunks[1] = PromptCache.StoredChunk(
            tokens: [7_777, 7_776, 7_775, 7_774], layers: chunks[1].layers,
            parentKey: chunks[1].parentKey, chunkKey: chunks[1].chunkKey,
            byteSize: chunks[1].byteSize, lastUsed: chunks[1].lastUsed)
        await cache.insert(modelID: modelID, chunks: chunks)

        let matched = await cache.lookupLongestPrefix(
            modelID: modelID, newTokens: tokens + [1], chunkSize: chunkSize
        ).consume()

        #expect(matched.count == 1, "the walk must stop right after the last genuinely-matching chunk")
    }

    // MARK: - evictAll / remove scoping, multi-model isolation

    @Test("evictAll drops chunk-store state for every model")
    func evictAllDropsChunkStoreForEveryModel() async {
        let cache = PromptCache()
        let modelA = "chunk-evict-all-a-\(UUID().uuidString)"
        let modelB = "chunk-evict-all-b-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        await cache.insert(
            modelID: modelA, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))
        await cache.insert(
            modelID: modelB, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        await cache.evictAll()

        #expect(await cache.chunkCount(modelID: modelA) == 0)
        #expect(await cache.chunkCount(modelID: modelB) == 0)
    }

    @Test("remove drops only the named model's chunk store, leaving other models' chunks intact")
    func removeIsPerModelForChunkStore() async {
        let cache = PromptCache()
        let modelA = "chunk-remove-a-\(UUID().uuidString)"
        let modelB = "chunk-remove-b-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        await cache.insert(
            modelID: modelA, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))
        await cache.insert(
            modelID: modelB, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        await cache.remove(modelID: modelA)

        #expect(await cache.chunkCount(modelID: modelA) == 0)
        #expect(await cache.chunkCount(modelID: modelB) == 1)
    }

    @Test(
        "chunks are isolated per model: inserting under one model id is invisible to a different model id"
    )
    func chunkStoreIsolationPerModel() async {
        let cache = PromptCache()
        let modelA = "chunk-isolation-a-\(UUID().uuidString)"
        let modelB = "chunk-isolation-b-\(UUID().uuidString)"
        let chunkSize = 4
        let tokens = Array(0 ..< chunkSize)
        await cache.insert(
            modelID: modelA, chunks: makeChunkChain(tokens: tokens, chunkSize: chunkSize))

        #expect(await cache.chunkCount(modelID: modelA) == 1)
        #expect(await cache.chunkCount(modelID: modelB) == 0, "model B must not see model A's chunks")

        let matchedB = await cache.lookupLongestPrefix(
            modelID: modelB, newTokens: tokens + [1], chunkSize: chunkSize
        ).consume()
        #expect(matchedB.isEmpty, "model B's lookup must never return model A's chunks")
    }
}

#endif
