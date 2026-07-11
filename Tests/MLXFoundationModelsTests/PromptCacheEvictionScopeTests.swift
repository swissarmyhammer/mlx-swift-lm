// Copyright © 2026 Apple Inc.
//
// Eviction-scope suite rewrite for kanban `t71kdmj`: full behavioral
// replacement for the deleted `PromptCacheSlotPoolTests.swift`'s
// `PromptCacheEvictionScopeTests` suite (old slot-pool semantics, removed at
// the `cthbfmw` cutover), re-established over chunk storage.
//
// Driven through the actor's public `resolve()`/`store()`/`evictAll()`/
// `remove()` surface -- the same level the deleted suite exercised (through
// `resolveOnce`/`storeWellFormedSlot`) -- rather than the lower `insert`/
// `lookupLongestPrefix` chunk-store primitives `PromptCacheChunkStoreTests`'s
// own "evictAll / remove scoping, multi-model isolation" section already
// covers directly. Testing at this higher level is deliberate, not
// redundant: it proves the guarantee holds through the exact call surface
// `Executor` actually uses, not just the underlying dictionary bookkeeping.
//
// Chunk semantics have no per-conversation "slot" to compare an
// `ObjectIdentifier` against (every `resolve()` assembles a fresh instance
// regardless of whether anything matched -- see `PromptCache.assemble`'s
// doc comment), so reuse vs. rebuild is observed the same way
// `PromptCacheChunkCutoverTests` does: a rebuild feeds the FULL prompt (no
// chunk matched, nothing left over to omit); a reuse feeds strictly fewer
// tokens than the full prompt (some prefix was served from stored chunks).
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

@Suite("PromptCache eviction API and per-model isolation (chunk storage, resolve()-level)")
struct PromptCacheEvictionScopeTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    private let chunkSize = 4

    @Test(
        "evictAll drops every model's chunk store; a post-eviction resolve for any model rebuilds from scratch"
    )
    func evictAllDropsAllModelsThroughResolve() async {
        let cache = PromptCache()
        await cache.setChunkSize(chunkSize)
        let model = PromptCacheProbeModel()
        let modelA = "evict-scope-all-a-\(UUID().uuidString)"
        let modelB = "evict-scope-all-b-\(UUID().uuidString)"
        let tokensA = Array(0 ..< (chunkSize * 3))
        let tokensB = Array(1_000 ..< (1_000 + chunkSize * 3))

        await cache.store(
            modelID: modelA, tokens: tokensA,
            cache: SendableBox([makeChunkableCache(tokenCount: tokensA.count)]))
        await cache.store(
            modelID: modelB, tokens: tokensB,
            cache: SendableBox([makeChunkableCache(tokenCount: tokensB.count)]))

        await cache.evictAll()

        let extendedA = tokensA + Array(90_000 ..< 90_005)
        let extendedB = tokensB + Array(91_000 ..< 91_005)
        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: extendedA, model: model)
        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: extendedB, model: model)

        #expect(
            resolvedA.tokensToFeed == extendedA,
            "evictAll must drop model A's chunks, forcing a full rebuild")
        #expect(
            resolvedB.tokensToFeed == extendedB,
            "evictAll must drop model B's chunks, forcing a full rebuild")
        #expect(await cache.chunkCount(modelID: modelA) == 0)
        #expect(await cache.chunkCount(modelID: modelB) == 0)
    }

    @Test("remove drops only the named model's chunk store, leaving other models' chunks reusable")
    func removeIsPerModelThroughResolve() async {
        let cache = PromptCache()
        await cache.setChunkSize(chunkSize)
        let model = PromptCacheProbeModel()
        let modelA = "evict-scope-remove-a-\(UUID().uuidString)"
        let modelB = "evict-scope-remove-b-\(UUID().uuidString)"
        let tokensA = Array(0 ..< (chunkSize * 3))
        let tokensB = Array(1_000 ..< (1_000 + chunkSize * 3))

        await cache.store(
            modelID: modelA, tokens: tokensA,
            cache: SendableBox([makeChunkableCache(tokenCount: tokensA.count)]))
        await cache.store(
            modelID: modelB, tokens: tokensB,
            cache: SendableBox([makeChunkableCache(tokenCount: tokensB.count)]))

        await cache.remove(modelID: modelA)

        let extendedA = tokensA + Array(90_000 ..< 90_005)
        let extendedB = tokensB + Array(91_000 ..< 91_005)
        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: extendedA, model: model)
        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: extendedB, model: model)

        #expect(
            resolvedA.tokensToFeed == extendedA,
            "remove must drop model A's chunks, forcing a full rebuild")
        #expect(
            resolvedB.tokensToFeed.count < extendedB.count,
            "remove must NOT disturb model B's chunks -- model B must still reuse")
    }

    @Test(
        """
        identical tokens never cross model ids: a different model id never reuses \
        another model's chunks, and the original model's chunks are left untouched
        """
    )
    func identicalTokensNeverCrossModelIDsThroughResolve() async {
        let cache = PromptCache()
        await cache.setChunkSize(chunkSize)
        let model = PromptCacheProbeModel()
        let modelA = "evict-scope-isolation-a-\(UUID().uuidString)"
        let modelB = "evict-scope-isolation-b-\(UUID().uuidString)"
        let tokens = Array(0 ..< (chunkSize * 3))

        await cache.store(
            modelID: modelA, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        // Model B asks with the EXACT same tokens: KV state is conditioned on
        // the model, not just the token sequence, so serving model A's
        // chunks to model B would silently corrupt B's attention.
        let extended = tokens + Array(92_000 ..< 92_005)
        let resolvedB = await resolveOnce(
            cache: cache, modelID: modelB, newTokens: extended, model: model)
        #expect(resolvedB.tokensToFeed == extended, "model B must never reuse model A's chunks")

        // And model A's chunks are still there for model A.
        let resolvedA = await resolveOnce(
            cache: cache, modelID: modelA, newTokens: extended, model: model)
        #expect(
            resolvedA.tokensToFeed.count < extended.count,
            "model A's chunks must be untouched by model B's resolve")
    }
}

#endif
