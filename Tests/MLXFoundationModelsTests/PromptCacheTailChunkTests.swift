// Copyright © 2026 Apple Inc.
//
// Tests for the TAIL chunk feature (kanban 5ra1wzm): storing the
// variable-length trailing partial span left uncovered by
// `PromptCache.sliceChunks`/`sliceTailChunk` (see `PromptCacheChunkTests.swift`'s
// own "sliceTailChunk" section for the pure-function tests), and wiring it
// into the actor's `store()`/`resolve()`/`assemble()` surface so a
// conversation prefix SHORTER than one full `chunkSize`-token chunk -- and a
// growing conversation's trailing sub-chunk remainder between full-chunk
// boundaries -- is reusable instead of silently dropped (the bug real-model
// integration testing exposed: `PromptCacheReuseTests`/`PromptCachePrewarmTests`
// both reported zero reuse for exactly this shape).
//
// Reuses `PromptCacheTestSupport.swift`'s `makeChunkableCache`/`resolveOnce`/
// `rawBufferAddress`/`mutateFirstElementInPlace` fixtures rather than
// duplicating them (per that file's own doc comment).
//
// Real tensor CONTENT is populated (`makeChunkableCache`) because `store()`
// routes through `sliceChunks`/`sliceTailChunk`, which require a genuine
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

@Suite("PromptCache tail chunk (partial-span, sub-chunkSize reuse)")
struct PromptCacheTailChunkTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - 1. Sub-chunkSize conversation becomes reusable

    @Test(
        "storing a sub-chunkSize-length conversation yields a reusable prefix -- a later resolve feeds only the new suffix"
    )
    func subChunkLengthConversationYieldsReusablePrefix() async {
        let cache = PromptCache()
        let modelID = "tail-subchunk-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 64
        await cache.setChunkSize(chunkSize)
        let tokens = Array(0 ..< 40)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        #expect(
            await cache.chunkCount(modelID: modelID) > 0,
            "a sub-chunkSize conversation must still store SOMETHING reusable (the tail), not nothing"
        )

        let continuation = tokens + Array(90_000 ..< 90_005)
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: continuation, model: model)

        #expect(
            resolved.tokensToFeed.count < continuation.count,
            "a continuation of a sub-chunkSize conversation must reuse the stored tail, not re-feed everything"
        )
        #expect(resolved.tokensToFeed == Array(continuation.suffix(5)))
    }

    @Test(
        "resolving the identical previously-stored sub-chunkSize conversation still leaves >=1 token to feed (cap-at-count-1 respected)"
    )
    func identicalSubChunkConversationRespectsCapAtCountMinusOne() async {
        let cache = PromptCache()
        let modelID = "tail-cap-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 64
        await cache.setChunkSize(chunkSize)
        let tokens = Array(0 ..< 40)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: tokens, model: model)

        #expect(
            resolved.tokensToFeed.count >= 1,
            "generation always needs at least one fresh token to produce new logits")
        #expect(
            resolved.tokensToFeed.count < tokens.count,
            "an identical, fully-stored sub-chunkSize prompt must never be fed back in its entirety")
    }

    // MARK: - 2. Divergence inside the tail

    @Test(
        "two conversations sharing a common tail prefix then diverging each resolve to their OWN correctly-matched continuation"
    )
    func divergingContinuationsFromASharedTailPrefixEachMatchTheirOwnCorrectLength() async {
        let cache = PromptCache()
        let modelID = "tail-divergence-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 8
        await cache.setChunkSize(chunkSize)

        // One full chunk (8 tokens) plus a 5-token tail -- the shared parent transcript.
        let sharedTokens = Array(0 ..< (chunkSize + 5))
        await cache.store(
            modelID: modelID, tokens: sharedTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: sharedTokens.count)]))

        // Diverges after sharing only the FIRST 3 of the tail's 5 tokens.
        let divergentContinuation =
            Array(sharedTokens.prefix(chunkSize + 3)) + Array(500_000 ..< 500_010)

        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: divergentContinuation, model: model)

        let expectedMatchedTokenCount = chunkSize + 3
        #expect(resolved.cache[0].offset == expectedMatchedTokenCount)
        #expect(
            resolved.tokensToFeed
                == Array(
                    divergentContinuation.suffix(
                        divergentContinuation.count - expectedMatchedTokenCount)))
    }

    /// Byte-exact proof (mirroring `PromptCacheAssembleTests`'s own style):
    /// `assemble(chunks:layerCount:lastChunkMatchedLength:)` must slice the
    /// tail down to the ACTUAL matched length before concatenating, and the
    /// result must be a genuinely independent buffer, not an alias of the
    /// stored tail's own tensor -- verified past the Swift wrapper via
    /// `rawBufferAddress(of:)`, not just shape/count.
    @Test(
        "assembling a partially-matched tail slices to the matched length, byte-exact against the source cache's own first K tokens"
    )
    func partialTailMatchAssemblesByteExactPrefixOfSourceCache() {
        let chunkSize = 8
        let tailTokenCount = 5
        let tokenCount = chunkSize + tailTokenCount
        let headDim = 4
        let cache = makeChunkableCache(tokenCount: tokenCount, headDim: headDim)
        let tokens = Array(0 ..< tokenCount)

        let fullChunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        let parentKey = fullChunks.last!.chunkKey
        let tail = PromptCache.sliceTailChunk(
            tokens: tokens, cache: [cache], chunkSize: chunkSize, parentKey: parentKey)!

        // Simulate a DIVERGING continuation that shares only the first 3 of
        // the tail's 5 stored tokens before diverging -- the matched length
        // a real `lookupTailMatch` longest-common-prefix walk would compute.
        let matchedTailLength = 3

        let assembled = PromptCache.assemble(
            chunks: fullChunks + [tail], layerCount: 1, lastChunkMatchedLength: matchedTailLength)

        let matchedTotal = chunkSize + matchedTailLength
        let expectedKeys = cache.state[0][.ellipsis, ..<matchedTotal, 0...]
        let expectedValues = cache.state[1][.ellipsis, ..<matchedTotal, 0...]

        #expect(assembled[0].offset == matchedTotal)
        #expect((assembled[0].state[0] .== expectedKeys).all().item())
        #expect((assembled[0].state[1] .== expectedValues).all().item())
        #expect(
            rawBufferAddress(of: assembled[0].state[0]) != rawBufferAddress(of: tail.layers[0].keys),
            "assemble() must allocate a fresh buffer, not merely slice-and-wrap the stored tail's own buffer"
        )
        #expect(
            rawBufferAddress(of: assembled[0].state[1])
                != rawBufferAddress(of: tail.layers[0].values),
            "assemble() must allocate a fresh buffer, not merely slice-and-wrap the stored tail's own buffer"
        )
    }

    // MARK: - 3. Dedup

    @Test("storing the identical tail twice does not duplicate byte accounting")
    func identicalTailStoredTwiceDoesNotDuplicateBytes() async {
        let cache = PromptCache()
        let modelID = "tail-dedup-\(UUID().uuidString)"
        let chunkSize = 64
        await cache.setChunkSize(chunkSize)
        let tokens = Array(0 ..< 40)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))
        let bytesAfterFirst = await cache.totalStoredByteCount()
        let countAfterFirst = await cache.chunkCount(modelID: modelID)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))
        let bytesAfterSecond = await cache.totalStoredByteCount()
        let countAfterSecond = await cache.chunkCount(modelID: modelID)

        #expect(
            bytesAfterFirst > 0,
            "storing a sub-chunkSize conversation (0 full chunks) must still account nonzero tail bytes"
        )
        #expect(
            bytesAfterSecond == bytesAfterFirst,
            "storing the identical tail again must not double-count bytes")
        #expect(
            countAfterSecond == countAfterFirst,
            "storing the identical tail again must not create a second stored entry")
    }

    // MARK: - 4. Eviction

    @Test("evicting a full chunk that a tail is chained under also reclaims the chained tail's bytes")
    func evictingAncestorFullChunkReclaimsChainedTailBytes() async {
        let cache = PromptCache()
        let modelID = "tail-eviction-\(UUID().uuidString)"
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)
        // 6 tokens: one full 4-token chunk plus a 2-token tail chained from it.
        let tokens = Array(0 ..< 6)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        let totalBeforeEviction = await cache.totalStoredByteCount()
        #expect(totalBeforeEviction > 0)
        #expect(
            await cache.chunkCount(modelID: modelID) == 2,
            "one full chunk plus one chained tail")

        // Force the full chunk (and therefore its chained tail, the only
        // lineage in this fresh actor) out via a byte budget too tight for
        // anything to survive.
        await cache.setByteBudget(1)

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "evicting the full chunk must transitively reclaim its chained tail too")
        #expect(await cache.totalStoredByteCount() == 0)
    }

    // MARK: - 5. Byte accounting

    @Test("totalStoredByteCount() includes tail bytes for a conversation with ZERO full chunks")
    func totalStoredByteCountIncludesTailBytesWithNoFullChunks() async {
        let cache = PromptCache()
        let modelID = "tail-byte-accounting-\(UUID().uuidString)"
        let chunkSize = 64
        await cache.setChunkSize(chunkSize)
        let tokens = Array(0 ..< 40)

        #expect(await cache.totalStoredByteCount() == 0)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))

        #expect(
            await cache.chunkCount(modelID: modelID) == 1,
            "a 40-token conversation at chunkSize 64 stores exactly one (tail) chunk")
        #expect(
            await cache.totalStoredByteCount() > 0,
            "the tail's bytes must be reflected in the global byte total")
    }
}

#endif
