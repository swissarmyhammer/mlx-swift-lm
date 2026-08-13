// Copyright © 2026 Apple Inc.
//
// Coverage for kanban `vgznm16`: a downstream consumer reported a suspected
// cross-session leak in the hybrid Mamba/attention checkpoint store
// (`PromptCache.hybridCheckpoints`) -- a full-multi-suite-only test failure,
// faster-than-normal yet with wrong output, on repeated fresh loads of the
// SAME model id (`mlx-community/Qwen3.6-27B-mxfp4`) where every suite ends
// with `container.model.evict()`. The suspected mechanisms were (a) eviction
// not actually clearing `hybridCheckpoints`, or (b) checkpoint resolution
// matching across two unrelated conversations.
//
// This file pins both mechanisms shut, at the same actor-level call surface
// `MLXLanguageModel` uses:
//
//   1. `remove(modelID:)` -- the exact call `MLXLanguageModel.evict()`
//      delegates to (`Self.promptCache.remove(modelID: modelID)`; the
//      `promptCache` instance itself is a `private static let`, so the
//      delegation is a source-verified one-liner with no seam to intercept,
//      and the decisive behavior lives here on the actor) -- drops EVERY
//      hybrid checkpoint stored for that model id, reclaiming its bytes,
//      while leaving other models' checkpoints untouched.
//   2. `evictAll()` drops every model's hybrid checkpoints.
//   3. Cross-conversation isolation through the real `resolve()` surface
//      with a synthetic hybrid-architecture model: after conversation A
//      checkpoints and the model id is evicted, an unrelated conversation B
//      on a fresh same-model-id "load" resolves exactly as it would with a
//      cold cache (full token feed, no checkpoint match).
//   4. COLLISION SAFETY: `resolveHybridCheckpoint` verifies candidates by
//      element-wise `Array(newTokens.prefix(len)) == checkpoint.tokens`,
//      never by trusting the `Hasher`-based `ChunkKey` alone. A genuine
//      key collision cannot be constructed deterministically (Swift's
//      `Hasher` is per-process seeded), so the guard is pinned with the
//      adversarial shape a collision would present: a stored checkpoint
//      whose tokens are the same length/structure as the new conversation's
//      prefix but differ element-wise -- it must never match.
//
// `store()`-backed tests populate a real hybrid stack (`makeChunkableCache`
// + `MambaCache`), which routes through `snapshotHybridCheckpoint`'s
// `ownedCopy` tensor evals -- a real GPU-device eval under plain
// `swift test`; see `TestBootstrap.swift`'s `MetalLibraryTestBootstrap`
// (kanban 23ff1zx, memory note `swiftpm-test-gpu-metallib-limit`) for why
// the `init()` bootstrap call below is required.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Minimal hybrid-architecture `LanguageModel` stand-in whose
/// `newCache(parameters:)` hands `PromptCache.resolve` a genuine
/// `[KVCacheSimple, MambaCache]` hybrid stack, so `resolve()` routes through
/// the hybrid-checkpoint mechanism (`isHybridMambaAttention` is `true`)
/// instead of the ordinary chunk store. `prepare`/`callAsFunction` are never
/// invoked by `PromptCache` and simply trap if they somehow were --
/// mirroring `PromptCacheProbeModel` (`PromptCacheTestSupport.swift`), which
/// this deliberately does not extend: that probe's `KVCacheDimensionProvider`
/// conformance builds an all-`KVCacheSimple` stack, the exact opposite of
/// what these tests need.
private final class HybridPromptCacheProbeModel: Module, MLXLMCommon.LanguageModel,
    @unchecked Sendable
{
    /// A fresh two-layer hybrid stack per call -- the shape that routes
    /// `resolve()` to `resolveHybridCheckpoint` rather than chunk lookup.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple(), MambaCache()]
    }

    /// Protocol conformance stub; not invoked by PromptCache and will trap if called.
    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    )
        throws -> PrepareResult
    {
        fatalError("not exercised by PromptCache tests")
    }

    /// Protocol conformance stub; not invoked by PromptCache and will trap if called.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised by PromptCache tests")
    }
}

@Suite("PromptCache hybrid checkpoint eviction scope and cross-conversation isolation")
struct PromptCacheHybridEvictionScopeTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    @Test("remove(modelID:) clears every hybrid checkpoint for that model id and reclaims its bytes")
    func removeClearsHybridCheckpointStore() async {
        let cache = PromptCache()
        let modelID = "hybrid-evict-remove-\(UUID().uuidString)"
        let tokens = Array(0 ..< 16)

        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: Array(tokens.prefix(8))))
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: Array(tokens.prefix(12))))
        #expect(await cache.hybridCheckpointCount(modelID: modelID) == 2)
        #expect(await cache.totalStoredByteCount() > 0)

        await cache.remove(modelID: modelID)

        #expect(
            await cache.hybridCheckpointCount(modelID: modelID) == 0,
            "remove(modelID:) must drop the model's ENTIRE hybrid checkpoint store")
        #expect(
            await cache.totalStoredByteCount() == 0,
            "remove(modelID:) must reclaim the dropped checkpoints' bytes")

        let resolved = await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: tokens)
            .consume()
        #expect(
            resolved == nil,
            "no stale checkpoint may remain resolvable after remove(modelID:)")
    }

    @Test("remove(modelID:) drops only the named model's hybrid checkpoints, leaving other models' resolvable")
    func removeIsPerModelForHybridCheckpoints() async {
        let cache = PromptCache()
        let modelA = "hybrid-evict-per-model-a-\(UUID().uuidString)"
        let modelB = "hybrid-evict-per-model-b-\(UUID().uuidString)"
        let tokensA = Array(0 ..< 10)
        let tokensB = Array(1_000 ..< 1_010)

        await cache.insertHybridCheckpoint(
            modelID: modelA, checkpoint: makeHybridCheckpoint(tokens: tokensA))
        await cache.insertHybridCheckpoint(
            modelID: modelB, checkpoint: makeHybridCheckpoint(tokens: tokensB))

        await cache.remove(modelID: modelA)

        #expect(await cache.hybridCheckpointCount(modelID: modelA) == 0)
        #expect(
            await cache.hybridCheckpointCount(modelID: modelB) == 1,
            "remove must NOT disturb other models' hybrid checkpoints")

        let resolvedB = await cache.resolveHybridCheckpoint(
            modelID: modelB, newTokens: tokensB + [42]
        ).consume()
        #expect(
            resolvedB?.tokens == tokensB,
            "model B's checkpoint must remain resolvable after model A's removal")
    }

    @Test("evictAll() drops every model's hybrid checkpoints and zeroes the byte accounting")
    func evictAllClearsEveryModelsHybridCheckpoints() async {
        let cache = PromptCache()
        let modelA = "hybrid-evict-all-a-\(UUID().uuidString)"
        let modelB = "hybrid-evict-all-b-\(UUID().uuidString)"

        await cache.insertHybridCheckpoint(
            modelID: modelA, checkpoint: makeHybridCheckpoint(tokens: Array(0 ..< 8)))
        await cache.insertHybridCheckpoint(
            modelID: modelB, checkpoint: makeHybridCheckpoint(tokens: Array(100 ..< 108)))
        #expect(await cache.totalStoredByteCount() > 0)

        await cache.evictAll()

        #expect(await cache.hybridCheckpointCount(modelID: modelA) == 0)
        #expect(await cache.hybridCheckpointCount(modelID: modelB) == 0)
        #expect(
            await cache.totalStoredByteCount() == 0,
            "evictAll must reclaim every dropped checkpoint's bytes")
    }

    @Test(
        """
        after conversation A checkpoints and the model id is evicted, an unrelated \
        conversation B on a fresh same-model-id load resolves exactly as a cold cache
        """
    )
    func evictedModelIDIsColdForTheNextConversation() async throws {
        let cache = PromptCache()
        let model = HybridPromptCacheProbeModel()
        let modelID = "hybrid-evict-cross-conversation-\(UUID().uuidString)"
        let tokensA = Array(0 ..< 12)

        // Conversation A completes a round: a real hybrid stack is stored,
        // producing a whole-stack checkpoint keyed by A's full token prefix.
        await cache.store(
            modelID: modelID, tokens: tokensA,
            cache: SendableBox([makeChunkableCache(tokenCount: tokensA.count), MambaCache()]))
        #expect(await cache.hybridCheckpointCount(modelID: modelID) == 1)

        // CONTROL: before eviction, a same-conversation continuation DOES
        // reuse the checkpoint (feeds only the suffix) -- proving the
        // cold-cache assertions below observe a signal this setup can
        // actually produce, not a vacuous always-full-feed.
        let continuationA = tokensA + [90, 91, 92]
        let control = try await resolveOnce(
            cache: cache, modelID: modelID, newTokens: continuationA, model: model)
        #expect(
            control.tokensToFeed == [90, 91, 92],
            "sanity: pre-eviction, the checkpoint must be reusable or this test proves nothing")

        // The model id is evicted -- the exact PromptCache call
        // `MLXLanguageModel.evict()` delegates to.
        await cache.remove(modelID: modelID)

        // A fresh same-model-id load drives a SECOND, unrelated conversation.
        let tokensB = Array(2_000 ..< 2_015)
        #expect(
            await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: tokensB).consume()
                == nil,
            "no checkpoint may match conversation B after eviction")
        let resolvedB = try await resolveOnce(
            cache: cache, modelID: modelID, newTokens: tokensB, model: model)
        #expect(
            resolvedB.tokensToFeed == tokensB,
            "conversation B must be fed in full -- identical to a cold cache")

        // Even replaying conversation A's own continuation is cold now: the
        // eviction removed the checkpoint outright, not just B's view of it.
        let replayedA = try await resolveOnce(
            cache: cache, modelID: modelID, newTokens: continuationA, model: model)
        #expect(
            replayedA.tokensToFeed == continuationA,
            "the control-path reuse must be gone after eviction")
    }

    @Test(
        """
        a stored checkpoint never matches a divergent conversation sharing its length \
        and structure -- the element-wise token-prefix guard, not the hash key, decides
        """
    )
    func divergentConversationsNeverCrossMatch() async throws {
        let cache = PromptCache()
        let model = HybridPromptCacheProbeModel()
        let modelID = "hybrid-divergent-no-cross-match-\(UUID().uuidString)"
        let tokensA = [7, 8, 9, 10, 11, 12]

        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: tokensA))

        // Conversation B's prefix has the same length and near-identical
        // content as A's stored checkpoint -- the adversarial shape a
        // hash-key collision would present. Only element-wise comparison of
        // the actual token content can reject it.
        let nearIdenticalB = [7, 8, 999, 10, 11, 12, 13]
        #expect(
            await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: nearIdenticalB)
                .consume() == nil,
            "a same-length, one-element-different prefix must never match")

        // And through the real resolve() surface: conversation B is fed in full.
        let resolvedB = try await resolveOnce(
            cache: cache, modelID: modelID, newTokens: nearIdenticalB, model: model)
        #expect(resolvedB.tokensToFeed == nearIdenticalB)

        // A fully disjoint conversation likewise never matches.
        let disjointB = Array(500 ..< 510)
        #expect(
            await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: disjointB)
                .consume() == nil)

        // The stored checkpoint itself is untouched by the failed lookups:
        // conversation A's genuine continuation still matches.
        let resolvedA = await cache.resolveHybridCheckpoint(
            modelID: modelID, newTokens: tokensA + [13]
        ).consume()
        #expect(resolvedA?.tokens == tokensA)
    }
}

#endif
