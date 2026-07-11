// Copyright © 2026 Apple Inc.
//
// Concurrency-suite rewrite for kanban `t71kdmj`: full behavioral
// replacement for the deleted `PromptCacheConcurrencyTests.swift` (old
// slot-pool semantics, removed at the `cthbfmw` cutover), re-established
// over chunk semantics.
//
// The OLD suite's entire premise no longer applies: it existed because
// `resolve()` used to REMOVE a checked-out slot before returning it, so a
// concurrent second caller for the same prefix could either steal an
// in-flight slot or find nothing -- "double-checkout" (two callers handed
// the identical `KVCache` instance) was a real hazard its `CheckoutTracker`
// guarded against. Since the chunk-store cutover, chunks are read-only and
// `resolve()` never removes anything from the store -- every call
// (concurrent or not) gets its own freshly `assemble`d, PRIVATE `[KVCache]`
// (see `PromptCache.assemble`'s doc comment: "no checkout, no steal, and no
// double-checkout hazard by construction"). So double-checkout is now
// IMPOSSIBLE by construction, not merely guarded against -- these tests
// instead prove the new, positive guarantee: concurrent resolves for the
// same prefix always return DISTINCT assembled instances (never the SAME
// instance handed to two callers), with equal content, and that this holds
// under sustained concurrent resolve/store churn against a single model id
// without corrupting the chunk store's bookkeeping.
//
// `PromptCacheAssembleTests.concurrentAssembleCallsReturnDistinctInstancesWithEqualContent`
// already proves the same "distinct instances, equal content" guarantee for
// `PromptCache.assemble` called directly, in isolation. These tests exercise
// one level up: genuine concurrent `resolve()` round trips through the
// actor's public entry point (matching prefix lookup, matched-chunk
// assembly, and -- in the churn test -- `store()`), which is what a real
// caller (`Executor`) actually races.
//
// Every result's cache instance is kept alive (via `SendableBox`, never
// consumed) until every concurrent task has finished, mirroring the deleted
// suite's own precaution: consuming (and thus dropping the last reference
// to) a cache instance while other tasks are still allocating risks the
// allocator reusing its exact address for a later instance, an
// `ObjectIdentifier` collision that would look like a false double-return.
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

@Suite("PromptCache actor-concurrency: distinct assembled instances, no store corruption")
struct PromptCacheConcurrencyTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    @Test(
        """
        many concurrent resolve() calls against the same stored prefix each receive a \
        DISTINCT assembled cache instance with equal content -- chunks are read-only \
        and every resolve gets its own assembled copy, so this can never race
        """
    )
    func concurrentResolvesForSamePrefixReturnDistinctInstancesWithEqualContent() async {
        let cache = PromptCache()
        let modelID = "concurrency-distinct-\(UUID().uuidString)"
        let model: any LanguageModel = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)

        let storedTokens = Array(0 ..< (chunkSize * 3))
        await cache.store(
            modelID: modelID, tokens: storedTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: storedTokens.count)]))

        let callerCount = 12
        let newTokens = storedTokens + Array(90_000 ..< 90_005)
        // `SendableBox` is single-use (`consume()` traps on a second call),
        // so each concurrent caller needs its OWN box wrapping the same
        // underlying model -- built here, outside any `@Sendable` closure,
        // since `model` itself (an `any LanguageModel` existential) isn't
        // Sendable and can't be captured directly by `group.addTask`.
        let modelBoxes = (0 ..< callerCount).map { _ in SendableBox(model) }

        let resultBoxes = await withTaskGroup(
            of: SendableBox<(cache: [KVCache], tokensToFeed: [Int])>.self
        ) { group in
            for i in 0 ..< callerCount {
                let modelBox = modelBoxes[i]
                group.addTask {
                    await cache.resolve(
                        modelID: modelID, newTokens: newTokens, model: modelBox, parameters: nil)
                }
            }
            var results: [SendableBox<(cache: [KVCache], tokensToFeed: [Int])>] = []
            for await box in group { results.append(box) }
            return results
        }

        #expect(resultBoxes.count == callerCount)
        let resolved = resultBoxes.map { $0.consume() }

        let identities = resolved.map { ObjectIdentifier($0.cache[0] as AnyObject) }
        #expect(
            Set(identities).count == callerCount,
            "expected \(callerCount) distinct assembled instances, got \(Set(identities).count) -- a concurrent resolve() handed the same instance to more than one caller"
        )

        let first = resolved[0].cache[0] as! KVCacheSimple
        for result in resolved.dropFirst() {
            let simple = result.cache[0] as! KVCacheSimple
            #expect(result.tokensToFeed == resolved[0].tokensToFeed)
            #expect(simple.offset == first.offset)
            #expect((simple.state[0] .== first.state[0]).all().item())
            #expect((simple.state[1] .== first.state[1]).all().item())
        }
    }

    @Test(
        """
        sustained parallel resolve/store churn against a shared model id never \
        corrupts the chunk store's bookkeeping: every concurrently-returned assembled \
        cache is a distinct instance, and the shared prefix's chunks dedup down to \
        exactly what was seeded -- never fewer (lost) or more (a dedup failure under \
        concurrent insert)
        """
    )
    func sustainedConcurrentChurnNeverCorruptsStoreBookkeeping() async {
        let cache = PromptCache()
        let modelID = "concurrency-churn-\(UUID().uuidString)"
        let model: any LanguageModel = PromptCacheProbeModel()
        let chunkSize = 4
        let headDim = 4
        await cache.setChunkSize(chunkSize)

        let seedTokens = Array(0 ..< (chunkSize * 2))
        await cache.store(
            modelID: modelID, tokens: seedTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: seedTokens.count, headDim: headDim)])
        )
        let seedChunkCount = await cache.chunkCount(modelID: modelID)
        #expect(seedChunkCount > 0, "the seed round must actually contribute chunks to churn against")

        let workerCount = 8
        let iterationsPerWorker = 6
        let totalCalls = workerCount * iterationsPerWorker
        let modelBoxes = (0 ..< totalCalls).map { _ in SendableBox(model) }

        let resultBoxes = await withTaskGroup(of: SendableBox<KVCacheSimple>.self) { group in
            for w in 0 ..< workerCount {
                for i in 0 ..< iterationsPerWorker {
                    let callIndex = w * iterationsPerWorker + i
                    let modelBox = modelBoxes[callIndex]
                    // Every call shares the seed prefix (a genuine candidate
                    // for the same stored chunks) but appends its OWN
                    // distinct single token, mirroring independent
                    // conversations continuing from a shared system prompt.
                    let ownTokens = seedTokens + [2_000 + callIndex]
                    group.addTask {
                        let resolved = await cache.resolve(
                            modelID: modelID, newTokens: ownTokens, model: modelBox,
                            parameters: nil
                        ).consume()
                        let assembledCache = resolved.cache[0] as! KVCacheSimple

                        // Simulate the rest of this round: feed the one
                        // unmatched prompt token plus one freshly generated
                        // token, then store the grown round back -- real
                        // parallel resolve/store churn against one model id.
                        let newTokenCount = resolved.tokensToFeed.count + 1
                        let base = Int32(500_000 + callIndex * 100)
                        let newKeys = MLXArray(
                            base ..< (base + Int32(newTokenCount * headDim)),
                            [1, 1, newTokenCount, headDim])
                        let newValues = MLXArray(
                            (base + 50) ..< (base + 50 + Int32(newTokenCount * headDim)),
                            [1, 1, newTokenCount, headDim])
                        _ = assembledCache.update(keys: newKeys, values: newValues)

                        let grownTokens = ownTokens + [9_000 + callIndex]
                        await cache.store(
                            modelID: modelID, tokens: grownTokens,
                            cache: SendableBox([assembledCache]))

                        return SendableBox(assembledCache)
                    }
                }
            }
            var results: [SendableBox<KVCacheSimple>] = []
            for await box in group { results.append(box) }
            return results
        }

        #expect(resultBoxes.count == totalCalls)
        let identities = resultBoxes.map { ObjectIdentifier($0.consume()) }
        #expect(
            Set(identities).count == totalCalls,
            "sustained concurrent resolve/store churn must never hand back the same assembled instance twice"
        )

        // Bookkeeping consistency: every call's grown tokens share the exact
        // same seed prefix and are only 2 tokens longer (short of another
        // full chunk boundary), so the churn must dedup down to exactly the
        // seeded chunks -- never fewer (corruption/loss) and never more (a
        // dedup failure under concurrent insert).
        #expect(await cache.chunkCount(modelID: modelID) == seedChunkCount)
    }
}

#endif
