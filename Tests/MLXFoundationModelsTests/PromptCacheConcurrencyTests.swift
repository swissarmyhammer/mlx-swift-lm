// Copyright © 2026 Apple Inc.
//
// Actor-concurrency regression test for `PromptCache`: interleaved
// `resolve`/`store` calls from concurrent `Task`s must never hand the same
// `KVCache` instance out to two callers at once. `resolve()` implements
// this by REMOVING (not merely reading) the selected slot before returning
// it -- see `PromptCache.resolve`'s doc comment -- so a concurrent caller
// either finds a different slot or nothing at all, never the same one.
//
// `resolve()`'s body has no internal `await`, so Swift's actor isolation
// already serializes any two concurrent calls to completion -- a call
// cannot literally be preempted mid-body by another call to the same
// actor. That is exactly the property this test exists to guard: if a
// future change introduces an `await` inside `resolve()` (e.g. for some
// lazily-computed value), true interleaving becomes possible, and remove-
// before-return is what keeps it safe. Running this through real
// `Task`-based concurrency (rather than reasoning about it) means a
// regression shows up here as soon as it becomes structurally possible.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Actor-concurrency tests proving concurrent `resolve`/`store` calls never hand the same `KVCache` instance to two callers.
///
/// Uses the shared `PromptCacheProbeModel` from `PromptCacheTestSupport.swift`.
@Suite("PromptCache actor-concurrency: no double-checkout of a KVCache instance")
struct PromptCacheConcurrencyTests {

    @Test(
        """
        many concurrent resolve() calls racing for one pre-stored slot each get a \
        distinct KVCache instance -- the slot is checked out exactly once, everyone \
        else safely rebuilds
        """
    )
    func concurrentResolveNeverDoubleChecksOutASlot() async {
        let cache = PromptCache()
        let model: any LanguageModel = PromptCacheProbeModel()
        let modelID = "concurrency-probe-\(UUID().uuidString)"

        let storedTokens = [1, 2, 3, 4, 5]
        let storedCache = KVCacheSimple()
        let storedIdentity = ObjectIdentifier(storedCache)
        await cache.store(
            modelID: modelID, tokens: storedTokens, cache: SendableBox([storedCache]))

        // N concurrent callers, each with a prompt that shares the stored
        // slot's tokens as a genuine prefix (so every caller is a
        // candidate for the SAME single slot) but is individually distinct
        // (so no two callers coincidentally ask for byte-identical work).
        let callerCount = 12
        // `SendableBox` is single-use (`consume()` traps on a second call),
        // so each concurrent caller needs its OWN box wrapping the same
        // underlying model -- built here, outside any `@Sendable` closure,
        // since `model` itself (an `any LanguageModel` existential) isn't
        // Sendable and can't be captured directly by `group.addTask`.
        let modelBoxes = (0 ..< callerCount).map { _ in SendableBox(model) }

        // Collect the raw result BOXES (still unconsumed) from every
        // concurrent caller before extracting any `ObjectIdentifier`.
        // Consuming (and thus dropping the last reference to) a cache
        // instance while OTHER callers are still concurrently allocating
        // fresh ones risks the allocator reusing that exact address for a
        // later instance -- an `ObjectIdentifier` collision that looks like
        // a double-checkout but is really just ARC reusing freed memory.
        // Keeping every result alive until the whole concurrent phase has
        // finished rules that out.
        let resultBoxes = await withTaskGroup(
            of: SendableBox<(cache: [KVCache], tokensToFeed: [Int])>.self
        ) { group in
            for i in 0 ..< callerCount {
                let modelBox = modelBoxes[i]
                group.addTask {
                    await cache.resolve(
                        modelID: modelID, newTokens: storedTokens + [100 + i],
                        model: modelBox, parameters: nil)
                }
            }
            var results: [SendableBox<(cache: [KVCache], tokensToFeed: [Int])>] = []
            for await box in group {
                results.append(box)
            }
            return results
        }

        // Only now (every concurrent caller has finished and every cache
        // instance is being kept alive by `resultBoxes`) extract identities.
        let identities = resultBoxes.map { box -> ObjectIdentifier in
            // Single-layer fake model -- exactly one cache per result.
            // `KVCache` isn't statically class-constrained, but every
            // conformer in this codebase is a class.
            ObjectIdentifier(box.consume().cache[0] as AnyObject)
        }

        #expect(identities.count == callerCount)

        // Invariant: every concurrent caller got a DISTINCT KVCache
        // instance -- if two callers had been handed the same instance
        // (a double-checkout), this set would be smaller than the caller
        // count.
        #expect(
            Set(identities).count == callerCount,
            "expected \(callerCount) distinct KVCache instances, got \(Set(identities).count) -- a KVCache instance was handed out to more than one caller"
        )

        // Invariant: the pre-stored slot was checked out by EXACTLY one
        // caller (the rest necessarily rebuilt via model.newCache, since
        // only one slot existed for this modelID).
        let reuseCount = identities.filter { $0 == storedIdentity }.count
        #expect(
            reuseCount == 1,
            "expected exactly 1 caller to reuse the pre-stored slot, got \(reuseCount)"
        )
    }

    @Test(
        """
        sustained concurrent resolve/store churn against a shared model id never \
        double-checks-out a KVCache instance, across many interleavings
        """
    )
    func sustainedConcurrentChurnNeverDoubleChecksOut() async {
        let cache = PromptCache()
        let model: any LanguageModel = PromptCacheProbeModel()
        let modelID = "concurrency-churn-\(UUID().uuidString)"

        // Seed one slot so early callers have something to contend for.
        let seedTokens = [1, 2, 3]
        await cache.store(
            modelID: modelID, tokens: seedTokens, cache: SendableBox([KVCacheSimple()]))

        let tracker = CheckoutTracker()
        let workerCount = 8
        let iterationsPerWorker = 6

        // One fresh `SendableBox` per resolve() call (see the comment in
        // the test above -- `SendableBox` is single-use), built upfront
        // outside any `@Sendable` closure.
        let modelBoxes = (0 ..< (workerCount * iterationsPerWorker)).map { _ in SendableBox(model) }

        await withTaskGroup(of: Void.self) { group in
            for w in 0 ..< workerCount {
                group.addTask {
                    for i in 0 ..< iterationsPerWorker {
                        let newTokens = seedTokens + [w, i]
                        let modelBox = modelBoxes[w * iterationsPerWorker + i]
                        let resolved = await cache.resolve(
                            modelID: modelID, newTokens: newTokens, model: modelBox,
                            parameters: nil
                        ).consume()
                        let identities = resolved.cache.map { ObjectIdentifier($0 as AnyObject) }

                        await tracker.checkOut(identities)
                        // Widen the window in which another concurrent
                        // caller could (incorrectly) observe this cache as
                        // still available, before checking it back in.
                        await Task.yield()
                        await tracker.checkIn(identities)

                        await cache.store(
                            modelID: modelID, tokens: newTokens,
                            cache: SendableBox(resolved.cache))
                    }
                }
            }
        }

        let violations = await tracker.violations
        #expect(
            violations.isEmpty,
            "double-checkout violations recorded: \(violations)"
        )
        // Sanity: this many concurrent resolve() calls against a shared
        // model id, with the slot cap at 4, must have produced at least
        // one genuine checkout in total (i.e. the test actually exercised
        // real work, not a vacuous zero-iteration pass).
        let totalCheckouts = await tracker.totalCheckouts
        #expect(totalCheckouts == workerCount * iterationsPerWorker)
    }
}

/// Tracks which `KVCache` instances (by `ObjectIdentifier`) are currently
/// checked out across concurrent `Task`s, recording a violation if the same
/// identity is checked out twice before being checked back in.
///
/// An actor itself, so its own bookkeeping can't race regardless of how many
/// `Task`s call into it concurrently.
private actor CheckoutTracker {
    private var outstanding: Set<ObjectIdentifier> = []
    private(set) var violations: [String] = []
    private(set) var totalCheckouts = 0

    func checkOut(_ identities: [ObjectIdentifier]) {
        totalCheckouts += 1
        for identity in identities where outstanding.contains(identity) {
            violations.append("KVCache instance \(identity) checked out while already outstanding")
        }
        outstanding.formUnion(identities)
    }

    func checkIn(_ identities: [ObjectIdentifier]) {
        outstanding.subtract(identities)
    }
}

#endif
