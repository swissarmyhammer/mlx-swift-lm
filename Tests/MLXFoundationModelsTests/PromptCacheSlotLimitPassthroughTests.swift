// Copyright © 2026 Apple Inc.
//
// Coverage for `MLXLanguageModel.setPromptCacheSlotLimit(_:)`: the public
// static passthrough that forwards to the process-global `PromptCache`
// actor's `setMaxSlotsPerModel(_:)`, mirroring the existing `evictAll()`
// passthrough pattern. `PromptCacheSlotPoolTests` already exercises
// `PromptCache.setMaxSlotsPerModel` directly against a locally-constructed
// actor instance; this file proves the *public* entry point actually reaches
// the ONE process-global instance `MLXLanguageModel` holds, by driving the
// same round-trip through `MLXLanguageModel`'s own internal
// `resolvePromptCache`/`storePromptCache` test seams (the exact functions
// `Executor` calls in production) instead of a fresh `PromptCache()`.
//
// `setMaxSlotsPerModel` is a single actor-wide value, not per-model, so
// lowering it here affects EVERY model's eviction cap process-wide for as
// long as it's set. A UUID-unique `modelID` isolates this test's SLOT
// STORAGE from other tests, but not from `MLXLanguageModel.evictAll()`,
// which is key-agnostic and drops every model's slots in the same
// process-global `promptCache` instance (see `ModelCacheEvictionTests.swift`
// and `TokenizerBiasCacheTests.swift`, both of which call it in this same
// test bundle). Nested under `FoundationModelsCacheTests` -- the shared
// `.serialized` parent those suites already extend -- so this suite never
// runs concurrently with an `evictAll()` caller and can't have its
// just-stored slots wiped mid-test by an unrelated suite.
import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

extension FoundationModelsCacheTests {

    @Suite("MLXLanguageModel.setPromptCacheSlotLimit process-global passthrough")
    struct PromptCacheSlotLimitPassthrough {

        private let model = PromptCacheProbeModel()

        @Test(
            """
            setPromptCacheSlotLimit(_:) reaches the process-global PromptCache: lowering it to 1 \
            evicts an older slot the next time a model's prompt cache is stored through \
            MLXLanguageModel's own resolve/store seam, not just a locally-constructed PromptCache
            """
        )
        func setPromptCacheSlotLimitEvictsThroughProcessGlobalCache() async {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            // `defer` bodies run synchronously and cannot `await`, so the
            // process-global limit is restored via an explicit call at the end
            // of this function instead -- reached regardless of whether the
            // `#expect`s below pass or fail, since `#expect` records a failure
            // without unwinding the function early.
            let modelID = "process-global-slot-limit-\(UUID().uuidString)"

            await MLXLanguageModel.setPromptCacheSlotLimit(1)

            let olderTokens = [1, 2, 3]
            let olderCache = makeSlotCache(tokenCount: olderTokens.count)
            await MLXLanguageModel.storePromptCache(modelID: modelID, tokens: olderTokens, cache: [olderCache])

            let newerTokens = [4, 5, 6]
            let newerCache = makeSlotCache(tokenCount: newerTokens.count)
            await MLXLanguageModel.storePromptCache(modelID: modelID, tokens: newerTokens, cache: [newerCache])

            // With the process-global limit lowered to 1 via the public
            // passthrough, storing the second slot must have evicted the first:
            // resolving its continuation forces a rebuild rather than reusing it.
            let resolvedOlder = await MLXLanguageModel.resolvePromptCache(
                modelID: modelID, newTokens: olderTokens + [9], model: model, parameters: nil)
            #expect(
                cacheIdentity(resolvedOlder) != ObjectIdentifier(olderCache),
                "setPromptCacheSlotLimit(1) must have reached the process-global cache and evicted the older slot")
            #expect(
                resolvedOlder.tokensToFeed == olderTokens + [9], "an evicted slot rebuilds from the full prompt")

            // The newest slot must have survived under the lowered cap.
            let resolvedNewer = await MLXLanguageModel.resolvePromptCache(
                modelID: modelID, newTokens: newerTokens + [9], model: model, parameters: nil)
            #expect(
                cacheIdentity(resolvedNewer) == ObjectIdentifier(newerCache),
                "the newest slot must survive under the limit set via the public passthrough")

            // Restore the default so no other test in this process observes a
            // process-global cap left at 1.
            await MLXLanguageModel.setPromptCacheSlotLimit(PromptCache.defaultMaxSlotsPerModel)
        }
    }
}

#endif
