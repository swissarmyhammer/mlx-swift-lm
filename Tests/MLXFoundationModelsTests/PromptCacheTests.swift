// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache re-encoded-token-count reconciliation")
struct PromptCacheReconciliationTests {

    @Test("exact count match trusts the re-encoded tokens unchanged")
    func exactCountMatchTrusted() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11, 12], actualGeneratedCount: 3)
        #expect(trusted == [10, 11, 12])
    }

    @Test(
        """
        one extra re-encoded token (GuidedGenerationLoop.run's natural-termination \
        off-by-one) drops the trailing token
        """
    )
    func offByOneDropsTrailingToken() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11, 12], actualGeneratedCount: 2)
        #expect(trusted == [10, 11])
    }

    @Test("a mismatch of more than one token is untrustworthy")
    func largerMismatchIsUntrustworthy() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11, 12], actualGeneratedCount: 1)
        #expect(trusted == nil)
    }

    @Test("fewer re-encoded tokens than the cache's real advance is untrustworthy")
    func fewerReencodedTokensIsUntrustworthy() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11], actualGeneratedCount: 3)
        #expect(trusted == nil)
    }
}

@Suite("PromptCache real-token-ID cache-advance reconciliation")
struct PromptCacheAdvanceReconciliationTests {

    @Test("exact match")
    func exactMatch() {
        let outcome = PromptCache.reconcileCacheAdvance(observedTokenCount: 5, cacheAdvance: 5)
        #expect(outcome == .matches)
    }

    @Test("cache one token ahead of observed IDs asks to trim the cache back by one")
    func oneAheadTrimsCacheByOne() {
        let outcome = PromptCache.reconcileCacheAdvance(observedTokenCount: 5, cacheAdvance: 6)
        #expect(outcome == .trimCacheByOne)
    }

    @Test("any larger gap is untrustworthy")
    func largerGapIsUntrustworthy() {
        #expect(
            PromptCache.reconcileCacheAdvance(observedTokenCount: 5, cacheAdvance: 7)
                == .untrustworthy)
    }

    @Test("cache behind the observed IDs is untrustworthy")
    func cacheBehindObservedIsUntrustworthy() {
        #expect(
            PromptCache.reconcileCacheAdvance(observedTokenCount: 5, cacheAdvance: 4)
                == .untrustworthy)
    }
}

/// Tests for `PromptCache.trimAndVerify`'s trim-then-confirm-offset verification logic.
@Suite("PromptCache trim-and-verify")
struct PromptCacheTrimAndVerifyTests {

    @Test("empty cache array is never trusted")
    func emptyCacheIsUntrusted() {
        #expect(PromptCache.trimAndVerify([], from: 10, to: 5) == false)
    }

    @Test("trimming an unbounded cache to the requested offset verifies true")
    func unboundedCacheTrimVerifies() {
        let cache = KVCacheSimple()
        cache.offset = 10
        let verified = PromptCache.trimAndVerify([cache], from: 10, to: 6)
        #expect(verified == true)
        #expect(cache.offset == 6)
    }

    @Test("a trim request inconsistent with the cache's real offset fails verification")
    func mismatchedAssumedOffsetFailsVerification() {
        // The cache is really only at offset 5, but the caller (incorrectly)
        // believes it is at 10 and asks to trim down to 3 -- trimPromptCache
        // computes numTokens = 10 - 3 = 7, but KVCacheSimple.trim can only
        // trim min(5, 7) = 5, landing the real offset at 0, not the
        // requested 3. Verification must catch this rather than trusting
        // the request.
        let cache = KVCacheSimple()
        cache.offset = 5
        let verified = PromptCache.trimAndVerify([cache], from: 10, to: 3)
        #expect(verified == false)
    }

    @Test(
        """
        ChunkedKVCache with startPosition > 0 can legitimately fall short of a \
        trim request -- trimAndVerify's rejection path catches it
        """
    )
    func chunkedCacheTrimShortfallIsRejected() {
        // ChunkedKVCache.trim is bounded by `offset - startPosition` (see
        // KVCache.swift), unlike KVCacheSimple.trim which is unbounded. A
        // real model run can legitimately produce startPosition > 0 (chunks
        // dropped off the front once the window fills); constructed here
        // directly via `metaState` (chunkSize, startPosition) rather than
        // requiring a real model to drive `maybeTrimFront()`.
        let cache = ChunkedKVCache(chunkSize: 4)
        cache.offset = 10
        cache.metaState = ["4", "6"]  // chunkSize: 4, startPosition: 6

        // trimAndVerify(from: 10, to: 2) asks trimPromptCache for
        // numTokens = 10 - 2 = 8, but ChunkedKVCache.trim can only trim
        // min(offset - startPosition, 8) = min(4, 8) = 4, landing the real
        // offset at 6, not the requested 2.
        let verified = PromptCache.trimAndVerify([cache], from: 10, to: 2)
        #expect(verified == false)
        #expect(
            cache.offset == 6,
            "trim should have been bounded by offset - startPosition, landing short of the request"
        )
    }
}

#endif
