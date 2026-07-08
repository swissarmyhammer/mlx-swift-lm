// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache prefix/trim/rebuild decision")
struct PromptCacheDecisionTests {

    @Test("empty cached tokens (first call) rebuilds regardless of trimmability")
    func firstCallRebuilds() {
        let decision = PromptCache.decide(
            cachedTokens: [], newTokens: [1, 2, 3], isTrimmable: true)
        #expect(decision == .rebuild)

        let decisionNotTrimmable = PromptCache.decide(
            cachedTokens: [], newTokens: [1, 2, 3], isTrimmable: false)
        #expect(decisionNotTrimmable == .rebuild)
    }

    @Test("exact prefix reuses the cache and feeds only the suffix")
    func exactPrefixReusesSuffix() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3], newTokens: [1, 2, 3, 4, 5], isTrimmable: true)
        #expect(decision == .reuseSuffix(count: 2))
    }

    @Test("exact prefix reuse does not depend on trimmability")
    func exactPrefixReusesSuffixWhenNotTrimmable() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3], newTokens: [1, 2, 3, 4, 5], isTrimmable: false)
        #expect(decision == .reuseSuffix(count: 2))
    }

    @Test("identical token sequences reuse with a zero-length suffix")
    func identicalSequencesReuseZeroSuffix() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3], newTokens: [1, 2, 3], isTrimmable: true)
        #expect(decision == .reuseSuffix(count: 0))
    }

    @Test("divergence with a trimmable cache trims to the common prefix")
    func divergenceTrimsWhenTrimmable() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3, 4], newTokens: [1, 2, 9, 10], isTrimmable: true)
        #expect(decision == .trimTo(commonPrefixLength: 2, thenSuffix: 2))
    }

    @Test("divergence with a non-trimmable cache rebuilds")
    func divergenceRebuildsWhenNotTrimmable() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3, 4], newTokens: [1, 2, 9, 10], isTrimmable: false)
        #expect(decision == .rebuild)
    }

    @Test("divergence at the very first token trims to an empty common prefix")
    func divergenceAtFirstTokenTrimsToZero() {
        let decision = PromptCache.decide(
            cachedTokens: [7, 8, 9], newTokens: [1, 2, 3, 4], isTrimmable: true)
        #expect(decision == .trimTo(commonPrefixLength: 0, thenSuffix: 4))
    }

    @Test("new tokens shorter than cached tokens still trims to the common prefix")
    func shorterNewTokensTrims() {
        let decision = PromptCache.decide(
            cachedTokens: [1, 2, 3, 4], newTokens: [1, 2, 3], isTrimmable: true)
        #expect(decision == .trimTo(commonPrefixLength: 3, thenSuffix: 0))
    }
}

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

@Suite("PromptCache multi-slot longest-common-prefix selection")
struct PromptCacheSlotSelectionTests {

    @Test("no candidates selects nothing")
    func noCandidatesSelectsNothing() {
        let index = PromptCache.selectSlot(candidates: [], newTokens: [1, 2, 3])
        #expect(index == nil)
    }

    @Test("single candidate with zero overlap selects nothing")
    func zeroOverlapSelectsNothing() {
        let index = PromptCache.selectSlot(
            candidates: [(tokens: [9, 9, 9], lastUsed: 1)], newTokens: [1, 2, 3])
        #expect(index == nil)
    }

    @Test("picks the candidate with the longest common prefix")
    func picksLongestCommonPrefix() {
        let index = PromptCache.selectSlot(
            candidates: [
                (tokens: [1, 2, 9, 9], lastUsed: 1),
                (tokens: [1, 2, 3, 9], lastUsed: 2),
                (tokens: [1, 9, 9, 9], lastUsed: 3),
            ],
            newTokens: [1, 2, 3, 4])
        #expect(index == 1)
    }

    @Test("ties in LCP length go to the most recently used candidate")
    func tiesGoToMostRecentlyUsed() {
        let index = PromptCache.selectSlot(
            candidates: [
                (tokens: [1, 2, 3], lastUsed: 5),
                (tokens: [1, 2, 9], lastUsed: 10),
                (tokens: [1, 2, 3], lastUsed: 1),
            ],
            newTokens: [1, 2, 3, 4])
        #expect(index == 0)
    }

    @Test("a full-length match still wins over a longer but partial candidate")
    func fullLengthMatchIsEligible() {
        let index = PromptCache.selectSlot(
            candidates: [
                (tokens: [1, 2, 3], lastUsed: 1),
                (tokens: [1, 2, 3, 4, 5, 6, 7, 9], lastUsed: 2),
            ],
            newTokens: [1, 2, 3])
        // Both share an LCP of 3; the more recently used wins the tie.
        #expect(index == 1)
    }
}

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
}

#endif
