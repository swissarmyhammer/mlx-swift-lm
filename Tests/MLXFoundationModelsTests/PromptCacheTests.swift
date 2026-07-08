// Copyright © 2026 Apple Inc.

import Foundation
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

#endif
