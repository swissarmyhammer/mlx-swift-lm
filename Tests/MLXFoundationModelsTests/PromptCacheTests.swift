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

// MARK: - Property-style / table-driven tests over randomly generated token sequences
//
// Swift Testing has no mainstream property-testing library in this repo's
// dependencies (checked Package.swift), so these run the same fixed,
// seeded pseudo-random sequence of cases on every invocation -- fully
// deterministic and reproducible, without adding a new dependency.

/// Deterministic, dependency-free PRNG (SplitMix64) so these tests generate
/// the exact same cases on every run, in CI and locally alike.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A short, sharply overlapping token vocabulary (0..<6) so randomly
/// generated sequences frequently share genuine prefixes -- with a large
/// vocabulary, almost every random pair would diverge at token 0, never
/// exercising `.reuseSuffix` or a nontrivial `.trimTo`.
private func randomTokenSequence(
    length: Int, vocabulary: Int, using generator: inout SplitMix64
) -> [Int] {
    (0 ..< length).map { _ in Int(generator.next() % UInt64(vocabulary)) }
}

/// Mirrors `Decision`'s documented semantics: the tokens assumed already
/// resident in the cache after applying `decision`, and the suffix a caller
/// would feed on top of them. Test-only -- independent of
/// `PromptCache.applyDecision`, which additionally handles trim
/// verification/fallback; this only checks `decide`'s own arithmetic.
private func residentPrefixAndFeedSuffix(
    for decision: PromptCache.Decision, cachedTokens: [Int], newTokens: [Int]
) -> (resident: [Int], suffix: [Int]) {
    switch decision {
    case .reuseSuffix(let count):
        return (cachedTokens, Array(newTokens.suffix(count)))
    case .trimTo(let commonPrefixLength, let thenSuffix):
        return (Array(newTokens.prefix(commonPrefixLength)), Array(newTokens.suffix(thenSuffix)))
    case .rebuild:
        return ([], newTokens)
    }
}

@Suite("PromptCache.decide property tests over randomly generated token sequences")
struct PromptCacheDecidePropertyTests {

    @Test("decide's invariants hold across many random (cachedTokens, newTokens, isTrimmable) cases")
    func decideInvariantsHoldAcrossRandomCases() {
        var generator = SplitMix64(seed: 0xC0FF_EE12_3456_789A)

        for _ in 0 ..< 2000 {
            let cachedLength = Int(generator.next() % 9)
            let newLength = Int(generator.next() % 9)
            let cachedTokens = randomTokenSequence(
                length: cachedLength, vocabulary: 4, using: &generator)
            let newTokens = randomTokenSequence(
                length: newLength, vocabulary: 4, using: &generator)
            let isTrimmable = generator.next() % 2 == 0

            let decision = PromptCache.decide(
                cachedTokens: cachedTokens, newTokens: newTokens, isTrimmable: isTrimmable)

            let isGenuinePrefix =
                !cachedTokens.isEmpty && newTokens.count >= cachedTokens.count
                && Array(newTokens.prefix(cachedTokens.count)) == cachedTokens

            switch decision {
            case .trimTo(let commonPrefixLength, let thenSuffix):
                // Invariant: never .trimTo when !isTrimmable.
                #expect(
                    isTrimmable,
                    "produced .trimTo with isTrimmable == false for cached=\(cachedTokens) new=\(newTokens)"
                )
                // Invariant: commonPrefixLength + thenSuffix == newTokens.count.
                #expect(
                    commonPrefixLength + thenSuffix == newTokens.count,
                    "commonPrefixLength(\(commonPrefixLength)) + thenSuffix(\(thenSuffix)) != newTokens.count(\(newTokens.count))"
                )
                // .trimTo only ever fires on genuine divergence, never on a prefix match.
                #expect(!isGenuinePrefix)

            case .reuseSuffix:
                // Invariant: .reuseSuffix iff cached tokens are genuinely a
                // prefix of new tokens (and non-empty -- decide always
                // rebuilds on an empty cache regardless of prefix triviality).
                #expect(
                    isGenuinePrefix,
                    "produced .reuseSuffix without a genuine prefix match for cached=\(cachedTokens) new=\(newTokens)"
                )

            case .rebuild:
                // Invariant (contrapositive of the .reuseSuffix iff): a
                // genuine prefix match never rebuilds.
                #expect(!isGenuinePrefix || cachedTokens.isEmpty)
            }

            // Invariant: .reuseSuffix iff isGenuinePrefix -- checked from the
            // other direction too (a genuine non-empty prefix always yields
            // .reuseSuffix, regardless of isTrimmable).
            if isGenuinePrefix {
                guard case .reuseSuffix = decision else {
                    Issue.record(
                        "genuine prefix match cached=\(cachedTokens) new=\(newTokens) did not produce .reuseSuffix (got \(decision))"
                    )
                    continue
                }
            }

            // Reconstruction invariant: feeding the decided suffix on top of
            // the decided resident prefix reproduces newTokens exactly, for
            // every decision kind (not just the ones with an interesting
            // suffix).
            let (resident, suffix) = residentPrefixAndFeedSuffix(
                for: decision, cachedTokens: cachedTokens, newTokens: newTokens)
            #expect(
                resident + suffix == newTokens,
                "reconstruction mismatch for decision \(decision): resident=\(resident) suffix=\(suffix) newTokens=\(newTokens)"
            )
        }
    }
}

@Suite("PromptCache.selectSlot property tests over randomly generated candidate pools")
struct PromptCacheSelectSlotPropertyTests {

    private func lcpLength(_ a: [Int], _ b: [Int]) -> Int {
        zip(a, b).prefix { $0 == $1 }.count
    }

    @Test("selectSlot's invariants hold across many random candidate pools")
    func selectSlotInvariantsHoldAcrossRandomCases() {
        var generator = SplitMix64(seed: 0x5EED_FACE_D00D_1234)

        for _ in 0 ..< 2000 {
            let candidateCount = Int(generator.next() % 5)
            let newLength = Int(generator.next() % 8)
            let newTokens = randomTokenSequence(length: newLength, vocabulary: 4, using: &generator)

            let candidates: [(tokens: [Int], lastUsed: Int)] = (0 ..< candidateCount).map { _ in
                let length = Int(generator.next() % 8)
                let tokens = randomTokenSequence(length: length, vocabulary: 4, using: &generator)
                let lastUsed = Int(generator.next() % 100)
                return (tokens: tokens, lastUsed: lastUsed)
            }

            let selected = PromptCache.selectSlot(candidates: candidates, newTokens: newTokens)
            let lcps = candidates.map { lcpLength($0.tokens, newTokens) }

            guard let selected else {
                // Invariant: nil iff every candidate has zero overlap.
                #expect(
                    lcps.allSatisfy { $0 == 0 },
                    "selectSlot returned nil despite a candidate with nonzero LCP: candidates=\(candidates) new=\(newTokens)"
                )
                continue
            }

            #expect(candidates.indices.contains(selected))
            let selectedLCP = lcps[selected]
            let selectedLastUsed = candidates[selected].lastUsed

            // Invariant: the selected candidate has nonzero overlap.
            #expect(selectedLCP > 0)

            // Invariant: no other candidate strictly beats the selection --
            // by LCP first, then by lastUsed on a tie.
            for (index, candidate) in candidates.enumerated() where index != selected {
                let candidateLCP = lcps[index]
                let dominatesOnLCP = candidateLCP > selectedLCP
                let tiesAndIsNewer = candidateLCP == selectedLCP && candidate.lastUsed > selectedLastUsed
                #expect(
                    !dominatesOnLCP && !tiesAndIsNewer,
                    """
                    candidate \(index) (lcp=\(candidateLCP), lastUsed=\(candidate.lastUsed)) should have \
                    beaten the selected candidate \(selected) (lcp=\(selectedLCP), lastUsed=\(selectedLastUsed))
                    """
                )
            }
        }
    }
}

#endif
