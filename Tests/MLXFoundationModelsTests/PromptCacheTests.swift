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

    @Test("two or more fewer re-encoded tokens than the cache's real advance is untrustworthy")
    func twoOrMoreFewerReencodedTokensIsUntrustworthy() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11], actualGeneratedCount: 4)
        #expect(trusted == nil)
    }

    @Test(
        """
        one fewer re-encoded token than the cache's real advance (terminal EOS/stop \
        token decodes to empty text) is trusted as-is
        """
    )
    func oneFewerReencodedTokenIsTrustedAsIs() {
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [10, 11, 12], actualGeneratedCount: 4)
        #expect(trusted == [10, 11, 12])
    }

    @Test(
        """
        a round that generated a SINGLE token which was itself the EOS/stop token \
        (decoding to empty text, so re-encoding yields zero tokens) is trusted as an \
        empty array rather than rejected
        """
    )
    func pureEOSSingleTokenRoundReencodesToEmptyAndIsTrustedAsIs() {
        // actualGeneratedCount: 1 means the cache's real offset advanced by
        // exactly one token this round, and that one token was the
        // EOS/stop token itself -- it decodes to no text at all, so
        // `emittedText` is empty and re-encoding it yields zero tokens.
        // `commitPromptCache(...generatedTokenIDs:)`'s own
        // `guard !generatedTokenIDs.isEmpty else { return }` then leaves
        // any prior stored cache entry untouched rather than storing or
        // removing anything for this round -- there is nothing new to
        // add, but (unlike the old `nil`-triggered `removePromptCache`)
        // a previously valid cache entry from an earlier round is no
        // longer needlessly destroyed by this round's edge case.
        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: [], actualGeneratedCount: 1)
        #expect(trusted == [])
    }

    // The four tests above exercise `reconcileGeneratedTokens` with hand-picked
    // integer literals. The two below instead drive it with token IDs a real
    // `Tokenizer` conformance actually produced from a realistic decoded model
    // response (`ByteTokenizer`, also used elsewhere in this file/target --
    // see `TestHelpers.swift` -- gives every byte its own token ID, a genuine
    // `encode(text:addSpecialTokens:)` call rather than a synthetic count),
    // proving `commitPromptCache(...emittedText:tokenizer:)`'s call site --
    // which always feeds it a real tokenizer's `encode` output, never a
    // hand-built array -- reconciles correctly for that shape of input.

    @Test("realistic decoded text, exact count match, trusts the re-encoded tokens unchanged")
    func realisticTextExactCountMatchTrusted() {
        let tokenizer = ByteTokenizer()
        let emittedText = "Sure! Here's a haiku about the ocean:\n\nWaves crash on the shore."
        let reencoded = tokenizer.encode(text: emittedText, addSpecialTokens: false)

        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: reencoded, actualGeneratedCount: reencoded.count)

        #expect(trusted == reencoded)
    }

    @Test(
        """
        realistic decoded text, cache advance one short (natural-stop-token \
        prefetch case), drops the trailing re-encoded token
        """
    )
    func realisticTextNaturalStopOffByOneDropsTrailingToken() {
        // Mirrors the doc comment on `commitPromptCache(...emittedText:tokenizer:)`:
        // `TokenIterator`'s next()-ahead prefetch discards the terminal
        // EOS/stop token that already advanced the cache without ever handing
        // it to the emitted-text stream, so the cache's real offset advance
        // legitimately lands one token short of the full re-encoding.
        let tokenizer = ByteTokenizer()
        let emittedText = "Absolutely, here is a short poem about autumn leaves falling gently."
        let reencoded = tokenizer.encode(text: emittedText, addSpecialTokens: false)
        let actualGeneratedCount = reencoded.count - 1

        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: reencoded, actualGeneratedCount: actualGeneratedCount)

        #expect(trusted == Array(reencoded.dropLast()))
    }

    @Test(
        """
        realistic decoded text, cache advance one MORE than the re-encoded tokens \
        (terminal EOS/stop token decodes to empty text) trusts the re-encoded tokens as-is
        """
    )
    func realisticTextTerminalEOSOffByOneTrustsReencodingAsIs() {
        // The mirror image of `realisticTextNaturalStopOffByOneDropsTrailingToken`
        // above: here the model's actual final generated token IS the EOS/stop
        // token itself. That token advances `cache.offset` (contributing to
        // `actualGeneratedCount`) but decodes to no text at all, so re-encoding
        // `emittedText` recovers one FEWER token than the cache's real advance.
        // Trusting the shorter re-encoding as-is (rather than rejecting it)
        // lets the caller's `reconcileCacheAdvance`/`trimCacheByOne` path bring
        // the cache back in sync instead of wiping the entire entry.
        let tokenizer = ByteTokenizer()
        let emittedText = "Absolutely, here is a short poem about autumn leaves falling gently."
        let reencoded = tokenizer.encode(text: emittedText, addSpecialTokens: false)
        let actualGeneratedCount = reencoded.count + 1

        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: reencoded, actualGeneratedCount: actualGeneratedCount)

        #expect(trusted == reencoded)
    }

    @Test(
        """
        the terminal-EOS-decodes-to-empty-text reconciliation composes with \
        reconcileCacheAdvance/trimAndVerify to correctly trim and store the cache
        """
    )
    func terminalEOSCaseComposesWithCacheAdvanceTrimAndVerify() {
        // End-to-end (at the pure-function level): proves the RESULT of
        // feeding `reconcileGeneratedTokens`'s trusted (shorter) array into
        // the same `reconcileCacheAdvance` + `trimAndVerify` machinery
        // `commitPromptCache(modelID:slot:generatedTokenIDs:)` uses is a
        // correctly trimmed cache -- not just that reconciliation returns
        // non-nil.
        let tokenizer = ByteTokenizer()
        let emittedText = "Absolutely, here is a short poem about autumn leaves falling gently."
        let reencoded = tokenizer.encode(text: emittedText, addSpecialTokens: false)
        let promptTokenCount = 20
        // The cache's real offset advance includes the EOS/stop token that
        // decoded to no text, so it is one MORE than the re-encoding recovers.
        let actualGeneratedCount = reencoded.count + 1

        let trusted = PromptCache.reconcileGeneratedTokens(
            reencoded: reencoded, actualGeneratedCount: actualGeneratedCount)
        #expect(trusted == reencoded)

        let outcome = PromptCache.reconcileCacheAdvance(
            observedTokenCount: trusted!.count, cacheAdvance: actualGeneratedCount)
        #expect(outcome == .trimCacheByOne)

        let cache = KVCacheSimple()
        cache.offset = promptTokenCount + actualGeneratedCount

        let verified = PromptCache.trimAndVerify(
            [cache], from: promptTokenCount + actualGeneratedCount,
            to: promptTokenCount + trusted!.count)

        #expect(verified == true)
        #expect(cache.offset == promptTokenCount + reencoded.count)
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

/// Tests for `PromptCache.planCacheStore`: the store-action decision a round
/// takes after `reconcileCacheAdvance`, given the cache's trimmability and the
/// identity of any fed-but-uncounted terminal stop token. This is the seam that
/// fixes hybrid Mamba/attention checkpoints being dropped on EOS-terminated
/// rounds (kanban `b3zpf2q`): a `.trimCacheByOne` gap on a non-trimmable cache
/// whose extra fed token is known must now store the EXTENDED sequence, not drop.
@Suite("PromptCache store-plan selection")
struct PromptCacheStorePlanTests {

    @Test("a matching advance stores the observed sequence as-is, regardless of trimmability")
    func matchesStores() {
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .matches, cacheIsTrimmable: true, fedStopToken: nil) == .store)
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .matches, cacheIsTrimmable: false, fedStopToken: 7) == .store)
    }

    @Test("a one-ahead advance on a trimmable cache trims then stores (pure-attention path)")
    func trimmableTrimsThenStores() {
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .trimCacheByOne, cacheIsTrimmable: true, fedStopToken: 7)
                == .trimThenStore)
    }

    @Test(
        "a one-ahead advance on a NON-trimmable cache with a known fed token stores the EXTENDED sequence"
    )
    func nonTrimmableKnownTokenStoresExtended() {
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .trimCacheByOne, cacheIsTrimmable: false, fedStopToken: 42)
                == .storeExtended(fedStopToken: 42))
    }

    @Test("a one-ahead advance on a non-trimmable cache with NO known fed token drops (can't guess)")
    func nonTrimmableUnknownTokenDrops() {
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .trimCacheByOne, cacheIsTrimmable: false, fedStopToken: nil)
                == .drop)
    }

    @Test("an untrustworthy advance always drops, whatever the trimmability or fed token")
    func untrustworthyDrops() {
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .untrustworthy, cacheIsTrimmable: false, fedStopToken: 7) == .drop)
        #expect(
            PromptCache.planCacheStore(
                reconciliation: .untrustworthy, cacheIsTrimmable: true, fedStopToken: nil) == .drop)
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
