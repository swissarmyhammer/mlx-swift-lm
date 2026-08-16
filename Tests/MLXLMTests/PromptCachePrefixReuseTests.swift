// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Unit tests for the two functions that APPLY a reuse decision to live caches:
/// ``reusablePromptPrefix(promptTokens:cachedTokens:caches:)`` and
/// ``rewindPromptCache(_:to:)``.
///
/// ``PromptCacheReusePolicyTests`` covers the decision table itself. This suite
/// covers what happens to real caches afterwards, thus it uses `KVCacheSimple`
/// and `RotatingKVCache` with their offsets set by hand. No model, no weights
/// and no GPU work are needed: every function under test reads `offset`,
/// `isTrimmable` and `trim(_:)` alone.
@Suite("A live cache serves the prefix of a new prompt")
struct PromptCachePrefixReuseTests {

    /// The sliding window of the rotating cache this suite builds.
    ///
    /// `RotatingKVCache.isTrimmable` is `offset < maxCacheSize`, thus a cache
    /// placed past this window answers `false` and no rewind is available.
    private static let slidingWindow = 8

    /// A pair of unbounded caches placed at `offset`.
    ///
    /// Two caches rather than one, because every function under test asks about
    /// EVERY cache and a single-cache fixture would hide a wrong `allSatisfy`.
    private func simpleCaches(offset: Int) -> [KVCache] {
        let caches = [KVCacheSimple(), KVCacheSimple()]
        for cache in caches {
            cache.offset = offset
        }
        return caches
    }

    // MARK: - Extending a cached prefix

    @Test("a prompt that extends the ledger feeds only its suffix")
    func aPromptThatExtendsTheLedgerFeedsOnlyItsSuffix() {
        let caches = simpleCaches(offset: 3)

        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 3, 4, 5], cachedTokens: [1, 2, 3], caches: caches)

        #expect(prefix == 3)
        #expect(caches.allSatisfy { $0.offset == 3 })
    }

    @Test("an empty cache holds nothing and takes the whole prompt")
    func anEmptyCacheHoldsNothingAndTakesTheWholePrompt() {
        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 3], cachedTokens: [], caches: simpleCaches(offset: 0))

        #expect(prefix == 0)
    }

    @Test("a repeated prompt has no suffix to feed and gives no reuse")
    func aRepeatedPromptHasNoSuffixToFeedAndGivesNoReuse() {
        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 3], cachedTokens: [1, 2, 3], caches: simpleCaches(offset: 3))

        #expect(prefix == nil)
    }

    @Test("a cache whose position disagrees with the ledger gives no reuse")
    func aCacheWhosePositionDisagreesWithTheLedgerGivesNoReuse() {
        // The ledger names three tokens and the caches hold four. Nothing may be
        // spliced onto a cache that holds more than the ledger describes.
        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 3, 4, 5], cachedTokens: [1, 2, 3],
            caches: simpleCaches(offset: 4))

        #expect(prefix == nil)
    }

    @Test("no cache at all gives no reuse")
    func noCacheAtAllGivesNoReuse() {
        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 3], cachedTokens: [1, 2], caches: [])

        #expect(prefix == nil)
    }

    // MARK: - Rewinding to a common prefix

    @Test("a divergent prompt rewinds the caches to the common prefix")
    func aDivergentPromptRewindsTheCachesToTheCommonPrefix() {
        let caches = simpleCaches(offset: 5)

        let prefix = reusablePromptPrefix(
            promptTokens: [1, 2, 9, 9], cachedTokens: [1, 2, 3, 4, 5], caches: caches)

        #expect(prefix == 2)
        #expect(caches.allSatisfy { $0.offset == 2 })
    }

    @Test("a divergent prompt gives no reuse when the caches cannot rewind")
    func aDivergentPromptGivesNoReuseWhenTheCachesCannotRewind() {
        // A rotating cache past its window has overwritten the keys a rewind
        // needs, thus it answers `isTrimmable` false and the whole cache goes.
        let rotating = RotatingKVCache(maxSize: Self.slidingWindow)
        rotating.offset = 5
        let caches: [KVCache] = [rotating]
        #expect(rotating.isTrimmable)
        rotating.offset = Self.slidingWindow + 5

        let prefix = reusablePromptPrefix(
            promptTokens: Array(1 ... 13).map { $0 == 3 ? 99 : $0 },
            cachedTokens: Array(1 ... 13),
            caches: caches)

        #expect(prefix == nil)
    }

    // MARK: - Rewinding directly

    @Test("a rewind lands every cache on the requested position")
    func aRewindLandsEveryCacheOnTheRequestedPosition() {
        let caches = simpleCaches(offset: 10)

        #expect(rewindPromptCache(caches, to: 6))
        #expect(caches.allSatisfy { $0.offset == 6 })
    }

    @Test("a rewind to the position a cache already holds keeps it")
    func aRewindToThePositionACacheAlreadyHoldsKeepsIt() {
        let caches = simpleCaches(offset: 6)

        #expect(rewindPromptCache(caches, to: 6))
        #expect(caches.allSatisfy { $0.offset == 6 })
    }

    @Test("a rewind ahead of the cache is refused")
    func aRewindAheadOfTheCacheIsRefused() {
        #expect(rewindPromptCache(simpleCaches(offset: 3), to: 5) == false)
    }

    @Test("a rewind of a cache past its sliding window is refused")
    func aRewindOfACachePastItsSlidingWindowIsRefused() {
        let rotating = RotatingKVCache(maxSize: Self.slidingWindow)
        rotating.offset = Self.slidingWindow + 5

        #expect(rewindPromptCache([rotating], to: 4) == false)
    }

    @Test("a rewind with no cache at all is refused")
    func aRewindWithNoCacheAtAllIsRefused() {
        #expect(rewindPromptCache([], to: 0) == false)
    }
}
