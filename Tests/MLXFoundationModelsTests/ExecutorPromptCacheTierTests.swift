// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels

/// Tests for ``ExecutorPromptCacheTier`` and ``ExecutorPromptCacheLRUTier``, which keep the
/// records of one tier of the prompt cache store, their byte total and, for the LRU tier, their
/// usage order.
///
/// No weights are needed. Each record is a byte count alone.
@Suite("A tier of the prompt cache store keeps its records, their bytes and their usage order")
struct ExecutorPromptCacheTierTests: PromptCacheKeyFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache-tier"

    /// A record of a fixed size.
    private struct Record: ExecutorPromptCacheTierRecord, Equatable {

        /// The bytes of the record.
        let byteCount: Int
    }

    /// The size of the first record of each test.
    private static let firstByteCount = 10

    /// The size of the second record of each test.
    private static let secondByteCount = 20

    /// A budget that both records exceed together, and neither alone.
    private static let budget = 25

    /// An LRU tier that holds `first` and then `second`, least recently used first.
    private func lruTierOfTwo() -> ExecutorPromptCacheLRUTier<Record> {
        var tier = ExecutorPromptCacheLRUTier<Record>()
        tier.insert(Record(byteCount: Self.firstByteCount), for: key("first"))
        tier.insert(Record(byteCount: Self.secondByteCount), for: key("second"))
        return tier
    }

    @Test("an insert adds the bytes of the record to the total")
    func anInsertAddsTheBytesOfTheRecord() {
        var tier = ExecutorPromptCacheTier<Record>()
        tier.insert(Record(byteCount: Self.firstByteCount), for: key("first"))
        tier.insert(Record(byteCount: Self.secondByteCount), for: key("second"))

        #expect(tier.byteCount == Self.firstByteCount + Self.secondByteCount)
    }

    @Test("an insert under a key the tier holds replaces the record and its bytes")
    func anInsertUnderAHeldKeyReplacesTheRecord() {
        var tier = ExecutorPromptCacheTier<Record>()
        tier.insert(Record(byteCount: Self.firstByteCount), for: key("first"))
        tier.insert(Record(byteCount: Self.secondByteCount), for: key("first"))

        #expect(tier.byteCount == Self.secondByteCount)
    }

    @Test("a removal gives the record back and subtracts its bytes")
    func aRemovalGivesTheRecordBack() {
        var tier = ExecutorPromptCacheTier<Record>()
        tier.insert(Record(byteCount: Self.firstByteCount), for: key("first"))

        let removed = tier.remove(key("first"))

        #expect(removed == Record(byteCount: Self.firstByteCount) && tier.byteCount == 0)
    }

    @Test("a removal of a key the tier does not hold gives nil")
    func aRemovalOfAMissingKeyGivesNil() {
        var tier = ExecutorPromptCacheTier<Record>()

        #expect(tier.remove(key("missing")) == nil)
    }

    @Test("an LRU tier over its budget gives up its least recently used record first")
    func anLRUTierOverItsBudgetGivesUpTheOldestRecord() {
        var tier = lruTierOfTwo()

        let evicted = tier.removeLeastRecentlyUsed(whileOver: Self.budget)

        #expect(evicted?.key == key("first"))
    }

    @Test("an LRU tier at or below its budget gives up nothing")
    func anLRUTierWithinItsBudgetGivesUpNothing() {
        var tier = lruTierOfTwo()

        #expect(
            tier.removeLeastRecentlyUsed(whileOver: Self.firstByteCount + Self.secondByteCount)
                == nil)
    }

    @Test("an LRU tier that removed a key no longer offers it for eviction")
    func anLRUTierForgetsTheOrderOfARemovedKey() {
        var tier = lruTierOfTwo()
        tier.remove(key("first"))

        let evicted = tier.removeLeastRecentlyUsed(whileOver: 0)

        #expect(evicted?.key == key("second"))
    }

    @Test("an LRU tier takes a key inserted again as its most recently used key")
    func anLRUTierTakesAReinsertedKeyAsTheNewest() {
        var tier = lruTierOfTwo()
        tier.insert(Record(byteCount: Self.firstByteCount), for: key("first"))

        let evicted = tier.removeLeastRecentlyUsed(whileOver: Self.budget)

        #expect(evicted?.key == key("second"))
    }

    @Test("an LRU tier over its budget passes over a record the eviction refuses")
    func anLRUTierPassesOverARefusedRecord() {
        var tier = lruTierOfTwo()

        let evicted = tier.removeLeastRecentlyUsed(
            whileOver: Self.budget, where: { $0.byteCount != Self.firstByteCount })

        #expect(evicted?.key == key("second"))
    }

    @Test("an LRU tier whose eviction refuses every record gives up nothing")
    func anLRUTierWhoseEvictionRefusesEveryRecordGivesUpNothing() {
        var tier = lruTierOfTwo()

        #expect(tier.removeLeastRecentlyUsed(whileOver: 0, where: { _ in false }) == nil)
    }
}

#endif
