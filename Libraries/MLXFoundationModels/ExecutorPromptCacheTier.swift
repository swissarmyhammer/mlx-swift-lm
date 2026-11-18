// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// A record that one tier of ``ExecutorPromptCacheStore`` keeps, and whose bytes the tier
/// counts.
protocol ExecutorPromptCacheTierRecord {

    /// The bytes the record holds on its tier.
    var byteCount: Int { get }
}

/// The records of one tier of ``ExecutorPromptCacheStore``, by session, and the sum of their
/// bytes.
///
/// The records and the total change together in ``insert(_:for:)`` and ``remove(_:)`` alone,
/// thus the total always equals the sum of the bytes of the records.
struct ExecutorPromptCacheTier<Record: ExecutorPromptCacheTierRecord> {

    /// The records, by session.
    private var records: [ExecutorPromptCacheKey: Record] = [:]

    /// The sum of ``ExecutorPromptCacheTierRecord/byteCount`` over the records.
    private(set) var byteCount = 0

    /// The sessions that hold a record.
    var keys: Dictionary<ExecutorPromptCacheKey, Record>.Keys { records.keys }

    /// How many sessions hold a record.
    var count: Int { records.count }

    /// The record of `key`, or nil when the tier holds none.
    ///
    /// - Parameter key: The session of the record.
    subscript(key: ExecutorPromptCacheKey) -> Record? { records[key] }

    /// Keeps `record` under `key`, in the place of the record that `key` held.
    ///
    /// - Parameters:
    ///   - record: The record.
    ///   - key: The session of the record.
    mutating func insert(_ record: Record, for key: ExecutorPromptCacheKey) {
        remove(key)
        records[key] = record
        byteCount += record.byteCount
    }

    /// Removes the record of `key` and its bytes.
    ///
    /// - Parameter key: The session of the record.
    /// - Returns: The record, or nil when the tier holds none for `key`.
    @discardableResult
    mutating func remove(_ key: ExecutorPromptCacheKey) -> Record? {
        guard let record = records.removeValue(forKey: key) else { return nil }
        byteCount -= record.byteCount
        return record
    }
}

/// An ``ExecutorPromptCacheTier`` that also keeps the order in which its sessions were used,
/// for a tier that a byte budget limits.
///
/// An insert makes its key the most recently used key. ``removeLeastRecentlyUsed(whileOver:)``
/// gives up the least recently used record while the total exceeds a budget.
struct ExecutorPromptCacheLRUTier<Record: ExecutorPromptCacheTierRecord> {

    /// The records and their byte total.
    private var tier = ExecutorPromptCacheTier<Record>()

    /// The sessions that hold a record, least recently used first.
    private var usageOrder: [ExecutorPromptCacheKey] = []

    /// The sum of ``ExecutorPromptCacheTierRecord/byteCount`` over the records.
    var byteCount: Int { tier.byteCount }

    /// The sessions that hold a record.
    var keys: Dictionary<ExecutorPromptCacheKey, Record>.Keys { tier.keys }

    /// How many sessions hold a record.
    var count: Int { tier.count }

    /// The record of `key`, or nil when the tier holds none.
    ///
    /// - Parameter key: The session of the record.
    subscript(key: ExecutorPromptCacheKey) -> Record? { tier[key] }

    /// Keeps `record` under `key` as the most recently used record, in the place of the record
    /// that `key` held.
    ///
    /// - Parameters:
    ///   - record: The record.
    ///   - key: The session of the record.
    mutating func insert(_ record: Record, for key: ExecutorPromptCacheKey) {
        remove(key)
        tier.insert(record, for: key)
        usageOrder.append(key)
    }

    /// Removes the record of `key`, its bytes and its place in the usage order.
    ///
    /// - Parameter key: The session of the record.
    /// - Returns: The record, or nil when the tier holds none for `key`.
    @discardableResult
    mutating func remove(_ key: ExecutorPromptCacheKey) -> Record? {
        guard let record = tier.remove(key) else { return nil }
        usageOrder.removeAll { $0 == key }
        return record
    }

    /// Removes the least recently used record when the total exceeds `budget`.
    ///
    /// A caller that applies a budget calls this in a loop until it gives nil.
    ///
    /// - Parameter budget: The most bytes the tier may hold.
    /// - Returns: The removed record and its session, or nil when the total is at or below
    ///   `budget` or the tier holds nothing.
    mutating func removeLeastRecentlyUsed(
        whileOver budget: Int
    ) -> (key: ExecutorPromptCacheKey, record: Record)? {
        guard byteCount > budget, let oldest = usageOrder.first, let record = remove(oldest)
        else { return nil }
        return (oldest, record)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
