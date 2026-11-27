// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// The store keeps the entry of a clustered model, whose caches live on the workers.
///
/// A ``RemoteKVCache`` holds no array in this process, and a prompt cache file cannot hold it.
/// The memory budget of the store thus does not apply to it, and the store never sends it to
/// disk: an eviction for the budget would drop it, and the next turn of its session would
/// start cold.
///
/// No worker runs. Each remote cache reports a fixed status over a transport that answers
/// every op with that status.
@Suite("The store keeps the remote cache of a clustered model in memory")
struct ExecutorPromptCacheRemoteEntryTests: PromptCacheKeyFixtures, PromptCacheSpoolFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache-remote"

    /// The bytes each remote cache reports on its worker: one gibibyte, far over
    /// ``memoryBudgetBytes``.
    private static let workerByteCount = 1 << 30

    /// The memory budget of the stores that keep one local entry: one mebibyte, far under
    /// ``workerByteCount``.
    private static let memoryBudgetBytes = 1 << 20

    @Test("a remote cache counts no bytes of this process")
    func aRemoteCacheCountsNoBytesOfThisProcess() {
        #expect(Self.remoteEntry().byteCount == 0)
    }

    @Test("a checkpoint of a remote cache counts no bytes of this process")
    func aCheckpointOfARemoteCacheCountsNoBytesOfThisProcess() throws {
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: [Self.remoteCache(canLoseRewind: true)], tokens: Self.tokens, state: nil))

        #expect(checkpoint.byteCount == 0)
    }

    @Test(
        "a remote entry over the memory budget stays at check-in, and the next check-out is .memory"
    )
    func aRemoteEntryOverTheBudgetStaysAtCheckIn() async {
        let store = await Self.store(
            in: Self.temporaryDirectory(), budget: Self.memoryBudgetBytes)
        let remote = Self.remoteEntry()

        await store.checkIn(key("remote"), remote)

        #expect(await store.checkOut(key("remote")) == .memory(remote))
    }

    @Test("a zero memory budget never spills a remote entry: it stays in memory")
    func aZeroBudgetNeverSpillsARemoteEntry() async {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 0)
        let remote = Self.remoteEntry()

        await store.checkIn(key("remote"), remote)
        await store.configure(memoryBudgetBytes: 0)
        await store.waitForSpills()

        #expect(await store.peek(key("remote")) === remote)
    }

    @Test("an eviction for the budget passes over the least recently used remote entry")
    func anEvictionPassesOverTheRemoteEntry() async {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (store, remote) = await Self.storeOverBudgetBehindARemoteEntry(in: directory)

        #expect(await store.peek(key("remote")) === remote)
    }

    @Test("an eviction for the budget still spills the local entry behind a remote entry")
    func anEvictionStillSpillsTheLocalEntryBehindARemoteEntry() async {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (store, _) = await Self.storeOverBudgetBehindARemoteEntry(in: directory)

        #expect(await store.checkOut(key("first-local")).spilledHandle != nil)
    }

    // MARK: - Fixtures

    /// A store whose budget holds one local entry, after three check-ins: a remote entry, then
    /// two local entries. The second local entry takes the total over the budget, and the
    /// writes of the eviction have ended.
    ///
    /// - Parameter directory: The spool folder of the store.
    /// - Returns: The store, and the remote entry it took first.
    private static func storeOverBudgetBehindARemoteEntry(
        in directory: URL
    ) async -> (store: ExecutorPromptCacheStore, remote: ExecutorPromptCacheEntry) {
        let store = await store(in: directory, budget: entry().byteCount)
        let remote = remoteEntry()
        await store.checkIn(key("remote"), remote)
        await store.checkIn(key("first-local"), entry())
        await store.checkIn(key("second-local"), entry())
        await store.waitForSpills()
        return (store, remote)
    }

    /// An entry whose one cache is a ``remoteCache(canLoseRewind:)`` that can always rewind.
    ///
    /// - Returns: The entry.
    private static func remoteEntry() -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(caches: [remoteCache(canLoseRewind: false)], tokens: tokens)
    }

    /// A ``RemoteKVCache`` that reports ``workerByteCount`` on its worker and holds
    /// ``PromptCacheSpoolFixtureShape/tokens``.
    ///
    /// - Parameter canLoseRewind: Whether the cache reports a layer that can lose its rewind,
    ///   which makes a checkpoint copy it.
    /// - Returns: The cache.
    private static func remoteCache(canLoseRewind: Bool) -> RemoteKVCache {
        let status = RemoteCacheStatus(
            offset: tokens.count, maxSize: nil, evictedTokenCount: 0, isTrimmable: true,
            maxTrimCount: tokens.count, rewindLimit: .always, canLoseRewind: canLoseRewind,
            residentByteCount: workerByteCount)
        let session = RemoteModelSession(transport: AnsweringStatusTransport(status: status))
        return RemoteKVCache(id: session.makeCacheID(), session: session, status: status)
    }
}

/// A transport with no worker behind it, which answers every op with the status of one
/// cache. A copy then reports the status of its source, and a release ignores its answer.
private struct AnsweringStatusTransport: RemoteModelTransport {

    /// The status of every answer.
    let status: RemoteCacheStatus

    /// Answers `request` with ``status``.
    ///
    /// - Parameter request: The request, which the transport ignores.
    /// - Returns: ``RemoteModelReply/cache(_:)`` of ``status``.
    func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        .cache(status)
    }
}

#endif  // FoundationModelsIntegration
