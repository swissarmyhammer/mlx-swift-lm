// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import os
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
/// No worker runs. Each remote cache reports a fixed status over a ``RecordingStatusTransport``,
/// which answers every op with that status.
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
        RemoteCacheFixture(workerByteCount: workerByteCount).entry()
    }

    /// A ``RemoteKVCache`` that reports ``workerByteCount`` on its worker and holds
    /// ``PromptCacheSpoolFixtureShape/tokens``.
    ///
    /// - Parameter canLoseRewind: Whether the cache reports a layer that can lose its rewind,
    ///   which makes a checkpoint copy it.
    /// - Returns: The cache.
    private static func remoteCache(canLoseRewind: Bool) -> RemoteKVCache {
        RemoteCacheFixture(workerByteCount: workerByteCount).cache(canLoseRewind: canLoseRewind)
    }
}

/// The store releases the least recently used remote entries on the host when their worker
/// bytes pass ``ExecutorPromptCacheStore/workerBudgetBytes``.
///
/// Every checked-in remote entry can go, the entry of a live session and the entry just
/// checked in among them: in a bench run every session is live. A release frees the caches on
/// the worker, and the next turn of that session starts cold. No worker runs: the caches of
/// each test answer over one ``RecordingStatusTransport``, which records each release.
@Suite("The store bounds the worker bytes of its remote entries")
struct ExecutorPromptCacheWorkerBudgetTests: PromptCacheKeyFixtures, PromptCacheSpoolFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache-worker-budget"

    /// The bytes each remote cache reports on its worker: one gibibyte.
    private static let workerByteCount = 1 << 30

    /// The number of live sessions that check in one remote entry each.
    private static let sessionCount = 3

    /// The worker budget that holds the remote entries of all but one of ``sessionCount``
    /// sessions.
    private static let workerBudgetBytes = (sessionCount - 1) * workerByteCount

    /// The memory budget of each store: one mebibyte, which holds one local entry.
    private static let memoryBudgetBytes = 1 << 20

    @Test("a remote entry counts the bytes of its caches on the worker")
    func aRemoteEntryCountsItsWorkerBytes() {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)

        #expect(fixture.entry().workerByteCount == Self.workerByteCount)
    }

    @Test("a remote entry counts the worker bytes of the copies of its checkpoint")
    func aRemoteEntryCountsTheWorkerBytesOfItsCheckpoint() throws {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let caches = [fixture.cache(canLoseRewind: true)]
        let checkpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: Self.tokens, state: nil))
        let entry = ExecutorPromptCacheEntry(
            caches: caches, tokens: Self.tokens, checkpoint: checkpoint)

        #expect(entry.workerByteCount == 2 * Self.workerByteCount)
    }

    @Test("a local entry counts no worker bytes")
    func aLocalEntryCountsNoWorkerBytes() {
        #expect(Self.entry().workerByteCount == 0)
    }

    @Test("the oldest of the live sessions over the worker budget is released on the host")
    func theOldestLiveSessionIsReleasedOnTheHost() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store(workerBudget: Self.workerBudgetBytes)

        let cacheIDs = await Self.checkInSessions(into: store, from: fixture)

        #expect(fixture.releasedIDs == [cacheIDs[0]])
    }

    @Test("the check-out of the released oldest session is cold")
    func theCheckOutOfTheReleasedSessionIsCold() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store(workerBudget: Self.workerBudgetBytes)
        _ = await Self.checkInSessions(into: store, from: fixture)

        #expect(await store.checkOut(Self.sessionKey(0)) == .none)
    }

    @Test("the check-outs of the newer sessions that fit the worker budget are .memory")
    func theCheckOutsOfTheNewerSessionsAreMemory() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store(workerBudget: Self.workerBudgetBytes)
        _ = await Self.checkInSessions(into: store, from: fixture)

        var inMemory: [Bool] = []
        for index in 1 ..< Self.sessionCount {
            inMemory.append(await store.checkOut(Self.sessionKey(index)).entry != nil)
        }

        #expect(inMemory == Array(repeating: true, count: Self.sessionCount - 1))
    }

    @Test("a single remote entry over the worker budget by itself is released at check-in")
    func aSingleRemoteEntryOverTheWorkerBudgetIsReleased() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store(workerBudget: Self.workerByteCount - 1)

        let cacheID = await Self.checkIn(Self.sessionKey(0), into: store, from: fixture)

        #expect(fixture.releasedIDs == [cacheID])
    }

    @Test("a smaller worker budget releases the remote entries over it at once")
    func aSmallerWorkerBudgetReleasesAtOnce() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store(workerBudget: Self.sessionCount * Self.workerByteCount)
        let cacheIDs = await Self.checkInSessions(into: store, from: fixture)

        await store.configure(workerBudgetBytes: 0)

        #expect(Set(fixture.releasedIDs) == Set(cacheIDs))
    }

    @Test("a zero worker budget keeps a local entry in memory")
    func aZeroWorkerBudgetKeepsALocalEntry() async {
        let store = await Self.store(workerBudget: 0)
        let local = Self.entry()

        await store.checkIn(Self.sessionKey(0), local)

        #expect(await store.peek(Self.sessionKey(0)) === local)
    }

    @Test("a store that no host configures takes the default worker budget")
    func aStoreTakesTheDefaultWorkerBudget() async {
        let store = ExecutorPromptCacheStore(directory: Self.temporaryDirectory())

        #expect(
            await store.workerBudgetBytes == ExecutorPromptCacheStore.defaultWorkerBudgetBytes)
    }

    @Test("the worker release line names the session and its worker bytes")
    func theWorkerReleaseLineNamesTheSessionAndItsWorkerBytes() {
        #expect(
            ExecutorPromptCacheReport.workerReleaseLine(
                key: Self.sessionKey(0), workerByteCount: Self.workerByteCount)
                == "prompt cache release model=test/prompt-cache-worker-budget session=session-0 "
                + "workerBytes=1073741824")
    }

    // MARK: - Fixtures

    /// The key of the live session `index`.
    ///
    /// - Parameter index: The place of the session in the order of the check-ins.
    /// - Returns: The key.
    private static func sessionKey(_ index: Int) -> ExecutorPromptCacheKey {
        key("session-\(index)")
    }

    /// A store with ``memoryBudgetBytes`` in memory and `workerBudget` on the worker. Nothing
    /// in this suite spills, thus the store needs no spool folder of its own.
    ///
    /// - Parameter workerBudget: The worker budget in bytes.
    /// - Returns: The store.
    private static func store(workerBudget: Int) async -> ExecutorPromptCacheStore {
        let store = await store(in: temporaryDirectory(), budget: memoryBudgetBytes)
        await store.configure(workerBudgetBytes: workerBudget)
        return store
    }

    /// Checks in one remote entry for each of ``sessionCount`` sessions, oldest first.
    ///
    /// - Parameters:
    ///   - store: The store.
    ///   - fixture: The maker of the remote caches.
    /// - Returns: The handle of the cache of each session, in the order of the check-ins.
    private static func checkInSessions(
        into store: ExecutorPromptCacheStore, from fixture: RemoteCacheFixture
    ) async -> [RemoteCacheID] {
        var cacheIDs: [RemoteCacheID] = []
        for index in 0 ..< sessionCount {
            cacheIDs.append(await checkIn(sessionKey(index), into: store, from: fixture))
        }
        return cacheIDs
    }

    /// Checks in a new remote entry under `key`, and keeps no reference to it: when the store
    /// drops the entry, its cache releases its handle on the worker.
    ///
    /// - Parameters:
    ///   - key: The session of the entry.
    ///   - store: The store.
    ///   - fixture: The maker of the remote cache.
    /// - Returns: The handle of the cache of the entry.
    private static func checkIn(
        _ key: ExecutorPromptCacheKey, into store: ExecutorPromptCacheStore,
        from fixture: RemoteCacheFixture
    ) async -> RemoteCacheID {
        let cache = fixture.cache(canLoseRewind: false)
        let cacheID = cache.id
        await store.checkIn(key, ExecutorPromptCacheEntry(caches: [cache], tokens: tokens))
        return cacheID
    }
}

/// Makes remote caches that report one status over one ``RecordingStatusTransport``, and reads
/// the handles the caches released.
private struct RemoteCacheFixture {

    /// The transport of every cache.
    let transport: RecordingStatusTransport

    /// The stream of requests of every cache, which gives each one its own handle.
    let session: RemoteModelSession

    /// The bytes each cache reports on its worker.
    let workerByteCount: Int

    /// Makes a fixture whose caches report `workerByteCount`.
    ///
    /// - Parameter workerByteCount: The bytes each cache reports on its worker.
    init(workerByteCount: Int) {
        self.workerByteCount = workerByteCount
        self.transport = RecordingStatusTransport(
            status: Self.status(workerByteCount: workerByteCount, canLoseRewind: false))
        self.session = RemoteModelSession(transport: transport)
    }

    /// The handles of the caches that were released on the worker, in order.
    var releasedIDs: [RemoteCacheID] { transport.releasedIDs }

    /// A ``RemoteKVCache`` that reports ``workerByteCount`` on its worker and holds
    /// ``PromptCacheSpoolFixtureShape/tokens``.
    ///
    /// - Parameter canLoseRewind: Whether the cache reports a layer that can lose its rewind,
    ///   which makes a checkpoint copy it.
    /// - Returns: The cache.
    func cache(canLoseRewind: Bool) -> RemoteKVCache {
        RemoteKVCache(
            id: session.makeCacheID(), session: session,
            status: Self.status(workerByteCount: workerByteCount, canLoseRewind: canLoseRewind))
    }

    /// An entry whose one cache is a ``cache(canLoseRewind:)`` that can always rewind.
    ///
    /// - Returns: The entry.
    func entry() -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(
            caches: [cache(canLoseRewind: false)], tokens: PromptCacheSpoolFixtureShape.tokens)
    }

    /// The status of a cache that holds ``PromptCacheSpoolFixtureShape/tokens``.
    ///
    /// - Parameters:
    ///   - workerByteCount: The bytes the cache reports on its worker.
    ///   - canLoseRewind: Whether a layer of the cache can lose its rewind.
    /// - Returns: The status.
    private static func status(workerByteCount: Int, canLoseRewind: Bool) -> RemoteCacheStatus {
        let tokenCount = PromptCacheSpoolFixtureShape.tokens.count
        return RemoteCacheStatus(
            offset: tokenCount, maxSize: nil, evictedTokenCount: 0, isTrimmable: true,
            maxTrimCount: tokenCount, rewindLimit: .always, canLoseRewind: canLoseRewind,
            residentByteCount: workerByteCount)
    }
}

/// A transport with no worker behind it, which answers every op with the status of one
/// cache and records the handle of each release. A copy then reports the status of its
/// source, and a release ignores its answer.
private final class RecordingStatusTransport: RemoteModelTransport {

    /// The status of every answer.
    let status: RemoteCacheStatus

    /// The handles of the released caches, in order. A cache releases its handle when it goes
    /// away, on the task that drops it.
    private let released = OSAllocatedUnfairLock<[RemoteCacheID]>(initialState: [])

    /// Makes a transport that answers with `status`.
    ///
    /// - Parameter status: The status of every answer.
    init(status: RemoteCacheStatus) {
        self.status = status
    }

    /// The handles of the released caches, in order.
    var releasedIDs: [RemoteCacheID] { released.withLock { $0 } }

    /// Records `request` when it is a release, and answers it with ``status``.
    ///
    /// - Parameter request: The request.
    /// - Returns: ``RemoteModelReply/cache(_:)`` of ``status``.
    func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        if case .release(let id) = request.op {
            released.withLock { $0.append(id) }
        }
        return .cache(status)
    }
}

#endif  // FoundationModelsIntegration
