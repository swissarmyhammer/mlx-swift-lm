// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMTestSupport
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

    @Test("a zero worker budget releases a remote entry that reports zero worker bytes")
    func aZeroWorkerBudgetReleasesAZeroByteRemoteEntry() async {
        let fixture = RemoteCacheFixture(workerByteCount: 0)
        let store = await Self.store(workerBudget: 0)

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

/// A copy of a session whose entry holds a remote cache
/// (``ExecutorPromptCacheStore/copy(modelID:from:to:)``) goes in like a check-in of a remote
/// entry, and a copy that the worker fails gives ``PromptCacheCopyResult/Refusal/copyFailed``.
///
/// No worker runs: the caches of each test answer over one ``RecordingStatusTransport``, which
/// copies, refuses each copy, or loses a member at each copy.
@Suite("A copy of a remote entry goes in like a check-in, and a failed copy keeps nothing")
struct ExecutorPromptCacheRemoteCopyTests: PromptCacheKeyFixtures, PromptCacheSpoolFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache-remote-copy"

    /// The bytes each remote cache reports on its worker: one gibibyte.
    private static let workerByteCount = 1 << 30

    /// The worker budget that holds the remote entries of two sessions: the source and its
    /// copy.
    private static let workerBudgetBytes = 2 * workerByteCount

    /// The memory budget of each store: one mebibyte.
    private static let memoryBudgetBytes = 1 << 20

    /// The member a copy loses.
    private static let lostMember = "m5u2.local"

    /// The session that each test copies.
    private static let source = "source"

    /// The session that each copy makes.
    private static let destination = "destination"

    /// The session that checks in after the source, and that no test copies.
    private static let other = "other"

    @Test("a copy over the worker budget releases the least recently used remote entry")
    func aCopyOverTheWorkerBudgetReleasesTheLeastRecentlyUsedRemoteEntry() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store()
        await store.checkIn(key(Self.other), fixture.entry())
        let otherID = await Self.cacheID(of: Self.other, in: store)
        await store.checkIn(key(Self.source), fixture.entry())

        _ = await Self.copySourceToDestination(in: store)

        #expect(fixture.releasedIDs == [otherID])
    }

    @Test("a copy over the worker budget keeps the source, which it touched")
    func aCopyOverTheWorkerBudgetKeepsTheSource() async {
        let fixture = RemoteCacheFixture(workerByteCount: Self.workerByteCount)
        let store = await Self.store()
        let sourceEntry = fixture.entry()
        await store.checkIn(key(Self.source), sourceEntry)
        await store.checkIn(key(Self.other), fixture.entry())

        _ = await Self.copySourceToDestination(in: store)

        #expect(await store.peek(key(Self.source)) === sourceEntry)
    }

    @Test("a copy that the worker refuses gives copyFailed")
    func aRefusedCopyGivesCopyFailed() async {
        let store = await Self.storeWithSource(copyAnswer: .refused).store

        #expect(await Self.copySourceToDestination(in: store) == .refused(.copyFailed))
    }

    @Test("a copy that the worker refuses keeps no entry for the destination")
    func aRefusedCopyKeepsNoDestinationEntry() async {
        let store = await Self.storeWithSource(copyAnswer: .refused).store

        _ = await Self.copySourceToDestination(in: store)

        #expect(await store.checkOut(key(Self.destination)) == .none)
    }

    @Test("a copy that the worker refuses keeps the source entry")
    func aRefusedCopyKeepsTheSourceEntry() async {
        let store = await Self.storeWithSource(copyAnswer: .refused).store
        let sourceEntry = await store.peek(key(Self.source))

        _ = await Self.copySourceToDestination(in: store)

        #expect(await store.peek(key(Self.source)) === sourceEntry)
    }

    @Test("a copy that the worker refuses leaves the model usable")
    func aRefusedCopyLeavesTheModelUsable() async {
        let (store, fixture) = await Self.storeWithSource(copyAnswer: .refused)

        _ = await Self.copySourceToDestination(in: store)

        #expect(throws: Never.self) { try fixture.session.beginRequest() }
    }

    @Test("a copy that the worker refuses releases no handle on the worker")
    func aRefusedCopyReleasesNoHandle() async {
        let (store, fixture) = await Self.storeWithSource(copyAnswer: .refused)

        _ = await Self.copySourceToDestination(in: store)

        #expect(fixture.releasedIDs.isEmpty)
    }

    @Test("a copy that loses a member gives copyFailed")
    func aCopyThatLosesAMemberGivesCopyFailed() async {
        let store = await Self.storeWithSource(copyAnswer: .lost(Self.lostMember)).store

        #expect(await Self.copySourceToDestination(in: store) == .refused(.copyFailed))
    }

    @Test("the next request after a copy that lost a member throws memberLost")
    func theNextRequestAfterALostMemberThrowsMemberLost() async {
        let (store, fixture) = await Self.storeWithSource(copyAnswer: .lost(Self.lostMember))

        _ = await Self.copySourceToDestination(in: store)

        #expect(throws: ClusterModelError.memberLost(Self.lostMember)) {
            try fixture.session.beginRequest()
        }
    }

    // MARK: - Fixtures

    /// A store with ``memoryBudgetBytes`` in memory and ``workerBudgetBytes`` on the worker.
    /// Nothing in this suite spills, thus the store needs no spool folder of its own.
    ///
    /// - Returns: The store.
    private static func store() async -> ExecutorPromptCacheStore {
        let store = await store(in: temporaryDirectory(), budget: memoryBudgetBytes)
        await store.configure(workerBudgetBytes: workerBudgetBytes)
        return store
    }

    /// A store that holds one remote entry of ``source``, whose transport answers each copy
    /// with `copyAnswer`.
    ///
    /// - Parameter copyAnswer: How the transport answers a copy.
    /// - Returns: The store, and the fixture of the remote caches.
    private static func storeWithSource(
        copyAnswer: RecordingStatusTransport.CopyAnswer
    ) async -> (store: ExecutorPromptCacheStore, fixture: RemoteCacheFixture) {
        let fixture = RemoteCacheFixture(workerByteCount: workerByteCount, copyAnswer: copyAnswer)
        let store = await store()
        await store.checkIn(key(source), fixture.entry())
        return (store, fixture)
    }

    /// Copies ``source`` to ``destination`` of ``modelID`` in `store`.
    ///
    /// - Parameter store: The store.
    /// - Returns: What the copy did.
    private static func copySourceToDestination(
        in store: ExecutorPromptCacheStore
    ) async -> PromptCacheCopyResult {
        await store.copy(modelID: .init(modelID), from: .init(source), to: .init(destination))
    }

    /// The handle of the remote cache of the entry of `sessionID` in `store`.
    ///
    /// - Parameters:
    ///   - sessionID: The session.
    ///   - store: The store.
    /// - Returns: The handle, or nil when the entry holds no remote cache.
    private static func cacheID(
        of sessionID: String, in store: ExecutorPromptCacheStore
    ) async -> RemoteCacheID? {
        (await store.peek(key(sessionID))?.caches.first as? RemoteKVCache)?.id
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
    /// - Parameters:
    ///   - workerByteCount: The bytes each cache reports on its worker.
    ///   - copyAnswer: How the transport answers a copy.
    init(
        workerByteCount: Int,
        copyAnswer: RecordingStatusTransport.CopyAnswer = .copied
    ) {
        self.workerByteCount = workerByteCount
        self.transport = RecordingStatusTransport(
            status: Self.status(workerByteCount: workerByteCount, canLoseRewind: false),
            copyAnswer: copyAnswer)
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
/// source, unless the transport refuses it or loses a member at it, and a release ignores its
/// answer.
private final class RecordingStatusTransport: RemoteModelTransport {

    /// How the transport answers a copy.
    enum CopyAnswer {

        /// With the status of its source, as a worker that copied the caches.
        case copied

        /// With ``RemoteModelReply/failed(_:)``, as a worker that ran the copy and failed it.
        case refused

        /// With an ``InProcessMemberLoss`` that lost the member, as a cluster that lost it.
        case lost(String)
    }

    /// The reason of the reply to a refused copy.
    static let refusalReason = "the worker refused the copy"

    /// The status of every answer.
    let status: RemoteCacheStatus

    /// How the transport answers a copy.
    let copyAnswer: CopyAnswer

    /// The handles of the released caches, in order. A cache releases its handle when it goes
    /// away, on the task that drops it.
    private let released = OSAllocatedUnfairLock<[RemoteCacheID]>(initialState: [])

    /// Makes a transport that answers with `status`.
    ///
    /// - Parameters:
    ///   - status: The status of every answer.
    ///   - copyAnswer: How the transport answers a copy.
    init(status: RemoteCacheStatus, copyAnswer: CopyAnswer = .copied) {
        self.status = status
        self.copyAnswer = copyAnswer
    }

    /// The handles of the released caches, in order.
    var releasedIDs: [RemoteCacheID] { released.withLock { $0 } }

    /// Records `request` when it is a release, and answers it with ``status``, a copy as
    /// ``copyAnswer`` says, and the begin or the end of a request with
    /// ``RemoteModelReply/done``.
    ///
    /// - Parameter request: The request.
    /// - Returns: ``RemoteModelReply/cache(_:)`` of ``status``,
    ///   ``RemoteModelReply/failed(_:)`` for a refused copy, or ``RemoteModelReply/done``.
    /// - Throws: The ``InProcessMemberLoss`` of a copy that loses a member.
    func send(_ request: RemoteModelRequest) throws -> RemoteModelReply {
        if case .release(let id) = request.op {
            released.withLock { $0.append(id) }
        }
        if case .copy = request.op {
            return try answerCopy()
        }
        if request.op == .begin || request.op == .end {
            return .done
        }
        return .cache(status)
    }

    /// The answer to a copy, as ``copyAnswer`` says.
    ///
    /// - Returns: ``RemoteModelReply/cache(_:)`` of ``status``, or
    ///   ``RemoteModelReply/failed(_:)`` for a refused copy.
    /// - Throws: The ``InProcessMemberLoss`` of a copy that loses a member.
    private func answerCopy() throws -> RemoteModelReply {
        switch copyAnswer {
        case .copied: return .cache(status)
        case .refused: return .failed(Self.refusalReason)
        case .lost(let member): throw InProcessMemberLoss(lostMember: member)
        }
    }
}

#endif  // FoundationModelsIntegration
