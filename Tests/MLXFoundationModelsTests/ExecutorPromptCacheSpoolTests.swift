// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import Synchronization
import Testing

@testable import MLXFoundationModels

/// Tests for the spool of ``ExecutorPromptCacheStore``: an entry that leaves memory for the
/// budget goes to one file through one serial writer, and a later check-out finds it.
///
/// No weights are needed. Each entry carries one `KVCacheSimple` with fixed values. Each test
/// has its own store in its own temporary folder. A test that must see a write in flight gives
/// the store a ``HeldWriter``, which holds each write until the test releases it.
@Suite("An evicted prompt cache goes to disk through one serial writer")
struct ExecutorPromptCacheSpoolTests {

    // MARK: - Fixture values

    /// The model that each key names. It holds a `/`, as a real model ID does.
    private static let modelID = "test-org/prompt-cache-spool"

    /// The ledger of each entry.
    private static let tokens = [1, 42, 7]

    /// The number of key/value heads of the cache.
    private static let headCount = 2

    /// The head dimension of the cache.
    private static let headDimension = 4

    /// The distance between the first values of two fixture arrays.
    private static let valueStride: Float = 100

    /// How many entries the serial-writer test spills.
    private static let spillCount = 4

    /// How long each write of the serial-writer test takes. A second writer that ran beside
    /// the first would start in this window and raise the count of concurrent writes.
    private static let overlapWindow: TimeInterval = 0.02

    /// The bytes that the spill-line test reports.
    private static let reportedByteCount = 4_096

    /// The write duration that the spill-line test reports.
    private static let reportedWriteDuration: Duration = .milliseconds(1_250)

    /// A second model, for the tests that remove the state of one model or one session.
    private static let otherModelID = "test-org/prompt-cache-spool-other"

    /// How many files the disk-budget test keeps, and how many entries the memory budget of the
    /// eviction test keeps.
    private static let keptFileCount = 2

    /// The free space that the default-disk-budget test gives.
    private static let availableCapacity = 400_000

    /// The process of the folder that the clean-up test marks stale.
    private static let stalePID: pid_t = 101

    /// The process of the folder that the clean-up test marks live but not signalable.
    private static let unsignalablePID: pid_t = 102

    /// The process of the folder that the clean-up test marks live.
    private static let livePID: pid_t = 103

    /// What the probe of the clean-up test gives for a live process: `kill` returns zero.
    private static let liveProbeResult: ExecutorPromptCacheStore.ProcessProbeResult = (
        killResult: 0, errorNumber: 0
    )

    /// What the probe of the clean-up test gives for each process that it marks. A process that
    /// is not in the table gets ``liveProbeResult``.
    private static let probeResults: [pid_t: ExecutorPromptCacheStore.ProcessProbeResult] = [
        stalePID: (killResult: -1, errorNumber: ESRCH),
        unsignalablePID: (killResult: -1, errorNumber: EPERM),
        livePID: liveProbeResult,
    ]

    /// The fixture arrays, in the order of their first values.
    private enum FixtureArray: Int {
        case keys
        case values
        case nextToken
    }

    // MARK: - Fixture builders

    /// The key of `sessionID` under ``modelID``.
    private static func key(_ sessionID: String) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
    }

    /// Makes the keys or the values of a block of tokens, with consecutive values.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens of the block.
    ///   - array: The fixture array. It sets the first value.
    /// - Returns: An array of shape `(1, heads, tokens, headDimension)`, as `float16`.
    private static func block(tokenCount: Int, array: FixtureArray) -> MLXArray {
        let shape = [1, headCount, tokenCount, headDimension]
        let count = shape.reduce(1, *)
        let start = Float(array.rawValue) * valueStride
        return MLXArray((0 ..< count).map { start + Float($0) }).reshaped(shape).asType(.float16)
    }

    /// Makes an entry with one `KVCacheSimple` that holds ``tokens``.
    ///
    /// - Returns: The entry.
    private static func entry() -> ExecutorPromptCacheEntry {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: block(tokenCount: tokens.count, array: .keys),
            values: block(tokenCount: tokens.count, array: .values))
        return ExecutorPromptCacheEntry(caches: [cache], tokens: tokens)
    }

    /// Makes the name of an empty temporary folder for one test. The folder is not made: the
    /// store makes it at its first write.
    ///
    /// - Returns: The URL of the folder.
    private static func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ExecutorPromptCacheSpoolTests-\(UUID().uuidString)")
    }

    /// Makes a store in `directory` with the budget `budget`.
    ///
    /// - Parameters:
    ///   - directory: The spool folder of the store.
    ///   - budget: The memory budget in bytes. Zero spills each check-in at once.
    ///   - writer: The writer of the spill files.
    /// - Returns: The store.
    private static func store(
        in directory: URL, budget: Int = 0,
        writer: @escaping ExecutorPromptCacheFileWriter = ExecutorPromptCacheStore.writeSpillFile
    ) async -> ExecutorPromptCacheStore {
        let store = ExecutorPromptCacheStore(directory: directory, writer: writer)
        await store.configure(memoryBudgetBytes: budget)
        return store
    }

    /// The names of the spill files in `directory`, sorted. A folder that does not exist
    /// holds none.
    ///
    /// - Parameter directory: The spool folder.
    /// - Returns: The file names.
    private static func fileNames(in directory: URL) throws -> [String] {
        guard FileManager.default.fileExists(atPath: directory.path(percentEncoded: false))
        else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    /// Makes the folder `name` in `root`, and makes `root` when it is not there.
    ///
    /// - Parameters:
    ///   - name: The name of the folder.
    ///   - root: The folder that holds it.
    /// - Returns: `name`.
    private static func makeFolder(_ name: String, in root: URL) throws -> String {
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(name, isDirectory: true),
            withIntermediateDirectories: true)
        return name
    }

    /// Records an issue unless `store` holds nothing for `key` and no file stays in `directory`:
    /// no entry in memory, no write that has not ended, no disk record and no byte on any tier.
    ///
    /// - Parameters:
    ///   - key: The key that the test removed.
    ///   - store: The store.
    ///   - directory: The spool folder of the store.
    private static func expectNothingStored(
        for key: ExecutorPromptCacheKey, in store: ExecutorPromptCacheStore, directory: URL
    ) async throws {
        #expect(await store.retainedByteCount == 0)
        #expect(await store.spillingByteCount == 0)
        #expect(await store.diskByteCount == 0)
        #expect(try fileNames(in: directory).isEmpty)
        #expect(await store.checkOut(key) == .none)
    }

    /// Records an issue unless two lists of arrays have equal shapes, types and values.
    ///
    /// - Parameters:
    ///   - actual: The arrays to check.
    ///   - expected: The arrays they must equal.
    private static func expectEqual(_ actual: [MLXArray], _ expected: [MLXArray]) {
        #expect(actual.count == expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            #expect(lhs.shape == rhs.shape)
            #expect(lhs.dtype == rhs.dtype)
            #expect(arrayEqual(lhs, rhs).item(Bool.self))
        }
    }

    // MARK: - Writers a test controls

    /// A writer that holds each write until the test releases it, and keeps a copy of each
    /// file that it writes.
    ///
    /// The store deletes a file that nobody needs when its write ends. The copy lets a test
    /// read the bytes that the writer wrote all the same.
    private final class HeldWriter: Sendable {

        /// The URL of each write, in the order the writes start.
        let startedWrites: AsyncStream<URL>

        /// Receives the URL of each write that starts.
        private let starts: AsyncStream<URL>.Continuation

        /// Holds each write until ``release()``.
        private let releases = DispatchSemaphore(value: 0)

        /// The folder of the copies.
        let copies: URL

        /// Creates a writer that keeps its copies in `copies`.
        ///
        /// - Parameter copies: The folder of the copies. The writer makes it.
        init(copies: URL) throws {
            (startedWrites, starts) = AsyncStream.makeStream(of: URL.self)
            self.copies = copies
            try FileManager.default.createDirectory(at: copies, withIntermediateDirectories: true)
        }

        /// Tells the test that a write started, waits for a release, and writes the file and
        /// its copy.
        ///
        /// - Parameters:
        ///   - input: The prepared entry.
        ///   - url: The URL of the file.
        func write(_ input: PromptCacheSaveInput, to url: URL) throws {
            starts.yield(url)
            releases.wait()
            try ExecutorPromptCacheStore.writeSpillFile(input, to: url)
            try FileManager.default.copyItem(at: url, to: copy(of: url))
        }

        /// Lets one held write run.
        func release() {
            releases.signal()
        }

        /// The URL of the copy of `url`.
        ///
        /// - Parameter url: The URL of a file that the writer wrote.
        /// - Returns: The URL of its copy.
        func copy(of url: URL) -> URL {
            copies.appendingPathComponent(url.lastPathComponent)
        }
    }

    /// A writer that counts the writes that run at the same time.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class CountingWriter: Sendable {

        /// The writes that run now, and the most that ran at one time.
        private let counts = Mutex((running: 0, most: 0))

        /// The most writes that ran at one time.
        var mostConcurrentWrites: Int {
            counts.withLock { $0.most }
        }

        /// Counts the write, holds it for ``overlapWindow``, and writes the file.
        ///
        /// - Parameters:
        ///   - input: The prepared entry.
        ///   - url: The URL of the file.
        func write(_ input: PromptCacheSaveInput, to url: URL) throws {
            counts.withLock {
                $0.running += 1
                $0.most = max($0.most, $0.running)
            }
            defer { counts.withLock { $0.running -= 1 } }
            Thread.sleep(forTimeInterval: ExecutorPromptCacheSpoolTests.overlapWindow)
            try ExecutorPromptCacheStore.writeSpillFile(input, to: url)
        }
    }

    // MARK: - Spill and check-out

    @Test("an entry evicted by the byte budget is on disk, and a check-out hands out its file")
    func anEvictedEntryIsOnDiskAndACheckOutHandsOutItsFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let oldest = Self.entry()
        let store = await Self.store(in: directory, budget: oldest.byteCount)
        let newest = Self.entry()

        await store.checkIn(Self.key("oldest"), oldest)
        await store.checkIn(Self.key("newest"), newest)
        await store.waitForSpills()

        let checkout = await store.checkOut(Self.key("oldest"))
        let handle = try #require(checkout.spilledHandle)
        #expect(handle.key == Self.key("oldest"))
        #expect(try Self.fileNames(in: directory) == [handle.url.lastPathComponent])
        let restored = try ExecutorPromptCacheFile.read(
            from: handle.url, key: handle.key, templates: [KVCacheSimple()])
        #expect(restored.tokens == Self.tokens)
        #expect(await store.checkOut(Self.key("newest")) == .memory(newest))
        #expect(await store.checkOut(Self.key("oldest")) == .none)
    }

    @Test("a check-out during a spill takes the same entry back, and no file stays")
    func aCheckOutDuringASpillTakesTheEntryBack() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        let entry = Self.entry()
        var starts = writer.startedWrites.makeAsyncIterator()

        await store.checkIn(Self.key("a"), entry)
        _ = await starts.next()
        #expect(await store.spillingByteCount == entry.byteCount)
        #expect(await store.retainedByteCount == 0)

        let checkout = await store.checkOut(Self.key("a"))
        #expect(await store.spillingByteCount == 0)
        writer.release()
        await store.waitForSpills()

        #expect(checkout == .memory(entry))
        #expect(await store.checkOut(Self.key("a")) == .none)
        #expect(try Self.fileNames(in: directory).isEmpty)
    }

    @Test("an update after a check-out during a spill does not change the file")
    func anUpdateAfterACheckOutDuringASpillDoesNotChangeTheFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()

        await store.checkIn(Self.key("a"), Self.entry())
        let url = try #require(await starts.next())
        let taken = try #require(await store.checkOut(Self.key("a")).entry)
        let token = Self.block(tokenCount: 1, array: .nextToken)
        _ = taken.caches[0].update(keys: token, values: token)
        writer.release()
        await store.waitForSpills()

        let written = try ExecutorPromptCacheFile.read(
            from: writer.copy(of: url), key: Self.key("a"), templates: [KVCacheSimple()])
        #expect(written.caches[0].offset == Self.tokens.count)
        Self.expectEqual(written.caches[0].state, Self.entry().caches[0].state)
    }

    @Test("a spill that a check-out and a later spill replaced leaves only the later file")
    func aReplacedSpillLeavesOnlyTheLaterFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()

        await store.checkIn(Self.key("a"), Self.entry())
        let firstURL = try #require(await starts.next())
        let taken = try #require(await store.checkOut(Self.key("a")).entry)
        await store.checkIn(Self.key("a"), taken)
        writer.release()
        let secondURL = try #require(await starts.next())
        writer.release()
        await store.waitForSpills()

        #expect(firstURL != secondURL)
        #expect(try Self.fileNames(in: directory) == [secondURL.lastPathComponent])
        #expect(
            await store.checkOut(Self.key("a"))
                == .spilled(ExecutorPromptCacheSpilledHandle(url: secondURL, key: Self.key("a"))))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("at most one write runs at a time")
    func atMostOneWriteRunsAtATime() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = CountingWriter()
        let store = await Self.store(in: directory, writer: writer.write)

        for session in 0 ..< Self.spillCount {
            await store.checkIn(Self.key("session-\(session)"), Self.entry())
        }
        await store.waitForSpills()

        #expect(writer.mostConcurrentWrites == 1)
        #expect(try Self.fileNames(in: directory).count == Self.spillCount)
    }

    @Test("check-outs and check-ins complete while a write runs")
    func checkOutsAndCheckInsCompleteWhileAWriteRuns() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let held = Self.entry()
        let store = await Self.store(in: directory, budget: held.byteCount, writer: writer.write)
        let other = Self.entry()
        var starts = writer.startedWrites.makeAsyncIterator()
        await store.checkIn(Self.key("held"), held)
        await store.checkIn(Self.key("other"), other)
        _ = await starts.next()

        // The write of "held" waits for a release. The actor must answer all the same.
        let checkout = await store.checkOut(Self.key("other"))
        await store.checkIn(Self.key("other"), other)
        let absent = await store.checkOut(Self.key("absent"))
        writer.release()
        await store.waitForSpills()

        #expect(checkout == .memory(other))
        #expect(absent == .none)
        #expect(await store.peek(Self.key("other")) === other)
        #expect(await store.checkOut(Self.key("held")).spilledHandle != nil)
    }

    @Test("a check-in of a key removes the older file of that key")
    func aCheckInRemovesTheOlderFileOfTheKey() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await store.checkIn(Self.key("a"), Self.entry())
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == 1)
        let newer = Self.entry()

        await store.configure(memoryBudgetBytes: newer.byteCount)
        await store.checkIn(Self.key("a"), newer)

        #expect(try Self.fileNames(in: directory).isEmpty)
        #expect(await store.checkOut(Self.key("a")) == .memory(newer))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("an executor pass that finds its cache on disk starts cold and deletes the file")
    func anExecutorPassThatFindsItsCacheOnDiskStartsColdAndDeletesTheFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let modelID = "probe/spilled-prompt-cache-\(UUID().uuidString)"
        let sessionKey = ExecutorPromptCacheKey(modelID: modelID, sessionID: "spilled-session")
        let store = await Self.store(in: directory)
        await store.checkIn(sessionKey, Self.entry())
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == 1)

        try await respondOnce(inside: store, modelID: modelID, sessionID: sessionKey.sessionID)

        #expect(try Self.fileNames(in: directory).isEmpty)
        #expect(await store.checkOut(sessionKey) == .none)
    }

    // MARK: - Disk budget

    @Test(
        "the disk budget deletes the least recently used files, and the disk total stays at or below it"
    )
    func theDiskBudgetDeletesTheLeastRecentlyUsedFiles() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await store.checkIn(Self.key("s1"), Self.entry())
        await store.waitForSpills()
        let fileByteCount = await store.diskByteCount
        #expect(fileByteCount > 0)

        await store.configure(diskBudgetBytes: Self.keptFileCount * fileByteCount)
        await store.checkIn(Self.key("s2"), Self.entry())
        await store.checkIn(Self.key("s3"), Self.entry())
        await store.waitForSpills()

        #expect(await store.diskByteCount == Self.keptFileCount * fileByteCount)
        let budget = await store.diskBudgetBytes
        #expect(await store.diskByteCount <= budget)
        #expect(try Self.fileNames(in: directory).count == Self.keptFileCount)
        #expect(await store.checkOut(Self.key("s1")) == .none)
        #expect(await store.checkOut(Self.key("s2")).spilledHandle != nil)
        #expect(await store.checkOut(Self.key("s3")).spilledHandle != nil)
    }

    @Test("a smaller disk budget deletes files at once")
    func aSmallerDiskBudgetDeletesFilesAtOnce() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await store.checkIn(Self.key("s1"), Self.entry())
        await store.waitForSpills()

        await store.configure(diskBudgetBytes: 0)

        #expect(await store.diskByteCount == 0)
        #expect(try Self.fileNames(in: directory).isEmpty)
        #expect(await store.checkOut(Self.key("s1")) == .none)
    }

    @Test("the default disk budget is one quarter of the free space it receives")
    func theDefaultDiskBudgetIsOneQuarterOfTheFreeSpace() {
        #expect(
            ExecutorPromptCacheStore.defaultDiskBudgetBytes(
                availableCapacity: Self.availableCapacity)
                == 100_000)
        #expect(ExecutorPromptCacheStore.defaultDiskBudgetBytes(availableCapacity: 0) == 0)
        #expect(ExecutorPromptCacheStore.defaultDiskBudgetBytes(availableCapacity: -1) == 0)
    }

    // MARK: - Folders of processes that do not run

    @Test("a process is stale only when kill fails with ESRCH")
    func aProcessIsStaleOnlyWhenKillFailsWithESRCH() {
        #expect(ExecutorPromptCacheStore.isStaleProcess(killResult: -1, errorNumber: ESRCH))
        #expect(!ExecutorPromptCacheStore.isStaleProcess(killResult: -1, errorNumber: EPERM))
        #expect(!ExecutorPromptCacheStore.isStaleProcess(killResult: 0, errorNumber: 0))
    }

    @Test(
        "the clean-up deletes the folder of a stale process and keeps the folders of live processes"
    )
    func theCleanUpDeletesOnlyTheFoldersOfStaleProcesses() throws {
        let root = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let stale = try Self.makeFolder("\(Self.stalePID)-\(UUID().uuidString)", in: root)
        let unsignalable = try Self.makeFolder(
            "\(Self.unsignalablePID)-\(UUID().uuidString)", in: root)
        let live = try Self.makeFolder("\(Self.livePID)-\(UUID().uuidString)", in: root)
        let foreign = try Self.makeFolder("not-a-spool-folder", in: root)

        ExecutorPromptCacheStore.removeStaleSpoolFolders(in: root) { pid in
            Self.probeResults[pid] ?? Self.liveProbeResult
        }

        #expect(try Self.fileNames(in: root) == [live, foreign, unsignalable].sorted())
        #expect(!(try Self.fileNames(in: root).contains(stale)))
    }

    // MARK: - Removal of one key

    @Test("remove of a key in memory leaves nothing for that key")
    func removeOfAKeyInMemoryLeavesNothing() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let entry = Self.entry()
        let store = await Self.store(in: directory, budget: entry.byteCount)
        await store.checkIn(Self.key("a"), entry)

        await store.remove(Self.key("a"))
        await store.waitForSpills()

        try await Self.expectNothingStored(for: Self.key("a"), in: store, directory: directory)
    }

    @Test("remove of a key whose write has not ended leaves nothing after the write ends")
    func removeOfASpillingKeyLeavesNothing() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()
        await store.checkIn(Self.key("a"), Self.entry())
        _ = await starts.next()

        await store.remove(Self.key("a"))
        writer.release()
        await store.waitForSpills()

        try await Self.expectNothingStored(for: Self.key("a"), in: store, directory: directory)
    }

    @Test("remove of a key on disk deletes its file")
    func removeOfAKeyOnDiskDeletesItsFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await store.checkIn(Self.key("a"), Self.entry())
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == 1)

        await store.remove(Self.key("a"))

        try await Self.expectNothingStored(for: Self.key("a"), in: store, directory: directory)
    }

    @Test("remove of an unknown key changes nothing")
    func removeOfAnUnknownKeyChangesNothing() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let kept = Self.entry()
        let store = await Self.store(in: directory, budget: kept.byteCount)
        await store.checkIn(Self.key("kept"), kept)

        await store.remove(Self.key("unknown"))

        #expect(await store.retainedByteCount == kept.byteCount)
        #expect(await store.checkOut(Self.key("kept")) == .memory(kept))
    }

    // MARK: - Removal of one session or one model

    @Test("evict(sessionID:) removes that session for every model and keeps other sessions")
    func evictSessionRemovesTheSessionForEveryModel() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let other = Self.entry()
        let store = await Self.store(in: directory, budget: other.byteCount)
        let firstModel = ExecutorPromptCacheKey(modelID: Self.modelID, sessionID: "shared")
        let secondModel = ExecutorPromptCacheKey(modelID: Self.otherModelID, sessionID: "shared")
        // Each check-in sends the one before it to disk.
        await store.checkIn(firstModel, Self.entry())
        await store.checkIn(secondModel, Self.entry())
        await store.checkIn(Self.key("other"), other)
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == 2)

        await store.evict(sessionID: "shared")

        #expect(try Self.fileNames(in: directory).isEmpty)
        #expect(await store.diskByteCount == 0)
        #expect(await store.checkOut(firstModel) == .none)
        #expect(await store.checkOut(secondModel) == .none)
        #expect(await store.checkOut(Self.key("other")) == .memory(other))
    }

    @Test(
        "evict(modelID:) removes the memory, spilling and disk state of one model and keeps other models"
    )
    func evictModelRemovesEveryStateOfThatModel() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        let evicted = { ExecutorPromptCacheKey(modelID: Self.modelID, sessionID: $0) }
        let kept = { ExecutorPromptCacheKey(modelID: Self.otherModelID, sessionID: $0) }
        // The first two writes run at once. The third write waits.
        writer.release()
        writer.release()
        await store.checkIn(evicted("disk"), Self.entry())
        await store.checkIn(kept("disk"), Self.entry())
        await store.waitForSpills()
        await store.checkIn(evicted("spilling"), Self.entry())
        let keptEntry = Self.entry()
        await store.configure(memoryBudgetBytes: Self.keptFileCount * keptEntry.byteCount)
        await store.checkIn(evicted("memory"), Self.entry())
        await store.checkIn(kept("memory"), keptEntry)

        await store.evict(modelID: Self.modelID)
        writer.release()
        await store.waitForSpills()

        #expect(await store.checkOut(evicted("disk")) == .none)
        #expect(await store.checkOut(evicted("spilling")) == .none)
        #expect(await store.checkOut(evicted("memory")) == .none)
        #expect(await store.spillingByteCount == 0)
        let keptFile = try #require(await store.checkOut(kept("disk")).spilledHandle)
        #expect(try Self.fileNames(in: directory) == [keptFile.url.lastPathComponent])
        #expect(await store.checkOut(kept("memory")) == .memory(keptEntry))
    }

    @Test("evict(modelID: nil) removes the files of every model")
    func evictOfEveryModelRemovesEveryFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await store.checkIn(Self.key("a"), Self.entry())
        await store.checkIn(
            ExecutorPromptCacheKey(modelID: Self.otherModelID, sessionID: "a"), Self.entry())
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == 2)

        await store.evict(modelID: nil)

        #expect(try Self.fileNames(in: directory).isEmpty)
        #expect(await store.diskByteCount == 0)
    }

    @Test("the spill line names the session, its bytes and the write duration")
    func theSpillLineNamesTheSessionTheBytesAndTheDuration() {
        let line = ExecutorPromptCacheReport.spillLine(
            key: Self.key("a"), byteCount: Self.reportedByteCount,
            duration: Self.reportedWriteDuration, outcome: .onDisk)

        #expect(
            line
                == "prompt cache spill model=test-org/prompt-cache-spool session=a bytes=4096 "
                + "seconds=1.250 result=on disk")
    }
}

#endif
