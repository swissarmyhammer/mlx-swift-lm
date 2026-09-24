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
