// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels

/// Tests for the public prompt cache API of ``MLXLanguageModel``: the two budgets, the release
/// of one session, and the byte totals.
///
/// Each test binds its own store with `ExecutorPromptCacheStore.$current.withValue(store)`, and
/// never touches the shared store. The fixtures come from ``PromptCacheSpoolFixtures``.
@Suite("The public prompt cache API sets the budgets and releases one session")
struct PromptCachePublicAPITests: PromptCacheSpoolFixtures {

    // MARK: - Fixture values

    /// The model whose sessions the tests release. It holds a `/`, as a real model ID does.
    private static let modelID = "test-org/prompt-cache-public-api"

    /// A second model. Its sessions must stay when the first model releases a session.
    private static let otherModelID = "test-org/prompt-cache-public-api-other"

    /// The session that each release test releases.
    private static let releasedSessionID = "released"

    /// A session of the same model that each release test keeps.
    private static let keptSessionID = "kept"

    /// How many sessions ``checkInThreeSessions(into:)`` checks in. A memory budget of this
    /// many entries keeps each of them in memory.
    private static let sessionCount = 3

    // MARK: - Fixture builders

    /// The model that the tests call. Its loader never runs.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func model() -> MLXLanguageModel {
        makeStubModel(modelID)
    }

    /// The key of `sessionID` under ``modelID``.
    private static func key(_ sessionID: String) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
    }

    /// The key of the released session under ``otherModelID``.
    private static var otherModelKey: ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: otherModelID, sessionID: releasedSessionID)
    }

    /// Checks in one entry for the released session, one for the kept session, and one for the
    /// released session of the other model, in this order.
    ///
    /// - Parameter store: The store.
    private static func checkInThreeSessions(into store: ExecutorPromptCacheStore) async {
        await store.checkIn(key(releasedSessionID), entry())
        await store.checkIn(key(keptSessionID), entry())
        await store.checkIn(otherModelKey, entry())
    }

    /// Releases the released session of ``modelID`` inside `store`.
    ///
    /// - Parameter store: The store that the call binds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func release(inside store: ExecutorPromptCacheStore) async {
        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model().releasePromptCache(sessionID: releasedSessionID)
        }
    }

    // MARK: - Release of one session

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a release of a session in memory removes it and keeps other sessions and models")
    func aReleaseInMemoryKeepsOtherSessionsAndModels() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let entryBytes = Self.entry().byteCount
        let store = await Self.store(in: directory, budget: Self.sessionCount * entryBytes)
        await Self.checkInThreeSessions(into: store)

        await Self.release(inside: store)

        #expect(await store.checkOut(Self.key(Self.releasedSessionID)) == .none)
        #expect(await store.checkOut(Self.key(Self.keptSessionID)).entry != nil)
        #expect(await store.checkOut(Self.otherModelKey).entry != nil)
        #expect(try Self.fileNames(in: directory).isEmpty)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a release of a session on disk deletes its file and keeps other sessions and models")
    func aReleaseOnDiskDeletesItsFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await Self.checkInThreeSessions(into: store)
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == Self.sessionCount)

        await Self.release(inside: store)

        #expect(await store.checkOut(Self.key(Self.releasedSessionID)) == .none)
        let kept = try #require(await store.checkOut(Self.key(Self.keptSessionID)).spilledHandle)
        let otherModel = try #require(await store.checkOut(Self.otherModelKey).spilledHandle)
        let remaining = [kept.url.lastPathComponent, otherModel.url.lastPathComponent].sorted()
        #expect(try Self.fileNames(in: directory) == remaining)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a release of a session during its spill leaves nothing after the write ends")
    func aReleaseDuringASpillLeavesNothing() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()
        await store.checkIn(Self.key(Self.releasedSessionID), Self.entry())
        _ = await starts.next()

        await Self.release(inside: store)
        writer.release()
        await store.waitForSpills()

        try await Self.expectNothingStored(
            for: Self.key(Self.releasedSessionID), in: store, directory: directory)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a release of an unknown session changes nothing")
    func aReleaseOfAnUnknownSessionChangesNothing() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let kept = Self.entry()
        let store = await Self.store(in: directory, budget: kept.byteCount)
        await store.checkIn(Self.key(Self.keptSessionID), kept)

        await Self.release(inside: store)

        #expect(await store.retainedByteCount == kept.byteCount)
        #expect(await store.checkOut(Self.key(Self.keptSessionID)) == .memory(kept))
    }

    // MARK: - Budgets

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a smaller memory budget sends the entries in memory to disk at once")
    func aSmallerMemoryBudgetEvictsAtOnce() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let entryBytes = Self.entry().byteCount
        let store = await Self.store(in: directory, budget: Self.sessionCount * entryBytes)
        await Self.checkInThreeSessions(into: store)

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await MLXLanguageModel.configurePromptCache(memoryBudgetBytes: 0)
        }

        #expect(await store.memoryBudgetBytes == 0)
        #expect(await store.retainedByteCount == 0)
        await store.waitForSpills()
        #expect(try Self.fileNames(in: directory).count == Self.sessionCount)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a smaller disk budget deletes the files at once")
    func aSmallerDiskBudgetDeletesTheFilesAtOnce() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        await Self.checkInThreeSessions(into: store)
        await store.waitForSpills()

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await MLXLanguageModel.configurePromptCache(diskBudgetBytes: 0)
        }

        #expect(await store.diskBudgetBytes == 0)
        #expect(await store.diskByteCount == 0)
        #expect(try Self.fileNames(in: directory).isEmpty)
    }

    // MARK: - Usage

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the usage reports the bytes in memory, in the spill and on disk")
    func theUsageReportsTheThreeByteTotals() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()
        // The first write runs at once. The second write waits.
        writer.release()
        await store.checkIn(Self.key("disk"), Self.entry())
        _ = await starts.next()
        await store.waitForSpills()
        await store.checkIn(Self.key("spilling"), Self.entry())
        _ = await starts.next()
        let inMemory = Self.entry()
        await store.configure(memoryBudgetBytes: inMemory.byteCount)
        await store.checkIn(Self.key("memory"), inMemory)

        let usage = await ExecutorPromptCacheStore.$current.withValue(store) {
            await MLXLanguageModel.promptCacheUsage
        }
        writer.release()
        await store.waitForSpills()

        #expect(usage.memoryBytes == inMemory.byteCount)
        #expect(usage.spillingBytes == inMemory.byteCount)
        let fileBytes = try await Self.diskBytes(of: Self.key("disk"), in: store)
        #expect(fileBytes > 0)
        #expect(usage.diskBytes == fileBytes)
    }

    /// The size of the file of `key` on disk.
    ///
    /// - Parameters:
    ///   - key: The key whose file is on disk.
    ///   - store: The store. The check-out takes the file out of it.
    /// - Returns: The size of the file in bytes.
    private static func diskBytes(
        of key: ExecutorPromptCacheKey, in store: ExecutorPromptCacheStore
    ) async throws -> Int {
        let handle = try #require(await store.checkOut(key).spilledHandle)
        return try #require(handle.url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
    }
}

#endif
