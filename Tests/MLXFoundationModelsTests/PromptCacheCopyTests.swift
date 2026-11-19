// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Tests for the copy of the prompt cache of one session to a new session
/// (``MLXLanguageModel/copyPromptCache(fromSession:toSession:)``), which a host calls when it
/// makes a fork.
///
/// The store tests check in fixture entries from ``PromptCacheSpoolFixtures``. The pass tests
/// run the executor over a scripted model, with no weights. Each test binds its own store.
@Suite("A copy of the prompt cache of a session gives a new session an independent cache")
struct PromptCacheCopyTests: PromptCacheSpoolFixtures, PromptCacheKeyFixtures {

    // MARK: - Fixture values

    /// The model of the store tests. It holds a `/`, as a real model ID does.
    static let modelID = "test-org/prompt-cache-copy"

    /// The session that each test copies.
    private static let source = "source"

    /// The session that each copy makes.
    private static let destination = "destination"

    /// A KV-cache configuration beside full precision, for a source with two configurations.
    private static let eightBit = KVCacheConfiguration(strategy: .affine(.eightBit))

    /// The text of the first message of each scripted session, as
    /// ``ScriptedSessionModel/transcript(firstEntryID:turns:)`` writes it.
    private static let firstMessage = "first turn"

    // MARK: - Store fixtures

    /// Copies ``source`` to ``destination`` of ``modelID`` in `store`.
    private static func copySourceToDestination(
        in store: ExecutorPromptCacheStore
    ) async -> PromptCacheCopyResult {
        await store.copy(
            modelID: .init(modelID), from: .init(source), to: .init(destination))
    }

    /// The state arrays of each cache of `entry`.
    private static func states(of entry: ExecutorPromptCacheEntry) -> [MLXArray] {
        entry.caches.flatMap(\.state)
    }

    // MARK: - The tiers

    @Test("a copy of an entry in memory gives new caches with the same tokens and arrays")
    func aCopyInMemoryGivesNewCachesWithTheSameTokensAndArrays() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceEntry = Self.entry(renderTokens: [7, 8])
        let store = await Self.store(in: directory, budget: 4 * sourceEntry.byteCount)
        await store.checkIn(Self.key(Self.source), sourceEntry)

        let result = await Self.copySourceToDestination(in: store)

        #expect(result == .copied(tokenCount: sourceEntry.tokens.count))
        let copied = try #require(await store.peek(Self.key(Self.destination)))
        #expect(copied !== sourceEntry)
        #expect(copied.tokens == sourceEntry.tokens)
        #expect(copied.renderTokens == sourceEntry.renderTokens)
        #expect(copied.caches.count == sourceEntry.caches.count)
        for (copy, original) in zip(copied.caches, sourceEntry.caches) {
            #expect(
                (copy as AnyObject) !== (original as AnyObject),
                "The copy must never share a cache object.")
            #expect(copy.offset == original.offset)
        }
        PromptCacheArrayExpectations.expectEqual(
            Self.states(of: copied), Self.states(of: sourceEntry), "copy")
        #expect(await store.peek(Self.key(Self.source)) === sourceEntry)
    }

    @Test("a write into the copy does not change the source, and the reverse")
    func aWriteIntoTheCopyDoesNotChangeTheSource() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceEntry = Self.entry()
        let store = await Self.store(in: directory, budget: 4 * sourceEntry.byteCount)
        await store.checkIn(Self.key(Self.source), sourceEntry)
        let sourceArrays = Self.states(of: sourceEntry).map { $0[.ellipsis] }
        eval(sourceArrays)
        _ = await Self.copySourceToDestination(in: store)
        let copied = try #require(await store.peek(Self.key(Self.destination)))
        let copiedArrays = Self.states(of: copied).map { $0[.ellipsis] }
        eval(copiedArrays)
        let tokenCount = sourceEntry.tokens.count

        // The copy rewinds two positions and writes new values there.
        #expect(copied.caches[0].trim(2) == 2)
        _ = copied.caches[0].update(
            keys: Self.block(tokenCount: 2, array: .values),
            values: Self.block(tokenCount: 2, array: .keys))
        eval(copied.caches.flatMap(\.state))

        #expect(sourceEntry.caches[0].offset == tokenCount)
        PromptCacheArrayExpectations.expectEqual(
            Self.states(of: sourceEntry), sourceArrays, "the source after a write of the copy")

        // The source takes one more position.
        _ = sourceEntry.caches[0].update(
            keys: Self.block(tokenCount: 1, array: .keys),
            values: Self.block(tokenCount: 1, array: .values))
        eval(sourceEntry.caches.flatMap(\.state))

        #expect(copied.caches[0].offset == tokenCount)
        #expect(
            !arrayEqual(Self.states(of: copied)[0], copiedArrays[0]).item(Bool.self),
            "The write of the copy must show in the copy.")
        #expect(Self.states(of: copied)[0].dim(2) == tokenCount)
    }

    @Test("a source with two KV-cache configurations gives two copies")
    func aSourceWithTwoConfigurationsGivesTwoCopies() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let fullPrecision = Self.entry(tokens: [1, 2, 3])
        let compressed = Self.entry(tokens: [1, 2, 3, 4, 5])
        let store = await Self.store(in: directory, budget: 8 * compressed.byteCount)
        await store.checkIn(Self.key(Self.source), fullPrecision)
        await store.checkIn(Self.configuredKey(Self.source), compressed)

        let result = await Self.copySourceToDestination(in: store)

        #expect(result == .copied(tokenCount: compressed.tokens.count))
        #expect(await store.peek(Self.key(Self.destination))?.tokens == fullPrecision.tokens)
        #expect(
            await store.peek(Self.configuredKey(Self.destination))?.tokens == compressed.tokens)
        #expect(await store.retainedSessionCount == 4)
    }

    @Test("a source whose write to disk has not ended gives a copy in memory")
    func aSpillingSourceGivesACopyInMemory() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let writer = try HeldWriter(copies: Self.temporaryDirectory())
        defer { try? FileManager.default.removeItem(at: writer.copies) }
        let store = await Self.store(in: directory, writer: writer.write)
        var starts = writer.startedWrites.makeAsyncIterator()
        let sourceEntry = Self.entry()
        await store.checkIn(Self.key(Self.source), sourceEntry)
        _ = await starts.next()
        await store.configure(memoryBudgetBytes: 4 * sourceEntry.byteCount)

        let result = await Self.copySourceToDestination(in: store)
        writer.release()
        await store.waitForSpills()

        #expect(result == .copied(tokenCount: sourceEntry.tokens.count))
        let copied = try #require(await store.peek(Self.key(Self.destination)))
        #expect(copied.tokens == sourceEntry.tokens)
        #expect(await store.retainedByteCount == copied.byteCount)
        let sourceFile = try #require(await store.checkOut(Self.key(Self.source)).spilledHandle)
        #expect(FileManager.default.fileExists(atPath: sourceFile.url.path))
    }

    @Test("a source on disk gives a clone of its file, and the file of the source stays")
    func aSourceOnDiskGivesACloneOfItsFile() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory)
        let sourceEntry = Self.entry(renderTokens: [9])
        await store.checkIn(Self.key(Self.source), sourceEntry)
        await store.waitForSpills()
        let diskBytesBefore = await store.diskByteCount

        let result = await Self.copySourceToDestination(in: store)

        #expect(result == .copied(tokenCount: sourceEntry.tokens.count))
        #expect(try Self.fileNames(in: directory).count == 2)
        #expect(await store.diskByteCount == 2 * diskBytesBefore)
        let copyFile = try #require(
            await store.checkOut(Self.key(Self.destination)).spilledHandle)
        defer { ExecutorPromptCacheFile.removeFile(at: copyFile.url) }
        #expect(copyFile.key == Self.key(Self.destination))
        #expect(copyFile.fileKey == Self.key(Self.source))
        let restored = try ExecutorPromptCacheFile.read(
            from: copyFile.url, key: copyFile.fileKey, templates: [KVCacheSimple()])
        #expect(restored.tokens == sourceEntry.tokens)
        #expect(restored.renderTokens == sourceEntry.renderTokens)
        PromptCacheArrayExpectations.expectEqual(
            Self.states(of: restored), Self.states(of: sourceEntry), "the restored copy")
        let sourceFile = try #require(await store.checkOut(Self.key(Self.source)).spilledHandle)
        #expect(sourceFile.url != copyFile.url)
        #expect(FileManager.default.fileExists(atPath: sourceFile.url.path))
    }

    // MARK: - Budget and release

    @Test("the copy counts its bytes in the usage")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theCopyCountsItsBytesInTheUsage() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceEntry = Self.entry()
        let store = await Self.store(in: directory, budget: 4 * sourceEntry.byteCount)
        await store.checkIn(Self.key(Self.source), sourceEntry)

        let (before, after) = await ExecutorPromptCacheStore.$current.withValue(store) {
            let before = await MLXLanguageModel.promptCacheUsage
            _ = await makeStubModel(Self.modelID).copyPromptCache(
                fromSession: Self.source, toSession: Self.destination)
            return (before, await MLXLanguageModel.promptCacheUsage)
        }

        let copied = try #require(await store.peek(Self.key(Self.destination)))
        #expect(copied.byteCount == sourceEntry.byteCount)
        #expect(after.memoryBytes == before.memoryBytes + copied.byteCount)
    }

    @Test("a small budget sends the least recently used entry to disk, not the source")
    func aSmallBudgetEvictsTheLeastRecentlyUsedEntryNotTheSource() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let entryBytes = Self.entry().byteCount
        let store = await Self.store(in: directory, budget: 3 * entryBytes)
        // The source is the least recently used entry before the copy.
        await store.checkIn(Self.key(Self.source), Self.entry())
        await store.checkIn(Self.key("older"), Self.entry())
        await store.checkIn(Self.key("newer"), Self.entry())

        let result = await Self.copySourceToDestination(in: store)
        await store.waitForSpills()

        #expect(result == .copied(tokenCount: Self.entry().tokens.count))
        #expect(await store.peek(Self.key(Self.source)) != nil)
        #expect(await store.peek(Self.key(Self.destination)) != nil)
        #expect(await store.peek(Self.key("newer")) != nil)
        #expect(await store.peek(Self.key("older")) == nil)
        #expect(await store.checkOut(Self.key("older")).spilledHandle != nil)
        #expect(await store.retainedByteCount == 3 * entryBytes)
    }

    @Test("a release of the copy leaves the source, and a release of the source leaves the copy")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aReleaseOfOneSessionLeavesTheOther() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceEntry = Self.entry()
        let store = await Self.store(in: directory, budget: 4 * sourceEntry.byteCount)
        await store.checkIn(Self.key(Self.source), sourceEntry)
        _ = await Self.copySourceToDestination(in: store)
        let copied = try #require(await store.peek(Self.key(Self.destination)))
        let model = makeStubModel(Self.modelID)

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.releasePromptCache(sessionID: Self.destination)
        }

        #expect(await store.peek(Self.key(Self.destination)) == nil)
        #expect(await store.peek(Self.key(Self.source)) === sourceEntry)

        _ = await Self.copySourceToDestination(in: store)
        let secondCopy = try #require(await store.peek(Self.key(Self.destination)))
        #expect(secondCopy !== copied)
        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.releasePromptCache(sessionID: Self.source)
        }

        #expect(await store.peek(Self.key(Self.source)) == nil)
        #expect(await store.peek(Self.key(Self.destination)) === secondCopy)
        #expect(secondCopy.caches[0].offset == sourceEntry.tokens.count)
    }

    // MARK: - What the store does not copy

    @Test("an unknown source gives sourceNotFound")
    func anUnknownSourceGivesSourceNotFound() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        await store.checkIn(
            Self.key(.init(Self.source), modelID: .init("other/model")), Self.entry())

        #expect(await Self.copySourceToDestination(in: store) == .sourceNotFound)
        #expect(await store.peek(Self.key(Self.destination)) == nil)
    }

    @Test("a destination with a cache is refused, and its cache does not change")
    func aDestinationWithACacheIsRefused() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        await store.checkIn(Self.key(Self.source), Self.entry())
        let destinationEntry = Self.entry(tokens: [1])
        await store.checkIn(Self.configuredKey(Self.destination), destinationEntry)

        #expect(
            await Self.copySourceToDestination(in: store) == .refused(.destinationHasCache))
        #expect(await store.peek(Self.configuredKey(Self.destination)) === destinationEntry)
        #expect(await store.peek(Self.key(Self.destination)) == nil)
    }

    @Test("a copy to the same session is refused")
    func aCopyToTheSameSessionIsRefused() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        await store.checkIn(Self.key(Self.source), Self.entry())

        let result = await store.copy(
            modelID: .init(Self.modelID), from: .init(Self.source), to: .init(Self.source))

        #expect(result == .refused(.sameSession))
        #expect(await store.retainedSessionCount == 1)
    }

    @Test("a source that a pass holds gives sourceInUse at once, and copies after its check-in")
    func aSourceThatAPassHoldsGivesSourceInUse() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        await store.checkIn(Self.key(Self.source), Self.entry())
        let held = try #require(await store.checkOut(Self.key(Self.source)).entry)
        #expect(await store.isCheckedOut(Self.key(Self.source)))

        #expect(await Self.copySourceToDestination(in: store) == .sourceInUse)
        #expect(await store.peek(Self.key(Self.destination)) == nil)

        await store.checkIn(Self.key(Self.source), held)
        #expect(!(await store.isCheckedOut(Self.key(Self.source))))

        #expect(
            await Self.copySourceToDestination(in: store)
                == .copied(tokenCount: held.tokens.count))
    }

    @Test("a source that a pass holds with no cache gives sourceInUse, not sourceNotFound")
    func aSourceHeldWithNoCacheGivesSourceInUse() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        #expect(await store.checkOut(Self.key(Self.source)) == .none)

        #expect(await Self.copySourceToDestination(in: store) == .sourceInUse)

        await store.checkIn(Self.key(Self.source), nil)
        #expect(await Self.copySourceToDestination(in: store) == .sourceNotFound)
    }

    @Test("a destination that a pass holds is refused")
    func aDestinationThatAPassHoldsIsRefused() async throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await Self.store(in: directory, budget: 4 * Self.entry().byteCount)
        await store.checkIn(Self.key(Self.source), Self.entry())
        _ = await store.checkOut(Self.key(Self.destination))

        #expect(await Self.copySourceToDestination(in: store) == .refused(.destinationInUse))
        await store.checkIn(Self.key(Self.destination), nil)
        #expect(await store.peek(Self.key(Self.destination)) == nil)
    }

    // MARK: - Passes of the executor

    @Test("the first pass of the copy reads only the new tokens, as a cold pass answers")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theFirstPassOfTheCopyReadsOnlyTheNewTokens() async throws {
        let session = try ScriptedCopySession()
        defer { session.remove() }
        try await session.respond(turns: 1, as: Self.source)
        let sourceEntry = try #require(await session.peek(Self.source))
        let sourceTokens = sourceEntry.tokens
        let sourceOffsets = sourceEntry.caches.map(\.offset)

        let result = await session.copy(from: Self.source, to: Self.destination)
        let warm = try await session.respond(turns: 2, as: Self.destination)
        let cold = try await session.respond(turns: 2, as: "cold")

        #expect(result == .copied(tokenCount: sourceTokens.count))
        // The render of the second turn parts from the ledger of the first turn where the
        // first turn generated its answer: the copy serves every token of the first message.
        #expect(warm.reusedTokenCount == Self.firstMessage.utf8.count)
        #expect(cold.reusedTokenCount == 0)
        #expect(warm.promptTokenCount == cold.promptTokenCount)
        #expect(warm.responseText == cold.responseText)
        let warmEntry = try #require(await session.peek(Self.destination))
        let coldEntry = try #require(await session.peek("cold"))
        #expect(warmEntry.tokens == coldEntry.tokens)
        #expect(warmEntry.caches.map(\.offset) == coldEntry.caches.map(\.offset))
        // The pass of the copy did not change the source.
        let sourceAfter = try #require(await session.peek(Self.source))
        #expect(sourceAfter === sourceEntry)
        #expect(sourceAfter.tokens == sourceTokens)
        #expect(sourceAfter.caches.map(\.offset) == sourceOffsets)
    }

    @Test("a pass of the source does not change the copy")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassOfTheSourceDoesNotChangeTheCopy() async throws {
        let session = try ScriptedCopySession()
        defer { session.remove() }
        try await session.respond(turns: 1, as: Self.source)
        _ = await session.copy(from: Self.source, to: Self.destination)
        let copied = try #require(await session.peek(Self.destination))
        let copiedTokens = copied.tokens
        let copiedOffsets = copied.caches.map(\.offset)

        let sourcePass = try await session.respond(turns: 2, as: Self.source)

        #expect(sourcePass.reusedTokenCount > 0)
        let sourceAfter = try #require(await session.peek(Self.source))
        #expect(sourceAfter.tokens != copiedTokens)
        let copyAfter = try #require(await session.peek(Self.destination))
        #expect(copyAfter === copied)
        #expect(copyAfter.tokens == copiedTokens)
        #expect(copyAfter.caches.map(\.offset) == copiedOffsets)
    }

    @Test("a copy whose render parts early rewinds, and the source keeps all its tokens")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCopyThatPartsEarlyRewindsAndTheSourceKeepsItsTokens() async throws {
        let session = try ScriptedCopySession()
        defer { session.remove() }
        try await session.respond(turns: 1, as: Self.source)
        try await session.respond(turns: 2, as: Self.source)
        let sourceEntry = try #require(await session.peek(Self.source))
        let sourceTokens = sourceEntry.tokens

        _ = await session.copy(from: Self.source, to: Self.destination)
        let forked = try await session.respond(
            over: Self.transcript(secondPrompt: "other"), as: Self.destination)

        // "first turn\nother" parts from "first turn\nturn 1..." after "first turn\n".
        let sharedPrefix = (Self.firstMessage + "\n").utf8.count
        #expect(forked.reusedTokenCount == sharedPrefix)
        #expect(sourceTokens.count > sharedPrefix)
        let sourceAfter = try #require(await session.peek(Self.source))
        #expect(sourceAfter === sourceEntry)
        #expect(sourceAfter.tokens == sourceTokens)
        #expect(sourceAfter.caches.allSatisfy { $0.offset == sourceTokens.count })
        let forkEntry = try #require(await session.peek(Self.destination))
        #expect(
            Array(forkEntry.tokens.prefix(sharedPrefix)) == Array(sourceTokens.prefix(sharedPrefix))
        )
    }

    @Test("a hybrid copy goes back to its checkpoint and never resumes a state past the cut")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aHybridCopyGoesBackToItsCheckpoint() async throws {
        let session = try ScriptedCopySession(recurrentLayerCount: 1)
        defer { session.remove() }
        try await session.respond(turns: 1, as: Self.source)
        let sourceEntry = try #require(await session.peek(Self.source))
        let checkpoint = try #require(sourceEntry.checkpointPosition)
        // The ledger holds the answer of the first turn past the checkpoint, and the recurrent
        // layer holds that answer: it cannot rewind into the first message.
        #expect(sourceEntry.tokens.count > checkpoint)

        _ = await session.copy(from: Self.source, to: Self.destination)
        let copied = try #require(await session.peek(Self.destination))
        #expect(copied.checkpointPosition == checkpoint)
        let warm = try await session.respond(turns: 2, as: Self.destination)
        let cold = try await session.respond(turns: 2, as: "cold")

        // The render of the second turn parts from the ledger at the end of the first
        // message, which is where the checkpoint stands. The pass reuses the checkpoint, not
        // the recurrent state at the end of the ledger.
        #expect(checkpoint == Self.firstMessage.utf8.count)
        #expect(warm.reusedTokenCount == checkpoint)
        let warmEntry = try #require(await session.peek(Self.destination))
        let coldEntry = try #require(await session.peek("cold"))
        #expect(warmEntry.tokens == coldEntry.tokens)
        #expect(warmEntry.caches.map(\.offset) == coldEntry.caches.map(\.offset))
        #expect(warmEntry.caches.allSatisfy { $0.offset == warmEntry.tokens.count })
        let sourceAfter = try #require(await session.peek(Self.source))
        #expect(sourceAfter === sourceEntry)
        #expect(sourceAfter.caches.allSatisfy { $0.offset == sourceEntry.tokens.count })
    }

    @Test("a copy of a source on disk serves the first pass of the copy")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCopyOnDiskServesTheFirstPassOfTheCopy() async throws {
        let session = try ScriptedCopySession()
        defer { session.remove() }
        await session.store.configure(memoryBudgetBytes: 0)
        try await session.respond(turns: 1, as: Self.source)
        await session.store.waitForSpills()
        #expect(await session.store.retainedSessionCount == 0)

        let result = await session.copy(from: Self.source, to: Self.destination)
        let warm = try await session.respond(turns: 2, as: Self.destination)

        guard case .copied = result else {
            Issue.record("The copy of a source on disk must copy: \(result)")
            return
        }
        #expect(warm.reusedTokenCount == Self.firstMessage.utf8.count)
        await session.store.waitForSpills()
        let sourceFile = try #require(
            await session.store.checkOut(session.key(Self.source)).spilledHandle)
        defer { ExecutorPromptCacheFile.removeFile(at: sourceFile.url) }
        #expect(FileManager.default.fileExists(atPath: sourceFile.url.path))
    }

    // MARK: - Helpers

    /// The key of `sessionID` under ``modelID`` and ``eightBit``.
    private static func configuredKey(_ sessionID: String) -> ExecutorPromptCacheKey {
        PromptCacheTestKey.make(
            modelID: .init(modelID), sessionID: .init(sessionID), kvCache: eightBit)
    }

    /// A transcript of the first message and one more prompt.
    ///
    /// - Parameter secondPrompt: the text of the second prompt.
    /// - Returns: the transcript.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func transcript(secondPrompt: String) -> Transcript {
        let first = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: firstMessage))])
        let second = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: secondPrompt))])
        return Transcript(entries: [.prompt(first), .prompt(second)])
    }
}

/// A scripted model with one KV cache layer, and a store of its own, for the passes of the
/// copy tests. Each pass binds its session with ``MLXLanguageModel/promptCacheScope``.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct ScriptedCopySession: PromptCacheSpoolFixtures {

    /// The most passes one test runs.
    private static let passCount = 4

    /// The folder that makes the model available.
    let weights: URL

    /// The spool folder of the store.
    let directory: URL

    /// The model.
    let model: MLXLanguageModel

    /// The store of the passes.
    let store: ExecutorPromptCacheStore

    /// Makes the model and the store.
    ///
    /// - Parameter recurrentLayerCount: the recurrent layers of the model beside its KV cache
    ///   layer.
    init(recurrentLayerCount: Int = 0) throws {
        weights = try makeScriptedWeightsDirectory()
        directory = Self.temporaryDirectory()
        model = ScriptedSessionModel.make(
            weights: weights,
            scripts: Array(repeating: ScriptedSessionModel.scriptedResponse, count: Self.passCount),
            recurrentLayerCount: recurrentLayerCount)
        store = ExecutorPromptCacheStore(
            directory: directory,
            capacityReader: PromptCacheSpoolFixtureVolume.capacity(ofVolumeOf:))
    }

    /// The key of `sessionID` for the model.
    func key(_ sessionID: String) -> ExecutorPromptCacheKey {
        PromptCacheTestKey.make(modelID: .init(model.modelID), sessionID: .init(sessionID))
    }

    /// The entry of `sessionID` in memory, or nil.
    func peek(_ sessionID: String) async -> ExecutorPromptCacheEntry? {
        await store.peek(key(sessionID))
    }

    /// Copies the cache of `source` to `destination` through the public API.
    func copy(from source: String, to destination: String) async -> PromptCacheCopyResult {
        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.copyPromptCache(fromSession: source, toSession: destination)
        }
    }

    /// Runs one pass over a transcript of `turns` prompts, bound to `sessionID`.
    @discardableResult
    func respond(turns: Int, as sessionID: String) async throws -> ScriptedPassResult {
        try await respond(
            over: ScriptedSessionModel.transcript(
                firstEntryID: "entry-\(UUID().uuidString)", turns: turns),
            as: sessionID)
    }

    /// Runs one pass over `transcript`, bound to `sessionID`.
    @discardableResult
    func respond(over transcript: Transcript, as sessionID: String) async throws
        -> ScriptedPassResult
    {
        try await MLXLanguageModel.$promptCacheScope.withValue(.session(sessionID)) {
            try await ScriptedExecutorPass.respond(over: transcript, model: model, inside: store)
        }
    }

    /// Removes the folders.
    func remove() {
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.removeItem(at: weights)
    }
}

#endif
