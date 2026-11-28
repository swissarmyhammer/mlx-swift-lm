// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A copy that keeps the first tokens of the ledger of its source
/// (``ExecutorPromptCacheStore/copy(modelID:from:to:keepingTokens:)``) holds exactly those
/// tokens when the caches rewind to them, or else the longest checkpoint of the source at or
/// below them, and it refuses more tokens than the ledger holds.
///
/// The entries are scripted caches checked into a store of each test. No weights are needed.
@Suite("A copy that keeps the first tokens of its source holds that prefix or a checkpoint")
struct PromptCacheCopyKeepingTokensTests {

    /// The model of every key here.
    private static let modelID = ExecutorPromptCacheKey.ModelID("test/keeping-tokens")

    /// The session that each test copies.
    private static let source = ExecutorPromptCacheKey.SessionID("source")

    /// The session that each copy makes.
    private static let destination = ExecutorPromptCacheKey.SessionID("destination")

    /// The tokens of the instructions of the hybrid entry, where it keeps its instructions
    /// checkpoint.
    private static let instructions = [1, 2]

    /// The tokens at the checkpoint at the end of the last message of the hybrid entry.
    private static let transcript = instructions + [3, 4]

    /// The ledger of each entry: the transcript, then the generated turn.
    private static let ledger = transcript + [5, 6]

    /// A token count between the end of the instructions and the end of the transcript.
    private static let insideThePrompt = instructions.count + 1

    /// The key of `sessionID` of ``modelID``.
    private static func key(_ sessionID: ExecutorPromptCacheKey.SessionID)
        -> ExecutorPromptCacheKey
    {
        PromptCacheTestKey.make(modelID: modelID, sessionID: sessionID)
    }

    /// A store whose writer writes nothing, holding `entry` for ``source``.
    ///
    /// - Parameter entry: the entry of the source.
    /// - Returns: the store.
    private static func store(holding entry: ExecutorPromptCacheEntry) async
        -> ExecutorPromptCacheStore
    {
        let store = PromptCacheTestStore.withoutWrites(
            folderNamed: "PromptCacheCopyKeepingTokensTests")
        await store.checkIn(key(source), entry)
        return store
    }

    /// Copies ``source`` to ``destination`` in `store`, keeping `tokenCount` tokens.
    private static func copy(
        in store: ExecutorPromptCacheStore, keepingTokens tokenCount: Int
    ) async -> PromptCacheCopyResult {
        await store.copy(
            modelID: modelID, from: source, to: destination, keepingTokens: tokenCount)
    }

    /// An entry of one attention layer that holds ``ledger``: its cache rewinds to any position.
    private static func attentionEntry() -> ExecutorPromptCacheEntry {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: ledger.count)
        return ExecutorPromptCacheEntry(caches: caches, tokens: ledger, renderTokens: transcript)
    }

    /// An entry of one attention layer and one recurrent layer that holds ``ledger``, with an
    /// instructions checkpoint at the end of ``instructions`` and, when `checkpointed`, a
    /// checkpoint at the end of ``transcript``.
    ///
    /// - Parameter checkpointed: whether the entry keeps its checkpoints.
    /// - Returns: the entry.
    /// - Throws: when a checkpoint is not taken.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func hybridEntry(checkpointed: Bool = true) throws -> ExecutorPromptCacheEntry {
        let caches = try hybridScriptedModel().newCache(parameters: nil)
        feed(caches, tokenCount: instructions.count)
        let instructionsCheckpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: instructions, state: nil))
        feed(caches, tokenCount: transcript.count - instructions.count)
        let checkpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: transcript, state: nil))
        feed(caches, tokenCount: ledger.count - transcript.count)
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: ledger, renderTokens: transcript,
            checkpoint: checkpointed ? checkpoint : nil,
            instructionsCheckpoint: checkpointed ? instructionsCheckpoint : nil)
    }

    @Test("caches that rewind keep exactly the tokens asked for")
    func cachesThatRewindKeepExactlyTheTokensAskedFor() async {
        let store = await Self.store(holding: Self.attentionEntry())

        let result = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(result == .copied(tokenCount: Self.insideThePrompt))
    }

    @Test("the kept ledger is the prefix of the ledger of the source")
    func theKeptLedgerIsThePrefixOfTheLedgerOfTheSource() async {
        let store = await Self.store(holding: Self.attentionEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(
            await store.peek(Self.key(Self.destination))?.tokens
                == Array(Self.ledger.prefix(Self.insideThePrompt)))
    }

    @Test("the kept caches stand at the end of the kept tokens")
    func theKeptCachesStandAtTheEndOfTheKeptTokens() async throws {
        let store = await Self.store(holding: Self.attentionEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        let copied = try #require(await store.peek(Self.key(Self.destination)))
        #expect(copied.caches.allSatisfy { $0.offset == Self.insideThePrompt })
    }

    @Test("a copy that keeps a prefix leaves the source whole")
    func aCopyThatKeepsAPrefixLeavesTheSourceWhole() async throws {
        let entry = Self.attentionEntry()
        let store = await Self.store(holding: entry)

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(entry.caches.allSatisfy { $0.offset == Self.ledger.count })
    }

    @Test("caches that cannot rewind keep the longest checkpoint at or below the tokens asked for")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func cachesThatCannotRewindKeepTheLongestCheckpointAtOrBelow() async throws {
        let store = await Self.store(holding: try Self.hybridEntry())

        let result = await Self.copy(in: store, keepingTokens: Self.transcript.count + 1)

        #expect(result == .copied(tokenCount: Self.transcript.count))
    }

    @Test("a count below the checkpoint keeps the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCountBelowTheCheckpointKeepsTheInstructionsCheckpoint() async throws {
        let store = await Self.store(holding: try Self.hybridEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(await store.peek(Self.key(Self.destination))?.tokens == Self.instructions)
    }

    @Test("caches restored to a checkpoint stand at the checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func cachesRestoredToACheckpointStandAtTheCheckpoint() async throws {
        let store = await Self.store(holding: try Self.hybridEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        let copied = try #require(await store.peek(Self.key(Self.destination)))
        #expect(copied.caches.allSatisfy { $0.offset == Self.instructions.count })
    }

    @Test("a copy restored to a checkpoint keeps the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCopyRestoredToACheckpointKeepsTheInstructionsCheckpoint() async throws {
        let store = await Self.store(holding: try Self.hybridEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.transcript.count)

        #expect(
            await store.peek(Self.key(Self.destination))?.instructionsCheckpoint?.tokens
                == Self.instructions)
    }

    @Test("a copy restored to a checkpoint leaves the source whole")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCopyRestoredToACheckpointLeavesTheSourceWhole() async throws {
        let entry = try Self.hybridEntry()
        let store = await Self.store(holding: entry)

        _ = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(entry.caches.allSatisfy { $0.offset == Self.ledger.count })
    }

    @Test("the whole ledger copies the whole entry")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theWholeLedgerCopiesTheWholeEntry() async throws {
        let store = await Self.store(holding: try Self.hybridEntry())

        let result = await Self.copy(in: store, keepingTokens: Self.ledger.count)

        #expect(result == .copied(tokenCount: Self.ledger.count))
    }

    @Test("more tokens than the ledger holds are refused")
    func moreTokensThanTheLedgerHoldsAreRefused() async {
        let store = await Self.store(holding: Self.attentionEntry())

        let result = await Self.copy(in: store, keepingTokens: Self.ledger.count + 1)

        #expect(result == .refused(.tokenCountBeyondLedger))
    }

    @Test("a refused count leaves no entry for the destination")
    func aRefusedCountLeavesNoEntryForTheDestination() async {
        let store = await Self.store(holding: Self.attentionEntry())

        _ = await Self.copy(in: store, keepingTokens: Self.ledger.count + 1)

        #expect(await store.peek(Self.key(Self.destination)) == nil)
    }

    @Test("caches that cannot rewind and keep no checkpoint at or below the count are refused")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func cachesWithNoCheckpointAtOrBelowTheCountAreRefused() async throws {
        let store = await Self.store(holding: try Self.hybridEntry(checkpointed: false))

        let result = await Self.copy(in: store, keepingTokens: Self.insideThePrompt)

        #expect(result == .refused(.noPrefixToKeep))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
