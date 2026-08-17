// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Unit tests for the pieces that let one session carry a prompt cache from one
/// turn to the next: the session key, the store and the slot.
///
/// No weights are needed. The store holds whatever entry it is given, thus the
/// entries here carry empty caches and a token ledger alone.
@Suite("A session carries its prompt cache between turns")
struct ExecutorPromptCacheTests {

    /// The model every key of this suite names.
    private static let modelID = "test/prompt-cache"

    /// A second model, to prove that two models never share one cache.
    private static let otherModelID = "test/prompt-cache-other"

    /// An entry that names `tokens` and carries one empty cache.
    private func entry(tokens: [Int]) -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(caches: [KVCacheSimple()], tokens: tokens)
    }

    /// A key for `sessionID` under ``modelID``.
    private func key(
        _ sessionID: String, modelID: String = ExecutorPromptCacheTests.modelID
    ) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
    }

    /// A transcript whose first entry carries `firstEntryID`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func transcript(firstEntryID: String, turns: Int = 1) -> Transcript {
        var entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    id: firstEntryID,
                    segments: [.text(Transcript.TextSegment(content: "first turn"))]))
        ]
        for turn in 1 ..< turns {
            entries.append(
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: "turn \(turn)"))])))
        }
        return Transcript(entries: entries)
    }

    // MARK: - Naming the session

    @Test("every turn of one session names the same cache")
    func everyTurnOfOneSessionNamesTheSameCache() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let firstTurn = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.modelID)
        let secondTurn = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a", turns: 3)),
            modelID: Self.modelID)

        #expect(firstTurn == secondTurn)
        #expect(firstTurn?.sessionID == "session-a")
    }

    @Test("two sessions never name the same cache")
    func twoSessionsNeverNameTheSameCache() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let first = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.modelID)
        let second = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-b")),
            modelID: Self.modelID)

        #expect(first != second)
    }

    @Test("one session on two models names two caches")
    func oneSessionOnTwoModelsNamesTwoCaches() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let onOneModel = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.modelID)
        let onAnother = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.otherModelID)

        #expect(onOneModel != onAnother)
    }

    @Test("an empty transcript names no session and carries no cache")
    func anEmptyTranscriptNamesNoSessionAndCarriesNoCache() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        #expect(
            MLXLanguageModel.Executor.sessionCacheKey(
                for: makeRequest(transcript: Transcript()), modelID: Self.modelID) == nil)
    }

    // MARK: - The store

    @Test("a checked-in cache comes back to the next turn of its session")
    func aCheckedInCacheComesBackToTheNextTurnOfItsSession() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        #expect(await store.checkOut(key("a"))?.tokens == [1, 2, 3])
    }

    @Test("a check-out takes the cache away, thus a second turn starts cold")
    func aCheckOutTakesTheCacheAwayThusASecondTurnStartsCold() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        _ = await store.checkOut(key("a"))

        #expect(await store.checkOut(key("a")) == nil)
    }

    @Test("checking in nothing releases the cache of that session")
    func checkingInNothingReleasesTheCacheOfThatSession() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        await store.checkIn(key("a"), nil)

        #expect(await store.checkOut(key("a")) == nil)
    }

    @Test("two sessions never read each other's cache")
    func twoSessionsNeverReadEachOthersCache() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))
        await store.checkIn(key("b"), entry(tokens: [7, 8]))

        #expect(await store.checkOut(key("a"))?.tokens == [1, 2, 3])
        #expect(await store.checkOut(key("b"))?.tokens == [7, 8])
    }

    @Test("the least recently used session loses its cache past the bound")
    func theLeastRecentlyUsedSessionLosesItsCachePastTheBound() async {
        let store = ExecutorPromptCacheStore()
        let bound = ExecutorPromptCacheStore.maximumRetainedSessions
        for session in 0 ... bound {
            await store.checkIn(key("session-\(session)"), entry(tokens: [session]))
        }

        #expect(await store.retainedSessionCount == bound)
        #expect(await store.checkOut(key("session-0")) == nil)
        #expect(await store.checkOut(key("session-\(bound)"))?.tokens == [bound])
    }

    @Test("evicting one model releases only the caches of that model")
    func evictingOneModelReleasesOnlyTheCachesOfThatModel() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1]))
        await store.checkIn(key("a", modelID: Self.otherModelID), entry(tokens: [2]))

        await store.evict(modelID: Self.modelID)

        #expect(await store.checkOut(key("a")) == nil)
        #expect(await store.checkOut(key("a", modelID: Self.otherModelID))?.tokens == [2])
    }

    @Test("evicting every model releases every cache")
    func evictingEveryModelReleasesEveryCache() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1]))
        await store.checkIn(key("a", modelID: Self.otherModelID), entry(tokens: [2]))

        await store.evict(modelID: nil)

        #expect(await store.retainedSessionCount == 0)
    }

    // MARK: - The slot

    @Test("a slot reports no reuse until a pass plans one")
    func aSlotReportsNoReuseUntilAPassPlansOne() {
        #expect(ExecutorPromptCacheSlot(entry(tokens: [1, 2, 3])).reusedTokenCount == 0)
    }

    @Test("a pass that carries no cache reports no reuse")
    func aPassThatCarriesNoCacheReportsNoReuse() {
        let slot = ExecutorPromptCacheSlot(entry(tokens: [1, 2, 3]))

        slot.carriesNoCache()

        #expect(slot.reusedTokenCount == 0)
    }

    @Test("a pass that commits nothing leaves the session cold")
    func aPassThatCommitsNothingLeavesTheSessionCold() {
        let slot = ExecutorPromptCacheSlot(entry(tokens: [1, 2, 3]))

        slot.commit(nil, generatedTokens: [])

        #expect(slot.entry == nil)
    }

    // MARK: - The ledger a finished pass leaves

    /// How many positions the rotating cache of this suite holds. A prompt
    /// longer than this stands past the window, thus the cache no longer
    /// rewinds.
    private static let slidingWindow = 8

    /// The number of sequences the key/value fixture carries.
    private static let batchSize = 1

    /// The number of attention heads the key/value fixture carries.
    private static let headCount = 2

    /// The width of one attention head of the key/value fixture.
    private static let headDimension = 4

    /// Takes every cache of `caches` to the position `tokenCount` tokens take
    /// it to.
    ///
    /// A ledger reads the POSITION of a cache and nothing else, thus a key of
    /// the right shape moves that position exactly as a prompt of the same
    /// length does.
    ///
    /// - Parameters:
    ///   - caches: the caches to feed.
    ///   - tokenCount: the number of tokens to feed.
    private func feed(_ caches: [KVCache], tokenCount: Int) {
        let keyValues = MLXArray.zeros([
            Self.batchSize, Self.headCount, tokenCount, Self.headDimension,
        ])
        for cache in caches {
            _ = cache.update(keys: keyValues, values: keyValues)
        }
    }

    /// A finished pass over `caches` that rendered `promptTokens`.
    private func plan(caches: [KVCache], promptTokens: [Int]) -> ExecutorPromptCachePlan {
        ExecutorPromptCachePlan(
            caches: caches,
            input: LMInput(tokens: MLXArray(promptTokens)),
            reusedTokenCount: 0,
            promptTokens: promptTokens)
    }

    @Test("a cache past its sliding window still carries a ledger to the next turn")
    func aCachePastItsSlidingWindowStillCarriesALedgerToTheNextTurn() {
        let caches: [KVCache] = [RotatingKVCache(maxSize: Self.slidingWindow)]
        let promptTokens = Array(1 ... 12)
        let generatedTokens = [101, 102, 103]
        feed(caches, tokenCount: promptTokens.count + generatedTokens.count)

        let committed = plan(caches: caches, promptTokens: promptTokens)
            .committed(generatedTokens: generatedTokens)

        #expect(!canTrimPromptCache(caches), "the premise: this cache cannot rewind")
        #expect(committed?.tokens == promptTokens + generatedTokens)
    }

    @Test("a cache that rewinds carries the same ledger")
    func aCacheThatRewindsCarriesTheSameLedger() {
        let caches: [KVCache] = [KVCacheSimple()]
        let promptTokens = [1, 2, 3, 4]
        let generatedTokens = [101, 102]
        feed(caches, tokenCount: promptTokens.count + generatedTokens.count)

        let committed = plan(caches: caches, promptTokens: promptTokens)
            .committed(generatedTokens: generatedTokens)

        #expect(committed?.tokens == promptTokens + generatedTokens)
    }

    @Test("a token the caches did not take stays out of the ledger")
    func aTokenTheCachesDidNotTakeStaysOutOfTheLedger() {
        let caches: [KVCache] = [KVCacheSimple()]
        let promptTokens = [1, 2, 3, 4]
        feed(caches, tokenCount: promptTokens.count + 1)

        let committed = plan(caches: caches, promptTokens: promptTokens)
            .committed(generatedTokens: [101, 102])

        #expect(committed?.tokens == promptTokens + [101])
    }

    @Test("caches that hold more than the pass generated leave the session cold")
    func cachesThatHoldMoreThanThePassGeneratedLeaveTheSessionCold() {
        let caches: [KVCache] = [KVCacheSimple()]
        let promptTokens = [1, 2, 3, 4]
        feed(caches, tokenCount: promptTokens.count + 3)

        #expect(
            plan(caches: caches, promptTokens: promptTokens)
                .committed(generatedTokens: [101, 102]) == nil)
    }

    @Test("caches that disagree on their position leave the session cold")
    func cachesThatDisagreeOnTheirPositionLeaveTheSessionCold() {
        let leading = KVCacheSimple()
        let lagging = KVCacheSimple()
        let promptTokens = [1, 2, 3, 4]
        feed([leading], tokenCount: promptTokens.count + 1)
        feed([lagging], tokenCount: promptTokens.count)

        #expect(
            plan(caches: [leading, lagging], promptTokens: promptTokens)
                .committed(generatedTokens: [101]) == nil)
    }
}

/// A request carrying `transcript` and nothing else.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeRequest(
    transcript: Transcript
) -> LanguageModelExecutorGenerationRequest {
    LanguageModelExecutorGenerationRequest(
        id: UUID(),
        transcript: transcript,
        enabledTools: [],
        generationOptions: GenerationOptions(),
        contextOptions: ContextOptions(),
        metadata: [:])
}

#endif  // FoundationModelsIntegration
