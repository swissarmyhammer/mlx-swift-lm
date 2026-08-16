// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
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

        slot.commit(nil)

        #expect(slot.entry == nil)
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
