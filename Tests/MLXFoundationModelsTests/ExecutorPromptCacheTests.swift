// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Unit tests for the pieces that let one session carry a prompt cache from one
/// turn to the next: the session key, the store, the slot and the plan of one
/// pass.
///
/// No weights are needed. The store holds whatever entry it is given, thus the
/// entries here carry small zero-filled caches and a token ledger alone. The
/// caches hold real bytes, because the store limits its memory by bytes.
@Suite("A session carries its prompt cache between turns")
struct ExecutorPromptCacheTests: PromptCacheKeyFixtures, PromptCachePlanFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache"

    /// A second model, to prove that two models never share one cache.
    private static let otherModelID = "test/prompt-cache-other"

    /// A store in a folder of its own, whose writer writes nothing.
    ///
    /// These tests read the memory tier alone, and
    /// `ExecutorPromptCacheSpoolTests` reads the files. A writer that writes
    /// nothing thus leaves no file, and two stores never share a file name.
    private func makeStore() -> ExecutorPromptCacheStore {
        ExecutorPromptCacheStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("ExecutorPromptCacheTests-\(UUID().uuidString)"),
            writer: { _, _ in })
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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func everyTurnOfOneSessionNamesTheSameCache() throws {
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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func twoSessionsNeverNameTheSameCache() throws {
        let first = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.modelID)
        let second = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-b")),
            modelID: Self.modelID)

        #expect(first != second)
    }

    @Test("one session on two models names two caches")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func oneSessionOnTwoModelsNamesTwoCaches() throws {
        let onOneModel = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.modelID)
        let onAnother = MLXLanguageModel.Executor.sessionCacheKey(
            for: makeRequest(transcript: transcript(firstEntryID: "session-a")),
            modelID: Self.otherModelID)

        #expect(onOneModel != onAnother)
    }

    @Test("an empty transcript names no session and carries no cache")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEmptyTranscriptNamesNoSessionAndCarriesNoCache() throws {
        #expect(
            MLXLanguageModel.Executor.sessionCacheKey(
                for: makeRequest(transcript: Transcript()), modelID: Self.modelID) == nil)
    }

    // MARK: - The store

    @Test("a checked-in cache comes back to the next turn of its session")
    func aCheckedInCacheComesBackToTheNextTurnOfItsSession() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        #expect(await store.checkOut(key("a")).entry?.tokens == [1, 2, 3])
    }

    @Test("a check-out takes the cache away, thus a second turn starts cold")
    func aCheckOutTakesTheCacheAwayThusASecondTurnStartsCold() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        _ = await store.checkOut(key("a"))

        #expect(await store.checkOut(key("a")) == .none)
    }

    @Test("checking in nothing releases the cache of that session")
    func checkingInNothingReleasesTheCacheOfThatSession() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        await store.checkIn(key("a"), nil)

        #expect(await store.checkOut(key("a")) == .none)
    }

    @Test("two sessions never read each other's cache")
    func twoSessionsNeverReadEachOthersCache() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))
        await store.checkIn(key("b"), entry(tokens: [7, 8]))

        #expect(await store.checkOut(key("a")).entry?.tokens == [1, 2, 3])
        #expect(await store.checkOut(key("b")).entry?.tokens == [7, 8])
    }

    @Test("the least recently used session leaves memory past the byte budget")
    func theLeastRecentlyUsedSessionLeavesMemoryPastTheByteBudget() async {
        let store = makeStore()
        let sessionsInBudget = Self.smallSessionsInBudget
        await store.configure(
            memoryBudgetBytes: sessionsInBudget * entry(tokens: [0]).byteCount)
        for session in 0 ... sessionsInBudget {
            await store.checkIn(key("session-\(session)"), entry(tokens: [session]))
        }

        #expect(await store.retainedSessionCount == sessionsInBudget)
        #expect(await store.peek(key("session-0")) == nil)
        #expect(
            await store.checkOut(key("session-\(sessionsInBudget)")).entry?.tokens
                == [sessionsInBudget])
    }

    @Test("evicting one model releases only the caches of that model")
    func evictingOneModelReleasesOnlyTheCachesOfThatModel() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1]))
        await store.checkIn(key("a", modelID: Self.otherModelID), entry(tokens: [2]))

        await store.evict(modelID: Self.modelID)

        #expect(await store.checkOut(key("a")) == .none)
        #expect(await store.checkOut(key("a", modelID: Self.otherModelID)).entry?.tokens == [2])
    }

    @Test("evicting every model releases every cache")
    func evictingEveryModelReleasesEveryCache() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1]))
        await store.checkIn(key("a", modelID: Self.otherModelID), entry(tokens: [2]))

        await store.evict(modelID: nil)

        #expect(await store.retainedSessionCount == 0)
    }

    // MARK: - Limiting the store by bytes

    /// How many small sessions the budget of the byte tests holds together.
    private static let smallSessionsInBudget = 3

    /// The token count of a large session. `KVCacheSimple` allocates its
    /// positions in steps of 256, thus this session holds four steps and a
    /// one-token session holds one.
    private static let largeSessionTokenCount = 1024

    /// A session that holds one allocation step of `KVCacheSimple`.
    private func smallEntry() -> ExecutorPromptCacheEntry {
        entry(tokens: [1])
    }

    /// A session that holds ``largeSessionTokenCount`` positions.
    private func largeEntry() -> ExecutorPromptCacheEntry {
        entry(tokens: Array(repeating: 1, count: Self.largeSessionTokenCount))
    }

    /// A store whose memory budget is `budget` bytes.
    private func store(budget: Int) async -> ExecutorPromptCacheStore {
        let store = makeStore()
        await store.configure(memoryBudgetBytes: budget)
        return store
    }

    @Test("an entry counts the bytes its caches hold")
    func anEntryCountsTheBytesItsCachesHold() {
        let caches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
        feed(caches, tokenCount: Self.largeSessionTokenCount)

        let entry = ExecutorPromptCacheEntry(
            caches: caches, tokens: Array(repeating: 1, count: Self.largeSessionTokenCount))

        #expect(entry.byteCount > 0)
        #expect(entry.byteCount == caches[0].residentByteCount + caches[1].residentByteCount)
    }

    @Test("an entry counts the arrays of the model state it carries")
    func anEntryCountsTheArraysOfTheModelStateItCarries() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 1)
        var state = LMOutput.State()
        let anchor = MLXArray.zeros([Self.largeSessionTokenCount])
        state[Self.arrayStateKey] = anchor

        let entry = ExecutorPromptCacheEntry(caches: caches, tokens: [1], state: state)

        #expect(entry.byteCount == caches[0].residentByteCount + anchor.nbytes)
    }

    /// A state key that holds an array, the way a VL model keeps an anchor.
    private static let arrayStateKey = LMOutput.Key<MLXArray>("test.array-anchor")

    @Test("the store never holds more bytes than its budget after a check-in")
    func theStoreNeverHoldsMoreBytesThanItsBudgetAfterACheckIn() async {
        let budget = largeEntry().byteCount + smallEntry().byteCount
        let store = await store(budget: budget)
        let entries = [largeEntry(), smallEntry(), largeEntry(), smallEntry(), largeEntry()]

        for (index, entry) in entries.enumerated() {
            await store.checkIn(key("session-\(index)"), entry)
            #expect(await store.retainedByteCount <= budget)
        }
    }

    @Test("many small sessions stay in memory together")
    func manySmallSessionsStayInMemoryTogether() async {
        let smallSessionsInOneLarge = largeEntry().byteCount / smallEntry().byteCount
        let store = await store(budget: largeEntry().byteCount)

        for session in 0 ..< smallSessionsInOneLarge {
            await store.checkIn(key("session-\(session)"), smallEntry())
        }

        #expect(await store.retainedSessionCount == smallSessionsInOneLarge)
        #expect(
            await store.retainedByteCount == smallSessionsInOneLarge * smallEntry().byteCount)
    }

    @Test("a few large sessions push each other out")
    func aFewLargeSessionsPushEachOtherOut() async {
        let store = await store(budget: largeEntry().byteCount + smallEntry().byteCount)

        await store.checkIn(key("first"), largeEntry())
        await store.checkIn(key("second"), largeEntry())

        #expect(await store.retainedSessionCount == 1)
        #expect(await store.peek(key("first")) == nil)
        #expect(await store.peek(key("second")) != nil)
    }

    @Test("the least recently used bytes leave first, and a check-out renews a session")
    func theLeastRecentlyUsedBytesLeaveFirstAndACheckOutRenewsASession() async {
        let store = await store(
            budget: Self.smallSessionsInBudget * smallEntry().byteCount)
        await store.checkIn(key("a"), smallEntry())
        await store.checkIn(key("b"), smallEntry())
        await store.checkIn(key("c"), smallEntry())

        // The turn of "a" takes its cache out and puts it back, thus "a" is
        // now the most recently used and "b" the least.
        let renewed = await store.checkOut(key("a")).entry
        await store.checkIn(key("a"), renewed)
        await store.checkIn(key("d"), smallEntry())

        #expect(await store.peek(key("b")) == nil)
        #expect(await store.peek(key("a")) != nil)
        #expect(await store.peek(key("c")) != nil)
        #expect(await store.peek(key("d")) != nil)
    }

    @Test("an entry larger than the whole budget is not kept, and evicts nothing")
    func anEntryLargerThanTheWholeBudgetIsNotKeptAndEvictsNothing() async {
        let small = smallEntry()
        let store = await store(budget: Self.smallSessionsInBudget * small.byteCount)
        await store.checkIn(key("small"), small)

        await store.checkIn(key("oversize"), largeEntry())

        #expect(await store.peek(key("oversize")) == nil)
        #expect(await store.peek(key("small")) != nil)
        #expect(await store.retainedByteCount == small.byteCount)
    }

    @Test("a lower budget evicts at once, least recently used first")
    func aLowerBudgetEvictsAtOnceLeastRecentlyUsedFirst() async {
        let store = await store(budget: largeEntry().byteCount)
        await store.checkIn(key("a"), smallEntry())
        await store.checkIn(key("b"), smallEntry())
        await store.checkIn(key("c"), smallEntry())

        await store.configure(memoryBudgetBytes: smallEntry().byteCount)

        #expect(await store.memoryBudgetBytes == smallEntry().byteCount)
        #expect(await store.retainedSessionCount == 1)
        #expect(await store.peek(key("c")) != nil)
        #expect(await store.retainedByteCount <= smallEntry().byteCount)
    }

    @Test("a higher budget evicts nothing")
    func aHigherBudgetEvictsNothing() async {
        let budget = Self.smallSessionsInBudget * smallEntry().byteCount
        let store = await store(budget: budget)
        await store.checkIn(key("a"), smallEntry())
        await store.checkIn(key("b"), smallEntry())
        let bytesBefore = await store.retainedByteCount

        await store.configure(memoryBudgetBytes: budget + largeEntry().byteCount)

        #expect(await store.retainedSessionCount == 2)
        #expect(await store.retainedByteCount == bytesBefore)
    }

    @Test("check-in, check-out and eviction keep the byte total correct")
    func checkInCheckOutAndEvictionKeepTheByteTotalCorrect() async {
        let store = await store(budget: largeEntry().byteCount * Self.smallSessionsInBudget)
        let large = largeEntry()
        let small = smallEntry()
        await store.checkIn(key("a"), large)
        await store.checkIn(key("b", modelID: Self.otherModelID), small)
        #expect(await store.retainedByteCount == large.byteCount + small.byteCount)

        // Checking in a new entry under a key already held replaces the old
        // bytes and does not add to them.
        await store.checkIn(key("a"), smallEntry())
        #expect(await store.retainedByteCount == small.byteCount * 2)

        _ = await store.checkOut(key("a"))
        #expect(await store.retainedByteCount == small.byteCount)

        await store.checkIn(key("a"), large)
        await store.evict(modelID: Self.modelID)
        #expect(await store.retainedByteCount == small.byteCount)

        await store.evict(modelID: nil)
        #expect(await store.retainedByteCount == 0)
    }

    @Test("the default budget is a quarter of the free working set")
    func theDefaultBudgetIsAQuarterOfTheFreeWorkingSet() {
        let workingSet = Self.largeSessionTokenCount * Self.smallSessionsInBudget
        let active = Self.largeSessionTokenCount

        #expect(
            ExecutorPromptCacheStore.defaultMemoryBudgetBytes(
                workingSet: workingSet, active: active)
                == (workingSet - active) / Self.quarter)
    }

    /// The divisor that takes a quarter of a value.
    private static let quarter = 4

    @Test("the default budget is zero when more memory is active than the working set")
    func theDefaultBudgetIsZeroWhenMoreMemoryIsActiveThanTheWorkingSet() {
        #expect(
            ExecutorPromptCacheStore.defaultMemoryBudgetBytes(
                workingSet: Self.largeSessionTokenCount,
                active: Self.largeSessionTokenCount * Self.smallSessionsInBudget) == 0)
        #expect(
            ExecutorPromptCacheStore.defaultMemoryBudgetBytes(workingSet: 0, active: Int.max)
                == 0)
    }

    @Test("a store with no budget from the host sets one from the device")
    func aStoreWithNoBudgetFromTheHostSetsOneFromTheDevice() async {
        // The test process holds far less active memory than the working set
        // of the device, thus a quarter of the free part is more than zero.
        #expect(await makeStore().memoryBudgetBytes > 0)
    }

    @Test("the eviction line names the session and its bytes")
    func theEvictionLineNamesTheSessionAndItsBytes() {
        #expect(
            ExecutorPromptCacheReport.evictionLine(
                key: key("session-1"), byteCount: Self.largeSessionTokenCount)
                == "prompt cache evict model=test/prompt-cache session=session-1 bytes=1024")
    }

    // MARK: - The store a task binds

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("an executor pass inside a bound store uses that store and not the shared one")
    func anExecutorPassInsideABoundStoreUsesThatStore() async throws {
        let modelID = "probe/bound-prompt-cache-store-\(UUID().uuidString)"
        let sessionKey = key("bound-session", modelID: modelID)
        let store = makeStore()
        await store.checkIn(sessionKey, entry(tokens: [1, 2, 3]))

        try await ScriptedExecutorPass.respondOnce(
            inside: store, modelID: modelID, sessionID: "bound-session")

        // The pass checked the seeded entry out of the bound store. The
        // scripted model holds no key/value cache, thus the pass checked
        // nothing back in.
        #expect(await store.peek(sessionKey) == nil)
        #expect(await ExecutorPromptCacheStore.shared.peek(sessionKey) == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("evicting a model inside a bound store releases the caches of that store")
    func evictingAModelInsideABoundStoreReleasesTheCachesOfThatStore() async {
        let model = makeStubModel("probe/bound-evict-\(UUID().uuidString)")
        let store = makeStore()
        await store.checkIn(key("a", modelID: model.modelID), smallEntry())

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.evict()
        }

        #expect(await store.retainedSessionCount == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("an eviction in another task leaves the caches of a store this task binds")
    func anEvictionInAnotherTaskLeavesTheCachesOfABoundStore() async {
        let model = makeStubModel("probe/foreign-evict-\(UUID().uuidString)")
        let sessionKey = key("a", modelID: model.modelID)
        let store = makeStore()
        await ExecutorPromptCacheStore.shared.checkIn(sessionKey, smallEntry())

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await ExecutorPromptCacheStore.current.checkIn(sessionKey, smallEntry())
            // A detached task does not inherit the binding, as a parallel suite does not.
            await Task.detached { await model.evict() }.value
        }

        // The eviction ran: it released the entry of the shared store, and only that one.
        #expect(await ExecutorPromptCacheStore.shared.peek(sessionKey) == nil)
        #expect(await store.peek(sessionKey) != nil)
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

    // MARK: - Reading the store without a check-out

    @Test("a peek reads the entry of a session and leaves it in the store")
    func aPeekReadsTheEntryOfASessionAndLeavesItInTheStore() async {
        let store = makeStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        #expect(await store.peek(key("a"))?.tokens == [1, 2, 3])
        #expect(await store.retainedSessionCount == 1)
        #expect(await store.checkOut(key("a")).entry?.tokens == [1, 2, 3])
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
