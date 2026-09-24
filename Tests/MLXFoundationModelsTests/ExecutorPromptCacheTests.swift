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
struct ExecutorPromptCacheTests {

    /// The model every key of this suite names.
    private static let modelID = "test/prompt-cache"

    /// A second model, to prove that two models never share one cache.
    private static let otherModelID = "test/prompt-cache-other"

    /// An entry that names `tokens` and carries one cache fed with one
    /// position for each token, thus the entry holds real bytes.
    private func entry(tokens: [Int]) -> ExecutorPromptCacheEntry {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: tokens.count)
        return ExecutorPromptCacheEntry(caches: caches, tokens: tokens)
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

    @Test("the least recently used session loses its cache past the byte budget")
    func theLeastRecentlyUsedSessionLosesItsCachePastTheByteBudget() async {
        let store = ExecutorPromptCacheStore()
        let sessionsInBudget = Self.smallSessionsInBudget
        await store.configure(
            memoryBudgetBytes: sessionsInBudget * entry(tokens: [0]).byteCount)
        for session in 0 ... sessionsInBudget {
            await store.checkIn(key("session-\(session)"), entry(tokens: [session]))
        }

        #expect(await store.retainedSessionCount == sessionsInBudget)
        #expect(await store.checkOut(key("session-0")) == nil)
        #expect(
            await store.checkOut(key("session-\(sessionsInBudget)"))?.tokens
                == [sessionsInBudget])
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
        let store = ExecutorPromptCacheStore()
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
        let renewed = await store.checkOut(key("a"))
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
        #expect(await ExecutorPromptCacheStore().memoryBudgetBytes > 0)
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
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        // A fresh identity keeps the process-wide model cache and the shared
        // store out of every other test.
        let modelID = "probe/bound-prompt-cache-store-\(UUID().uuidString)"
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [],
            weightsLocation: { _ in weights },
            load: { _, _ in makeScriptedContainer(modelID: modelID, rounds: ["A"]) })
        let executor = try makeMLXExecutor(for: model)
        let request = makeRequest(transcript: transcript(firstEntryID: "bound-session"))
        let sessionKey = key("bound-session", modelID: modelID)
        let store = ExecutorPromptCacheStore()
        await store.checkIn(sessionKey, entry(tokens: [1, 2, 3]))
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside the
        // executor or every send parks it.
        let consumer = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { consumer.cancel() }

        try await ExecutorPromptCacheStore.$current.withValue(store) {
            try await executor.respond(to: request, model: model, streamingInto: channel)
        }

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
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a", modelID: model.modelID), smallEntry())

        await ExecutorPromptCacheStore.$current.withValue(store) {
            await model.evict()
        }

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

    /// A finished pass over `caches` that rendered `promptTokens` and fed the
    /// whole render, thus the caches represent the render once it is fed.
    private func plan(caches: [KVCache], promptTokens: [Int]) -> ExecutorPromptCachePlan {
        ExecutorPromptCachePlan(
            caches: caches,
            input: LMInput(tokens: MLXArray(promptTokens)),
            reusedTokenCount: 0,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: nil,
            decision: .cold)
    }

    /// A recurrent cache placed at `position`, the way a Qwen 3.5 linear
    /// layer places it after `position` tokens.
    private func recurrentCache(at position: Int) -> MambaCache {
        let cache = MambaCache()
        cache.advancePosition(by: position)
        return cache
    }

    @Test("a hybrid stack whose caches agree on their position commits to a ledger")
    func aHybridStackWhoseCachesAgreeOnTheirPositionCommitsToALedger() {
        // One recurrent cache beside one attention cache, both past the render
        // and the generated tokens: the ledger names both.
        let promptTokens = [1, 2, 3, 4]
        let generatedTokens = [101, 102]
        let attention = KVCacheSimple()
        feed([attention], tokenCount: promptTokens.count + generatedTokens.count)
        let caches: [KVCache] = [
            recurrentCache(at: promptTokens.count + generatedTokens.count), attention,
        ]

        let committed = plan(caches: caches, promptTokens: promptTokens)
            .committed(generatedTokens: generatedTokens)

        #expect(!canTrimPromptCache(caches), "the premise: a recurrent cache cannot rewind")
        #expect(committed?.tokens == promptTokens + generatedTokens)
    }

    @Test("a recurrent cache that reports no position leaves the session cold")
    func aRecurrentCacheThatReportsNoPositionLeavesTheSessionCold() {
        // The defect of card ^xx5g893: a linear layer that never moves its
        // cache's position leaves the recurrent caches at zero while the
        // attention caches stand past the prompt. The two cannot share one
        // ledger, thus the session starts cold. Every cache, recurrent or
        // attention, must report the position.
        let promptTokens = [1, 2, 3, 4]
        let attention = KVCacheSimple()
        feed([attention], tokenCount: promptTokens.count + 1)

        #expect(
            plan(caches: [recurrentCache(at: 0), attention], promptTokens: promptTokens)
                .committed(generatedTokens: [101]) == nil)
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

    // MARK: - Planning one pass over a processor's input

    /// `tokens` the way a VLM processor batches a text-only prompt: one row.
    private func oneRow(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens).expandedDimensions(axis: 0)
    }

    /// A text-only input of one batched row, masked with `mask`, or with a
    /// mask that marks every token present when `mask` is nil.
    ///
    /// This is the input `MuseGlimmerProcessor.prepare(input:)` gives for a
    /// prompt that carries no image.
    private func batchedInput(_ tokens: [Int], mask: [Int]? = nil) -> LMInput {
        let tokenArray = oneRow(tokens)
        let maskArray = mask.map { oneRow($0) } ?? ones(like: tokenArray)
        return LMInput(text: .init(tokens: tokenArray, mask: maskArray.asType(.int8)))
    }

    /// Plans `input` against an entry whose caches hold `cachedTokens`, or
    /// against no entry when `cachedTokens` is nil.
    ///
    /// The model owns no attention layer, thus a fresh cache is empty, which
    /// is all a plan needs here.
    private func plan(
        _ input: LMInput, cachedTokens: [Int]? = nil
    ) throws -> ExecutorPromptCachePlan? {
        let entry = cachedTokens.map { tokens -> ExecutorPromptCacheEntry in
            let caches: [KVCache] = [KVCacheSimple()]
            feed(caches, tokenCount: tokens.count)
            return ExecutorPromptCacheEntry(caches: caches, tokens: tokens)
        }
        return try ExecutorPromptCachePlan.make(
            reusing: entry, input: input, model: ScriptedLanguageModel(rounds: []),
            parameters: GenerateParameters())
    }

    @Test("a text-only prompt a VLM processor batched to one row gets a plan")
    func aTextOnlyPromptAVLMProcessorBatchedToOneRowGetsAPlan() throws {
        let planned = try plan(batchedInput([1, 2, 3]))

        #expect(planned?.promptTokens == [1, 2, 3])
        #expect(planned?.reusedTokenCount == 0)
    }

    @Test("that plan reuses the prefix an earlier turn left, in the shape the model expects")
    func thatPlanReusesThePrefixAnEarlierTurnLeftInTheShapeTheModelExpects() throws {
        let planned = try #require(
            try plan(batchedInput([1, 2, 3, 4, 5]), cachedTokens: [1, 2, 3]))

        #expect(planned.reusedTokenCount == 3)
        #expect(planned.input.text.tokens.shape == [1, 2])
        #expect(planned.input.text.tokens.asArray(Int.self) == [4, 5])
        #expect(planned.input.text.mask?.shape == [1, 2])
        #expect(planned.input.text.mask?.dtype == .int8)
        #expect(planned.input.text.mask?.asArray(Int.self) == [1, 1])
    }

    @Test("a mask that holds a zero gets no plan")
    func aMaskThatHoldsAZeroGetsNoPlan() throws {
        #expect(try plan(batchedInput([1, 2, 3], mask: [1, 1, 0])) == nil)
    }

    @Test("a batch of more than one row gets no plan")
    func aBatchOfMoreThanOneRowGetsNoPlan() throws {
        let twoRows = MLXArray([1, 2, 3, 4, 5, 6]).reshaped([2, 3])
        let input = LMInput(text: .init(tokens: twoRows, mask: ones(like: twoRows)))

        #expect(try plan(input) == nil)
    }

    @Test("an image keeps a prompt out of the plan")
    func anImageKeepsAPromptOutOfThePlan() throws {
        let text = batchedInput([1, 2, 3]).text
        let input = LMInput(text: text, image: .init(pixels: MLXArray.zeros([1, 1])))

        #expect(try plan(input) == nil)
    }

    // MARK: - Planning one pass with a protocol rule

    /// The commit token of the splicing fixtures below.
    private static let commit = 9

    /// A rule that keeps the tokens the model wrote and feeds the tail of the
    /// render after its commit, the way a committed-turn rule does.
    private struct SplicingRule: PromptCacheReuseRule {
        func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
            guard turn.promptTokens.starts(with: cache.previousRenderTokens),
                let commitIndex = turn.promptTokens.firstIndex(
                    of: ExecutorPromptCacheTests.commit)
            else { return nil }
            let suffixStart = commitIndex + 1
            return .appendSuffix(
                suffixStart: suffixStart,
                representedTokens: cache.cachedTokens + turn.promptTokens[suffixStart...])
        }
    }

    @Test("a protocol rule splices the render's tail onto the tokens the model wrote")
    func aProtocolRuleSplicesTheRendersTailOntoTheTokensTheModelWrote() throws {
        // The last pass rendered [1, 2] and the model wrote [70, commit]. The
        // new render writes 71 where the model wrote 70, and a recurrent cache
        // cannot rewind, thus only the rule can serve this turn.
        let ledger = [1, 2, 70, Self.commit]
        let caches: [KVCache] = [recurrentCache(at: ledger.count)]
        let entry = ExecutorPromptCacheEntry(caches: caches, tokens: ledger, renderTokens: [1, 2])
        let render = [1, 2, 71, Self.commit, 20, 21]

        let planned = try #require(
            try ExecutorPromptCachePlan.make(
                reusing: entry, input: LMInput(tokens: MLXArray(render)),
                model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters(),
                protocolRules: [SplicingRule()]))

        #expect(planned.reusedTokenCount == 4)
        #expect(planned.input.text.tokens.asArray(Int.self) == [20, 21])
        #expect(planned.representedTokens == [1, 2, 70, Self.commit, 20, 21])
    }

    @Test("a spliced pass commits the tokens the model wrote, not the render")
    func aSplicedPassCommitsTheTokensTheModelWroteNotTheRender() throws {
        let ledger = [1, 2, 70, Self.commit]
        let recurrent = recurrentCache(at: ledger.count)
        let entry = ExecutorPromptCacheEntry(
            caches: [recurrent], tokens: ledger, renderTokens: [1, 2])
        let render = [1, 2, 71, Self.commit, 20, 21]
        let planned = try #require(
            try ExecutorPromptCachePlan.make(
                reusing: entry, input: LMInput(tokens: MLXArray(render)),
                model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters(),
                protocolRules: [SplicingRule()]))
        // The pass feeds the two-token tail and generates one token.
        recurrent.advancePosition(by: 2 + 1)

        let committed = planned.committed(generatedTokens: [30])

        #expect(committed?.tokens == [1, 2, 70, Self.commit, 20, 21, 30])
        #expect(committed?.renderTokens == render)
    }

    @Test("a plan without a rule records the render it fed for the next pass")
    func aPlanWithoutARuleRecordsTheRenderItFedForTheNextPass() throws {
        let caches: [KVCache] = [KVCacheSimple()]
        let promptTokens = [1, 2, 3]
        feed(caches, tokenCount: promptTokens.count)

        let committed = plan(caches: caches, promptTokens: promptTokens)
            .committed(generatedTokens: [])

        #expect(committed?.renderTokens == promptTokens)
    }

    // MARK: - Carrying model state

    /// The state key of the fixtures below, the way a Qwen 3.5 VL model keys
    /// its M-RoPE anchor.
    private static let anchorKey = LMOutput.Key<Int>("test.anchor")

    /// A state that holds `anchor` under ``anchorKey``.
    private func state(anchor: Int) -> LMOutput.State {
        var state = LMOutput.State()
        state[Self.anchorKey] = anchor
        return state
    }

    /// An entry whose caches hold [1, 2, 3] and whose state holds `anchor`.
    private func anchoredEntry(anchor: Int) -> ExecutorPromptCacheEntry {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 3)
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: [1, 2, 3], state: state(anchor: anchor))
    }

    @Test("a plan seeds the model state the entry carries")
    func aPlanSeedsTheModelStateTheEntryCarries() throws {
        let planned = try #require(
            try ExecutorPromptCachePlan.make(
                reusing: anchoredEntry(anchor: 7), input: LMInput(tokens: MLXArray([1, 2, 3, 4])),
                model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters()))

        #expect(planned.reusedTokenCount == 3)
        #expect(planned.state?[Self.anchorKey] == 7)
    }

    @Test("a carried model state keeps the caches from a rewind")
    func aCarriedModelStateKeepsTheCachesFromARewind() throws {
        // The state is anchored to the prefill that made it, thus a render
        // that rewrites a cached token gets fresh caches and no state.
        let entry = anchoredEntry(anchor: 7)

        let planned = try #require(
            try ExecutorPromptCachePlan.make(
                reusing: entry, input: LMInput(tokens: MLXArray([1, 2, 9])),
                model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters()))

        #expect(planned.reusedTokenCount == 0)
        #expect(planned.state == nil)
        #expect(entry.caches.allSatisfy { $0.offset == 3 })
    }

    @Test("a commit keeps the state the prefill left for the next turn")
    func aCommitKeepsTheStateThePrefillLeftForTheNextTurn() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 3)

        let committed = plan(caches: caches, promptTokens: [1, 2, 3])
            .committed(generatedTokens: [], state: state(anchor: 11))

        #expect(committed?.state?[Self.anchorKey] == 11)
    }

    @Test("a slot hands the prepared state to the entry it commits")
    func aSlotHandsThePreparedStateToTheEntryItCommits() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 3)
        let slot = ExecutorPromptCacheSlot(nil)

        slot.commit(
            plan(caches: caches, promptTokens: [1, 2, 3]), generatedTokens: [],
            state: state(anchor: 5))

        #expect(slot.entry?.state?[Self.anchorKey] == 5)
    }

    // MARK: - Reading the store without a check-out

    @Test("a peek reads the entry of a session and leaves it in the store")
    func aPeekReadsTheEntryOfASessionAndLeavesItInTheStore() async {
        let store = ExecutorPromptCacheStore()
        await store.checkIn(key("a"), entry(tokens: [1, 2, 3]))

        #expect(await store.peek(key("a"))?.tokens == [1, 2, 3])
        #expect(await store.retainedSessionCount == 1)
        #expect(await store.checkOut(key("a"))?.tokens == [1, 2, 3])
    }

    // MARK: - Naming the rule that decided a pass

    /// An entry whose one recurrent cache stands at the end of `ledger`, thus
    /// the caches cannot rewind and a render that parts from the ledger
    /// rebuilds.
    private func recurrentEntry(ledger: [Int]) -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(caches: [recurrentCache(at: ledger.count)], tokens: ledger)
    }

    /// Plans `render` against `entry` with no protocol rule.
    private func plan(
        render: [Int], reusing entry: ExecutorPromptCacheEntry?
    ) throws -> ExecutorPromptCachePlan? {
        try ExecutorPromptCachePlan.make(
            reusing: entry, input: LMInput(tokens: MLXArray(render)),
            model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters())
    }

    @Test("a pass with no carried entry names the cold rule")
    func aPassWithNoCarriedEntryNamesTheColdRule() throws {
        #expect(try plan(render: [1, 2, 3], reusing: nil)?.decision == .cold)
    }

    @Test("a render that extends the ledger names the extend rule")
    func aRenderThatExtendsTheLedgerNamesTheExtendRule() throws {
        let planned = try plan(batchedInput([1, 2, 3, 4]), cachedTokens: [1, 2, 3])

        #expect(planned?.decision == .extend)
    }

    @Test("a protocol rule that splices names the splice rule")
    func aProtocolRuleThatSplicesNamesTheSpliceRule() throws {
        let ledger = [1, 2, 70, Self.commit]
        let entry = ExecutorPromptCacheEntry(
            caches: [recurrentCache(at: ledger.count)], tokens: ledger, renderTokens: [1, 2])

        let planned = try ExecutorPromptCachePlan.make(
            reusing: entry, input: LMInput(tokens: MLXArray([1, 2, 71, Self.commit, 20, 21])),
            model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters(),
            protocolRules: [SplicingRule()])

        #expect(planned?.decision == .splice)
    }

    @Test("a rewind names the rewind rule and the seam it rewound to")
    func aRewindNamesTheRewindRuleAndTheSeamItRewoundTo() throws {
        let planned = try plan(batchedInput([1, 2, 9, 9]), cachedTokens: [1, 2, 3, 4, 5])

        #expect(
            planned?.decision
                == .rewind(
                    ExecutorPromptCacheDivergence(
                        index: 2, renderTokens: [9, 9], ledgerTokens: [3, 4, 5])))
    }

    @Test("a rebuild names the rebuild rule and the seam the caches could not rewind to")
    func aRebuildNamesTheRebuildRuleAndTheSeamTheCachesCouldNotRewindTo() throws {
        let planned = try plan(
            render: [1, 2, 9, 9], reusing: recurrentEntry(ledger: [1, 2, 3, 4]))

        #expect(planned?.reusedTokenCount == 0)
        #expect(
            planned?.decision
                == .rebuild(
                    ExecutorPromptCacheDivergence(
                        index: 2, renderTokens: [9, 9], ledgerTokens: [3, 4])))
    }

    @Test("a seam keeps a short window of tokens on each side")
    func aSeamKeepsAShortWindowOfTokensOnEachSide() {
        let window = ExecutorPromptCacheDivergence.reportedTokenCount
        let shared = [1, 2]
        let render = shared + Array(repeating: 8, count: window * 2)
        let ledger = shared + Array(repeating: 9, count: window * 2)

        let seam = ExecutorPromptCacheDivergence(render: render, ledger: ledger)

        #expect(seam.index == shared.count)
        #expect(seam.renderTokens == Array(repeating: 8, count: window))
        #expect(seam.ledgerTokens == Array(repeating: 9, count: window))
    }

    @Test("a seam at the end of one side keeps no token on that side")
    func aSeamAtTheEndOfOneSideKeepsNoTokenOnThatSide() {
        let seam = ExecutorPromptCacheDivergence(render: [1, 2], ledger: [1, 2, 3])

        #expect(
            seam == ExecutorPromptCacheDivergence(index: 2, renderTokens: [], ledgerTokens: [3]))
    }

    // MARK: - Why a finished pass checks nothing in

    @Test("caches that disagree on their position name every position")
    func cachesThatDisagreeOnTheirPositionNameEveryPosition() {
        let leading = KVCacheSimple()
        let lagging = KVCacheSimple()
        let promptTokens = [1, 2, 3, 4]
        feed([leading], tokenCount: promptTokens.count + 1)
        feed([lagging], tokenCount: promptTokens.count)

        let outcome = plan(caches: [leading, lagging], promptTokens: promptTokens)
            .commitOutcome(generatedTokens: [101])

        #expect(outcome.refusal == .positionsDisagree([5, 4]))
    }

    @Test("caches behind the ledger name their position and the ledger length")
    func cachesBehindTheLedgerNameTheirPositionAndTheLedgerLength() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 3)

        let outcome = plan(caches: caches, promptTokens: [1, 2, 3, 4])
            .commitOutcome(generatedTokens: [])

        #expect(outcome.refusal == .behindTheLedger(position: 3, ledgerLength: 4))
    }

    @Test("caches past the generation name what the pass generated")
    func cachesPastTheGenerationNameWhatThePassGenerated() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 7)

        let outcome = plan(caches: caches, promptTokens: [1, 2, 3, 4])
            .commitOutcome(generatedTokens: [101])

        #expect(
            outcome.refusal
                == .pastTheGeneration(position: 7, ledgerLength: 4, generatedTokenCount: 1))
    }

    @Test("a pass with no cache at all names that")
    func aPassWithNoCacheAtAllNamesThat() {
        let outcome = plan(caches: [], promptTokens: [1, 2]).commitOutcome(generatedTokens: [])

        #expect(outcome.refusal == .noCaches)
    }

    @Test("a good commit is checked in and refuses nothing")
    func aGoodCommitIsCheckedInAndRefusesNothing() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 4)

        let outcome = plan(caches: caches, promptTokens: [1, 2, 3])
            .commitOutcome(generatedTokens: [101])

        #expect(outcome.refusal == nil)
        #expect(outcome.entry?.tokens == [1, 2, 3, 101])
    }

    // MARK: - The log line of one pass

    /// Decodes a token window the way the report tests read it: each token
    /// as its number, separated by one space.
    private func decodeAsNumbers(_ tokens: [Int]) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    /// The session every report line of this section names.
    private var reportedKey: ExecutorPromptCacheKey { key("session-1") }

    @Test("the plan line of an extension names the counts and the rule")
    func thePlanLineOfAnExtensionNamesTheCountsAndTheRule() throws {
        let planned = try plan(batchedInput([1, 2, 3, 4]), cachedTokens: [1, 2, 3])

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "rendered=4 reused=3 fed=1 rule=extend")
    }

    @Test("the plan line of a rebuild names the seam and decodes each side")
    func thePlanLineOfARebuildNamesTheSeamAndDecodesEachSide() throws {
        let planned = try plan(
            render: [1, 2, 9, 9], reusing: recurrentEntry(ledger: [1, 2, 3, 4]))

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "rendered=4 reused=0 fed=4 rule=rebuild divergence=2 "
                + "render=<<<9 9>>> ledger=<<<3 4>>>")
    }

    @Test("the plan line of a rewind names the seam it rewound to")
    func thePlanLineOfARewindNamesTheSeamItRewoundTo() throws {
        let planned = try plan(batchedInput([1, 2, 9]), cachedTokens: [1, 2, 3, 4])

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "rendered=3 reused=2 fed=1 rule=rewind divergence=2 "
                + "render=<<<9>>> ledger=<<<3 4>>>")
    }

    @Test("the plan line of a cold pass names the cold rule")
    func thePlanLineOfAColdPassNamesTheColdRule() throws {
        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, plan: try plan(render: [1, 2, 3], reusing: nil),
            decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "rendered=3 reused=0 fed=3 rule=cold")
    }

    @Test("the plan line of a pass with no plan says why")
    func thePlanLineOfAPassWithNoPlanSaysWhy() {
        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, plan: nil, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "rule=none (the input carries media, a batch or a mask)")
    }

    @Test("a pass with no session key names no session")
    func aPassWithNoSessionKeyNamesNoSession() {
        let line = ExecutorPromptCacheReport.planLine(
            key: nil, plan: nil, decodeTokens: decodeAsNumbers)

        #expect(line.hasPrefix("prompt cache plan model=none session=none "))
    }

    @Test("the commit line names the ledger length checked in")
    func theCommitLineNamesTheLedgerLengthCheckedIn() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 4)
        let outcome = plan(caches: caches, promptTokens: [1, 2, 3])
            .commitOutcome(generatedTokens: [101])

        let line = ExecutorPromptCacheReport.commitLine(key: reportedKey, outcome: outcome)

        #expect(
            line == "prompt cache commit model=test/prompt-cache session=session-1 ledger=4")
    }

    @Test("the commit line names why nothing was checked in")
    func theCommitLineNamesWhyNothingWasCheckedIn() {
        let prefix = "prompt cache commit model=test/prompt-cache session=session-1 "
        let lines = [
            ExecutorPromptCacheCommitRefusal.noPlan,
            .noCaches,
            .positionsDisagree([5, 0]),
            .behindTheLedger(position: 3, ledgerLength: 4),
            .pastTheGeneration(position: 7, ledgerLength: 4, generatedTokenCount: 1),
        ].map { ExecutorPromptCacheReport.commitLine(key: reportedKey, outcome: .refused($0)) }

        #expect(
            lines == [
                prefix + "checked in nothing: the pass carried no plan",
                prefix + "checked in nothing: the pass carried no cache",
                prefix + "checked in nothing: the caches disagree on their position [5, 0]",
                prefix + "checked in nothing: the caches stand at 3, behind the 4-token ledger",
                prefix + "checked in nothing: the caches stand at 7, past the 4-token ledger "
                    + "and the 1 generated tokens",
            ])
    }

    @Test("a slot reports one plan line and one commit line for each pass")
    func aSlotReportsOnePlanLineAndOneCommitLineForEachPass() throws {
        var lines: [String] = []
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: 3)
        let entry = ExecutorPromptCacheEntry(caches: caches, tokens: [1, 2, 3])
        let slot = ExecutorPromptCacheSlot(entry, key: reportedKey) { lines.append($0) }

        let planned = try slot.plan(
            input: LMInput(tokens: MLXArray([1, 2, 3, 4])),
            model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters(),
            decodeTokens: decodeAsNumbers)
        feed(caches, tokenCount: 1)
        slot.commit(planned, generatedTokens: [])

        #expect(
            lines == [
                "prompt cache plan model=test/prompt-cache session=session-1 "
                    + "rendered=4 reused=3 fed=1 rule=extend",
                "prompt cache commit model=test/prompt-cache session=session-1 ledger=4",
            ])
    }

    @Test("a guided pass reports that it owns its cache")
    func aGuidedPassReportsThatItOwnsItsCache() {
        var lines: [String] = []
        let slot = ExecutorPromptCacheSlot(nil, key: reportedKey) { lines.append($0) }

        slot.carriesNoCache()
        slot.commit(nil, generatedTokens: [])

        #expect(
            lines == [
                "prompt cache plan model=test/prompt-cache session=session-1 "
                    + "rule=guided (the guided pass owns its cache and carries none)",
                "prompt cache commit model=test/prompt-cache session=session-1 "
                    + "checked in nothing: the pass carried no plan",
            ])
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
