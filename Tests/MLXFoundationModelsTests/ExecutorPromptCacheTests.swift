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

    /// A finished pass over `caches` that rendered `promptTokens` and fed the
    /// whole render, thus the caches represent the render once it is fed.
    private func plan(caches: [KVCache], promptTokens: [Int]) -> ExecutorPromptCachePlan {
        ExecutorPromptCachePlan(
            caches: caches,
            input: LMInput(tokens: MLXArray(promptTokens)),
            reusedTokenCount: 0,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: nil)
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
