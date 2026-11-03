// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Unit tests for the plan of one pass over the prompt cache of a session:
/// the ledger a finished pass leaves, the rule that decides each pass, the
/// model state an entry carries, and the log lines of each pass.
///
/// No weights are needed. The entries carry small zero-filled caches and a
/// token ledger alone.
@Suite("A pass plans its prompt cache and commits a ledger")
struct ExecutorPromptCachePlanTests: PromptCacheKeyFixtures, PromptCachePlanFixtures {

    /// The model every key of this suite names.
    static let modelID = "test/prompt-cache"

    // MARK: - The ledger a finished pass leaves

    /// How many positions the rotating cache of this suite holds. A prompt
    /// longer than this stands past the window, thus the cache no longer
    /// rewinds.
    private static let slidingWindow = 8

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
        cache.advance(position)
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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func plan(
        _ input: LMInput, cachedTokens: [Int]? = nil
    ) throws -> ExecutorPromptCachePlan? {
        try ExecutorPromptCachePlan.make(
            reusing: cachedTokens.map { entry(tokens: $0) }, input: input,
            model: ScriptedLanguageModel(rounds: []), parameters: GenerateParameters())
    }

    @Test("a text-only prompt a VLM processor batched to one row gets a plan")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aTextOnlyPromptAVLMProcessorBatchedToOneRowGetsAPlan() throws {
        let planned = try plan(batchedInput([1, 2, 3]))

        #expect(planned?.promptTokens == [1, 2, 3])
        #expect(planned?.reusedTokenCount == 0)
    }

    @Test("that plan reuses the prefix an earlier turn left, in the shape the model expects")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aMaskThatHoldsAZeroGetsNoPlan() throws {
        #expect(try plan(batchedInput([1, 2, 3], mask: [1, 1, 0])) == nil)
    }

    @Test("a batch of more than one row gets no plan")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aBatchOfMoreThanOneRowGetsNoPlan() throws {
        let twoRows = MLXArray([1, 2, 3, 4, 5, 6]).reshaped([2, 3])
        let input = LMInput(text: .init(tokens: twoRows, mask: ones(like: twoRows)))

        #expect(try plan(input) == nil)
    }

    @Test("an image keeps a prompt out of the plan")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anImageKeepsAPromptOutOfThePlan() throws {
        let text = batchedInput([1, 2, 3]).text
        let input = LMInput(text: text, image: .init(pixels: MLXArray.zeros([1, 1])))

        #expect(try plan(input) == nil)
    }

    // MARK: - Planning one pass with a protocol rule

    /// The commit token of the splicing fixtures below.
    private static let commit = 9

    /// The render of the splicing fixtures: the last pass rendered [1, 2] and
    /// the model wrote [70, commit]; this render writes 71 where the model
    /// wrote 70, and adds a tail after the commit.
    private static let spliceRender = [1, 2, 71, commit, 20, 21]

    /// The entry of the splicing fixtures: the ledger `[1, 2, 70, commit]` in
    /// one recurrent cache, which cannot rewind, and the render `[1, 2]` of
    /// the last pass.
    private func spliceFixture() -> ExecutorPromptCacheEntry {
        let ledger = [1, 2, 70, Self.commit]
        return ExecutorPromptCacheEntry(
            caches: [recurrentCache(at: ledger.count)], tokens: ledger, renderTokens: [1, 2])
    }

    @Test("a protocol rule splices the render's tail onto the tokens the model wrote")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aProtocolRuleSplicesTheRendersTailOntoTheTokensTheModelWrote() throws {
        // The new render writes 71 where the model wrote 70, and a recurrent
        // cache cannot rewind, thus only the rule can serve this turn.
        let planned = try #require(
            try planPromptCache(
                render: Self.spliceRender, reusing: spliceFixture(),
                protocolRules: [SplicingRule(commit: Self.commit)]))

        #expect(planned.reusedTokenCount == 4)
        #expect(planned.input.text.tokens.asArray(Int.self) == [20, 21])
        #expect(planned.representedTokens == [1, 2, 70, Self.commit, 20, 21])
    }

    @Test("a spliced pass commits the tokens the model wrote, not the render")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSplicedPassCommitsTheTokensTheModelWroteNotTheRender() throws {
        let entry = spliceFixture()
        let recurrent = try #require(entry.caches.first as? MambaCache)
        let planned = try #require(
            try planPromptCache(
                render: Self.spliceRender, reusing: entry,
                protocolRules: [SplicingRule(commit: Self.commit)]))
        // The pass feeds the two-token tail and generates one token.
        recurrent.advance(2 + 1)

        let committed = planned.committed(generatedTokens: [30])

        #expect(committed?.tokens == [1, 2, 70, Self.commit, 20, 21, 30])
        #expect(committed?.renderTokens == Self.spliceRender)
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
        entry(tokens: [1, 2, 3], state: state(anchor: anchor))
    }

    @Test("a plan seeds the model state the entry carries")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPlanSeedsTheModelStateTheEntryCarries() throws {
        let planned = try #require(
            try planPromptCache(render: [1, 2, 3, 4], reusing: anchoredEntry(anchor: 7)))

        #expect(planned.reusedTokenCount == 3)
        #expect(planned.state?[Self.anchorKey] == 7)
    }

    @Test("a carried model state keeps the caches from a rewind")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCarriedModelStateKeepsTheCachesFromARewind() throws {
        // The state is anchored to the prefill that made it, thus a render
        // that rewrites a cached token gets fresh caches and no state.
        let entry = anchoredEntry(anchor: 7)

        let planned = try #require(try planPromptCache(render: [1, 2, 9], reusing: entry))

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

    // MARK: - Naming the rule that decided a pass

    /// An entry whose one recurrent cache stands at the end of `ledger`, thus
    /// the caches cannot rewind and a render that parts from the ledger
    /// rebuilds.
    private func recurrentEntry(ledger: [Int]) -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(caches: [recurrentCache(at: ledger.count)], tokens: ledger)
    }

    @Test("a pass with no carried entry names the cold rule")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassWithNoCarriedEntryNamesTheColdRule() throws {
        #expect(try planPromptCache(render: [1, 2, 3], reusing: nil)?.decision == .cold)
    }

    @Test("a render that extends the ledger names the extend rule")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatExtendsTheLedgerNamesTheExtendRule() throws {
        let planned = try plan(batchedInput([1, 2, 3, 4]), cachedTokens: [1, 2, 3])

        #expect(planned?.decision == .extend)
    }

    @Test("a protocol rule that splices names the splice rule")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aProtocolRuleThatSplicesNamesTheSpliceRule() throws {
        let planned = try planPromptCache(
            render: Self.spliceRender, reusing: spliceFixture(),
            protocolRules: [SplicingRule(commit: Self.commit)])

        #expect(planned?.decision == .splice)
    }

    @Test("a rewind names the rewind rule and the seam it rewound to")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRewindNamesTheRewindRuleAndTheSeamItRewoundTo() throws {
        let planned = try plan(batchedInput([1, 2, 9, 9]), cachedTokens: [1, 2, 3, 4, 5])

        #expect(
            planned?.decision
                == .rewind(
                    ExecutorPromptCacheDivergence(
                        index: 2, renderTokens: [9, 9], ledgerTokens: [3, 4, 5])))
    }

    @Test("a rebuild names the rebuild rule and the seam the caches could not rewind to")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRebuildNamesTheRebuildRuleAndTheSeamTheCachesCouldNotRewindTo() throws {
        let planned = try planPromptCache(
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

    /// The restore time, in milliseconds, that the plan line of a restored pass reports.
    private static let reportedRestoreMilliseconds = 1_250

    /// The restore time the plan line of a restored pass reports.
    private static let reportedRestoreDuration: Duration = .milliseconds(
        reportedRestoreMilliseconds)

    @Test("the plan line of an extension names the counts and the rule")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfAnExtensionNamesTheCountsAndTheRule() throws {
        let planned = try plan(batchedInput([1, 2, 3, 4]), cachedTokens: [1, 2, 3])

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .memory, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=memory rendered=4 reused=3 fed=1 rule=extend")
    }

    @Test("the plan line of a rebuild names the seam and decodes each side")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfARebuildNamesTheSeamAndDecodesEachSide() throws {
        let planned = try planPromptCache(
            render: [1, 2, 9, 9], reusing: recurrentEntry(ledger: [1, 2, 3, 4]))

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .memory, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=memory rendered=4 reused=0 fed=4 rule=rebuild divergence=2 "
                + "render=<<<9 9>>> ledger=<<<3 4>>>")
    }

    @Test("the plan line of a rewind names the seam it rewound to")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfARewindNamesTheSeamItRewoundTo() throws {
        let planned = try plan(batchedInput([1, 2, 9]), cachedTokens: [1, 2, 3, 4])

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .memory, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=memory rendered=3 reused=2 fed=1 rule=rewind divergence=2 "
                + "render=<<<9>>> ledger=<<<3 4>>>")
    }

    @Test("the plan line of a cold pass names the cold rule")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfAColdPassNamesTheColdRule() throws {
        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .none,
            plan: try planPromptCache(render: [1, 2, 3], reusing: nil),
            decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=none rendered=3 reused=0 fed=3 rule=cold")
    }

    @Test("a carried entry whose caches hold nothing plans as cold")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCarriedEntryWhoseCachesHoldNothingPlansAsCold() throws {
        let emptyEntry = ExecutorPromptCacheEntry(caches: [KVCacheSimple()], tokens: [])

        let planned = try #require(try planPromptCache(render: [1, 2, 3], reusing: emptyEntry))
        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .memory, plan: planned, decodeTokens: decodeAsNumbers)

        #expect(planned.decision == .cold)
        #expect(planned.reusedTokenCount == 0)
        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=memory rendered=3 reused=0 fed=3 rule=cold")
    }

    @Test("the plan line of a restored pass names the disk and the restore time")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfARestoredPassNamesTheDiskAndTheRestoreTime() throws {
        let planned = try plan(batchedInput([1, 2, 3, 4]), cachedTokens: [1, 2, 3])

        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .disk(restoreDuration: Self.reportedRestoreDuration),
            plan: planned, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=disk restoreSeconds=1.250 rendered=4 reused=3 fed=1 rule=extend")
    }

    @Test("the plan line of a pass with no plan says why")
    func thePlanLineOfAPassWithNoPlanSaysWhy() {
        let line = ExecutorPromptCacheReport.planLine(
            key: reportedKey, source: .memory, plan: nil, decodeTokens: decodeAsNumbers)

        #expect(
            line
                == "prompt cache plan model=test/prompt-cache session=session-1 "
                + "source=memory rule=none (the input carries media, a batch or a mask)")
    }

    @Test("a pass with no session key names no session")
    func aPassWithNoSessionKeyNamesNoSession() {
        let line = ExecutorPromptCacheReport.planLine(
            key: nil, source: .none, plan: nil, decodeTokens: decodeAsNumbers)

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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
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
                    + "source=memory rendered=4 reused=3 fed=1 rule=extend",
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

#endif  // FoundationModelsIntegration
