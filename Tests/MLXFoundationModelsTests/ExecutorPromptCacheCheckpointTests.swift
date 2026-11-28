// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A hybrid model keeps a checkpoint of its recurrent state at the end of the
/// last message of each prompt, thus a next render that drops the generated
/// turn reuses the prompt up to that point and does not rebuild.
///
/// The measured case (card ^8qkdk0b): Router rejects a tool call, and the
/// next render keeps every message of the turn, drops the generation prompt
/// and the turn the model generated, and adds a user message. The recurrent
/// layers cannot rewind into the generation prompt, thus only a checkpoint
/// serves that render.
///
/// These passes run the executor over scripted models. No weights are needed.
@Suite("A hybrid model restores its checkpoint when the next render drops the generated turn")
struct ExecutorPromptCacheCheckpointTests {

    /// The generation prompt that the measured-shape processor writes after
    /// the last message. Its first letter is not the first letter of a
    /// message, thus the next render parts from it at its first token.
    private static let generationPrompt = "assistant:"

    /// The text of the first message of every session here, as
    /// ``ScriptedSessionModel/transcript(firstEntryID:turns:)`` writes it.
    private static let firstMessage = "first turn"

    /// The line break between two messages of a render.
    private static let messageSeparator = "\n"

    /// The end of the first message of every session here, the line break
    /// after it included: as far as the next render agrees with the first.
    private static let firstMessageEnd = (firstMessage + messageSeparator).utf8.count

    /// A store in a folder of its own, whose writer writes nothing. The
    /// budget comes from the device, thus each entry stays in memory.
    private func makeStore() -> ExecutorPromptCacheStore {
        PromptCacheTestStore.withoutWrites(folderNamed: "ExecutorPromptCacheCheckpointTests")
    }

    /// Runs the first turn of a new session, and then a second turn whose
    /// render keeps the first message and adds one user message, with no
    /// generated turn between them.
    ///
    /// - Parameters:
    ///   - processor: renders the prompt of each pass.
    ///   - recurrentLayerCount: the recurrent layers of the model.
    /// - Returns: what the first pass and the second pass streamed.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func runDroppedTurn(
        processor: any UserInputProcessor, recurrentLayerCount: Int
    ) async throws -> (first: ScriptedPassResult, second: ScriptedPassResult) {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(
            weights: weights,
            scripts: Array(
                repeating: ScriptedSessionModel.scriptedResponse,
                count: ScriptedSessionModel.maximumPassCount),
            processor: processor, recurrentLayerCount: recurrentLayerCount)
        let store = makeStore()
        let sessionID = "checkpoint-\(UUID().uuidString)"

        let first = try await ScriptedExecutorPass.respond(
            over: ScriptedSessionModel.transcript(firstEntryID: sessionID),
            model: model, inside: store)
        let second = try await ScriptedExecutorPass.respond(
            over: ScriptedSessionModel.transcript(firstEntryID: sessionID, turns: 2),
            model: model, inside: store)
        return (first, second)
    }

    /// Runs ``runDroppedTurn(processor:recurrentLayerCount:)`` with the processor that writes
    /// ``generationPrompt`` after the last message.
    ///
    /// - Parameter recurrentLayerCount: the recurrent layers of the model.
    /// - Returns: what the first pass and the second pass streamed.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func runGenerationPromptDroppedTurn(
        recurrentLayerCount: Int = 1
    ) async throws -> (first: ScriptedPassResult, second: ScriptedPassResult) {
        try await runDroppedTurn(
            processor: GenerationPromptBytesInputProcessor(generationPrompt: Self.generationPrompt),
            recurrentLayerCount: recurrentLayerCount)
    }

    @Test("the measured shape reuses the prompt up to the end of the last message")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theMeasuredShapeReusesThePromptUpToTheEndOfTheLastMessage() async throws {
        // Turn 1 renders "first turn\nassistant:". Turn 2 renders
        // "first turn\nturn 1\nassistant:": it agrees with turn 1 through the
        // line break after the last message of turn 1, and no further.
        let passes = try await runGenerationPromptDroppedTurn()

        #expect(passes.second.reusedTokenCount == Self.firstMessageEnd)
    }

    @Test("a split prefill still reports every prompt token of the pass")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSplitPrefillStillReportsEveryPromptTokenOfThePass() async throws {
        let firstRender =
            Self.firstMessage + Self.messageSeparator + Self.generationPrompt

        let passes = try await runGenerationPromptDroppedTurn()

        #expect(passes.first.reusedTokenCount == 0)
        #expect(passes.first.promptTokenCount == firstRender.utf8.count)
    }

    @Test("with no generation prompt the checkpoint stands at the end of the prompt")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func withNoGenerationPromptTheCheckpointStandsAtTheEndOfThePrompt() async throws {
        // Turn 1 renders "first turn" and writes nothing after the last
        // message, thus the end of the last message is the end of the
        // prompt.
        let passes = try await runDroppedTurn(
            processor: PromptBytesInputProcessor(), recurrentLayerCount: 1)

        #expect(passes.second.reusedTokenCount == Self.firstMessage.utf8.count)
    }

    @Test("a model that is not hybrid still rewinds to the shared prefix")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aModelThatIsNotHybridStillRewindsToTheSharedPrefix() async throws {
        // An attention cache rewinds to any position, thus the render reuses
        // all it shares with the ledger, the line break included.
        let passes = try await runGenerationPromptDroppedTurn(recurrentLayerCount: 0)

        #expect(passes.second.reusedTokenCount == Self.firstMessageEnd)
    }

    // MARK: - Planning one pass with a checkpoint

    /// The tokens the checkpoint fixtures represent at the checkpoint.
    private static let checkpointTokens = [1, 2, 3]

    /// The ledger of the checkpoint fixtures: the tokens at the checkpoint,
    /// then the generation prompt and the generated turn of the pass.
    private static let ledger = checkpointTokens + [4, 5, 6]

    /// The key that the report lines of the checkpoint fixture name.
    private static let checkpointKey = PromptCacheTestKey.make(
        modelID: .init("test/checkpoint"), sessionID: .init("s"))

    /// The model the plan fixtures make fresh caches with: one attention
    /// layer and one recurrent layer.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func hybridModel() -> ScriptedLanguageModel {
        ScriptedLanguageModel(rounds: [], cacheLayerCount: 1, recurrentLayerCount: 1)
    }

    /// An entry whose hybrid caches hold ``ledger`` and carry a checkpoint at
    /// the end of ``checkpointTokens``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func entryWithCheckpoint() throws -> ExecutorPromptCacheEntry {
        let caches = try hybridModel().newCache(parameters: nil)
        feed(caches, tokenCount: Self.checkpointTokens.count)
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: caches, tokens: Self.checkpointTokens, state: nil))
        feed(caches, tokenCount: Self.ledger.count - Self.checkpointTokens.count)
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: Self.ledger, checkpoint: checkpoint)
    }

    /// Plans a pass over `render` that reuses ``entryWithCheckpoint()`` on ``hybridModel()``.
    ///
    /// - Parameters:
    ///   - render: the tokens of the next render.
    ///   - protocolRules: the cache-reuse rules of the response protocol.
    /// - Returns: the plan, or nil when the input cannot be planned.
    /// - Throws: the error of the plan or of the fixture.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func planCheckpointRender(
        _ render: [Int], protocolRules: [any PromptCacheReuseRule] = []
    ) throws -> ExecutorPromptCachePlan? {
        try planPromptCache(
            render: render, reusing: entryWithCheckpoint(), model: hybridModel(),
            protocolRules: protocolRules)
    }

    @Test("a render that drops the generated turn goes back to the checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatDropsTheGeneratedTurnGoesBackToTheCheckpoint() throws {
        let render = Self.checkpointTokens + [9, 9]

        let planned = try #require(try planCheckpointRender(render))

        #expect(planned.decision == .restore(.init(render: render, ledger: Self.ledger)))
        #expect(planned.reusedTokenCount == Self.checkpointTokens.count)
        #expect(planned.input.text.tokens.asArray(Int.self) == [9, 9])
        #expect(planned.caches.allSatisfy { $0.offset == Self.checkpointTokens.count })
    }

    @Test("the plan line of a restore names the rule and the seam")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePlanLineOfARestoreNamesTheRuleAndTheSeam() throws {
        let planned = try planCheckpointRender(Self.checkpointTokens + [9, 9])

        let line = ExecutorPromptCacheReport.planLine(
            key: Self.checkpointKey,
            source: .memory, plan: planned,
            decodeTokens: { $0.map(String.init).joined(separator: " ") })

        #expect(
            line
                == "prompt cache plan model=test/checkpoint session=s source=memory "
                + "rendered=5 reused=3 fed=2 rule=restore divergence=3 "
                + "render=<<<9 9>>> ledger=<<<4 5 6>>>")
    }

    @Test("a render that parts before the checkpoint still rebuilds")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatPartsBeforeTheCheckpointStillRebuilds() throws {
        let render = [1, 9, 9]

        let planned = try #require(try planCheckpointRender(render))

        #expect(planned.decision == .rebuild(.init(render: render, ledger: Self.ledger)))
        #expect(planned.reusedTokenCount == 0)
    }

    @Test("a render that extends the ledger still extends and does not restore")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatExtendsTheLedgerStillExtendsAndDoesNotRestore() throws {
        let planned = try #require(try planCheckpointRender(Self.ledger + [7]))

        #expect(planned.decision == .extend)
        #expect(planned.reusedTokenCount == Self.ledger.count)
    }

    /// The token the splicing rule takes as the commit of the turn the model
    /// wrote: the first token after the checkpoint in the render of the
    /// splice test, thus the rule feeds the last token of that render.
    private static let spliceCommit = 9

    @Test("a protocol rule still splices and does not restore")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aProtocolRuleStillSplicesAndDoesNotRestore() throws {
        let planned = try #require(
            try planCheckpointRender(
                Self.checkpointTokens + [9, 9],
                protocolRules: [SplicingRule(commit: Self.spliceCommit)]))

        #expect(planned.decision == .splice)
    }

    @Test("caches that all rewind take no checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func cachesThatAllRewindTakeNoCheckpoint() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: Self.checkpointTokens.count)

        #expect(
            PromptCacheCheckpoint(caches: caches, tokens: Self.checkpointTokens, state: nil)
                == nil)
    }

    @Test("only a model with a layer that cannot rewind is served by a checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func onlyAModelWithALayerThatCannotRewindIsServedByACheckpoint() {
        #expect(PromptCacheCheckpoint.applies(to: hybridModel()))
        #expect(
            !PromptCacheCheckpoint.applies(
                to: ScriptedLanguageModel(rounds: [], cacheLayerCount: 1)))
    }

    @Test("an entry counts the bytes of its checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEntryCountsTheBytesOfItsCheckpoint() throws {
        let entry = try entryWithCheckpoint()
        let checkpoint = try #require(entry.checkpoint)

        #expect(checkpoint.byteCount > 0)
        #expect(
            entry.byteCount
                == entry.caches.reduce(0) { $0 + $1.residentByteCount } + checkpoint.byteCount)
    }

    @Test("a boundary before the end of the prompt is fed before generation")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aBoundaryBeforeTheEndOfThePromptIsFedBeforeGeneration() throws {
        let model = hybridModel()
        let render = Self.ledger
        let planned = try #require(
            try ExecutorPromptCachePlan.make(
                reusing: nil, input: LMInput(tokens: MLXArray(render)), model: model,
                parameters: GenerateParameters()))

        let split = try planned.prefillingToCheckpoint(
            at: Self.checkpointTokens.count, model: model, parameters: GenerateParameters())

        #expect(split.input.text.tokens.asArray(Int.self) == [4, 5, 6])
        #expect(split.prefilledTokenCount == Self.checkpointTokens.count)
        #expect(split.checkpoint?.tokens == Self.checkpointTokens)
        #expect(split.caches.allSatisfy { $0.offset == Self.checkpointTokens.count })
    }

    /// Plans a pass of `slot` over `render` without a report of its tokens.
    private static func plan(
        in slot: ExecutorPromptCacheSlot, render: [Int], model: any MLXLMCommon.LanguageModel,
        transcriptBoundary: Int?
    ) throws -> ExecutorPromptCachePlan? {
        try slot.plan(
            input: LMInput(tokens: MLXArray(render)), model: model,
            parameters: GenerateParameters(),
            checkpointBoundaries: ExecutorCheckpointBoundaries(transcriptEnd: transcriptBoundary),
            decodeTokens: { _ in "" })
    }

    /// Plans a pass over the checkpoint tokens whose transcript ends with the prompt.
    private static func promptEndPlan(
        in slot: ExecutorPromptCacheSlot, model: any MLXLMCommon.LanguageModel
    ) throws -> ExecutorPromptCachePlan? {
        try plan(
            in: slot, render: checkpointTokens, model: model,
            transcriptBoundary: checkpointTokens.count)
    }

    @Test("a boundary at the end of the prompt is taken when the prefill ends")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aBoundaryAtTheEndOfThePromptIsTakenWhenThePrefillEnds() throws {
        let model = hybridModel()
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })
        let planned = try #require(
            try Self.promptEndPlan(in: slot, model: model))
        #expect(slot.prefilledTokenCount == 0)

        feed(planned.caches, tokenCount: Self.checkpointTokens.count)
        slot.prefillDidEnd(state: nil)
        feed(planned.caches, tokenCount: 1)
        slot.commit(planned, generatedTokens: [101])

        #expect(slot.entry?.checkpoint?.tokens == Self.checkpointTokens)
    }

    /// The render of the pass after the stale pass: it parts from the ledger
    /// at its first token, thus the hybrid caches rebuild from fresh caches
    /// and the caches of the earlier pass keep their position.
    private static let partedRender = [9, 9]

    @Test("a commit drops the prompt-end plan of a pass whose prefill did not report its end")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCommitDropsThePromptEndPlanOfAPassWhosePrefillDidNotReportItsEnd() throws {
        let model = hybridModel()
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })
        let stale = try #require(
            try Self.promptEndPlan(in: slot, model: model))
        feed(stale.caches, tokenCount: Self.checkpointTokens.count)
        slot.commit(stale, generatedTokens: [])

        let next = try #require(
            try Self.plan(
                in: slot, render: Self.partedRender, model: model, transcriptBoundary: nil))
        feed(next.caches, tokenCount: Self.partedRender.count)
        slot.prefillDidEnd(state: nil)
        slot.commit(next, generatedTokens: [])

        #expect(slot.entry?.checkpoint == nil)
    }

    @Test("the commit line names the position of the checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theCommitLineNamesThePositionOfTheCheckpoint() throws {
        let line = ExecutorPromptCacheReport.commitLine(
            key: Self.checkpointKey,
            outcome: .checkedIn(try entryWithCheckpoint()))

        #expect(line == "prompt cache commit model=test/checkpoint session=s ledger=6 checkpoint=3")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
