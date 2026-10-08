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

    /// A store in a folder of its own, whose writer writes nothing. The
    /// budget comes from the device, thus each entry stays in memory.
    private func makeStore() -> ExecutorPromptCacheStore {
        ExecutorPromptCacheStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("ExecutorPromptCacheCheckpointTests-\(UUID().uuidString)"),
            writer: { _, _ in })
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

    @Test("the measured shape reuses the prompt up to the end of the last message")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theMeasuredShapeReusesThePromptUpToTheEndOfTheLastMessage() async throws {
        // Turn 1 renders "first turn\nassistant:". Turn 2 renders
        // "first turn\nturn 1\nassistant:": it agrees with turn 1 through the
        // line break after the last message of turn 1, and no further.
        let lastMessageEnd = (Self.firstMessage + Self.messageSeparator).utf8.count

        let passes = try await runDroppedTurn(
            processor: GenerationPromptBytesInputProcessor(
                generationPrompt: Self.generationPrompt),
            recurrentLayerCount: 1)

        #expect(passes.second.reusedTokenCount == lastMessageEnd)
    }

    @Test("a split prefill still reports every prompt token of the pass")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSplitPrefillStillReportsEveryPromptTokenOfThePass() async throws {
        let firstRender =
            Self.firstMessage + Self.messageSeparator + Self.generationPrompt

        let passes = try await runDroppedTurn(
            processor: GenerationPromptBytesInputProcessor(
                generationPrompt: Self.generationPrompt),
            recurrentLayerCount: 1)

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
        let sharedPrefix = (Self.firstMessage + Self.messageSeparator).utf8.count

        let passes = try await runDroppedTurn(
            processor: GenerationPromptBytesInputProcessor(
                generationPrompt: Self.generationPrompt),
            recurrentLayerCount: 0)

        #expect(passes.second.reusedTokenCount == sharedPrefix)
    }

    // MARK: - Planning one pass with a checkpoint

    /// The tokens the checkpoint fixtures represent at the checkpoint.
    private static let checkpointTokens = [1, 2, 3]

    /// The ledger of the checkpoint fixtures: the tokens at the checkpoint,
    /// then the generation prompt and the generated turn of the pass.
    private static let ledger = checkpointTokens + [4, 5, 6]

    /// The shape of the zero arrays the fixtures write into each cache.
    private static let fixtureShape = [1, 1, 1, 1]

    /// The model the plan fixtures make fresh caches with: one attention
    /// layer and one recurrent layer.
    private func hybridModel() -> ScriptedLanguageModel {
        ScriptedLanguageModel(rounds: [], cacheLayerCount: 1, recurrentLayerCount: 1)
    }

    /// Moves each cache of `caches` past `tokenCount` more tokens, the way a
    /// forward pass of a hybrid model does.
    ///
    /// - Parameters:
    ///   - caches: the caches to feed.
    ///   - tokenCount: the number of tokens to feed.
    private func feed(_ caches: [KVCache], tokenCount: Int) {
        let keyValues = MLXArray.zeros([1, 1, tokenCount, 1])
        for cache in caches {
            if let recurrent = cache as? MambaCache {
                recurrent[0] = MLXArray.zeros(Self.fixtureShape)
                recurrent[1] = MLXArray.zeros(Self.fixtureShape)
                recurrent.advance(tokenCount)
            } else {
                _ = cache.update(keys: keyValues, values: keyValues)
            }
        }
    }

    /// An entry whose hybrid caches hold ``ledger`` and carry a checkpoint at
    /// the end of ``checkpointTokens``.
    private func entryWithCheckpoint() throws -> ExecutorPromptCacheEntry {
        let caches = try hybridModel().newCache(parameters: nil)
        feed(caches, tokenCount: Self.checkpointTokens.count)
        let checkpoint = try #require(
            ExecutorPromptCacheCheckpoint(
                caches: caches, tokens: Self.checkpointTokens, state: nil))
        feed(caches, tokenCount: Self.ledger.count - Self.checkpointTokens.count)
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: Self.ledger, checkpoint: checkpoint)
    }

    /// Plans `render` against `entry`.
    private func plan(
        render: [Int], reusing entry: ExecutorPromptCacheEntry,
        protocolRules: [any PromptCacheReuseRule] = []
    ) throws -> ExecutorPromptCachePlan? {
        try ExecutorPromptCachePlan.make(
            reusing: entry, input: LMInput(tokens: MLXArray(render)), model: hybridModel(),
            parameters: GenerateParameters(), protocolRules: protocolRules)
    }

    @Test("a render that drops the generated turn goes back to the checkpoint")
    func aRenderThatDropsTheGeneratedTurnGoesBackToTheCheckpoint() throws {
        let render = Self.checkpointTokens + [9, 9]

        let planned = try #require(try plan(render: render, reusing: entryWithCheckpoint()))

        #expect(planned.decision == .restore(.init(render: render, ledger: Self.ledger)))
        #expect(planned.reusedTokenCount == Self.checkpointTokens.count)
        #expect(planned.input.text.tokens.asArray(Int.self) == [9, 9])
        #expect(planned.caches.allSatisfy { $0.offset == Self.checkpointTokens.count })
    }

    @Test("the plan line of a restore names the rule and the seam")
    func thePlanLineOfARestoreNamesTheRuleAndTheSeam() throws {
        let planned = try plan(
            render: Self.checkpointTokens + [9, 9], reusing: entryWithCheckpoint())

        let line = ExecutorPromptCacheReport.planLine(
            key: ExecutorPromptCacheKey(modelID: "test/checkpoint", sessionID: "s"),
            source: .memory, plan: planned,
            decodeTokens: { $0.map(String.init).joined(separator: " ") })

        #expect(
            line
                == "prompt cache plan model=test/checkpoint session=s source=memory "
                + "rendered=5 reused=3 fed=2 rule=restore divergence=3 "
                + "render=<<<9 9>>> ledger=<<<4 5 6>>>")
    }

    @Test("a render that parts before the checkpoint still rebuilds")
    func aRenderThatPartsBeforeTheCheckpointStillRebuilds() throws {
        let render = [1, 9, 9]

        let planned = try #require(try plan(render: render, reusing: entryWithCheckpoint()))

        #expect(planned.decision == .rebuild(.init(render: render, ledger: Self.ledger)))
        #expect(planned.reusedTokenCount == 0)
    }

    @Test("a render that extends the ledger still extends and does not restore")
    func aRenderThatExtendsTheLedgerStillExtendsAndDoesNotRestore() throws {
        let planned = try #require(
            try plan(render: Self.ledger + [7], reusing: entryWithCheckpoint()))

        #expect(planned.decision == .extend)
        #expect(planned.reusedTokenCount == Self.ledger.count)
    }

    /// A rule that keeps the whole ledger and feeds the last token of the
    /// render, the way a committed-turn rule splices.
    private struct SpliceLastTokenRule: PromptCacheReuseRule {
        func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
            guard let last = turn.promptTokens.last else { return nil }
            return .appendSuffix(
                suffixStart: turn.promptTokens.count - 1,
                representedTokens: cache.cachedTokens + [last])
        }
    }

    @Test("a protocol rule still splices and does not restore")
    func aProtocolRuleStillSplicesAndDoesNotRestore() throws {
        let planned = try #require(
            try plan(
                render: Self.checkpointTokens + [9, 9], reusing: entryWithCheckpoint(),
                protocolRules: [SpliceLastTokenRule()]))

        #expect(planned.decision == .splice)
    }

    @Test("caches that all rewind take no checkpoint")
    func cachesThatAllRewindTakeNoCheckpoint() {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: Self.checkpointTokens.count)

        #expect(
            ExecutorPromptCacheCheckpoint(caches: caches, tokens: Self.checkpointTokens, state: nil)
                == nil)
    }

    @Test("only a model with a layer that cannot rewind is served by a checkpoint")
    func onlyAModelWithALayerThatCannotRewindIsServedByACheckpoint() {
        #expect(ExecutorPromptCacheCheckpoint.applies(to: hybridModel()))
        #expect(
            !ExecutorPromptCacheCheckpoint.applies(
                to: ScriptedLanguageModel(rounds: [], cacheLayerCount: 1)))
    }

    @Test("an entry counts the bytes of its checkpoint")
    func anEntryCountsTheBytesOfItsCheckpoint() throws {
        let entry = try entryWithCheckpoint()
        let checkpoint = try #require(entry.checkpoint)

        #expect(checkpoint.byteCount > 0)
        #expect(
            entry.byteCount
                == entry.caches.reduce(0) { $0 + $1.residentByteCount } + checkpoint.byteCount)
    }

    @Test("a boundary before the end of the prompt is fed before generation")
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

    @Test("a boundary at the end of the prompt is taken when the prefill ends")
    func aBoundaryAtTheEndOfThePromptIsTakenWhenThePrefillEnds() throws {
        let model = hybridModel()
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })
        let planned = try #require(
            try slot.plan(
                input: LMInput(tokens: MLXArray(Self.checkpointTokens)), model: model,
                parameters: GenerateParameters(),
                transcriptBoundary: Self.checkpointTokens.count,
                decodeTokens: { _ in "" }))
        #expect(slot.prefilledTokenCount == 0)

        feed(planned.caches, tokenCount: Self.checkpointTokens.count)
        slot.prefillDidEnd(state: nil)
        feed(planned.caches, tokenCount: 1)
        slot.commit(planned, generatedTokens: [101])

        #expect(slot.entry?.checkpoint?.tokens == Self.checkpointTokens)
    }

    @Test("the commit line names the position of the checkpoint")
    func theCommitLineNamesThePositionOfTheCheckpoint() throws {
        let line = ExecutorPromptCacheReport.commitLine(
            key: ExecutorPromptCacheKey(modelID: "test/checkpoint", sessionID: "s"),
            outcome: .checkedIn(try entryWithCheckpoint()))

        #expect(line == "prompt cache commit model=test/checkpoint session=s ledger=6 checkpoint=3")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
