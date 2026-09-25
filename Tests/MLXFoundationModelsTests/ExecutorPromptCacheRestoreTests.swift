// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXNN
import Synchronization
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// Proves that `MLXLanguageModel.Executor` restores a prompt cache that went to disk, and that
/// the file of that cache never stays after the turn.
///
/// Each test binds its own store with `ExecutorPromptCacheStore.$current`. The memory budget
/// of the store is zero, thus each check-in goes to disk at once, and a later turn of the
/// session finds its cache on disk only. The memory control of the cache-echo tests is the
/// one exception: its budget keeps the cache in memory.
///
/// The executor is available from iOS 27, macOS 27 and visionOS 27. On an earlier system each
/// executor test records an issue, thus it fails and does not pass with no assertion.
@Suite("The executor restores a prompt cache that went to disk")
struct ExecutorPromptCacheRestoreTests: PromptCacheSpoolFixtures {

    /// The first entry of the cold control, a session that the store never saw.
    private static let coldSessionID = "cold-control"

    /// The generation of the spill file of the slot tests.
    private static let slotFileGeneration: UInt64 = 1

    /// The bytes that the corrupt-file test writes in place of the spilled file.
    private static let corruptBytes = Data("this is not a prompt cache".utf8)

    /// The model of the slot tests: one KV cache layer, and no script.
    private static let slotModelID = "test-org/prompt-cache-restore"

    /// The rendered prompt of the slot tests. It extends the ledger of the fixture entry by one
    /// token.
    private static let slotPromptTokens =
        PromptCacheSpoolFixtureShape.tokens + [
            PromptCacheSpoolFixtureShape.tokenCount
        ]

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    // MARK: - The executor

    @Test("a session whose cache went to disk comes back warm, with the answer of a cold run")
    func aSpilledSessionComesBackWarm() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASpilledSessionComesBackWarm()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a successful restore deletes the spilled file")
    func aSuccessfulRestoreDeletesTheSpilledFile() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASuccessfulRestoreDeletesTheSpilledFile()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a corrupt spilled file gives a cold turn that succeeds, and the file is deleted")
    func aCorruptSpilledFileGivesAColdTurn() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectACorruptSpilledFileGivesAColdTurn()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn cancelled after the check-out and before the restore leaves no file")
    func aCancelledTurnLeavesNoFile() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectACancelledTurnLeavesNoFile()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn whose container work throws before the restore leaves no file")
    func aFailedTurnLeavesNoFile() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAFailedTurnLeavesNoFile()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn restored from disk gives the output of a turn restored from memory")
    func aDiskRestoreGivesTheOutputOfAMemoryRestore() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectADiskRestoreGivesTheOutputOfAMemoryRestore()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a turn restored from a spilled file with changed values gives a different output")
    func aDiskRestoreOfChangedValuesGivesADifferentOutput() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectADiskRestoreOfChangedValuesGivesADifferentOutput()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The slot

    @Test("a slot restores a spilled entry one time, and its plan line names the disk")
    func aSlotRestoresASpilledEntryOneTime() throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try expectASlotRestoresASpilledEntryOneTime()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a slot whose spilled file is missing reports the failure and plans a cold pass")
    func aSlotWhoseSpilledFileIsMissingPlansAColdPass() throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try expectASlotWhoseSpilledFileIsMissingPlansAColdPass()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a committed Qwen turn restored from disk splices, and its plan line names the disk")
    func aCommittedTurnRestoredFromDiskSplices() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectACommittedTurnRestoredFromDiskSplices()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a committed turn restored from disk with no render on record does not splice")
    func aCommittedTurnWithNoRenderDoesNotSplice() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectACommittedTurnWithNoRenderDoesNotSplice()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The checks

    /// An entry that holds a committed Qwen turn goes to disk and comes back through the slot.
    /// The Qwen rule then keeps the tokens that the model wrote and feeds only the tool
    /// response. Thus the file carries the render that the rule needs.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectACommittedTurnRestoredFromDiskSplices() async throws {
        let turn = try await CommittedTurn.planAfterRestore(
            renderTokens: CommittedTurn.previousRender)

        #expect(turn.restoredRenderTokens == CommittedTurn.previousRender)
        #expect(turn.plan.decision == .splice)
        #expect(turn.plan.reusedTokenCount == CommittedTurn.ledger.count)
        #expect(turn.plan.representedTokens == CommittedTurn.ledger + CommittedTurn.toolResponse)
        #expect(turn.plan.input.text.tokens.asArray(Int.self) == CommittedTurn.toolResponse)
        let line = try #require(turn.lines.first)
        #expect(turn.lines.count == 1)
        #expect(line.hasPrefix(Self.planLinePrefix(of: CommittedTurn.key) + "source=disk "))
        #expect(
            line.hasSuffix(
                "rendered=\(CommittedTurn.nextRender.count) reused=\(CommittedTurn.ledger.count) "
                    + "fed=\(CommittedTurn.toolResponse.count) rule=splice"))
    }

    /// Control: the same chain with an entry that records no render. The Qwen rule declines,
    /// thus the plan does not keep the committed turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectACommittedTurnWithNoRenderDoesNotSplice() async throws {
        let turn = try await CommittedTurn.planAfterRestore(renderTokens: [])

        #expect(turn.restoredRenderTokens.isEmpty)
        #expect(turn.plan.decision != .splice)
        #expect(turn.plan.reusedTokenCount < CommittedTurn.ledger.count)
        let line = try #require(turn.lines.first)
        #expect(line.hasPrefix(Self.planLinePrefix(of: CommittedTurn.key) + "source=disk "))
        #expect(!line.contains("rule=splice"))
    }

    /// A slot reads its spilled entry at the first restore call and not at the second, and the
    /// plan line of its first pass names the disk and the restore time.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASlotRestoresASpilledEntryOneTime() throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let handle = try Self.writeSpillFile(in: directory)
        var lines: [String] = []
        let slot = ExecutorPromptCacheSlot(spilled: handle, key: handle.key) { lines.append($0) }
        let model = ScriptedLanguageModel(
            rounds: [], cacheLayerCount: ScriptedSessionModel.cacheLayerCount)

        slot.restoreIfPending(model: model, parameters: GenerateParameters())
        let restored = try #require(slot.entry)
        slot.restoreIfPending(model: model, parameters: GenerateParameters())
        let plan = try slot.plan(
            input: LMInput(tokens: MLXArray(Self.slotPromptTokens)), model: model,
            parameters: GenerateParameters(), decodeTokens: { _ in "" })

        #expect(restored.tokens == Self.tokens)
        #expect(plan?.reusedTokenCount == Self.tokens.count)
        let line = try #require(lines.first)
        #expect(lines.count == 1)
        #expect(line.hasPrefix(Self.planLinePrefix(of: handle.key) + "source=disk restoreSeconds="))
        #expect(line.hasSuffix("rendered=4 reused=3 fed=1 rule=extend"))
    }

    /// A slot whose spilled file is missing reports one failure line, and its first pass
    /// plans cold.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASlotWhoseSpilledFileIsMissingPlansAColdPass() throws {
        let directory = Self.temporaryDirectory()
        let key = ExecutorPromptCacheKey(
            modelID: Self.slotModelID, sessionID: SpilledSessionValues.sessionID)
        let handle = ExecutorPromptCacheSpilledHandle(
            url: directory.appendingPathComponent("missing.safetensors"), key: key)
        var lines: [String] = []
        let slot = ExecutorPromptCacheSlot(spilled: handle, key: key) { lines.append($0) }
        let model = ScriptedLanguageModel(
            rounds: [], cacheLayerCount: ScriptedSessionModel.cacheLayerCount)

        slot.restoreIfPending(model: model, parameters: GenerateParameters())
        _ = try slot.plan(
            input: LMInput(tokens: MLXArray(Self.slotPromptTokens)), model: model,
            parameters: GenerateParameters(), decodeTokens: { _ in "" })

        #expect(lines.count == 2)
        let failure = try #require(lines.first)
        #expect(
            failure.hasPrefix(
                "prompt cache restore model=\(Self.slotModelID) session=\(SpilledSessionValues.sessionID) "
                    + "failed, the turn starts cold: "))
        #expect(
            lines.last
                == Self.planLinePrefix(of: key) + "source=none rendered=4 reused=0 fed=4 rule=cold")
    }

    /// The second turn of a spilled session reuses the prompt of the first turn, and its answer
    /// is the answer of a cold run of the same transcript.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASpilledSessionComesBackWarm() async throws {
        let session = try await SpilledSession.make()
        defer { session.remove() }
        #expect(await session.store.retainedByteCount == 0, "The cache must be on disk only.")

        let warm = try await session.respond(firstEntryID: SpilledSessionValues.sessionID)
        let cold = try await session.respond(firstEntryID: Self.coldSessionID)

        #expect(warm.reusedTokenCount > 0, "The restored session must start warm.")
        #expect(cold.reusedTokenCount == 0, "The cold control must reuse nothing.")
        #expect(warm.responseText == cold.responseText)
    }

    /// The later turn of a session whose cache went to disk reuses the tokens and gives the text
    /// of the same turn of a session whose cache stayed in memory. The model emits the tokens
    /// that its cache holds, thus the text is correct only when the restore gives the correct
    /// values.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectADiskRestoreGivesTheOutputOfAMemoryRestore() async throws {
        let memory = try await CacheEchoSession.run(budget: CacheEchoSession.memoryBudgetBytes)
        let disk = try await CacheEchoSession.run(budget: 0)

        #expect(memory.spilledFileCount == 0, "The memory run must keep its cache in memory.")
        #expect(memory.retainedByteCount > 0, "The memory run must keep its cache in memory.")
        #expect(disk.spilledFileCount == 1, "The disk run must spill its cache to one file.")
        #expect(disk.retainedByteCount == 0, "The disk run must keep no cache in memory.")
        #expect(memory.turn.reusedTokenCount > 0, "The memory run must start warm.")
        #expect(
            memory.turn.responseText.utf8.count == CacheEchoLanguageModel.responseTokenCount,
            "The model must emit one token for each position that it echoes.")
        #expect(disk.turn.reusedTokenCount == memory.turn.reusedTokenCount)
        #expect(disk.turn.responseText == memory.turn.responseText)
    }

    /// Negative control: a spilled file whose array values changed, and whose header stays
    /// valid, restores and reuses the same tokens, but gives a different text. Thus the text
    /// check of ``expectADiskRestoreGivesTheOutputOfAMemoryRestore()`` sees wrong values.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectADiskRestoreOfChangedValuesGivesADifferentOutput() async throws {
        let memory = try await CacheEchoSession.run(budget: CacheEchoSession.memoryBudgetBytes)
        let changed = try await CacheEchoSession.run(budget: 0, changesSpilledValues: true)

        #expect(changed.spilledFileCount == 1, "The changed run must spill its cache to one file.")
        #expect(
            changed.turn.reusedTokenCount == memory.turn.reusedTokenCount,
            "The changed file must restore, thus its header must stay valid.")
        #expect(changed.turn.responseText != memory.turn.responseText)
    }

    /// A turn that restores the spilled cache deletes the file of that cache. The check-in of
    /// the turn writes a new file under a new name.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASuccessfulRestoreDeletesTheSpilledFile() async throws {
        let session = try await SpilledSession.make()
        defer { session.remove() }

        let warm = try await session.respond(firstEntryID: SpilledSessionValues.sessionID)
        await session.store.waitForSpills()

        #expect(warm.reusedTokenCount > 0)
        let files = try Self.fileNames(in: session.directory)
        #expect(!files.contains(session.spilledFileName))
        #expect(files.count == 1, "The check-in of the restored turn spills one new file.")
    }

    /// A spilled file that does not read back gives a cold turn that succeeds, and the turn
    /// deletes the file.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectACorruptSpilledFileGivesAColdTurn() async throws {
        let session = try await SpilledSession.make()
        defer { session.remove() }
        try Self.corruptBytes.write(to: session.spilledFileURL)

        let turn = try await session.respond(firstEntryID: SpilledSessionValues.sessionID)
        await session.store.waitForSpills()

        #expect(turn.reusedTokenCount == 0, "A cache that does not read back gives a cold turn.")
        #expect(turn.responseText == ScriptedSessionModel.scriptedResponse)
        #expect(try !Self.fileNames(in: session.directory).contains(session.spilledFileName))
    }

    /// A turn that is cancelled while the render holds -- after the check-out gave the spilled
    /// file and before the restore reads it -- deletes the file.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectACancelledTurnLeavesNoFile() async throws {
        let session = try await SpilledSession.make()
        defer { session.remove() }
        var heldRenders = session.gate.heldRenders.makeAsyncIterator()
        session.gate.nextAction = .hold

        let turn = Task { try await session.respond(firstEntryID: SpilledSessionValues.sessionID) }
        _ = await heldRenders.next()
        #expect(await session.store.checkOut(session.key) == .none)
        #expect(try Self.fileNames(in: session.directory) == [session.spilledFileName])
        turn.cancel()
        let outcome = await turn.result

        #expect(throws: CancellationError.self) { try outcome.get() }
        #expect(try Self.fileNames(in: session.directory).isEmpty)
    }

    /// A turn whose work in the model container throws before the restore deletes the file.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAFailedTurnLeavesNoFile() async throws {
        let session = try await SpilledSession.make()
        defer { session.remove() }
        session.gate.nextAction = .fail

        await #expect(throws: ScriptedRenderError.failed) {
            try await session.respond(firstEntryID: SpilledSessionValues.sessionID)
        }
        #expect(try Self.fileNames(in: session.directory).isEmpty)
        #expect(await session.store.checkOut(session.key) == .none)
    }

    // MARK: - Fixtures

    /// The start of each plan line of the slot tests.
    ///
    /// - Parameter key: the session of the slot.
    /// - Returns: the head of the line, up to the source.
    private static func planLinePrefix(of key: ExecutorPromptCacheKey) -> String {
        "prompt cache plan model=\(key.modelID) session=\(key.sessionID) "
    }

    /// Writes the fixture entry to a spill file in `directory`.
    ///
    /// - Parameter directory: the folder of the file. The function makes it.
    /// - Returns: the handle of the file.
    private static func writeSpillFile(
        in directory: URL
    ) throws -> ExecutorPromptCacheSpilledHandle {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let key = ExecutorPromptCacheKey(
            modelID: slotModelID, sessionID: SpilledSessionValues.sessionID)
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: slotFileGeneration))
        try ExecutorPromptCacheFile.write(
            ExecutorPromptCacheFile.prepare(entry(), key: key), to: url)
        return ExecutorPromptCacheSpilledHandle(url: url, key: key)
    }
}

// MARK: - A session whose first turn went to disk

/// The values of each spilled session.
private enum SpilledSessionValues {

    /// The first entry of the session.
    static let sessionID = "spilled-session"

    /// The number of prompts of the later turn of the session.
    static let laterTurnCount = 2
}

/// A scripted session whose first turn ran and whose cache went to disk.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct SpilledSession: PromptCacheSpoolFixtures {

    /// The spool folder of the store.
    let directory: URL

    /// The folder that makes the model available.
    let weights: URL

    /// The store of the session. Its memory budget is zero.
    let store: ExecutorPromptCacheStore

    /// The scripted model of the session.
    let model: MLXLanguageModel

    /// Decides what the next render of the model does.
    let gate: ScriptedRenderGate

    /// The name of the file that the first turn spilled.
    let spilledFileName: String

    /// The key of the session.
    var key: ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: model.modelID, sessionID: SpilledSessionValues.sessionID)
    }

    /// The URL of the file that the first turn spilled.
    var spilledFileURL: URL { directory.appendingPathComponent(spilledFileName) }

    /// Runs the first turn of a new session, and waits until its cache is on disk.
    ///
    /// - Returns: the session.
    /// - Throws: the error of the turn, or an issue when the turn spilled no file.
    static func make() async throws -> SpilledSession {
        let directory = temporaryDirectory()
        let weights = try makeScriptedWeightsDirectory()
        let gate = ScriptedRenderGate()
        let model = ScriptedSessionModel.make(
            weights: weights, processor: GatedPromptBytesInputProcessor(gate: gate))
        let store = await store(in: directory)

        try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: SpilledSessionValues.sessionID),
            model: model,
            inside: store)
        await store.waitForSpills()

        let files = try fileNames(in: directory)
        #expect(files.count == 1, "The first turn must spill one file.")
        return SpilledSession(
            directory: directory, weights: weights, store: store, model: model, gate: gate,
            spilledFileName: try #require(files.first))
    }

    /// Runs one later turn over a transcript whose first entry is `firstEntryID`.
    ///
    /// - Parameter firstEntryID: the first entry, which names the session of the turn.
    /// - Returns: what the turn streamed.
    /// - Throws: the error of the executor.
    func respond(firstEntryID: String) async throws -> ScriptedPassResult {
        try await ScriptedExecutorPass.respond(
            over: ScriptedSessionModel.transcript(
                firstEntryID: firstEntryID, turns: SpilledSessionValues.laterTurnCount),
            model: model, inside: store)
    }

    /// Removes the spool folder and the weights folder.
    func remove() {
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.removeItem(at: weights)
    }
}

// MARK: - A committed Qwen turn that went to disk

/// What the first pass after the restore of a ``CommittedTurn`` entry planned.
private struct CommittedTurnPlan {

    /// The render that the restored entry records.
    let restoredRenderTokens: [Int]

    /// The plan of the pass.
    let plan: ExecutorPromptCachePlan

    /// Each log line of the slot, in order.
    let lines: [String]
}

/// A Qwen agent round whose entry goes to disk between two turns.
///
/// The fixture has the shape of `QwenCommittedTurnRuleTests`. The ledger holds the render of
/// the last pass and the turn that the model wrote, which ends at the commit token. The next
/// render writes that turn again in its own tokens, and adds a tool response after the commit.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private enum CommittedTurn: PromptCacheSpoolFixtures {

    /// Stands in for the `<|im_end|>` commit token.
    static let commit = 1

    /// Stands in for a token that only the model wrote.
    static let generatedOnly = 70

    /// Stands in for the token that the render writes where the model wrote ``generatedOnly``.
    static let renderedOnly = 71

    /// Stands in for the tokens of the conversation before the model turn.
    static let conversationToken = 10

    /// Stands in for the generation prompt that opens the model turn.
    static let generationPromptToken = 11

    /// Stands in for the token that opens the tool response.
    static let toolResponseOpenToken = 20

    /// Stands in for the result that the tool response carries.
    static let toolResultToken = 21

    /// The render of the last pass. It ends at the generation prompt.
    static let previousRender = [conversationToken, generationPromptToken]

    /// The tool response that the next render adds after the commit.
    static let toolResponse = [toolResponseOpenToken, toolResultToken]

    /// The tokens of the entry: the last render and the turn that the model wrote.
    static let ledger = previousRender + [generatedOnly, commit]

    /// The next render: the committed turn in the tokens of the template, then the tool
    /// response.
    static let nextRender = previousRender + [renderedOnly, commit] + toolResponse

    /// The session of the entry.
    static let key = ExecutorPromptCacheKey(
        modelID: "test-org/prompt-cache-committed-turn", sessionID: "committed-turn-session")

    /// Checks an entry of ``ledger`` into a store whose memory budget is zero, waits for the
    /// spill, checks it out, restores it through a slot, and plans ``nextRender`` with the Qwen
    /// rule.
    ///
    /// - Parameter renderTokens: The render that the entry records, or empty for no render.
    /// - Returns: What the pass planned.
    /// - Throws: An issue when the check-out gives no file or the slot gives no plan, or the
    ///   error of the plan.
    static func planAfterRestore(renderTokens: [Int]) async throws -> CommittedTurnPlan {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = await store(in: directory)
        await store.checkIn(key, entry(tokens: ledger, renderTokens: renderTokens))
        await store.waitForSpills()
        let handle = try #require(
            await store.checkOut(key).spilledHandle, "The entry must come back from disk.")
        return try plan(restoring: handle)
    }

    /// Restores the file of `handle` through a slot, and plans ``nextRender`` with the Qwen
    /// rule.
    ///
    /// - Parameter handle: The spilled file of the entry.
    /// - Returns: What the pass planned.
    /// - Throws: An issue when the slot gives no plan, or the error of the plan.
    private static func plan(
        restoring handle: ExecutorPromptCacheSpilledHandle
    ) throws -> CommittedTurnPlan {
        var lines: [String] = []
        let slot = ExecutorPromptCacheSlot(spilled: handle, key: key) { lines.append($0) }
        let model = ScriptedLanguageModel(
            rounds: [], cacheLayerCount: ScriptedSessionModel.cacheLayerCount)

        slot.restoreIfPending(model: model, parameters: GenerateParameters())
        let restored = try #require(slot.entry, "The spilled file must read back.")
        let plan = try #require(
            try slot.plan(
                input: LMInput(tokens: MLXArray(nextRender)), model: model,
                parameters: GenerateParameters(),
                protocolRules: [QwenCommittedTurnRule(endOfTurnToken: commit)],
                decodeTokens: { _ in "" }))
        return CommittedTurnPlan(
            restoredRenderTokens: restored.renderTokens, plan: plan, lines: lines)
    }
}

// MARK: - A session whose output depends on the restored cache

/// What one ``CacheEchoSession`` run measured.
private struct CacheEchoRun {

    /// The bytes that the store held in memory after the first turn.
    let retainedByteCount: Int

    /// The number of spill files after the first turn.
    let spilledFileCount: Int

    /// What the later turn streamed.
    let turn: ScriptedPassResult
}

/// Runs a two-turn session of a ``CacheEchoLanguageModel`` inside a store of a given memory
/// budget. The later turn restores the cache of the first turn from memory or from disk.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private enum CacheEchoSession: PromptCacheSpoolFixtures {

    /// A memory budget that keeps the cache of each run in memory.
    static let memoryBudgetBytes = 1 << 30

    /// The first entry of the session.
    private static let sessionID = "cache-echo-session"

    /// The number of prompts of the later turn of the session.
    private static let laterTurnCount = 2

    /// The value that ``changeValues(at:)`` adds to each value of a spill file.
    private static let valueChange = 1

    /// Runs the first turn and the later turn of a new session with a new model.
    ///
    /// - Parameters:
    ///   - budget: The memory budget of the store in bytes. Zero spills each check-in at once.
    ///   - changesSpilledValues: When true, adds ``valueChange`` to each value of each spill
    ///     file before the later turn. The header of each file stays valid.
    /// - Returns: What the run measured.
    /// - Throws: The error of a turn or of the file change.
    static func run(budget: Int, changesSpilledValues: Bool = false) async throws -> CacheEchoRun {
        let directory = temporaryDirectory()
        let weights = try makeScriptedWeightsDirectory()
        defer {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: weights)
        }
        let model = CacheEchoLanguageModel.make(weights: weights)
        let store = await store(in: directory, budget: budget)

        try await ScriptedExecutorPass.run(
            over: ScriptedSessionModel.transcript(firstEntryID: sessionID), model: model,
            inside: store)
        await store.waitForSpills()
        let files = try fileNames(in: directory)
        for name in files where changesSpilledValues {
            try changeValues(at: directory.appendingPathComponent(name))
        }
        let retainedByteCount = await store.retainedByteCount

        let turn = try await ScriptedExecutorPass.respond(
            over: ScriptedSessionModel.transcript(firstEntryID: sessionID, turns: laterTurnCount),
            model: model, inside: store)
        return CacheEchoRun(
            retainedByteCount: retainedByteCount, spilledFileCount: files.count, turn: turn)
    }

    /// Adds ``valueChange`` to each array value of the spill file at `url`. Each array keeps
    /// its type and its shape, and the metadata stays the same, thus the header stays valid.
    ///
    /// - Parameter url: The URL of the spill file.
    /// - Throws: The error of the safetensors load or save.
    private static func changeValues(at url: URL) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        let changed = arrays.mapValues { ($0 + valueChange).asType($0.dtype) }
        eval(changed.values)
        try save(arrays: changed, metadata: metadata, url: url)
    }
}

/// A model whose output depends on the values of its cache.
///
/// Each forward pass writes the token IDs of its input into each cache as keys and values. The
/// pass at step `n` of a round emits the token that the first cache holds at position `n`.
/// Thus a round echoes the start of the cached prompt, and a cache with wrong values gives a
/// wrong echo.
private final class CacheEchoLanguageModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{

    /// The number of tokens that each round emits before it stops.
    static let responseTokenCount = 4

    /// The key/value heads of the one cache layer.
    private static let cacheHeadCount = 1

    /// The width of each key and value that the model writes into its cache.
    private static let cacheHeadDimension = 1

    /// The logit of the token that the model emits.
    private static let selectedLogit: Float = 100

    /// The logit of each other token.
    private static let rejectedLogit: Float = -100

    /// One key/value head for the one cache layer.
    var kvHeads: [Int] { [Self.cacheHeadCount] }

    /// The number of forward passes of the round so far.
    private var step = 0

    /// Makes the model under a new identity, thus the process-wide model cache keeps it apart
    /// from each other test.
    ///
    /// - Parameter weights: The directory that makes the model available.
    /// - Returns: The model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    static func make(weights: URL) -> MLXLanguageModel {
        let modelID = "probe/cache-echo-\(UUID().uuidString)"
        let configuration = ModelConfiguration(id: modelID)
        return MLXLanguageModel(
            configuration: configuration,
            capabilities: [],
            weightsLocation: { _ in weights },
            load: { _, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration, model: CacheEchoLanguageModel(),
                        processor: PromptBytesInputProcessor(),
                        tokenizer: ScriptedByteTokenizer()))
            })
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        step = 0
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        defer { step += 1 }
        let cached = write(tokens: inputs, into: cache ?? [])
        let positions = Swift.max(inputs.size, 1)
        var logits = Array(
            repeating: Self.rejectedLogit, count: positions * ScriptedLanguageModel.vocabularySize)
        // Only the last row is read as the next-token distribution.
        logits[(positions - 1) * ScriptedLanguageModel.vocabularySize + nextToken(from: cached)] =
            Self.selectedLogit
        return MLXArray(logits, [1, positions, ScriptedLanguageModel.vocabularySize])
    }

    /// Writes the token IDs of `tokens` into each of `caches` as keys and values.
    ///
    /// - Parameters:
    ///   - tokens: The input of the forward pass.
    ///   - caches: The caches of the forward pass.
    /// - Returns: All the keys that the first cache holds after the write, in position order.
    private func write(tokens: MLXArray, into caches: [KVCache]) -> [Float] {
        let tokenCount = tokens.size
        guard tokenCount > 0 else { return [] }
        let keyValues = tokens.asType(.float32).reshaped([
            1, Self.cacheHeadCount, tokenCount, Self.cacheHeadDimension,
        ])
        let allKeys = caches.map { $0.update(keys: keyValues, values: keyValues).0 }
        return allKeys.first?.asType(.float32).asArray(Float.self) ?? []
    }

    /// The token that the pass emits: the cached token at the position of ``step``, or
    /// end-of-text after ``responseTokenCount`` tokens, or when the cache holds no valid token
    /// at that position.
    ///
    /// - Parameter cached: All the keys of the first cache, in position order.
    /// - Returns: The token ID.
    private func nextToken(from cached: [Float]) -> Int {
        guard step < Self.responseTokenCount, cached.indices.contains(step) else {
            return ScriptedByteTokenizer.endOfTextByte
        }
        let token = Int(cached[step])
        guard (0 ..< ScriptedLanguageModel.vocabularySize).contains(token) else {
            return ScriptedByteTokenizer.endOfTextByte
        }
        return token
    }
}

// MARK: - A render the test controls

/// What the next render of a ``GatedPromptBytesInputProcessor`` does.
private enum ScriptedRenderAction: Sendable {

    /// Renders the prompt.
    case render

    /// Holds until the task is cancelled.
    case hold

    /// Throws ``ScriptedRenderError/failed``.
    case fail
}

/// Why a gated render did not give a prompt.
private enum ScriptedRenderError: Error, Equatable {

    /// The test told the render to fail.
    case failed

    /// A held render was not cancelled in ``ScriptedRenderGate/holdLimit``.
    case holdExpired
}

/// Decides what the next render of a ``GatedPromptBytesInputProcessor`` does, and tells the
/// test when a render holds.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class ScriptedRenderGate: Sendable {

    /// How many seconds a held render waits for its cancellation before it fails.
    static let holdLimitSeconds = 60

    /// How long a held render waits for its cancellation before it fails.
    static let holdLimit: Duration = .seconds(holdLimitSeconds)

    /// One element for each render that starts to hold.
    let heldRenders: AsyncStream<Void>

    /// Receives one element for each render that starts to hold.
    private let holds: AsyncStream<Void>.Continuation

    /// What the next render does.
    private let action = Mutex(ScriptedRenderAction.render)

    /// Creates a gate whose renders render.
    init() {
        (heldRenders, holds) = AsyncStream.makeStream(of: Void.self)
    }

    /// What the next render does.
    var nextAction: ScriptedRenderAction {
        get { action.withLock { $0 } }
        set { action.withLock { $0 = newValue } }
    }

    /// Tells the test that a render holds, and waits for the cancellation of the task.
    ///
    /// - Throws: `CancellationError` when the task is cancelled, or
    ///   ``ScriptedRenderError/holdExpired`` after ``holdLimit``.
    func hold() async throws -> Never {
        holds.yield()
        try await Task.sleep(for: Self.holdLimit)
        throw ScriptedRenderError.holdExpired
    }
}

/// A ``PromptBytesInputProcessor`` whose render a ``ScriptedRenderGate`` can hold or fail. The
/// render runs inside the model container, after the check-out and before the restore.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct GatedPromptBytesInputProcessor: UserInputProcessor {

    /// Decides what each render does.
    let gate: ScriptedRenderGate

    /// Renders the prompt of `input`, holds, or fails, as the gate says.
    func prepare(input: UserInput) async throws -> LMInput {
        switch gate.nextAction {
        case .render:
            return try await PromptBytesInputProcessor().prepare(input: input)
        case .hold:
            try await gate.hold()
        case .fail:
            throw ScriptedRenderError.failed
        }
    }
}

#endif  // FoundationModelsIntegration
