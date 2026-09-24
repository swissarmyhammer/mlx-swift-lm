// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Synchronization
import Testing

@testable import MLXFoundationModels

/// Proves that `MLXLanguageModel.Executor` restores a prompt cache that went to disk, and that
/// the file of that cache never stays after the turn.
///
/// Each test binds its own store with `ExecutorPromptCacheStore.$current`. The memory budget
/// of the store is zero, thus each check-in goes to disk at once, and a later turn of the
/// session finds its cache on disk only.
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

    // MARK: - The checks

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
