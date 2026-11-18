// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import Synchronization

// MARK: - Carrying the cache through one response

/// Carries the prompt cache of one session through one response.
///
/// `MLXLanguageModel.Executor.respond(to:model:streamingInto:)` checks the
/// session's cache out before it generates and checks this slot's entry back in
/// when it finishes, whatever the outcome.
///
/// The slot writes one log line when a pass is planned and one when it is
/// committed, at `info` level in the `com.apple.FoundationModels-MLX`
/// subsystem, thus `log show` names every rebuild of an agent run.
///
/// A session whose cache went to disk gives the slot a pending restore. The
/// restore reads the file into the fresh caches of the model, thus it runs
/// inside the model container: ``restoreIfPending(model:parameters:speculation:)`` runs it
/// one time, before the first plan. The executor, not the slot, deletes the
/// file.
///
/// A response runs one generation pass at a time, and every pass runs inside the
/// model container, which serializes the passes of one model. This box is thus
/// written by one task at a time, and the `@unchecked Sendable` conformance
/// rests on that.
// swiftlint:disable:next no_unchecked_sendable Every pass of one response runs inside the model container, which serializes the passes of one model, thus one task at a time writes the slot.
final class ExecutorPromptCacheSlot: @unchecked Sendable {

    /// What the slot holds for the next pass.
    private enum Carried {

        /// No cache: the next pass starts cold.
        case none

        /// A cache that stayed in memory.
        case memory(ExecutorPromptCacheEntry)

        /// A cache that the slot read back from disk, and how long the read took.
        case restored(ExecutorPromptCacheEntry, restoreDuration: Duration)

        /// A cache on disk that the slot did not read yet.
        case pendingRestore(ExecutorPromptCacheSpilledHandle)
    }

    /// What the slot holds for the next pass.
    private var carried: Carried

    /// The cache the session carries into its next turn, or nil when the next
    /// turn must start cold. A cache on disk that the slot did not read is not
    /// an entry.
    var entry: ExecutorPromptCacheEntry? {
        switch carried {
        case .memory(let entry), .restored(let entry, _): entry
        case .none, .pendingRestore: nil
        }
    }

    /// Where the cache of the next pass comes from.
    private var source: ExecutorPromptCacheSource {
        switch carried {
        case .memory: .memory
        case .restored(_, let restoreDuration): .disk(restoreDuration: restoreDuration)
        case .none, .pendingRestore: .none
        }
    }

    /// The prompt tokens the last planned pass did not feed to the model.
    ///
    /// Zero until a pass carries a cache. A pass that builds its own cache --
    /// the guided loop does -- leaves the value at zero, which is the measured
    /// truth for that pass.
    private(set) var reusedTokenCount = 0

    /// The prompt tokens the last planned pass fed to the model before its
    /// generation started, to take a checkpoint. The generation does not
    /// count them, thus the usage of the pass adds them back.
    private(set) var prefilledTokenCount = 0

    /// The tokens of the previous answer that the last planned pass fed ahead
    /// of its prompt (see ``ExecutorPromptCachePlan/uncommittedTokens``). The
    /// generation or the checkpoint feed counts them among the tokens it fed,
    /// and they are no prompt tokens, thus the usage of the pass takes them
    /// out.
    private(set) var uncommittedTokenCount = 0

    /// The plan of the pass that takes its checkpoint at the end of the
    /// prompt, when the generation reports that its prefill ended. Nil when
    /// the pass takes no checkpoint there.
    private var promptEndCheckpointPlan: ExecutorPromptCachePlan?

    /// The checkpoint the pass took at the end of its prompt, or nil.
    private var promptEndCheckpoint: PromptCacheCheckpoint?

    /// The session the cache belongs to, which every log line names, or nil
    /// when the request names no session.
    private let key: ExecutorPromptCacheKey?

    /// Receives each log line. The unified log by default; a test reads the
    /// lines through its own sink.
    private let report: (String) -> Void

    /// Creates a slot holding the entry a session checked out.
    ///
    /// - Parameters:
    ///   - entry: the entry the session checked out, or nil for a cold session.
    ///   - key: the session the entry belongs to, or nil when the request
    ///     names no session.
    ///   - report: receives each log line of the response.
    convenience init(
        _ entry: ExecutorPromptCacheEntry?, key: ExecutorPromptCacheKey? = nil,
        report: @escaping (String) -> Void = ExecutorPromptCacheLog.info
    ) {
        self.init(carrying: entry.map { .memory($0) } ?? .none, key: key, report: report)
    }

    /// Creates a slot whose session found its cache on disk. The slot reads
    /// the file at ``restoreIfPending(model:parameters:speculation:)``.
    ///
    /// - Parameters:
    ///   - handle: the file of the cache, which the check-out gave.
    ///   - key: the session the cache belongs to.
    ///   - report: receives each log line of the response.
    convenience init(
        spilled handle: ExecutorPromptCacheSpilledHandle, key: ExecutorPromptCacheKey?,
        report: @escaping (String) -> Void = ExecutorPromptCacheLog.info
    ) {
        self.init(carrying: .pendingRestore(handle), key: key, report: report)
    }

    /// Creates a slot holding what a check-out of the session gave.
    ///
    /// - Parameters:
    ///   - checkout: the result of the check-out.
    ///   - key: the session of the request, or nil when the request names no
    ///     session.
    convenience init(_ checkout: ExecutorPromptCacheCheckout, key: ExecutorPromptCacheKey?) {
        switch checkout {
        case .memory(let entry): self.init(entry, key: key)
        case .spilled(let handle): self.init(spilled: handle, key: key)
        case .none: self.init(nil, key: key)
        }
    }

    /// Creates a slot that holds `carried`.
    private init(
        carrying carried: Carried, key: ExecutorPromptCacheKey?,
        report: @escaping (String) -> Void
    ) {
        self.carried = carried
        self.key = key
        self.report = report
    }

    /// Reads the cache on disk into the fresh caches of `model`, when the
    /// slot holds a cache on disk that it did not read yet.
    ///
    /// The executor calls this method before each plan. Only the first call
    /// reads. A file that does not read back -- a missing file, a corrupt
    /// file, another cache class, a wrong offset -- writes one log line, and
    /// the pass starts cold: a failed restore never fails the turn.
    ///
    /// A drafter state in the file that does not read back writes its own log
    /// line, and only the drafter of the pass starts cold.
    ///
    /// - Parameters:
    ///   - model: the model that makes the fresh caches.
    ///   - parameters: the generation parameters the fresh caches must match.
    ///   - speculation: the MTP drafter of the pass, which reads the drafter
    ///     state of the file, or nil. Without a resumable drafter the entry
    ///     keeps no drafter state.
    func restoreIfPending(
        model: any LanguageModel, parameters: GenerateParameters,
        speculation: MTPSpeculation? = nil
    ) {
        guard case .pendingRestore(let handle) = carried else { return }
        carried = .none
        do {
            let restored = try ExecutorPromptCacheTiming.measure {
                try ExecutorPromptCacheFile.read(
                    from: handle.url, key: handle.key,
                    templates: model.newCache(parameters: parameters),
                    drafter: speculation?.resumableDrafter, parameters: parameters, report: report)
            }
            carried = .restored(restored.value, restoreDuration: restored.duration)
        } catch {
            report(ExecutorPromptCacheReport.restoreFailureLine(key: handle.key, error: error))
        }
    }

    /// Plans the pass that is about to run, records what that pass reuses,
    /// and logs the decision.
    ///
    /// The slot gives up its entry: the pass owns the caches until
    /// ``commit(_:generatedTokens:state:drafterState:)`` takes them back.
    ///
    /// - Parameters:
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    ///   - transcriptBoundary: the index of the prompt where a next render
    ///     of the same messages plus one new message parts from it, or nil
    ///     when the pass takes no checkpoint. The pass takes a checkpoint of
    ///     a hybrid model there: before its generation when the boundary is
    ///     before the end of the prompt, and when
    ///     ``prefillDidEnd(state:drafterState:)`` reports the end of the prefill when the
    ///     boundary is the end of the prompt.
    ///   - speculation: the MTP drafter of the pass, or nil. The plan resumes
    ///     the drafter state the entry or its checkpoint keeps for that
    ///     drafter, and a checkpoint before the end of the prompt keeps the
    ///     drafter state at its position.
    ///   - decodeTokens: decodes the tokens on each side of a seam for the log
    ///     line of a rewind, a restore or a rebuild.
    /// - Returns: the plan, or nil when the pass carries no cache.
    /// - Throws: the error of the new caches, or `CancellationError` when the
    ///   task is cancelled while the pass feeds the prompt up to the
    ///   checkpoint.
    func plan(
        input: LMInput, model: any LanguageModel, parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = [], transcriptBoundary: Int? = nil,
        speculation: MTPSpeculation? = nil, decodeTokens: ([Int]) -> String
    ) throws -> ExecutorPromptCachePlan? {
        let carriedSource = source
        var plan = try ExecutorPromptCachePlan.make(
            reusing: entry, input: input, model: model, parameters: parameters,
            protocolRules: protocolRules, drafterID: speculation?.resumableDrafter?.drafterID)
        carried = .none
        reusedTokenCount = plan?.reusedTokenCount ?? 0
        report(
            ExecutorPromptCacheReport.planLine(
                key: key, source: carriedSource, plan: plan, decodeTokens: decodeTokens))
        if let planned = plan, let transcriptBoundary {
            plan = try planned.prefillingToCheckpoint(
                at: transcriptBoundary, model: model, parameters: parameters,
                speculation: speculation)
            promptEndCheckpointPlan =
                transcriptBoundary == planned.promptTokens.count ? planned : nil
        }
        prefilledTokenCount = plan?.prefilledTokenCount ?? 0
        uncommittedTokenCount = plan?.uncommittedTokens.count ?? 0
        return plan
    }

    /// Takes the checkpoint at the end of the prompt, when the planned pass
    /// asked for one there.
    ///
    /// The generation calls this method when its prefill ended and before it
    /// feeds its first generated token, thus the caches stand at the end of
    /// the prompt.
    ///
    /// - Parameters:
    ///   - state: the model state the prefill left.
    ///   - drafterState: reads the MTP drafter continuation at the end of the
    ///     prompt, or gives nil when the generation does not speculate with a
    ///     resumable drafter. It is read only when the pass takes the
    ///     checkpoint, because the read copies the drafter caches.
    func prefillDidEnd(
        state: LMOutput.State?, drafterState: () -> MTPDrafterContinuation? = { nil }
    ) {
        promptEndCheckpoint = promptEndCheckpointPlan.flatMap {
            $0.checkpointAtPromptEnd(state: state, drafterState: drafterState())
        }
        promptEndCheckpointPlan = nil
    }

    /// Records that the pass about to run carries no cache from an earlier turn.
    ///
    /// The guided loop owns its key/value cache and accepts none from a caller,
    /// thus a guided pass reuses nothing and must report nothing, even when an
    /// earlier pass of the same response reused a prefix.
    ///
    /// The slot also gives up what it carries. The ledger of that cache does
    /// not record the guided pass, thus the response checks in nothing and the
    /// next turn of the session starts cold.
    func carriesNoCache() {
        carried = .none
        reusedTokenCount = 0
        prefilledTokenCount = 0
        uncommittedTokenCount = 0
        promptEndCheckpointPlan = nil
        promptEndCheckpoint = nil
        report(ExecutorPromptCacheReport.guidedLine(key: key))
    }

    /// Records the cache a finished pass leaves for the next turn, and logs
    /// the ledger length or the reason nothing is left.
    ///
    /// The commit ends the pass: it drops the prompt-end checkpoint plan and
    /// the checkpoint of the pass, thus a later pass of the slot never takes a
    /// checkpoint from a plan of an earlier pass whose prefill did not report
    /// its end.
    ///
    /// - Parameters:
    ///   - plan: the plan the pass ran, or nil when the pass carried no plan.
    ///     The entry carries the checkpoint of the plan, or else the
    ///     checkpoint ``prefillDidEnd(state:drafterState:)`` took.
    ///   - generatedTokens: the tokens the pass generated, in order.
    ///   - state: the model state the prefill of the pass left, or nil.
    ///   - drafterState: the MTP drafter continuation the generation of the
    ///     pass handed over when it ended, or nil.
    func commit(
        _ plan: ExecutorPromptCachePlan?, generatedTokens: [Int], state: LMOutput.State? = nil,
        drafterState: MTPDrafterContinuation? = nil
    ) {
        var plan = plan
        if plan?.checkpoint == nil {
            plan?.checkpoint = promptEndCheckpoint
        }
        promptEndCheckpointPlan = nil
        promptEndCheckpoint = nil
        let outcome =
            plan?.commitOutcome(
                generatedTokens: generatedTokens, state: state, drafterState: drafterState)
            ?? .refused(.noPlan)
        carried = outcome.entry.map { .memory($0) } ?? .none
        report(ExecutorPromptCacheReport.commitLine(key: key, outcome: outcome))
    }
}

/// Holds one value a generation hands over when it ends, until the pass commits
/// its cache.
///
/// The generation task hands the value over and the response task takes it
/// after it awaited that task, thus a `Mutex` holds it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class ExecutorHandOffReceiver<Value: Sendable>: Sendable {

    /// The value the generation handed over, or nil.
    private let received = Mutex<Value?>(nil)

    /// Creates a receiver that holds no value.
    init() {}

    /// The callback a generation hands its value to when it ends.
    var receive: @Sendable (Value) -> Void {
        { value in self.received.withLock { $0 = value } }
    }

    /// The value the generation handed over, or nil when it handed none or did
    /// not end yet. The receiver gives the value up: a second take gives nil.
    func take() -> Value? {
        received.withLock { $0.take() }
    }
}

/// Holds what the generation of one executor pass hands over when it ends --
/// the MTP drafter continuation and the caches it last wrote into -- until the
/// pass commits its cache.
///
/// Each pass makes one, hands ``generationHandOff`` to its generation, and
/// commits through ``commit(_:generatedTokens:state:to:)`` once that
/// generation ended.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct ExecutorGenerationHandOff: Sendable {

    /// Receives the MTP drafter continuation of the generation, which the next
    /// turn resumes.
    private let drafterState = ExecutorHandOffReceiver<MTPDrafterContinuation?>()

    /// Receives the caches the generation last wrote into, which the next turn
    /// carries in the place of the caches of the plan.
    private let generationCache = ExecutorHandOffReceiver<SendableBox<[KVCache]>>()

    /// Creates a hand-off that holds nothing yet.
    init() {}

    /// The hand-off a generation gives what it left to when it ends.
    var generationHandOff: GenerationHandOff {
        GenerationHandOff(
            drafterContinuation: drafterState.receive,
            generationCache: generationCache.receive)
    }

    /// Commits the cache of the pass to `promptCache`, with the caches and the
    /// drafter continuation the generation handed over.
    ///
    /// - Parameters:
    ///   - plan: the plan that the pass ran, or nil when the pass had no plan.
    ///   - generatedTokens: the tokens the generation fed into the caches.
    ///   - state: the model state the prefill of the pass left.
    ///   - promptCache: the slot that keeps the cache for the next turn.
    func commit(
        _ plan: ExecutorPromptCachePlan?, generatedTokens: [Int], state: LMOutput.State?,
        to promptCache: ExecutorPromptCacheSlot
    ) {
        promptCache.commit(
            plan?.holding(generationCache: generationCache.take()?.consume()),
            generatedTokens: generatedTokens, state: state,
            drafterState: drafterState.take() ?? nil)
    }
}

/// Records what the prefill of one executor pass left, and hands the end of that prefill to
/// the prompt cache slot of the pass.
///
/// The generation calls ``record(state:drafterState:)`` once, while it builds its iterator and
/// before the pass reads its first token. The pass then commits ``state`` with its cache.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class ExecutorPrefillRecorder {

    /// The slot that takes the checkpoint at the end of the prompt.
    private let promptCache: ExecutorPromptCacheSlot

    /// The model state the prefill left, or nil before the prefill ended.
    private(set) var state: LMOutput.State?

    /// Creates a recorder that hands the end of the prefill to `promptCache`.
    ///
    /// - Parameter promptCache: the slot of the pass.
    init(promptCache: ExecutorPromptCacheSlot) {
        self.promptCache = promptCache
    }

    /// Records the model state the prefill left and reports the end of the prefill to the
    /// slot. A generation takes this method as its `preparedState` callback.
    ///
    /// - Parameters:
    ///   - state: the model state the prefill left.
    ///   - drafterState: reads the MTP drafter continuation at the end of the prompt, as
    ///     ``ExecutorPromptCacheSlot/prefillDidEnd(state:drafterState:)`` takes it.
    func record(state: LMOutput.State?, drafterState: () -> MTPDrafterContinuation?) {
        self.state = state
        promptCache.prefillDidEnd(state: state, drafterState: drafterState)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
