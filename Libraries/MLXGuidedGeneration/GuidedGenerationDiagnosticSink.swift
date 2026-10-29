// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Diagnostic capture for the tool-call grammar/tokenizer boundary and the MTP rounds of a
/// guided generation.
///
/// A test binds an instance with `GuidedGenerationDiagnosticSink.$current.withValue(_:)`
/// around a real generation call, drives the executor, then reads the accumulated signals.
/// Outside a test, `MLXFoundationModels` binds a sink made with `recordsDecoding: false`
/// around each guided pass of a call that speculates, and reads the MTP rounds and
/// fallbacks of the pass for the response metadata of the call. Such a sink records no
/// decision, margin or token, so the decoder pays nothing for it. With no sink bound, every
/// recording site is a nil-guarded no-op.
///
/// Reference type so writes made deep in the synchronous generation loop are
/// visible to the test afterward. `@unchecked Sendable` because it is mutated
/// only within the single generation task, never concurrently.
public final class GuidedGenerationDiagnosticSink: @unchecked Sendable {

    /// Task-local injection point: a test's sink, the speculation-only sink of a guided pass
    /// that speculates, or `nil`.
    @TaskLocal public static var current: GuidedGenerationDiagnosticSink?

    /// Whether the decoder records its decisions, margins, tokens and termination here. A sink
    /// that does not records only the MTP rounds, rollbacks and fallbacks.
    public let recordsDecoding: Bool

    /// Model-sampled token IDs, in generation order. Grammar-forced tokens are
    /// recorded separately in `fastForwardTokenIDs`.
    public private(set) var sampledTokenIDs: [Int] = []

    /// Grammar-forced fast-forward token IDs, in generation order.
    public private(set) var fastForwardTokenIDs: [Int] = []

    /// Whether the grammar reached a stop state (vs. exhausting the budget).
    public private(set) var grammarTerminated = false

    /// Total tokens generated (sampled + fast-forward), as the loop counts them.
    public private(set) var generatedTokenCount = 0

    /// The exact buffer handed to the tool-call parser.
    public private(set) var finalBuffer: String?

    /// True when generation hit the token budget before the grammar stopped.
    public private(set) var incompleteOutput = false

    /// Whether `finalBuffer` parsed as a valid tool call (an object with a
    /// `name` field). `nil` until the parser runs.
    public private(set) var parsedAsToolCall: Bool?

    /// The parsed tool name, when the buffer parsed as a tool call.
    public private(set) var parsedName: String?

    /// Number of synchronous executor-side guided emit boundaries observed.
    public private(set) var emitCount = 0

    /// Number of required-tool reasoning close boundaries observed.
    public private(set) var toolReasoningCloseCount = 0

    /// The position of each decision, in generation order: plain decoding records one per
    /// decided token, and an MTP round one per draft the target accepted and one per token the
    /// target decided itself.
    public private(set) var decisions: [GuidedDecisionRecord] = []

    /// The gap between the two best constrained logits of each token the target decided from
    /// its own logits, in order: every sampled token of plain decoding, and the token after the
    /// accepted drafts of an MTP round. A gap near zero marks a near-tie, where decoding noise
    /// can flip the decision.
    public private(set) var decisionMargins: [Float] = []

    /// The drafts the MTP rounds proposed.
    public private(set) var draftedTokenCount = 0

    /// The drafts the target accepted.
    public private(set) var acceptedDraftTokenCount = 0

    /// Each rollback of the grammar matcher after a rejected draft, in order.
    public private(set) var rollbacks: [GuidedRollbackRecord] = []

    /// Why the configured MTP drafter did not speculate, in words, when it cannot draft for
    /// the target; `nil` otherwise.
    public private(set) var speculationFallbackReason: String?

    /// Why each guided generation whose configured MTP drafter cannot draft for the target
    /// decoded plainly, one reason for each such generation, in order.
    public private(set) var speculationFallbacks: [SpeculativeDecodingFallbackReason] = []

    private let cancelAfterEmitCount: Int?
    private let cancelOnToolReasoningClose: Bool

    /// Creates a sink.
    ///
    /// - Parameters:
    ///   - cancelAfterEmitCount: The emit boundary at which ``recordEmit()`` cancels its task,
    ///     or `nil` for none.
    ///   - cancelOnToolReasoningClose: Whether ``recordToolReasoningClose()`` cancels its task.
    ///   - recordsDecoding: Whether the decoder records its decisions, margins, tokens and
    ///     termination; `false` records only the MTP rounds, rollbacks and fallbacks.
    public init(
        cancelAfterEmitCount: Int? = nil,
        cancelOnToolReasoningClose: Bool = false,
        recordsDecoding: Bool = true
    ) {
        self.cancelAfterEmitCount = cancelAfterEmitCount
        self.cancelOnToolReasoningClose = cancelOnToolReasoningClose
        self.recordsDecoding = recordsDecoding
    }

    // MARK: Recording (called from generation; no-ops when the sink is unbound)

    public func recordSampledToken(_ id: Int) {
        sampledTokenIDs.append(id)
    }

    public func recordFastForwardToken(_ id: Int) {
        fastForwardTokenIDs.append(id)
    }

    public func recordTermination(grammarTerminated: Bool, generatedTokenCount: Int) {
        self.grammarTerminated = grammarTerminated
        self.generatedTokenCount = generatedTokenCount
    }

    public func recordBuffer(_ buffer: String, incompleteOutput: Bool) {
        self.finalBuffer = buffer
        self.incompleteOutput = incompleteOutput
    }

    public func recordParse(parsedAsToolCall: Bool, parsedName: String?) {
        self.parsedAsToolCall = parsedAsToolCall
        self.parsedName = parsedName
    }

    /// Records a guided loop's synchronous emit boundary and optionally
    /// cancels its calling task. Cancellation is opt-in and used only by tests
    /// that bind this sink through ``current``.
    public func recordEmit() {
        emitCount += 1
        if emitCount == cancelAfterEmitCount {
            withUnsafeCurrentTask { $0?.cancel() }
        }
    }

    /// Records the boundary where Phase 1 has consumed the reasoning close
    /// marker and optionally cancels its calling task.
    public func recordToolReasoningClose() {
        toolReasoningCloseCount += 1
        if cancelOnToolReasoningClose {
            withUnsafeCurrentTask { $0?.cancel() }
        }
    }

    /// Records the position of one decision.
    ///
    /// - Parameter decision: The position's mask and bias.
    public func recordDecision(_ decision: GuidedDecisionRecord) {
        decisions.append(decision)
    }

    /// Records the gap between the two best constrained logits of one decision.
    ///
    /// - Parameter margin: The gap.
    public func recordDecisionMargin(_ margin: Float) {
        decisionMargins.append(margin)
    }

    /// Records one MTP round.
    ///
    /// - Parameters:
    ///   - draftedTokenCount: The drafts the round proposed.
    ///   - acceptedDraftTokenCount: The drafts the target accepted.
    public func recordSpeculativeRound(draftedTokenCount: Int, acceptedDraftTokenCount: Int) {
        self.draftedTokenCount += draftedTokenCount
        self.acceptedDraftTokenCount += acceptedDraftTokenCount
    }

    /// Records one rollback of the grammar matcher.
    ///
    /// - Parameter rollback: The drafts and the matcher acceptances the rollback undid.
    public func recordRollback(_ rollback: GuidedRollbackRecord) {
        rollbacks.append(rollback)
    }

    /// Records why the configured MTP drafter decodes plainly.
    ///
    /// - Parameters:
    ///   - statedReason: The drafter, the target, and why the drafter cannot draft for the
    ///     target.
    ///   - reason: The fallback reason the generation reports.
    public func recordSpeculationFallback(
        _ statedReason: String, reason: SpeculativeDecodingFallbackReason
    ) {
        speculationFallbackReason = statedReason
        speculationFallbacks.append(reason)
    }
}

/// The position of one guided decision: the grammar mask and the logit bias the decision read.
public struct GuidedDecisionRecord: Equatable, Sendable {

    /// The tokens the generation counted before the decision.
    public let tokenCount: Int

    /// The grammar's packed token bitmask.
    public let mask: [Int32]

    /// Whether the grammar mask excludes tokens.
    public let needsApply: Bool

    /// Whether the grammar had terminated.
    public let isTerminated: Bool

    /// The bias the decision added to the logits, or `nil` for none.
    public let bias: [Float]?
}

/// One rollback of the grammar matcher after the target rejected a draft.
public struct GuidedRollbackRecord: Equatable, Sendable {

    /// The drafts the rollback undid: the rejected draft and the drafts after it.
    public let rejectedDraftCount: Int

    /// The tokens the matcher rolled back: the drafts and the tokens each forced.
    public let acceptanceCount: Int
}
