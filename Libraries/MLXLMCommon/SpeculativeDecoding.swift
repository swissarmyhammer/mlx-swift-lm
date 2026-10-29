// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Why a configured ``ChatSession`` selected ordinary generation instead of speculation.
public enum SpeculativeDecodingFallbackReason: Sendable, Equatable {
    /// The drafter requires greedy sampling, but the temperature is nonzero.
    case unsupportedSampling

    /// The prepared input contains image, video, or audio tensors.
    case unsupportedMedia

    /// The initial MTP cache-staging probe refused speculation.
    case unsupportedCache

    /// Required speculative state is missing, misaligned, or cannot follow the cache operation.
    case unavailableContinuation

    /// The draft-model memory policy selected ordinary generation.
    case memoryBudgetExceeded

    /// The MTP drafter cannot draft for the session's model. Call
    /// ``MTPDrafterContainer/validateCompatibility(with:)`` to learn why.
    case incompatibleDrafter
}

private func defaultSpeculativeDecodingMemoryLimit() -> Int? {
    guard let bytes = GPU.maxRecommendedWorkingSetBytes(), bytes > 0 else {
        return nil
    }
    return bytes
}

/// Runtime counters for a speculative decoding pass.
public struct SpeculativeDecodingTelemetry: Sendable, Equatable {
    /// Number of speculative decoding rounds.
    public private(set) var roundCount: Int

    /// Number of tokens proposed by the draft model.
    public private(set) var draftTokenCount: Int

    /// Number of draft tokens accepted by the target model.
    public private(set) var acceptedDraftTokenCount: Int

    /// Number of target-model verification calls.
    public private(set) var targetModelCallCount: Int

    /// Number of draft-model calls.
    public private(set) var draftModelCallCount: Int

    /// Number of token positions evaluated by the target model during verification.
    public private(set) var targetVerifiedTokenCount: Int

    /// Number of tokens emitted from speculative rounds, including correction and bonus tokens.
    public private(set) var emittedTokenCount: Int

    /// The rounds that drafted a token at each depth: element `k` counts the rounds that drafted
    /// more than `k` tokens, so it describes depth `k + 1`.
    public private(set) var draftedRoundCountsByDepth: [Int] = []

    /// The rounds that accepted the draft at each depth: element `k` counts the rounds that
    /// accepted more than `k` drafts, so it describes depth `k + 1`.
    public private(set) var acceptedRoundCountsByDepth: [Int] = []

    /// The times the adaptive guard of an MTP stream paused speculation because its rounds lost
    /// to plain decode (see ``MTPAdaptivePolicy``).
    public private(set) var adaptivePauseCount = 0

    /// The times the adaptive guard of an MTP stream probed plain decode to measure it.
    public private(set) var adaptiveProbeCount = 0

    /// The tokens the adaptive guard of an MTP stream decoded plainly, in its probes and pauses.
    /// ``emittedTokenCount`` counts them too.
    public private(set) var adaptivePlainTokenCount = 0

    /// The times the adaptive guard of an MTP stream changed its block size.
    public private(set) var adaptiveBlockSizeChangeCount = 0

    /// The block size the adaptive guard of an MTP stream last chose, or `nil` for a stream
    /// without a guard.
    public private(set) var adaptiveBlockSize: Int?

    public init(
        roundCount: Int = 0,
        draftTokenCount: Int = 0,
        acceptedDraftTokenCount: Int = 0,
        targetModelCallCount: Int = 0,
        draftModelCallCount: Int = 0,
        targetVerifiedTokenCount: Int = 0,
        emittedTokenCount: Int = 0
    ) {
        self.roundCount = roundCount
        self.draftTokenCount = draftTokenCount
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetModelCallCount = targetModelCallCount
        self.draftModelCallCount = draftModelCallCount
        self.targetVerifiedTokenCount = targetVerifiedTokenCount
        self.emittedTokenCount = emittedTokenCount
    }

    /// Number of draft tokens rejected by the target model.
    public var rejectedDraftTokenCount: Int {
        max(0, draftTokenCount - acceptedDraftTokenCount)
    }

    /// Fraction of drafted tokens accepted by the target model.
    public var acceptanceRate: Double {
        guard draftTokenCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(draftTokenCount)
    }

    /// Mean accepted draft tokens per speculative round.
    public var meanAcceptedDraftTokensPerRound: Double {
        guard roundCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(roundCount)
    }

    /// Mean emitted tokens per target-model verification call.
    public var meanEmittedTokensPerTargetCall: Double {
        guard targetModelCallCount > 0 else { return 0 }
        return Double(emittedTokenCount) / Double(targetModelCallCount)
    }

    /// The share of the rounds that drafted a token at `depth` whose draft there the target
    /// accepted, or `nil` when no round drafted that deep.
    ///
    /// - Parameter depth: The position of the draft in its round: 1 is the first draft after the
    ///   bonus token.
    /// - Returns: The acceptance rate at `depth`, from 0 to 1.
    public func acceptanceRate(atDepth depth: Int) -> Double? {
        let index = depth - 1
        guard draftedRoundCountsByDepth.indices.contains(index) else { return nil }
        return Double(acceptedRoundCountsByDepth[index])
            / Double(draftedRoundCountsByDepth[index])
    }

    package mutating func recordRound(
        drafted: Int,
        accepted: Int,
        targetVerified: Int,
        draftModelCalls: Int? = nil
    ) {
        roundCount += 1
        draftTokenCount += drafted
        acceptedDraftTokenCount += accepted
        targetModelCallCount += 1
        draftModelCallCount += draftModelCalls ?? drafted
        targetVerifiedTokenCount += targetVerified
        Self.countRound(
            throughDepth: drafted, in: &draftedRoundCountsByDepth, depthCount: drafted)
        Self.countRound(
            throughDepth: accepted, in: &acceptedRoundCountsByDepth,
            depthCount: draftedRoundCountsByDepth.count)
    }

    /// Counts one round at each depth from 1 to `depth`, after growing `counts` with zeros to
    /// `depthCount` depths.
    private static func countRound(
        throughDepth depth: Int, in counts: inout [Int], depthCount: Int
    ) {
        counts.append(contentsOf: repeatElement(0, count: Swift.max(0, depthCount - counts.count)))
        for index in 0 ..< Swift.max(0, depth) {
            counts[index] += 1
        }
    }

    package mutating func recordGeneratedToken() {
        emittedTokenCount += 1
    }

    package mutating func discardGeneratedToken() {
        emittedTokenCount = max(0, emittedTokenCount - 1)
    }

    /// Counts one token the adaptive guard decoded plainly.
    package mutating func recordAdaptivePlainToken() {
        adaptivePlainTokenCount += 1
    }

    /// Takes the decisions of an adaptive guard so far: its pauses, its probes, its block size
    /// changes, and the block size it last chose.
    ///
    /// - Parameter controller: The decisions of the guard.
    package mutating func recordAdaptiveDecisions(of controller: MTPAdaptiveController) {
        adaptivePauseCount = controller.pauseCount
        adaptiveProbeCount = controller.probeCount
        adaptiveBlockSizeChangeCount = controller.blockSizeChangeCount
        adaptiveBlockSize = controller.blockSize
    }
}

/// Action to take when speculative decoding exceeds a memory budget.
public enum SpeculativeDecodingMemoryAction: Sendable, Hashable {
    /// Use speculative decoding even if the estimate exceeds the budget.
    case allow

    /// Fall back to regular generation.
    case fallbackToDefault

    /// Throw an error instead of silently falling back.
    case fail
}

/// Result of evaluating speculative decoding against a memory policy.
public struct SpeculativeDecodingMemoryEvaluation: Sendable, Equatable {
    /// Estimated main-model parameter bytes.
    public let mainModelBytes: Int

    /// Estimated draft-model parameter bytes.
    public let draftModelBytes: Int

    /// Additional caller-provided budget for KV cache, workspace, or other resident data.
    public let additionalBytes: Int

    /// Total estimated resident bytes for speculative decoding.
    public var estimatedBytes: Int {
        mainModelBytes + draftModelBytes + additionalBytes
    }

    /// Memory budget used for the decision. `nil` means no budget was available.
    public let limitBytes: Int?

    /// Action selected by the policy.
    public let action: SpeculativeDecodingMemoryAction

    /// Whether speculative decoding is within the available budget.
    public var isWithinBudget: Bool {
        guard let limitBytes else { return true }
        return estimatedBytes <= limitBytes
    }

    /// Whether speculative decoding should be used.
    public var shouldUseSpeculativeDecoding: Bool {
        isWithinBudget || action == .allow
    }
}

/// Policy for gating auxiliary-model speculative decoding by resident memory estimates.
public struct SpeculativeDecodingMemoryPolicy: Sendable, Hashable {
    /// Optional absolute budget in bytes. When nil, no budget is enforced.
    public let limitBytes: Int?

    /// Extra bytes to reserve for KV cache, workspace, or application memory.
    public let additionalBytes: Int

    /// Action to take when the estimate exceeds the budget.
    public let action: SpeculativeDecodingMemoryAction

    public init(
        limitBytes: Int? = nil,
        additionalBytes: Int = 0,
        action: SpeculativeDecodingMemoryAction = .fallbackToDefault
    ) {
        self.limitBytes = limitBytes
        self.additionalBytes = max(0, additionalBytes)
        self.action = action
    }

    /// Default policy using `GPU.maxRecommendedWorkingSetBytes()` when available.
    public static var recommendedWorkingSet: Self {
        Self(limitBytes: defaultSpeculativeDecodingMemoryLimit())
    }

    /// Evaluate explicit byte estimates. This is useful before loading a draft model.
    public func evaluate(
        mainModelBytes: Int,
        draftModelBytes: Int
    ) -> SpeculativeDecodingMemoryEvaluation {
        SpeculativeDecodingMemoryEvaluation(
            mainModelBytes: max(0, mainModelBytes),
            draftModelBytes: max(0, draftModelBytes),
            additionalBytes: additionalBytes,
            limitBytes: limitBytes,
            action: action
        )
    }

    /// Whether speculation runs under this policy for the given byte estimates.
    ///
    /// - Parameters:
    ///   - mainModelBytes: The estimated bytes of the main model.
    ///   - draftModelBytes: The estimated bytes of the draft model or MTP drafter, with any
    ///     drafter state the caller carries.
    /// - Returns: `true` when the policy admits speculation, `false` when it falls back to
    ///   decoding without speculation.
    /// - Throws: ``SpeculativeDecodingMemoryError`` when the policy does not admit speculation
    ///   and its action is ``SpeculativeDecodingMemoryAction/fail``.
    package func admitsSpeculation(mainModelBytes: Int, draftModelBytes: Int) throws -> Bool {
        let evaluation = evaluate(mainModelBytes: mainModelBytes, draftModelBytes: draftModelBytes)
        guard evaluation.shouldUseSpeculativeDecoding else {
            if evaluation.action == .fail {
                throw SpeculativeDecodingMemoryError(evaluation: evaluation)
            }
            return false
        }
        return true
    }

    /// Whether speculation with `draftModel` for `mainModel` runs under this policy.
    ///
    /// - Parameters:
    ///   - mainModel: The main model.
    ///   - draftModel: The loaded draft model.
    /// - Returns: `true` when the policy admits speculation, `false` when it falls back to
    ///   decoding without speculation.
    /// - Throws: ``SpeculativeDecodingMemoryError`` when the policy does not admit speculation
    ///   and its action is ``SpeculativeDecodingMemoryAction/fail``.
    package func admitsSpeculation(
        mainModel: any LanguageModel, draftModel: any LanguageModel
    ) throws -> Bool {
        try admitsSpeculation(
            mainModelBytes: Self.modelWeightBytes(mainModel),
            draftModelBytes: Self.modelWeightBytes(draftModel))
    }

    /// The bytes of the parameters of `model`.
    ///
    /// - Parameter model: A target, draft or MTP drafter model.
    /// - Returns: The sum of the bytes of its parameters.
    package static func modelWeightBytes(_ model: Module) -> Int {
        model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }
}

public struct SpeculativeDecodingMemoryError: Error, Sendable {
    public let evaluation: SpeculativeDecodingMemoryEvaluation

    public init(evaluation: SpeculativeDecodingMemoryEvaluation) {
        self.evaluation = evaluation
    }
}
