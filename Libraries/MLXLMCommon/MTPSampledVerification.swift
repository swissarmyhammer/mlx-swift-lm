// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How an MTP stream with a stateful drafter (Qwen MTP) verifies its drafts when it samples,
/// that is at a temperature above zero.
///
/// Both modes emit tokens with the distribution of plain decode. At temperature zero neither
/// applies: the drafts are argmax tokens, and the verifier keeps the drafts its argmax tokens
/// match.
public enum MTPSampledVerification: Sendable, CaseIterable {

    /// The drafter proposes argmax tokens, which read no random state. The verifier samples
    /// each verify row with the stream's sampler and keeps the drafts its samples match, then
    /// emits its first sample after them.
    ///
    /// A draft is accepted with the probability the target gives the draft token. The
    /// default: one GPU readback verifies a round, and on Qwen3.8-27B-4bit at temperature 0.7
    /// and top-p 0.95 it decoded faster than ``rejectionSampling`` overall.
    case sampleAndMatch

    /// The drafter samples each proposal from its own distribution `q`, with a random state of
    /// its own. The verifier accepts a draft `x` with probability `min(1, p(x) / q(x))`, and on a
    /// rejection samples the correction from the normalized residual `max(p - q, 0)`.
    ///
    /// A draft is accepted with probability `1 - TV(p, q)`.
    case rejectionSampling
}

/// The batched verify walks of an MTP round: the target samples every verify row in one call,
/// and one GPU readback decides how many drafts the samples match.
///
/// A walk keeps the drafts the target samples match and then emits the first sample after them,
/// the correction at a mismatch or the bonus after a full match. Each row conditions on the
/// drafts before it, which equal the emitted tokens up to the first mismatch, and the rows past
/// it are discarded, so the emitted tokens are samples of the target.
enum BatchedMTPVerification {

    /// The emitted tokens of a walk, the drafts it accepted, and its final token.
    typealias Result = (tokens: [Int], accepted: Int, finalToken: MLXArray)

    /// Verifies `draftTokens` against the target tokens `sampler` draws from each row of `logits`.
    ///
    /// - Parameters:
    ///   - logits: The processed target logits of the verify rows, shape `[numDraft + 1, vocabulary]`.
    ///   - draftTokens: The drafts, shape `[numDraft]`.
    ///   - sampler: The sampler of the stream.
    /// - Returns: The accepted drafts and the sample after them, the number of accepted drafts,
    ///   and the final token, shape `[1]`.
    static func sampled(
        logits: MLXArray, draftTokens: MLXArray, sampler: any LogitSampler
    ) -> Result {
        matching(targetTokens: sampler.sample(logits: logits), draftTokens: draftTokens)
    }

    /// Accepts the longest prefix of `draftTokens` that `targetTokens` matches.
    ///
    /// - Parameters:
    ///   - targetTokens: The target token of each verify row, shape `[numDraft + 1]`.
    ///   - draftTokens: The drafts, shape `[numDraft]`.
    /// - Returns: The accepted drafts and the target token after them, the number of accepted
    ///   drafts, and the final token, shape `[1]`.
    static func matching(targetTokens: MLXArray, draftTokens: MLXArray) -> Result {
        // Materialize drafts and target tokens together with one GPU readback.
        let tokens = concatenated([draftTokens.asType(targetTokens.dtype), targetTokens])
            .asArray(Int.self)
        let numDraft = draftTokens.size
        var accepted = 0
        while accepted < numDraft, tokens[accepted] == tokens[numDraft + accepted] {
            accepted += 1
        }
        return (
            Array(tokens[numDraft ..< (numDraft + accepted + 1)]), accepted,
            targetTokens[accepted].reshaped([1])
        )
    }
}
