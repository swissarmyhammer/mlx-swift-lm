// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// The greedy MTP verification one row at a time: the reference the batched verification of
/// `MTPSpeculativeTokenIterator` must match, and the baseline its benchmark times it against.
public enum SequentialGreedyMTPVerification {

    /// Samples each verify row with `ArgMaxSampler` and stops at the first row whose token
    /// differs from its draft, or after the bonus row when every draft matches.
    ///
    /// - Parameters:
    ///   - logits: The verify logits, one row for each draft and one bonus row.
    ///   - draftTokens: The drafts, one fewer than the rows of `logits`.
    /// - Returns: The emitted tokens, the count of accepted drafts, and the last emitted token
    ///   as the sampler returned it.
    public static func verify(
        logits: MLXArray, draftTokens: MLXArray
    ) -> (tokens: [Int], accepted: Int, finalToken: MLXArray) {
        eval(draftTokens)
        let drafts = draftTokens.asArray(Int.self)
        let sampler = ArgMaxSampler()
        var tokens: [Int] = []
        var accepted = 0
        for row in 0 ... drafts.count {
            let token = sampler.sample(logits: logits[row ..< (row + 1)])
            eval(token)
            let value = token.item(Int.self)
            tokens.append(value)
            if row == drafts.count || value != drafts[row] {
                return (tokens, accepted, token)
            }
            accepted += 1
        }
        preconditionFailure("verification must produce a correction or bonus")
    }
}
