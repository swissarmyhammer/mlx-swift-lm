// Copyright © 2026 Apple Inc.
//
// The logits of a speculative verify forward and of plain decode over the same
// draft tokens, shared by the verify tests of the text and the vision-language
// Qwen 3.5 models. Each model gives one forward over its own caches as a
// closure, so both suites run one implementation of the verify state setup.

import MLX
import MLXLMCommon

/// The logits of a verify forward and of plain decode over the same tokens.
enum VerifyForwardLogits {

    /// One forward of a model over some tokens of one sequence, on the
    /// caches the closure holds.
    ///
    /// - Parameters:
    ///   - tokens: The token ids, shape `(1, tokens)`.
    ///   - state: The state of the forward before, or `nil` for none.
    /// - Returns: The output of the forward.
    typealias Forward = (_ tokens: MLXArray, _ state: LMOutput.State?) -> LMOutput

    /// The logits of one verify forward of `draft`: the drafter state asked
    /// of the target and the recurrent snapshots kept, as the MTP iterator
    /// asks them.
    ///
    /// - Parameters:
    ///   - draft: The token ids of the verify, shape `(1, tokens)`.
    ///   - prompt: The state the prompt left, or `nil` when it left none.
    ///   - forward: One forward of the model over the caches of the prompt.
    /// - Returns: The float32 logits, shape `(1, tokens, vocabulary)`.
    static func verify(
        _ draft: MLXArray, after prompt: LMOutput.State?, forward: Forward
    ) -> MLXArray {
        var state = prompt ?? LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        return forward(draft, state).logits.asType(.float32)
    }

    /// The logits of plain decode over `draft`, one token at a time, each
    /// forward on the state the forward before left.
    ///
    /// - Parameters:
    ///   - draft: The token ids to decode, shape `(1, tokens)`.
    ///   - prompt: The state the prompt left, or `nil` when it left none.
    ///   - forward: One forward of the model over the caches of the prompt.
    /// - Returns: The float32 logits, shape `(1, tokens, vocabulary)`.
    static func decode(
        _ draft: MLXArray, after prompt: LMOutput.State?, forward: Forward
    ) -> MLXArray {
        var state = prompt
        let rows = (0 ..< draft.dim(1)).map { row in
            let output = forward(draft[0..., row ..< row + 1], state)
            state = output.state ?? state
            return output.logits
        }
        return concatenated(rows, axis: 1).asType(.float32)
    }
}
