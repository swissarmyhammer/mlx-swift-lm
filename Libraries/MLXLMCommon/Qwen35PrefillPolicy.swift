// Copyright © 2026 Apple Inc.

import Foundation

/// The prefill of Qwen 3.5, which the text models of `MLXLLM` and the
/// vision-language model of `MLXVLM` share.
package enum Qwen35PrefillPolicy {

    /// The prefill step when the caller sets none: the prefill chunk of mlx-vlm
    /// and mlx-lm. A forward of more tokens keeps the GPU busier than the
    /// generic step of ``PrefillParameters/defaultStepSize``, for more peak
    /// memory. A step the caller sets still wins.
    ///
    /// The Qwen 3.5 models give this step as their
    /// ``LanguageModel/defaultPrefillStepSize``, thus the feeds of a prompt
    /// cache up to a checkpoint (see `PromptCacheCheckpoint`) take it too. The
    /// step does not move a checkpoint of the hybrid caches: each feed ends at
    /// a message boundary, never at a prefill chunk boundary.
    package static let defaultStepSize = 2048

    /// The largest prefill step that keeps a forward off the head-dimension
    /// split (`dsplit`) attention kernel of MLX.
    ///
    /// On a GPU with neural accelerators (NAX), MLX selects the `dsplit`
    /// kernel for a causal attention with a head dimension of 256 and 1024 or
    /// more query tokens. The mlx that mlx-swift 0.33.0 carries cannot build
    /// that kernel (ml-explore/mlx#4832): the process stops with "Unable to
    /// build metal library from source".
    package static let largestStepBelowHeadDimensionSplit = 1023

    /// The largest prefill step of a Qwen 3.5 model, or nil when the model
    /// has no limit.
    ///
    /// ``PrefillAttentionKernel`` does the prefill attention when the query
    /// heads of each KV head make whole head groups. Else the prefill goes to
    /// `MLXFast.scaledDotProductAttention`, and a head dimension of 256 then
    /// needs a step of at most ``largestStepBelowHeadDimensionSplit``. Thus
    /// Qwen3.5-9B (16 query heads, 4 KV heads) gets the limit, and
    /// Qwen3.8-27B (24 query heads, 4 KV heads) keeps
    /// ``defaultStepSize``.
    ///
    /// - Parameters:
    ///   - queryHeads: The query heads of a full-attention layer.
    ///   - kvHeads: The KV heads of a full-attention layer.
    ///   - headDimension: The width of one attention head.
    /// - Returns: ``largestStepBelowHeadDimensionSplit``, or nil.
    package static func maximumStepSize(queryHeads: Int, kvHeads: Int, headDimension: Int)
        -> Int?
    {
        guard headDimension == 256,
            !PrefillAttentionKernel.headGroupsFit(queryHeads: queryHeads, kvHeads: kvHeads)
        else { return nil }
        return largestStepBelowHeadDimensionSplit
    }
}
