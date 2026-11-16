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
    /// The step does not move a checkpoint of the hybrid caches: a prompt cache
    /// takes each checkpoint after a feed that ends at a message boundary (see
    /// `PromptCacheCheckpoint`), never at a prefill chunk boundary.
    package static let defaultStepSize = 2048
}
