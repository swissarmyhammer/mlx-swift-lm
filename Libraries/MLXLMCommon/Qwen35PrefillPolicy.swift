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
}
