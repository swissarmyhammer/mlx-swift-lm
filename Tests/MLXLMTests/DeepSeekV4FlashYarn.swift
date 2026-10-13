// Copyright © 2026 Apple Inc.

import MLX

@testable import MLXLLM

/// The YaRN rotary position of a DeepSeek-V4-Flash layer that holds a
/// compressor, for the DeepSeek-V4 test suites.
///
/// More than one suite turns rows with the inverse-frequency table of such a
/// layer. Each suite reads the table and its numbers here, thus every suite
/// checks the same table.
enum DeepSeekV4FlashYarn {

    /// The number of trailing head dimensions that take a position
    /// (`qk_rope_head_dim`).
    static let ropeDim = 64

    /// The `compress_rope_theta` of the DeepSeek-V4-Flash checkpoint.
    static let compressRopeTheta: Float = 160_000

    /// The YaRN `factor` of the DeepSeek-V4-Flash checkpoint.
    static let factor: Float = 16

    /// The YaRN `original_max_position_embeddings` of the DeepSeek-V4-Flash
    /// checkpoint.
    static let originalMaxPositionEmbeddings = 65_536

    /// The YaRN `beta_fast` of the DeepSeek-V4-Flash checkpoint.
    static let betaFast: Float = 32

    /// The YaRN `beta_slow` of the DeepSeek-V4-Flash checkpoint.
    static let betaSlow: Float = 1

    /// The inverse-frequency table of a DeepSeek-V4-Flash layer with a
    /// compressor: `compress_rope_theta` with the YaRN scaling.
    ///
    /// - Returns: The table, shape `(ropeDim / 2)`, in float32.
    static func inverseFrequency() -> MLXArray {
        DeepSeekV4Math.yarnInvFreq(
            dim: ropeDim, base: compressRopeTheta,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            factor: factor, betaFast: betaFast, betaSlow: betaSlow)
    }
}
