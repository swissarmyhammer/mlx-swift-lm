// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// A tiny Qwen 3.5 target whose two layers are both full attention, thus every cache trims, on
/// seeded random weights over the vocabulary of ``Qwen35MTPFixture``.
public enum Qwen35AttentionOnlyTarget {

    /// The seed of the weights.
    private static let seed: UInt64 = 21

    /// The decoder layers, every one full attention.
    private static let layerCount = 2

    /// Makes the target.
    ///
    /// - Returns: The target.
    /// - Throws: When the configuration does not decode.
    public static func make() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, hiddenLayers: layerCount, fullAttentionInterval: 1,
                    hiddenSize: Qwen35MTPFixture.hiddenSize,
                    vocabularySize: Qwen35MTPFixture.vocabularySize, tieWordEmbeddings: false
                ).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35TextModel(configuration)
        }
        eval(model)
        return model
    }
}
