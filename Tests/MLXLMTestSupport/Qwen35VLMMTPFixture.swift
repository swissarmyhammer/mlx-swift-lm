// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// A tiny Qwen 3.5 vision-language target with its MTP drafter, on seeded random weights.
///
/// The language model has the hybrid layout of ``Qwen35MTPFixture``: one Gated DeltaNet
/// (linear, recurrent) layer, then one full-attention layer. The VLM target routes a long or
/// warm prompt through its windowed continuation forward, which the text target does not have,
/// so a test of the MTP prefill reaches that path with this fixture. Both models take their
/// weights from a seed, thus every run of a test decodes the same tokens.
public struct Qwen35VLMMTPFixture {

    /// The target model.
    public let target: Qwen35

    /// The MTP drafter of the target.
    public let drafter: Qwen35VLMNextNDraftModel

    /// Makes the target and the drafter from `seed`.
    ///
    /// - Parameter seed: The seed of the weights.
    /// - Throws: The error of the configuration decoder.
    public init(seed: UInt64 = Qwen35MTPFixture.defaultSeed) throws {
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.visionLanguage(
                    textConfigJSON: Qwen35FixtureJSON.text(
                        mtpLayers: Qwen35MTPFixture.mtpLayers,
                        hiddenLayers: Qwen35MTPFixture.hiddenLayers,
                        fullAttentionInterval: Qwen35MTPFixture.fullAttentionInterval,
                        hiddenSize: Qwen35MTPFixture.hiddenSize,
                        vocabularySize: Qwen35MTPFixture.vocabularySize,
                        tieWordEmbeddings: false)
                ).utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35(configuration)
        }
        let drafter = withRandomState(
            MLXRandom.RandomState(seed: seed &+ Qwen35MTPFixture.drafterSeedOffset)
        ) {
            Qwen35VLMNextNDraftModel(configuration)
        }
        eval(target, drafter)
        self.target = target
        self.drafter = drafter
    }
}
