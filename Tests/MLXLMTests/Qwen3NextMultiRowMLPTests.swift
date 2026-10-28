// Copyright © 2026 Apple Inc.
//
// `Qwen3NextMLP`, which Qwen3-Next shares with the Qwen3.5 models and their
// MTP drafter, runs its projections through ``MultiRowQuantizedMatvec``.
// Plain decode keeps its bits, and inside a speculative verify the rows get
// the bits of a plain decode.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen3NextMultiRowMLPTests: XCTestCase {

    /// A tiny Qwen3-Next with a dense MLP in every layer, whose MLP widths fit
    /// the kernel after a 4-bit quantization in groups of 64.
    private static let configurationJSON = """
        {
            "hidden_size": 512, "num_hidden_layers": 4, "intermediate_size": 1024,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
            "linear_num_value_heads": 2, "linear_num_key_heads": 2,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 512, "mlp_only_layers": [0, 1, 2, 3],
            "moe_intermediate_size": 512, "rms_norm_eps": 1e-6, "vocab_size": 32,
            "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "max_position_embeddings": 128, "norm_topk_prob": true,
            "tie_word_embeddings": true, "attention_bias": false,
            "full_attention_interval": 2
        }
        """

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 43

    /// The seed of the rows of the MLP test.
    private static let rowSeed: UInt64 = 44

    /// The width of the hidden state of ``configurationJSON``.
    private static let hiddenSize = 512

    /// The decoder layers of ``configurationJSON``, each with a dense MLP.
    private static let hiddenLayers = 4

    private static let decodeTokens: [Int32] = [3, 8, 1, 5, 2]

    /// The rows of a verify: the fewest the kernel runs.
    private static let verifyRows = MultiRowQuantizedMatvec.fewestRows

    /// A model on seeded weights in bfloat16, quantized.
    ///
    /// - Parameter matvecOfMLX: True to give every projection the matvec of
    ///   MLX.
    private func makeModel(matvecOfMLX: Bool = false) throws -> Qwen3NextModel {
        let configuration = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: Data(Self.configurationJSON.utf8))
        return try MultiRowProjectionFixture.seededQuantized(
            seed: Self.weightSeed, matvecOfMLX: matvecOfMLX
        ) {
            Qwen3NextModel(configuration)
        }
    }

    private func mlps(of model: Qwen3NextModel) -> [Qwen3NextMLP] {
        model.model.layers.compactMap { $0.mlp as? Qwen3NextMLP }
    }

    private func decodeLogits(_ model: Qwen3NextModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        return Self.decodeTokens.map { token in
            let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
            return logits
        }
    }

    func testPlainDecodeLogitsKeepTheirBits() throws {
        let model = try makeModel()
        let reference = try makeModel(matvecOfMLX: true)
        XCTAssertEqual(mlps(of: model).count, Self.hiddenLayers)

        let got = try decodeLogits(model)
        let want = try decodeLogits(reference)

        for (step, (gotLogits, wantLogits)) in zip(got, want).enumerated() {
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(gotLogits, wantLogits), 0,
                "decode step \(step): logits differ")
        }
    }

    func testVerifyRowsOfTheMLPMatchSingleRowDecode() throws {
        let model = try makeModel()
        let mlp = try XCTUnwrap(mlps(of: model).first)
        let x = MLXRandom.normal(
            [1, Self.verifyRows, Self.hiddenSize], key: MLXRandom.key(Self.rowSeed)
        ).asType(.bfloat16)

        let (got, want) = MultiRowQuantizedMatvec.run(asSpeculativeVerify: true) {
            (
                mlp(x),
                concatenated(
                    (0 ..< Self.verifyRows).map { mlp(x[0..., $0 ..< ($0 + 1), 0...]) }, axis: 1)
            )
        }
        eval(got, want)

        XCTAssertEqual(BitwiseComparison.mismatchCount(got, want), 0)
    }
}
