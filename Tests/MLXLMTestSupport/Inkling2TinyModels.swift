// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

/// Tiny Inkling 2 targets and DSpark drafters on seeded random weights, which the Inkling 2
/// tests of every bundle build from a `config.json`.
public enum Inkling2TinyModels {

    /// The seed of the target's random weights.
    private static let targetSeed: UInt64 = 5

    /// The seed of the drafter's random weights.
    private static let drafterSeed: UInt64 = 11

    /// The seed of the initial weights the layers of a model draw before the random weights
    /// replace them. A task-local state of its own keeps those draws off the global random
    /// stream, whose lazy key every later draw of any test would otherwise extend.
    private static let initializerSeed: UInt64 = 13

    /// The scale of the random normal weights.
    private static let weightScale: Float = 0.3

    /// The group size of the NVFP4 experts.
    private static let expertGroupSize = 16

    /// The bits of the NVFP4 experts.
    private static let expertBits = 4

    /// The values one `uint32` of an NVFP4 weight packs.
    private static let valuesPerPackedWord = 8

    /// The axis of the experts of a packed NVFP4 expert weight.
    private static let expertAxis = 0

    /// The axis of the output rows of a packed NVFP4 expert weight.
    private static let rowAxis = 1

    /// The axis of the packed `uint32` words of a packed NVFP4 expert weight.
    private static let packedAxis = 2

    /// The tensors a loaded checkpoint keeps in float32 when the rest is bfloat16.
    private static let float32Suffixes = [
        ".query", "mlp.gate.weight", "e_score_correction_bias",
        "forget_gate_proj.weight", "message_gate_proj.weight", ".global_scale",
    ]

    /// The `config.json` of the tiny Inkling 2 target of `Inkling2ModelTests`: six layers in
    /// blocks of two, where layers 2 and 3 loop twice. Layers 3 and 5 attend to all positions,
    /// the others to a window of 8. Its heads and its expert width split across two processes.
    public static let targetJSON = """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 50,
            "unpadded_vocab_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "sliding_window_size": 8, "local_layer_ids": [0, 1, 2, 4],
            "d_rel": 8, "rel_extent": 16, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 2.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
              "is_varloop": false
            }
          }
        }
        """

    /// The `config.json` of a tiny Inkling 2 target with a DSpark head, as
    /// `mlx-community/Inkling-2-Mini-oQ4e-mtp` carries its MTP configuration: the target of
    /// `Inkling2ModelTests`, whose eight layer visits are the prelude, two passes of the loop
    /// core and the coda, and a drafter of two draft layers, a window of six, so it attends the
    /// newest five context rows, blocks of four, and the feature taps `[1, 3, 5, 8]`.
    ///
    /// - Parameters:
    ///   - vocabularySize: the rows of the embedding and of the vocabulary projection.
    ///   - unpaddedVocabularySize: the tokens of the vocabulary that are not padding.
    /// - Returns: the `config.json`.
    public static func draftJSON(vocabularySize: Int, unpaddedVocabularySize: Int) -> String {
        """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": \(vocabularySize),
            "unpadded_vocab_size": \(unpaddedVocabularySize), "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 32, "sliding_window_size": 8,
            "local_layer_ids": [0, 1, 2, 4], "d_rel": 8, "rel_extent": 16,
            "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
            "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4,
            "n_layers_per_attn_res_block": 2, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "intermediate_size": 32, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "route_scale": 2.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
              "is_varloop": false
            }
          },
          "mtp_config": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
            "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "local_layer_ids": [0],
            "sliding_window_size": 6, "block_size": 4, "markov_rank": 8,
            "markov_head_type": "rnn_hidden", "q_bias": false, "o_bias": false,
            "target_layer_ids": [1, 3, 5, 8]
          }
        }
        """
    }

    /// The result of `body`, whose random draws come from a task-local state of `seed`.
    ///
    /// Task-local rather than `MLXRandom.seed`: parallel tests must not share (or perturb) the
    /// global random stream, or the weights differ from run to run.
    ///
    /// - Parameters:
    ///   - seed: the seed of the random state of `body`.
    ///   - body: the work that draws random values.
    /// - Returns: the result of `body`.
    /// - Throws: what `body` throws.
    private static func seeded<Result>(_ seed: UInt64, _ body: () throws -> Result) rethrows
        -> Result
    {
        try withRandomState(MLXRandom.RandomState(seed: seed), body: body)
    }

    /// A target of `json` with random weights and NVFP4 experts. In bfloat16, the tensors that
    /// the checkpoint stores in float32 stay float32, and the model is prepared as after a load.
    ///
    /// - Parameters:
    ///   - json: the `config.json` of the model.
    ///   - bfloat16: whether the model runs in bfloat16 as a loaded checkpoint does.
    /// - Returns: the target.
    /// - Throws: when `json` does not decode, or the weights do not fit the model.
    public static func target(json: String, bfloat16: Bool = false) throws -> Inkling2 {
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: Data(json.utf8))
        let model = seeded(initializerSeed) { Inkling2(config) }
        try seeded(targetSeed) {
            var parameters: [String: MLXArray] = [:]
            for (key, value) in model.parameters().flattened() where value.dtype == .float32 {
                parameters[key] = MLXRandom.normal(value.shape) * weightScale
            }
            for (key, value) in model.parameters().flattened()
            where key.hasSuffix("_proj.weight") && value.dtype == .uint32 {
                let prefix = String(key.dropLast(".weight".count))
                let (rows, packed) = (value.dim(rowAxis), value.dim(packedAxis))
                let dense =
                    MLXRandom.normal([value.dim(expertAxis), rows, packed * valuesPerPackedWord])
                    * weightScale
                let experts = (0 ..< dense.dim(expertAxis)).map { quantizedExpert(dense[$0]) }
                parameters[key] = stacked(experts.map(\.weight))
                parameters["\(prefix).scales"] = stacked(experts.map(\.scales))
                parameters["\(prefix).global_scale"] = stacked(experts.map(\.globalScale))
            }
            if bfloat16 {
                for (key, value) in parameters
                where value.dtype == .float32 && !float32Suffixes.contains(where: key.hasSuffix) {
                    parameters[key] = value.asType(.bfloat16)
                }
            }
            try model.update(
                parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
            if bfloat16 {
                try model.prepare()
            }
            eval(model)
        }
        return model
    }

    /// One expert's `[rows, columns]` weights in NVFP4 with the global scale of a checkpoint:
    /// the largest magnitude of the weights, which the dequantization multiplies the group
    /// scales by. Without it, `MLX.quantized` writes group scales that a global scale of 1
    /// shrinks by `6 * 448`, the largest FP4 value times the largest FP8 scale, so the experts
    /// would output almost zero.
    ///
    /// - Parameter dense: the weights of one expert.
    /// - Returns: the packed weights, the group scales and the global scale.
    private static func quantizedExpert(_ dense: MLXArray) -> (
        weight: MLXArray, scales: MLXArray, globalScale: MLXArray
    ) {
        let globalScale = abs(dense).max()
        let (weight, scales, _) = MLX.quantized(
            dense, groupSize: expertGroupSize, bits: expertBits, mode: .nvfp4,
            globalScale: globalScale)
        return (weight, scales, globalScale)
    }

    /// A DSpark drafter of the `mtp_config` of `json` with random weights.
    ///
    /// - Parameter json: the `config.json` of a checkpoint that ships the drafter.
    /// - Returns: the drafter.
    /// - Throws: when `json` does not decode, or the weights do not fit the drafter.
    public static func drafter(json: String) throws -> Inkling2DraftModel {
        let configuration = try JSONDecoder().decode(
            Inkling2DraftConfiguration.self, from: Data(json.utf8))
        let drafter = seeded(initializerSeed) { Inkling2DraftModel(configuration) }
        let parameters = seeded(drafterSeed) {
            Dictionary(
                uniqueKeysWithValues: drafter.parameters().flattened().map { key, value in
                    (key, MLXRandom.normal(value.shape) * weightScale)
                })
        }
        try drafter.update(
            parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        eval(drafter)
        return drafter
    }
}
