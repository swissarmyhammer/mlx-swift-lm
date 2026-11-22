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

    /// The scale of the random normal weights.
    private static let weightScale: Float = 0.3

    /// The group size of the NVFP4 experts.
    private static let expertGroupSize = 16

    /// The bits of the NVFP4 experts.
    private static let expertBits = 4

    /// The values one `uint32` of an NVFP4 weight packs.
    private static let valuesPerPackedWord = 8

    /// The tensors a loaded checkpoint keeps in float32 when the rest is bfloat16.
    private static let float32Suffixes = [
        ".query", "mlp.gate.weight", "e_score_correction_bias",
        "forget_gate_proj.weight", "message_gate_proj.weight", ".global_scale",
    ]

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
        let model = Inkling2(config)
        MLXRandom.seed(targetSeed)
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() where value.dtype == .float32 {
            parameters[key] = MLXRandom.normal(value.shape) * weightScale
        }
        for (key, value) in model.parameters().flattened() where key.hasSuffix(".global_scale") {
            parameters[key] = MLXArray.ones(value.shape)
        }
        for (key, value) in model.parameters().flattened()
        where key.hasSuffix("_proj.weight") && value.dtype == .uint32 {
            let prefix = String(key.dropLast(".weight".count))
            let (rows, packed) = (value.dim(1), value.dim(2))
            let dense =
                MLXRandom.normal([value.dim(0), rows, packed * valuesPerPackedWord]) * weightScale
            let (weight, scales, _) = MLX.quantized(
                dense, groupSize: expertGroupSize, bits: expertBits, mode: .nvfp4)
            parameters[key] = weight
            parameters["\(prefix).scales"] = scales
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
        return model
    }

    /// A DSpark drafter of the `mtp_config` of `json` with random weights.
    ///
    /// - Parameter json: the `config.json` of a checkpoint that ships the drafter.
    /// - Returns: the drafter.
    /// - Throws: when `json` does not decode, or the weights do not fit the drafter.
    public static func drafter(json: String) throws -> Inkling2DraftModel {
        let drafter = Inkling2DraftModel(
            try JSONDecoder().decode(Inkling2DraftConfiguration.self, from: Data(json.utf8)))
        MLXRandom.seed(drafterSeed)
        let parameters = Dictionary(
            uniqueKeysWithValues: drafter.parameters().flattened().map { key, value in
                (key, MLXRandom.normal(value.shape) * weightScale)
            })
        try drafter.update(
            parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        eval(drafter)
        return drafter
    }
}
