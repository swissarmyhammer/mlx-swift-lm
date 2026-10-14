// Copyright © 2026 Apple Inc.
//
// The compiled functions of the DeepSeek-V4 decode step.
//
// A decode step of DeepSeek-V4 costs CPU time, not GPU time. A profile of one
// step at 128 context (task ^rnnxh8d) shows the GPU busy for about 9 ms of a
// step of about 53 ms. MLX encodes each primitive on the CPU, thus the cost of
// a step follows the number of primitives and not the arithmetic. `compile`
// fuses each chain of elementwise operations into one Metal kernel, which
// removes primitives.
//
// The Python reference of mlx-vlm (Blaizzy/mlx-vlm, `deepseek_v4/language.py`)
// compiles the same functions: `_expert_select`, `_hash_expert_select`,
// `_limited_swiglu`, `_hc_split_sinkhorn_ops` and `_hc_expand_op`.
//
// Four rules hold for every function in this file:
//
//  1. Each array the body reads is an argument. A compiled function keeps an
//     array it captures as a constant of its trace, thus a captured weight
//     would freeze at the value of the first call. The `compile` overloads of
//     `MLXLMCommon` take a `@Sendable` body, which cannot capture an
//     `MLXArray`.
//  2. A body captures a Swift value only when the value is a constant of the
//     checkpoint: a limit, an epsilon, a count. A value that changes on each
//     step, such as the position offset, must be an `MLXArray` argument. A
//     captured `Int` offset adds one trace for each token (mlx-lm PR 1189:
//     decode fell from 29 to 17.7 tok/s after 6K tokens).
//  3. A function compiled with a shape serves only a block of one token. Such
//     a function keeps one trace for each input shape, and MLX never drops a
//     trace, thus each new prefill length would add one more trace for each
//     layer. `Qwen35.swift` makes the same choice.
//  4. A function compiled without a shape (`compile(shapeless: true)`) serves
//     each block, prefill and decode, with one trace. MLX replays such a
//     trace on a new shape through the `output_shapes` of each primitive.
//     Elementwise operations, an expanded axis, a transpose, a matrix
//     multiply and a reduction give an output shape; a slice and a reshape to
//     a fixed shape do not. Thus only a body free of slices and fixed
//     reshapes may be shapeless. The clamped SwiGLU, the expand, the combine
//     and the activations of the mHC split are shapeless. At prefill each one
//     fuses its elementwise chain over a whole block into one kernel, which
//     reads and writes each large float32 tensor one time instead of one time
//     for each operation.
//
// Each compiled function gives the same bits as its op path. The parity tests
// in Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift compare the bits.
//
// A decoder layer also runs each decode step in two larger traces,
// `DeepSeekV4DecoderLayer.decodeStep` in DeepSeekV4.swift. MLX traces the body
// of a compiled function that it calls inside another trace, thus a function
// of this file becomes a part of the larger trace there.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The compiled functions of the DeepSeek-V4 decode step. The shapeless ones
/// also serve each prefill block.
///
/// Each function here wraps one op-path function of the DeepSeek-V4 port in
/// `compile`. A factory takes the constants of the checkpoint and returns the
/// compiled function, thus each layer builds its own function one time, in
/// its `init`.
enum DeepSeekV4CompiledDecode {

    /// The axis of a `(batch, tokens, ...)` block that holds its tokens.
    static let tokenAxis = 1

    /// Tells whether a block is one decode step, thus whether it reads the
    /// functions compiled with a shape.
    ///
    /// - Parameter block: A block of shape `(batch, tokens, ...)`.
    /// - Returns: True when the block holds one token.
    static func isDecodeStep(_ block: MLXArray) -> Bool {
        block.dim(tokenAxis) == 1
    }

    // MARK: - The clamped SwiGLU

    /// Compiles ``DeepSeekV4Math/clampedSwiGLU(gate:up:limit:)`` without a
    /// shape, thus one trace serves each block.
    ///
    /// - Parameter limit: The `swiglu_limit` of the checkpoint.
    /// - Returns: A function of `(gate, up)` that gives the activation.
    static func clampedSwiGLU(limit: Float) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
        compile(shapeless: true) { gate, up in
            DeepSeekV4Math.clampedSwiGLU(gate: gate, up: up, limit: limit)
        }
    }

    // MARK: - The manifold hyper-connection

    /// The compiled ``ManifoldStream/expand(blockOutput:residual:post:comb:)``.
    ///
    /// The expand reads no constant of the checkpoint, thus every layer shares
    /// this one function. It is shapeless, thus one trace serves each block.
    ///
    /// The arguments are `(blockOutput, residual, post, comb)`.
    static let expand: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
        shapeless: true
    ) { blockOutput, residual, post, comb in
        ManifoldStream.expand(blockOutput: blockOutput, residual: residual, post: post, comb: comb)
    }

    /// Compiles ``DeepSeekV4Math/hcSplitActivations(_:eps:)`` without a shape,
    /// thus one trace of each field serves each block.
    ///
    /// The slices of the split stay outside, in
    /// ``DeepSeekV4Math/hcSplitOperands(mixes:scale:base:hcMult:)``, because
    /// a shapeless function can hold no slice. Each field takes its own
    /// compiled function, because each field reads its own three arrays.
    ///
    /// - Parameter hcEps: The epsilon of the split, from `hc_eps`.
    /// - Returns: A function of the split operands that gives `pre`, `post`
    ///   and `comb`.
    static func splitActivations(
        hcEps: Float
    ) -> @Sendable (DeepSeekV4Math.HCSplitOperands) -> (
        pre: MLXArray, post: MLXArray, comb: MLXArray
    ) {
        let preWeights = shapelessField { DeepSeekV4Math.hcPreWeights($0, eps: hcEps) }
        let postGains = shapelessField { DeepSeekV4Math.hcPostGains($0) }
        let combRows = shapelessField { DeepSeekV4Math.hcCombRows($0, eps: hcEps) }
        return { operands in
            (
                pre: preWeights(operands.pre), post: postGains(operands.post),
                comb: combRows(operands.comb)
            )
        }
    }

    /// Compiles the activation of one field of the mHC split without a shape.
    ///
    /// - Parameter activation: The op path of the field.
    /// - Returns: A function of the field that gives its activation.
    private static func shapelessField(
        _ activation: @escaping @Sendable (DeepSeekV4Math.HCSplitField) -> MLXArray
    ) -> @Sendable (DeepSeekV4Math.HCSplitField) -> MLXArray {
        let compiled = compile(shapeless: true) { mixes, scale, base in
            activation(DeepSeekV4Math.HCSplitField(mixes: mixes, scale: scale, base: base))
        }
        return { field in compiled(field.mixes, field.scale, field.base) }
    }

    /// Compiles
    /// ``ManifoldStream/collapseFields(of:projection:normWeight:scale:base:hcMult:normEps:hcEps:)``.
    ///
    /// - Parameters:
    ///   - hcMult: The number of parallel copies of the residual stream.
    ///   - normEps: The epsilon of the RMS reduction, from `rms_norm_eps`.
    ///   - hcEps: The epsilon of the split, from `hc_eps`.
    /// - Returns: A function of `(stream, projection, normWeight, scale, base)`
    ///   that gives `(stream in float32, pre, post, comb)`.
    static func collapseFields(
        hcMult: Int, normEps: Float, hcEps: Float
    )
        -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
    {
        compile { stream, projection, normWeight, scale, base in
            let fields = ManifoldStream.collapseFields(
                of: stream, projection: projection, normWeight: normWeight, scale: scale,
                base: base, hcMult: hcMult, normEps: normEps, hcEps: hcEps)
            return (fields.stream, fields.pre, fields.post, fields.comb)
        }
    }

    // MARK: - The routing gate

    /// Compiles ``DeepSeekV4ExpertRouting/topK(_:weight:bias:)``.
    ///
    /// - Parameter routing: The routing constants of the checkpoint.
    /// - Returns: A function of `(x, weight, bias)` that gives
    ///   `(indices, weights)`.
    static func topKRouting(
        _ routing: DeepSeekV4ExpertRouting
    ) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
        compile { x, weight, bias in
            let routed = routing.topK(x, weight: weight, bias: bias)
            return (routed.indices, routed.weights)
        }
    }

    /// Compiles ``DeepSeekV4ExpertRouting/hashed(_:weight:table:inputIds:)``.
    ///
    /// - Parameter routing: The routing constants of the checkpoint.
    /// - Returns: A function of `(x, weight, table, inputIds)` that gives
    ///   `(indices, weights)`.
    static func hashRouting(
        _ routing: DeepSeekV4ExpertRouting
    ) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
        compile { x, weight, table, inputIds in
            let routed = routing.hashed(x, weight: weight, table: table, inputIds: inputIds)
            return (routed.indices, routed.weights)
        }
    }

    // MARK: - The mixture-of-experts combine

    /// The compiled ``DeepSeekV4MoE/combine(_:routed:weights:shared:)`` of a
    /// layer that holds a shared expert. It is shapeless, thus one trace
    /// serves each block.
    ///
    /// The arguments are `(x, routed, weights, shared)`.
    static let combineWithSharedExpert:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
            x, routed, weights, shared in
            DeepSeekV4MoE.combine(x, routed: routed, weights: weights, shared: shared)
        }

    /// The compiled ``DeepSeekV4MoE/combine(_:routed:weights:shared:)`` of a
    /// layer that holds no shared expert. It is shapeless, thus one trace
    /// serves each block.
    ///
    /// The arguments are `(x, routed, weights)`.
    static let combineRoutedExperts: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { x, routed, weights in
            DeepSeekV4MoE.combine(x, routed: routed, weights: weights, shared: nil)
        }
}
