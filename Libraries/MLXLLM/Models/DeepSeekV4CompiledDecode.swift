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
// Three rules hold for every function in this file:
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
//  3. Only a block of one token reads a compiled function. A compiled function
//     keeps one trace for each input shape, and each new prefill length would
//     add one more trace for each layer. Prefill is bound by its matrix
//     multiplies, thus it keeps the op path. `Qwen35.swift` makes the same
//     choice.
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

/// The compiled functions of the DeepSeek-V4 decode step.
///
/// Each function here wraps one op-path function of the DeepSeek-V4 port in
/// `compile`. A factory takes the constants of the checkpoint and returns the
/// compiled function, thus each layer builds its own function one time, in
/// its `init`.
enum DeepSeekV4CompiledDecode {

    /// The axis of a `(batch, tokens, ...)` block that holds its tokens.
    static let tokenAxis = 1

    /// Tells whether a block is one decode step, thus whether it reads the
    /// compiled functions.
    ///
    /// - Parameter block: A block of shape `(batch, tokens, ...)`.
    /// - Returns: True when the block holds one token.
    static func isDecodeStep(_ block: MLXArray) -> Bool {
        block.dim(tokenAxis) == 1
    }

    // MARK: - The clamped SwiGLU

    /// Compiles ``DeepSeekV4Math/clampedSwiGLU(gate:up:limit:)``.
    ///
    /// - Parameter limit: The `swiglu_limit` of the checkpoint.
    /// - Returns: A function of `(gate, up)` that gives the activation.
    static func clampedSwiGLU(limit: Float) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
        compile { gate, up in
            DeepSeekV4Math.clampedSwiGLU(gate: gate, up: up, limit: limit)
        }
    }

    // MARK: - The manifold hyper-connection

    /// The compiled ``ManifoldStream/expand(blockOutput:residual:post:comb:)``.
    ///
    /// The expand reads no constant of the checkpoint, thus every layer shares
    /// this one function and its one trace.
    ///
    /// The arguments are `(blockOutput, residual, post, comb)`.
    static let expand: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = compile {
        blockOutput, residual, post, comb in
        ManifoldStream.expand(blockOutput: blockOutput, residual: residual, post: post, comb: comb)
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
    /// layer that holds a shared expert.
    ///
    /// The arguments are `(x, routed, weights, shared)`.
    static let combineWithSharedExpert:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = compile {
            x, routed, weights, shared in
            DeepSeekV4MoE.combine(x, routed: routed, weights: weights, shared: shared)
        }

    /// The compiled ``DeepSeekV4MoE/combine(_:routed:weights:shared:)`` of a
    /// layer that holds no shared expert.
    ///
    /// The arguments are `(x, routed, weights)`.
    static let combineRoutedExperts: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile { x, routed, weights in
            DeepSeekV4MoE.combine(x, routed: routed, weights: weights, shared: nil)
        }
}
