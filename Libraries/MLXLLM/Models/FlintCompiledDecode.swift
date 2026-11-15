// Copyright © 2026 Apple Inc.
//
// The compiled functions of the Flint decode step.
//
// A decode step of a quantized Flint costs CPU time: MLX builds and encodes
// each primitive on the CPU, thus the cost of a step follows the number of
// primitives and not the arithmetic (task ^pg2hsb8). `compile` fuses each
// chain of elementwise operations into one Metal kernel, which removes
// primitives. Each of the 47 MoE layers of the published checkpoint runs the
// two functions of this file once for each token:
//
//  - ``FlintCompiledDecode/routerScores`` fuses the float32 cast of the gate
//    logits, the sigmoid and the bias add ahead of the fused router kernel.
//  - ``FlintCompiledDecode/combine(_:)`` fuses the epsilon add and the divide
//    of the weight normalization into one kernel, and the cast back to the
//    activation dtype, the add of the shared expert and the add of the
//    residual stream after the weighted sum into another.
//
// No fusion joins the router kernel, the matmuls or the reductions.
//
// `FlintSparseMoeBlock.compiledDecodeStep` runs the whole decode step of a
// block, with these two functions, in one larger trace. MLX traces the body
// of a compiled function that it calls inside another trace, thus the
// functions of this file become a part of that trace, and the CPU builds one
// primitive for the block instead of one for each step. Inside it, MLX fuses
// the router scores as two kernels, the cast with the sigmoid and then the
// bias add, because the router kernel reads the scores as well.
//
// Three rules hold for every function in this file, as they hold for
// `DeepSeekV4CompiledDecode.swift`:
//
//  1. Each array the body reads is an argument. A compiled function keeps an
//     array it captures as a constant of its trace, thus a captured weight
//     would freeze at the value of the first call.
//  2. A body captures a Swift value only when the value is a constant of the
//     checkpoint, such as the routing of ``FlintExpertRouting``.
//  3. Each function is shapeless: its body holds no slice and no reshape to a
//     fixed shape, thus one trace serves each shape. Only a block of one
//     token reads them; prefill keeps the op path.
//
// Each compiled function gives the same bits as its op path. The parity tests
// in Tests/MLXLMTests/FlintCompiledDecodeTests.swift compare the bits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The compiled functions of the Flint decode step.
///
/// Each function here wraps the op path of ``FlintExpertRouting`` in
/// `compile`. A factory takes the routing of the checkpoint and returns the
/// compiled function, thus each MoE block builds its own function one time,
/// in its `init`.
enum FlintCompiledDecode {

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

    /// The compiled ``FlintExpertRouting/scores(logits:bias:)``.
    ///
    /// It reads no constant of the checkpoint, thus every block shares this
    /// one function.
    ///
    /// The arguments are `(logits, bias)`, and the results are
    /// `(scores, choice)`.
    static let routerScores: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) = compile(
        shapeless: true
    ) { logits, bias in
        let router = FlintExpertRouting.scores(logits: logits, bias: bias)
        return (router.scores, router.choice)
    }

    /// Compiles the residual add of
    /// ``FlintExpertRouting/combined(_:routed:weights:shared:)`` for a block
    /// with a shared expert.
    ///
    /// - Parameter routing: The routing constants of the checkpoint.
    /// - Returns: A function of `(x, routed, weights, shared, residual)` that
    ///   gives `residual` plus the block output.
    static func combine(
        _ routing: FlintExpertRouting
    ) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
        compile(shapeless: true) { x, routed, weights, shared, residual in
            residual + routing.combined(x, routed: routed, weights: weights, shared: shared)
        }
    }

    /// Compiles the residual add of
    /// ``FlintExpertRouting/combined(_:routed:weights:shared:)`` for a block
    /// without a shared expert.
    ///
    /// - Parameter routing: The routing constants of the checkpoint.
    /// - Returns: A function of `(x, routed, weights, residual)` that gives
    ///   `residual` plus the block output.
    static func combineRoutedExperts(
        _ routing: FlintExpertRouting
    ) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
        compile(shapeless: true) { x, routed, weights, residual in
            residual + routing.combined(x, routed: routed, weights: weights, shared: nil)
        }
    }
}
