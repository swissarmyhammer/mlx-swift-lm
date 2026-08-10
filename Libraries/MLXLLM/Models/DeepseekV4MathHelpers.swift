// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Two details of ``DeepseekV4Math/yarnInvFreq(dim:base:originalMaxPositionEmbeddings:factor:betaFast:betaSlow:)``
// do not come from that file, because a later transcription of it in
// scouzi1966/mlx-swift-lm,
// Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift @ e1852869ce61ded0d23b76df3757e9b75c77c1f5,
// corrects the first one, and the DeepSeek-V4 Python reference
// (Thump604/mlx-lm @ deepseek-v4-support-fixes, mlx_lm/models/deepseek_v4.py,
// `DeepseekV4RoPE.__init__`) decides both:
//
//  1. `betaFast` gives the low end of the correction range and `betaSlow`
//     gives the high end. The file above has the two the other way around.
//     `YarnRoPE` in this repository, Libraries/MLXLMCommon/RoPEUtils.swift,
//     also reads them this way.
//  2. A degenerate range widens as `high += 0.001` when `high` equals `low`,
//     and the ramp then divides by `high - low`. The file above divides by
//     `max(high - low, 0.001)` instead, which hides the effect of the
//     `high = min(high, dim - 1)` clamp whenever that clamp pulls `high`
//     below `low`.

import Foundation
import MLX
import MLXNN

/// The pure math of the DeepSeek-V4 forward pass.
///
/// Every function here reads its arguments and returns a value. None of them
/// holds state, reads a cache, or owns a weight, thus a test gives each one a
/// synthetic tensor and needs no checkpoint.
enum DeepseekV4Math {

    // MARK: - Manifold hyper-connections

    /// The number of one-value-for-each-copy fields the mixes vector holds
    /// ahead of the flattened mixing matrix: one `pre` weight and one `post`
    /// weight for each parallel copy of the residual stream.
    private static let splitFieldCount = 2

    /// The place of the `pre` scale inside the learned scale vector.
    private static let preScaleIndex = 0

    /// The place of the `post` scale inside the learned scale vector.
    private static let postScaleIndex = 1

    /// The place of the `comb` scale inside the learned scale vector.
    private static let combScaleIndex = 2

    /// The `post` field is a sigmoid opened to the range 0 through 2, thus one
    /// block output can double before the residual stream takes it back.
    private static let postGain: Float = 2

    /// A sum over this axis gives one number for each row, thus a division by
    /// that sum makes each row of the mixing matrix add up to 1.
    private static let rowSumAxis = -1

    /// A sum over this axis gives one number for each column, thus a division
    /// by that sum makes each column of the mixing matrix add up to 1.
    private static let columnSumAxis = -2

    /// Splits the mHC mixes vector into the three fields one layer needs.
    ///
    /// The last axis of `mixes` holds `(2 + hcMult) * hcMult` numbers: the
    /// first `hcMult` make `pre`, the next `hcMult` make `post`, and the
    /// remaining `hcMult * hcMult` reshape into the mixing matrix `comb`.
    ///
    /// `comb` starts as a softmax over each row and then takes `iters`
    /// Sinkhorn steps: one column normalization, then `iters - 1` pairs of a
    /// row normalization and a column normalization. The matrix that comes out
    /// is doubly stochastic, thus the expand step it drives holds the norm of
    /// the residual stream.
    ///
    /// Every value moves to float32 first. The Sinkhorn steps divide by sums
    /// of small numbers, and float16 loses those sums.
    ///
    /// - Parameters:
    ///   - mixes: The projected mixes, shape `(..., (2 + hcMult) * hcMult)`.
    ///   - scale: The three learned scales, one for each field, shape `(3)`.
    ///   - base: The learned bias, shape `((2 + hcMult) * hcMult)`.
    ///   - hcMult: The number of parallel copies of the residual stream.
    ///   - iters: The number of Sinkhorn steps, from `hc_sinkhorn_iters`.
    ///   - eps: The epsilon the Sinkhorn steps add, from `hc_eps`.
    /// - Returns: `pre` and `post`, each shape `(..., hcMult)`, and `comb`,
    ///   shape `(..., hcMult, hcMult)`.
    static func hcSplitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int,
        eps: Float
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        let mixWidth = (splitFieldCount + hcMult) * hcMult
        precondition(
            mixes.shape.last == mixWidth,
            "the last axis of mixes must be (2 + hcMult) * hcMult = \(mixWidth), "
                + "and it is \(mixes.shape.last ?? -1)")

        let mixesF32 = mixes.asType(.float32)
        let scaleF32 = scale.asType(.float32)
        let baseF32 = base.asType(.float32)

        let splitWidth = splitFieldCount * hcMult
        let pre =
            sigmoid(
                mixesF32[.ellipsis, 0 ..< hcMult] * scaleF32[preScaleIndex]
                    + baseF32[0 ..< hcMult]) + eps
        let post =
            postGain
            * sigmoid(
                mixesF32[.ellipsis, hcMult ..< splitWidth] * scaleF32[postScaleIndex]
                    + baseF32[hcMult ..< splitWidth])

        let combFlat = mixesF32[.ellipsis, splitWidth...]
        let combShape = Array(combFlat.shape.dropLast()) + [hcMult, hcMult]
        let combLogits =
            (combFlat * scaleF32[combScaleIndex]).reshaped(combShape)
            + baseF32[splitWidth...].reshaped([hcMult, hcMult])

        var comb = softmax(combLogits, axis: rowSumAxis, precise: true) + eps
        comb = normalized(comb, summingOver: columnSumAxis, eps: eps)
        for _ in 0 ..< max(iters - 1, 0) {
            comb = normalized(comb, summingOver: rowSumAxis, eps: eps)
            comb = normalized(comb, summingOver: columnSumAxis, eps: eps)
        }

        return (pre: pre, post: post, comb: comb)
    }

    /// Divides `x` by its own sum over `axis`, keeping the shape of `x`.
    private static func normalized(
        _ x: MLXArray, summingOver axis: Int, eps: Float
    ) -> MLXArray {
        x / (x.sum(axis: axis, keepDims: true) + eps)
    }

    // MARK: - Partial rotary position

    /// Two dimensions make one rotary pair, thus a head of width `d` holds
    /// `d / 2` pairs.
    private static let rotaryPairWidth = 2

    /// Turns the trailing rotary dimensions of `x` by the given angle.
    ///
    /// DeepSeek-V4 gives a position to the last `ropeDim` dimensions of a head
    /// and leaves the dimensions before them with no position at all. On the
    /// DeepSeek-V4-Flash head of width 512 with `qk_rope_head_dim` 64, the
    /// first 448 numbers of each head come back unchanged.
    ///
    /// The rotation is the traditional one: it turns adjacent pairs
    /// `(x[0], x[1])`, `(x[2], x[3])` and so on, not the two halves of the
    /// head. The other convention mixes position across the head and the model
    /// then repeats one token.
    ///
    /// - Parameters:
    ///   - x: The head values, shape `(..., headDim)`.
    ///   - cos: The cosine of each angle, shape `(..., ropeDim / 2)`.
    ///   - sin: The sine of each angle, shape `(..., ropeDim / 2)`.
    ///   - ropeDim: The number of trailing dimensions that take a position.
    /// - Returns: An array of the shape of `x`.
    static func applyPartialRoPE(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, ropeDim: Int
    ) -> MLXArray {
        partialRoPE(x, cos: cos, sin: sin, ropeDim: ropeDim, inverse: false)
    }

    /// Turns the trailing rotary dimensions of `x` back by the given angle.
    ///
    /// This is the conjugate of
    /// ``applyPartialRoPE(_:cos:sin:ropeDim:)``: it flips the sign of the sine,
    /// thus it undoes that rotation to the precision of the arithmetic.
    ///
    /// DeepSeek-V4 runs it on the output of attention, which strips the
    /// position back out before the residual stream adds the block result.
    ///
    /// - Parameters:
    ///   - x: The head values, shape `(..., headDim)`.
    ///   - cos: The cosine of each angle, shape `(..., ropeDim / 2)`.
    ///   - sin: The sine of each angle, shape `(..., ropeDim / 2)`.
    ///   - ropeDim: The number of trailing dimensions that take a position.
    /// - Returns: An array of the shape of `x`.
    static func applyInversePartialRoPE(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, ropeDim: Int
    ) -> MLXArray {
        partialRoPE(x, cos: cos, sin: sin, ropeDim: ropeDim, inverse: true)
    }

    private static func partialRoPE(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, ropeDim: Int, inverse: Bool
    ) -> MLXArray {
        guard let headDim = x.shape.last else {
            preconditionFailure("x must have at least one axis")
        }
        precondition(
            ropeDim <= headDim,
            "ropeDim \(ropeDim) must not be wider than the head, which is \(headDim)")

        let noPositionDim = headDim - ropeDim
        guard noPositionDim > 0 else {
            return rotatePairs(x, cos: cos, sin: sin, inverse: inverse)
        }
        let noPosition = x[.ellipsis, 0 ..< noPositionDim]
        let rotated = rotatePairs(
            x[.ellipsis, noPositionDim...], cos: cos, sin: sin, inverse: inverse)
        return concatenated([noPosition, rotated], axis: -1)
    }

    /// Turns each adjacent pair of the last axis of `x`.
    ///
    /// The cosine and the sine move to the dtype of `x` first. The DeepSeek-V4
    /// runtime builds the angle table in float32 and casts it down before it
    /// rotates; without that cast float32 spreads from the table into the
    /// whole residual stream and each later expert casts its weights too.
    private static func rotatePairs(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        let cosine = cos.asType(x.dtype)
        let sine = (inverse ? -sin : sin).asType(x.dtype)
        let pairCount = (x.shape.last ?? 0) / rotaryPairWidth
        let paired = x.reshaped(Array(x.shape.dropLast()) + [pairCount, rotaryPairWidth])
        let real = paired[.ellipsis, 0]
        let imaginary = paired[.ellipsis, 1]
        let turnedReal = real * cosine - imaginary * sine
        let turnedImaginary = real * sine + imaginary * cosine
        return stacked([turnedReal, turnedImaginary], axis: -1).reshaped(x.shape)
    }

    // MARK: - YaRN inverse frequency

    /// One full turn of the circle, in radians. Doubling `pi` this way is
    /// exact in binary floating point, thus it is the same number as
    /// `2 * pi`.
    private static let fullTurn: Float = .pi + .pi

    /// The width a degenerate correction range takes, so that the ramp never
    /// divides by zero.
    private static let degenerateRangeWidth: Float = 0.001

    /// Builds the YaRN inverse-frequency table of one rotary layer.
    ///
    /// The table starts as `1 / base ^ (2 * i / dim)`. YaRN then reads a ramp
    /// across the table: a dimension whose wavelength is shorter than
    /// `betaFast` turns keeps its plain frequency, a dimension whose
    /// wavelength is longer than `betaSlow` turns takes the frequency divided
    /// by `factor`, and the dimensions between the two mix the pair.
    ///
    /// The high end of that ramp clamps to `dim - 1`, which is the table's own
    /// end. Without the clamp a long original context puts the high end past
    /// the table and every dimension keeps its plain frequency, which is the
    /// wrong answer for a long context.
    ///
    /// - Parameters:
    ///   - dim: The rotary width. The table holds `dim / 2` entries.
    ///   - base: The rope theta of this layer.
    ///   - originalMaxPositionEmbeddings: The context the checkpoint trained
    ///     on, before YaRN widened it.
    ///   - factor: The YaRN scaling. A factor of 1 is no scaling.
    ///   - betaFast: The turn count that sets the low end of the ramp.
    ///   - betaSlow: The turn count that sets the high end of the ramp.
    /// - Returns: The table, shape `(dim / 2)`, in float32.
    static func yarnInvFreq(
        dim: Int,
        base: Float,
        originalMaxPositionEmbeddings: Int,
        factor: Float,
        betaFast: Float,
        betaSlow: Float
    ) -> MLXArray {
        let pairCount = dim / rotaryPairWidth
        var plain: [Float] = []
        plain.reserveCapacity(pairCount)
        for pair in 0 ..< pairCount {
            plain.append(1 / pow(base, Float(rotaryPairWidth * pair) / Float(dim)))
        }
        let plainTable = MLXArray(plain)
        guard factor != 1 else { return plainTable }

        func correctionDim(_ turns: Float) -> Float {
            Float(dim) * log(Float(originalMaxPositionEmbeddings) / (turns * fullTurn))
                / (Float(rotaryPairWidth) * log(base))
        }

        let low = max(floor(correctionDim(betaFast)), 0)
        var high = min(ceil(correctionDim(betaSlow)), Float(dim - 1))
        if low == high { high += degenerateRangeWidth }

        var smooth: [Float] = []
        smooth.reserveCapacity(pairCount)
        for pair in 0 ..< pairCount {
            let ramp = (Float(pair) - low) / (high - low)
            smooth.append(1 - max(0, min(1, ramp)))
        }
        let smoothTable = MLXArray(smooth)
        return plainTable / factor * (1 - smoothTable) + plainTable * smoothTable
    }

    // MARK: - Mixture-of-experts gate scoring

    /// Scores the routing logits as `sqrt(softplus(logits))`.
    ///
    /// DeepSeek-V4 reads this in place of the softmax the earlier families
    /// read. It rises with the logit and it does not make the scores add up to
    /// 1, thus the hash-routed layers and the gated layers read one scoring
    /// function.
    ///
    /// `softplus` arrives as `logAddExp(logits, 0)`, which stays finite at a
    /// large positive logit and falls to zero at a large negative one.
    ///
    /// - Parameter logits: The routing logits.
    /// - Returns: The scores, of the shape of `logits`.
    static func sqrtSoftplus(_ logits: MLXArray) -> MLXArray {
        sqrt(logAddExp(logits, zeros(like: logits)))
    }

    // MARK: - Clamped SwiGLU

    /// Reads the DeepSeek-V4 expert activation `silu(gate) * up` with a limit.
    ///
    /// The limit holds `up` inside `-limit` through `limit` and holds `gate`
    /// below `limit`. The two sides are not the same: `silu` already falls to
    /// zero at a large negative gate, thus only the high side of the gate can
    /// run away, while `up` multiplies and can run away on either side. This
    /// asymmetry is the reference's, and a symmetric clamp gives a different
    /// answer at a large negative gate.
    ///
    /// A limit of zero or less turns the clamp off.
    ///
    /// The arithmetic runs in float32 and the answer casts back to the dtype
    /// of `gate`. Across 43 layers of experts a float16 multiply of two
    /// clamped values loses too much for the down projection that follows.
    ///
    /// - Parameters:
    ///   - gate: The gate projection.
    ///   - up: The up projection.
    ///   - limit: The `swiglu_limit` of the checkpoint.
    /// - Returns: The activation, of the shape of `gate` and its dtype.
    static func clampedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
        let outputDType = gate.dtype
        var gateF32 = gate.asType(.float32)
        var upF32 = up.asType(.float32)
        if limit > 0 {
            gateF32 = minimum(gateF32, limit)
            upF32 = clip(upF32, min: -limit, max: limit)
        }
        return (silu(gateF32) * upF32).asType(outputDType)
    }

    // MARK: - Routed expert reduction

    /// The axis of a routed-expert stack that holds one expert.
    private static let routedExpertAxis = -2

    /// Adds the routed expert outputs of each token in float32.
    ///
    /// Each output already holds the weight of its route, thus this is a
    /// plain sum. The sum runs in float32 whatever the activation dtype is,
    /// because a bfloat16 running total drops a small expert beside a large
    /// one: `1024 + 1` rounds back to `1024`.
    ///
    /// - Parameter routed: The expert outputs, shape `(..., experts, width)`.
    /// - Returns: The sum, shape `(..., width)`, in float32. The caller adds
    ///   the shared expert and casts one time.
    static func reduceRoutedExpertsFP32(_ routed: MLXArray) -> MLXArray {
        routed.asType(.float32).sum(axis: routedExpertAxis)
    }
}
