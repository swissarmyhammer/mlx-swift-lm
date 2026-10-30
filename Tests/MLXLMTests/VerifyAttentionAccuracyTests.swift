// Copyright © 2026 Apple Inc.
//
// The accuracy of ``VerifyAttentionKernel`` against the SDPA of MLX.
//
// The kernel is not bit-exact with the SDPA of MLX, thus a verify through it
// can accept other drafts than a verify through the SDPA. This suite measures
// each path against an attention in float32 that it builds from plain MLX
// operations (a matrix product, a softmax and a second matrix product), over
// the same inputs in the dtype of the model. The kernel must be no farther
// from that reference than the SDPA of MLX is, up to a small slack.
//
// The error is the root mean square of the gap over every output number. The
// largest gap of one number is not a measure of accuracy here: both paths
// round their output to the model dtype, and the one number with the largest
// gap is the one whose rounding moved it most, on either path.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct VerifyAttentionAccuracyTests {

    /// The query token counts of a verify that the kernel reads. MLX reads
    /// the fewer of them with its vector kernel, and the rest with its
    /// unfused SDPA.
    static let rowCounts = Array(
        VerifyAttentionKernel.fewestQueries ... VerifyAttentionKernel.mostQueries)

    /// The key counts of the accuracy test: the fewest keys the kernel reads,
    /// a key count that ends inside a key tile, two key counts near 2k, one
    /// of them with blocks that end inside a key tile, and a key count near
    /// 8k that ends inside a key tile.
    static let keyCounts = [DecodeAttentionKernel.fewestKeys, 1500, 2048, 2100, 8195]

    /// The dtypes of the model weights and the key cache that a verify reads.
    static let modelDtypes: [DType] = [.bfloat16, .float16]

    /// The masks the kernel reads.
    static let causalMasks = [true, false]

    /// The part by which the kernel error may pass the error of the SDPA of
    /// MLX. Both paths round their output to the model dtype, thus most of
    /// their error is the same rounding, and only the rest of it differs.
    static let slack: Float = 0.05

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_030, keys: 20_261_031, values: 20_261_032)

    /// The error of an output against a reference: the root mean square of
    /// their gap, as a part of the root mean square of the reference.
    ///
    /// - Parameters:
    ///   - got: The output of a path, in any float dtype.
    ///   - reference: The reference output, in float32.
    /// - Returns: The relative error.
    static func relativeError(_ got: MLXArray, _ reference: MLXArray) -> Float {
        let gap = got.asType(.float32) - reference
        return (sqrt(mean(gap * gap)) / sqrt(mean(reference * reference))).item(Float.self)
    }

    /// Tells if a path is no less accurate than a baseline path, up to
    /// ``slack``.
    ///
    /// - Parameters:
    ///   - error: The error of the path, from ``relativeError(_:_:)``.
    ///   - baseline: The error of the baseline path.
    /// - Returns: True when `error` is at most `baseline` and its slack.
    static func isNoLessAccurate(_ error: Float, than baseline: Float) -> Bool {
        error <= baseline * (1 + slack)
    }

    /// The scaled dot-product attention of the operands in float32, from
    /// plain MLX operations, with its scores rounded to one dtype before the
    /// softmax. Each query head reads the keys and the values of its KV head,
    /// and a causal query `t` of `T` reads the keys up to key `keys - T + t`.
    ///
    /// - Parameters:
    ///   - operands: The operands, in any float dtype.
    ///   - causal: True for the causal mask, false for no mask.
    ///   - scoreDtype: The dtype the scores are rounded to; float32 keeps
    ///     them whole.
    /// - Returns: The attention output in float32, shape
    ///   `(batch, queryHeads, tokens, headDim)`.
    static func attention(
        _ operands: AttentionKernelOperands, causal: Bool, roundingScoresTo scoreDtype: DType
    ) -> MLXArray {
        let headAxis = AttentionKernelOperands.headAxis
        let rowAxis = AttentionKernelOperands.rowAxis
        let queries = operands.queries.asType(.float32) * operands.scale
        let group = queries.dim(headAxis) / operands.keys.dim(headAxis)
        let keys = repeated(operands.keys.asType(.float32), count: group, axis: headAxis)
        let values = repeated(operands.values.asType(.float32), count: group, axis: headAxis)
        var scores = matmul(queries, keys.swappedAxes(rowAxis, AttentionKernelOperands.widthAxis))
            .asType(scoreDtype).asType(.float32)
        if causal {
            let tokens = queries.dim(rowAxis)
            let keyCount = keys.dim(rowAxis)
            let lastKey = arange(keyCount - tokens, keyCount).expandedDimensions(axis: 1)
            scores = which(arange(keyCount) .<= lastKey, scores, MLXArray(-Float.infinity))
        }
        return matmul(softmax(scores, axis: -1, precise: true), values)
    }

    /// The float32 reference attention of the operands: ``attention(_:causal:roundingScoresTo:)``
    /// with whole float32 scores.
    static func reference(_ operands: AttentionKernelOperands, causal: Bool) -> MLXArray {
        attention(operands, causal: causal, roundingScoresTo: .float32)
    }

    /// The kernel output over the operands, with the mask of a causal flag.
    static func kernelOutput(_ operands: AttentionKernelOperands, causal: Bool) -> MLXArray {
        VerifyAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, causal: causal)
    }

    /// The SDPA of MLX over the operands in their own dtype, with the mask
    /// of a causal flag.
    static func sdpaOutput(_ operands: AttentionKernelOperands, causal: Bool) -> MLXArray {
        operands.sdpa(mask: operands.mask(AttentionMaskKind(causal: causal)))
    }

    /// Gives the check the Qwen3.8 operands of one shape in each model dtype,
    /// with each mask, and with the kernel error against the float32
    /// reference.
    ///
    /// - Parameters:
    ///   - rows: The number of query tokens.
    ///   - keys: The number of cached keys.
    ///   - check: The check of one call: the operands, the causal flag, the
    ///     float32 reference and the kernel error.
    static func forEachCall(
        rows: Int, keys: Int,
        _ check: (
            _ operands: AttentionKernelOperands, _ causal: Bool, _ reference: MLXArray,
            _ kernelError: Float
        ) -> Void
    ) {
        for dtype in Self.modelDtypes {
            let operands = Self.random.qwen38(queries: rows, keys: keys, dtype: dtype)
            for causal in Self.causalMasks {
                let reference = Self.reference(operands, causal: causal)
                let kernel = Self.relativeError(
                    Self.kernelOutput(operands, causal: causal), reference)
                check(operands, causal, reference, kernel)
            }
        }
    }

    /// For each model dtype and each mask, the kernel is no farther from the
    /// float32 reference than the SDPA of MLX is, up to ``slack``.
    @Test(arguments: rowCounts, keyCounts)
    func theKernelIsAsAccurateAsTheSDPA(rows: Int, keys: Int) {
        Self.forEachCall(rows: rows, keys: keys) { operands, causal, reference, kernel in
            let sdpa = Self.relativeError(Self.sdpaOutput(operands, causal: causal), reference)

            #expect(
                Self.isNoLessAccurate(kernel, than: sdpa),
                "\(operands.testDescription), causal \(causal): kernel error \(kernel), SDPA error \(sdpa)"
            )
        }
    }

    /// The positive control of the accuracy test: an attention that rounds
    /// its scores to the model dtype before the softmax is less accurate
    /// than the kernel, past the slack, thus the check tells a less accurate
    /// path from the kernel.
    @Test(arguments: rowCounts, keyCounts)
    func roundedScoresAreLessAccurateThanTheKernel(rows: Int, keys: Int) {
        Self.forEachCall(rows: rows, keys: keys) { operands, causal, reference, kernel in
            let dtype = operands.queries.dtype
            let rounded = Self.relativeError(
                Self.attention(operands, causal: causal, roundingScoresTo: dtype).asType(dtype),
                reference)

            #expect(
                !Self.isNoLessAccurate(rounded, than: kernel),
                "\(operands.testDescription), causal \(causal): rounded-score error \(rounded), kernel error \(kernel)"
            )
        }
    }
}
