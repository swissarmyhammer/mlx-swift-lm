// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Three details do not come from that file. The DeepSeek-V4 Python reference
// -- Thump604/mlx-lm @ deepseek-v4-support-fixes, mlx_lm/models/deepseek_v4.py,
// `HyperConnection.hc_pre`, `HyperConnection.hc_post` and
// `HyperHead.__call__` -- decides each one:
//
//  1. The epsilon of the RMS reduction. The Python holds two epsilons and
//     reads each one in one place: `norm_eps`, which is `rms_norm_eps`, goes
//     into the `rsqrt` of the reduction, and `hc_eps` goes into the sigmoid
//     and into the Sinkhorn steps. The file above reads `hc_eps` in the
//     reduction of the collapse, and it reads `rms_norm_eps` as the `hc_eps`
//     of the head, thus it swaps the two on each side. The DeepSeek-V4-Flash
//     checkpoint gives 1e-6 for both keys, thus the swap gives the same
//     numbers there and different numbers on a checkpoint whose two keys
//     differ. This file reads each epsilon where the Python reads it.
//  2. The dtype of the collapse and of the expand. The Python casts the
//     hidden states to float32, runs the whole reduction, the weighted sum
//     and the expand in float32, and casts one time at the end. The file
//     above casts the normalized states back to the dtype of the model
//     before the mixes projection, and casts `pre`, `post` and `comb` down
//     to that dtype before the sums. On bfloat16 activations the two answers
//     differ, and the comment of that file states why the float32 path is
//     the one that holds: the reduction runs over `hcMult * hiddenSize`
//     numbers, which is 16384 on DeepSeek-V4-Flash, and bfloat16 loses that
//     sum.
//  3. The Sinkhorn split. The file above and this file both call one shared
//     routine. This file calls the two halves of
//     `DeepSeekV4Math.hcSplitSinkhorn`, from
//     Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, which is the
//     transcription of the Python `hc_split_sinkhorn`, and it hands them the
//     `hc_eps` of the checkpoint. The routine holds no epsilon of its own.
//     On the GPU the collapse gives the split activations, the Sinkhorn
//     steps and the weighted sum to one fused Metal kernel,
//     Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift, and the RMS
//     reduction and the mixes projection to one more fused kernel,
//     Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift. On another
//     device it calls the split half, `DeepSeekV4Math.hcSplitFields`, and
//     then the Sinkhorn half, `DeepSeekV4Math.sinkhorn`. A parity test holds
//     the Sinkhorn kernel to the bits of the op path, and a tolerance test
//     holds the mixes kernel to the numbers of the op path.
//
// The Python reference above does not decide how the expand reads the mixing
// matrix `comb`. Its `hc_post` reads `comb` the wrong way round. The official
// DeepSeek-V4 code decides that reading, and
// `DeepSeekV4HyperConnection.expand` states the source.
//
// The names of the three checkpoint tensors are `fn`, `base` and `scale`,
// which are the names the Python gives them. A DeepSeek-V4 checkpoint spells
// them `hc_attn_fn`, `hc_ffn_fn` and `hc_head_fn` and so on, and the Python
// `sanitize` maps each spelling onto the plain name. The weight-load side of
// that map is its own work and is not in this file.

import Foundation
import MLX
import MLXNN

// MARK: - The shape of a parallel residual stream

/// The axes of a `(batch, tokens, copies, width)` residual stream.
///
/// DeepSeek-V4 carries `hc_mult` parallel copies of the residual stream
/// between the blocks, thus every tensor of this file has one more axis than
/// the usual `(batch, tokens, width)`.
private enum StreamAxis {
    /// The axis that holds the batch.
    static let batch = 0
    /// The axis that holds the tokens.
    static let token = 1
    /// The axis that holds the parallel copies of the residual stream.
    static let copy = 2
    /// The axis that holds the numbers of one copy.
    static let width = 3
    /// The number of axes a parallel residual stream carries.
    static let count = 4
}

/// The axes of a `(batch, tokens, copies, copies)` mixing matrix `comb`.
///
/// `comb[j][k]` is the weight that copy `j` of the residual stream gives to
/// copy `k` of the answer of the expand.
private enum MixingAxis {
    /// The axis that holds the copy of the residual stream the weight reads.
    static let source = 2
    /// The axis that holds the copy of the answer the weight writes.
    static let target = 3
}

/// The two steps the collapse of ``DeepSeekV4HyperConnection`` and the
/// reduction of ``DeepSeekV4HyperHead`` both take.
///
/// The two layers read one mixes projection and one weighted sum over the
/// copies. They differ only in what they do with the mixes vector, thus the
/// steps live here and each layer calls them.
///
/// The enum is internal, not private, because the parity test of
/// ``DeepSeekV4SinkhornKernel`` reads ``weightedSum(of:weights:)`` as the op
/// path the kernel must agree with.
enum ManifoldStream {

    /// Stops a residual stream whose shape is not the one the layer expects.
    ///
    /// - Parameters:
    ///   - stream: The residual stream.
    ///   - copyCount: The number of parallel copies the stream must carry.
    ///   - width: The width one copy must carry.
    static func checkShape(of stream: MLXArray, copyCount: Int, width: Int) {
        precondition(
            stream.ndim == StreamAxis.count,
            "the residual stream must carry \(StreamAxis.count) axes, "
                + "and it carries \(stream.ndim)")
        precondition(
            stream.dim(StreamAxis.copy) == copyCount,
            "the copy axis must hold \(copyCount) copies, "
                + "and it holds \(stream.dim(StreamAxis.copy))")
        precondition(
            stream.dim(StreamAxis.width) == width,
            "one copy must hold \(width) numbers, "
                + "and it holds \(stream.dim(StreamAxis.width))")
    }

    /// Normalizes the flattened copies of the residual stream and projects
    /// them onto the mixes vector.
    ///
    /// The Python reads `rsqrt(mean(x * x) + norm_eps)` over the flattened
    /// copies and multiplies the projection by it. `MLXFast.rmsNorm` with a
    /// weight of every value 1 is that same reduction, and the multiply moves
    /// ahead of the projection, which is the same number because the
    /// reduction gives one scalar for each row.
    ///
    /// - Parameters:
    ///   - stream: The residual stream in float32, shape
    ///     `(batch, tokens, copies, width)`, already checked by
    ///     ``checkShape(of:copyCount:width:)``.
    ///   - projection: The mixes projection, shape `(mixes, copies * width)`.
    ///   - normWeight: The weight of the reduction, every value 1, shape
    ///     `(copies * width)`.
    ///   - normEps: The epsilon of the reduction, from `rms_norm_eps`.
    /// - Returns: The mixes vector, shape `(batch, tokens, mixes)`, in float32.
    static func mixes(
        of stream: MLXArray, projection: MLXArray, normWeight: MLXArray, normEps: Float
    ) -> MLXArray {
        let batch = stream.dim(StreamAxis.batch)
        let tokens = stream.dim(StreamAxis.token)
        let hcDim = stream.dim(StreamAxis.copy) * stream.dim(StreamAxis.width)
        let flattened = stream.reshaped([batch, tokens, hcDim])
        let normalized = MLXFast.rmsNorm(
            flattened, weight: normWeight.asType(.float32), eps: normEps)
        return normalized.matmul(projection.asType(.float32).transposed())
    }

    /// Adds the copies of the residual stream up, one weight for each copy.
    ///
    /// - Parameters:
    ///   - stream: The residual stream in float32, shape
    ///     `(batch, tokens, copies, width)`.
    ///   - weights: One weight for each copy, shape `(batch, tokens, copies)`.
    ///     The trailing axis of 1 the sum adds spreads each weight across the
    ///     width of its own copy.
    /// - Returns: The single stream, shape `(batch, tokens, width)`, in
    ///   float32.
    static func weightedSum(of stream: MLXArray, weights: MLXArray) -> MLXArray {
        (weights.expandedDimensions(axis: -1) * stream).sum(axis: StreamAxis.copy)
    }

    /// Takes every step of the collapse of ``DeepSeekV4HyperConnection`` that
    /// comes before the Sinkhorn steps: the float32 cast of the stream, the
    /// mixes projection and the split of the mixes vector.
    ///
    /// ``DeepSeekV4CompiledDecode/collapseFields(hcMult:normEps:hcEps:)``
    /// compiles this function for the decode step.
    ///
    /// - Parameters:
    ///   - stream: The residual stream, shape `(batch, tokens, copies, width)`,
    ///     already checked by ``checkShape(of:copyCount:width:)``.
    ///   - projection: The mixes projection, shape `(mixes, copies * width)`.
    ///   - normWeight: The weight of the reduction, every value 1, shape
    ///     `(copies * width)`.
    ///   - scale: The three learned scales, one for each field, shape `(3)`.
    ///   - base: The learned bias of the mixes vector, shape `(mixes)`.
    ///   - hcMult: The number of parallel copies of the residual stream.
    ///   - normEps: The epsilon of the reduction, from `rms_norm_eps`.
    ///   - hcEps: The epsilon of the split, from `hc_eps`.
    /// - Returns: `stream`, the residual stream in float32; `pre` and `post`,
    ///   each shape `(batch, tokens, copies)`; and `comb`, the row softmax of
    ///   the mixing logits plus `hcEps`, shape `(batch, tokens, copies,
    ///   copies)`. The Sinkhorn steps still have to run on `comb`.
    static func collapseFields(
        of stream: MLXArray, projection: MLXArray, normWeight: MLXArray, scale: MLXArray,
        base: MLXArray, hcMult: Int, normEps: Float, hcEps: Float
    ) -> (stream: MLXArray, pre: MLXArray, post: MLXArray, comb: MLXArray) {
        collapseFields(
            of: stream, projection: projection, normWeight: normWeight, scale: scale, base: base,
            hcMult: hcMult, normEps: normEps
        ) { DeepSeekV4Math.hcSplitActivations($0, eps: hcEps) }
    }

    /// Takes the steps of
    /// ``collapseFields(of:projection:normWeight:scale:base:hcMult:normEps:hcEps:)``
    /// with the split activations the caller gives.
    ///
    /// A prefill block gives the shapeless compiled activations of
    /// ``DeepSeekV4CompiledDecode/splitActivations(hcEps:)``, which give the
    /// bits of the op path.
    ///
    /// - Parameters:
    ///   - stream: The residual stream, shape `(batch, tokens, copies, width)`,
    ///     already checked by ``checkShape(of:copyCount:width:)``.
    ///   - projection: The mixes projection, shape `(mixes, copies * width)`.
    ///   - normWeight: The weight of the reduction, every value 1, shape
    ///     `(copies * width)`.
    ///   - scale: The three learned scales, one for each field, shape `(3)`.
    ///   - base: The learned bias of the mixes vector, shape `(mixes)`.
    ///   - hcMult: The number of parallel copies of the residual stream.
    ///   - normEps: The epsilon of the reduction, from `rms_norm_eps`.
    ///   - activations: The activations of the three split fields.
    /// - Returns: The fields
    ///   ``collapseFields(of:projection:normWeight:scale:base:hcMult:normEps:hcEps:)``
    ///   gives.
    static func collapseFields(
        of stream: MLXArray, projection: MLXArray, normWeight: MLXArray, scale: MLXArray,
        base: MLXArray, hcMult: Int, normEps: Float,
        activations: (DeepSeekV4Math.HCSplitOperands) -> (
            pre: MLXArray, post: MLXArray, comb: MLXArray
        )
    ) -> (stream: MLXArray, pre: MLXArray, post: MLXArray, comb: MLXArray) {
        let streamF32 = stream.asType(.float32)
        let mixes = mixes(
            of: streamF32, projection: projection, normWeight: normWeight, normEps: normEps)
        let fields = activations(
            DeepSeekV4Math.hcSplitOperands(mixes: mixes, scale: scale, base: base, hcMult: hcMult))
        return (stream: streamF32, pre: fields.pre, post: fields.post, comb: fields.comb)
    }

    /// Writes a block output back into the parallel copies of the residual
    /// stream, as the op path.
    ///
    /// ``DeepSeekV4HyperConnection/expand(blockOutput:residual:post:comb:)``
    /// states the reading of `comb` this function makes and its source, and
    /// ``DeepSeekV4CompiledDecode/expand`` compiles this function for each
    /// block.
    ///
    /// - Parameters:
    ///   - blockOutput: The output of the sub-layer, shape
    ///     `(batch, tokens, width)`.
    ///   - residual: The residual stream the matching collapse read, shape
    ///     `(batch, tokens, copies, width)`.
    ///   - post: The gain each copy gives the block output, shape
    ///     `(batch, tokens, copies)`.
    ///   - comb: The doubly stochastic mixing matrix, shape
    ///     `(batch, tokens, copies, copies)`.
    /// - Returns: The residual stream of the next sub-layer, shape
    ///   `(batch, tokens, copies, width)`, in the dtype of `blockOutput`.
    static func expand(
        blockOutput: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        let outputDType = blockOutput.dtype
        // `post` holds one gain for each copy and carries no width axis, and
        // `blockOutput` holds one width and carries no copy axis, thus each
        // one takes the axis the other carries.
        let newTerm =
            post.asType(.float32).expandedDimensions(axis: -1)
            * blockOutput.asType(.float32).expandedDimensions(axis: StreamAxis.copy)
        let residualTerm = comb.asType(.float32)
            .swappedAxes(MixingAxis.source, MixingAxis.target)
            .matmul(residual.asType(.float32))
        return (newTerm + residualTerm).asType(outputDType)
    }
}

// MARK: - Hyper-connection

/// The manifold hyper-connection of one DeepSeek-V4 sub-layer.
///
/// DeepSeek-V4 does not carry one residual stream between its blocks. It
/// carries `hc_mult` parallel copies of it. Each sub-layer -- the attention
/// half and the mixture-of-experts half -- reads those copies through a
/// collapse, runs its block on the single stream that comes out, and writes
/// the block output back through an expand.
///
/// The collapse and the expand share one projection. The collapse reads the
/// flattened copies, normalizes them, and projects them onto a mixes vector of
/// `(2 + hcMult) * hcMult` numbers. `DeepSeekV4Math.hcSplitSinkhorn` splits
/// that vector into three fields: `pre`, which weighs each copy on the way in,
/// `post`, which weighs the block output on the way out, and `comb`, a doubly
/// stochastic mixing matrix that carries the copies across.
///
/// `comb` is square and doubly stochastic, thus its transpose carries the same
/// shape and the same row and column sums. An expand that read the matrix the
/// wrong way round would therefore pass every shape check and answer numbers
/// that look reasonable and are wrong.
/// ``expand(blockOutput:residual:post:comb:)`` states the reading it makes,
/// and the tests hold it to values rather than to shapes.
class DeepSeekV4HyperConnection: Module {

    /// The number of learned scales, one for each of the three fields the
    /// mixes vector splits into.
    private static let scaleFieldCount = 3

    /// The number of parallel copies of the residual stream.
    let hcMult: Int

    /// The width of the residual stream.
    let hiddenSize: Int

    /// The number of Sinkhorn steps that make the mixing matrix doubly
    /// stochastic, from `hc_sinkhorn_iters`.
    let sinkhornIterations: Int

    /// The epsilon of the sigmoid and of the Sinkhorn steps, from `hc_eps`.
    let hcEps: Float

    /// The epsilon of the RMS reduction, from `rms_norm_eps`.
    let normEps: Float

    /// The mixes projection, shape
    /// `((2 + hcMult) * hcMult, hcMult * hiddenSize)`.
    @ParameterInfo(key: "fn") var mixProjection: MLXArray

    /// The learned bias of the mixes vector, shape `((2 + hcMult) * hcMult)`.
    @ParameterInfo(key: "base") var mixBias: MLXArray

    /// The learned scale of each of the three fields, shape `(3)`.
    @ParameterInfo(key: "scale") var mixScale: MLXArray

    /// The weight of the RMS reduction, which is every value 1.
    ///
    /// The Python reduction is a plain `rsqrt` of the mean square and carries
    /// no learned gain. `MLXFast.rmsNorm` asks for a weight, thus this vector
    /// stands in for the gain the Python does not have.
    ///
    /// The name starts with an underscore on purpose. mlx-swift collects every
    /// stored `MLXArray` of a `Module` as a parameter unless its name starts
    /// with an underscore, and this vector comes from the configuration rather
    /// than from the checkpoint. Without the underscore a weight-load check
    /// would demand a tensor that no DeepSeek-V4 checkpoint holds.
    private let _normWeight: MLXArray

    /// The compiled steps of the collapse before the Sinkhorn steps, which a
    /// decode step off the GPU reads.
    ///
    /// The function captures the constants of this layer alone, and takes
    /// every weight as an argument on each call.
    private let compiledCollapseFields:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )

    /// The shapeless compiled activations of the split, which a block of
    /// more than one token off the GPU reads.
    private let compiledSplitActivations:
        @Sendable (DeepSeekV4Math.HCSplitOperands) -> (
            pre: MLXArray, post: MLXArray, comb: MLXArray
        )

    /// Builds the hyper-connection of one sub-layer.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    init(configuration: DeepSeekV4Configuration) {
        self.hcMult = configuration.hcMult
        self.hiddenSize = configuration.hiddenSize
        self.sinkhornIterations = configuration.hcSinkhornIters
        self.hcEps = configuration.hcEps
        self.normEps = configuration.rmsNormEps

        let mixWidth = DeepSeekV4Math.mixWidth(hcMult: configuration.hcMult)
        let hcDim = configuration.hcMult * configuration.hiddenSize
        self._mixProjection.wrappedValue = MLXArray.zeros([mixWidth, hcDim])
        self._mixBias.wrappedValue = MLXArray.zeros([mixWidth])
        self._mixScale.wrappedValue = MLXArray.zeros([Self.scaleFieldCount])
        self._normWeight = MLXArray.ones([hcDim])
        self.compiledCollapseFields = DeepSeekV4CompiledDecode.collapseFields(
            hcMult: configuration.hcMult, normEps: configuration.rmsNormEps,
            hcEps: configuration.hcEps)
        self.compiledSplitActivations = DeepSeekV4CompiledDecode.splitActivations(
            hcEps: configuration.hcEps)
    }

    /// Reads the parallel copies of the residual stream down to one stream.
    ///
    /// The answer also carries the two fields the matching expand needs, so
    /// that the mixes projection runs one time for each sub-layer rather than
    /// two times.
    ///
    /// On the GPU the collapse takes two fused kernels and no other
    /// dispatch. ``DeepSeekV4MixesKernel`` takes the RMS reduction and the
    /// mixes projection, and ``DeepSeekV4SinkhornKernel`` takes the split
    /// activations, the Sinkhorn steps and the weighted sum. Both read the
    /// stream in its own dtype. The mixes kernel adds its products in another
    /// order than the op path, thus the two paths differ in the last float32
    /// places of the mixes. A stream the mixes kernel cannot read takes the
    /// mixes projection as MLX operations. On another device, or with no
    /// Sinkhorn step, the collapse takes every step as MLX operations.
    ///
    /// - Parameter stream: The residual stream, shape
    ///   `(batch, tokens, hcMult, hiddenSize)`.
    /// - Returns: `collapsed`, the single stream the block reads, shape
    ///   `(batch, tokens, hiddenSize)`; `post`, the gain each copy gives the
    ///   block output, shape `(batch, tokens, hcMult)`; and `comb`, the doubly
    ///   stochastic mixing matrix, shape `(batch, tokens, hcMult, hcMult)`.
    func collapse(
        _ stream: MLXArray
    ) -> (collapsed: MLXArray, post: MLXArray, comb: MLXArray) {
        ManifoldStream.checkShape(of: stream, copyCount: hcMult, width: hiddenSize)
        guard DeepSeekV4SinkhornKernel.canRun(on: stream, iters: sinkhornIterations) else {
            let fields = collapseFields(stream)
            let comb = DeepSeekV4Math.sinkhorn(fields.comb, iters: sinkhornIterations, eps: hcEps)
            let collapsed = ManifoldStream.weightedSum(of: fields.stream, weights: fields.pre)
            return (collapsed: collapsed.asType(stream.dtype), post: fields.post, comb: comb)
        }

        let fused = DeepSeekV4SinkhornKernel.splitSinkhornCollapse(
            mixes: mixes(of: stream), scale: mixScale, base: mixBias, stream: stream,
            iters: sinkhornIterations, eps: hcEps)
        return (collapsed: fused.collapsed, post: fused.post, comb: fused.comb)
    }

    /// Takes the RMS reduction and the mixes projection of the collapse on
    /// the GPU.
    ///
    /// - Parameter stream: The residual stream, already checked.
    /// - Returns: The mixes vector, shape `(batch, tokens, mixes)`, in
    ///   float32.
    private func mixes(of stream: MLXArray) -> MLXArray {
        guard DeepSeekV4MixesKernel.canRun(on: stream) else {
            return ManifoldStream.mixes(
                of: stream.asType(.float32), projection: mixProjection, normWeight: _normWeight,
                normEps: normEps)
        }
        return DeepSeekV4MixesKernel.mixes(of: stream, projection: mixProjection, normEps: normEps)
    }

    /// Takes the steps of the collapse that come before the Sinkhorn steps,
    /// on the path off the GPU.
    ///
    /// A decode step reads the whole compiled function. A longer block reads
    /// the op path for the projection and the slices, and the shapeless
    /// compiled split activations. Each path gives the bits of the op path.
    ///
    /// - Parameter stream: The residual stream, already checked.
    /// - Returns: The fields
    ///   ``ManifoldStream/collapseFields(of:projection:normWeight:scale:base:hcMult:normEps:hcEps:)``
    ///   answers.
    private func collapseFields(
        _ stream: MLXArray
    ) -> (stream: MLXArray, pre: MLXArray, post: MLXArray, comb: MLXArray) {
        guard DeepSeekV4CompiledDecode.isDecodeStep(stream) else {
            return ManifoldStream.collapseFields(
                of: stream, projection: mixProjection, normWeight: _normWeight,
                scale: mixScale, base: mixBias, hcMult: hcMult, normEps: normEps,
                activations: compiledSplitActivations)
        }
        let (streamF32, pre, post, comb) = compiledCollapseFields(
            stream, mixProjection, _normWeight, mixScale, mixBias)
        return (stream: streamF32, pre: pre, post: post, comb: comb)
    }

    /// Writes the block output back into the parallel copies of the residual
    /// stream.
    ///
    /// The answer is the sum of two terms. The new term gives copy `k` the
    /// block output scaled by `post[k]`. The residual term mixes the copies
    /// the matching collapse read: copy `k` of the answer takes
    /// `sum over j of comb[j][k] * residual[j]`, thus column `k` of the mixing
    /// matrix decides answer copy `k`, and row `j` decides how far residual
    /// copy `j` reaches. The residual term is therefore `comb` transposed,
    /// times `residual`.
    ///
    /// That reading is the one the official DeepSeek-V4 code makes:
    /// deepseek-ai/DeepSeek-V4-Flash @
    /// 60d8d70770c6776ff598c94bb586a859a38244f1, `inference/model.py`,
    /// `Block.hc_post`, which adds
    /// `torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`. The
    /// sum runs over the first matrix axis of `comb`. Apple mlx-lm and
    /// Blaizzy mlx-vlm read `comb` the same way. The Thump604 reference that
    /// the header of this file names reads `comb` without the transpose, and
    /// that reading is wrong. The layout of `comb` is the same in the
    /// official code and in this file: `DeepSeekV4Math.hcSplitFields`
    /// reshapes the mixes row-major and takes the softmax over the last axis,
    /// as the official `hc_split_sinkhorn` kernel does.
    ///
    /// On the GPU one fused kernel, ``DeepSeekV4ExpandKernel``, takes the
    /// whole expand. It adds the products in another order than the op path,
    /// thus the two paths differ in the last float32 places. A block output
    /// the kernel cannot read, and every block off the GPU, takes the
    /// compiled op path ``DeepSeekV4CompiledDecode/expand``.
    ///
    /// - Parameters:
    ///   - blockOutput: The output of the sub-layer, shape
    ///     `(batch, tokens, hiddenSize)`.
    ///   - residual: The residual stream the matching collapse read, shape
    ///     `(batch, tokens, hcMult, hiddenSize)`.
    ///   - post: The gain each copy gives the block output, shape
    ///     `(batch, tokens, hcMult)`.
    ///   - comb: The doubly stochastic mixing matrix, shape
    ///     `(batch, tokens, hcMult, hcMult)`.
    /// - Returns: The residual stream of the next sub-layer, shape
    ///   `(batch, tokens, hcMult, hiddenSize)`.
    func expand(
        blockOutput: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        guard DeepSeekV4ExpandKernel.canRun(blockOutput: blockOutput, residual: residual) else {
            return DeepSeekV4CompiledDecode.expand(blockOutput, residual, post, comb)
        }
        return DeepSeekV4ExpandKernel.expand(
            blockOutput: blockOutput, residual: residual, post: post, comb: comb)
    }
}

// MARK: - Hyper head

/// The manifold hyper-connection at the top of a DeepSeek-V4 stack.
///
/// The last block hands out `hc_mult` parallel copies of the residual stream,
/// and the final norm and the language-model head read one stream. This
/// reduction weighs each copy by a sigmoid of its own mixes value and adds the
/// copies up.
///
/// It is the simpler half of ``DeepSeekV4HyperConnection``: it reads the same
/// mixes projection, and it takes no Sinkhorn step, because it collapses the
/// copies one time and never expands them again.
class DeepSeekV4HyperHead: Module {

    /// The number of learned scales. The head holds one scale, where a
    /// hyper-connection holds one for each of its three fields.
    private static let scaleFieldCount = 1

    /// The number of parallel copies of the residual stream.
    let hcMult: Int

    /// The width of the residual stream.
    let hiddenSize: Int

    /// The epsilon the sigmoid adds, from `hc_eps`.
    let hcEps: Float

    /// The epsilon of the RMS reduction, from `rms_norm_eps`.
    let normEps: Float

    /// The mixes projection, shape `(hcMult, hcMult * hiddenSize)`.
    @ParameterInfo(key: "fn") var mixProjection: MLXArray

    /// The learned bias, one value for each copy, shape `(hcMult)`.
    @ParameterInfo(key: "base") var mixBias: MLXArray

    /// The learned scale, shape `(1)`.
    @ParameterInfo(key: "scale") var mixScale: MLXArray

    /// The weight of the RMS reduction, which is every value 1.
    ///
    /// The leading underscore keeps mlx-swift from collecting this constant as
    /// a checkpoint parameter, the same way it does in
    /// ``DeepSeekV4HyperConnection``.
    private let _normWeight: MLXArray

    /// Builds the reduction at the top of the stack.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    init(configuration: DeepSeekV4Configuration) {
        self.hcMult = configuration.hcMult
        self.hiddenSize = configuration.hiddenSize
        self.hcEps = configuration.hcEps
        self.normEps = configuration.rmsNormEps

        let hcDim = configuration.hcMult * configuration.hiddenSize
        self._mixProjection.wrappedValue = MLXArray.zeros([configuration.hcMult, hcDim])
        self._mixBias.wrappedValue = MLXArray.zeros([configuration.hcMult])
        self._mixScale.wrappedValue = MLXArray.zeros([Self.scaleFieldCount])
        self._normWeight = MLXArray.ones([hcDim])
    }

    /// Reads the parallel copies of the residual stream down to one stream.
    ///
    /// - Parameter stream: The residual stream, shape
    ///   `(batch, tokens, hcMult, hiddenSize)`.
    /// - Returns: The single stream the final norm reads, shape
    ///   `(batch, tokens, hiddenSize)`.
    func callAsFunction(_ stream: MLXArray) -> MLXArray {
        ManifoldStream.checkShape(of: stream, copyCount: hcMult, width: hiddenSize)
        let streamF32 = stream.asType(.float32)
        let mixes = ManifoldStream.mixes(
            of: streamF32, projection: mixProjection, normWeight: _normWeight,
            normEps: normEps)

        // The scale vector holds one value, thus it broadcasts across the
        // mixes as it is. An `Int` subscript would add a `Gather` node.
        let scale = mixScale.asType(.float32)
        let weights = sigmoid(mixes * scale + mixBias.asType(.float32)) + hcEps
        return ManifoldStream.weightedSum(of: streamF32, weights: weights).asType(stream.dtype)
    }
}
