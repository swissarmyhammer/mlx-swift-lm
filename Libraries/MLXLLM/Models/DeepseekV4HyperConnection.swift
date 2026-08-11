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
//     routine. This file calls `DeepseekV4Math.hcSplitSinkhorn`, from
//     Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift, which is the
//     transcription of the Python `hc_split_sinkhorn`, and it hands that
//     routine the `hc_eps` of the checkpoint. The routine holds no epsilon
//     of its own.
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

/// The two steps the collapse of ``DeepseekV4HyperConnection`` and the
/// reduction of ``DeepseekV4HyperHead`` both take.
///
/// The two layers read one mixes projection and one weighted sum over the
/// copies. They differ only in what they do with the mixes vector, thus the
/// steps live here and each layer calls them.
private enum ManifoldStream {

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
/// `(2 + hcMult) * hcMult` numbers. `DeepseekV4Math.hcSplitSinkhorn` splits
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
class DeepseekV4HyperConnection: Module {

    /// The number of one-value-for-each-copy fields the mixes vector holds
    /// ahead of the flattened mixing matrix: one `pre` weight and one `post`
    /// weight for each copy.
    private static let splitFieldCount = 2

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

    /// Builds the hyper-connection of one sub-layer.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    init(configuration: DeepseekV4Configuration) {
        self.hcMult = configuration.hcMult
        self.hiddenSize = configuration.hiddenSize
        self.sinkhornIterations = configuration.hcSinkhornIters
        self.hcEps = configuration.hcEps
        self.normEps = configuration.rmsNormEps

        let mixWidth = (Self.splitFieldCount + configuration.hcMult) * configuration.hcMult
        let hcDim = configuration.hcMult * configuration.hiddenSize
        self._mixProjection.wrappedValue = MLXArray.zeros([mixWidth, hcDim])
        self._mixBias.wrappedValue = MLXArray.zeros([mixWidth])
        self._mixScale.wrappedValue = MLXArray.zeros([Self.scaleFieldCount])
        self._normWeight = MLXArray.ones([hcDim])
    }

    /// Reads the parallel copies of the residual stream down to one stream.
    ///
    /// The answer also carries the two fields the matching expand needs, so
    /// that the mixes projection runs one time for each sub-layer rather than
    /// two times.
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
        let streamF32 = stream.asType(.float32)
        let mixes = ManifoldStream.mixes(
            of: streamF32, projection: mixProjection, normWeight: _normWeight,
            normEps: normEps)

        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: mixScale, base: mixBias,
            hcMult: hcMult, iters: sinkhornIterations, eps: hcEps)

        let collapsed = ManifoldStream.weightedSum(of: streamF32, weights: pre)
        return (collapsed: collapsed.asType(stream.dtype), post: post, comb: comb)
    }

    /// Writes the block output back into the parallel copies of the residual
    /// stream.
    ///
    /// The answer is the sum of two terms. The new term gives copy `i` the
    /// block output scaled by `post[i]`. The residual term mixes the copies
    /// the matching collapse read: copy `i` of the answer takes
    /// `sum over j of comb[i][j] * residual[j]`, thus row `i` of the mixing
    /// matrix decides answer copy `i`, and column `j` decides how far residual
    /// copy `j` reaches. That reading is the one the Python
    /// `einsum("bsij,bsjd->bsid", comb, residual)` makes, and it is the whole
    /// axis alignment of the manifold hyper-connections.
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
        let outputDType = blockOutput.dtype
        // `post` holds one gain for each copy and carries no width axis, and
        // `blockOutput` holds one width and carries no copy axis, thus each
        // one takes the axis the other carries.
        let newTerm =
            post.asType(.float32).expandedDimensions(axis: -1)
            * blockOutput.asType(.float32).expandedDimensions(axis: StreamAxis.copy)
        let residualTerm = comb.asType(.float32).matmul(residual.asType(.float32))
        return (newTerm + residualTerm).asType(outputDType)
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
/// It is the simpler half of ``DeepseekV4HyperConnection``: it reads the same
/// mixes projection, and it takes no Sinkhorn step, because it collapses the
/// copies one time and never expands them again.
class DeepseekV4HyperHead: Module {

    /// The number of learned scales. The head holds one scale, where a
    /// hyper-connection holds one for each of its three fields.
    private static let scaleFieldCount = 1

    /// The place of the learned scale inside the scale vector.
    private static let scaleIndex = 0

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
    /// ``DeepseekV4HyperConnection``.
    private let _normWeight: MLXArray

    /// Builds the reduction at the top of the stack.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    init(configuration: DeepseekV4Configuration) {
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

        let scale = mixScale.asType(.float32)[Self.scaleIndex]
        let weights = sigmoid(mixes * scale + mixBias.asType(.float32)) + hcEps
        return ManifoldStream.weightedSum(of: streamF32, weights: weights).asType(stream.dtype)
    }
}
