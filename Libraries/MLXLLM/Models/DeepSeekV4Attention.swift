// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Three details do not come from that file. The DeepSeek-V4 Python
// reference -- Thump604/mlx-lm @ deepseek-v4-support-fixes,
// mlx_lm/models/deepseek_v4.py, `DeepseekV4RoPE` and `V4Attention` --
// decides each one:
//
//  1. The compress ratio of a layer. The file above invents a pattern when
//     the checkpoint gives no `compress_ratios` list: layer 0 and the last
//     layer take 0, and the layers between alternate 4 and 128. The Python
//     reads `ratios[layer_idx] if layer_idx < len(ratios) else 0` and
//     invents nothing. This file reads `hasCompressor(layer:)` and
//     `ropeTheta(forLayer:)` from ``DeepSeekV4Configuration``, which give
//     the Python answer.
//  2. The inverse-frequency table is not a checkpoint tensor. The Python
//     hides it behind a tuple and says so: "This is derived from config,
//     not a checkpoint parameter". The file above holds it as a plain
//     `invFreq`, which puts `attn.rope.invFreq` into the module parameters
//     and thus into every weight-load check, although no checkpoint gives
//     that key. The leading underscore below keeps mlx-swift's
//     `filterValidParameters` from collecting it, the same way
//     `YarnRoPE._freqs` does in Libraries/MLXLMCommon/RoPEUtils.swift.
//  3. The quantization mode of the grouped output projection. The file
//     above calls `quantizedMatmul` without a `mode:`, thus it reads the
//     affine mode whatever mode the checkpoint gives. The Python passes
//     `mode=self.wo_a.mode`. Every `attn.*` tensor of the published
//     DeepSeek-V4 checkpoint is affine, thus an omission gives the same
//     numbers on that checkpoint. An attention tensor in another mode
//     would give the wrong numbers. This file passes the mode of the
//     layer.
//
// The compressor of a layer whose compress ratio is more than 0 is
// Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift, and the indexer of a
// layer whose compress ratio is 4 is
// Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift. The block below reads both:
// it pools each block through the compressor into
// Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, asks the indexer which pooled
// chunks each query reads, and attends over the sliding window joined to those
// chunks.
//
// A fourth detail does not come from the file above either. That file gathers
// the top-k pooled rows of each query at decode time and builds a mask only
// for prefill. This file answers ONE selection mask for a block of any length,
// because ``DeepSeekV4Indexer`` already gives a Boolean mask that holds the
// block-causal rule. A masked chunk takes no softmax weight, thus the numbers
// are the numbers a gather gives, and the decode path and the prefill path
// stay one path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Rotary position

/// The rotary position of one DeepSeek-V4 attention layer.
///
/// DeepSeek-V4 gives each layer its own rope theta. A layer with no
/// compressor turns its positions with `rope_theta` and takes no YaRN
/// scaling. A layer with a compressor turns them with `compress_rope_theta`
/// and takes the YaRN scaling of the checkpoint, which widens the context.
///
/// The layer hands out a cosine table and a sine table rather than a rotated
/// tensor, because DeepSeek-V4 reads the same tables three times in one
/// block: forward on the queries, forward on the keys, and backward on the
/// attention output.
final class DeepSeekV4RoPE: Module {

    /// The inverse frequency of each rotary pair, in float32.
    ///
    /// The name starts with an underscore on purpose. mlx-swift collects
    /// every stored `MLXArray` of a `Module` as a parameter unless its name
    /// starts with an underscore, and this table comes from the
    /// configuration rather than from the checkpoint. Without the
    /// underscore a weight-load check would demand a `rope.invFreq` tensor
    /// that no DeepSeek-V4 checkpoint holds.
    private let _inverseFrequency: MLXArray

    /// A YaRN scaling that leaves every frequency alone.
    private static let noScaling: Float = 1

    /// The `beta_fast` of the DeepSeek-V4 reference, read when the
    /// `rope_scaling` block gives no value of its own.
    private static let defaultBetaFast: Float = 32

    /// The `beta_slow` of the DeepSeek-V4 reference, read when the
    /// `rope_scaling` block gives no value of its own.
    private static let defaultBetaSlow: Float = 1

    /// The names the `rope_scaling` block gives a YaRN scaling.
    private static let yarnTypeNames: Set<String> = ["yarn", "deepseek_yarn"]

    /// Builds the rotary table of one layer.
    ///
    /// - Parameters:
    ///   - dim: The number of trailing head dimensions that take a position.
    ///   - base: The rope theta of this layer.
    ///   - factor: The YaRN scaling. A factor of 1 is no scaling.
    ///   - originalMaxPositionEmbeddings: The context the checkpoint trained
    ///     on, before YaRN widened it.
    ///   - betaFast: The turn count that sets the low end of the YaRN ramp.
    ///   - betaSlow: The turn count that sets the high end of the YaRN ramp.
    init(
        dim: Int,
        base: Float,
        factor: Float,
        originalMaxPositionEmbeddings: Int,
        betaFast: Float,
        betaSlow: Float
    ) {
        self._inverseFrequency = DeepSeekV4Math.yarnInvFreq(
            dim: dim,
            base: base,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            factor: factor,
            betaFast: betaFast,
            betaSlow: betaSlow)
    }

    /// Builds the rotary table one layer of a checkpoint asks for.
    ///
    /// A layer with a compressor takes `compress_rope_theta` and the YaRN
    /// scaling of the `rope_scaling` block. Every other layer takes
    /// `rope_theta` and no scaling at all.
    ///
    /// A `rope_scaling` block that names no YaRN type, or that leaves out
    /// `factor` or `original_max_position_embeddings`, gives no scaling.
    /// The Python reference demands both keys and stops without them; this
    /// port keeps the plain table instead of inventing a value.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer.
    convenience init(configuration: DeepSeekV4Configuration, layer: Int) {
        let scaling =
            configuration.hasCompressor(layer: layer)
            ? Self.yarnScaling(from: configuration.ropeScaling) : nil
        self.init(
            dim: configuration.qkRopeHeadDim,
            base: configuration.ropeTheta(forLayer: layer),
            factor: scaling?.factor ?? Self.noScaling,
            originalMaxPositionEmbeddings: scaling?.originalMaxPositionEmbeddings ?? 0,
            betaFast: scaling?.betaFast ?? Self.defaultBetaFast,
            betaSlow: scaling?.betaSlow ?? Self.defaultBetaSlow)
    }

    /// The four numbers a YaRN scaling needs.
    private struct YarnScaling {
        /// The factor the scaling widens the context by.
        let factor: Float

        /// The context the checkpoint trained on, before YaRN widened it.
        let originalMaxPositionEmbeddings: Int

        /// The turn count that sets the low end of the YaRN ramp.
        let betaFast: Float

        /// The turn count that sets the high end of the YaRN ramp.
        let betaSlow: Float
    }

    /// Reads a YaRN scaling out of a `rope_scaling` block, or gives `nil`
    /// when the block names no YaRN type or leaves a needed key out.
    private static func yarnScaling(from block: [String: StringOrNumber]?) -> YarnScaling? {
        guard let block else { return nil }
        let name = block["type"] ?? block["rope_type"]
        guard case .string(let typeName) = name, yarnTypeNames.contains(typeName) else {
            return nil
        }
        guard let factor = block["factor"]?.asFloat(),
            let originalMax = block["original_max_position_embeddings"]?.asInt()
        else {
            return nil
        }
        return YarnScaling(
            factor: factor,
            originalMaxPositionEmbeddings: originalMax,
            betaFast: block["beta_fast"]?.asFloat() ?? defaultBetaFast,
            betaSlow: block["beta_slow"]?.asFloat() ?? defaultBetaSlow)
    }

    /// Gives the cosine and the sine of each angle of a run of positions.
    ///
    /// - Parameters:
    ///   - offset: The position of the first token of the run.
    ///   - length: The number of tokens in the run.
    /// - Returns: Two tables, each of shape `(length, dim / 2)`, in float32.
    func cosSin(offset: Int, length: Int) -> (cos: MLXArray, sin: MLXArray) {
        cosSin(positions: MLXArray(Int32(offset) ..< Int32(offset + length)))
    }

    /// Gives the cosine and the sine of each angle of the given positions.
    ///
    /// The positions need not stand in a run. ``DeepSeekV4Compressor`` reads
    /// one position for each pooled chunk, and two neighbouring chunks stand a
    /// whole compress ratio apart.
    ///
    /// - Parameter positions: The positions, shape `(count)`.
    /// - Returns: Two tables, each of shape `(count, dim / 2)`, in float32.
    func cosSin(positions: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let angles =
            positions.asType(.float32).expandedDimensions(axis: -1)
            * _inverseFrequency.expandedDimensions(axis: 0)
        return (cos: MLX.cos(angles), sin: MLX.sin(angles))
    }
}

// MARK: - Attention

/// One DeepSeek-V4 attention layer.
///
/// The layer differs from every earlier DeepSeek family on four counts:
///
/// - **A partial rotary position.** Only the last `qk_rope_head_dim`
///   dimensions of a head take a position. The dimensions before them carry
///   no position at all.
/// - **A learned sink.** Each head holds one logit that joins the softmax
///   ahead of the keys and leaves again after it, thus a head can hold back
///   from every key it sees.
/// - **An inverse rotation on the output.** The block turns the rotary
///   dimensions of the attention output backward, which takes the position
///   back out before the residual stream reads the result.
/// - **A grouped low-rank output projection.** The heads split into
///   `o_groups` groups, each group reads its own low-rank matrix, and one
///   wide projection then returns the residual width.
///
/// One latent key/value head serves all query heads, thus the keys and the
/// values are one and the same tensor.
final class DeepSeekV4Attention: Module {

    /// The number of query heads.
    let headCount: Int

    /// The width of one head.
    let headDim: Int

    /// The number of trailing head dimensions that take a position.
    let ropeDim: Int

    /// The number of head groups the output projection reads.
    let outputGroups: Int

    /// The rank of the low-rank output projection of one group.
    let outputLoraRank: Int

    /// The epsilon of every norm in this layer.
    let normEps: Float

    /// True when the softmax reads the learned sink of each head.
    let useAttnSink: Bool

    /// The number of most recent keys a query reads directly.
    ///
    /// Every layer attends inside this window. A layer that also holds a
    /// compressor reads the rest of the context through the pooled chunks
    /// beside the window; a layer whose compress ratio is 0 reads the window
    /// and nothing else.
    let slidingWindow: Int

    /// The factor the attention scores take before the softmax.
    let scale: Float

    /// The rotary position of this layer.
    let rope: DeepSeekV4RoPE

    /// The first half of the low-rank query projection, which reads the block
    /// input down to `q_lora_rank`.
    @ModuleInfo(key: "wq_a") var wqA: Linear

    /// The second half of the low-rank query projection, which reads that rank
    /// out to every query head.
    @ModuleInfo(key: "wq_b") var wqB: Linear

    /// The projection of the one latent key/value head, which serves every
    /// query head.
    @ModuleInfo(key: "wkv") var wkv: Linear

    /// The grouped low-rank half of the output projection. Each head group
    /// reads its own rows of this matrix and no other, thus the projection is
    /// block diagonal.
    @ModuleInfo(key: "wo_a") var woA: Linear

    /// The wide half of the output projection, which gives the residual width
    /// back.
    @ModuleInfo(key: "wo_b") var woB: Linear

    /// The norm between the two halves of the query projection.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm

    /// The norm of the one latent key/value head.
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    /// One learned logit for each query head, shape `(headCount)`.
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    /// The top-k chunk selector of this layer, or `nil` on a layer that holds
    /// no indexer.
    ///
    /// The selector holds the `attn.indexer.*` tensors of the checkpoint,
    /// which include the `attn.indexer.compressor.*` tensors of the pooled
    /// keys it scores. It answers which pooled chunks each query of a block
    /// reads, and the attention mask of this layer then holds that answer
    /// beside the sliding window.
    @ModuleInfo(key: "indexer") var indexer: DeepSeekV4Indexer?

    /// The key/value compressor of this layer, or `nil` on a layer whose
    /// compress ratio is 0.
    ///
    /// The compressor holds the `attn.compressor.*` tensors of the checkpoint,
    /// and it pools the chunks the global context is read through. The
    /// published DeepSeek-V4-Flash checkpoint names those tensors on the 41
    /// layers 2 to 42 alone, thus a compressor on layer 0 or layer 1 would
    /// fail the weight load rather than sit unused.
    ///
    /// A layer that holds one reads the sparse path: the sliding window joined
    /// to the pooled chunks. A layer that holds none reads the window alone.
    @ModuleInfo(key: "compressor") var compressor: DeepSeekV4Compressor?

    /// DeepSeek-V4 keeps one latent key/value head and sends it to every
    /// query head, thus `wkv` gives one head of `head_dim` numbers.
    private static let latentHeadCount = 1

    /// The axis positions of a `(batch, tokens, heads, width)` tensor.
    ///
    /// The grouped output projection reads the same layout, with one head
    /// group on the head axis and the numbers of one group on the width
    /// axis.
    private enum BatchMajorAxis {
        /// The axis that holds the batch.
        static let batch = 0
        /// The axis that holds the tokens.
        static let token = 1
        /// The axis that holds the heads, or the head groups.
        static let head = 2
        /// The axis that holds the numbers of one head, or of one group.
        static let width = 3
    }

    /// The axis order that swaps the head axis and the token axis of a
    /// `(batch, tokens, heads, width)` tensor. The swap is its own inverse,
    /// thus the same order takes the result back again.
    private static let headMajor = [
        BatchMajorAxis.batch, BatchMajorAxis.head, BatchMajorAxis.token, BatchMajorAxis.width,
    ]

    /// Builds one attention layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this attention belongs to.
    init(configuration: DeepSeekV4Configuration, layer: Int) {
        self.headCount = configuration.numAttentionHeads
        self.headDim = configuration.headDim
        self.ropeDim = configuration.qkRopeHeadDim
        self.outputGroups = configuration.oGroups
        self.outputLoraRank = configuration.oLoraRank
        self.normEps = configuration.rmsNormEps
        self.useAttnSink = configuration.useAttnSink
        self.slidingWindow = configuration.slidingWindow
        self.scale = 1 / sqrt(Float(configuration.headDim))
        self.rope = DeepSeekV4RoPE(configuration: configuration, layer: layer)

        let hidden = configuration.hiddenSize
        let queryLoraRank = configuration.qLoraRank
        let headWidth = configuration.numAttentionHeads * configuration.headDim

        self._wqA.wrappedValue = Linear(hidden, queryLoraRank, bias: false)
        self._wqB.wrappedValue = Linear(queryLoraRank, headWidth, bias: false)
        self._wkv.wrappedValue = Linear(hidden, configuration.headDim, bias: false)
        self._woA.wrappedValue = Linear(
            headWidth / configuration.oGroups,
            configuration.oGroups * configuration.oLoraRank,
            bias: false)
        self._woB.wrappedValue = Linear(
            configuration.oGroups * configuration.oLoraRank, hidden, bias: false)
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: queryLoraRank, eps: configuration.rmsNormEps)
        self._kvNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim, eps: configuration.rmsNormEps)
        self._attnSink.wrappedValue = zeros([configuration.numAttentionHeads])

        if configuration.hasIndexer(layer: layer) {
            self._indexer.wrappedValue = DeepSeekV4Indexer(
                configuration: configuration, layer: layer)
        }
        if configuration.hasCompressor(layer: layer) {
            self._compressor.wrappedValue = DeepSeekV4Compressor(
                configuration: configuration, layer: layer, headDim: configuration.headDim)
        }
    }

    /// Reads one block of tokens.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - mask: The attention mask of this block.
    ///   - cache: The key/value cache of this layer, or `nil` for a run that
    ///     keeps no history.
    /// - Returns: The block output, shape `(batch, tokens, hidden)`.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let offset = cache?.offset ?? 0
        let angles = rope.cosSin(offset: offset, length: length)
        let cosTable = angles.cos.expandedDimensions(axes: [0, 1])
        let sinTable = angles.sin.expandedDimensions(axes: [0, 1])

        let queryResidual = qNorm(wqA(x))
        let queries = rotatedQueries(
            queryResidual, batch: batch, length: length, cos: cosTable, sin: sinTable)
        let keyValues = rotatedKeyValues(
            x, batch: batch, length: length, cos: cosTable, sin: sinTable)

        // The window mask must read the offset the cache stood at BEFORE the
        // keys of this block joined it.
        let windowMask = compressor == nil
            ? mask
            : cache?.makeMask(n: length, windowSize: slidingWindow, returnArray: true)
                ?? makeAttentionMask(
                    n: length, cache: nil, windowSize: slidingWindow, returnArray: true)

        let attended = attentionOutput(
            x, queries: queries, keyValues: keyValues, queryResidual: queryResidual,
            windowMask: windowMask, cache: cache, cos: cosTable, sin: sinTable, offset: offset)

        // The keys carried the position into the scores. Turning the output
        // backward takes it out again, thus the residual stream reads a
        // result with no position in it.
        let straightened = DeepSeekV4Math.applyInversePartialRoPE(
            attended, cos: cosTable, sin: sinTable, ropeDim: ropeDim)
        let flattened =
            straightened
            .transposed(axes: Self.headMajor)
            .reshaped(batch, length, headCount * headDim)
        return woB(groupedOutputProjection(flattened))
    }

    // MARK: - The two attention paths

    /// The attention output of one block.
    ///
    /// A layer whose compress ratio is 0 reads the sliding window alone, which
    /// is the plain path this file always had. A layer that holds a compressor
    /// reads that window AND the pooled chunks of everything before it.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queries: The rotated queries of this block.
    ///   - keyValues: The rotated keys of this block, which are also its
    ///     values.
    ///   - queryResidual: The low-rank query of this block, after its norm.
    ///   - windowMask: The mask of the sliding window alone.
    ///   - cache: The key/value cache of this layer, or `nil`.
    ///   - cos: The cosine of each rotary angle of this block.
    ///   - sin: The sine of each rotary angle of this block.
    ///   - offset: The absolute position of the first token of this block.
    /// - Returns: The attention output, shape
    ///   `(batch, heads, tokens, headDim)`.
    private func attentionOutput(
        _ x: MLXArray,
        queries: MLXArray,
        keyValues: MLXArray,
        queryResidual: MLXArray,
        windowMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        cos: MLXArray,
        sin: MLXArray,
        offset: Int
    ) -> MLXArray {
        let sinks = useAttnSink ? attnSink.asType(queries.dtype) : nil
        guard let compressor else {
            // `attentionWithCacheUpdate` feeds SDPA the very arrays the cache
            // returned. A block that dropped that return and fed its own tensor
            // instead leaks one Metal buffer for each layer and each token --
            // `ml-explore/mlx-lm` issue 1662.
            return attentionWithCacheUpdate(
                queries: queries,
                keys: keyValues,
                values: keyValues,
                cache: cache,
                scale: scale,
                mask: windowMask,
                sinks: sinks)
        }

        let pooled = pooledChunks(x, compressor: compressor, cache: cache, offset: offset)
        let windowKeys = cache.map { $0.update(keys: keyValues, values: keyValues).0 } ?? keyValues
        guard pooled.attention.dim(Self.chunkAxis) > 0 else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: windowKeys, values: windowKeys,
                scale: scale, mask: windowMask, sinks: sinks)
        }

        // The pooled chunks arrive as a second run of keys behind the window,
        // thus the mask reads `[window visibility | chunk visibility]`.
        let batch = x.dim(0)
        let length = x.dim(1)
        let chunkMask = chunkVisibility(
            x, queryResidual: queryResidual, pooled: pooled, cos: cos, sin: sin,
            offset: offset, batch: batch, length: length)
        let joinedMask = concatenated(
            [
                broadcast(
                    windowVisibility(windowMask, length: length, keyCount: windowKeys.dim(2)),
                    to: [batch, 1, length, windowKeys.dim(2)]),
                chunkMask,
            ], axis: -1)
        let keys = concatenated(
            [windowKeys, pooled.attention.expandedDimensions(axis: Self.latentHeadAxis)],
            axis: Self.keyAxis)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: keys,
            scale: scale, mask: .array(joinedMask), sinks: sinks)
    }

    /// The axis a `(batch, chunks, headDim)` pool holds its chunks on.
    private static let chunkAxis = 1

    /// The axis a `(batch, heads, keys, width)` tensor holds its one latent
    /// key/value head on.
    private static let latentHeadAxis = 1

    /// The axis a `(batch, heads, keys, width)` tensor holds its keys on.
    private static let keyAxis = 2

    /// The chunks of each branch of this layer.
    private struct PooledChunks {
        /// The chunks the compressor of the attention pooled, shape
        /// `(batch, chunks, headDim)`.
        let attention: MLXArray

        /// The chunks the compressor inside the indexer pooled, or `nil` on a
        /// layer that holds no indexer.
        let indexer: MLXArray?
    }

    /// Pools one block into every branch of this layer.
    ///
    /// A run that keeps a cache pools through it, thus the chunks of every
    /// earlier block stay and the tokens of a chunk that is not whole yet wait
    /// for the block that ends it. A run that keeps no cache pools the block
    /// it is given, which is the whole context of such a run.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - compressor: The compressor of this layer.
    ///   - cache: The key/value cache of this layer, or `nil`.
    ///   - offset: The absolute position of the first token of this block.
    /// - Returns: The chunks of each branch.
    private func pooledChunks(
        _ x: MLXArray, compressor: DeepSeekV4Compressor, cache: KVCache?, offset: Int
    ) -> PooledChunks {
        guard let cache else {
            return PooledChunks(
                attention: compressor(x, rope: rope, offset: offset),
                indexer: indexer.map { $0.compressor(x, rope: rope, offset: offset) })
        }
        guard let pooledCache = cache as? DeepSeekV4Cache else {
            preconditionFailure(
                "a layer whose compress ratio is more than 0 needs a DeepSeekV4Cache, and it "
                    + "got a \(type(of: cache)). Build the cache with "
                    + "DeepSeekV4Model.newCache(parameters:).")
        }
        var indexerChunks: MLXArray?
        if let branch = pooledCache.indexerChunks, let selector = indexer {
            indexerChunks = branch.pooled(
                x, through: selector.compressor, rope: rope, offset: offset)
        }
        return PooledChunks(
            attention: pooledCache.attentionChunks.pooled(
                x, through: compressor, rope: rope, offset: offset),
            indexer: indexerChunks)
    }

    /// The chunks each query of this block reads.
    ///
    /// A layer that holds an indexer asks it, and the answer already holds the
    /// block-causal rule. A layer that holds no indexer reads every chunk that
    /// stands wholly behind its query.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queryResidual: The low-rank query of this block, after its norm.
    ///   - pooled: The chunks of each branch.
    ///   - cos: The cosine of each rotary angle of this block.
    ///   - sin: The sine of each rotary angle of this block.
    ///   - offset: The absolute position of the first token of this block.
    ///   - batch: The number of sequences in this block.
    ///   - length: The number of tokens in this block.
    /// - Returns: The visibility, shape `(batch, 1, length, chunks)`.
    private func chunkVisibility(
        _ x: MLXArray,
        queryResidual: MLXArray,
        pooled: PooledChunks,
        cos: MLXArray,
        sin: MLXArray,
        offset: Int,
        batch: Int,
        length: Int
    ) -> MLXArray {
        let chunkCount = pooled.attention.dim(Self.chunkAxis)
        if let selector = indexer, let pooledKeys = pooled.indexer {
            return selector(
                x, queryResidual: queryResidual, pooledKeys: pooledKeys,
                cos: cos, sin: sin, offset: offset)
        }
        let visible = DeepSeekV4Math.pooledChunkVisibility(
            queryCount: length, offset: offset, chunkCount: chunkCount,
            chunkWidth: compressorChunkWidth)
        return broadcast(
            visible.expandedDimensions(axis: Self.latentHeadAxis),
            to: [batch, 1, length, chunkCount])
    }

    /// The number of raw positions one pooled chunk of this layer covers.
    ///
    /// A layer that holds no compressor never reads this, thus the fallback of
    /// 1 stands for a layer that pools nothing.
    private var compressorChunkWidth: Int {
        compressor?.chunkWidth ?? 1
    }

    /// The window mask, as a Boolean array of the shape the joined mask needs.
    ///
    /// - Parameters:
    ///   - mask: The mask the sliding window answered.
    ///   - length: The number of tokens in this block.
    ///   - keyCount: The number of keys the window holds.
    /// - Returns: The visibility, shape `(1, 1, length, keyCount)`.
    private func windowVisibility(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode, length: Int, keyCount: Int
    ) -> MLXArray {
        switch mask {
        case .array(let visible):
            precondition(
                visible.ndim == Self.windowMaskAxisCount,
                "a window mask must hold one axis for the queries and one for the keys, and "
                    + "this one holds \(visible.ndim)")
            return visible.expandedDimensions(axes: [0, 1])
        case .none:
            // A block of one token reads every key the window holds, thus the
            // window answers no mask at all.
            return MLXArray.ones([1, 1, length, keyCount], dtype: .bool)
        default:
            preconditionFailure(
                "a window mask taken with returnArray true must be an array or none, and this "
                    + "one is \(mask)")
        }
    }

    /// The number of axes a window mask array holds: one for the queries and
    /// one for the keys.
    private static let windowMaskAxisCount = 2

    /// Projects the queries, norms each head, and turns the rotary
    /// dimensions forward.
    ///
    /// The head norm runs in float32 whatever the activation dtype is. It
    /// divides by a mean of squares over 512 numbers, and float16 loses that
    /// sum.
    private func rotatedQueries(
        _ queryResidual: MLXArray, batch: Int, length: Int, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let projected = wqB(queryResidual).reshaped(batch, length, headCount, headDim)
        let wide = projected.asType(.float32)
        let normed = wide * rsqrt((wide * wide).mean(axis: -1, keepDims: true) + normEps)
        return DeepSeekV4Math.applyPartialRoPE(
            normed.asType(projected.dtype).transposed(axes: Self.headMajor),
            cos: cos, sin: sin, ropeDim: ropeDim)
    }

    /// Projects the one latent key/value head and turns its rotary
    /// dimensions forward.
    private func rotatedKeyValues(
        _ x: MLXArray, batch: Int, length: Int, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let projected = kvNorm(wkv(x))
            .reshaped(batch, length, Self.latentHeadCount, headDim)
            .transposed(axes: Self.headMajor)
        return DeepSeekV4Math.applyPartialRoPE(projected, cos: cos, sin: sin, ropeDim: ropeDim)
    }

    /// Reads the grouped low-rank half of the output projection.
    ///
    /// The heads split into ``outputGroups`` groups of equal width. Group
    /// `g` reads rows `g * oLoraRank` through `(g + 1) * oLoraRank` of
    /// `wo_a` and nothing else, thus the projection is block diagonal and a
    /// plain `wo_a(x)` would give a different answer. `wo_a` carries no
    /// bias, thus this half adds none.
    ///
    /// - Parameter output: The attention output, shape
    ///   `(batch, tokens, headCount * headDim)`.
    /// - Returns: The low-rank result, shape
    ///   `(batch, tokens, outputGroups * outputLoraRank)`.
    func groupedOutputProjection(_ output: MLXArray) -> MLXArray {
        let batch = output.dim(0)
        let length = output.dim(1)
        let features = (headCount * headDim) / outputGroups
        let grouped = output.reshaped(batch, length, outputGroups, features)

        if let quantized = woA as? QuantizedLinear {
            return quantizedGroupedProjection(grouped, weight: quantized)
                .reshaped(batch, length, outputGroups * outputLoraRank)
        }
        return einsum(
            "bsgd,grd->bsgr", grouped,
            woA.weight.reshaped(outputGroups, outputLoraRank, features)
        ).reshaped(batch, length, outputGroups * outputLoraRank)
    }

    /// The axis positions of a `(groups, batch, tokens, width)` tensor, the
    /// layout one batched quantized matrix multiply reads.
    private enum GroupMajorAxis {
        /// The axis that holds the head groups.
        static let group = 0
        /// The axis that holds the batch.
        static let batch = 1
        /// The axis that holds the tokens.
        static let token = 2
        /// The axis that holds the numbers of one group.
        static let width = 3
    }

    /// The axis order that puts the group axis of a
    /// `(batch, tokens, groups, features)` tensor first, so that one batched
    /// quantized matrix multiply reads every group at once.
    private static let groupMajor = [
        BatchMajorAxis.head, BatchMajorAxis.batch, BatchMajorAxis.token, BatchMajorAxis.width,
    ]

    /// The axis order that takes a `(groups, batch, tokens, rank)` result
    /// back to `(batch, tokens, groups, rank)`.
    private static let groupMinor = [
        GroupMajorAxis.batch, GroupMajorAxis.token, GroupMajorAxis.group, GroupMajorAxis.width,
    ]

    /// Runs the grouped projection against a packed `wo_a`.
    ///
    /// A packed weight cannot be reshaped element by element, thus the group
    /// axis moves to the front and one batched `quantizedMatmul` reads every
    /// group. The last axis of each slab keeps its packed width, which `-1`
    /// works out.
    private func quantizedGroupedProjection(
        _ grouped: MLXArray, weight quantized: QuantizedLinear
    ) -> MLXArray {
        let groupFirst = grouped.transposed(axes: Self.groupMajor)
        let packed = quantized.weight
            .reshaped(outputGroups, outputLoraRank, -1)
            .expandedDimensions(axis: 1)
        let scales = quantized.scales
            .reshaped(outputGroups, outputLoraRank, -1)
            .expandedDimensions(axis: 1)
        let biases = quantized.biases?
            .reshaped(outputGroups, outputLoraRank, -1)
            .expandedDimensions(axis: 1)
        let projected = quantizedMM(
            groupFirst, packed,
            scales: scales, biases: biases,
            transpose: true,
            groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        return projected.transposed(axes: Self.groupMinor)
    }
}
