// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from scouzi1966/mlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4Compressor.swift @ 07e1b806cc7e7291d05e1d3a95e3a04b3139e531
// which gives the Osaurus AI copyright line above. Manual transcription of the
// `Compressor` part alone, lines 860 to 1226; no git ancestry.
//
// The `Indexer` part of that same file is
// Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, task `^r92pjcr`.
//
// Five details do not come from that file.
//
//  1. **The pooling holds no state.** The file above takes a
//     `DeepseekV4Cache` as an argument and writes its pooled chunks into it.
//     This file pools the block it is given and holds nothing, which is the
//     `v4Cache == nil` path of the file above.
//     ``DeepSeekV4ChunkCache`` in Libraries/MLXLLM/Models/DeepSeekV4Cache.swift
//     is what keeps the chunks across calls, and it drives this file rather
//     than living inside it.
//  2. **Any batch size.** `updateProjected` of the file above opens with
//     `precondition(projectedKV.dim(0) == 1 && projectedGate.dim(0) == 1)`,
//     because its pool windows and its offsets belong to one request. Nothing
//     here belongs to a request, thus this file reads any batch.
//  3. **The position of a chunk is its first raw position.** The comment of
//     the file above says "chunk centers", and the line under that comment
//     reads `arange(pooled_count) * ratio + pool_base`, which is the FIRST raw
//     position of each chunk. This file follows the line, not the comment.
//  4. **No activation quantization-aware round trip.** The file above reads an
//     `activationQATEnabled` flag of its own configuration. The published
//     DeepSeek-V4-Flash `config.json` names no such key, thus this port leaves
//     the round trip out. ``DeepSeekV4Indexer`` states the same.
//  5. **The rotary tables come from the layer rope.** The file above reads
//     `rope.invFreq` and builds the tables itself. That property would put
//     `attn.rope.invFreq` into the module parameters and thus into the
//     weight-load check, which is the very trap
//     Libraries/MLXLLM/Models/DeepSeekV4Attention.swift records. This file
//     asks ``DeepSeekV4RoPE/cosSin(positions:)`` for the tables instead.
//
// ``DeepSeekV4Attention`` and ``DeepSeekV4Indexer`` each hold a compressor,
// thus the `attn.compressor.*` and `attn.indexer.compressor.*` tensors of the
// checkpoint load. The attention path pools every block through them and
// attends over the chunks beside its sliding window.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The key/value compressor of one DeepSeek-V4 attention layer.
///
/// DeepSeek-V4 reads a long context through pooled chunks rather than through
/// every key. This module makes those chunks: it pools each run of
/// `compress_ratio` tokens into one chunk with a learned softmax gate, norms
/// the result, and turns it by the rotary position of the chunk.
///
/// A layer of the fine compress ratio pools **with overlap**: chunk `c` reads
/// the tokens of chunk `c - 1` as well as its own, thus one chunk covers twice
/// the tokens and the two projections answer twice the pooled width. A layer
/// of a coarser ratio reads its own tokens alone. Either way chunk `c` ends at
/// raw position `(c + 1) * chunkWidth - 1`, which is the rule
/// ``DeepSeekV4Indexer`` scores against.
final class DeepSeekV4Compressor: Module {

    /// The number of raw positions one pooled chunk advances, which is the
    /// compress ratio of this layer.
    let chunkWidth: Int

    /// The width of one pooled chunk.
    ///
    /// The compressor of an attention layer pools to `head_dim`, and the
    /// compressor inside an indexer pools to `index_head_dim`.
    let headDim: Int

    /// The number of values the two projections answer for each token.
    ///
    /// An overlapping layer answers twice ``headDim``: the leading half feeds
    /// the chunk that follows, and the trailing half feeds the chunk the token
    /// stands in.
    let projectionWidth: Int

    /// True when each pooled chunk also reads the chunk before it.
    let poolsWithOverlap: Bool

    /// The number of trailing head dimensions that take a position.
    let ropeDim: Int

    /// The projection that makes the values one chunk pools, from `wkv`.
    @ModuleInfo(key: "wkv") var wkv: Linear

    /// The projection that makes the gate logits one chunk pools by, from
    /// `wgate`.
    @ModuleInfo(key: "wgate") var wgate: Linear

    /// The learned position bias inside one chunk, shape
    /// `(chunkWidth, projectionWidth)`.
    ///
    /// The gate logits take it before the softmax, thus the pooling can favour
    /// one place of a chunk over another whatever the tokens hold.
    @ParameterInfo(key: "ape") var ape: MLXArray

    /// The norm each pooled chunk takes before it is turned by its position.
    @ModuleInfo(key: "norm") var norm: RMSNorm

    /// The compress ratio that pools with overlap.
    ///
    /// It is the same ratio that carries an indexer, thus
    /// ``DeepSeekV4Configuration/indexerCompressRatio`` names it. The published
    /// DeepSeek-V4-Flash checkpoint states the rule in its tensor shapes: `ape`
    /// is `(4, 2 * head_dim)` on a ratio-4 layer and `(128, head_dim)` on a
    /// ratio-128 layer.
    private static let overlapRatio = DeepSeekV4Configuration.indexerCompressRatio

    /// The factor an overlapping layer widens its projections by, because each
    /// pooled chunk reads two chunks of tokens rather than one.
    private static let overlapWidthFactor = 2

    /// The axis positions of a `(batch, chunks, position, width)` window
    /// tensor, which is the layout the pooling reads.
    private enum WindowAxis {
        /// The axis that holds the batch.
        static let batch = 0
        /// The axis that holds the pooled chunks.
        static let chunk = 1
        /// The axis that holds the raw positions inside one chunk.
        static let position = 2
    }

    /// The gate logit a padded position takes.
    ///
    /// The softmax gives it no weight at all, thus the leading half of the
    /// first chunk of an overlapping layer -- which stands before the first
    /// token -- adds nothing to that chunk.
    private static let paddedGateLogit: Float = -.infinity

    /// The value a padded position takes. The gate gives it no weight, thus
    /// the value only has to be finite.
    private static let paddedValue: Float = 0

    /// Builds the compressor of one layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this compressor belongs to.
    ///   - headDim: The width of one pooled chunk. An attention layer gives
    ///     `head_dim`, and an indexer gives `index_head_dim`.
    init(configuration: DeepSeekV4Configuration, layer: Int, headDim: Int) {
        precondition(
            configuration.hasCompressor(layer: layer),
            "layer \(layer) has no compressor, thus its compress ratio is 0")

        let ratio = configuration.compressRatio(ofLayer: layer)
        let overlaps = ratio == Self.overlapRatio
        let width = headDim * (overlaps ? Self.overlapWidthFactor : 1)

        self.chunkWidth = ratio
        self.headDim = headDim
        self.projectionWidth = width
        self.poolsWithOverlap = overlaps
        self.ropeDim = configuration.qkRopeHeadDim

        self._wkv.wrappedValue = Linear(configuration.hiddenSize, width, bias: false)
        self._wgate.wrappedValue = Linear(configuration.hiddenSize, width, bias: false)
        self._ape.wrappedValue = zeros([ratio, width])
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
    }

    /// The chunks this layer pools out of one block of tokens.
    ///
    /// The block gives one chunk for each whole run of ``chunkWidth`` tokens in
    /// it. The tokens of an incomplete run at the end give no chunk, because a
    /// chunk stands for a run that has ended.
    ///
    /// The arithmetic runs in float32 up to the norm. The gate is a softmax
    /// over as many as `2 * chunkWidth` logits, and bfloat16 loses that sum.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - rope: The rotary position of this layer, which turns each chunk by
    ///     the position that chunk starts at.
    ///   - offset: The absolute position of the first token of the block. The
    ///     caller gives a block that starts a chunk, thus chunk `c` covers the
    ///     positions `offset + c * chunkWidth` onward.
    /// - Returns: The pooled chunks, shape `(batch, chunks, headDim)`, in the
    ///   dtype of `x`.
    func callAsFunction(_ x: MLXArray, rope: DeepSeekV4RoPE, offset: Int) -> MLXArray {
        let batch = x.dim(0)
        let chunkCount = x.dim(1) / chunkWidth
        guard chunkCount > 0 else {
            return zeros([batch, 0, headDim], dtype: x.dtype)
        }

        let windows = pooledWindows(x, batch: batch, chunkCount: chunkCount)
        let weights = softmax(windows.gates, axis: WindowAxis.position, precise: true)
        let pooled = norm((windows.values * weights).sum(axis: WindowAxis.position).asType(x.dtype))

        let positions =
            MLXArray(Int32(0) ..< Int32(chunkCount)) * Int32(chunkWidth) + Int32(offset)
        let angles = rope.cosSin(positions: positions)
        return DeepSeekV4Math.applyPartialRoPE(
            pooled,
            cos: angles.cos.expandedDimensions(axis: WindowAxis.batch),
            sin: angles.sin.expandedDimensions(axis: WindowAxis.batch),
            ropeDim: ropeDim)
    }

    /// The values and the gate logits of each chunk, ready for the softmax.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - batch: The number of sequences in the block.
    ///   - chunkCount: The number of whole chunks the block holds.
    /// - Returns: Two tensors of shape
    ///   `(batch, chunkCount, positions, headDim)`, in float32, where
    ///   `positions` is `chunkWidth` on a layer that does not overlap and
    ///   twice that on a layer that does.
    private func pooledWindows(
        _ x: MLXArray, batch: Int, chunkCount: Int
    ) -> (values: MLXArray, gates: MLXArray) {
        // The DeepSeek-V4 runtime stages the compressor in float32 from the
        // projections onward, thus the cast stands ahead of them rather than
        // after them.
        let whole = x[0..., 0 ..< (chunkCount * chunkWidth), 0...].asType(.float32)
        let shape = [batch, chunkCount, chunkWidth, projectionWidth]
        let values = wkv(whole).reshaped(shape)
        let gates = wgate(whole).reshaped(shape) + ape.asType(.float32)

        guard poolsWithOverlap else { return (values: values, gates: gates) }
        return (
            values: overlapped(values, padding: Self.paddedValue),
            gates: overlapped(gates, padding: Self.paddedGateLogit)
        )
    }

    /// Spreads each chunk of an overlapping layer over two chunks of tokens.
    ///
    /// The trailing ``headDim`` values of chunk `c` stay with chunk `c`, and
    /// the leading ``headDim`` values move on to chunk `c + 1`. Chunk 0 has no
    /// chunk before it, thus it takes `padding` there.
    ///
    /// - Parameters:
    ///   - windows: The projected chunks, shape
    ///     `(batch, chunks, chunkWidth, 2 * headDim)`.
    ///   - padding: The value the places before the first token take.
    /// - Returns: The chunks, shape `(batch, chunks, 2 * chunkWidth, headDim)`.
    private func overlapped(_ windows: MLXArray, padding: Float) -> MLXArray {
        let batch = windows.dim(WindowAxis.batch)
        let chunkCount = windows.dim(WindowAxis.chunk)
        let leading = windows[.ellipsis, 0 ..< headDim]
        let trailing = windows[.ellipsis, headDim...]
        let padded = full([batch, 1, chunkWidth, headDim], values: padding)
            .asType(windows.dtype)
        let handedOn = concatenated(
            [padded, leading[0..., 0 ..< (chunkCount - 1), 0..., 0...]], axis: WindowAxis.chunk)
        return concatenated([handedOn, trailing], axis: WindowAxis.position)
    }
}
