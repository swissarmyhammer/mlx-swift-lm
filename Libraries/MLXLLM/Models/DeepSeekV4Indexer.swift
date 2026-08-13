// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4Compressor.swift @ 4546a5d720e7013adffdbddd728c6106e4f9e637
// Manual transcription; no git ancestry.
//
// The `Indexer` half of DeepSeek-V4 compressed sparse attention. The
// `Compressor` half, which pools the keys this file scores against, is task
// `^tty95f4`.
//
// Four details do not come from that file.
//
//  1. **A block of one token takes the causal mask too.** The mask of the
//     file above sits in `DeepseekV4Math.causalMaskedIndexerScores`, and that
//     function opens with `guard S > 1 ... else { return scores }`. A decode
//     step carries one token, thus every decode step of that file ranks
//     unmasked scores. At absolute position 5 with a chunk width of 4 and 3
//     pooled chunks, chunk 1 and chunk 2 hold positions 4 through 11, which
//     the query has not read yet, and the ranking may pick them. This file
//     masks each block, of any length.
//  2. **The answer is a selection mask, not a list of chunk indices.** The
//     file above answers `argPartition` output directly. When fewer chunks
//     stand behind a query than the budget holds, that output fills the rest
//     of the row with chunks the query may not read, and a later step must
//     take them back. This file answers a Boolean mask that already holds the
//     causal rule, thus a row carries exactly the chunks its query may read
//     and never more.
//  3. **The pooled keys arrive as an argument.** The file above holds a
//     private compressor and calls it. This repository lands the compressor
//     with task `^tty95f4`, thus the keys arrive from the caller and this
//     file holds no state.
//  4. **No activation quantization-aware round trip.** The file above reads
//     an `activationQATEnabled` flag of its own configuration. The published
//     DeepSeek-V4-Flash `config.json` names no such key, thus this port
//     leaves the round trip out.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The top-k chunk selector of one DeepSeek-V4 attention layer.
///
/// DeepSeek-V4 reads its global context through a compressor, which pools each
/// run of `compress_ratio` tokens into one chunk. A layer whose compress ratio
/// is ``DeepSeekV4Configuration/indexerCompressRatio`` holds this selector
/// beside that compressor, and the selector answers which chunks each query
/// reads.
///
/// The selection is block causal. Chunk `c` covers the raw positions
/// `c * chunkWidth` through `(c + 1) * chunkWidth - 1`, thus a query at
/// absolute position `q` may read that chunk only when the whole chunk stands
/// behind it. Each query keeps the ``topK`` visible chunks of the highest
/// score, or every visible chunk when fewer than ``topK`` stand behind it.
class DeepSeekV4Indexer: Module {

    /// The number of indexer heads, from `index_n_heads`.
    let headCount: Int

    /// The width of one indexer head, from `index_head_dim`.
    let headDim: Int

    /// The number of pooled chunks one query keeps, from `index_topk`.
    let topK: Int

    /// The number of trailing head dimensions that take a position.
    let ropeDim: Int

    /// The number of raw positions one pooled chunk covers, which is the
    /// compress ratio of this layer.
    let chunkWidth: Int

    /// The factor the chunk scores take, which holds the scores at one size
    /// whatever the head width is.
    let scale: Float

    /// The factor each head weight takes, which holds the sum over the heads
    /// at one size whatever the head count is.
    let headWeightScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    /// The axis a `(batch, heads, tokens, width)` tensor holds its heads on.
    ///
    /// The selection mask keeps the same axis with one entry, because every
    /// attention head of the layer reads the one selection.
    private static let headAxis = 1

    /// The axis a `(batch, tokens, heads, width)` tensor holds its heads on.
    ///
    /// A swap of this axis with ``headAxis`` puts the heads ahead of the
    /// tokens, which is the layout the score matmul reads.
    private static let batchMajorHeadAxis = 2

    /// The axis a stack of matrices holds its rows on. A swap of it with the
    /// last axis turns each matrix of the stack around.
    private static let matrixRowAxis = -2

    /// The axis the ranked chunk indices of one query sit on, after the
    /// comparison against every chunk index adds an axis at the end.
    private static let rankedChunkAxis = -2

    /// The score a chunk that its query may not read takes.
    ///
    /// Every real score stands above it, thus the ranking picks a masked chunk
    /// only after it has picked every visible one.
    private static let maskedScore: Float = -1e30

    /// Builds the selector of one layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this selector belongs to.
    init(configuration: DeepSeekV4Configuration, layer: Int) {
        precondition(
            configuration.hasIndexer(layer: layer),
            "layer \(layer) has no indexer, thus its compress ratio is not "
                + "\(DeepSeekV4Configuration.indexerCompressRatio)")
        precondition(
            configuration.indexTopK > 0,
            "index_topk must be more than 0, and it is \(configuration.indexTopK)")

        self.headCount = configuration.indexNHeads
        self.headDim = configuration.indexHeadDim
        self.topK = configuration.indexTopK
        self.ropeDim = configuration.qkRopeHeadDim
        self.chunkWidth = configuration.compressRatio(ofLayer: layer)
        self.scale = 1 / sqrt(Float(configuration.indexHeadDim))
        self.headWeightScale = 1 / sqrt(Float(configuration.indexNHeads))

        self._wqB.wrappedValue = Linear(
            configuration.qLoraRank,
            configuration.indexNHeads * configuration.indexHeadDim,
            bias: false)
        self._weightsProj.wrappedValue = Linear(
            configuration.hiddenSize, configuration.indexNHeads, bias: false)
    }

    /// Picks the pooled chunks each query of one block reads.
    ///
    /// The answer holds exactly the smaller of ``topK`` and the number of
    /// chunks that stand wholly behind the query, and it never holds a chunk
    /// that reaches past the query.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queryResidual: The low-rank query of this layer, after its norm,
    ///     shape `(batch, tokens, qLoraRank)`.
    ///   - pooledKeys: The chunks the compressor pooled, shape
    ///     `(batch, chunks, headDim)`.
    ///   - cos: The cosine of each rotary angle of this block.
    ///   - sin: The sine of each rotary angle of this block.
    ///   - offset: The absolute position of the first token of this block.
    /// - Returns: The selection, shape `(batch, 1, tokens, chunks)`, where a
    ///   true entry names a chunk its query reads.
    func callAsFunction(
        _ x: MLXArray,
        queryResidual: MLXArray,
        pooledKeys: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        offset: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let chunkCount = pooledKeys.dim(1)
        let visible = chunkVisibility(
            queryCount: length, offset: offset, chunkCount: chunkCount)

        // Every chunk fits inside the budget, thus a ranking would keep them
        // all. The score path answers this very mask and costs one matmul for
        // each head.
        guard chunkCount > topK else {
            return broadcast(
                visible.expandedDimensions(axis: Self.headAxis),
                to: [batch, 1, length, chunkCount])
        }

        let scores = chunkScores(
            x, queryResidual: queryResidual, pooledKeys: pooledKeys, cos: cos, sin: sin)
        let masked = MLX.where(visible, scores, MLXArray(Self.maskedScore))
        let ranked = argPartition(-masked, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
        let chunkIndices = MLXArray(Int32(0) ..< Int32(chunkCount))
        let picked =
            (ranked.asType(.int32).expandedDimensions(axis: -1) .== chunkIndices)
            .any(axis: Self.rankedChunkAxis)

        // The ranking spends its whole budget even when fewer chunks stand
        // behind the query, thus the visibility decides again here.
        return (picked .&& visible).expandedDimensions(axis: Self.headAxis)
    }

    /// The block-causal visibility of each pooled chunk.
    ///
    /// Chunk `c` covers the raw positions `c * chunkWidth` through
    /// `(c + 1) * chunkWidth - 1`. A query at absolute position `q` reads that
    /// chunk when the whole chunk stands behind it, thus when
    /// `(c + 1) * chunkWidth <= q + 1`.
    ///
    /// - Parameters:
    ///   - queryCount: The number of tokens in this block.
    ///   - offset: The absolute position of the first token of this block.
    ///   - chunkCount: The number of chunks the compressor pooled.
    /// - Returns: The visibility, shape `(1, queryCount, chunkCount)`.
    private func chunkVisibility(queryCount: Int, offset: Int, chunkCount: Int) -> MLXArray {
        let positions = MLXArray(Int32(offset) ..< Int32(offset + queryCount))
            .reshaped(1, queryCount, 1)
        let chunkEnds =
            (MLXArray(Int32(0) ..< Int32(chunkCount)) + 1) * Int32(chunkWidth)
        return chunkEnds.reshaped(1, 1, chunkCount) .<= (positions + 1)
    }

    /// Scores each pooled chunk against each query of one block.
    ///
    /// Each head scores the chunks on its own, the negative scores fall to
    /// zero, and one learned weight for each head then adds the heads into the
    /// one score the ranking reads. The arithmetic runs in float32, because a
    /// sum over 64 heads loses too much in bfloat16.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queryResidual: The low-rank query of this layer, after its norm.
    ///   - pooledKeys: The chunks the compressor pooled.
    ///   - cos: The cosine of each rotary angle of this block.
    ///   - sin: The sine of each rotary angle of this block.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`, in float32.
    private func chunkScores(
        _ x: MLXArray,
        queryResidual: MLXArray,
        pooledKeys: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> MLXArray {
        let projected = wqB(queryResidual)
            .reshaped(x.dim(0), x.dim(1), headCount, headDim)
            .swappedAxes(Self.headAxis, Self.batchMajorHeadAxis)
        let queries = DeepSeekV4Math.applyPartialRoPE(
            projected, cos: cos, sin: sin, ropeDim: ropeDim
        ).asType(.float32)
        let keys = pooledKeys.asType(.float32)
            .expandedDimensions(axis: Self.headAxis)
            .swappedAxes(-1, Self.matrixRowAxis)
        let perHead = maximum(queries.matmul(keys), 0) * scale
        let headWeights = (weightsProj(x).asType(.float32) * headWeightScale)
            .swappedAxes(-1, Self.matrixRowAxis)
            .expandedDimensions(axis: -1)
        return (perHead * headWeights).sum(axis: Self.headAxis)
    }
}
