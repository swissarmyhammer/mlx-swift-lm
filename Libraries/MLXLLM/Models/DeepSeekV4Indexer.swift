// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4Compressor.swift @ 4546a5d720e7013adffdbddd728c6106e4f9e637
// Manual transcription; no git ancestry.
//
// The `Indexer` half of DeepSeek-V4 compressed sparse attention. The
// `Compressor` half, which pools the keys this file scores against, is
// Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift.
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
//  2. **A chunk the query may not read ranks last.** The file above answers
//     `argPartition` output of the raw scores. This file gives each chunk
//     that reaches past its query the lowest score first, thus the ranking
//     spends its budget on visible chunks before any other. When fewer chunks
//     stand behind a query than the budget holds, the rest of the row still
//     names chunks the query may not read, and the attention hides them with
//     `DeepSeekV4Math.pickedChunkVisibility`.
//  3. **The pooled keys arrive as an argument.** The file above calls its own
//     compressor inside the score path, which ties the score path to the
//     pooled cache that compressor reads. This file holds the compressor, so
//     that the `attn.indexer.compressor.*` tensors of the checkpoint load,
//     and takes the pooled keys as an argument, so that the score path holds
//     no state.
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
final class DeepSeekV4Indexer: Module {

    /// The number of indexer heads, from `index_n_heads`.
    let headCount: Int

    /// The width of one indexer head, from `index_head_dim`.
    let headDim: Int

    /// The number of pooled chunks one query keeps, from `index_topk`.
    let topK: Int

    /// The number of raw positions one pooled chunk covers, which is the
    /// compress ratio of this layer.
    let chunkWidth: Int

    /// The factor the chunk scores take, which holds the scores at one size
    /// whatever the head width is.
    let scale: Float

    /// The factor each head weight takes, which holds the sum over the heads
    /// at one size whatever the head count is.
    let headWeightScale: Float

    /// The projection that opens the low-rank query into the indexer queries,
    /// from `wq_b`.
    ///
    /// It reads the `qLoraRank` query residual of the layer and answers
    /// `headCount * headDim` values, which the score path then cuts into one
    /// query for each indexer head. These queries score the pooled chunks.
    @ModuleInfo(key: "wq_b") var wqB: Linear

    /// The projection that gives each indexer head its weight, from
    /// `weights_proj`.
    ///
    /// It reads the block input and answers one value for each of the
    /// `headCount` heads. The score path multiplies the chunk scores of each
    /// head by that value before it adds the heads into the one score the
    /// ranking reads.
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    /// The compressor that pools the keys this selector scores, from
    /// `compressor`.
    ///
    /// It is a compressor of its own, beside the compressor of the attention
    /// layer: this one pools to ``DeepSeekV4Configuration/indexHeadDim`` and
    /// the other pools to `head_dim`. Both pool the same tokens at the same
    /// compress ratio, thus the two answer the same number of chunks and a
    /// chunk this selector picks names a chunk of the attention pool.
    ///
    /// The caller pools with this module and hands the answer to
    /// ``rankedChunks(_:queryResidual:pooledKeys:rope:offset:scoreByteLimit:)``.
    @ModuleInfo(key: "compressor") var compressor: DeepSeekV4Compressor

    /// The axis a `(batch, heads, tokens, width)` tensor holds its heads on.
    private static let headAxis = 1

    /// The axis a `(batch, tokens, heads, width)` tensor holds its heads on.
    ///
    /// A swap of this axis with ``headAxis`` puts the heads ahead of the
    /// tokens, which is the layout the score matmul reads.
    private static let batchMajorHeadAxis = 2

    /// The axis a stack of matrices holds its rows on. A swap of it with the
    /// last axis turns each matrix of the stack around.
    private static let matrixRowAxis = -2

    /// The dtype the chunk scores use.
    ///
    /// The arithmetic runs in float32, because a sum over 64 heads loses too
    /// much in bfloat16. The indexer branch of ``DeepSeekV4Cache`` keeps its
    /// pool in this dtype, thus a decode step reads the pool with no cast.
    static let scoreDType = DType.float32

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
        self.chunkWidth = configuration.compressRatio(ofLayer: layer)
        self.scale = 1 / sqrt(Float(configuration.indexHeadDim))
        self.headWeightScale = 1 / sqrt(Float(configuration.indexNHeads))

        self._wqB.wrappedValue = Linear(
            configuration.qLoraRank,
            configuration.indexNHeads * configuration.indexHeadDim,
            bias: false)
        self._weightsProj.wrappedValue = Linear(
            configuration.hiddenSize, configuration.indexNHeads, bias: false)
        self._compressor.wrappedValue = DeepSeekV4Compressor(
            configuration: configuration, layer: layer,
            headDim: configuration.indexHeadDim)
    }

    // MARK: - The ranking

    /// The most bytes the per-head scores of one tile of query rows hold.
    ///
    /// At the real checkpoint a 32k context pools 8192 chunks, and one query
    /// row of one sequence scores them with 64 heads in float32, which is
    /// 2 MiB. Thus 1 GiB holds a tile of 512 rows, a quarter of one prefill
    /// step of 2048 tokens.
    static let oneTileScoreBytes = 1 << 30

    /// The fewest query rows a tile holds, so that each tile makes progress.
    private static let fewestRowsPerTile = 1

    /// The bytes of one float32 score.
    private static let scoreBytes = MemoryLayout<Float>.size

    /// The axis a `(batch, tokens, ...)` tensor holds its tokens on.
    private static let tokenAxis = 1

    /// The axis a `(batch, chunks, headDim)` pool holds its chunks on.
    private static let chunkAxis = 1

    /// The axis a `(batch, tokens, chunks)` score tensor holds its chunks on.
    private static let scoreChunkAxis = -1

    /// The most query rows one tile of the ranking scores.
    ///
    /// The per-head scores of a tile stay inside `scoreByteLimit`. When one
    /// row alone breaks the limit, the answer is one row, thus each tile
    /// still scores a row.
    ///
    /// - Parameters:
    ///   - batch: The number of sequences.
    ///   - heads: The number of indexer heads.
    ///   - chunks: The number of pooled chunks each query row scores.
    ///   - scoreByteLimit: The most bytes the per-head scores of one tile
    ///     hold.
    /// - Returns: The number of query rows for each tile, 1 or more.
    static func queryRowsPerScoreTile(
        batch: Int, heads: Int, chunks: Int, scoreByteLimit: Int
    ) -> Int {
        max(fewestRowsPerTile, scoreByteLimit / (batch * heads * chunks * scoreBytes))
    }

    /// The indices of the ``topK`` chunks of the highest score for each query
    /// of one block.
    ///
    /// The ranking reads the block-causal rule: a chunk that reaches past its
    /// query ranks after every chunk the query may read. The answer is the
    /// whole budget for each query, thus a query that has fewer than
    /// ``topK`` chunks behind it also gets indices of chunks it may not read.
    /// A caller that reads the indices of such a query must mask them.
    ///
    /// A decode step carries one token, and every chunk of its pool stands
    /// wholly behind that token. This function then builds no visibility at
    /// all, and a caller needs no mask.
    ///
    /// The queries go in tiles of
    /// ``queryRowsPerScoreTile(batch:heads:chunks:scoreByteLimit:)`` rows, so
    /// that the per-head scores of one tile stay inside `scoreByteLimit`.
    /// Each query row ranks on its own, thus the tiles give the answer of one
    /// whole block.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queryResidual: The low-rank query of this layer, after its norm,
    ///     shape `(batch, tokens, qLoraRank)`.
    ///   - pooledKeys: The chunks the compressor pooled, shape
    ///     `(batch, chunks, headDim)`. It must hold more than ``topK`` chunks.
    ///   - rope: The rotary position of this layer, which turns the queries.
    ///   - offset: The absolute position of the first token of this block.
    ///   - scoreByteLimit: The most bytes the per-head scores of one tile
    ///     hold.
    /// - Returns: The chunk indices, shape `(batch, tokens, topK)`, in
    ///   ascending index order for each query.
    func rankedChunks(
        _ x: MLXArray,
        queryResidual: MLXArray,
        pooledKeys: MLXArray,
        rope: DeepSeekV4RoPE,
        offset: Int,
        scoreByteLimit: Int = oneTileScoreBytes
    ) -> MLXArray {
        let chunkCount = pooledKeys.dim(Self.chunkAxis)
        precondition(
            chunkCount > topK,
            "a ranking needs more than \(topK) chunks, and the pool holds \(chunkCount)")
        let queryCount = x.dim(Self.tokenAxis)
        let lastChunkEnd = chunkCount * chunkWidth - 1
        let visible =
            lastChunkEnd <= offset
            ? nil
            : chunkVisibility(queryCount: queryCount, offset: offset, chunkCount: chunkCount)
        let operands = scoreOperands(
            x, queryResidual: queryResidual, pooledKeys: pooledKeys, rope: rope, offset: offset)
        let rowsPerTile = Self.queryRowsPerScoreTile(
            batch: x.dim(0), heads: headCount, chunks: chunkCount, scoreByteLimit: scoreByteLimit)
        return DeepSeekV4Math.tiledRows(
            count: queryCount, rowsPerTile: rowsPerTile, axis: Self.tokenAxis,
            whole: { topChunks(of: chunkScores(operands), visible: visible) },
            tile: { rows in
                topChunks(of: chunkScores(operands.tile(rows)), visible: visible?[0..., rows])
            })
    }

    /// The indices of the ``topK`` chunks of the highest score.
    ///
    /// On the GPU, ``DeepSeekV4TopKKernel`` picks the chunks. MLX
    /// `argPartition` is a full sort of each row, and the kernel takes a few
    /// passes over the row instead. On another device, `argPartition` picks
    /// the chunks, and a sort gives them the order of the kernel.
    ///
    /// - Parameters:
    ///   - scores: The chunk scores, shape `(batch, tokens, chunks)`.
    ///   - visible: The block-causal visibility of each chunk, or `nil` when
    ///     every query may read every chunk.
    /// - Returns: The chunk indices, shape `(batch, tokens, topK)`, in
    ///   ascending index order for each query.
    private func topChunks(of scores: MLXArray, visible: MLXArray?) -> MLXArray {
        let ranked = visible.map { MLX.where($0, scores, MLXArray(Self.maskedScore)) } ?? scores
        guard DeepSeekV4TopKKernel.canRun else {
            let picks = argPartition(-ranked, kth: topK - 1, axis: Self.scoreChunkAxis)
            return sorted(picks[.ellipsis, 0 ..< topK], axis: Self.scoreChunkAxis)
        }
        return DeepSeekV4TopKKernel.topIndices(ranked, count: topK)
    }

    /// The block-causal visibility of each pooled chunk.
    ///
    /// The rule is ``DeepSeekV4Math/pooledChunkVisibility(queryCount:offset:chunkCount:chunkWidth:)``,
    /// which the attention mask of a compressed layer reads as well.
    ///
    /// - Parameters:
    ///   - queryCount: The number of tokens in this block.
    ///   - offset: The absolute position of the first token of this block.
    ///   - chunkCount: The number of chunks the compressor pooled.
    /// - Returns: The visibility, shape `(1, queryCount, chunkCount)`.
    private func chunkVisibility(queryCount: Int, offset: Int, chunkCount: Int) -> MLXArray {
        DeepSeekV4Math.pooledChunkVisibility(
            queryCount: queryCount, offset: offset, chunkCount: chunkCount,
            chunkWidth: chunkWidth)
    }

    // MARK: - The chunk scores

    /// The ``scoreDType`` operands of the chunk scores of one block, or of
    /// one tile of its query rows.
    private struct ScoreOperands {
        /// The rotated queries, shape `(batch, heads, tokens, headDim)`.
        let queries: MLXArray

        /// The pooled keys, turned for the score matmul, shape
        /// `(batch, 1, headDim, chunks)`.
        ///
        /// Every tile of a block reads this one array, thus a cast of a pool
        /// in another dtype runs one time for each block and not one time
        /// for each tile.
        let keys: MLXArray

        /// The weight of each head for each query, shape
        /// `(batch, heads, tokens, 1)`.
        let headWeights: MLXArray

        /// The factor each per-head score takes, a float32 scalar.
        let scale: MLXArray

        /// The operands as one tile of query rows reads them.
        ///
        /// - Parameter rows: The query rows of the tile.
        /// - Returns: The queries and the head weights of those rows alone,
        ///   with the same keys and scale.
        func tile(_ rows: Range<Int>) -> ScoreOperands {
            ScoreOperands(
                queries: queries[0..., 0..., rows], keys: keys,
                headWeights: headWeights[0..., 0..., rows], scale: scale)
        }
    }

    /// The operands of the chunk scores of one block, in ``scoreDType``.
    ///
    /// A pool that is already in ``scoreDType``, as the pool of the cache is,
    /// takes no cast: MLX gives back the same array for a cast to its own
    /// dtype. A pool in another dtype, as the pool of a call with no cache
    /// is, takes one cast for the block.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hidden)`.
    ///   - queryResidual: The low-rank query of this layer, after its norm.
    ///   - pooledKeys: The chunks the compressor pooled.
    ///   - rope: The rotary position of this layer, which turns the queries.
    ///   - offset: The absolute position of the first token of this block.
    /// - Returns: The queries, the keys, the head weights and the scale, in
    ///   ``scoreDType``.
    private func scoreOperands(
        _ x: MLXArray,
        queryResidual: MLXArray,
        pooledKeys: MLXArray,
        rope: DeepSeekV4RoPE,
        offset: Int
    ) -> ScoreOperands {
        let projected = wqB(queryResidual)
            .reshaped(x.dim(0), x.dim(Self.tokenAxis), headCount, headDim)
            .swappedAxes(Self.headAxis, Self.batchMajorHeadAxis)
        let queries = rope(projected, offset: DeepSeekV4RoPE.position(offset))
            .asType(Self.scoreDType)
        let keys = pooledKeys.asType(Self.scoreDType)
            .expandedDimensions(axis: Self.headAxis)
            .swappedAxes(-1, Self.matrixRowAxis)
        let headWeights = (weightsProj(x).asType(Self.scoreDType) * headWeightScale)
            .swappedAxes(-1, Self.matrixRowAxis)
            .expandedDimensions(axis: -1)
        return ScoreOperands(
            queries: queries, keys: keys, headWeights: headWeights, scale: MLXArray(scale))
    }

    /// The score of each pooled chunk against each query the operands hold.
    ///
    /// Each head scores the chunks on its own matmul, and
    /// ``headWeightedSum(_:headWeights:scale:)`` adds the heads into the one
    /// score the ranking reads, in one compiled kernel.
    ///
    /// - Parameter operands: The operands of one block or of one tile.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`, in float32.
    private func chunkScores(_ operands: ScoreOperands) -> MLXArray {
        Self.compiledHeadWeightedSum(
            operands.queries.matmul(operands.keys), operands.headWeights, operands.scale)
    }

    /// Adds the per-head chunk scores into one score for each query and
    /// chunk.
    ///
    /// A negative score falls to zero, each score takes `scale`, and one
    /// learned weight for each head then adds the heads. The order of the
    /// steps is the order of the reference, thus the sum is the same number.
    ///
    /// - Parameters:
    ///   - scores: The per-head scores, shape `(batch, heads, tokens, chunks)`.
    ///   - headWeights: The weight of each head for each query, shape
    ///     `(batch, heads, tokens, 1)`.
    ///   - scale: The factor each score takes, a float32 scalar.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`.
    static func headWeightedSum(
        _ scores: MLXArray, headWeights: MLXArray, scale: MLXArray
    ) -> MLXArray {
        (maximum(scores, 0) * scale * headWeights).sum(axis: headAxis)
    }

    /// The compiled ``headWeightedSum(_:headWeights:scale:)``.
    ///
    /// The function is shapeless, thus one trace serves each block length and
    /// each pool size. Its elementwise steps fuse into one kernel, which reads
    /// the per-head scores one time instead of one time for each step. At a
    /// 32k context those scores hold 4 GiB for one prefill step.
    ///
    /// The scale is an argument and not a captured constant. A captured
    /// scale becomes a constant of the fused kernel, and that kernel then
    /// gives sums that differ from the op path in the last bit.
    ///
    /// The arguments are `(scores, headWeights, scale)`.
    static let compiledHeadWeightedSum: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { scores, headWeights, scale in
            headWeightedSum(scores, headWeights: headWeights, scale: scale)
        }
}
