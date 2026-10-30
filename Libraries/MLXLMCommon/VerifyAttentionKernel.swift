// Copyright © 2026 Apple Inc.
//
// Two Metal kernels for the attention of the few query tokens of a
// speculative verify over a long key cache, for grouped-query attention with a
// head width of 256, as Qwen3.8 (24 query heads, 4 KV heads).
//
// The SDPA of MLX reads a verify of 2 to 5 tokens with its vector kernel,
// which gives one SIMD group to each query head and each token, and adds the
// keys one dot product and one SIMD reduction at a time. Its cost grows with
// the tokens times the keys: at a 100k context each token past the first adds
// about 6 ms over the 16 full-attention layers of Qwen3.8, one more read of the
// whole cache. These kernels read each key tile once for every token and
// every query head of one KV head, as 8x8 SIMD-group matrices, thus the cost
// of a verify of 3 to 8 tokens does not grow with its tokens. At 32k and 100k
// keys they cost about 2.4 times one decode step over the same cache, which a
// verify of 3 tokens or more makes up for; a verify of 2 tokens keeps the SDPA
// of MLX.
//
//  1. The block pass. ``PrefillAttentionKernel`` shares its threadgroup layout
//     and its online softmax: a threadgroup reads one block of the keys for 8
//     token rows and the query heads of one KV head, and gives, for each token
//     row of each head, the largest score, the sum of the weights, and the
//     weighted sum of the values, in float32.
//  2. The combine pass of ``DecodeAttentionKernel``. Each threadgroup reads the
//     blocks of one token row of one query head and gives its attention
//     output.

import Foundation
import MLX

/// The attention of the few query tokens of a speculative verify over a long
/// key cache, for grouped-query attention with a head width of 256, in two
/// Metal kernels.
///
/// The kernel output is not bit-exact with the SDPA of MLX: the kernel adds
/// the keys in blocks and tiles with an online softmax, in float32, in another
/// order, and it rounds the softmax weights to the input dtype for the value
/// product.
package enum VerifyAttentionKernel {

    /// The fewest query tokens the kernel reads. One token keeps the route of
    /// ``DecodeAttentionKernel``, and the SDPA of MLX reads two tokens faster
    /// than this kernel.
    static let fewestQueries = 3

    /// The most query tokens the kernel reads: the token rows of one
    /// SIMD-group matrix.
    static let mostQueries = 8

    /// The number of keys one block holds when the block count is not at a
    /// limit.
    private static let targetKeysPerBlock = 512

    /// The fewest blocks of one KV head.
    private static let fewestBlocks = 4

    /// The most blocks of one KV head. More blocks write more block results
    /// for the combine pass to read, one row of `headDim` numbers for each
    /// token row of each query head.
    private static let mostBlocks = 64

    /// The query heads of one KV head of Qwen3.8, which one threadgroup reads
    /// whole: 12 SIMD groups hold them, and their key tiles and the score
    /// exchange still fit in 32 KB of threadgroup memory.
    private static let wholeKVHeadGroup = 6

    /// The query heads one threadgroup of ``PrefillAttentionKernel`` reads.
    private static let prefillHeadGroup = 3

    /// The block results of one block: the weighted values, the largest
    /// score and the sum of the weights of each token row.
    ///
    /// The largest score of a token row with no key in the block is the
    /// lowest float, thus its block takes no weight in the combine pass. The
    /// online softmax keeps scores in base 2; the combine pass reads them in
    /// natural-log units.
    private static let resultSource = """
        if (!loader && token < queryCount) {
            uint block = threadgroup_position_in_grid.z;
            size_t row = ((size_t(sequence) * queryHeads + queryHead) * queryCount + token)
                * threadgroups_per_grid.z + block;
            UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    partials[row * DIM + dimFirst + d * matrixSide + at.x + e] =
                        out[d].thread_elements()[e];
                }
            }
            if (dimFirst == 0 && at.x == 0) {
                largestScores[row] = rowMax == maskedScore
                    ? -metal::numeric_limits<float>::max() : rowMax * M_LN2_F;
                weightTotals[row] = rowSum;
            }
        }
        """

    /// The key range of a threadgroup: block `z` of `blockKeys` keys of the
    /// keys its token rows read.
    private static let blockRangeSource = """
        int loopFirst = int(threadgroup_position_in_grid.z) * blockKeys[0];
        int loopEnd = min(keyEnd, loopFirst + blockKeys[0]);

        """

    /// The block pass kernel. It reads the strides of the queries, the keys
    /// and the values, so a cache slice reaches it with no copy.
    private static let blockKernel = MLXFast.metalKernel(
        name: "verify_attention_blocks",
        inputNames: ["queries", "keys", "values", "scale", "blockKeys"],
        outputNames: ["partials", "largestScores", "weightTotals"],
        source: PrefillAttentionKernel.setupSource + blockRangeSource
            + PrefillAttentionKernel.loopSource + resultSource,
        header: PrefillAttentionKernel.header,
        ensureRowContiguous: false)

    /// Tells if the kernels run one attention call: the threadgroup layout of
    /// ``PrefillAttentionKernel`` holds its shapes, it has
    /// ``fewestQueries`` to ``mostQueries`` query tokens over at least
    /// ``DecodeAttentionKernel/fewestKeys`` keys, and its mask is causal, or
    /// there is no mask.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The keys, shape `(batch, kvHeads, keys, headDim)`.
    ///   - values: The values, of the shape of `keys`.
    ///   - mask: The mask of the call.
    /// - Returns: True when the kernels run the call.
    static func canRun(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        PrefillAttentionKernel.layoutHolds(queries: queries, keys: keys, values: values)
            && (fewestQueries ... mostQueries).contains(queries.dim(HeadRowsAxis.row))
            && keys.dim(HeadRowsAxis.row) >= DecodeAttentionKernel.fewestKeys
            && PrefillAttentionKernel.causalFlag(of: mask) != nil
    }

    /// The number of blocks the keys of one KV head make: the power of two
    /// at or below `keys / 512`, from 4 to 64 blocks. More, shorter blocks
    /// measured slower: each writes block results for the combine pass.
    ///
    /// - Parameter keys: The number of cached keys, 1 or more.
    /// - Returns: The block count.
    static func blockCount(keys: Int) -> Int {
        DecodeAttentionKernel.blockCount(
            keys: keys, targetKeysPerBlock: targetKeysPerBlock, fewest: fewestBlocks,
            most: mostBlocks)
    }

    /// The query heads one threadgroup reads: every query head of its KV
    /// head when they make ``wholeKVHeadGroup`` heads, thus each key tile is
    /// read once, else the head group of ``PrefillAttentionKernel``.
    ///
    /// - Parameters:
    ///   - queryHeads: The number of query heads.
    ///   - kvHeads: The number of KV heads, which divides `queryHeads`.
    /// - Returns: The query heads of one threadgroup.
    static func headsPerThreadgroup(queryHeads: Int, kvHeads: Int) -> Int {
        let group = queryHeads / kvHeads
        return group % wholeKVHeadGroup == 0 ? wholeKVHeadGroup : prefillHeadGroup
    }

    /// The attention output of each query token over the keys.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The keys, shape `(batch, kvHeads, keys, headDim)`. A slice
    ///     of the first rows of a larger cache buffer reaches the kernel with
    ///     no copy.
    ///   - values: The values, of the shape of `keys`.
    ///   - scale: The factor each score takes before the softmax.
    ///   - causal: True when query `t` reads only the keys up to key
    ///     `keys - tokens + t`, the causal mask of MLX.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, causal: Bool
    ) -> MLXArray {
        precondition(
            canRun(queries: queries, keys: keys, values: values, mask: causal ? .causal : .none),
            "the verify attention kernel cannot read queries \(queries.shape) "
                + "over keys \(keys.shape)")
        let keyCount = keys.dim(HeadRowsAxis.row)
        let blocks = blockCount(keys: keyCount)
        let rows =
            queries.dim(HeadRowsAxis.batch) * queries.dim(HeadRowsAxis.head)
            * queries.dim(HeadRowsAxis.row)
        let results = PrefillAttentionKernel.run(
            blockKernel,
            inputs: [
                queries, keys, values, MLXArray([scale]),
                MLXArray([
                    Int32(DecodeAttentionKernel.keysPerBlock(keys: keyCount, blocks: blocks))
                ]),
            ],
            queries: queries, keys: keys, causal: causal, keyBlocks: blocks,
            headsPerThreadgroup: headsPerThreadgroup(
                queryHeads: queries.dim(HeadRowsAxis.head), kvHeads: keys.dim(HeadRowsAxis.head)),
            outputShapes: [
                [rows, blocks, queries.dim(HeadRowsAxis.width)], [rows, blocks], [rows, blocks],
            ],
            outputDTypes: [.float32, .float32, .float32])
        return DecodeAttentionKernel.combine(results, like: queries)
    }

    /// The scaled dot-product attention of a call: these kernels when they
    /// run the call, else the route of ``DecodeAttentionKernel``, which reads
    /// a decode step over a long cache and keeps the SDPA of MLX for every
    /// other call.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The keys, shape `(batch, kvHeads, keys, headDim)`.
    ///   - values: The values, of the shape of `keys`.
    ///   - scale: The factor each score takes before the softmax.
    ///   - mask: The mask of the call.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    package static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard canRun(queries: queries, keys: keys, values: values, mask: mask),
            let causal = PrefillAttentionKernel.causalFlag(of: mask)
        else {
            return DecodeAttentionKernel.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        return attention(queries: queries, keys: keys, values: values, scale: scale, causal: causal)
    }
}
