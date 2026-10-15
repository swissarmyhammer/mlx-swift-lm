// Copyright © 2026 Apple Inc.
//
// The fused lightning-indexer Metal kernel of DeepSeek-V4.
//
// The method follows the kernel `kernel_lightning_indexer` of llama.cpp,
// ggml/src/ggml-metal/kernels/fa_aux.metal (PR 25893): simdgroup matrix
// products of the queries and the pooled keys, then the ReLU, the head weight
// and the sum over the heads in the same kernel, thus the per-head scores
// never go to device memory. Manual transcription of the method; no git
// ancestry. Four details do not come from that kernel:
//
//  1. The queries and the keys stay in float32, and the products add in
//     float32. The llama.cpp kernel moves both to float16, which changes more
//     scores than this port accepts.
//  2. A threadgroup scores all heads of one token against one tile of
//     chunks, in four SIMD groups: two halves of the heads times two halves
//     of the chunks. Each SIMD group keeps its sums in registers for the
//     whole head width. The llama.cpp kernel keeps one tile of keys and runs
//     over the queries and over tiles of eight heads.
//  3. One thread for each chunk adds the heads in head order, with the order
//     of operations of the op path: `(max(s, 0) * scale) * weight`.
//  4. The kernel writes the masked score for each chunk that reaches past
//     its query, and a tile of chunks that is hidden from a token takes no
//     products for that token.

import Foundation
import MLX

/// One Metal kernel that gives the indexer score of each pooled chunk for
/// each query, with the per-head scores kept in the threadgroup.
///
/// The op path of the indexer scores each head with a matmul, which writes
/// `(batch, heads, tokens, chunks)` float32 scores, and then adds the heads
/// in a second kernel. At a 32k context one prefill step of 2048 tokens
/// writes and reads 4 GiB there, for each indexer layer. This kernel writes
/// only the `(batch, tokens, chunks)` answer.
///
/// The score of chunk `c` for query `t` is
/// `sum_h (max(q_th . k_c, 0) * scale) * w_th`, or the masked score when the
/// chunk reaches past the query.
enum DeepSeekV4IndexerScoreKernel {

    /// The block-causal rule the kernel applies to each score.
    struct Visibility {
        /// The absolute position of the first token of the block.
        let offset: Int

        /// The number of raw positions one pooled chunk covers.
        let chunkWidth: Int

        /// The score of a chunk that its query may not read.
        let maskedScore: Float
    }

    /// The place of each value in the `scalars` input of the kernel.
    private enum ScalarSlot {
        /// The factor each per-head score takes.
        static let scale = 0
        /// The score of a chunk that its query may not read.
        static let maskedScore = 1
        /// The number of values.
        static let count = 2
    }

    /// The number of pooled chunks one threadgroup scores.
    private static let blockChunks = 64

    /// The number of head-width columns one tile load brings into the
    /// threadgroup.
    private static let blockDepth = 32

    /// The number of tokens one threadgroup scores, one after the other.
    /// More than one token for each threadgroup spreads the launch cost of a
    /// threadgroup over more work.
    private static let blockTokens = 4

    /// The number of SIMD groups that split the heads of one token.
    private static let rowGroups = 2

    /// The number of SIMD groups that split the chunks of one tile.
    private static let columnGroups = 2

    /// The floats each tile row takes after its `blockDepth` values, which
    /// keeps the rows of a fragment load in different memory banks.
    private static let tilePad = 4

    /// The number of threads in one threadgroup.
    private static let threadsPerGroup =
        rowGroups * columnGroups * DeepSeekV4MetalKernel.simdGroupWidth

    /// The axis of the `(batch, heads, tokens, headDim)` queries that holds
    /// the heads.
    private static let queryHeadAxis = 1

    /// The axis of the `(batch, heads, tokens, headDim)` queries that holds
    /// the tokens.
    private static let queryTokenAxis = 2

    /// The axis of the `(batch, chunks, headDim)` keys that holds the chunks.
    private static let keyChunkAxis = 1

    /// The Metal source. The template arguments are `HEADS`, the number of
    /// heads; `DIM`, the head width; `CHUNK_WIDTH`, the raw positions of one
    /// chunk; `BLOCK_CHUNKS`, `BLOCK_DEPTH` and `BLOCK_TOKENS`, the tile
    /// sizes; `ROW_GROUPS` and `COLUMN_GROUPS`, the split of the tile over
    /// the SIMD groups; `SIMD_WIDTH`; `TILE_PAD`; `VECTOR_WIDTH`, the floats
    /// of one load; `SCALE_SLOT` and `MASKED_SCORE_SLOT`, the places in
    /// `scalars`; and `QUERY_TOKEN_AXIS` and `KEY_CHUNK_AXIS`, the axes of the
    /// input shapes that hold the token count and the chunk count. The two
    /// counts come from the input shapes, thus a pool that grows does not
    /// compile a new specialization.
    ///
    /// The head tile pads the heads up to a multiple of the rows of the SIMD
    /// groups, and a head past `HEADS`, a column past `DIM` and a chunk past
    /// the pool load as zero. A zero query row gives a zero score, and a zero
    /// weight takes it out of the sum.
    ///
    /// The per-head sums of one token share the threadgroup memory of the
    /// tiles: the last barrier of the product loop ends every read of the
    /// tiles before the first store of the sums.
    ///
    /// The source keeps no `device` pointer into an input, because an input
    /// can hold fewer values than
    /// ``DeepSeekV4MetalKernel/smallestDeviceInput``. Thus the source indexes
    /// each input from a base offset, and the tile load of the header takes
    /// either address space.
    private static let source = """
        constexpr uint matrixSide = 8;
        static_assert(BLOCK_CHUNKS % (COLUMN_GROUPS * matrixSide) == 0, "whole matrices of chunks");
        static_assert(BLOCK_DEPTH % matrixSide == 0, "whole matrices of the head width");
        constexpr uint rowSpan = ROW_GROUPS * matrixSide;
        constexpr uint paddedHeads = (HEADS + rowSpan - 1) / rowSpan * rowSpan;
        constexpr uint rowFragments = paddedHeads / rowSpan;
        constexpr uint columnFragments = BLOCK_CHUNKS / (COLUMN_GROUPS * matrixSide);
        constexpr uint threads = ROW_GROUPS * COLUMN_GROUPS * SIMD_WIDTH;
        constexpr uint stride = BLOCK_DEPTH + TILE_PAD;
        constexpr uint tileFloats = (paddedHeads + BLOCK_CHUNKS) * stride;
        constexpr uint sumFloats = paddedHeads * BLOCK_CHUNKS;

        threadgroup float shared[tileFloats > sumFloats ? tileFloats : sumFloats];
        threadgroup float* queryTile = shared;
        threadgroup float* keyTile = shared + paddedHeads * stride;
        threadgroup float* headSums = shared;
        threadgroup float headWeights[paddedHeads];

        uint lane = thread_index_in_threadgroup;
        uint simdGroup = simdgroup_index_in_threadgroup;
        uint rowFirst = (simdGroup / COLUMN_GROUPS) * rowFragments * matrixSide;
        uint columnFirst = (simdGroup % COLUMN_GROUPS) * columnFragments * matrixSide;
        uint sequence = threadgroup_position_in_grid.z;
        uint tokens = queries_shape[QUERY_TOKEN_AXIS];
        uint chunks = keys_shape[KEY_CHUNK_AXIS];
        uint firstChunk = threadgroup_position_in_grid.x * BLOCK_CHUNKS;
        uint tileChunks = min(uint(BLOCK_CHUNKS), chunks - firstChunk);
        float scale = scalars[SCALE_SLOT];
        float maskedScore = scalars[MASKED_SCORE_SLOT];
        size_t keyBase = (size_t(sequence) * chunks + firstChunk) * DIM;

        for (uint step = 0; step < BLOCK_TOKENS; step++) {
            uint token = threadgroup_position_in_grid.y * BLOCK_TOKENS + step;
            if (token >= tokens) {
                break;
            }
            int position = offsets[0] + int(token);
            device float* scoreRow = scores + (size_t(sequence) * tokens + token) * chunks + firstChunk;
            if (!chunk_is_visible(firstChunk, position, CHUNK_WIDTH)) {
                for (uint c = lane; c < tileChunks; c += threads) {
                    scoreRow[c] = maskedScore;
                }
                continue;
            }
            size_t queryBase = ((size_t(sequence) * HEADS) * tokens + token) * DIM;

            simdgroup_float8x8 sums[rowFragments][columnFragments];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < rowFragments; r++) {
                #pragma clang loop unroll(full)
                for (uint c = 0; c < columnFragments; c++) {
                    sums[r][c] = make_filled_simdgroup_matrix<float, matrixSide>(0.0f);
                }
            }

            for (uint depth = 0; depth < DIM; depth += BLOCK_DEPTH) {
                load_tile<paddedHeads, BLOCK_DEPTH, stride, threads, DIM, VECTOR_WIDTH>(
                    queryTile, queries, queryBase, size_t(tokens) * DIM, HEADS, depth, lane);
                load_tile<BLOCK_CHUNKS, BLOCK_DEPTH, stride, threads, DIM, VECTOR_WIDTH>(
                    keyTile, keys, keyBase, DIM, tileChunks, depth, lane);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < BLOCK_DEPTH; k += matrixSide) {
                    simdgroup_float8x8 a[rowFragments];
                    simdgroup_float8x8 b[columnFragments];
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < rowFragments; r++) {
                        simdgroup_load(a[r], queryTile + (rowFirst + r * matrixSide) * stride + k, stride);
                    }
                    #pragma clang loop unroll(full)
                    for (uint c = 0; c < columnFragments; c++) {
                        simdgroup_load(
                            b[c], keyTile + (columnFirst + c * matrixSide) * stride + k, stride,
                            ulong2(0, 0), true);
                    }
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < rowFragments; r++) {
                        #pragma clang loop unroll(full)
                        for (uint c = 0; c < columnFragments; c++) {
                            simdgroup_multiply_accumulate(sums[r][c], a[r], b[c], sums[r][c]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            #pragma clang loop unroll(full)
            for (uint r = 0; r < rowFragments; r++) {
                #pragma clang loop unroll(full)
                for (uint c = 0; c < columnFragments; c++) {
                    simdgroup_store(
                        sums[r][c],
                        headSums + (rowFirst + r * matrixSide) * BLOCK_CHUNKS + columnFirst
                            + c * matrixSide,
                        BLOCK_CHUNKS);
                }
            }
            size_t weightBase = (size_t(sequence) * tokens + token) * HEADS;
            for (uint h = lane; h < paddedHeads; h += threads) {
                headWeights[h] = h < HEADS ? weights[weightBase + h] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = lane; c < tileChunks; c += threads) {
                float score = 0.0f;
                for (uint h = 0; h < HEADS; h++) {
                    score += max(headSums[h * BLOCK_CHUNKS + c], 0.0f) * scale * headWeights[h];
                }
                scoreRow[c] = chunk_is_visible(firstChunk + c, position, CHUNK_WIDTH)
                    ? score : maskedScore;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    /// The Metal header.
    ///
    /// `chunk_is_visible` is the block-causal rule of
    /// ``DeepSeekV4Math/pooledChunkVisibility(queryCount:offset:chunkCount:chunkWidth:)``:
    /// a query reads a chunk only when the whole chunk stands behind it.
    ///
    /// `load_tile` copies `DEPTH` columns, from column `first`, of the first
    /// `ROWS` rows of a row-major matrix into a threadgroup tile of row
    /// stride `STRIDE`, one float4 at a time. The matrix starts `base` floats
    /// into `source`, and `rowStride` is the distance in floats between two
    /// of its rows. A row from `count` on and a column from `WIDTH` on load
    /// as zero. The source is a template argument, and `load_four` has one
    /// form for each address space, because an input of fewer values than
    /// ``DeepSeekV4MetalKernel/smallestDeviceInput`` is in the `constant`
    /// address space.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header = """
        inline bool chunk_is_visible(uint chunk, int position, int chunkWidth) {
            return int(chunk + 1) * chunkWidth <= position + 1;
        }

        inline float4 load_four(const device float* source, size_t index) {
            return *((const device float4*)(source + index));
        }

        inline float4 load_four(const constant float* source, size_t index) {
            return *((const constant float4*)(source + index));
        }

        template <
            uint ROWS, uint DEPTH, uint STRIDE, uint THREADS, uint WIDTH, uint VECTOR,
            typename Source>
        inline void load_tile(
            threadgroup float* tile, Source source, size_t base, size_t rowStride, uint count,
            uint first, uint lane
        ) {
            static_assert(VECTOR == 4, "the tile loads read float4");
            constexpr uint vectorsPerRow = DEPTH / VECTOR;
            #pragma clang loop unroll(full)
            for (uint i = lane; i < ROWS * vectorsPerRow; i += THREADS) {
                uint row = i / vectorsPerRow;
                uint column = (i % vectorsPerRow) * VECTOR;
                float4 value = (row < count && first + column < WIDTH)
                    ? load_four(source, base + row * rowStride + first + column)
                    : float4(0.0f);
                *((threadgroup float4*)(tile + row * STRIDE + column)) = value;
            }
        }

        """

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_lightning_indexer",
        inputNames: ["queries", "keys", "weights", "scalars", "offsets"],
        outputNames: ["scores"],
        source: source,
        header: header)

    /// Tells if the kernel can run. It is a Metal kernel, thus it runs only
    /// when the default device is the GPU, and its tile loads read the head
    /// width in whole vectors of ``DeepSeekV4MetalKernel/vectorWidth``
    /// floats.
    ///
    /// - Parameter headDim: The width of one indexer head.
    /// - Returns: True when the kernel can score heads of this width.
    static func canRun(headDim: Int) -> Bool {
        DeepSeekV4MetalKernel.deviceIsGPU && headDim % DeepSeekV4MetalKernel.vectorWidth == 0
    }

    /// The indexer score of each pooled chunk for each query.
    ///
    /// - Parameters:
    ///   - queries: The rotated queries, shape `(batch, heads, tokens, headDim)`.
    ///   - keys: The pooled keys, shape `(batch, chunks, headDim)`. It must
    ///     hold one chunk or more.
    ///   - headWeights: The weight of each head for each query, shape
    ///     `(batch, tokens, heads)`.
    ///   - scale: The factor each per-head score takes.
    ///   - visibility: The block-causal rule of the block.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`, in float32. A
    ///   chunk that reaches past its query takes `visibility.maskedScore`.
    static func scores(
        queries: MLXArray, keys: MLXArray, headWeights: MLXArray, scale: Float,
        visibility: Visibility
    ) -> MLXArray {
        let batch = queries.dim(0)
        let heads = queries.dim(queryHeadAxis)
        let tokens = queries.dim(queryTokenAxis)
        let headDim = queries.dim(DeepSeekV4MetalKernel.widthAxis)
        let chunks = keys.dim(keyChunkAxis)
        let vectorWidth = DeepSeekV4MetalKernel.vectorWidth
        precondition(
            headDim % vectorWidth == 0,
            "the kernel reads heads of a multiple of \(vectorWidth) floats, not \(headDim)")
        precondition(chunks > 0, "the kernel scores one chunk or more")
        precondition(
            keys.shape == [batch, chunks, headDim] && headWeights.shape == [batch, tokens, heads],
            "keys \(keys.shape) and head weights \(headWeights.shape) do not fit queries \(queries.shape)"
        )
        var scalars = [Float](repeating: 0, count: ScalarSlot.count)
        scalars[ScalarSlot.scale] = scale
        scalars[ScalarSlot.maskedScore] = visibility.maskedScore
        let chunkTiles = DeepSeekV4MetalKernel.groupCount(covering: chunks, groupSize: blockChunks)
        let tokenTiles = DeepSeekV4MetalKernel.groupCount(covering: tokens, groupSize: blockTokens)
        return kernel(
            [
                queries.asType(.float32), keys.asType(.float32), headWeights.asType(.float32),
                MLXArray(scalars), MLXArray([Int32(visibility.offset)]),
            ],
            template: [
                ("HEADS", heads), ("DIM", headDim), ("CHUNK_WIDTH", visibility.chunkWidth),
                ("BLOCK_CHUNKS", blockChunks), ("BLOCK_DEPTH", blockDepth),
                ("BLOCK_TOKENS", blockTokens), ("ROW_GROUPS", rowGroups),
                ("COLUMN_GROUPS", columnGroups),
                ("SIMD_WIDTH", DeepSeekV4MetalKernel.simdGroupWidth),
                ("TILE_PAD", tilePad), ("VECTOR_WIDTH", vectorWidth),
                ("SCALE_SLOT", ScalarSlot.scale), ("MASKED_SCORE_SLOT", ScalarSlot.maskedScore),
                ("QUERY_TOKEN_AXIS", queryTokenAxis), ("KEY_CHUNK_AXIS", keyChunkAxis),
            ],
            grid: (chunkTiles * threadsPerGroup, tokenTiles, batch),
            threadGroup: (threadsPerGroup, 1, 1),
            outputShapes: [[batch, tokens, chunks]],
            outputDTypes: [.float32])[0]
    }
}
