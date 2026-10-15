// Copyright © 2026 Apple Inc.
//
// One Metal flash-attention kernel for the prefill blocks of a grouped-query
// model with a head width of 256, as Qwen3.8 (24 query heads, 4 KV heads).
//
// MLX has no fused SDPA kernel for this head width when a call has more than
// a few queries. Its op path writes the full float32 score tensor of each
// layer: at a 32k context one prefill block of 2048 tokens writes
// `(24, 2048, 32768)` scores, then reads them again for the softmax and for
// the value product. This kernel holds the scores of one key tile in
// registers and adds the keys with an online softmax, in float32, thus it
// writes only the output.
//
//  1. A threadgroup reads 8 query tokens of 3 query heads that share one KV
//     head. Two SIMD groups hold each head, each one half of the head width,
//     and the partial scores of the two halves meet in threadgroup memory.
//     The scores and the output are 8x8 SIMD-group matrices.
//  2. Four more SIMD groups only load the key and value tiles into
//     threadgroup memory. They load the next tile while the other SIMD groups
//     read the current one, into a second value tile and into the key tile
//     after the scores of the current tile are done.
//  3. A causal block reads only the keys up to its last token. The tiles
//     that stand fully behind the first token of the block take no mask.

import Foundation
import MLX

/// The attention of a prefill block over its key cache, for grouped-query
/// attention with a head width of 256, in one Metal kernel.
///
/// The kernel output is not bit-exact with the SDPA of MLX: the kernel adds
/// the keys in tiles with an online softmax, in float32, in another order,
/// and it rounds the softmax weights to the input dtype for the value
/// product.
package enum PrefillAttentionKernel {

    /// The fewest query tokens the kernel reads. Below this count the other
    /// attention routes are as fast, and a decode step keeps
    /// ``DecodeAttentionKernel``.
    static let fewestQueries = 64

    /// The only head width the kernel reads.
    private static let headWidth = 256

    /// The number of query heads of one KV head that one threadgroup reads.
    private static let threadgroupHeads = 3

    /// The number of SIMD groups that share the head width of one query head.
    private static let headParts = 2

    /// The number of SIMD groups of one threadgroup that only load tiles.
    private static let loaderSimdGroups = 4

    /// The number of query tokens one threadgroup reads: one row of a
    /// SIMD-group matrix for each token.
    private static let tokensPerThreadgroup = 8

    /// The number of value tiles in threadgroup memory. The loaders fill one
    /// while the other SIMD groups read the other.
    private static let valueTiles = 2

    /// The padding of each tile row, in values. It moves the rows of a tile
    /// to other threadgroup memory banks.
    private static let tileRowPadding = 8

    /// The number of keys of one tile for 2-byte dtypes.
    private static let halfTileKeys = 16

    /// The number of keys of one tile for float32, whose tiles take two
    /// times the threadgroup memory of one 2-byte tile.
    private static let float32TileKeys = 8

    /// The dtypes the kernel reads.
    private static let floatTypes: [DType] = [.float32, .float16, .bfloat16]

    /// The Metal header.
    ///
    /// The upper-case constants are the layout constants of this type and
    /// the ``HeadRowsAxis`` axes. Lanes `lane`, `lane ^ nearRowLane` and
    /// `lane ^ farRowLane` hold the elements of one matrix row, thus one row
    /// reduction takes two shuffles.
    ///
    /// `TileRows` reads rows of one KV head from device memory. `copy_tiles`
    /// copies one tile of keys and one tile of values to threadgroup memory:
    /// each thread issues all its loads first, then all its stores, so that
    /// its loads are in flight at the same time. With `WIDE`, one load reads
    /// 16 bytes; this needs side-by-side head numbers and rows on 16-byte
    /// steps, the layout of every key cache. `load_tiles` picks the form.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header =
        metalSimdWidthDeclaration + metalMatrixLaneDeclaration + """
            constexpr constant int THREADGROUP_HEADS = \(threadgroupHeads);
            constexpr constant int HEAD_PARTS = \(headParts);
            constexpr constant int LOADER_SIMD_GROUPS = \(loaderSimdGroups);
            constexpr constant int TOKENS = \(tokensPerThreadgroup);
            constexpr constant int VALUE_TILES = \(valueTiles);
            constexpr constant int TILE_PADDING = \(tileRowPadding);
            constexpr constant int SEQUENCE_AXIS = \(HeadRowsAxis.batch);
            constexpr constant int HEAD_AXIS = \(HeadRowsAxis.head);
            constexpr constant int ROW_AXIS = \(HeadRowsAxis.row);
            constexpr constant int WIDTH_AXIS = \(HeadRowsAxis.width);
            constexpr constant ushort nearRowLane = 1;
            constexpr constant ushort farRowLane = lanesPerQuad * quadColumnBit;
            constexpr constant float maskedScore = -INFINITY;
            #define UNROLL_FULL _Pragma("clang loop unroll(full)")

            static_assert(TOKENS == matrixSide, "one matrix row for each query token");
            static_assert(HEAD_PARTS == 2, "each SIMD group adds the partial scores of one peer");

            template <typename T>
            struct TileRows {
                const device T* first;
                int64_t rowStep;
                int64_t numberStep;

                template <typename Unit>
                inline Unit read(int row, int column) const {
                    return *(const device Unit*)(first + row * rowStep + column * numberStep);
                }
            };

            template <typename T, typename Strides>
            inline bool wide_rows(Strides strides) {
                constexpr int64_t perChunk = sizeof(uint4) / sizeof(T);
                return strides[WIDTH_AXIS] == 1 && strides[ROW_AXIS] % perChunk == 0
                    && strides[HEAD_AXIS] % perChunk == 0 && strides[SEQUENCE_AXIS] % perChunk == 0;
            }

            template <typename T, int DIM, int ROWS, int THREADS, bool WIDE>
            inline void copy_tiles(
                threadgroup T* keyTile, threadgroup T* valueTile, TileRows<T> keys, TileRows<T> values,
                int first, int count, uint threadIndex
            ) {
                typedef typename metal::conditional<WIDE, uint4, T>::type Unit;
                constexpr int perUnit = sizeof(Unit) / sizeof(T);
                constexpr int unitsPerRow = DIM / perUnit;
                constexpr int units = ROWS * unitsPerRow;
                constexpr int perThread = (units + THREADS - 1) / THREADS;
                constexpr int tileStride = DIM + TILE_PADDING;
                Unit keyUnits[perThread];
                Unit valueUnits[perThread];
                UNROLL_FULL for (int i = 0; i < perThread; i++) {
                    int unit = threadIndex + i * THREADS;
                    int row = unit / unitsPerRow;
                    int column = (unit % unitsPerRow) * perUnit;
                    bool real = (units % THREADS == 0 || unit < units) && first + row < count;
                    keyUnits[i] = real ? keys.template read<Unit>(first + row, column) : Unit(0);
                    valueUnits[i] = real ? values.template read<Unit>(first + row, column) : Unit(0);
                }
                UNROLL_FULL for (int i = 0; i < perThread; i++) {
                    int unit = threadIndex + i * THREADS;
                    if (units % THREADS == 0 || unit < units) {
                        int row = unit / unitsPerRow;
                        int column = (unit % unitsPerRow) * perUnit;
                        *(threadgroup Unit*)(keyTile + row * tileStride + column) = keyUnits[i];
                        *(threadgroup Unit*)(valueTile + row * tileStride + column) = valueUnits[i];
                    }
                }
            }

            template <typename T, int DIM, int ROWS, int THREADS>
            inline void load_tiles(
                bool wide, threadgroup T* keyTile, threadgroup T* valueTile, TileRows<T> keys,
                TileRows<T> values, int first, int count, uint threadIndex
            ) {
                if (wide) {
                    copy_tiles<T, DIM, ROWS, THREADS, true>(keyTile, valueTile, keys, values, first, count, threadIndex);
                } else {
                    copy_tiles<T, DIM, ROWS, THREADS, false>(keyTile, valueTile, keys, values, first, count, threadIndex);
                }
            }

            """

    /// The kernel body.
    ///
    /// SIMD group `g` below `THREADGROUP_HEADS * HEAD_PARTS` holds query head
    /// `g / HEAD_PARTS` of the threadgroup and the head-width part
    /// `g % HEAD_PARTS`; the SIMD groups above it are loaders. Each tile
    /// takes two threadgroup barriers on every SIMD group: one after the
    /// partial scores, when the key tile is free, and one at the end, when
    /// the next tiles are in place.
    private static let source = """
        constexpr int partDim = DIM / HEAD_PARTS;
        constexpr int dimFrags = partDim / matrixSide;
        constexpr int keyFrags = BK / matrixSide;
        constexpr int laneScores = keyFrags * elementsPerLane;
        constexpr int computeGroups = THREADGROUP_HEADS * HEAD_PARTS;
        constexpr int computeThreads = computeGroups * SIMD_WIDTH;
        constexpr int allThreads = (computeGroups + LOADER_SIMD_GROUPS) * SIMD_WIDTH;
        constexpr int loaderThreads = LOADER_SIMD_GROUPS * SIMD_WIDTH;
        constexpr int tileSize = BK * (DIM + TILE_PADDING);
        constexpr int tileStride = DIM + TILE_PADDING;
        constexpr int headGroups = GQA / THREADGROUP_HEADS;
        static_assert(GQA % THREADGROUP_HEADS == 0, "whole head groups for each KV head");
        static_assert(partDim % matrixSide == 0 && BK % matrixSide == 0, "whole matrices");
        threadgroup T tiles[(1 + VALUE_TILES) * tileSize];
        threadgroup float exchange[computeGroups * laneScores * SIMD_WIDTH];

        ushort lane = thread_index_in_simdgroup;
        ushort simdGroup = simdgroup_index_in_threadgroup;
        uint threadIndex = thread_position_in_threadgroup.x;
        bool loader = simdGroup >= computeGroups;
        int queryCount = queries_shape[ROW_AXIS];
        int keyCount = keys_shape[ROW_AXIS];
        uint kvHeads = keys_shape[HEAD_AXIS];
        uint queryHeads = queries_shape[HEAD_AXIS];
        uint place = threadgroup_position_in_grid.y;
        uint sequence = place / (kvHeads * headGroups);
        uint kvHead = (place / headGroups) % kvHeads;
        uint head = min(uint(simdGroup / HEAD_PARTS), uint(THREADGROUP_HEADS - 1));
        uint queryHead = kvHead * GQA + (place % headGroups) * THREADGROUP_HEADS + head;
        int dimFirst = (simdGroup % HEAD_PARTS) * partDim;
        int firstToken = threadgroup_position_in_grid.x * TOKENS;
        int offset = keyCount - queryCount;
        int keyEnd = CAUSAL ? min(keyCount, offset + firstToken + TOKENS) : keyCount;
        int openEnd = CAUSAL ? min(keyCount, offset + firstToken + 1) : keyCount;
        short2 at = lane_coordinate(lane);
        int token = firstToken + at.y;

        TileRows<T> keyRows = {
            keys + sequence * keys_strides[SEQUENCE_AXIS] + kvHead * keys_strides[HEAD_AXIS],
            keys_strides[ROW_AXIS], keys_strides[WIDTH_AXIS]};
        TileRows<T> valueRows = {
            values + sequence * values_strides[SEQUENCE_AXIS] + kvHead * values_strides[HEAD_AXIS],
            values_strides[ROW_AXIS], values_strides[WIDTH_AXIS]};
        bool wide = wide_rows<T>(keys_strides) && wide_rows<T>(values_strides);
        const device T* queryRows = queries + sequence * queries_strides[SEQUENCE_AXIS]
            + queryHead * queries_strides[HEAD_AXIS];

        // The queries of this token row and head-width part, and an empty
        // output.
        simdgroup_matrix<T, matrixSide, matrixSide> query[dimFrags];
        simdgroup_matrix<float, matrixSide, matrixSide> out[dimFrags];
        UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
            UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                int column = dimFirst + d * matrixSide + at.x + e;
                query[d].thread_elements()[e] = token < queryCount
                    ? queryRows[token * queries_strides[ROW_AXIS] + column * queries_strides[WIDTH_AXIS]]
                    : T(0);
            }
            out[d] = make_filled_simdgroup_matrix<float, matrixSide, matrixSide>(0.0f);
        }
        float rowMax = maskedScore;
        float rowSum = 0.0f;
        float factor = scale[0] * M_LOG2E_F;

        load_tiles<T, DIM, BK, allThreads>(
            wide, tiles, tiles + tileSize, keyRows, valueRows, 0, keyCount, threadIndex);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int tile = 0;
        for (int first = 0; first < keyEnd; first += BK, tile++) {
            threadgroup T* keyTile = tiles;
            threadgroup T* valueTile = tiles + (1 + tile % VALUE_TILES) * tileSize;

            // A loader waits for the partial scores, which free the key
            // tile, and loads the next tiles.
            if (loader) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                int next = first + BK;
                if (next < keyEnd) {
                    load_tiles<T, DIM, BK, loaderThreads>(
                        wide, keyTile, tiles + (1 + (tile + 1) % VALUE_TILES) * tileSize, keyRows,
                        valueRows, next, keyCount, threadIndex - computeThreads);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }

            // The partial scores of this head-width part.
            simdgroup_matrix<float, matrixSide, matrixSide> scores[keyFrags];
            UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                scores[k] = make_filled_simdgroup_matrix<float, matrixSide, matrixSide>(0.0f);
            }
            UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
                simdgroup_matrix<T, matrixSide, matrixSide> keyMatrix[keyFrags];
                simdgroup_barrier(mem_flags::mem_none);
                UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                    simdgroup_load(
                        keyMatrix[k], keyTile + k * matrixSide * tileStride + dimFirst + d * matrixSide,
                        tileStride, ulong2(0, 0), true);
                }
                simdgroup_barrier(mem_flags::mem_none);
                UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                    simdgroup_multiply_accumulate(scores[k], query[d], keyMatrix[k], scores[k]);
                }
            }

            // The whole scores: each SIMD group adds the part of its peer.
            UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    exchange[(simdGroup * laneScores + k * elementsPerLane + e) * SIMD_WIDTH + lane] =
                        scores[k].thread_elements()[e];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ushort peer = simdGroup ^ (HEAD_PARTS - 1);
            UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    scores[k].thread_elements()[e] +=
                        exchange[(peer * laneScores + k * elementsPerLane + e) * SIMD_WIDTH + lane];
                }
            }

            // The online softmax, in base 2. Only a tile that reaches past
            // the first token of the block, or past the last key, takes the
            // mask.
            bool masked = first + BK > openEnd;
            float tileMax = maskedScore;
            UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    float score = scores[k].thread_elements()[e] * factor;
                    if (masked) {
                        int key = first + k * matrixSide + at.x + e;
                        bool visible = key < keyCount && (!CAUSAL || key <= offset + token);
                        score = visible ? score : maskedScore;
                    }
                    scores[k].thread_elements()[e] = score;
                    tileMax = max(tileMax, score);
                }
            }
            tileMax = max(tileMax, simd_shuffle_xor(tileMax, nearRowLane));
            tileMax = max(tileMax, simd_shuffle_xor(tileMax, farRowLane));
            float largest = max(rowMax, tileMax);
            float correction = fast::exp2(rowMax - largest);
            rowMax = largest;
            float tileSum = 0.0f;
            simdgroup_matrix<T, matrixSide, matrixSide> weights[keyFrags];
            UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    float weight = fast::exp2(scores[k].thread_elements()[e] - largest);
                    weights[k].thread_elements()[e] = T(weight);
                    tileSum += weight;
                }
            }
            tileSum += simd_shuffle_xor(tileSum, nearRowLane);
            tileSum += simd_shuffle_xor(tileSum, farRowLane);
            rowSum = rowSum * correction + tileSum;
            UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
                out[d].thread_elements() *= correction;
            }

            // The weighted values.
            UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
                simdgroup_matrix<T, matrixSide, matrixSide> valueMatrix[keyFrags];
                simdgroup_barrier(mem_flags::mem_none);
                UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                    simdgroup_load(
                        valueMatrix[k], valueTile + k * matrixSide * tileStride + dimFirst + d * matrixSide,
                        tileStride);
                }
                simdgroup_barrier(mem_flags::mem_none);
                UNROLL_FULL for (int k = 0; k < keyFrags; k++) {
                    simdgroup_multiply_accumulate(out[d], weights[k], valueMatrix[k], out[d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (!loader && token < queryCount) {
            float inverse = 1.0f / rowSum;
            size_t rowBase = ((size_t(sequence) * queryHeads + queryHead) * queryCount + token) * DIM;
            UNROLL_FULL for (int d = 0; d < dimFrags; d++) {
                UNROLL_FULL for (int e = 0; e < elementsPerLane; e++) {
                    output[rowBase + dimFirst + d * matrixSide + at.x + e] =
                        T(out[d].thread_elements()[e] * inverse);
                }
            }
        }
        """

    /// The kernel. It reads the strides of the queries, the keys and the
    /// values, so a cache slice reaches it with no copy.
    private static let kernel = MLXFast.metalKernel(
        name: "prefill_attention",
        inputNames: ["queries", "keys", "values", "scale"],
        outputNames: ["output"],
        source: source,
        header: header,
        ensureRowContiguous: false)

    /// Tells if the kernel runs one attention call: the default device is
    /// the GPU, the call has at least ``fewestQueries`` query tokens, the
    /// heads are 256 numbers wide, each threadgroup gets a whole head group,
    /// and the mask is causal over at least as many keys as queries, or
    /// there is no mask.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The keys, shape `(batch, kvHeads, keys, headDim)`.
    ///   - values: The values, of the shape of `keys`.
    ///   - mask: The mask of the call.
    /// - Returns: True when the kernel runs the call.
    static func canRun(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu,
            queries.ndim == HeadRowsAxis.count, keys.ndim == HeadRowsAxis.count,
            values.shape == keys.shape,
            queries.dim(HeadRowsAxis.row) >= fewestQueries, keys.dim(HeadRowsAxis.row) > 0,
            queries.dim(HeadRowsAxis.batch) == keys.dim(HeadRowsAxis.batch),
            queries.dim(HeadRowsAxis.width) == headWidth, keys.dim(HeadRowsAxis.width) == headWidth,
            floatTypes.contains(queries.dtype),
            keys.dtype == queries.dtype, values.dtype == queries.dtype,
            headGroupsFit(queryHeads: queries.dim(HeadRowsAxis.head), kvHeads: keys.dim(HeadRowsAxis.head)),
            let causal = causalFlag(of: mask)
        else { return false }
        return !causal || keys.dim(HeadRowsAxis.row) >= queries.dim(HeadRowsAxis.row)
    }

    /// Tells if the query heads of each KV head make whole threadgroup head
    /// groups.
    private static func headGroupsFit(queryHeads: Int, kvHeads: Int) -> Bool {
        guard kvHeads > 0, queryHeads % kvHeads == 0 else { return false }
        return (queryHeads / kvHeads) % threadgroupHeads == 0
    }

    /// The causal flag of a mask the kernel reads: true for a causal mask,
    /// false for no mask, and `nil` for a mask the kernel does not read.
    private static func causalFlag(of mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool? {
        if case .causal = mask { return true }
        if case .none = mask { return false }
        return nil
    }

    /// The number of keys of one tile for a dtype: the most keys whose key
    /// tile and two value tiles fit in threadgroup memory.
    private static func tileKeys(_ dtype: DType) -> Int {
        dtype == .float32 ? float32TileKeys : halfTileKeys
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
            "the prefill attention kernel cannot read queries \(queries.shape) "
                + "over keys \(keys.shape)")
        let groupSize = queries.dim(HeadRowsAxis.head) / keys.dim(HeadRowsAxis.head)
        let threads = (threadgroupHeads * headParts + loaderSimdGroups) * metalSimdWidth
        let tokenBlocks =
            (queries.dim(HeadRowsAxis.row) + tokensPerThreadgroup - 1) / tokensPerThreadgroup
        let headBlocks =
            keys.dim(HeadRowsAxis.batch) * keys.dim(HeadRowsAxis.head) * (groupSize / threadgroupHeads)
        return kernel(
            [queries, keys, values, MLXArray([scale])],
            template: [
                ("T", queries.dtype), ("DIM", headWidth), ("GQA", groupSize),
                ("BK", tileKeys(queries.dtype)), ("CAUSAL", causal),
            ],
            grid: (threads * tokenBlocks, headBlocks, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype])[0]
    }

    /// The scaled dot-product attention of a call: this kernel when it runs
    /// the call, else the route of ``DecodeAttentionKernel``, which reads a
    /// decode step over a long cache and keeps the SDPA of MLX for every
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
            let causal = causalFlag(of: mask)
        else {
            return DecodeAttentionKernel.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        return attention(queries: queries, keys: keys, values: values, scale: scale, causal: causal)
    }
}
