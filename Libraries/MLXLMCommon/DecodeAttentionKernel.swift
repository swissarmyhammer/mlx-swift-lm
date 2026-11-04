// Copyright © 2026 Apple Inc.
//
// Two Metal kernels for the attention of one query token over a long key
// cache, for grouped-query attention.
//
// The SDPA of MLX gives one SIMD group to each query head, thus each SIMD
// group of a threadgroup reads the same key and value rows again. At a 32k
// context it reads the cache of the Qwen3.8 shape (24 query heads, 4 KV heads,
// head width 256) at about 465 GB/s. These kernels give each SIMD group its
// own key rows and half of the query heads of one KV head, so two SIMD groups
// share each row. They read the same cache at about 600 GB/s.
//
//  1. The block pass. Each threadgroup reads one block of the keys of one KV
//     head and gives, for each query head of that KV head, the largest score,
//     the sum of the weights, and the weighted sum of the values, in float32.
//  2. The combine pass. Each threadgroup reads the blocks of one query head
//     and gives the attention output.

import Foundation
import MLX

/// The attention of one query token over a long key cache, for grouped-query
/// attention, in two Metal kernels.
///
/// The kernel output is not bit-exact with the SDPA of MLX: the kernel adds
/// the keys in blocks with an online softmax, in float32, in another order.
package enum DecodeAttentionKernel {

    /// The fewest cached keys the kernel reads. Below this count the SDPA of
    /// MLX is faster, because it reads the keys in one pass.
    static let fewestKeys = 1024

    /// The number of groups the query heads of one KV head make. Each SIMD
    /// group holds one group of heads, and the SIMD groups of the other
    /// groups read the same key rows.
    private static let headGroups = 2

    /// The number of SIMD groups of one threadgroup of the block pass.
    private static let blockSimdGroups = 4

    /// The number of SIMD groups of one threadgroup of the combine pass.
    private static let combineSimdGroups = 8

    /// The most query numbers one lane holds in registers: the heads of one
    /// group times the numbers of one head that one lane reads. The Qwen3.8
    /// shape holds 3 heads of 8 numbers.
    private static let mostQueryNumbersPerLane = 24

    /// The number of keys one block holds when the block count is not at a
    /// limit.
    private static let targetKeysPerBlock = 128

    /// The fewest blocks of one KV head.
    private static let fewestBlocks = 32

    /// The most blocks of one KV head.
    private static let mostBlocks = 128

    /// The axis of the `(rows, blocks)` block results that holds the blocks.
    private static let blockAxis = 1

    /// The dtypes the kernels read.
    private static let floatTypes: [DType] = [.float32, .float16, .bfloat16]

    /// The Metal header.
    ///
    /// The upper-case constants are the layout constants of this type and
    /// the ``HeadRowsAxis`` axes, written once here so that a kernel call
    /// names only the template arguments that change with the shape. A short
    /// call list keeps the graph build of a decode step cheap.
    ///
    /// `lowestScore` stands below every real score. `UNROLL_FULL` unrolls a
    /// loop with a count the compiler knows. Each loop that indexes the
    /// per-lane arrays must unroll, else the arrays leave the registers.
    ///
    /// `LaneRows` reads the numbers of one lane from the rows of one KV
    /// head. With `UNIT_STEP` the numbers stand side by side, which is the
    /// layout of every key cache, and the reads join into wide loads.
    ///
    /// `GroupSoftmax` holds the queries and the online softmax of the heads
    /// of one head group, and adds one key stream to them.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header =
        metalSimdWidthDeclaration + """
            constexpr constant int HEAD_GROUPS = \(headGroups);
            constexpr constant int BLOCK_SIMD_GROUPS = \(blockSimdGroups);
            constexpr constant int COMBINE_SIMD_GROUPS = \(combineSimdGroups);
            constexpr constant int SEQUENCE_AXIS = \(HeadRowsAxis.batch);
            constexpr constant int HEAD_AXIS = \(HeadRowsAxis.head);
            constexpr constant int ROW_AXIS = \(HeadRowsAxis.row);
            constexpr constant int WIDTH_AXIS = \(HeadRowsAxis.width);
            constexpr constant int BLOCK_AXIS = \(blockAxis);
            constexpr constant float lowestScore = -metal::numeric_limits<float>::max();
            #define UNROLL_FULL _Pragma("clang loop unroll(full)")

            template <typename T>
            struct LaneRows {
                const device T* first;
                int64_t rowStep;
                int64_t numberStep;

                template <bool UNIT_STEP, int COUNT>
                inline void read(int row, thread float (&numbers)[COUNT]) const {
                    const device T* source = first + row * rowStep;
                    UNROLL_FULL for (int number = 0; number < COUNT; number++) {
                        numbers[number] = float(source[UNIT_STEP ? number : number * numberStep]);
                    }
                }
            };

            template <int HEADS, int NUMBERS>
            struct GroupSoftmax {
                float query[HEADS][NUMBERS];
                float output[HEADS][NUMBERS];
                float largest[HEADS];
                float total[HEADS];

                inline void add(thread const float (&keyRow)[NUMBERS], thread const float (&valueRow)[NUMBERS]) {
                    UNROLL_FULL for (int head = 0; head < HEADS; head++) {
                        float score = 0.0f;
                        UNROLL_FULL for (int number = 0; number < NUMBERS; number++) {
                            score += query[head][number] * keyRow[number];
                        }
                        score = simd_sum(score);
                        float next = max(largest[head], score);
                        float factor = exp(largest[head] - next);
                        float weight = exp(score - next);
                        largest[head] = next;
                        total[head] = total[head] * factor + weight;
                        UNROLL_FULL for (int number = 0; number < NUMBERS; number++) {
                            output[head][number] = output[head][number] * factor + weight * valueRow[number];
                        }
                    }
                }

                template <bool UNIT_STEP, typename T>
                inline void add_stream(LaneRows<T> keys, LaneRows<T> values, int first, int end, int step) {
                    for (int key = first; key < end; key += step) {
                        float keyRow[NUMBERS];
                        float valueRow[NUMBERS];
                        keys.template read<UNIT_STEP>(key, keyRow);
                        values.template read<UNIT_STEP>(key, valueRow);
                        add(keyRow, valueRow);
                    }
                }
            };

            """

    /// The block pass.
    ///
    /// SIMD group `g` holds the query heads of group `g % HEAD_GROUPS` and
    /// reads key `first + g / HEAD_GROUPS` of the block, then each key
    /// `BLOCK_SIMD_GROUPS / HEAD_GROUPS` rows further. One lane holds
    /// `DIM / SIMD_WIDTH` neighbouring numbers of each head. After the keys,
    /// the threadgroup joins the key streams of each head in threadgroup
    /// memory and writes one row of block results for each query head.
    private static let blockSource = """
        constexpr int numbersPerLane = DIM / SIMD_WIDTH;
        constexpr int headsPerGroup = GQA / HEAD_GROUPS;
        constexpr int keyStreams = BLOCK_SIMD_GROUPS / HEAD_GROUPS;
        threadgroup float streamLargest[keyStreams * GQA];
        threadgroup float streamTotal[keyStreams * GQA];
        threadgroup float streamOutput[keyStreams * GQA * DIM];

        uint lane = thread_index_in_simdgroup;
        uint simdGroup = simdgroup_index_in_threadgroup;
        uint headGroup = simdGroup % HEAD_GROUPS;
        uint keyStream = simdGroup / HEAD_GROUPS;
        uint kvHeads = keys_shape[HEAD_AXIS];
        uint queryHeads = queries_shape[HEAD_AXIS];
        uint sequence = threadgroup_position_in_grid.y / kvHeads;
        uint kvHead = threadgroup_position_in_grid.y % kvHeads;
        uint block = threadgroup_position_in_grid.z;
        uint blocks = threadgroups_per_grid.z;
        int first = int(block) * blockKeys[0];
        int end = min(int(keys_shape[ROW_AXIS]), first + blockKeys[0]);
        uint laneFirstNumber = lane * numbersPerLane;
        uint kvFirstHead = kvHead * GQA;
        uint groupFirstHead = kvFirstHead + headGroup * headsPerGroup;

        LaneRows<T> keyRows = {
            keys + sequence * keys_strides[SEQUENCE_AXIS] + kvHead * keys_strides[HEAD_AXIS]
                + laneFirstNumber * keys_strides[WIDTH_AXIS],
            keys_strides[ROW_AXIS], keys_strides[WIDTH_AXIS]};
        LaneRows<T> valueRows = {
            values + sequence * values_strides[SEQUENCE_AXIS] + kvHead * values_strides[HEAD_AXIS]
                + laneFirstNumber * values_strides[WIDTH_AXIS],
            values_strides[ROW_AXIS], values_strides[WIDTH_AXIS]};
        const device T* queryRows = queries + sequence * queries_strides[SEQUENCE_AXIS]
            + laneFirstNumber * queries_strides[WIDTH_AXIS];

        // The queries of the group, times the scale, and an empty softmax for
        // each head.
        GroupSoftmax<headsPerGroup, numbersPerLane> softmax;
        UNROLL_FULL for (int head = 0; head < headsPerGroup; head++) {
            UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
                size_t at = (groupFirstHead + head) * queries_strides[HEAD_AXIS]
                    + number * queries_strides[WIDTH_AXIS];
                softmax.query[head][number] = float(queryRows[at]) * scale[0];
                softmax.output[head][number] = 0.0f;
            }
            softmax.largest[head] = lowestScore;
            softmax.total[head] = 0.0f;
        }

        // The online softmax over the keys of this key stream. One uniform
        // test picks the loop for side-by-side numbers.
        int streamFirst = first + int(keyStream);
        if (keys_strides[WIDTH_AXIS] == 1 && values_strides[WIDTH_AXIS] == 1) {
            softmax.template add_stream<true>(keyRows, valueRows, streamFirst, end, keyStreams);
        } else {
            softmax.template add_stream<false>(keyRows, valueRows, streamFirst, end, keyStreams);
        }

        // Join the key streams of each head.
        UNROLL_FULL for (int head = 0; head < headsPerGroup; head++) {
            uint slot = keyStream * GQA + headGroup * headsPerGroup + head;
            if (lane == 0) {
                streamLargest[slot] = softmax.largest[head];
                streamTotal[slot] = softmax.total[head];
            }
            UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
                streamOutput[slot * DIM + laneFirstNumber + number] = softmax.output[head][number];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint head = simdGroup; head < GQA; head += BLOCK_SIMD_GROUPS) {
            float blockLargest = lowestScore;
            UNROLL_FULL for (int stream = 0; stream < keyStreams; stream++) {
                blockLargest = max(blockLargest, streamLargest[stream * GQA + head]);
            }
            float blockTotal = 0.0f;
            float blockOutput[numbersPerLane] = {};
            UNROLL_FULL for (int stream = 0; stream < keyStreams; stream++) {
                uint slot = stream * GQA + head;
                float weight = exp(streamLargest[slot] - blockLargest);
                blockTotal += streamTotal[slot] * weight;
                UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
                    blockOutput[number] += streamOutput[slot * DIM + laneFirstNumber + number] * weight;
                }
            }
            size_t row = (size_t(sequence) * queryHeads + kvFirstHead + head) * blocks + block;
            UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
                partials[row * DIM + laneFirstNumber + number] = blockOutput[number];
            }
            if (lane == 0) {
                largestScores[row] = blockLargest;
                weightTotals[row] = blockTotal;
            }
        }
        """

    /// The combine pass.
    ///
    /// SIMD group `g` reads block `g` of one query head and each block
    /// `COMBINE_SIMD_GROUPS` further. The threadgroup then adds the SIMD
    /// groups in threadgroup memory and writes the attention output of the
    /// head.
    private static let combineSource = """
        constexpr int numbersPerLane = DIM / SIMD_WIDTH;
        threadgroup float groupOutput[COMBINE_SIMD_GROUPS * DIM];
        threadgroup float groupTotal[COMBINE_SIMD_GROUPS];

        uint lane = thread_index_in_simdgroup;
        uint simdGroup = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.y;
        int blocks = largestScores_shape[BLOCK_AXIS];
        uint laneFirstNumber = lane * numbersPerLane;
        const device float* rowLargest = largestScores + size_t(row) * blocks;
        const device float* rowTotal = weightTotals + size_t(row) * blocks;
        const device float* rowPartials = partials + size_t(row) * blocks * DIM + laneFirstNumber;

        float largest = lowestScore;
        for (int block = int(lane); block < blocks; block += SIMD_WIDTH) {
            largest = max(largest, rowLargest[block]);
        }
        largest = simd_max(largest);

        float total = 0.0f;
        float sum[numbersPerLane] = {};
        for (int block = int(simdGroup); block < blocks; block += COMBINE_SIMD_GROUPS) {
            float weight = exp(rowLargest[block] - largest);
            total += rowTotal[block] * weight;
            UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
                sum[number] += rowPartials[block * DIM + number] * weight;
            }
        }
        UNROLL_FULL for (int number = 0; number < numbersPerLane; number++) {
            groupOutput[simdGroup * DIM + laneFirstNumber + number] = sum[number];
        }
        if (lane == 0) {
            groupTotal[simdGroup] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint column = simdGroup * SIMD_WIDTH + lane; column < DIM;
             column += COMBINE_SIMD_GROUPS * SIMD_WIDTH) {
            float value = 0.0f;
            float weights = 0.0f;
            for (int group = 0; group < COMBINE_SIMD_GROUPS; group++) {
                value += groupOutput[group * DIM + column];
                weights += groupTotal[group];
            }
            output[size_t(row) * DIM + column] = T(value / weights);
        }
        """

    /// The block pass kernel. It reads the strides of the queries, the keys
    /// and the values, so a cache slice reaches it with no copy.
    private static let blockKernel = MLXFast.metalKernel(
        name: "decode_attention_blocks",
        inputNames: ["queries", "keys", "values", "scale", "blockKeys"],
        outputNames: ["partials", "largestScores", "weightTotals"],
        source: blockSource,
        header: header,
        ensureRowContiguous: false)

    /// The combine pass kernel.
    private static let combineKernel = MLXFast.metalKernel(
        name: "decode_attention_combine",
        inputNames: ["partials", "largestScores", "weightTotals"],
        outputNames: ["output"],
        source: combineSource,
        header: header)

    /// The number of blocks the keys of one KV head make: the power of two
    /// at or below `keys / 128`, from 32 to 128 blocks. Each block then holds
    /// enough keys to hide the cost of its block results, and the blocks of
    /// all KV heads fill the GPU.
    ///
    /// - Parameter keys: The number of cached keys, 1 or more.
    /// - Returns: The block count.
    static func blockCount(keys: Int) -> Int {
        blockCount(
            keys: keys, targetKeysPerBlock: targetKeysPerBlock, fewest: fewestBlocks,
            most: mostBlocks)
    }

    /// The number of blocks some keys make, which ``VerifyAttentionKernel``
    /// shares: the power of two at or below `keys / targetKeysPerBlock`,
    /// from `fewest` to `most` blocks.
    ///
    /// - Parameters:
    ///   - keys: The number of keys, 1 or more.
    ///   - targetKeysPerBlock: The keys one block holds when the block count
    ///     is not at a limit.
    ///   - fewest: The fewest blocks, a power of two.
    ///   - most: The most blocks, a power of two.
    /// - Returns: The block count.
    static func blockCount(keys: Int, targetKeysPerBlock: Int, fewest: Int, most: Int) -> Int {
        let wanted = max(keys / targetKeysPerBlock, 1)
        let powerOfTwo = 1 << (Int.bitWidth - 1 - wanted.leadingZeroBitCount)
        return min(max(powerOfTwo, fewest), most)
    }

    /// The number of keys of each block but the last, which holds the rest.
    ///
    /// - Parameters:
    ///   - keys: The number of cached keys.
    ///   - blocks: The block count.
    /// - Returns: The fewest keys for each block that cover all keys.
    static func keysPerBlock(keys: Int, blocks: Int) -> Int {
        (keys + blocks - 1) / blocks
    }

    /// Tells if the kernels run one attention call: the default device is
    /// the GPU, one query token reads at least ``fewestKeys`` keys, and the
    /// head shape fits the registers of one lane.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The cached keys, shape `(batch, kvHeads, keys, headDim)`.
    ///   - values: The cached values, of the shape of `keys`.
    /// - Returns: True when the kernels run the call.
    static func canRun(queries: MLXArray, keys: MLXArray, values: MLXArray) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu,
            queries.ndim == HeadRowsAxis.count, keys.ndim == HeadRowsAxis.count,
            values.shape == keys.shape,
            queries.dim(HeadRowsAxis.row) == 1, keys.dim(HeadRowsAxis.row) >= fewestKeys,
            queries.dim(HeadRowsAxis.batch) == keys.dim(HeadRowsAxis.batch),
            queries.dim(HeadRowsAxis.width) == keys.dim(HeadRowsAxis.width),
            floatTypes.contains(queries.dtype),
            keys.dtype == queries.dtype, values.dtype == queries.dtype
        else { return false }
        return headShapeFits(
            queryHeads: queries.dim(HeadRowsAxis.head), kvHeads: keys.dim(HeadRowsAxis.head),
            headDim: queries.dim(HeadRowsAxis.width))
    }

    /// Tells if one SIMD group holds the queries of one head group in
    /// registers: the query heads make an even number for each KV head, and
    /// the lanes share the head width evenly.
    private static func headShapeFits(queryHeads: Int, kvHeads: Int, headDim: Int) -> Bool {
        guard kvHeads > 0, queryHeads % kvHeads == 0, headDim % metalSimdWidth == 0 else {
            return false
        }
        let group = queryHeads / kvHeads
        guard group % headGroups == 0 else { return false }
        return (group / headGroups) * (headDim / metalSimdWidth) <= mostQueryNumbersPerLane
    }

    /// The attention output of one query token over every cached key, with
    /// no mask.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, 1, headDim)`.
    ///   - keys: The cached keys, shape `(batch, kvHeads, keys, headDim)`. A
    ///     slice of a larger cache buffer reaches the kernel with no copy.
    ///   - values: The cached values, of the shape of `keys`.
    ///   - scale: The factor each score takes before the softmax.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray {
        precondition(
            canRun(queries: queries, keys: keys, values: values),
            "the decode attention kernel cannot read queries \(queries.shape) "
                + "over keys \(keys.shape)")
        let rows = queries.dim(HeadRowsAxis.batch) * queries.dim(HeadRowsAxis.head)
        let headDim = queries.dim(HeadRowsAxis.width)
        let keyCount = keys.dim(HeadRowsAxis.row)
        let blocks = blockCount(keys: keyCount)
        let blockThreads = blockSimdGroups * metalSimdWidth
        let results = blockKernel(
            [
                queries, keys, values, MLXArray([scale]),
                MLXArray([Int32(keysPerBlock(keys: keyCount, blocks: blocks))]),
            ],
            template: [
                ("T", queries.dtype), ("DIM", headDim),
                ("GQA", queries.dim(HeadRowsAxis.head) / keys.dim(HeadRowsAxis.head)),
            ],
            grid: (
                blockThreads, keys.dim(HeadRowsAxis.batch) * keys.dim(HeadRowsAxis.head), blocks
            ),
            threadGroup: (blockThreads, 1, 1),
            outputShapes: [[rows, blocks, headDim], [rows, blocks], [rows, blocks]],
            outputDTypes: [.float32, .float32, .float32])
        return combine(results, like: queries)
    }

    /// The combine pass over the block results of each query row, which
    /// ``VerifyAttentionKernel`` shares: the attention output of each row.
    ///
    /// - Parameters:
    ///   - results: The block results, in float32: the weighted sums of the
    ///     values, shape `(rows, blocks, headDim)`, the largest score of each
    ///     block in natural-log units, shape `(rows, blocks)`, and the sum of
    ///     the weights of each block, shape `(rows, blocks)`. Row `r` is row
    ///     `r` of the `(batch, queryHeads, tokens)` rows of `queries`.
    ///   - queries: The queries, whose shape and dtype the output takes.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    static func combine(_ results: [MLXArray], like queries: MLXArray) -> MLXArray {
        let rows = results[0].dim(0)
        return combineKernel(
            results,
            template: [("T", queries.dtype), ("DIM", queries.dim(HeadRowsAxis.width))],
            grid: (metalSimdWidth, rows * combineSimdGroups, 1),
            threadGroup: (metalSimdWidth, combineSimdGroups, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype])[0]
    }

    /// The scaled dot-product attention of a call: the kernels when they run
    /// the call and it has no mask, else the SDPA of MLX.
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
        guard case .none = mask, canRun(queries: queries, keys: keys, values: values) else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        return attention(queries: queries, keys: keys, values: values, scale: scale)
    }
}
