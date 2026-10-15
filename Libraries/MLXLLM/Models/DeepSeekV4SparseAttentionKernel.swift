// Copyright © 2026 Apple Inc.
//
// The fused sparse attention Metal kernel of DeepSeek-V4.
//
// The method follows the sparse flash attention of llama.cpp,
// ggml/src/ggml-metal/kernels/fa_vec_common.metal (PR 28098 and PR 29377):
// one kernel reads the window keys and the pooled keys by index, adds them
// with an online softmax that starts from the sink, and reads each pooled key
// as its own value. Manual transcription of the method; no git ancestry. Four
// details do not come from that kernel:
//
//  1. A threadgroup reads one query token and a block of heads. The heads of
//     DeepSeek-V4 share one latent key head, thus the scores of one key tile
//     are one small matrix product, and the kernel runs it on the SIMD-group
//     matrices in float32.
//  2. Each SIMD group holds one slice of the head width of the queries and of
//     the output. The partial scores of the slices meet in the threadgroup,
//     and one SIMD group reads the softmax of each head, one lane for each key
//     of the tile.
//  3. A query reads only its own window rows: at most one window of rows, in
//     one range that the shapes give. The keys of the block that the window
//     mask hides take no product at all.
//  4. A pick that reaches past its query, and a key past the end of the
//     list, load as zero and take no weight.

import Foundation
import MLX

/// One Metal kernel that gives the DeepSeek-V4 attention output of each query
/// of a prefill block over its window rows and its pooled rows.
///
/// The op path joins the window keys of the block to the pooled chunks, scores
/// every joined key for every query and hides most of them with a mask. At a
/// 32k context one prefill step of 2048 tokens writes a float32 score tensor
/// of `(64, 2048, 2175 + picks)` for each layer, and the ranked layers also
/// copy the 512 picked chunks of each query. This kernel reads only the keys
/// each query may see and writes only the output.
///
/// Query `t` of a block of `tokens` queries stands at row
/// `windowKeys - tokens + t` of the window keys, the layout a
/// `RotatingKVCache` gives back, and it reads the window rows from
/// `row - window + 1` to `row`. It also reads the pooled rows
/// ``PooledRows`` names, each one only when the whole chunk stands behind the
/// query, and the learned sink of its head.
///
/// One threadgroup reads one query token, thus a decode step gives only one
/// threadgroup for each block of heads and leaves most of the GPU idle. Such
/// a call keeps the op path; ``canRun(queries:)`` tells which calls fill the
/// GPU.
enum DeepSeekV4SparseAttentionKernel {

    /// The pooled chunks one attention call reads beside the window. Each
    /// chunk is its own value.
    struct PooledRows {
        /// The pooled chunks, shape `(batch, chunks, headDim)`. It holds one
        /// chunk or more.
        let pool: MLXArray

        /// The chunks each query reads, shape `(batch, tokens, picks)`, or
        /// `nil` when each query reads every chunk that stands behind it.
        let picked: MLXArray?

        /// The absolute position of the first query of the block.
        let offset: Int

        /// The number of raw positions one pooled chunk covers.
        let chunkWidth: Int
    }

    /// The window rows one attention call reads.
    struct WindowRows {
        /// The window keys, shape `(batch, 1, windowKeys, headDim)`. The last
        /// `tokens` rows are the keys of the block, and the rows ahead of
        /// them are the earlier rows the cache gives back, in time order.
        let keys: MLXArray

        /// The window values, of the shape of ``keys``. A DeepSeek-V4 cache
        /// gives the keys back as the values.
        let values: MLXArray

        /// The sliding window: the most recent rows one query reads.
        let window: Int
    }

    /// The number of heads one threadgroup reads: two rows of SIMD-group
    /// matrices.
    private static let headBlock = 16

    /// The head-width columns of the queries and of the output that one SIMD
    /// group holds.
    private static let dimSlice = 64

    /// The number of neighbouring head-width columns one lane loads at a
    /// time. The head width must be a multiple of it.
    private static let columnsPerLoad = 2

    /// The fewest threadgroups a call must give to run the kernel. A call of
    /// fewer threadgroups leaves most of the GPU idle, and the op path reads
    /// it faster: a decode step of 64 heads gives 4 threadgroups, and a
    /// prefill step of 2048 tokens gives 8192.
    private static let fewestThreadgroups = 256

    /// The axis of the `(batch, heads, tokens, headDim)` queries that holds
    /// the heads.
    private static let headAxis = 1

    /// The axis of the queries that holds the tokens, and the axis of the
    /// `(batch, 1, windowKeys, headDim)` window keys that holds the keys.
    private static let tokenAxis = 2

    /// The axis of the `(batch, chunks, headDim)` pool that holds the chunks.
    private static let chunkAxis = 1

    /// The axis of the `(batch, tokens, picks)` picks that holds the picks of
    /// one query.
    private static let pickAxis = 2

    /// The place of each value in the `scalars` input of the kernel.
    private enum ScalarSlot {
        /// The factor each score takes before the softmax.
        static let scale = 0
        /// The number of values.
        static let count = 1
    }

    /// The Metal header.
    ///
    /// `lane_coordinate` gives the row and the first column of the two
    /// elements one lane holds in an 8x8 SIMD-group matrix, the layout MLX
    /// reads in `BaseMMAFrag::get_coord`. The kernel fills the matrices of
    /// the queries and of the keys one lane at a time with it, because the
    /// keys of a tile stand at rows that an index gives.
    ///
    /// `load_pair` reads two neighbouring values as one load. A bfloat16 pair
    /// is one 32-bit word, and each half moves to the high half of a float.
    /// The template form reads any other pointer one value at a time.
    ///
    /// The three pool types answer how many pooled rows a query reads and
    /// which chunk each one is: none, every chunk that stands behind the
    /// query, or the chunks a ranking picked. A row of `-1` takes no weight.
    ///
    /// `attend_query` holds the whole method for one query token and one
    /// block of heads. One key tile holds one key for each lane of a SIMD
    /// group. The pointer types are template arguments, because an input of
    /// fewer values than ``DeepSeekV4MetalKernel/smallestDeviceInput`` is in
    /// the `constant` address space.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header = """
        constexpr constant short matrixSide = 8;
        constexpr constant short elementsPerLane = 2;
        constexpr constant short lanesPerQuad = 4;
        constexpr constant short quadRowBit = 4;
        constexpr constant short quadColumnBit = 2;
        constexpr constant uint bfloat16Bits = 16;
        constexpr constant uint highHalfMask = 0xFFFF0000u;

        inline short2 lane_coordinate(ushort lane) {
            short quad = lane / lanesPerQuad;
            short row = (quad & quadRowBit) + (lane / elementsPerLane) % lanesPerQuad;
            short column = (quad & quadColumnBit) * elementsPerLane + (lane % elementsPerLane) * elementsPerLane;
            return short2(column, row);
        }

        inline float2 load_pair(const device float* source, size_t index) {
            return *(const device float2*)(source + index);
        }

        inline float2 load_pair(const device half* source, size_t index) {
            return float2(*(const device half2*)(source + index));
        }

        inline float2 load_pair(const device bfloat16_t* source, size_t index) {
            uint word = *(const device uint*)(source + index);
            return float2(as_type<float>(word << bfloat16Bits), as_type<float>(word & highHalfMask));
        }

        template <typename Source>
        inline float2 load_pair(Source source, size_t index) {
            return float2(float(source[index]), float(source[index + 1]));
        }

        struct NoPool {
            inline uint count(int) const { return 0; }
            inline int row(uint, int) const { return -1; }
        };

        struct EveryVisiblePool {
            uint chunks;
            int chunkWidth;
            inline uint count(int position) const {
                return min(chunks, uint((position + 1) / chunkWidth));
            }
            inline int row(uint pick, int) const { return int(pick); }
        };

        template <typename Picks>
        struct PickedPool {
            Picks picks;
            size_t first;
            uint pickCount;
            uint chunks;
            int chunkWidth;
            inline uint count(int) const { return pickCount; }
            inline int row(uint pick, int position) const {
                int chunk = int(picks[first + pick]);
                bool readable = chunk >= 0 && uint(chunk) < chunks
                    && (chunk + 1) * chunkWidth <= position + 1;
                return readable ? chunk : -1;
            }
        };

        template <typename Picks>
        inline PickedPool<Picks> picked_pool(
            Picks picks, size_t first, uint pickCount, uint chunks, int chunkWidth
        ) {
            return {picks, first, pickCount, chunks, chunkWidth};
        }

        struct QueryPlace {
            uint sequence;
            uint token;
            uint tokens;
            uint headFirst;
            int position;
            uint windowFirst;
            uint windowCount;
            size_t windowBase;
            size_t poolBase;
        };

        struct KeyRow {
            int row;
            int pooled;
        };

        template <typename Pool>
        inline KeyRow key_row(uint key, thread const QueryPlace& place, thread const Pool& rows, uint total) {
            if (key < place.windowCount) {
                return {int(place.windowFirst + key), 0};
            }
            if (key >= total) {
                return {-1, 0};
            }
            return {rows.row(key - place.windowCount, place.position), 1};
        }

        inline KeyRow key_of_lane(KeyRow mine, ushort lane) {
            return {simd_shuffle(mine.row, lane), simd_shuffle(mine.pooled, lane)};
        }

        template <int DIM, typename WindowRows, typename PoolRows>
        inline simdgroup_float8x8 key_matrix(
            WindowRows window, PoolRows pool, thread const QueryPlace& place, KeyRow key, uint column
        ) {
            float2 pair = float2(0.0f);
            if (key.row >= 0 && column < DIM) {
                size_t index = (key.pooled ? place.poolBase : place.windowBase) + size_t(key.row) * DIM + column;
                pair = key.pooled ? load_pair(pool, index) : load_pair(window, index);
            }
            simdgroup_float8x8 matrix;
            matrix.thread_elements()[0] = pair.x;
            matrix.thread_elements()[1] = pair.y;
            return matrix;
        }

        template <
            typename T, int HEADS, int DIM, int HEAD_BLOCK, int DIM_SLICE, int SIMD_GROUPS, int SIMD_WIDTH,
            bool HAS_SINKS, typename Queries, typename WindowKeys, typename WindowValues, typename PoolKeys,
            typename Pool, typename Sinks>
        inline void attend_query(
            Queries queries, WindowKeys windowKeys, WindowValues windowValues, PoolKeys pool,
            thread const Pool& rows, Sinks sinks, float scale, thread const QueryPlace& place,
            device T* output, threadgroup float* partial, threadgroup float* weights,
            threadgroup float* rowMax, threadgroup float* rowSum, threadgroup float* correction,
            ushort simdGroup, ushort lane
        ) {
            constexpr int headFrags = HEAD_BLOCK / matrixSide;
            constexpr int dimFrags = DIM_SLICE / matrixSide;
            constexpr int keyFrags = SIMD_WIDTH / matrixSide;
            constexpr int tileScores = SIMD_WIDTH * HEAD_BLOCK;
            static_assert(HEAD_BLOCK % matrixSide == 0, "whole matrices of heads");
            static_assert(DIM_SLICE % matrixSide == 0, "whole matrices of the head width");
            static_assert(SIMD_WIDTH % matrixSide == 0, "whole matrices of keys");

            short2 at = lane_coordinate(lane);
            uint dimFirst = uint(simdGroup) * DIM_SLICE;
            uint threadIndex = uint(simdGroup) * SIMD_WIDTH + lane;
            for (uint head = threadIndex; head < HEAD_BLOCK; head += SIMD_GROUPS * SIMD_WIDTH) {
                bool hasSink = HAS_SINKS && place.headFirst + head < HEADS;
                rowMax[head] = hasSink ? float(sinks[place.headFirst + head]) : -INFINITY;
                rowSum[head] = hasSink ? 1.0f : 0.0f;
            }

            simdgroup_float8x8 queryT[dimFrags][headFrags];
            simdgroup_float8x8 out[headFrags][dimFrags];
            #pragma clang loop unroll(full)
            for (int d = 0; d < dimFrags; d++) {
                uint column = dimFirst + d * matrixSide + at.y;
                #pragma clang loop unroll(full)
                for (int r = 0; r < headFrags; r++) {
                    uint head = place.headFirst + r * matrixSide + at.x;
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < elementsPerLane; e++) {
                        float value = 0.0f;
                        if (head + e < HEADS && column < DIM) {
                            size_t index = ((size_t(place.sequence) * HEADS + head + e) * place.tokens + place.token) * DIM + column;
                            value = float(queries[index]) * scale;
                        }
                        queryT[d][r].thread_elements()[e] = value;
                    }
                    out[r][d] = make_filled_simdgroup_matrix<float, matrixSide>(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint total = place.windowCount + rows.count(place.position);
            for (uint tileFirst = 0; tileFirst < total; tileFirst += SIMD_WIDTH) {
                KeyRow mine = key_row(tileFirst + lane, place, rows, total);

                #pragma clang loop unroll(full)
                for (int k = 0; k < keyFrags; k++) {
                    KeyRow key = key_of_lane(mine, ushort(k * matrixSide + at.y));
                    simdgroup_float8x8 scores[headFrags];
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < headFrags; r++) {
                        scores[r] = make_filled_simdgroup_matrix<float, matrixSide>(0.0f);
                    }
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < dimFrags; d++) {
                        simdgroup_float8x8 keys = key_matrix<DIM>(
                            windowKeys, pool, place, key, dimFirst + d * matrixSide + at.x);
                        #pragma clang loop unroll(full)
                        for (int r = 0; r < headFrags; r++) {
                            simdgroup_multiply_accumulate(scores[r], keys, queryT[d][r], scores[r]);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < headFrags; r++) {
                        simdgroup_store(
                            scores[r],
                            partial + simdGroup * tileScores + k * matrixSide * HEAD_BLOCK + r * matrixSide,
                            HEAD_BLOCK);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int head = simdGroup; head < HEAD_BLOCK; head += SIMD_GROUPS) {
                    float score = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int group = 0; group < SIMD_GROUPS; group++) {
                        score += partial[group * tileScores + lane * HEAD_BLOCK + head];
                    }
                    score = mine.row >= 0 ? score : -INFINITY;
                    float previous = rowMax[head];
                    float largest = max(previous, simd_max(score));
                    bool seen = largest > -INFINITY;
                    float weight = seen ? exp(score - largest) : 0.0f;
                    float factor = seen ? exp(previous - largest) : 1.0f;
                    float tileSum = simd_sum(weight);
                    weights[lane * HEAD_BLOCK + head] = weight;
                    if (lane == 0) {
                        rowMax[head] = largest;
                        rowSum[head] = rowSum[head] * factor + tileSum;
                        correction[head] = factor;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma clang loop unroll(full)
                for (int r = 0; r < headFrags; r++) {
                    float factor = correction[r * matrixSide + at.y];
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < dimFrags; d++) {
                        out[r][d].thread_elements() *= factor;
                    }
                }
                #pragma clang loop unroll(full)
                for (int k = 0; k < keyFrags; k++) {
                    simdgroup_float8x8 weightMatrix[headFrags];
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < headFrags; r++) {
                        simdgroup_load(
                            weightMatrix[r], weights + k * matrixSide * HEAD_BLOCK + r * matrixSide,
                            HEAD_BLOCK, ulong2(0, 0), true);
                    }
                    KeyRow key = key_of_lane(mine, ushort(k * matrixSide + at.y));
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < dimFrags; d++) {
                        simdgroup_float8x8 values = key_matrix<DIM>(
                            windowValues, pool, place, key, dimFirst + d * matrixSide + at.x);
                        #pragma clang loop unroll(full)
                        for (int r = 0; r < headFrags; r++) {
                            simdgroup_multiply_accumulate(out[r][d], weightMatrix[r], values, out[r][d]);
                        }
                    }
                }
            }

            #pragma clang loop unroll(full)
            for (int r = 0; r < headFrags; r++) {
                uint head = place.headFirst + r * matrixSide + at.y;
                if (head >= HEADS) {
                    continue;
                }
                float inverse = 1.0f / rowSum[r * matrixSide + at.y];
                size_t rowBase = ((size_t(place.sequence) * HEADS + head) * place.tokens + place.token) * DIM;
                #pragma clang loop unroll(full)
                for (int d = 0; d < dimFrags; d++) {
                    uint column = dimFirst + d * matrixSide + at.x;
                    if (column >= DIM) {
                        continue;
                    }
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < elementsPerLane; e++) {
                        output[rowBase + column + e] = T(out[r][d].thread_elements()[e] * inverse);
                    }
                }
            }
        }

        """

    /// The start of every kernel body: the threadgroup memory, and the place
    /// of the query token and of the head block this threadgroup reads.
    ///
    /// A pool setup follows it and names `rows`, `position`, `poolBase` and
    /// `poolKeys`; ``bodyEnd`` then makes the call.
    private static let bodyStart = """
        constexpr int headBlocks = (HEADS + HEAD_BLOCK - 1) / HEAD_BLOCK;
        threadgroup float partial[SIMD_GROUPS * SIMD_WIDTH * HEAD_BLOCK];
        threadgroup float weights[SIMD_WIDTH * HEAD_BLOCK];
        threadgroup float rowMax[HEAD_BLOCK];
        threadgroup float rowSum[HEAD_BLOCK];
        threadgroup float correction[HEAD_BLOCK];

        uint sequence = threadgroup_position_in_grid.z;
        uint token = threadgroup_position_in_grid.y / headBlocks;
        uint headFirst = (threadgroup_position_in_grid.y % headBlocks) * HEAD_BLOCK;
        uint tokens = queries_shape[TOKEN_AXIS];
        uint windowLength = windowKeys_shape[TOKEN_AXIS];
        uint queryRow = windowLength - tokens + token;
        uint windowFirst = queryRow + 1 > uint(WINDOW) ? queryRow + 1 - WINDOW : 0;

        """

    /// The end of every kernel body: the call of `attend_query`.
    private static let bodyEnd = """

        QueryPlace place = {
            sequence, token, tokens, headFirst, position, windowFirst, queryRow - windowFirst + 1,
            size_t(sequence) * windowLength * DIM, poolBase};
        attend_query<T, HEADS, DIM, HEAD_BLOCK, DIM_SLICE, SIMD_GROUPS, SIMD_WIDTH, HAS_SINKS>(
            queries, windowKeys, windowValues, poolKeys, rows, sinks, scalars[SCALE_SLOT], place,
            output, partial, weights, rowMax, rowSum, correction,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    /// The pool setup of a call that reads the window alone. The pool keys
    /// name the window keys, which no row reads.
    private static let windowOnlySetup = """
        NoPool rows;
        int position = 0;
        size_t poolBase = 0;
        auto poolKeys = windowKeys;
        """

    /// The pool setup of a call that reads every chunk behind each query.
    private static let everyVisibleSetup = """
        uint chunks = pool_shape[CHUNK_AXIS];
        EveryVisiblePool rows = {chunks, CHUNK_WIDTH};
        int position = offsets[0] + int(token);
        size_t poolBase = size_t(sequence) * chunks * DIM;
        auto poolKeys = pool;
        """

    /// The pool setup of a call that reads the chunks each query picked.
    private static let pickedSetup = """
        uint chunks = pool_shape[CHUNK_AXIS];
        uint pickCount = picked_shape[PICK_AXIS];
        auto rows = picked_pool(
            picked, (size_t(sequence) * tokens + token) * pickCount, pickCount, chunks, CHUNK_WIDTH);
        int position = offsets[0] + int(token);
        size_t poolBase = size_t(sequence) * chunks * DIM;
        auto poolKeys = pool;
        """

    /// Builds one compiled kernel. MLX compiles one specialization for each
    /// set of template arguments and keeps it.
    ///
    /// - Parameters:
    ///   - name: The kernel name.
    ///   - poolInputs: The names of the inputs after the common five.
    ///   - setup: The pool setup of the body.
    /// - Returns: The kernel.
    private static func makeKernel(
        name: String, poolInputs: [String], setup: String
    ) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: name,
            inputNames: ["queries", "windowKeys", "windowValues", "sinks", "scalars"] + poolInputs,
            outputNames: ["output"],
            source: bodyStart + setup + bodyEnd,
            header: header)
    }

    /// The kernel of a call that reads the window alone.
    private static let windowOnlyKernel = makeKernel(
        name: "deepseek_v4_window_attention", poolInputs: [], setup: windowOnlySetup)

    /// The kernel of a call that reads every chunk behind each query.
    private static let everyVisibleKernel = makeKernel(
        name: "deepseek_v4_pooled_attention", poolInputs: ["pool", "offsets"],
        setup: everyVisibleSetup)

    /// The kernel of a call that reads the chunks each query picked.
    private static let pickedKernel = makeKernel(
        name: "deepseek_v4_sparse_attention", poolInputs: ["pool", "offsets", "picked"],
        setup: pickedSetup)

    /// Tells if the kernel runs a call: the default device is the GPU, the
    /// paired loads can read the head width, and the call gives at least
    /// ``fewestThreadgroups`` threadgroups, one for each query token and
    /// block of heads.
    ///
    /// - Parameter queries: The queries of the call, shape
    ///   `(batch, heads, tokens, headDim)`.
    /// - Returns: True when the kernel runs the call; false when the call
    ///   keeps the op path.
    static func canRun(queries: MLXArray) -> Bool {
        let threadgroups =
            queries.dim(0) * queries.dim(tokenAxis) * headBlocks(of: queries.dim(headAxis))
        return DeepSeekV4MetalKernel.deviceIsGPU
            && queries.dim(DeepSeekV4MetalKernel.widthAxis) % columnsPerLoad == 0
            && threadgroups >= fewestThreadgroups
    }

    /// The number of threadgroups that read the heads of one query token.
    ///
    /// - Parameter heads: The number of query heads.
    /// - Returns: The number of blocks of ``headBlock`` heads that cover them.
    private static func headBlocks(of heads: Int) -> Int {
        DeepSeekV4MetalKernel.groupCount(covering: heads, groupSize: headBlock)
    }

    /// The attention output of each query over its window rows, its pooled
    /// rows and its sink.
    ///
    /// - Parameters:
    ///   - queries: The rotated queries, shape `(batch, heads, tokens, headDim)`.
    ///   - window: The window rows of the block.
    ///   - pooled: The pooled chunks the block reads, or `nil` for a layer
    ///     with no compressor.
    ///   - scale: The factor each score takes before the softmax.
    ///   - sinks: The learned sink of each head, shape `(heads)`, or `nil`.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    static func attention(
        queries: MLXArray, window: WindowRows, pooled: PooledRows?, scale: Float,
        sinks: MLXArray?
    ) -> MLXArray {
        let batch = queries.dim(0)
        let heads = queries.dim(headAxis)
        let tokens = queries.dim(tokenAxis)
        let headDim = queries.dim(DeepSeekV4MetalKernel.widthAxis)
        let dtype = queries.dtype
        precondition(
            headDim % columnsPerLoad == 0,
            "the kernel reads heads of a multiple of \(columnsPerLoad) numbers, not \(headDim)")
        precondition(
            window.keys.dim(tokenAxis) >= tokens && window.values.shape == window.keys.shape,
            "the window holds the \(tokens) keys of the block and one value for each key, and "
                + "keys \(window.keys.shape) with values \(window.values.shape) do not")
        var scalars = [Float](repeating: 0, count: ScalarSlot.count)
        scalars[ScalarSlot.scale] = scale
        let simdGroups = DeepSeekV4MetalKernel.groupCount(covering: headDim, groupSize: dimSlice)
        let common = [
            queries, window.keys.asType(dtype), window.values.asType(dtype),
            sinks?.asType(dtype) ?? MLXArray.zeros([heads], dtype: dtype), MLXArray(scalars),
        ]
        let (kernel, poolInputs) = Self.kernel(for: pooled, dtype: dtype)
        let simdGroupWidth = DeepSeekV4MetalKernel.simdGroupWidth
        let threads = simdGroups * simdGroupWidth
        return kernel(
            common + poolInputs,
            template: [
                ("T", dtype), ("HEADS", heads), ("DIM", headDim), ("WINDOW", window.window),
                ("CHUNK_WIDTH", pooled?.chunkWidth ?? 1), ("HAS_SINKS", sinks != nil),
                ("HEAD_BLOCK", headBlock), ("DIM_SLICE", dimSlice), ("SIMD_GROUPS", simdGroups),
                ("SIMD_WIDTH", simdGroupWidth), ("SCALE_SLOT", ScalarSlot.scale),
                ("TOKEN_AXIS", tokenAxis), ("CHUNK_AXIS", chunkAxis), ("PICK_AXIS", pickAxis),
            ],
            grid: (threads, tokens * headBlocks(of: heads), batch),
            threadGroup: (threads, 1, 1),
            outputShapes: [queries.shape],
            outputDTypes: [dtype])[0]
    }

    /// The kernel one call reads and the pool inputs it takes.
    ///
    /// - Parameters:
    ///   - pooled: The pooled chunks of the call, or `nil`.
    ///   - dtype: The dtype of the queries, which the pool takes.
    /// - Returns: The kernel, and its inputs after the common five.
    private static func kernel(
        for pooled: PooledRows?, dtype: DType
    ) -> (MLXFast.MLXFastKernel, [MLXArray]) {
        guard let pooled else {
            return (windowOnlyKernel, [])
        }
        precondition(pooled.pool.dim(chunkAxis) > 0, "a pool holds one chunk or more")
        let poolInputs = [pooled.pool.asType(dtype), MLXArray([Int32(pooled.offset)])]
        guard let picked = pooled.picked else {
            return (everyVisibleKernel, poolInputs)
        }
        return (pickedKernel, poolInputs + [picked.asType(.int32)])
    }
}
