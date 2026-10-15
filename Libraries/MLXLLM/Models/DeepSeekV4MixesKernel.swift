// Copyright © 2026 Apple Inc.
//
// The fused Metal kernel of the mixes projection of the DeepSeek-V4
// hyper-connection collapse.
//
// The layout follows the GLM-5.3 kernels of osaurus-ai/vmlx-swift,
// Libraries/MLXLLM/Models/MHCFusedDecode.swift (commit 01abc39f) and
// MHCFusedPrefill.swift (commit f8130127): the kernel reads the residual
// stream in its own dtype, adds the sum of squares of the reduction and the
// dot products of the projection in one pass, and multiplies each dot product
// by the inverse RMS at the end. Manual transcription; no git ancestry.
//
// Two details do not come from those kernels:
//
//  1. The vmlx-swift kernels take a fixed count of four copies and a fixed
//     width of the mixes vector. This kernel takes the width of the flattened
//     copies, the width of the mixes vector and the dtypes as template
//     arguments.
//  2. The vmlx-swift decode kernel gives one mix row to each threadgroup, and
//     its prefill kernel gives two tokens to each threadgroup. This kernel
//     takes one source with two layouts. The layout of a decode step gives a
//     few mix rows of one token to each threadgroup, thus the threadgroups
//     split the rows of the projection. The layout of a prefill block gives
//     one token to each lane of a SIMD group, thus the lanes of one SIMD
//     group read the same projection value and each projection value goes
//     to 32 tokens with one read. A scan over the layouts at 32k context on
//     an M3 Ultra measured 0.022 ms for one decode token and 0.38 ms for a
//     block of 2048 tokens, against 0.036 ms and 1.13 ms for the op path.

import Foundation
import MLX

/// One Metal kernel for the RMS reduction and the mixes projection of one
/// DeepSeek-V4 hyper-connection collapse.
///
/// As separate MLX operations, the projection is a float32 copy of the
/// stream, the RMS norm and the matrix multiply: three dispatches that each
/// read and write the whole float32 stream. The kernel reads the stream one
/// time in its own dtype, and it writes only the mixes vector.
///
/// The kernel multiplies the sum of the dot products by the inverse RMS at
/// the end, where the op path multiplies each value of the stream by it
/// first, and it adds the products in another order than the MLX matrix
/// multiply. Thus its answer is not the bits of the op path: the two differ
/// in the last float32 places.
enum DeepSeekV4MixesKernel {

    /// The arrangement of tokens, mix rows and threads in one threadgroup.
    private struct Layout {

        /// The number of tokens of one threadgroup, one in each group of
        /// `simdGroupWidth / tokensPerGroup` lanes. It divides the width of
        /// one SIMD group.
        let tokensPerGroup: Int

        /// The number of rows of the projection one threadgroup reads.
        let rowsPerGroup: Int

        /// The number of threads of one threadgroup.
        let threads: Int
    }

    /// The number of mix rows one threadgroup of a decode step reads. The 24
    /// rows of DeepSeek-V4-Flash thus take 12 threadgroups.
    private static let stepRowsPerGroup = 2

    /// The number of threads of one threadgroup of a decode step.
    private static let stepThreads = 512

    /// The number of mix rows one threadgroup of a prefill block reads.
    private static let blockRowsPerGroup = 12

    /// The number of threads of one threadgroup of a prefill block.
    private static let blockThreads = 256

    /// The layout of a decode step: one token in each threadgroup, and the
    /// threadgroups split the rows of the projection.
    private static let stepLayout = Layout(
        tokensPerGroup: 1, rowsPerGroup: stepRowsPerGroup, threads: stepThreads)

    /// The layout of a prefill block: one token in each lane of a SIMD group.
    private static let blockLayout = Layout(
        tokensPerGroup: simdGroupWidth, rowsPerGroup: blockRowsPerGroup, threads: blockThreads)

    /// The smallest number of tokens that takes ``blockLayout``. The scan
    /// measured the two layouts equal at about 256 tokens.
    static let blockLayoutMinimumTokens = 256

    /// The number of lanes in one Apple GPU SIMD group.
    private static let simdGroupWidth = 32

    /// The number of values the kernel reads in one vector load.
    private static let vectorWidth = 4

    /// The smallest input MLX gives a kernel in the `device` address space.
    /// MLX gives an input of fewer values in the `constant` address space,
    /// and the vector loads of the kernel need a `device` pointer.
    private static let smallestDeviceInput = 8

    /// The number of trailing axes of the residual stream that one token
    /// holds: the copy axis and the width axis.
    private static let tokenAxisCount = 2

    /// The Metal source. The template arguments are `T`, the dtype of the
    /// stream; `W`, the dtype of the projection; `FEATURES`, the width of the
    /// flattened copies of one token; `MIX_WIDTH`, the width of the mixes
    /// vector; `TOKENS_PER_GROUP`, `ROWS_PER_GROUP` and `THREADS`, the
    /// layout; `SIMD_WIDTH`, the number of lanes in one SIMD group;
    /// `VECTOR_WIDTH`, the number of values of one vector load; and
    /// `EPS_BITS`, the bit pattern of the float32 epsilon of the reduction.
    /// The input `tokens` holds the number of tokens of the stream.
    ///
    /// Each SIMD group reads its own span of the features. In the span, lane
    /// `l` takes token `l % TOKENS_PER_GROUP` and each
    /// `SIMD_WIDTH / TOKENS_PER_GROUP`-th vector from place
    /// `l / TOKENS_PER_GROUP`. A lane adds the squares of its values and the
    /// dot product of its values with each projection row. The XOR shuffles
    /// then add the lanes of one token, the first `TOKENS_PER_GROUP` lanes
    /// write the sums of each SIMD group into threadgroup memory, and after
    /// the barrier the threads add the SIMD groups and write the mixes.
    ///
    /// A lane past the last token reads the last token and writes nothing,
    /// and a row past the last mix row reads the last row and writes
    /// nothing, thus each lane runs every shuffle.
    private static let source = """
        constexpr uint rowGroups = (MIX_WIDTH + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
        constexpr uint simdGroups = THREADS / SIMD_WIDTH;
        constexpr uint lanesPerToken = SIMD_WIDTH / TOKENS_PER_GROUP;
        constexpr uint vectors = FEATURES / VECTOR_WIDTH;
        constexpr uint span = (vectors + simdGroups - 1) / simdGroups;
        constexpr uint squareSlot = ROWS_PER_GROUP;
        constexpr uint lastMixRow = MIX_WIDTH - 1;
        typedef vec<float, VECTOR_WIDTH> floatVector;
        threadgroup float partial[simdGroups][ROWS_PER_GROUP + 1][TOKENS_PER_GROUP];

        uint group = threadgroup_position_in_grid.x;
        uint firstToken = (group / rowGroups) * TOKENS_PER_GROUP;
        uint firstRow = (group % rowGroups) * ROWS_PER_GROUP;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        uint slot = lane % TOKENS_PER_GROUP;
        uint tokenCount = tokens[0];
        uint token = min(firstToken + slot, tokenCount - 1);
        const device vec<T, VECTOR_WIDTH>* values =
            (const device vec<T, VECTOR_WIDTH>*)stream + token * vectors;
        const device vec<W, VECTOR_WIDTH>* rows = (const device vec<W, VECTOR_WIDTH>*)projection;

        float sums[ROWS_PER_GROUP];
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            sums[r] = 0.0f;
        }
        float squares = 0.0f;
        uint last = min((simd + 1) * span, vectors);
        for (uint v = simd * span + lane / TOKENS_PER_GROUP; v < last; v += lanesPerToken) {
            floatVector value = floatVector(values[v]);
            squares += dot(value, value);
            for (uint r = 0; r < ROWS_PER_GROUP; r++) {
                uint row = min(firstRow + r, lastMixRow);
                sums[r] += dot(value, floatVector(rows[row * vectors + v]));
            }
        }
        for (uint offset = TOKENS_PER_GROUP; offset < SIMD_WIDTH; offset <<= 1) {
            squares += simd_shuffle_xor(squares, (ushort)offset);
            for (uint r = 0; r < ROWS_PER_GROUP; r++) {
                sums[r] += simd_shuffle_xor(sums[r], (ushort)offset);
            }
        }
        if (lane < TOKENS_PER_GROUP) {
            for (uint r = 0; r < ROWS_PER_GROUP; r++) {
                partial[simd][r][lane] = sums[r];
            }
            partial[simd][squareSlot][lane] = squares;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float eps = as_type<float>(EPS_BITS);
        for (uint i = thread_index_in_threadgroup; i < ROWS_PER_GROUP * TOKENS_PER_GROUP;
             i += THREADS) {
            uint s = i % TOKENS_PER_GROUP;
            uint r = i / TOKENS_PER_GROUP;
            uint t = firstToken + s;
            uint row = firstRow + r;
            if (t < tokenCount && row < MIX_WIDTH) {
                float total = 0.0f;
                float square = 0.0f;
                for (uint g = 0; g < simdGroups; g++) {
                    total += partial[g][r][s];
                    square += partial[g][squareSlot][s];
                }
                mixes[t * MIX_WIDTH + row] =
                    total * metal::precise::rsqrt(square / FEATURES + eps);
            }
        }
        """

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_hyper_connection_mixes",
        inputNames: ["stream", "projection", "tokens"],
        outputNames: ["mixes"],
        source: source)

    /// Tells if the kernel can take the mixes of `stream`.
    ///
    /// The kernel is a Metal kernel, thus it runs only when the default
    /// device is the GPU. It reads the flattened copies of one token in
    /// vector loads through a `device` pointer, thus the flattened width must
    /// be a multiple of the vector width and the stream must hold enough
    /// values for MLX to give it in the `device` address space.
    ///
    /// - Parameter stream: The residual stream, shape `(..., copies, width)`.
    /// - Returns: `true` when ``mixes(of:projection:normEps:)`` can take the
    ///   stream.
    static func canRun(on stream: MLXArray) -> Bool {
        let features = stream.shape.suffix(tokenAxisCount).reduce(1, *)
        return Device.defaultDevice().deviceType == .gpu && stream.size >= smallestDeviceInput
            && features % vectorWidth == 0
    }

    /// Takes the RMS reduction of the flattened copies of each token and the
    /// mixes projection in one dispatch.
    ///
    /// The answer is `(x @ projection^T) * rsqrt(mean(x * x) + normEps)` for
    /// the flattened copies `x` of each token, in float32. That is the answer
    /// of ``ManifoldStream/mixes(of:projection:normWeight:normEps:)`` with a
    /// weight of every value 1, within the last float32 places.
    ///
    /// - Parameters:
    ///   - stream: The residual stream, shape `(..., copies, width)`, in any
    ///     float dtype. ``canRun(on:)`` must accept it.
    ///   - projection: The mixes projection, shape
    ///     `(mixes, copies * width)`.
    ///   - normEps: The epsilon of the reduction, from `rms_norm_eps`.
    /// - Returns: The mixes vector, shape `(..., mixes)`, in float32.
    static func mixes(of stream: MLXArray, projection: MLXArray, normEps: Float) -> MLXArray {
        let leading = Array(stream.shape.dropLast(tokenAxisCount))
        let features = stream.shape.suffix(tokenAxisCount).reduce(1, *)
        let mixWidth = projection.dim(0)
        precondition(
            projection.shape == [mixWidth, features],
            "the projection must be shape \([mixWidth, features]), and it is \(projection.shape)")
        let tokens = leading.reduce(1, *)
        let layout = tokens >= blockLayoutMinimumTokens ? blockLayout : stepLayout
        let tokenGroups = (tokens + layout.tokensPerGroup - 1) / layout.tokensPerGroup
        let rowGroups = (mixWidth + layout.rowsPerGroup - 1) / layout.rowsPerGroup
        return kernel(
            [stream, projection, MLXArray([Int32(tokens)])],
            template: [
                ("T", stream.dtype), ("W", projection.dtype), ("FEATURES", features),
                ("MIX_WIDTH", mixWidth), ("TOKENS_PER_GROUP", layout.tokensPerGroup),
                ("ROWS_PER_GROUP", layout.rowsPerGroup), ("THREADS", layout.threads),
                ("SIMD_WIDTH", simdGroupWidth), ("VECTOR_WIDTH", vectorWidth),
                ("EPS_BITS", normEps.templateBits),
            ],
            grid: (tokenGroups * rowGroups * layout.threads, 1, 1),
            threadGroup: (layout.threads, 1, 1),
            outputShapes: [leading + [mixWidth]],
            outputDTypes: [.float32])[0]
    }
}
