// Copyright © 2026 Apple Inc.
//
// One Metal kernel for the query and key norms, the M-RoPE and the head layout
// of one Qwen 3.5 decode token.
//
// A decode step of Qwen 3.5 costs CPU time and GPU dispatches for each
// primitive (task ^6jf4g0c): the CPU builds and encodes each one, and each
// dispatch waits for the one before it. From the `q_proj` and `k_proj`
// outputs, the op path of one token is a split of each query from its output
// gate, an RMS norm of the queries, an RMS norm of the keys, the swap of the
// token axis and the head axis of each part, and the M-RoPE of the queries and
// of the keys: a slice of the rotated numbers, two products, the half turn
// (two slices, a negation and a concatenation), a sum and the concatenation of
// the numbers past the rotation, for each part. This kernel does all of it in
// one dispatch, and reads the cosines and the sines the step computes once for
// every layer.
//
// The kernel gives the SAME BITS as those MLX kernels. SwiftPM compiles the
// MLX kernels with the fast math of the Metal compiler, which contracts a
// multiply and an add into one fused multiply-add, and MLX compiles this
// kernel at run time without fast math. Thus the source writes each fused
// multiply-add the MLX kernels round with as an explicit `fma`, and each
// rounding of an MLX kernel as a value of the dtype `T`:
//
//  - `rms_single_row`, for a row of `headDim` numbers: one thread for each
//    4 neighbouring numbers, thus one SIMD group of 32 threads for each
//    128 numbers. Each thread adds the squares of its numbers with `fma`,
//    `simd_sum` adds the threads of each SIMD group, and a second `simd_sum`
//    adds the sums of the SIMD groups from the lanes of one SIMD group, with
//    zero in every other lane. The inverse is
//    `precise::rsqrt(sum / width + eps)`, and the output is
//    `weight * T(x * inverse)`, rounded in the dtype `T`. One SIMD group of
//    this kernel reads the numbers of each SIMD group of the MLX kernel in
//    turn, in the same lanes, and adds them the same way.
//  - The M-RoPE of `applyMultimodalRotaryPosEmb`: each MLX operation rounds
//    in `T`, thus number `d` of the rotated part turns to
//    `T(x[d] * cos[d]) + T(halfTurn[d] * sin[d])`, with `halfTurn[d] = -x[d + R/2]`
//    in the first half of the `R` rotated numbers and `x[d - R/2]` in the
//    second half. The numbers past `R` keep their values.
//
// Tests/MLXLMTests/Qwen35DecodeHeadsKernelTests.swift compares the bits.

import Foundation
import MLX
import MLXLMCommon

/// The query and key norms, the M-RoPE and the head layout of one Qwen 3.5
/// decode token, in one Metal kernel.
///
/// One SIMD group reads one head. Lane `l` holds the numbers
/// `128g + 4l ..< 128g + 4l + 4` of each group `g` of 128 numbers, as thread
/// `32g + l` of the MLX RMS norm does, thus `simd_sum` adds the same values in
/// the same lanes. The rotated numbers lie in the first group, and the M-RoPE
/// pairs number `d` of the first half of them with number `d + R/2` of the
/// second half, which lane `l + R/8` holds; one shuffle exchanges the pair.
struct Qwen35DecodeHeadsKernel {

    /// The numbers one thread of the MLX RMS norm reads (`RMS_N_READS`).
    private static let numbersPerLane = 4

    /// The numbers one SIMD group reads in one pass: one number group for
    /// each lane.
    static let groupWidth = numbersPerLane * metalSimdWidth

    /// The widest row the MLX RMS norm reads in one pass of its threads
    /// (`RMS_LOOPED_LIMIT`); a wider row takes a loop that adds in another
    /// order.
    private static let widestSinglePassRow = 4096

    /// The halves of each head of the `q_proj` output: the query, then the
    /// output gate.
    private static let gatedQueryParts = 2

    /// The halves of the rotated numbers that M-RoPE pairs.
    private static let rotaryHalves = 2

    /// The token count of a decode step.
    private static let decodeTokens = 1

    /// The index of the query heads among the outputs of the kernel.
    private static let queriesOutput = 0

    /// The index of the key heads among the outputs of the kernel.
    private static let keysOutput = 1

    /// The dtypes the kernel reads.
    private static let floatTypes: [DType] = [.float32, .float16, .bfloat16]

    private static let source = """
        constexpr int perLane = PER_LANE;
        constexpr int groupWidth = perLane * SIMD_WIDTH;
        constexpr int groups = DIM / groupWidth;
        constexpr int rotaryLanes = ROTARY_DIM / perLane;
        constexpr int pairLanes = rotaryLanes / 2;
        constexpr int heads = QUERY_HEADS + KV_HEADS;

        uint lane = thread_index_in_simdgroup;
        uint sequence = threadgroup_position_in_grid.y / heads;
        int head = int(threadgroup_position_in_grid.y % heads);
        bool isQuery = head < QUERY_HEADS;
        int kvHead = head - QUERY_HEADS;

        const device T* row = isQuery
            ? queryProjection + (size_t(sequence) * QUERY_HEADS + head) * QUERY_STRIDE
            : keyProjection + (size_t(sequence) * KV_HEADS + kvHead) * DIM;
        const device T* weight = isQuery ? queryWeight : keyWeight;
        float eps = as_type<float>(isQuery ? Q_EPS_BITS : K_EPS_BITS);
        device T* out = isQuery
            ? queries + (size_t(sequence) * QUERY_HEADS + head) * DIM
            : keys + (size_t(sequence) * KV_HEADS + kvHead) * DIM;

        float x[groups][perLane];
        float groupSum = 0;
        for (int g = 0; g < groups; g++) {
            float squares = 0;
            for (int i = 0; i < perLane; i++) {
                x[g][i] = row[g * groupWidth + lane * perLane + i];
                squares = fma(x[g][i], x[g][i], squares);
            }
            squares = simd_sum(squares);
            if (lane == uint(g)) {
                groupSum = squares;
            }
        }
        float inverse = metal::precise::rsqrt(simd_sum(groupSum) / DIM + eps);

        T normed[groups][perLane];
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < perLane; i++) {
                int d = g * groupWidth + lane * perLane + i;
                normed[g][i] = weight[d] * static_cast<T>(x[g][i] * inverse);
            }
        }

        bool rotates = lane < uint(rotaryLanes);
        bool firstHalf = lane < uint(pairLanes);
        ushort partnerLane = ushort(
            !rotates ? lane : firstHalf ? lane + pairLanes : lane - pairLanes);
        const device T* cosines = cosine + size_t(sequence) * ROTARY_DIM;
        const device T* sines = sine + size_t(sequence) * ROTARY_DIM;
        for (int i = 0; i < perLane; i++) {
            T partner = static_cast<T>(
                simd_shuffle(static_cast<float>(normed[0][i]), partnerLane));
            int d = lane * perLane + i;
            if (rotates) {
                T halfTurn = firstHalf ? T(-partner) : partner;
                T turned = normed[0][i] * cosines[d];
                T shifted = halfTurn * sines[d];
                out[d] = turned + shifted;
            } else {
                out[d] = normed[0][i];
            }
        }
        for (int g = 1; g < groups; g++) {
            for (int i = 0; i < perLane; i++) {
                out[g * groupWidth + lane * perLane + i] = normed[g][i];
            }
        }
        """

    /// The query heads of one sequence.
    private let queryHeads: Int

    /// The key heads of one sequence.
    private let kvHeads: Int

    /// The numbers of one head.
    private let headDim: Int

    /// The rotated numbers at the start of each head.
    private let rotaryDim: Int

    /// The kernel of the layer. Its header holds the constants of the
    /// layer, thus a call names only the dtype: each template argument costs
    /// graph-build time on every decode step.
    private let kernel: MLXFast.MLXFastKernel

    /// The kernel of a layer, or `nil` when the kernel does not reproduce
    /// its norms or its rotation: a head that is not a whole number of
    /// ``groupWidth``-number groups, or wider than one pass of the MLX RMS
    /// norm, or a rotation that is odd in its lanes or wider than one group.
    ///
    /// - Parameters:
    ///   - queryHeads: The query heads of one sequence.
    ///   - kvHeads: The key heads of one sequence.
    ///   - headDim: The numbers of one head.
    ///   - rotaryDim: The rotated numbers at the start of each head.
    ///   - queryEps: The epsilon of the query norm.
    ///   - keyEps: The epsilon of the key norm.
    init?(
        queryHeads: Int, kvHeads: Int, headDim: Int, rotaryDim: Int, queryEps: Float,
        keyEps: Float
    ) {
        let rotaryLanes = rotaryDim / Self.numbersPerLane
        guard headDim > 0, headDim.isMultiple(of: Self.groupWidth),
            headDim <= Self.widestSinglePassRow,
            rotaryDim > 0, rotaryDim.isMultiple(of: Self.numbersPerLane),
            rotaryLanes.isMultiple(of: Self.rotaryHalves), rotaryDim <= Self.groupWidth
        else { return nil }
        self.queryHeads = queryHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.rotaryDim = rotaryDim
        let counts: KeyValuePairs = [
            "PER_LANE": Self.numbersPerLane, "DIM": headDim, "QUERY_HEADS": queryHeads,
            "KV_HEADS": kvHeads,
            "ROTARY_DIM": rotaryDim, "QUERY_STRIDE": Self.gatedQueryParts * headDim,
        ]
        let bitPatterns: KeyValuePairs = [
            "Q_EPS_BITS": queryEps.bitPattern, "K_EPS_BITS": keyEps.bitPattern,
        ]
        let header =
            counts.map { "constexpr constant int \($0.key) = \($0.value);\n" }
            + bitPatterns.map { "constexpr constant uint \($0.key) = \($0.value)u;\n" }
        // MLX keeps one library for each kernel name, thus the name holds
        // every constant of the header.
        let name =
            (["qwen35_decode_heads"] + counts.map { "\($0.value)" }
            + bitPatterns.map { "\($0.value)" }).joined(separator: "_")
        self.kernel = MLXFast.metalKernel(
            name: name,
            inputNames: [
                "queryProjection", "keyProjection", "queryWeight", "keyWeight", "cosine", "sine",
            ],
            outputNames: ["queries", "keys"], source: Self.source,
            header: metalSimdWidthDeclaration + header.joined())
    }

    /// The normed and rotated heads of one decode token from its `q_proj`
    /// and `k_proj` outputs, with the bits of the op path, or `nil` when the
    /// kernel cannot read them.
    ///
    /// The kernel runs when the default device is the GPU, the projections
    /// hold one token of the heads of the layer in a float dtype, both norm
    /// weights hold one head of that dtype, and the rotation holds the
    /// rotated numbers of that token in that dtype.
    ///
    /// - Parameters:
    ///   - queryProjection: The `q_proj` output, shape
    ///     `[B, 1, queryHeads * 2 * headDim]`: the query, then the output
    ///     gate, of each head.
    ///   - keyProjection: The `k_proj` output, shape
    ///     `[B, 1, kvHeads * headDim]`.
    ///   - queryWeight: The weight of the RMS norm of each query head.
    ///   - keyWeight: The weight of the RMS norm of each key head.
    ///   - rotation: The M-RoPE cosines and sines of the token, each
    ///     `[B, 1, rotaryDim]`.
    /// - Returns: The queries, `[B, queryHeads, 1, headDim]`, and the keys,
    ///   `[B, kvHeads, 1, headDim]`, or `nil`.
    func heads(
        queryProjection: MLXArray, keyProjection: MLXArray, queryWeight: MLXArray,
        keyWeight: MLXArray, rotation: Qwen35Language.RotaryValues
    ) -> (queries: MLXArray, keys: MLXArray)? {
        let dtype = queryProjection.dtype
        let batchSize = queryProjection.dim(0)
        let widths = [
            (queryProjection, Self.gatedQueryParts * queryHeads * headDim),
            (keyProjection, kvHeads * headDim), (rotation.cos, rotaryDim),
            (rotation.sin, rotaryDim),
        ]
        guard Device.defaultDevice().deviceType == .gpu, Self.floatTypes.contains(dtype),
            widths.allSatisfy({ array, width in
                array.shape == [batchSize, Self.decodeTokens, width] && array.dtype == dtype
            }),
            [queryWeight, keyWeight].allSatisfy({ $0.shape == [headDim] && $0.dtype == dtype })
        else { return nil }
        let shapes = [queryHeads, kvHeads].map {
            [batchSize, $0, Self.decodeTokens, headDim]
        }
        let results = kernel(
            [queryProjection, keyProjection, queryWeight, keyWeight, rotation.cos, rotation.sin],
            template: [("T", dtype)],
            grid: (metalSimdWidth, batchSize * (queryHeads + kvHeads), 1),
            threadGroup: (metalSimdWidth, 1, 1),
            outputShapes: shapes,
            outputDTypes: shapes.map { _ in dtype })
        return (results[Self.queriesOutput], results[Self.keysOutput])
    }
}
