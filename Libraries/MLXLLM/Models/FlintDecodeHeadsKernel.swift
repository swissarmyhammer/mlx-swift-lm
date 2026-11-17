// Copyright © 2026 Apple Inc.
//
// One Metal kernel for the query and key norms, the RoPE and the head layout
// of one Flint decode token.
//
// A decode step of Flint costs CPU time and GPU dispatches for each primitive
// (task ^nanqj4p): the CPU builds and encodes each one, and each dispatch
// waits for the one before it. From the fused `qkv_proj` output, the op path
// of one token is a split into heads, an RMS norm of the queries, an RMS norm
// of the keys, the swap of the token axis and the head axis of each part,
// and the RoPE of the queries and of the keys: four dispatches in a layer
// with RoPE, two in a layer of `no_rope_layer_indices`. This kernel does all
// of it in one dispatch.
//
// The kernel gives the SAME BITS as those MLX kernels. SwiftPM compiles the
// MLX kernels with the fast math of the Metal compiler, which contracts a
// multiply and an add into one fused multiply-add, and MLX compiles this
// kernel at run time without fast math. Thus the source writes each fused
// multiply-add the MLX kernels round with as an explicit `fma`, and the fast
// `exp2` the MLX RoPE reads as `fast::exp2`:
//
//  - `rms_single_row`, for a row of 128 numbers: each of 32 lanes adds the
//    squares of its 4 neighbouring numbers with `fma`, `simd_sum` adds the
//    lanes. The MLX source divides the sum by the width, and its fast math
//    multiplies by the reciprocal of the width instead, thus the inverse is
//    `precise::rsqrt(sum * (1 / width) + eps)`. The output is
//    `weight * T(x * inverse)`, rounded in the dtype `T`.
//  - `rope_single`, not traditional: position `d` of the first half of a
//    head turns by `offset * scale * exp2(-d / half * log2(base))`, with
//    `fma(x1, cos, -(x2 * sin))` for the first half and
//    `fma(x2, cos, x1 * sin)` for the second half.
//
// Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift compares the bits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// How a Flint layer rotates its query and key heads.
enum FlintHeadRotation: Sendable, Equatable {
    /// A layer of `no_rope_layer_indices`: the heads keep their values.
    case unrotated
    /// The RoPE of MLX over the whole head, not traditional, with the base
    /// and the position scale of the layer.
    case rotary(base: Float, scale: Float)
}

/// The query and key norms, the RoPE and the head layout of one Flint decode
/// token, in one Metal kernel.
///
/// One SIMD group reads one head of the projection. Lane `l` holds the
/// numbers `4l ..< 4l + 4` of the head, as a lane of the MLX RMS norm does,
/// thus `simd_sum` adds the same values in the same lanes. The RoPE pairs
/// number `d` of the first half with number `d + 64` of the second half,
/// which lane `l ^ 16` holds, and one XOR shuffle exchanges the pair.
struct FlintDecodeHeadsKernel {

    /// The numbers one lane of the MLX RMS norm reads (`RMS_N_READS`).
    private static let numbersPerLane = 4

    /// The head width the kernel reads: one number group for each lane of
    /// one SIMD group, the width of one MLX RMS norm threadgroup.
    static let headDim = numbersPerLane * metalSimdWidth

    /// The axis of the projection that holds the tokens.
    private static let tokenAxis = 1

    /// The axes of the projection: `[B, L, width]`.
    private static let projectionAxes = 3

    /// The token count of a decode step.
    private static let decodeTokens = 1

    /// The KV projections of a fused `qkv_proj`: the keys and the values.
    private static let kvProjections = 2

    /// The index of the query heads among the outputs of the kernel.
    private static let queriesOutput = 0

    /// The index of the key heads among the outputs of the kernel.
    private static let keysOutput = 1

    /// The index of the value heads among the outputs of the kernel.
    private static let valuesOutput = 2

    /// The dtypes the kernel reads.
    private static let floatTypes: [DType] = [.float32, .float16, .bfloat16]

    private static let source = """
        constexpr int perLane = DIM / SIMD_WIDTH;
        constexpr int halfDim = DIM / 2;
        constexpr int pairLanes = SIMD_WIDTH / 2;
        constexpr int heads = QUERY_HEADS + 2 * KV_HEADS;

        uint lane = thread_index_in_simdgroup;
        uint sequence = threadgroup_position_in_grid.y / heads;
        int head = int(threadgroup_position_in_grid.y % heads);
        const device T* row = projection + (size_t(sequence) * heads + head) * DIM + lane * perLane;

        if (head >= QUERY_HEADS + KV_HEADS) {
            int valueHead = head - QUERY_HEADS - KV_HEADS;
            device T* out = values + (size_t(sequence) * KV_HEADS + valueHead) * DIM + lane * perLane;
            for (int i = 0; i < perLane; i++) {
                out[i] = row[i];
            }
            return;
        }

        bool isQuery = head < QUERY_HEADS;
        const device T* weight = (isQuery ? queryWeight : keyWeight) + lane * perLane;
        float eps = as_type<float>(isQuery ? Q_EPS_BITS : K_EPS_BITS);
        device T* out = isQuery
            ? queries + (size_t(sequence) * QUERY_HEADS + head) * DIM + lane * perLane
            : keys + (size_t(sequence) * KV_HEADS + head - QUERY_HEADS) * DIM + lane * perLane;

        float x[perLane];
        float squares = 0;
        for (int i = 0; i < perLane; i++) {
            x[i] = row[i];
            squares = fma(x[i], x[i], squares);
        }
        squares = simd_sum(squares);
        float inverse = metal::precise::rsqrt(squares * (1.0f / DIM) + eps);
        T normed[perLane];
        for (int i = 0; i < perLane; i++) {
            normed[i] = weight[i] * static_cast<T>(x[i] * inverse);
        }

        if (!ROTATES) {
            for (int i = 0; i < perLane; i++) {
                out[i] = normed[i];
            }
            return;
        }

        bool firstHalf = lane < uint(pairLanes);
        float position = as_type<float>(SCALE_BITS) * static_cast<float>(offset[0]);
        for (int i = 0; i < perLane; i++) {
            float mine = static_cast<float>(normed[i]);
            float partner = simd_shuffle_xor(mine, ushort(pairLanes));
            int pair = int(lane % uint(pairLanes)) * perLane + i;
            float d = static_cast<float>(pair) / static_cast<float>(halfDim);
            float theta = position * metal::fast::exp2(-d * as_type<float>(LOG2_BASE_BITS));
            float cosine = metal::fast::cos(theta);
            float sine = metal::fast::sin(theta);
            float rotated = firstHalf
                ? fma(mine, cosine, -(partner * sine))
                : fma(mine, cosine, partner * sine);
            out[i] = static_cast<T>(rotated);
        }
        """

    /// The query heads of one sequence.
    private let queryHeads: Int

    /// The key heads of one sequence, and the value heads.
    private let kvHeads: Int

    /// The kernel of the layer. Its header holds the constants of the
    /// layer, thus a call names only the dtype: each template argument costs
    /// graph-build time on every decode step.
    private let kernel: MLXFast.MLXFastKernel

    /// The kernel of a layer, or `nil` when its heads are not ``headDim``
    /// numbers wide.
    ///
    /// - Parameters:
    ///   - queryHeads: The query heads of one sequence.
    ///   - kvHeads: The key heads of one sequence, and the value heads.
    ///   - headDim: The numbers of one head.
    ///   - queryEps: The epsilon of the query norm.
    ///   - keyEps: The epsilon of the key norm.
    ///   - rotation: The rotation of the layer.
    init?(
        queryHeads: Int, kvHeads: Int, headDim: Int, queryEps: Float, keyEps: Float,
        rotation: FlintHeadRotation
    ) {
        guard headDim == Self.headDim else { return nil }
        self.queryHeads = queryHeads
        self.kvHeads = kvHeads
        let (base, scale) = rotation.parameters
        let counts: KeyValuePairs = [
            "DIM": Self.headDim, "QUERY_HEADS": queryHeads, "KV_HEADS": kvHeads,
            "ROTATES": rotation == .unrotated ? 0 : 1,
        ]
        let bitPatterns: KeyValuePairs = [
            "Q_EPS_BITS": queryEps.bitPattern, "K_EPS_BITS": keyEps.bitPattern,
            "SCALE_BITS": scale.bitPattern, "LOG2_BASE_BITS": log2(base).bitPattern,
        ]
        let header =
            counts.map { "constexpr constant int \($0.key) = \($0.value);\n" }
            + bitPatterns.map { "constexpr constant uint \($0.key) = \($0.value)u;\n" }
        // MLX keeps one library for each kernel name, thus the name holds
        // every constant of the header.
        let name =
            (["flint_decode_heads"] + counts.map { "\($0.value)" }
            + bitPatterns.map { "\($0.value)" }).joined(separator: "_")
        self.kernel = MLXFast.metalKernel(
            name: name, inputNames: ["projection", "queryWeight", "keyWeight", "offset"],
            outputNames: ["queries", "keys", "values"], source: Self.source,
            header: metalSimdWidthDeclaration + header.joined())
    }

    /// The heads of one decode token from its fused `qkv_proj` projection,
    /// with the bits of the op path, or `nil` when the kernel cannot read
    /// the projection.
    ///
    /// The kernel runs when the default device is the GPU, the projection
    /// holds one token of the heads of the layer in a float dtype, and both
    /// norm weights hold ``headDim`` numbers of that dtype.
    ///
    /// - Parameters:
    ///   - projection: The `qkv_proj` output, shape
    ///     `[B, 1, (queryHeads + 2 * kvHeads) * headDim]`: the query heads,
    ///     then the key heads, then the value heads.
    ///   - queryWeight: The weight of the RMS norm of each query head.
    ///   - keyWeight: The weight of the RMS norm of each key head.
    ///   - offset: The position of the token.
    /// - Returns: The normed and rotated queries and keys and the values,
    ///   each `[B, heads, 1, headDim]`, or `nil`.
    func heads(
        of projection: MLXArray, queryWeight: MLXArray, keyWeight: MLXArray, offset: Int
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray)? {
        let headCount = queryHeads + Self.kvProjections * kvHeads
        guard canRun(projection, headCount: headCount, weights: [queryWeight, keyWeight])
        else { return nil }
        let batchSize = projection.dim(0)
        let shapes = [queryHeads, kvHeads, kvHeads].map {
            [batchSize, $0, Self.decodeTokens, Self.headDim]
        }
        let results = kernel(
            [projection, queryWeight, keyWeight, MLXArray([Int32(offset)])],
            template: [("T", projection.dtype)],
            grid: (metalSimdWidth, batchSize * headCount, 1),
            threadGroup: (metalSimdWidth, 1, 1),
            outputShapes: shapes,
            outputDTypes: shapes.map { _ in projection.dtype })
        return (
            results[Self.queriesOutput], results[Self.keysOutput], results[Self.valuesOutput]
        )
    }

    /// Tells if the kernel can read `projection` with norm `weights`.
    private func canRun(_ projection: MLXArray, headCount: Int, weights: [MLXArray]) -> Bool {
        Device.defaultDevice().deviceType == .gpu
            && projection.ndim == Self.projectionAxes
            && projection.dim(Self.tokenAxis) == Self.decodeTokens
            && projection.dim(-1) == headCount * Self.headDim
            && Self.floatTypes.contains(projection.dtype)
            && weights.allSatisfy { $0.shape == [Self.headDim] && $0.dtype == projection.dtype }
    }
}

extension FlintHeadRotation {
    /// The base and the position scale the kernel reads. An unrotated layer
    /// reads neither, so it passes the values of a plain RoPE.
    fileprivate var parameters: (base: Float, scale: Float) {
        switch self {
        case .unrotated: (1, 1)
        case .rotary(let base, let scale): (base, scale)
        }
    }
}
