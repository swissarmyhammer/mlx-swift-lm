// Copyright © 2026 Apple Inc.
//
// One Metal kernel for a residual add and the RMS norm that reads its sum.
//
// A decoder layer adds the output of each block to the residual stream, and
// the next norm reads that sum: the norm before the feed-forward block, or the
// input norm of the next layer, or the final norm (task ^ftj9mae). MLX runs the
// add and the norm as two dispatches, and `compile` does not fuse them,
// because the norm is a primitive of its own and not an elementwise operation.
// This kernel writes the sum and its norm in one dispatch.
//
// The kernel gives the SAME BITS as the two MLX kernels. The sum rounds in
// the dtype `T` first, as the MLX `Add` does, and the norm reads that rounded
// sum. The norm follows `rms_single_row` for a row of at most
// `RMS_LOOPED_LIMIT` numbers and `rms_looped` for a wider row, with the
// threads of the MLX dispatch:
//
//  - One threadgroup reads one row. Thread `t` reads the numbers
//    `p * P + 4t ..< p * P + 4t + 4` of each pass `p`, where `P` is four
//    numbers for each thread of the threadgroup. A row of at most
//    `RMS_LOOPED_LIMIT` numbers takes one pass, with the fewest whole SIMD
//    groups that hold a thread for each 4 numbers; a wider row takes the
//    1024 threads of the MLX looped kernel, in as many passes as it needs.
//  - Each thread adds the squares of its numbers with `fma`. SwiftPM compiles
//    the MLX kernels with the fast math of the Metal compiler, which contracts
//    the multiply and the add into one fused multiply-add, and MLX compiles
//    this kernel at run time without fast math, thus the source writes the
//    `fma`.
//  - `simd_sum` adds the threads of each SIMD group, and a second `simd_sum`
//    adds the sums of the SIMD groups from the lanes of the first SIMD group,
//    with zero in every lane past the last SIMD group. The MLX source divides
//    the sum by the width, and its fast math multiplies by the reciprocal of
//    the width instead, which rounds differently for a width that is not a
//    power of two. Thus the inverse is
//    `precise::rsqrt(sum * (1 / width) + eps)`, and the output is
//    `weight * T(x * inverse)`, rounded in the dtype `T`.
//  - In float32 the conversion to `T` rounds nothing, and the fast math of
//    the looped MLX kernel reorders the two products of a row wider than
//    `RMS_LOOPED_LIMIT` to `x * (weight * inverse)`. The kernel writes that
//    order for such a row in float32.
//
// Tests/MLXLMTests/ResidualRMSNormKernelTests.swift compares the bits.

import Foundation
import MLX
import MLXNN

/// A residual add and the RMS norm of its sum, in one Metal kernel.
///
/// For each row it writes `sum = residual + addend`, rounded in the dtype of
/// the operands, and `normed = MLXFast.rmsNorm(sum, weight:, eps:)`, with the
/// bits of those two MLX operations.
package struct ResidualRMSNormKernel {

    /// The numbers one thread of the MLX RMS norm reads in one pass
    /// (`RMS_N_READS`).
    private static let numbersPerThread = 4

    /// The widest row the MLX RMS norm reads in one pass of its threads
    /// (`RMS_LOOPED_LIMIT`); a wider row takes its looped kernel.
    private static let widestSinglePassRow = 4096

    /// The threads of one threadgroup of the looped MLX RMS norm: the most
    /// threads its pipeline takes in one threadgroup on an Apple GPU.
    private static let loopedThreads = 1024

    /// The dtypes the kernel reads.
    private static let floatTypes: [DType] = [.float32, .float16, .bfloat16]

    /// The index of the sum among the outputs of the kernel.
    private static let sumOutput = 0

    /// The index of the normed sum among the outputs of the kernel.
    private static let normedOutput = 1

    private static let source = """
        constexpr int perThread = PER_THREAD;
        constexpr int passWidth = THREADS * perThread;
        constexpr int passes = (WIDTH + passWidth - 1) / passWidth;

        threadgroup float simdSums[SIMD_WIDTH];
        threadgroup float rowInverse[1];

        uint threadIndex = thread_position_in_threadgroup.x;
        uint simdLane = thread_index_in_simdgroup;
        uint simdGroup = simdgroup_index_in_threadgroup;
        size_t rowStart = size_t(threadgroup_position_in_grid.y) * WIDTH;
        float eps = as_type<float>(EPS_BITS);

        T rowSums[passes][perThread];
        float squares = 0;
        for (int p = 0; p < passes; p++) {
            for (int i = 0; i < perThread; i++) {
                int d = p * passWidth + int(threadIndex) * perThread + i;
                if (d < WIDTH) {
                    T added = residual[rowStart + d] + addend[rowStart + d];
                    rowSums[p][i] = added;
                    sum[rowStart + d] = added;
                    float x = static_cast<float>(added);
                    squares = fma(x, x, squares);
                }
            }
        }
        squares = simd_sum(squares);
        if (simdGroup == 0) {
            simdSums[simdLane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdLane == 0) {
            simdSums[simdGroup] = squares;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdGroup == 0) {
            float total = simd_sum(simdSums[simdLane]);
            if (simdLane == 0) {
                rowInverse[0] = metal::precise::rsqrt(total * (1.0f / WIDTH) + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inverse = rowInverse[0];
        for (int p = 0; p < passes; p++) {
            for (int i = 0; i < perThread; i++) {
                int d = p * passWidth + int(threadIndex) * perThread + i;
                if (d < WIDTH) {
                    float x = static_cast<float>(rowSums[p][i]);
                    normed[rowStart + d] = LOOPED_ROW && metal::is_same_v<T, float>
                        ? static_cast<T>(x * (static_cast<float>(weight[d]) * inverse))
                        : weight[d] * static_cast<T>(x * inverse);
                }
            }
        }
        """

    /// The numbers of one row.
    private let width: Int

    /// The threads of one threadgroup, the threads the MLX RMS norm
    /// dispatches for a row of ``width`` numbers.
    private let threads: Int

    /// The kernel of the row width and the epsilon. Its header holds both,
    /// thus a call names only the dtype.
    private let kernel: MLXFast.MLXFastKernel

    /// The kernel of a row width and an epsilon, or `nil` for an empty row.
    ///
    /// - Parameters:
    ///   - width: The numbers of one row.
    ///   - eps: The epsilon of the norm.
    package init?(width: Int, eps: Float) {
        guard width > 0 else { return nil }
        self.width = width
        self.threads = Self.threads(width: width)
        let counts: KeyValuePairs = [
            "PER_THREAD": Self.numbersPerThread, "WIDTH": width, "THREADS": threads,
            "LOOPED_ROW": width > Self.widestSinglePassRow ? 1 : 0,
        ]
        let header =
            counts.map { "constexpr constant int \($0.key) = \($0.value);\n" }
            + ["constexpr constant uint EPS_BITS = \(eps.bitPattern)u;\n"]
        // MLX keeps one library for each kernel name, thus the name holds
        // every constant of the header.
        let name =
            (["residual_rms_norm"] + counts.map { "\($0.value)" } + ["\(eps.bitPattern)"])
            .joined(separator: "_")
        self.kernel = MLXFast.metalKernel(
            name: name, inputNames: ["residual", "addend", "weight"],
            outputNames: ["sum", "normed"], source: Self.source,
            header: metalSimdWidthDeclaration + header.joined())
    }

    /// The threads the MLX RMS norm dispatches for one row of `width`
    /// numbers: the fewest whole SIMD groups that hold a thread for each
    /// ``numbersPerThread`` numbers, or ``loopedThreads`` for a row wider
    /// than ``widestSinglePassRow``.
    private static func threads(width: Int) -> Int {
        guard width <= widestSinglePassRow else { return loopedThreads }
        let threadsNeeded = (width + numbersPerThread - 1) / numbersPerThread
        let simdGroups = (threadsNeeded + metalSimdWidth - 1) / metalSimdWidth
        return simdGroups * metalSimdWidth
    }

    /// The sum of `residual` and `addend` and its RMS norm, with the bits of
    /// the op path, or `nil` when the kernel cannot read them.
    ///
    /// The kernel runs when the default device is the GPU, the residual and
    /// the addend have one shape whose last axis holds the row width, and
    /// the residual, the addend and the weight hold one float dtype.
    ///
    /// - Parameters:
    ///   - residual: The residual stream, shape `[..., width]`.
    ///   - addend: The block output the layer adds to it, the same shape.
    ///   - weight: The weight of the norm, shape `[width]`.
    /// - Returns: The sum and its norm, each the shape of `residual`, or
    ///   `nil`.
    package func callAsFunction(
        residual: MLXArray, addend: MLXArray, weight: MLXArray
    ) -> (sum: MLXArray, normed: MLXArray)? {
        let dtype = residual.dtype
        guard Device.defaultDevice().deviceType == .gpu, Self.floatTypes.contains(dtype),
            residual.ndim > 0, residual.shape == addend.shape, residual.shape.last == width,
            weight.shape == [width],
            [addend, weight].allSatisfy({ $0.dtype == dtype })
        else { return nil }
        let rows = residual.size / width
        let results = kernel(
            [residual, addend, weight],
            template: [("T", dtype)],
            grid: (threads, rows, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [residual.shape, residual.shape],
            outputDTypes: [dtype, dtype])
        return (results[Self.sumOutput], results[Self.normedOutput])
    }
}

extension RMSNorm {
    /// The sum `residual + addend` and its norm, as a residual add and this
    /// norm give them, in one ``ResidualRMSNormKernel`` dispatch when it can.
    ///
    /// A subclass of `RMSNorm` may norm in a way of its own, thus it takes
    /// the op path, as does a sum the kernel cannot read.
    ///
    /// - Parameters:
    ///   - residual: The residual stream, shape `[..., width]`.
    ///   - addend: The block output the layer adds to it, the same shape.
    /// - Returns: The sum and its norm.
    package func normalizingSum(
        of residual: MLXArray, adding addend: MLXArray
    ) -> (sum: MLXArray, normed: MLXArray) {
        if type(of: self) == RMSNorm.self,
            let kernel = ResidualRMSNormKernel(width: weight.size, eps: eps),
            let fused = kernel(residual: residual, addend: addend, weight: weight)
        {
            return fused
        }
        let sum = residual + addend
        return (sum, self(sum))
    }
}
