// Copyright © 2026 Apple Inc.
//
// The fused Metal kernel of the DeepSeek-V4 hyper-connection expand.
//
// The kernel follows the kernel `vmlx_mhc_prefill_place` of
// osaurus-ai/vmlx-swift, Libraries/MLXLLM/Models/MHCFusedPrefill.swift
// (commit f8130127): each thread reads the copies of the residual stream at
// one place of the width, mixes them in float32, adds the block output
// scaled by `post`, and writes each answer copy in the dtype of the block
// output. Manual transcription; no git ancestry.
//
// Two details do not come from that kernel:
//
//  1. The vmlx-swift kernel reads exactly four copies, one value at a time.
//     This kernel takes the copy count as a template argument, and each
//     thread reads a vector of 4 values of the width, thus one thread holds
//     4 places and the loads move 4 times the data.
//  2. The vmlx-swift kernel takes the block output and the residual stream
//     in one dtype. This kernel takes each one in its own dtype, as the op
//     path does.

import Foundation
import MLX

/// One Metal kernel for the expand of one DeepSeek-V4 hyper-connection.
///
/// As separate MLX operations, the expand is a float32 copy of the residual
/// stream, a batched matrix multiply of the mixing matrix by it, and one
/// compiled elementwise kernel for the new term, the sum and the cast. Each
/// one reads or writes the whole float32 stream. The kernel reads the stream
/// one time in its own dtype and writes the answer one time.
///
/// ``DeepSeekV4HyperConnection/expand(blockOutput:residual:post:comb:)``
/// states the reading of `comb` this kernel makes and its source: copy `k` of
/// the answer is `post[k] * blockOutput + sum over j of comb[j][k] *
/// residual[j]`. The kernel adds the products in another order than the MLX
/// matrix multiply, thus its answer is not the bits of the op path.
enum DeepSeekV4ExpandKernel {

    /// The number of values of the width one thread reads in one vector
    /// load.
    private static let vectorWidth = 4

    /// The number of threads of one threadgroup.
    private static let threadsPerGroup = 256

    /// The smallest input MLX gives a kernel in the `device` address space.
    /// MLX gives an input of fewer values in the `constant` address space,
    /// and the vector loads of the kernel need a `device` pointer.
    private static let smallestDeviceInput = 8

    /// The axis of the residual stream that holds the copies.
    private static let copyAxis = -2

    /// The axis of the residual stream and of the block output that holds
    /// the numbers of one copy.
    private static let widthAxis = -1

    /// The number of trailing axes of the residual stream that one token
    /// holds: the copy axis and the width axis.
    private static let tokenAxisCount = 2

    /// The Metal source. The template arguments are `T`, the dtype of the
    /// block output and of the answer; `R`, the dtype of the residual
    /// stream; `HC`, the number of copies; `D`, the width of one copy; and
    /// `VECTOR_WIDTH`, the number of values of one vector load.
    ///
    /// Thread `i` takes vector `i % (D / VECTOR_WIDTH)` of the width of token
    /// `i / (D / VECTOR_WIDTH)`. The source indexes `post` and `comb` and
    /// keeps no pointer into them, because MLX gives an input of fewer than
    /// 8 values in the `constant` address space, and the `post` of one
    /// token of 4 copies holds 4 values.
    private static let source = """
        typedef vec<float, VECTOR_WIDTH> floatVector;
        constexpr uint vectors = D / VECTOR_WIDTH;
        uint index = thread_position_in_grid.x;
        uint token = index / vectors;
        uint place = index % vectors;

        const device vec<R, VECTOR_WIDTH>* copies =
            (const device vec<R, VECTOR_WIDTH>*)residual + token * HC * vectors + place;
        floatVector residualCopies[HC];
        for (uint j = 0; j < HC; j++) {
            residualCopies[j] = floatVector(copies[j * vectors]);
        }
        floatVector output = floatVector(
            ((const device vec<T, VECTOR_WIDTH>*)blockOutput)[token * vectors + place]);
        device vec<T, VECTOR_WIDTH>* answer =
            (device vec<T, VECTOR_WIDTH>*)expanded + token * HC * vectors + place;
        for (uint k = 0; k < HC; k++) {
            floatVector mixed = floatVector(0.0f);
            for (uint j = 0; j < HC; j++) {
                mixed += comb[(token * HC + j) * HC + k] * residualCopies[j];
            }
            answer[k * vectors] = vec<T, VECTOR_WIDTH>(post[token * HC + k] * output + mixed);
        }
        """

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_hyper_connection_expand",
        inputNames: ["blockOutput", "residual", "post", "comb"],
        outputNames: ["expanded"],
        source: source)

    /// Tells if the kernel can take the expand of `blockOutput` into
    /// `residual`.
    ///
    /// The kernel is a Metal kernel, thus it runs only when the default
    /// device is the GPU. It reads the width in vector loads through a
    /// `device` pointer, thus the width must be a multiple of the vector
    /// width and the block output must hold enough values for MLX to give it
    /// in the `device` address space. It broadcasts nothing, thus the block
    /// output must be the shape of the residual stream without its copy
    /// axis.
    ///
    /// - Parameters:
    ///   - blockOutput: The output of the sub-layer, shape `(..., width)`.
    ///   - residual: The residual stream, shape `(..., copies, width)`.
    /// - Returns: `true` when
    ///   ``expand(blockOutput:residual:post:comb:)`` can take the expand.
    static func canRun(blockOutput: MLXArray, residual: MLXArray) -> Bool {
        Device.defaultDevice().deviceType == .gpu && blockOutput.size >= smallestDeviceInput
            && blockOutput.dim(widthAxis) % vectorWidth == 0
            && blockOutput.shape == blockOutputShape(of: residual)
    }

    /// The shape of the block output that matches a residual stream: the
    /// shape of the stream without its copy axis.
    ///
    /// - Parameter residual: The residual stream, shape
    ///   `(..., copies, width)`.
    /// - Returns: The shape `(..., width)`.
    private static func blockOutputShape(of residual: MLXArray) -> [Int] {
        Array(residual.shape.dropLast(tokenAxisCount)) + [residual.dim(widthAxis)]
    }

    /// Writes a block output back into the parallel copies of the residual
    /// stream in one dispatch.
    ///
    /// The answer is the answer of
    /// ``ManifoldStream/expand(blockOutput:residual:post:comb:)`` within the
    /// last float32 places, changed to the dtype of `blockOutput`.
    ///
    /// - Parameters:
    ///   - blockOutput: The output of the sub-layer, shape `(..., width)`.
    ///     ``canRun(blockOutput:residual:)`` must accept it.
    ///   - residual: The residual stream the matching collapse read, shape
    ///     `(..., copies, width)`.
    ///   - post: The gain each copy gives the block output, shape
    ///     `(..., copies)`.
    ///   - comb: The doubly stochastic mixing matrix, shape
    ///     `(..., copies, copies)`.
    /// - Returns: The residual stream of the next sub-layer, of the shape of
    ///   `residual`, in the dtype of `blockOutput`.
    static func expand(
        blockOutput: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        precondition(
            blockOutput.shape == blockOutputShape(of: residual),
            "the block output must be shape \(blockOutputShape(of: residual)), "
                + "and it is \(blockOutput.shape)")
        return kernel(
            [blockOutput, residual, post.asType(.float32), comb.asType(.float32)],
            template: [
                ("T", blockOutput.dtype), ("R", residual.dtype), ("HC", residual.dim(copyAxis)),
                ("D", residual.dim(widthAxis)), ("VECTOR_WIDTH", vectorWidth),
            ],
            grid: (blockOutput.size / vectorWidth, 1, 1),
            threadGroup: (threadsPerGroup, 1, 1),
            outputShapes: [residual.shape],
            outputDTypes: [blockOutput.dtype])[0]
    }
}
