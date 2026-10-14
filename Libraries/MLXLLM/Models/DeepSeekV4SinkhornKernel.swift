// Copyright © 2026 Apple Inc.
//
// The fused Metal kernel of the DeepSeek-V4 hyper-connection collapse.
//
// The Sinkhorn half follows the kernel `deepseek_v41_sinkhorn` of
// ml-explore/mlx-lm, mlx_lm/models/deepseek_v41.py (pull request 1895): one
// thread takes the whole Sinkhorn loop of one mixing matrix, in the order of
// the op path. The weighted-sum half follows the second phase of the kernel
// `hc_sinkhorn_collapse` of Blaizzy/mlx-vlm,
// mlx_vlm/models/deepseek_v4/hyper_connection.py: all the threads of the
// threadgroup read the copies of one token in float4 groups. Manual
// transcription; no git ancestry.
//
// Three details do not come from those kernels:
//
//  1. The softmax and the sigmoids stay on the op path. The mlx-vlm kernel
//     also fuses them, but it adds `eps` to the softmax denominator and it
//     reads `fast::exp`, thus its numbers are not the numbers of
//     `DeepSeekV4Math.hcSplitFields`. This kernel reads the fields that
//     function gives and changes no step of the math.
//  2. The mlx-vlm kernel reads exactly four copies. This kernel takes the
//     copy count, the step count and the width as template arguments, and
//     it reads the width one float at a time when the width is not a
//     multiple of four.
//  3. The mlx-lm kernel gives the whole Sinkhorn loop to one thread. This
//     kernel gives one row of the mixing matrix to each lane of the first
//     SIMD group, and a column normalization reads the column through
//     `simd_shuffle`. Each sum still adds the values in index order, and
//     each step still divides, thus the bits do not change. The Sinkhorn
//     half took about 57 us with one thread and takes about 14 us with one
//     lane for each row.

import Foundation
import MLX

/// One Metal kernel for the Sinkhorn steps and the weighted sum of one
/// DeepSeek-V4 hyper-connection collapse.
///
/// As separate MLX operations, the Sinkhorn steps of one collapse are
/// `2 * iters - 1` normalizations of three operations each, and the weighted
/// sum is two more. With `hc_sinkhorn_iters` 20 that is about 120 kernels for
/// each collapse, and DeepSeek-V4-Flash runs 86 collapses for each token.
/// This kernel does the same work in one dispatch.
///
/// The kernel takes one threadgroup for each token. The first SIMD group of
/// the threadgroup takes the Sinkhorn steps of the mixing matrix of that
/// token, one row in each lane, and all the threads of the threadgroup take
/// the weighted sum, which does not read the mixing matrix. The two halves
/// thus need no barrier.
enum DeepSeekV4SinkhornKernel {

    /// The number of threads in the threadgroup of one token.
    private static let threadsPerToken = 256

    /// The number of lanes in one Apple GPU SIMD group. The Sinkhorn half
    /// runs in the first SIMD group of the threadgroup, one row of the mixing
    /// matrix in each lane, thus ``threadsPerToken`` must not be less than
    /// this number.
    private static let simdGroupWidth = 32

    /// The axis of the residual stream that holds the copies.
    private static let copyAxis = -2

    /// The axis of the residual stream that holds the numbers of one copy.
    private static let widthAxis = -1

    /// The number of trailing axes of the residual stream that one token
    /// holds: the copy axis and the width axis.
    private static let tokenAxisCount = 2

    /// The Metal source. The template arguments are `HC`, the number of
    /// copies; `ITERS`, the number of Sinkhorn steps; `D`, the width of one
    /// copy; `THREADS`, the threadgroup width; `SIMD_WIDTH`, the number of
    /// lanes in one SIMD group; and `EPS_BITS`, the bit pattern of the
    /// float32 epsilon.
    ///
    /// The Sinkhorn loop is one column normalization, then `ITERS - 1` pairs
    /// of a row normalization and a column normalization. Each normalization
    /// adds the values in index order and divides by the sum plus epsilon,
    /// which is the order of `DeepSeekV4Math.sinkhorn(_:iters:eps:)`.
    ///
    /// Lane `r` of the first SIMD group keeps row `r` in registers. A row
    /// normalization adds the row of the lane. A column normalization reads
    /// value `c` of each row 0 to `HC - 1` through `simd_shuffle`. All the
    /// lanes of the SIMD group must run each `simd_shuffle`, thus a lane past
    /// the last row also runs the loop, on a copy of row 0, and does not
    /// write its answer.
    ///
    /// The source indexes `comb_in` and does not keep a pointer into it. MLX
    /// gives an input of fewer than 8 values in the `constant` address space,
    /// and a `device` pointer to such an input does not compile.
    private static let source = """
        uint token = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        const float eps = as_type<float>(EPS_BITS);

        if (lane < SIMD_WIDTH) {
            uint row = lane < HC ? lane : 0;
            uint first = (token * HC + row) * HC;
            float v[HC];
            #pragma unroll
            for (int b = 0; b < HC; b++) {
                v[b] = comb_in[first + b];
            }
            for (int step = 0; step < 2 * ITERS - 1; step++) {
                if (step % 2 == 1) {
                    float total = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < HC; b++) {
                        total += v[b];
                    }
                    float divisor = total + eps;
                    #pragma unroll
                    for (int b = 0; b < HC; b++) {
                        v[b] = v[b] / divisor;
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < HC; c++) {
                        float total = 0.0f;
                        #pragma unroll
                        for (int a = 0; a < HC; a++) {
                            total += simd_shuffle(v[c], (ushort)a);
                        }
                        v[c] = v[c] / (total + eps);
                    }
                }
            }
            if (lane < HC) {
                #pragma unroll
                for (int b = 0; b < HC; b++) {
                    comb[first + b] = v[b];
                }
            }
        }

        float weight[HC];
        for (int c = 0; c < HC; c++) {
            weight[c] = pre[token * HC + c];
        }
        const device float* copies = stream + token * HC * D;
        device float* single = collapsed + token * D;
        if (D % 4 == 0) {
            constexpr uint groups = D / 4;
            const device float4* copies4 = (const device float4*)copies;
            device float4* single4 = (device float4*)single;
            for (uint i = lane; i < groups; i += THREADS) {
                float4 total = float4(0.0f);
                for (int c = 0; c < HC; c++) {
                    total += weight[c] * copies4[c * groups + i];
                }
                single4[i] = total;
            }
        } else {
            for (uint d = lane; d < D; d += THREADS) {
                float total = 0.0f;
                for (int c = 0; c < HC; c++) {
                    total += weight[c] * copies[c * D + d];
                }
                single[d] = total;
            }
        }
        """

    /// The Metal header. It stops the compiler from fusing a multiply and an
    /// add into one `fma`. The op path rounds each product before the sum
    /// adds it, and an `fma` skips that rounding, thus without this pragma
    /// the weighted sum moves by one unit in the last place and the greedy
    /// output of a long generation can change. MLX writes the kernel
    /// signature directly after the header, thus the header ends in a line
    /// break.
    private static let header = "#pragma METAL fp contract(off)\n"

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_sinkhorn_collapse",
        inputNames: ["comb_in", "pre", "stream"],
        outputNames: ["comb", "collapsed"],
        source: source,
        header: header)

    /// Tells if the fused kernel can take the collapse of `stream`.
    ///
    /// The kernel is a Metal kernel, thus it runs only when the default
    /// device is the GPU. It needs at least one Sinkhorn step, because the op
    /// path always takes the first column normalization, and it needs at
    /// least one token. It keeps one row of the mixing matrix in each lane of
    /// one SIMD group, thus the copy count must not be more than the width of
    /// one SIMD group.
    ///
    /// - Parameters:
    ///   - stream: The residual stream, shape `(..., copies, width)`.
    ///   - iters: The number of Sinkhorn steps, from `hc_sinkhorn_iters`.
    /// - Returns: `true` when ``sinkhornCollapse(comb:pre:stream:iters:eps:)``
    ///   can take the collapse.
    static func canRun(on stream: MLXArray, iters: Int) -> Bool {
        Device.defaultDevice().deviceType == .gpu && iters > 0 && stream.size > 0
            && stream.dim(copyAxis) <= simdGroupWidth
    }

    /// Takes the Sinkhorn steps of `comb` and the weighted sum of `stream` in
    /// one dispatch.
    ///
    /// The answer has the same float32 bits as the answer of
    /// `DeepSeekV4Math.sinkhorn(_:iters:eps:)` on `comb` and of the weighted
    /// sum of the copies of `stream` by `pre` as separate MLX operations.
    ///
    /// - Parameters:
    ///   - comb: The row softmax of the mixing logits plus `eps`, shape
    ///     `(..., copies, copies)`.
    ///   - pre: One weight for each copy, shape `(..., copies)`.
    ///   - stream: The residual stream, shape `(..., copies, width)`.
    ///   - iters: The number of Sinkhorn steps, from `hc_sinkhorn_iters`. It
    ///     must be 1 or more.
    ///   - eps: The epsilon the Sinkhorn steps add, from `hc_eps`.
    /// - Returns: `comb`, the doubly stochastic matrix, of the shape of the
    ///   input `comb`, and `collapsed`, the single stream, shape
    ///   `(..., width)`. Both are float32.
    static func sinkhornCollapse(
        comb: MLXArray, pre: MLXArray, stream: MLXArray, iters: Int, eps: Float
    ) -> (comb: MLXArray, collapsed: MLXArray) {
        precondition(iters > 0, "the fused kernel needs 1 or more Sinkhorn steps, not \(iters)")
        let copies = stream.dim(copyAxis)
        precondition(
            copies <= simdGroupWidth,
            "the fused kernel takes \(simdGroupWidth) or fewer copies, not \(copies)")
        let width = stream.dim(widthAxis)
        let leading = Array(stream.shape.dropLast(tokenAxisCount))
        precondition(
            comb.shape == leading + [copies, copies],
            "comb must be shape \(leading + [copies, copies]), and it is \(comb.shape)")
        precondition(
            pre.shape == leading + [copies],
            "pre must be shape \(leading + [copies]), and it is \(pre.shape)")

        let tokens = leading.reduce(1, *)
        let epsBits = Int(Int32(bitPattern: eps.bitPattern))
        let outputs = kernel(
            [comb.asType(.float32), pre.asType(.float32), stream.asType(.float32)],
            template: [
                ("HC", copies), ("ITERS", iters), ("D", width),
                ("THREADS", threadsPerToken), ("SIMD_WIDTH", simdGroupWidth),
                ("EPS_BITS", epsBits),
            ],
            grid: (tokens * threadsPerToken, 1, 1),
            threadGroup: (threadsPerToken, 1, 1),
            outputShapes: [comb.shape, leading + [width]],
            outputDTypes: [.float32, .float32])
        return (comb: outputs[0], collapsed: outputs[1])
    }
}
