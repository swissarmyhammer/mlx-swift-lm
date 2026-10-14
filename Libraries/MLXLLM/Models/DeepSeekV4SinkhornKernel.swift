// Copyright © 2026 Apple Inc.
//
// The fused Metal kernel of the DeepSeek-V4 hyper-connection collapse.
//
// The Sinkhorn half follows the kernel `deepseek_v41_sinkhorn` of
// ml-explore/mlx-lm, mlx_lm/models/deepseek_v41.py (pull request 1895): one
// thread takes the whole Sinkhorn loop of one mixing matrix, in the order of
// the op path. The split activations and the weighted sum follow the kernel
// `hc_sinkhorn_collapse` of Blaizzy/mlx-vlm,
// mlx_vlm/models/deepseek_v4/hyper_connection.py: the kernel reads the mixes
// vector, the scales and the bias, takes the sigmoids and the row softmax
// itself, and all the threads of the threadgroup read the copies of one token
// in float4 groups. Manual transcription; no git ancestry.
//
// Four details do not come from those kernels:
//
//  1. The activations give the bits of `DeepSeekV4Math.hcSplitFields`. The
//     mlx-vlm kernel adds `eps` to the softmax denominator and reads
//     `fast::exp` in its sigmoids, thus its numbers are not the numbers of
//     the op path. This kernel takes each step of the MLX primitives in
//     their own order and with their own exponential: the sigmoid of
//     `Sigmoid` in kernels/unary_ops.h, which reads `metal::exp`, and the
//     row softmax of `softmax_single_row` in kernels/softmax.h, which reads
//     `fast::exp` also when the op path asks for a precise softmax.
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
//  4. Each thread of the threadgroup takes the `pre` sigmoids itself, thus
//     the weighted sum needs no barrier and no threadgroup memory.

import Foundation
import MLX

/// One Metal kernel for the split activations, the Sinkhorn steps and the
/// weighted sum of one DeepSeek-V4 hyper-connection collapse.
///
/// As separate MLX operations, the split activations of one collapse are
/// five kernels: one for each sigmoid field, one for the logits of the
/// mixing matrix, the row softmax, and the epsilon the softmax adds. The
/// Sinkhorn steps are `2 * iters - 1` normalizations of three operations
/// each, and the weighted sum is two more. DeepSeek-V4-Flash runs 86
/// collapses for each token. This kernel does the same work in one dispatch.
///
/// The kernel takes one threadgroup for each token. The first SIMD group of
/// the threadgroup takes the row softmax and the Sinkhorn steps of the mixing
/// matrix of that token, one row in each lane, and the `post` gains. All the
/// threads of the threadgroup take the `pre` weights and the weighted sum,
/// which does not read the mixing matrix. The two halves thus need no
/// barrier.
enum DeepSeekV4SinkhornKernel {

    /// The number of threads in the threadgroup of one token.
    private static let threadsPerToken = 256

    /// The number of lanes in one Apple GPU SIMD group. The Sinkhorn half
    /// runs in the first SIMD group of the threadgroup, one row of the mixing
    /// matrix in each lane, thus ``threadsPerToken`` must not be less than
    /// this number.
    private static let simdGroupWidth = 32

    /// The number of values one thread of the MLX softmax kernel reads and
    /// adds in index order, `SOFTMAX_N_READS` in kernels/defines.h. The
    /// kernel adds each group of this many values of a softmax row first,
    /// and then adds the group sums, as the MLX kernel does.
    private static let softmaxReadWidth = 4

    /// The axis of the residual stream that holds the copies.
    private static let copyAxis = -2

    /// The axis of the residual stream that holds the numbers of one copy.
    private static let widthAxis = -1

    /// The number of trailing axes of the residual stream that one token
    /// holds: the copy axis and the width axis.
    private static let tokenAxisCount = 2

    /// The largest place of a field scale inside the learned scale vector.
    /// The scale vector must hold one more value than this place.
    private static let lastScaleIndex = max(
        DeepSeekV4Math.preScaleIndex, DeepSeekV4Math.postScaleIndex,
        DeepSeekV4Math.combScaleIndex)

    /// The outputs of the kernel, in the order of the kernel output list. The
    /// name of each case is the name of its output in the Metal source, and
    /// the raw value of each case is its place in the output list.
    private enum Output: Int, CaseIterable {
        /// The gain of each copy, shape `(..., copies)`.
        case post
        /// The doubly stochastic matrix, shape `(..., copies, copies)`.
        case comb
        /// The single stream, shape `(..., width)`.
        case collapsed

        /// The name of the output in the Metal source.
        var name: String { String(describing: self) }

        /// The shape of the output.
        ///
        /// - Parameters:
        ///   - leading: The leading axes of the residual stream, one for each
        ///     token axis.
        ///   - copies: The number of copies of the residual stream.
        ///   - width: The width of one copy.
        /// - Returns: The shape of the output for these axes.
        func shape(leading: [Int], copies: Int, width: Int) -> [Int] {
            switch self {
            case .post: leading + [copies]
            case .comb: leading + [copies, copies]
            case .collapsed: leading + [width]
            }
        }
    }

    /// The Metal source. The template arguments are `HC`, the number of
    /// copies; `ITERS`, the number of Sinkhorn steps; `D`, the width of one
    /// copy; `THREADS`, the threadgroup width; `SIMD_WIDTH`, the number of
    /// lanes in one SIMD group; `MIX_WIDTH`, the width of the mixes vector;
    /// `COMB_OFFSET`, the place of the first logit of the mixing matrix in
    /// the mixes vector; `PRE_SCALE`, `POST_SCALE` and `COMB_SCALE`, the
    /// place of each field scale in the scale vector; `SOFTMAX_READS`, the
    /// group width of the softmax sum; and `EPS_BITS` and `POST_GAIN_BITS`,
    /// the bit patterns of the float32 epsilon and of the `post` gain. The
    /// `pre` field starts at place 0 of the mixes vector and the `post`
    /// field follows its `HC` values.
    ///
    /// The row softmax takes the steps of the MLX softmax kernel: the row
    /// maximum, `fast::exp` of each value less the maximum, the sum of each
    /// group of `SOFTMAX_READS` values in index order and then the sum of
    /// the group sums, and the product of each value by the inverse of the
    /// sum. It then adds `eps`, as the op path does.
    ///
    /// The Sinkhorn loop is one column normalization, then `ITERS - 1` pairs
    /// of a row normalization and a column normalization. Each normalization
    /// adds the values in index order and divides by the sum plus epsilon,
    /// which is the order of `DeepSeekV4Math.sinkhorn(_:iters:eps:)`.
    /// `stepsPerPair` is the number of normalizations in one pair; an odd
    /// step is a row normalization and an even step is a column
    /// normalization.
    ///
    /// The weighted sum reads the width in groups of `vectorWidth`, the
    /// number of floats in one `float4`, when the width is a multiple of
    /// that number.
    ///
    /// Lane `r` of the first SIMD group keeps row `r` in registers. A row
    /// normalization adds the row of the lane. A column normalization reads
    /// value `c` of each row 0 to `HC - 1` through `simd_shuffle`. All the
    /// lanes of the SIMD group must run each `simd_shuffle`, thus a lane past
    /// the last row also runs the loop, on a copy of row 0, and does not
    /// write its answer.
    ///
    /// The source indexes `mixes`, `scale` and `base` and keeps no pointer
    /// into them. MLX gives an input of fewer than 8 values in the
    /// `constant` address space, and a `device` pointer to such an input does
    /// not compile.
    private static let source = """
        uint token = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        const float eps = as_type<float>(EPS_BITS);
        uint mixFirst = token * MIX_WIDTH;

        if (lane < SIMD_WIDTH) {
            uint row = lane < HC ? lane : 0;
            uint first = (token * HC + row) * HC;
            float v[HC];
            const float combScale = scale[COMB_SCALE];
            float rowMax = -FLT_MAX;
            #pragma unroll
            for (int b = 0; b < HC; b++) {
                uint place = COMB_OFFSET + row * HC + b;
                v[b] = mixes[mixFirst + place] * combScale + base[place];
                rowMax = (rowMax < v[b]) ? v[b] : rowMax;
            }
            float rowSum = 0.0f;
            #pragma unroll
            for (int group = 0; group < HC; group += SOFTMAX_READS) {
                float groupSum = 0.0f;
                #pragma unroll
                for (int b = group; b < group + SOFTMAX_READS && b < HC; b++) {
                    v[b] = fast::exp(v[b] - rowMax);
                    groupSum += v[b];
                }
                rowSum += groupSum;
            }
            float inverse = 1 / rowSum;
            #pragma unroll
            for (int b = 0; b < HC; b++) {
                v[b] = v[b] * inverse + eps;
            }
            constexpr int stepsPerPair = 2;
            for (int step = 0; step < stepsPerPair * ITERS - 1; step++) {
                if (step % stepsPerPair == 1) {
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
                uint place = HC + lane;
                float logit = mixes[mixFirst + place] * scale[POST_SCALE] + base[place];
                post[token * HC + lane] = as_type<float>(POST_GAIN_BITS) * mlx_sigmoid(logit);
            }
        }

        float weight[HC];
        const float preScale = scale[PRE_SCALE];
        for (int c = 0; c < HC; c++) {
            weight[c] = mlx_sigmoid(mixes[mixFirst + c] * preScale + base[c]) + eps;
        }
        const device float* copies = stream + token * HC * D;
        device float* single = collapsed + token * D;
        constexpr uint vectorWidth = sizeof(float4) / sizeof(float);
        if (D % vectorWidth == 0) {
            constexpr uint groups = D / vectorWidth;
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

    /// The Metal header.
    ///
    /// The pragma stops the compiler from fusing a multiply and an add into
    /// one `fma`. The op path rounds each product before the sum adds it, and
    /// an `fma` skips that rounding, thus without this pragma the weighted
    /// sum moves by one unit in the last place and the greedy output of a
    /// long generation can change.
    ///
    /// `mlx_sigmoid` is the `Sigmoid` operator of MLX, kernels/unary_ops.h,
    /// for a float32 value, with the same `metal::exp`. The two are compiled
    /// with the same options, thus they give the same bits. A sigmoid with
    /// `metal::precise::exp` does not: the parity test measured one unit in
    /// the last place on some `pre` and `post` values.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header = """
        #pragma METAL fp contract(off)
        inline float mlx_sigmoid(float x) {
            float y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }

        """

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_split_sinkhorn_collapse",
        inputNames: ["mixes", "scale", "base", "stream"],
        outputNames: Output.allCases.map(\.name),
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
    /// - Returns: `true` when
    ///   ``splitSinkhornCollapse(mixes:scale:base:stream:iters:eps:)`` can
    ///   take the collapse.
    static func canRun(on stream: MLXArray, iters: Int) -> Bool {
        Device.defaultDevice().deviceType == .gpu && iters > 0 && stream.size > 0
            && stream.dim(copyAxis) <= simdGroupWidth
    }

    /// Takes the split activations, the Sinkhorn steps and the weighted sum
    /// of one collapse in one dispatch.
    ///
    /// The answer has the same float32 bits as the answer of
    /// `DeepSeekV4Math.hcSplitSinkhorn(mixes:scale:base:hcMult:iters:eps:)`
    /// on the mixes and of the weighted sum of the copies of `stream` by its
    /// `pre` weights, as separate MLX operations. Each input moves to float32
    /// first, as on the op path.
    ///
    /// - Parameters:
    ///   - mixes: The projected mixes, shape `(..., (2 + copies) * copies)`.
    ///   - scale: The three learned scales, one for each field, shape `(3)`.
    ///   - base: The learned bias, shape `((2 + copies) * copies)`.
    ///   - stream: The residual stream, shape `(..., copies, width)`.
    ///   - iters: The number of Sinkhorn steps, from `hc_sinkhorn_iters`. It
    ///     must be 1 or more.
    ///   - eps: The epsilon of the split and of the Sinkhorn steps, from
    ///     `hc_eps`.
    /// - Returns: `post`, the gain of each copy, shape `(..., copies)`;
    ///   `comb`, the doubly stochastic matrix, shape `(..., copies, copies)`;
    ///   and `collapsed`, the single stream, shape `(..., width)`. All three
    ///   are float32.
    static func splitSinkhornCollapse(
        mixes: MLXArray, scale: MLXArray, base: MLXArray, stream: MLXArray, iters: Int,
        eps: Float
    ) -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray) {
        precondition(iters > 0, "the fused kernel needs 1 or more Sinkhorn steps, not \(iters)")
        let copies = stream.dim(copyAxis)
        precondition(
            copies <= simdGroupWidth,
            "the fused kernel takes \(simdGroupWidth) or fewer copies, not \(copies)")
        let width = stream.dim(widthAxis)
        let leading = Array(stream.shape.dropLast(tokenAxisCount))
        let mixWidth = DeepSeekV4Math.mixWidth(hcMult: copies)
        precondition(
            mixes.shape == leading + [mixWidth],
            "mixes must be shape \(leading + [mixWidth]), and it is \(mixes.shape)")
        precondition(
            base.shape == [mixWidth], "base must be shape \([mixWidth]), and it is \(base.shape)")
        precondition(
            scale.size > lastScaleIndex,
            "scale must hold \(lastScaleIndex + 1) values, and it holds \(scale.size)")

        let outputs = kernel(
            [
                mixes.asType(.float32), scale.asType(.float32), base.asType(.float32),
                stream.asType(.float32),
            ],
            template: templateArguments(copies: copies, iters: iters, width: width, eps: eps),
            grid: (leading.reduce(1, *) * threadsPerToken, 1, 1),
            threadGroup: (threadsPerToken, 1, 1),
            outputShapes: Output.allCases.map {
                $0.shape(leading: leading, copies: copies, width: width)
            },
            outputDTypes: Output.allCases.map { _ in .float32 })
        return (
            post: outputs[Output.post.rawValue], comb: outputs[Output.comb.rawValue],
            collapsed: outputs[Output.collapsed.rawValue]
        )
    }

    /// The template arguments of one specialization of the kernel. The
    /// source names each one.
    ///
    /// - Parameters:
    ///   - copies: The number of copies of the residual stream.
    ///   - iters: The number of Sinkhorn steps.
    ///   - width: The width of one copy.
    ///   - eps: The epsilon of the split and of the Sinkhorn steps.
    /// - Returns: The template arguments, by name.
    private static func templateArguments(
        copies: Int, iters: Int, width: Int, eps: Float
    ) -> [(String, any KernelTemplateArg)] {
        [
            ("HC", copies), ("ITERS", iters), ("D", width),
            ("THREADS", threadsPerToken), ("SIMD_WIDTH", simdGroupWidth),
            ("MIX_WIDTH", DeepSeekV4Math.mixWidth(hcMult: copies)),
            ("COMB_OFFSET", DeepSeekV4Math.splitFieldCount * copies),
            ("PRE_SCALE", DeepSeekV4Math.preScaleIndex),
            ("POST_SCALE", DeepSeekV4Math.postScaleIndex),
            ("COMB_SCALE", DeepSeekV4Math.combScaleIndex),
            ("SOFTMAX_READS", softmaxReadWidth),
            ("EPS_BITS", bitPattern(of: eps)),
            ("POST_GAIN_BITS", bitPattern(of: DeepSeekV4Math.postGain)),
        ]
    }

    /// The bit pattern of a float32 value as a template argument. The source
    /// reads it back with `as_type<float>`, thus the kernel sees the exact
    /// value and not a decimal text of it.
    ///
    /// - Parameter value: The float32 value.
    /// - Returns: The bits of `value`, as a signed integer.
    private static func bitPattern(of value: Float) -> Int {
        Int(Int32(bitPattern: value.bitPattern))
    }
}
