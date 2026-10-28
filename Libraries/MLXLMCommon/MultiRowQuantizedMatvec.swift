// Copyright © 2026 Apple Inc.
//
// A Metal kernel for the quantized projections of a speculative verify of 3
// to 8 rows, which reads each weight from memory one time for all rows.
//
// For 2 or more rows MLX runs `qmv_wide`, which reads the weights again for
// each tile of rows and sums each row in another order than `qmv_fast`, the
// kernel of a plain decode of one row. The matmuls of a Qwen3.8 step on the
// M5 Ultra then cost 1.6 times a plain step for 3 rows, and 3.4 times for 6
// rows. This kernel gives each lane the input numbers that `qmv_fast` gives
// it (MLX `quantized.h` and `fp_quantized.h`) and adds them in the same
// order, so each output row has the same bits as the plain decode of that
// row. It decodes the weights of each block one time and reuses them for
// every row; it costs 1.3 times a plain step for 3 rows and 2.6 times for 6.

import Foundation
import MLX
import MLXNN

/// The quantized projection of a speculative verify of 3 to 8 rows, which
/// reads each weight one time for all rows.
///
/// Each output row has the same bits as `quantizedMM` over that row alone,
/// the matvec of a plain decode step. Thus a verify that runs its
/// projections through ``quantizedMatmulRows(_:_:)`` gives the logits of a
/// plain decode for the quantized matmuls.
///
/// A model runs its projections through ``projection(_:by:)``, which takes
/// the kernel only inside the scope of a speculative verify
/// (``run(asSpeculativeVerify:_:)``). Every other call of 3 to 8 rows keeps
/// the matvec of MLX and its bits.
public enum MultiRowQuantizedMatvec {

    /// The fewest rows the kernel runs. One row keeps the `qmv_fast` of MLX.
    /// Two rows keep `qmv_wide`: the 2-token MTP verify of Qwen3.8 ran about
    /// 11% slower end to end through the kernel on the M5 Ultra.
    static let fewestRows = 3

    /// The most rows the kernel runs, the longest verify it serves.
    static let mostRows = 8

    /// The most rows one threadgroup holds. A longer call splits its rows
    /// over threadgroups of equal size side by side in the grid, so the
    /// second threadgroup of an output tile reads the weights from the cache.
    static let mostRowsPerThreadgroup = 6

    /// The number of SIMD groups of one threadgroup, as in `qmv_fast`.
    private static let simdGroups = 2

    /// The output rows one SIMD group computes, as in `qmv_fast`.
    private static let resultsPerSimdGroup = 4

    /// The packed words one lane reads from one weight row in each block, as
    /// in `qmv_fast` for 4 and 8 bits.
    private static let packsPerLane = 2

    /// The bits of one packed word of the weights.
    private static let packBits = 32

    /// The narrow affine bit width: `qdot` reads four weights from each
    /// 16-bit half of a packed word.
    private static let narrowAffineBits = 4

    /// The wide affine bit width: `qdot` reads one weight from each byte.
    private static let wideAffineBits = 8

    /// The affine bit widths the kernel reads.
    private static let affineBits: Set<Int> = [narrowAffineBits, wideAffineBits]

    /// The smallest affine group size the kernel reads.
    private static let smallAffineGroupSize = 32

    /// The default affine group size of MLX.
    private static let defaultAffineGroupSize = 64

    /// The largest affine group size the kernel reads.
    private static let largeAffineGroupSize = 128

    /// The affine group sizes the kernel reads.
    private static let affineGroupSizes: Set<Int> = [
        smallAffineGroupSize, defaultAffineGroupSize, largeAffineGroupSize,
    ]

    /// The bit width and the group size of mxfp4.
    private static let mxfp4Bits = 4
    private static let mxfp4GroupSize = 32

    /// The axes of the packed weights: `(N, K * BITS / 32)`.
    private static let weightAxes = 2

    /// The input dtypes the kernel reads.
    private static let inputTypes: Set<DType> = [.bfloat16, .float16]

    /// The Metal header: the vector load and the dot products of `qmv_fast`
    /// and `fp_qmv_fast` with their order of operations, and the two kernel
    /// bodies.
    ///
    /// `qdot` of MLX multiplies each input number by the masked integer of
    /// its weight (`ws[i] & 0x00f0` for the second number of a 4-bit pack),
    /// or by the decoded fp4 value. The `decode_*` functions turn the same
    /// integers and fp4 values into half precision one time for each block,
    /// which is exact, and the `*_dot` functions run the sums of `qdot` over
    /// them.
    ///
    /// Each body decodes one block of its four output rows, then reads the
    /// input rows one after the other. The SIMD sum closes each row as in
    /// `qmv_fast`.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header =
        metalSimdWidthDeclaration + """
            constexpr constant int SIMD_GROUPS = \(simdGroups);
            constexpr constant int RESULTS = \(resultsPerSimdGroup);
            constexpr constant int PACKS = \(packsPerLane);
            constexpr constant int PACK_BITS = \(packBits);
            constexpr constant int BYTES_PER_PACK = PACK_BITS / 8;
            #define UNROLL_FULL _Pragma("clang loop unroll(full)")

            /// A decoded weight. Every masked integer and fp4 value the
            /// kernels decode is exact in half precision.
            typedef half decoded_weight;

            template <typename T, int VALUES, int BITS>
            inline float affine_load_vector(const device T* x, thread float (&x_thread)[VALUES]) {
                float sum = 0;
                if (BITS == 4) {
                    UNROLL_FULL for (int i = 0; i < VALUES; i += 4) {
                        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                        x_thread[i] = x[i];
                        x_thread[i + 1] = x[i + 1] / 16.0f;
                        x_thread[i + 2] = x[i + 2] / 256.0f;
                        x_thread[i + 3] = x[i + 3] / 4096.0f;
                    }
                } else {
                    UNROLL_FULL for (int i = 0; i < VALUES; i++) {
                        sum += x[i];
                        x_thread[i] = x[i];
                    }
                }
                return sum;
            }

            template <typename T, int VALUES>
            inline void load_vector(const device T* x, thread float (&x_thread)[VALUES]) {
                UNROLL_FULL for (int i = 0; i < VALUES; i++) {
                    x_thread[i] = x[i];
                }
            }

            /// The masked integers `qdot` multiplies by: for 4 bits each
            /// 16-bit half of a word gives `half & 0x000f`, `half & 0x00f0`,
            /// `half & 0x0f00` and `half & 0xf000`; for 8 bits each byte.
            template <int BITS, int VALUES>
            inline void decode_affine(const device uint32_t* wl, thread decoded_weight (&w)[VALUES]) {
                UNROLL_FULL for (int pack = 0; pack < PACKS; pack++) {
                    const uint32_t word = wl[pack];
                    if (BITS == 4) {
                        UNROLL_FULL for (int half_index = 0; half_index < 2; half_index++) {
                            const uint32_t half_word = (word >> (16 * half_index)) & 0xffff;
                            UNROLL_FULL for (int j = 0; j < 4; j++) {
                                w[pack * 8 + half_index * 4 + j] = half_word & (0xfu << (4 * j));
                            }
                        }
                    } else {
                        UNROLL_FULL for (int j = 0; j < 4; j++) {
                            w[pack * 4 + j] = (word >> (8 * j)) & 0xff;
                        }
                    }
                }
            }

            /// The fp4 values `fp_qdot` multiplies by, nibble by nibble.
            template <int VALUES>
            inline void decode_fp4(const device uint32_t* wl, thread decoded_weight (&w)[VALUES]) {
                UNROLL_FULL for (int pack = 0; pack < PACKS; pack++) {
                    const uint32_t word = wl[pack];
                    UNROLL_FULL for (int j = 0; j < 8; j++) {
                        const uint8_t bits = (word >> (4 * j)) & 0xf;
                        half converted = as_type<half>(ushort((bits & 7) << 9));
                        converted *= 16384.0;
                        w[pack * 8 + j] = bits & 8 ? -converted : converted;
                    }
                }
            }

            inline float fp8_e8m0_value(uint8_t bits) {
                uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
                return as_type<float>(out);
            }

            template <int BITS, int VALUES>
            inline float affine_dot(
                thread const decoded_weight (&w)[VALUES], thread const float (&x_thread)[VALUES],
                float scale, float bias, float sum) {
                float accum = 0;
                if (BITS == 4) {
                    UNROLL_FULL for (int i = 0; i < (VALUES / 4); i++) {
                        accum +=
                            (x_thread[4 * i] * w[4 * i] +
                             x_thread[4 * i + 1] * w[4 * i + 1] +
                             x_thread[4 * i + 2] * w[4 * i + 2] +
                             x_thread[4 * i + 3] * w[4 * i + 3]);
                    }
                } else {
                    UNROLL_FULL for (int i = 0; i < VALUES; i++) {
                        accum += x_thread[i] * w[i];
                    }
                }
                return scale * accum + sum * bias;
            }

            template <int VALUES>
            inline float fp4_dot(
                thread const decoded_weight (&w)[VALUES], thread const float (&x_thread)[VALUES],
                float scale) {
                float accum = 0;
                UNROLL_FULL for (int i = 0; i < (VALUES / 4); i++) {
                    accum +=
                        (x_thread[4 * i] * w[4 * i] +
                         x_thread[4 * i + 1] * w[4 * i + 1] +
                         x_thread[4 * i + 2] * w[4 * i + 2] +
                         x_thread[4 * i + 3] * w[4 * i + 3]);
                }
                return scale * accum;
            }

            /// The place of one lane in the weights, the inputs and the
            /// outputs, as in `qmv_fast`.
            template <int BITS, int GROUP_SIZE>
            struct LaneLayout {
                static constexpr constant int pack_factor = PACK_BITS / BITS;
                static constexpr constant int values = pack_factor * PACKS;
                static constexpr constant int block = values * SIMD_WIDTH;
                static constexpr constant int scale_step = GROUP_SIZE / values;

                int row_bytes;
                int row_groups;
                int out_row;

                LaneLayout(int in_vec_size, uint3 tid, uint simd_gid) thread
                    : row_bytes(in_vec_size * BYTES_PER_PACK / pack_factor),
                      row_groups(in_vec_size / GROUP_SIZE),
                      out_row(tid.y * (SIMD_GROUPS * RESULTS) + simd_gid * RESULTS) {}

                int weight_offset(uint simd_lid) const thread {
                    return out_row * row_bytes + simd_lid * PACKS * BYTES_PER_PACK;
                }

                int scale_offset(uint simd_lid) const thread {
                    return out_row * row_groups + simd_lid / scale_step;
                }
            };

            /// Closes each row with the SIMD sum of `qmv_fast` and writes it.
            template <typename T, int ROWS>
            inline void store_rows(
                thread float (&result)[ROWS][RESULTS], device T* y, int out_vec_size,
                int row_count, uint simd_lid) {
                UNROLL_FULL for (int r = 0; r < ROWS; r++) {
                    UNROLL_FULL for (int row = 0; row < RESULTS; row++) {
                        float total = simd_sum(result[r][row]);
                        if (simd_lid == 0 && r < row_count) {
                            y[r * out_vec_size + row] = static_cast<T>(total);
                        }
                    }
                }
            }

            template <typename T, int GROUP_SIZE, int BITS, int ROWS>
            inline void affine_rows(
                const device T* x, const device uint32_t* w, const device T* scales,
                const device T* biases, device T* y, int in_vec_size, int out_vec_size,
                int first_row, int row_count, uint3 tid, uint simd_gid, uint simd_lid) {
                typedef LaneLayout<BITS, GROUP_SIZE> Layout;
                const Layout layout(in_vec_size, tid, simd_gid);
                const device uint8_t* ws = (const device uint8_t*)w + layout.weight_offset(simd_lid);
                scales += layout.scale_offset(simd_lid);
                biases += layout.scale_offset(simd_lid);
                x += first_row * in_vec_size + simd_lid * Layout::values;
                y += first_row * out_vec_size + layout.out_row;

                float result[ROWS][RESULTS] = {};
                for (int k = 0; k < in_vec_size; k += Layout::block) {
                    decoded_weight weights[RESULTS][Layout::values];
                    float s[RESULTS];
                    float b[RESULTS];
                    UNROLL_FULL for (int row = 0; row < RESULTS; row++) {
                        decode_affine<BITS, Layout::values>(
                            (const device uint32_t*)(ws + row * layout.row_bytes), weights[row]);
                        s[row] = scales[row * layout.row_groups];
                        b[row] = biases[row * layout.row_groups];
                    }
                    UNROLL_FULL for (int r = 0; r < ROWS; r++) {
                        if (r < row_count) {
                            float x_thread[Layout::values];
                            const float sum = affine_load_vector<T, Layout::values, BITS>(
                                x + r * in_vec_size, x_thread);
                            UNROLL_FULL for (int row = 0; row < RESULTS; row++) {
                                result[r][row] += affine_dot<BITS, Layout::values>(
                                    weights[row], x_thread, s[row], b[row], sum);
                            }
                        }
                    }
                    ws += Layout::block * BYTES_PER_PACK / Layout::pack_factor;
                    scales += Layout::block / GROUP_SIZE;
                    biases += Layout::block / GROUP_SIZE;
                    x += Layout::block;
                }
                store_rows<T, ROWS>(result, y, out_vec_size, row_count, simd_lid);
            }

            template <typename T, int GROUP_SIZE, int ROWS>
            inline void mxfp4_rows(
                const device T* x, const device uint32_t* w, const device uint8_t* scales,
                device T* y, int in_vec_size, int out_vec_size, int first_row, int row_count,
                uint3 tid, uint simd_gid, uint simd_lid) {
                typedef LaneLayout<4, GROUP_SIZE> Layout;
                const Layout layout(in_vec_size, tid, simd_gid);
                const device uint8_t* ws = (const device uint8_t*)w + layout.weight_offset(simd_lid);
                scales += layout.scale_offset(simd_lid);
                x += first_row * in_vec_size + simd_lid * Layout::values;
                y += first_row * out_vec_size + layout.out_row;

                float result[ROWS][RESULTS] = {};
                for (int k = 0; k < in_vec_size; k += Layout::block) {
                    decoded_weight weights[RESULTS][Layout::values];
                    float s[RESULTS];
                    UNROLL_FULL for (int row = 0; row < RESULTS; row++) {
                        decode_fp4<Layout::values>(
                            (const device uint32_t*)(ws + row * layout.row_bytes), weights[row]);
                        s[row] = fp8_e8m0_value(scales[row * layout.row_groups]);
                    }
                    UNROLL_FULL for (int r = 0; r < ROWS; r++) {
                        if (r < row_count) {
                            float x_thread[Layout::values];
                            load_vector<T, Layout::values>(x + r * in_vec_size, x_thread);
                            UNROLL_FULL for (int row = 0; row < RESULTS; row++) {
                                result[r][row] += fp4_dot<Layout::values>(
                                    weights[row], x_thread, s[row]);
                            }
                        }
                    }
                    ws += Layout::block * BYTES_PER_PACK / Layout::pack_factor;
                    scales += Layout::block / GROUP_SIZE;
                    x += Layout::block;
                }
                store_rows<T, ROWS>(result, y, out_vec_size, row_count, simd_lid);
            }

            """

    /// The start of the kernel bodies: the rows of this threadgroup. The row
    /// groups of one output tile stand side by side along the grid x axis.
    private static let rowRangeSource = """
        const int first_row = int(threadgroup_position_in_grid.x) * ROWS;
        const int row_count = min(ROWS, int(x_shape[0]) - first_row);

        """

    /// The kernel body for affine weights. `x` is `(rows, K)`, `w` is
    /// `(N, K * BITS / 32)`, and `scales` and `biases` are `(N, K / GROUP_SIZE)`.
    private static let affineSource =
        rowRangeSource + """
            affine_rows<T, GROUP_SIZE, BITS, ROWS>(
                x, w, scales, biases, y, int(x_shape[1]), int(w_shape[0]), first_row, row_count,
                threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """

    /// The kernel body for mxfp4 weights. `scales` is `(N, K / 32)` of e8m0
    /// bytes.
    private static let mxfp4Source =
        rowRangeSource + """
            mxfp4_rows<T, GROUP_SIZE, ROWS>(
                x, w, scales, y, int(x_shape[1]), int(w_shape[0]), first_row, row_count,
                threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """

    private static let affineKernel = MLXFast.metalKernel(
        name: "multi_row_affine_qmv",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: affineSource,
        header: header)

    private static let mxfp4Kernel = MLXFast.metalKernel(
        name: "multi_row_mxfp4_qmv",
        inputNames: ["x", "w", "scales"],
        outputNames: ["y"],
        source: mxfp4Source,
        header: header)

    /// The projection of `x` by `layer`, with each output row bit-identical
    /// to the projection of that row alone.
    ///
    /// The kernel runs when ``canRun(_:_:)`` holds: 3 to 8 rows through a
    /// stock `QuantizedLinear` whose single-row matvec is the `qmv_fast` of
    /// MLX. Every other call, one or two rows among them, runs `layer(x)`.
    ///
    /// - Parameters:
    ///   - x: The input, shape `(..., K)`.
    ///   - layer: The projection. A layer that is not a stock
    ///     `QuantizedLinear` always runs its own call.
    /// - Returns: The projection, shape `(..., N)`, of the dtype of `x`.
    public static func quantizedMatmulRows(_ x: MLXArray, _ layer: Linear) -> MLXArray {
        guard canRun(x, layer), let quantized = layer as? QuantizedLinear else {
            return layer(x)
        }
        let output = project(x.reshaped(-1, x.dim(-1)), quantized)
            .reshaped(Array(x.shape.dropLast()) + [quantized.weight.dim(0)])
        guard let bias = quantized.bias else { return output }
        return output + bias
    }

    /// The scope of a speculative verify: true while
    /// ``run(asSpeculativeVerify:_:)`` runs the forward pass of a verify.
    private static let speculativeVerifyScope = TaskLocal<Bool>(wrappedValue: false)

    /// Runs `body`, the forward pass of a model call, inside or outside the
    /// scope of a speculative verify.
    ///
    /// Inside the scope, ``projection(_:by:)`` runs 3 to 8 rows through the
    /// kernel. A model enters it only for a verify, so a prefill (the prompt
    /// tail after a checkpoint among them), a plain decode step and the
    /// drafter keep the matvec of MLX. A compiled trace of a verify reads the
    /// scope when it traces, inside the scope.
    ///
    /// - Parameters:
    ///   - isSpeculativeVerify: True when `body` is the forward pass of a
    ///     speculative verify.
    ///   - body: The forward pass.
    /// - Returns: The result of `body`.
    package static func run<Result>(
        asSpeculativeVerify isSpeculativeVerify: Bool, _ body: () throws -> Result
    ) rethrows -> Result {
        try speculativeVerifyScope.withValue(isSpeculativeVerify, operation: body)
    }

    /// The projection of `x` by `layer`: through
    /// ``quantizedMatmulRows(_:_:)`` inside the scope of a speculative verify
    /// (``run(asSpeculativeVerify:_:)``), else `layer(x)`.
    ///
    /// - Parameters:
    ///   - x: The input, shape `(..., K)`.
    ///   - layer: The projection.
    /// - Returns: The projection, shape `(..., N)`.
    package static func projection(_ x: MLXArray, by layer: Linear) -> MLXArray {
        guard speculativeVerifyScope.get() else { return layer(x) }
        return quantizedMatmulRows(x, layer)
    }

    /// Tells if the kernel runs the projection of `x` by `layer`.
    ///
    /// It runs on the GPU for 3 to 8 rows of float16 or bfloat16, through a
    /// stock `QuantizedLinear` with no global scale, of affine 4 or 8 bits in
    /// groups of 32, 64 or 128, or of mxfp4, when MLX runs `qmv_fast` for one
    /// row: the outputs fill whole tiles of 8 and the input width fills whole
    /// blocks of the kernel.
    ///
    /// - Parameters:
    ///   - x: The input, shape `(..., K)`.
    ///   - layer: The projection.
    /// - Returns: True when ``quantizedMatmulRows(_:_:)`` runs the kernel.
    public static func canRun(_ x: MLXArray, _ layer: Linear) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu,
            type(of: layer) == QuantizedLinear.self,
            let quantized = layer as? QuantizedLinear,
            quantized.globalScale == nil,
            readsFormat(quantized),
            inputTypes.contains(x.dtype),
            x.ndim >= 1
        else { return false }
        let width = x.dim(-1)
        let rows = width > 0 ? x.size / width : 0
        return (fewestRows ... mostRows).contains(rows)
            && fitsTiles(quantized, width: width, inputType: x.dtype)
    }

    /// Tells if the kernel reads the quantization format of `layer`.
    private static func readsFormat(_ layer: QuantizedLinear) -> Bool {
        switch layer.mode {
        case .affine:
            return affineBits.contains(layer.bits)
                && affineGroupSizes.contains(layer.groupSize)
                && layer.biases != nil
        case .mxfp4:
            return layer.bits == mxfp4Bits && layer.groupSize == mxfp4GroupSize
        case .mxfp8, .nvfp4:
            return false
        }
    }

    /// Tells if the weights, the scales and the biases of `layer` have the
    /// layout of a `(N, width)` projection that the single-row `qmv_fast` of
    /// MLX runs: the outputs fill whole tiles of 8 and the input fills whole
    /// blocks of one SIMD group.
    private static func fitsTiles(_ layer: QuantizedLinear, width: Int, inputType: DType) -> Bool {
        let outputs = layer.weight.dim(0)
        let valuesPerLane = (packBits / layer.bits) * packsPerLane
        let blockWidth = valuesPerLane * metalSimdWidth
        let tileRows = simdGroups * resultsPerSimdGroup
        let groups = width / layer.groupSize
        let scaleType: DType = layer.mode == .affine ? inputType : .uint8
        return layer.weight.ndim == weightAxes && layer.weight.dtype == .uint32
            && layer.weight.dim(1) * packBits == width * layer.bits
            && layer.scales.shape == [outputs, groups] && layer.scales.dtype == scaleType
            && (layer.biases.map { $0.shape == [outputs, groups] && $0.dtype == inputType }
                ?? true)
            && outputs > 0 && outputs % tileRows == 0
            && width > 0 && width % blockWidth == 0
    }

    /// The rows each threadgroup holds: the rows split evenly over the
    /// fewest threadgroups that hold at most ``mostRowsPerThreadgroup``.
    static func rowsPerThreadgroup(rows: Int) -> Int {
        let threadgroups = (rows + mostRowsPerThreadgroup - 1) / mostRowsPerThreadgroup
        return (rows + threadgroups - 1) / threadgroups
    }

    /// Runs the kernel over `x`, shape `(rows, K)`.
    private static func project(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray {
        let rows = x.dim(0)
        let outputs = layer.weight.dim(0)
        let rowsPerGroup = rowsPerThreadgroup(rows: rows)
        let rowGroups = (rows + rowsPerGroup - 1) / rowsPerGroup
        let tileRows = simdGroups * resultsPerSimdGroup
        let grid = (metalSimdWidth * rowGroups, simdGroups * (outputs / tileRows), 1)
        let threadGroup = (metalSimdWidth, simdGroups, 1)
        var template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("GROUP_SIZE", layer.groupSize), ("ROWS", rowsPerGroup),
        ]
        let inputs: [MLXArray]
        let kernel: MLXFast.MLXFastKernel
        if let biases = layer.biases, layer.mode == .affine {
            template.append(("BITS", layer.bits))
            inputs = [x, layer.weight, layer.scales, biases]
            kernel = affineKernel
        } else {
            inputs = [x, layer.weight, layer.scales]
            kernel = mxfp4Kernel
        }
        return kernel(
            inputs, template: template, grid: grid, threadGroup: threadGroup,
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
    }
}
