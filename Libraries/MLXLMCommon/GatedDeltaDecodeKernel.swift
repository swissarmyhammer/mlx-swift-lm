// Copyright © 2026 Apple Inc.
//
// One Metal kernel for the body of a gated delta net (GDN) layer on one
// decode token.
//
// Between the fused input projection and the output projection, the op path
// of a GDN layer runs about 13 small kernels on one token: the conv taps, the
// conv state shift, the SiLU, the q and k RMSNorm and scale, the beta and
// decay gates, the delta rule, the gated RMSNorm and the gate. Each of these
// kernels reads and writes a few kilobytes, thus on a decode step the start
// and the end of each kernel cost more than its work.
//
// This kernel does all of them. One threadgroup holds one value head:
//
//  1. Each thread computes the conv taps and the SiLU of some channels of the
//     q and k head of the value head and of the value head itself, into
//     threadgroup memory. The grid also writes the shifted conv state.
//  2. Each SIMD group normalizes q and k, computes the gates, and runs the
//     delta rule on its rows of the recurrent state.
//  3. The first SIMD group applies the gated RMSNorm to the output of the
//     head.
//
// Each step rounds where the op path rounds and computes the sigmoid and the
// softplus in the input type, as the compiled MLX kernels do, thus the kernel
// output stays inside a small tolerance of the op path.

import Foundation
import MLX

/// The body of a gated delta net layer on one decode token, in one Metal
/// kernel.
package enum GatedDeltaDecodeKernel {

    /// The weights of one gated delta net layer that the kernel reads.
    package struct Weights {
        /// The depthwise conv weight, shape `(convChannels, taps, 1)`.
        let conv: MLXArray
        /// The log of the decay rate of each value head, shape `(valueHeads)`.
        let aLog: MLXArray
        /// The bias of the decay step of each value head, shape `(valueHeads)`.
        let dtBias: MLXArray
        /// The weight of the gated RMSNorm, shape `(valueHeadDim)`.
        let norm: MLXArray
        /// The epsilon of the gated RMSNorm.
        let normEps: Float
        /// The epsilon of the q and k RMSNorm.
        let queryKeyNormEps: Float
        /// The factor of the normalized q, before its rounding to the input
        /// type.
        let queryScale: Float
        /// The factor of the normalized k, before its rounding to the input
        /// type.
        let keyScale: Float

        /// Collects the weights of one layer.
        ///
        /// - Parameters:
        ///   - conv: The depthwise conv weight, shape `(convChannels, taps, 1)`.
        ///   - aLog: The log of the decay rate of each value head.
        ///   - dtBias: The bias of the decay step of each value head.
        ///   - norm: The weight of the gated RMSNorm.
        ///   - normEps: The epsilon of the gated RMSNorm.
        ///   - queryKeyNormEps: The epsilon of the q and k RMSNorm.
        ///   - queryScale: The factor of the normalized q.
        ///   - keyScale: The factor of the normalized k.
        package init(
            conv: MLXArray, aLog: MLXArray, dtBias: MLXArray, norm: MLXArray,
            normEps: Float, queryKeyNormEps: Float, queryScale: Float, keyScale: Float
        ) {
            self.conv = conv
            self.aLog = aLog
            self.dtBias = dtBias
            self.norm = norm
            self.normEps = normEps
            self.queryKeyNormEps = queryKeyNormEps
            self.queryScale = queryScale
            self.keyScale = keyScale
        }

        /// The scalar parameters of the kernel, in the order of ``Parameter``.
        var parameters: MLXArray {
            var values = [Float](repeating: 0, count: Parameter.count)
            values[Parameter.queryScale] = queryScale
            values[Parameter.keyScale] = keyScale
            values[Parameter.queryKeyNormEps] = queryKeyNormEps
            values[Parameter.normEps] = normEps
            return MLXArray(values)
        }
    }

    /// The places of the scalar parameters in the `params` input of the
    /// kernel.
    private enum Parameter {
        /// The factor of the normalized q.
        static let queryScale = 0
        /// The factor of the normalized k.
        static let keyScale = 1
        /// The epsilon of the q and k RMSNorm.
        static let queryKeyNormEps = 2
        /// The epsilon of the gated RMSNorm.
        static let normEps = 3
        /// The number of parameters.
        static let count = 4
    }

    /// The per-token inputs of one decode step: the slices of the fused input
    /// projection.
    package struct Inputs {
        /// The conv input of the token, shape `(batch, 1, convChannels)`: the
        /// keys of the q heads, of the k heads, then of the value heads.
        let qkv: MLXArray
        /// The gate of the output, shape `(batch, 1, valueHeads, valueHeadDim)`.
        let z: MLXArray
        /// The beta input, shape `(batch, 1, valueHeads)`.
        let b: MLXArray
        /// The decay input, shape `(batch, 1, valueHeads)`.
        let a: MLXArray

        /// Collects the inputs of one token.
        ///
        /// - Parameters:
        ///   - qkv: The conv input, shape `(batch, 1, convChannels)`.
        ///   - z: The gate of the output, shape `(batch, 1, valueHeads, valueHeadDim)`.
        ///   - b: The beta input, shape `(batch, 1, valueHeads)`.
        ///   - a: The decay input, shape `(batch, 1, valueHeads)`.
        package init(qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray) {
            self.qkv = qkv
            self.z = z
            self.b = b
            self.a = a
        }
    }

    /// The output of one decode step.
    package struct Step {
        /// The gated, normalized output of the heads, shape
        /// `(batch, 1, valueHeads * valueHeadDim)`, the input of the output
        /// projection.
        package let output: MLXArray
        /// The conv state after the token, shape `(batch, taps - 1, convChannels)`.
        package let convState: MLXArray
        /// The recurrent state after the token, shape
        /// `(batch, valueHeads, valueHeadDim, keyHeadDim)`, in float32.
        package let recurrentState: MLXArray
    }

    /// The most threads of one threadgroup: the limit of Metal.
    private static let mostThreadgroupThreads = 1024

    /// The most SIMD groups of one threadgroup.
    private static let mostSimdGroups = mostThreadgroupThreads / metalSimdWidth

    /// The places of the outputs of the kernel, in the order of its
    /// `outputNames`.
    private enum Output {
        /// The gated, normalized output of the heads.
        static let output = 0
        /// The recurrent state after the token.
        static let recurrentState = 1
        /// The conv state after the token.
        static let convState = 2
    }

    /// The most bytes of threadgroup memory of one threadgroup: 32 KB, the
    /// limit of Metal.
    private static let mostThreadgroupBytes = 32_768

    /// The most float32 numbers the threadgroup memory of the kernel holds.
    private static let mostThreadgroupNumbers = mostThreadgroupBytes / MemoryLayout<Float>.size

    /// The dtypes of the token inputs the kernel reads.
    private static let inputTypes: [DType] = [.float16, .bfloat16]

    /// The axes of the `(batch, 1, channels)` token inputs and of the
    /// `(batch, heads, rows, width)` recurrent state.
    private enum Axis {
        /// The axis that holds the sequences.
        static let batch = 0
        /// The axis that holds the tokens of the token inputs.
        static let token = 1
        /// The axis that holds the channels of the conv input and the rows of
        /// the conv state.
        static let channel = 2
        /// The axis that holds the value heads of the recurrent state.
        static let head = 1
        /// The axis that holds the value rows of the recurrent state.
        static let valueRow = 2
        /// The axis that holds the key numbers of the recurrent state.
        static let keyNumber = 3
        /// The number of axes of the recurrent state.
        static let stateCount = 4
        /// The axis that holds the taps of the conv weight.
        static let tap = 1
    }

    /// The fewest conv taps: the conv state holds `taps - 1` rows.
    private static let fewestTaps = 2

    /// The number of key channel blocks of the conv input: the q block and
    /// the k block.
    private static let queryKeyBlocks = 2

    /// The Metal header: the shared gated delta helpers, then the helpers of
    /// this kernel.
    ///
    /// `sigmoid_of` and `softplus_of` copy the `Sigmoid` and `LogAddExp`
    /// operators of MLX, so that in the input type they round as the
    /// compiled MLX kernels round. `conv_silu` computes the conv taps of one
    /// channel in float32, as `decodeConv` of the model does, then the SiLU in
    /// the input type.
    private static let header =
        gatedDeltaMetalHeader + """
            template <typename T>
            inline T sigmoid_of(T x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            template <typename T>
            inline T softplus_of(T x) {
              T zero = T(0);
              T largest = metal::max(x, zero);
              T smallest = metal::min(x, zero);
              return (largest == metal::numeric_limits<T>::infinity())
                  ? largest
                  : (largest + log1p(metal::exp(smallest - largest)));
            }

            template <typename T, typename W>
            inline float conv_silu(
                const device T* state, const device T* token, const device W* weight,
                uint channel, int taps, int channels) {
              float sum = float(state[channel]) * float(weight[channel * taps]);
              for (int tap = 1; tap < taps; ++tap) {
                float x = tap < taps - 1 ? float(state[tap * channels + channel]) : float(token[channel]);
                sum = sum + x * float(weight[channel * taps + tap]);
              }
              T pre = static_cast<T>(sum);
              return float(pre * sigmoid_of(pre));
            }

            """

    /// The kernel body.
    ///
    /// Threadgroup `n` holds value head `n % Hv` of sequence `n / Hv`. SIMD
    /// group `g` holds the value rows `g * ROWS ..< (g + 1) * ROWS`. `params`
    /// holds the scalars of ``Parameter``. `head_inputs` holds the q head,
    /// the k head and the value head, in this order.
    private static let source = """
        constexpr int QK_BLOCKS = \(queryKeyBlocks);
        constexpr int Q_SCALE = \(Parameter.queryScale);
        constexpr int K_SCALE = \(Parameter.keyScale);
        constexpr int QK_EPS = \(Parameter.queryKeyNormEps);
        constexpr int NORM_EPS = \(Parameter.normEps);
        constexpr int KEY_CHANNELS = Hk * Dk;
        constexpr int CHANNELS = QK_BLOCKS * KEY_CHANNELS + Hv * Dv;
        constexpr int NUMBERS = Dk / SIMD_WIDTH;
        constexpr int THREADS = SIMD_GROUPS * SIMD_WIDTH;
        constexpr int ROWS = Dv / SIMD_GROUPS;
        constexpr int V_AT = QK_BLOCKS * Dk;
        constexpr int STATE_ROWS = TAPS - 1;
        threadgroup float head_inputs[V_AT + Dv];
        threadgroup float head_output[Dv];

        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint thread_in_group = simd_group * SIMD_WIDTH + lane;
        uint n = threadgroup_position_in_grid.z;
        uint sequence = n / Hv;
        uint value_head = n % Hv;
        uint key_head = value_head / (Hv / Hk);

        const device InT* token = qkv + sequence * CHANNELS;
        const device InT* state = conv_state + sequence * STATE_ROWS * CHANNELS;

        // The conv state after the token: every thread of the grid writes at
        // most a few numbers.
        uint state_numbers = STATE_ROWS * CHANNELS;
        uint grid_threads = threads_per_grid.x * threads_per_grid.y * threads_per_grid.z;
        uint grid_thread = (thread_position_in_grid.z * threads_per_grid.y + thread_position_in_grid.y)
            * threads_per_grid.x + thread_position_in_grid.x;
        for (uint i = grid_thread; i < threadgroups_per_grid.z / Hv * state_numbers; i += grid_threads) {
          uint from = i / state_numbers;
          uint row = (i % state_numbers) / CHANNELS;
          uint channel = i % CHANNELS;
          conv_out[i] = row + 1 < STATE_ROWS
              ? conv_state[(from * STATE_ROWS + row + 1) * CHANNELS + channel]
              : qkv[from * CHANNELS + channel];
        }

        // The conv and the SiLU of the q head, the k head and the value head.
        for (uint i = thread_in_group; i < V_AT + Dv; i += THREADS) {
          uint channel = i < Dk ? key_head * Dk + i
              : i < V_AT ? KEY_CHANNELS + key_head * Dk + (i - Dk)
              : QK_BLOCKS * KEY_CHANNELS + value_head * Dv + (i - V_AT);
          head_inputs[i] = conv_silu(state, token, conv_w, channel, TAPS, CHANNELS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The q and k RMSNorm and scale, rounded as the op path rounds.
        float q_row[NUMBERS];
        float k_row[NUMBERS];
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int i = 0; i < NUMBERS; ++i) {
          q_row[i] = head_inputs[lane * NUMBERS + i];
          k_row[i] = head_inputs[Dk + lane * NUMBERS + i];
          q_sum += q_row[i] * q_row[i];
          k_sum += k_row[i] * k_row[i];
        }
        float q_inverse = metal::precise::rsqrt(simd_sum(q_sum) / Dk + params[QK_EPS]);
        float k_inverse = metal::precise::rsqrt(simd_sum(k_sum) / Dk + params[QK_EPS]);
        InT q_scale = static_cast<InT>(params[Q_SCALE]);
        InT k_scale = static_cast<InT>(params[K_SCALE]);
        for (int i = 0; i < NUMBERS; ++i) {
          q_row[i] = float(static_cast<InT>(q_row[i] * q_inverse) * q_scale);
          k_row[i] = float(static_cast<InT>(k_row[i] * k_inverse) * k_scale);
        }

        // The gates of the head.
        uint gate_at = sequence * Hv + value_head;
        float beta = float(sigmoid_of(static_cast<InT>(b_in[gate_at])));
        InT shifted = static_cast<InT>(a_in[gate_at]) + static_cast<InT>(dt_bias[value_head]);
        float g = metal::precise::exp(
            -metal::precise::exp(float(a_log[value_head])) * float(softplus_of(shifted)));

        // The delta rule on the rows of this SIMD group.
        for (int r = 0; r < ROWS; ++r) {
          uint value_row = simd_group * ROWS + r;
          size_t row_at = (size_t(n) * Dv + value_row) * Dk + lane * NUMBERS;
          float row[NUMBERS];
          for (int i = 0; i < NUMBERS; ++i) {
            row[i] = static_cast<float>(state_in[row_at + i]);
          }
          float out = gated_delta_row(row, k_row, q_row, g, beta, head_inputs[V_AT + value_row]);
          for (int i = 0; i < NUMBERS; ++i) {
            state_out[row_at + i] = static_cast<StT>(row[i]);
          }
          if (lane == 0) {
            head_output[value_row] = float(static_cast<InT>(out));
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The gated RMSNorm of the head.
        if (simd_group == 0) {
          float sum = 0.0f;
          for (int i = lane; i < Dv; i += SIMD_WIDTH) {
            sum += head_output[i] * head_output[i];
          }
          float inverse = metal::precise::rsqrt(simd_sum(sum) / Dv + params[NORM_EPS]);
          for (int i = lane; i < Dv; i += SIMD_WIDTH) {
            InT normed = static_cast<InT>(norm_w[i]) * static_cast<InT>(head_output[i] * inverse);
            float gate = float(z[size_t(n) * Dv + i]);
            y[size_t(n) * Dv + i] = static_cast<InT>(gate * sigmoid_of(gate) * float(normed));
          }
        }
        """

    /// The kernel.
    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_decode",
        inputNames: [
            "qkv", "z", "b_in", "a_in", "a_log", "dt_bias", "conv_w", "conv_state", "state_in",
            "norm_w", "params",
        ],
        outputNames: ["y", "state_out", "conv_out"],
        source: source,
        header: header)

    /// The head shape of one layer, read from the shapes of a step.
    private struct HeadShape {
        let keyHeads: Int
        let keyHeadDim: Int
        let valueHeads: Int
        let valueHeadDim: Int
        let taps: Int

        /// The number of SIMD groups of one threadgroup: one for each value
        /// row, at most ``mostSimdGroups``.
        var simdGroups: Int { min(valueHeadDim, mostSimdGroups) }
    }

    /// The head shape of a step, or nil when the shapes do not make one.
    private static func headShape(
        inputs: Inputs, recurrentState: MLXArray, weights: Weights
    ) -> HeadShape? {
        guard recurrentState.ndim == Axis.stateCount, inputs.qkv.ndim == Axis.channel + 1
        else { return nil }
        let valueHeads = recurrentState.dim(Axis.head)
        let valueHeadDim = recurrentState.dim(Axis.valueRow)
        let keyHeadDim = recurrentState.dim(Axis.keyNumber)
        let keyChannels = inputs.qkv.dim(Axis.channel) - valueHeads * valueHeadDim
        let keyBlockWidth = queryKeyBlocks * keyHeadDim
        guard valueHeads > 0, keyHeadDim > 0, keyChannels > 0, keyChannels % keyBlockWidth == 0
        else { return nil }
        return HeadShape(
            keyHeads: keyChannels / keyBlockWidth, keyHeadDim: keyHeadDim,
            valueHeads: valueHeads, valueHeadDim: valueHeadDim,
            taps: weights.conv.dim(Axis.tap))
    }

    /// Tells if the kernel runs a head shape: the lanes share the key numbers
    /// evenly, the SIMD groups share the value rows evenly, the value heads
    /// share the key heads evenly, and the head fits the threadgroup memory:
    /// the q, k and value inputs of the head and the value output of the head.
    private static func fits(_ shape: HeadShape) -> Bool {
        let threadgroupNumbers =
            queryKeyBlocks * shape.keyHeadDim + shape.valueHeadDim + shape.valueHeadDim
        return shape.keyHeadDim % metalSimdWidth == 0
            && shape.valueHeadDim % shape.simdGroups == 0
            && shape.valueHeads % shape.keyHeads == 0
            && shape.taps >= fewestTaps
            && threadgroupNumbers <= mostThreadgroupNumbers
    }

    /// Tells if the shapes of the operands agree with the head shape.
    private static func shapesAgree(
        _ shape: HeadShape, inputs: Inputs, convState: MLXArray, recurrentState: MLXArray,
        weights: Weights
    ) -> Bool {
        let batch = inputs.qkv.dim(Axis.batch)
        let channels = inputs.qkv.dim(Axis.channel)
        let gates = [batch, 1, shape.valueHeads]
        return inputs.qkv.dim(Axis.token) == 1
            && inputs.z.shape == [batch, 1, shape.valueHeads, shape.valueHeadDim]
            && inputs.b.shape == gates && inputs.a.shape == gates
            && convState.shape == [batch, shape.taps - 1, channels]
            && recurrentState.dim(Axis.batch) == batch
            && weights.conv.shape == [channels, shape.taps, 1]
            && weights.aLog.shape == [shape.valueHeads]
            && weights.dtBias.shape == [shape.valueHeads]
            && weights.norm.shape == [shape.valueHeadDim]
    }

    /// Tells if the dtypes of the operands are the ones the kernel reads: the
    /// token inputs, the conv state, the decay bias and the norm weight in
    /// one float16 or bfloat16 type, and the recurrent state in float32.
    private static func dtypesAgree(
        inputs: Inputs, convState: MLXArray, recurrentState: MLXArray, weights: Weights
    ) -> Bool {
        let type = inputs.qkv.dtype
        return inputTypes.contains(type)
            && [inputs.z, inputs.b, inputs.a, convState, weights.dtBias, weights.norm]
                .allSatisfy { $0.dtype == type }
            && recurrentState.dtype == .float32
    }

    /// Tells if the kernel runs one decode step: the default device is the
    /// GPU, the step holds one token, and the shapes and the dtypes of the
    /// operands are the ones the kernel reads.
    ///
    /// - Parameters:
    ///   - inputs: The token inputs.
    ///   - convState: The conv state before the token.
    ///   - recurrentState: The recurrent state before the token.
    ///   - weights: The weights of the layer.
    /// - Returns: True when ``step(inputs:convState:recurrentState:weights:)``
    ///   runs the step.
    package static func canRun(
        inputs: Inputs, convState: MLXArray, recurrentState: MLXArray, weights: Weights
    ) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu,
            let shape = headShape(inputs: inputs, recurrentState: recurrentState, weights: weights)
        else { return false }
        return fits(shape)
            && shapesAgree(
                shape, inputs: inputs, convState: convState, recurrentState: recurrentState,
                weights: weights)
            && dtypesAgree(
                inputs: inputs, convState: convState, recurrentState: recurrentState,
                weights: weights)
    }

    /// One decode step of one gated delta net layer.
    ///
    /// - Parameters:
    ///   - inputs: The token inputs.
    ///   - convState: The conv state before the token.
    ///   - recurrentState: The recurrent state before the token, in float32.
    ///   - weights: The weights of the layer.
    /// - Returns: The output of the heads and the states after the token.
    /// - Precondition: ``canRun(inputs:convState:recurrentState:weights:)``.
    package static func step(
        inputs: Inputs, convState: MLXArray, recurrentState: MLXArray, weights: Weights
    ) -> Step {
        precondition(
            canRun(
                inputs: inputs, convState: convState, recurrentState: recurrentState,
                weights: weights),
            "the gated delta decode kernel cannot run a step of conv input \(inputs.qkv.shape) "
                + "and recurrent state \(recurrentState.shape)")
        guard
            let shape = headShape(inputs: inputs, recurrentState: recurrentState, weights: weights)
        else { preconditionFailure("canRun checks the head shape") }
        let batch = inputs.qkv.dim(Axis.batch)
        let outputs = kernel(
            [
                inputs.qkv, inputs.z, inputs.b, inputs.a, weights.aLog, weights.dtBias,
                weights.conv, convState, recurrentState, weights.norm, weights.parameters,
            ],
            template: [
                ("InT", inputs.qkv.dtype), ("StT", recurrentState.dtype),
                ("Dk", shape.keyHeadDim), ("Dv", shape.valueHeadDim), ("Hk", shape.keyHeads),
                ("Hv", shape.valueHeads), ("TAPS", shape.taps), ("SIMD_GROUPS", shape.simdGroups),
            ],
            grid: (metalSimdWidth, shape.simdGroups, batch * shape.valueHeads),
            threadGroup: (metalSimdWidth, shape.simdGroups, 1),
            outputShapes: [
                [batch, 1, shape.valueHeads * shape.valueHeadDim], recurrentState.shape,
                convState.shape,
            ],
            outputDTypes: [inputs.qkv.dtype, recurrentState.dtype, convState.dtype])
        return Step(
            output: outputs[Output.output], convState: outputs[Output.convState],
            recurrentState: outputs[Output.recurrentState])
    }
}
