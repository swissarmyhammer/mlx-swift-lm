// Copyright © 2026 Apple Inc.
//
// One Metal kernel for the body of a gated delta net (GDN) layer on one
// decode token, or on the few tokens of a speculative verify.
//
// Between the fused input projection and the output projection, the op path
// of a GDN layer runs about 13 small kernels on one token: the conv taps, the
// conv state shift, the SiLU, the q and k RMSNorm and scale, the beta and
// decay gates, the delta rule, the gated RMSNorm and the gate. Each of these
// kernels reads and writes a few kilobytes, thus on a decode step the start
// and the end of each kernel cost more than its work.
//
// This kernel does all of them. One threadgroup holds one value head, and
// its SIMD groups keep their rows of the recurrent state in registers over
// the tokens. For each token:
//
//  1. Each thread computes the conv taps and the SiLU of some channels of the
//     q and k head of the value head and of the value head itself, into
//     threadgroup memory. The taps read the conv input window: the conv state,
//     then the tokens.
//  2. Each SIMD group normalizes q and k, computes the gates, and runs the
//     delta rule on its rows of the recurrent state. A verify also writes the
//     rows after the token into the snapshot slot of the token.
//  3. The first SIMD group applies the gated RMSNorm to the output of the
//     head.
//
// The grid also writes the conv state after the last token, and the SIMD
// groups write the recurrent state after the last token.
//
// Each step rounds where the op path rounds and computes the sigmoid and the
// softplus in the input type, as the compiled MLX kernels do, thus the kernel
// output stays inside a small tolerance of the op path. A run of several
// tokens computes each token as a run of that one token does, thus each
// snapshot slot holds the bits of the state that one-token runs reach.
//
// The snapshot slots follow the design of the `kernel_gated_delta_net_impl`
// Metal kernel of llama.cpp: a rollback of a verify selects one slot, and the
// conv state of a slot is a view of the conv input window.

import Foundation
import MLX
import MLXNN

/// The body of a gated delta net layer on one decode token, or on the tokens
/// of a speculative verify, in one Metal kernel.
package enum GatedDeltaDecodeKernel {

    /// The most tokens of one run: the most tokens of a speculative verify
    /// the kernel holds.
    package static let mostTokens = 6

    /// The tokens of one decode step.
    package static let decodeTokens = 1

    /// The most tokens a rollback of a verify the kernel holds rewinds: its
    /// drafts. A round always keeps the bonus token before them.
    package static let mostRewoundTokens = mostTokens - decodeTokens

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

    /// The per-token inputs of one run: the slices of the fused input
    /// projection, for `tokens` tokens of each sequence.
    package struct Inputs {
        /// The conv input of the tokens, shape `(batch, tokens, convChannels)`:
        /// the keys of the q heads, of the k heads, then of the value heads.
        let qkv: MLXArray
        /// The gate of the output, shape
        /// `(batch, tokens, valueHeads, valueHeadDim)`.
        let z: MLXArray
        /// The beta input, shape `(batch, tokens, valueHeads)`.
        let b: MLXArray
        /// The decay input, shape `(batch, tokens, valueHeads)`.
        let a: MLXArray

        /// Collects the inputs of the tokens.
        ///
        /// - Parameters:
        ///   - qkv: The conv input, shape `(batch, tokens, convChannels)`.
        ///   - z: The gate of the output, shape
        ///     `(batch, tokens, valueHeads, valueHeadDim)`.
        ///   - b: The beta input, shape `(batch, tokens, valueHeads)`.
        ///   - a: The decay input, shape `(batch, tokens, valueHeads)`.
        package init(qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray) {
            self.qkv = qkv
            self.z = z
            self.b = b
            self.a = a
        }

        /// The number of tokens of each sequence.
        var tokenCount: Int { qkv.dim(GatedDeltaAxis.step) }
    }

    /// The output of one run.
    package struct Step {
        /// The gated, normalized output of the heads, shape
        /// `(batch, tokens, valueHeads * valueHeadDim)`, the input of the
        /// output projection.
        package let output: MLXArray
        /// The conv state after the last token, shape
        /// `(batch, taps - 1, convChannels)`.
        package let convState: MLXArray
        /// The recurrent state after the last token, shape
        /// `(batch, valueHeads, valueHeadDim, keyHeadDim)`, in float32.
        package let recurrentState: MLXArray
        /// The states after each token, when the run takes snapshots.
        package let snapshots: Snapshots?
    }

    /// The states after each token of a speculative verify: one snapshot slot
    /// of the recurrent state for each token, and the conv input window that
    /// holds the conv state after each token.
    ///
    /// A rollback of `r` tokens of a verify of `T` tokens selects the states
    /// after token `T - r`. The slot of the recurrent state is a view of one
    /// `(tokens, …)` buffer, and the conv state is a view of the window: a
    /// holder that keeps a selected state copies it, so that the buffer can go.
    package struct Snapshots {
        /// The recurrent state after each token, shape
        /// `(tokens, batch, valueHeads, valueHeadDim, keyHeadDim)`, in
        /// float32: slot `t` holds the state after token `t + 1`.
        package let recurrent: MLXArray
        /// The conv input window, shape
        /// `(batch, taps - 1 + tokens, convChannels)`: the conv state before
        /// the first token, then the conv input of each token.
        package let convWindow: MLXArray

        /// Collects the states after each token of a verify.
        ///
        /// - Parameters:
        ///   - recurrent: The recurrent state after each token, shape
        ///     `(tokens, batch, valueHeads, valueHeadDim, keyHeadDim)`.
        ///   - convWindow: The conv input window, shape
        ///     `(batch, taps - 1 + tokens, convChannels)`.
        package init(recurrent: MLXArray, convWindow: MLXArray) {
            self.recurrent = recurrent
            self.convWindow = convWindow
        }

        /// The number of tokens of the verify: the number of slots.
        package var tokenCount: Int { recurrent.dim(SnapshotAxis.token) }

        /// The number of rows of one conv state: the conv taps less one.
        private var convStateRows: Int { convWindow.dim(GatedDeltaAxis.step) - tokenCount }

        /// The recurrent state after a token of the verify: a view of its slot.
        ///
        /// - Parameter token: The token, from 1 to ``tokenCount``.
        /// - Returns: The recurrent state, shape
        ///   `(batch, valueHeads, valueHeadDim, keyHeadDim)`.
        package func recurrentState(afterToken token: Int) -> MLXArray {
            precondition(
                (1 ... tokenCount).contains(token),
                "a verify of \(tokenCount) tokens has no slot after token \(token)")
            return recurrent[token - 1]
        }

        /// The conv state after a token of the verify: a view of the window.
        ///
        /// - Parameter token: The token, from 1 to ``tokenCount``.
        /// - Returns: The conv state, shape `(batch, taps - 1, convChannels)`.
        package func convState(afterToken token: Int) -> MLXArray {
            precondition(
                (1 ... tokenCount).contains(token),
                "a verify of \(tokenCount) tokens has no conv state after token \(token)")
            return convWindow[0..., token ..< token + convStateRows]
        }
    }

    /// The axes of the snapshot slots of the recurrent state.
    private enum SnapshotAxis {
        /// The axis that holds the slot of each token.
        static let token = 0
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
        /// The recurrent state after the last token.
        static let recurrentState = 1
        /// The conv state after the last token.
        static let convState = 2
        /// The recurrent state after each token, of a run that takes
        /// snapshots.
        static let snapshots = 3
    }

    /// The most bytes of threadgroup memory of one threadgroup: 32 KB, the
    /// limit of Metal.
    private static let mostThreadgroupBytes = 32_768

    /// The most float32 numbers the threadgroup memory of the kernel holds.
    private static let mostThreadgroupNumbers = mostThreadgroupBytes / MemoryLayout<Float>.size

    /// The dtypes of the token inputs the kernel reads.
    private static let inputTypes: [DType] = [.float16, .bfloat16]

    /// The axes of this kernel that ``GatedDeltaAxis`` and ``HeadRowsAxis``
    /// do not name. The `(batch, tokens, channels)` conv input holds its
    /// sequences and its tokens on the ``GatedDeltaAxis`` axes. The recurrent state is a
    /// ``HeadRowsAxis`` tensor.
    private enum Axis {
        /// The axis that holds the channels of the conv input and the rows of
        /// the conv state.
        static let channel = 2
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
    /// compiled MLX kernels round. `window_at` reads one row of the conv input
    /// window: a row of the conv state, then the conv input of a token.
    /// `conv_silu` computes the conv taps of one channel for one token in
    /// float32, as `decodeConv` of the model does, then the SiLU in the input
    /// type.
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

            template <typename T>
            inline float window_at(
                const device T* state, const device T* tokens, int row, uint channel,
                int state_rows, int channels) {
              return row < state_rows
                  ? float(state[row * channels + channel])
                  : float(tokens[(row - state_rows) * channels + channel]);
            }

            template <typename T, typename W>
            inline float conv_silu(
                const device T* state, const device T* tokens, const device W* weight,
                uint channel, int token, int taps, int channels) {
              float sum = window_at(state, tokens, token, channel, taps - 1, channels)
                  * float(weight[channel * taps]);
              for (int tap = 1; tap < taps; ++tap) {
                float x = window_at(state, tokens, token + tap, channel, taps - 1, channels);
                sum = sum + x * float(weight[channel * taps + tap]);
              }
              T pre = static_cast<T>(sum);
              return float(pre * sigmoid_of(pre));
            }

            """

    /// The kernel body.
    ///
    /// Threadgroup `n` holds value head `n % Hv` of sequence `n / Hv`. SIMD
    /// group `g` holds the value rows `g * ROWS ..< (g + 1) * ROWS` in
    /// registers over the `TOKENS` tokens. `params` holds the scalars of
    /// ``Parameter``. `head_inputs` holds the q head, the k head and the value
    /// head of the current token, in this order.
    ///
    /// Two barriers order each token: the first one between the conv and the
    /// delta rule, the second one between the delta rule and the gated
    /// RMSNorm. The next token writes `head_inputs` only after the second
    /// barrier, when every read of them is done, and writes `head_output` only
    /// after its own first barrier, which the first SIMD group reaches when its
    /// gated RMSNorm is done.
    ///
    /// - Parameter snapshotWrite: The Metal statements that write the rows of
    ///   the SIMD group after token `t` into its snapshot slot, or nothing.
    /// - Returns: The kernel body.
    private static func source(snapshotWrite: String) -> String {
        """
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
        uint heads = threadgroups_per_grid.z;
        uint sequence = n / Hv;
        uint value_head = n % Hv;
        uint key_head = value_head / (Hv / Hk);

        const device InT* tokens = qkv + sequence * TOKENS * CHANNELS;
        const device InT* state = conv_state + sequence * STATE_ROWS * CHANNELS;

        // The conv state after the last token, rows TOKENS onward of the conv
        // input window: every thread of the grid writes at most a few numbers.
        uint state_numbers = STATE_ROWS * CHANNELS;
        uint grid_threads = threads_per_grid.x * threads_per_grid.y * threads_per_grid.z;
        uint grid_thread = (thread_position_in_grid.z * threads_per_grid.y + thread_position_in_grid.y)
            * threads_per_grid.x + thread_position_in_grid.x;
        for (uint i = grid_thread; i < heads / Hv * state_numbers; i += grid_threads) {
          uint from = i / state_numbers;
          uint row = (i % state_numbers) / CHANNELS + TOKENS;
          uint channel = i % CHANNELS;
          conv_out[i] = row < STATE_ROWS
              ? conv_state[(from * STATE_ROWS + row) * CHANNELS + channel]
              : qkv[(from * TOKENS + row - STATE_ROWS) * CHANNELS + channel];
        }

        // The rows of this SIMD group, which stay in registers over the tokens.
        float rows[ROWS][NUMBERS];
        for (int r = 0; r < ROWS; ++r) {
          size_t row_at = (size_t(n) * Dv + simd_group * ROWS + r) * Dk + lane * NUMBERS;
          for (int i = 0; i < NUMBERS; ++i) {
            rows[r][i] = static_cast<float>(state_in[row_at + i]);
          }
        }

        for (int t = 0; t < TOKENS; ++t) {
          // The conv and the SiLU of the q head, the k head and the value head.
          for (uint i = thread_in_group; i < V_AT + Dv; i += THREADS) {
            uint channel = i < Dk ? key_head * Dk + i
                : i < V_AT ? KEY_CHANNELS + key_head * Dk + (i - Dk)
                : QK_BLOCKS * KEY_CHANNELS + value_head * Dv + (i - V_AT);
            head_inputs[i] = conv_silu(state, tokens, conv_w, channel, t, TAPS, CHANNELS);
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
          size_t token_head = (size_t(sequence) * TOKENS + t) * Hv + value_head;
          float beta = float(sigmoid_of(static_cast<InT>(b_in[token_head])));
          InT shifted = static_cast<InT>(a_in[token_head]) + static_cast<InT>(dt_bias[value_head]);
          float g = metal::precise::exp(
              -metal::precise::exp(float(a_log[value_head])) * float(softplus_of(shifted)));

          // The delta rule on the rows of this SIMD group.
          for (int r = 0; r < ROWS; ++r) {
            uint value_row = simd_group * ROWS + r;
            float out = gated_delta_row(rows[r], k_row, q_row, g, beta, head_inputs[V_AT + value_row]);
            \(snapshotWrite)
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
              float gate = float(z[token_head * Dv + i]);
              y[token_head * Dv + i] = static_cast<InT>(gate * sigmoid_of(gate) * float(normed));
            }
          }
        }

        // The recurrent state after the last token.
        for (int r = 0; r < ROWS; ++r) {
          size_t row_at = (size_t(n) * Dv + simd_group * ROWS + r) * Dk + lane * NUMBERS;
          for (int i = 0; i < NUMBERS; ++i) {
            state_out[row_at + i] = static_cast<StT>(rows[r][i]);
          }
        }
        """
    }

    /// The Metal statements of a run that takes snapshots: they write the rows
    /// of the SIMD group after token `t` into slot `t` of `snapshots`, shape
    /// `(tokens, batch * valueHeads, valueHeadDim, keyHeadDim)`.
    private static let snapshotWrite = """
        size_t slot_at = ((size_t(t) * heads + n) * Dv + value_row) * Dk + lane * NUMBERS;
                    for (int i = 0; i < NUMBERS; ++i) {
                      snapshots[slot_at + i] = static_cast<StT>(rows[r][i]);
                    }
        """

    /// The input names of both kernels.
    private static let inputNames = [
        "qkv", "z", "b_in", "a_in", "a_log", "dt_bias", "conv_w", "conv_state", "state_in",
        "norm_w", "params",
    ]

    /// The output names of a run without snapshots, in the order of ``Output``.
    private static let outputNames = ["y", "state_out", "conv_out"]

    /// The kernel of a run without snapshots: a decode step, or the accepted
    /// tokens of a verify.
    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_decode",
        inputNames: inputNames,
        outputNames: outputNames,
        source: source(snapshotWrite: ""),
        header: header)

    /// The kernel of a speculative verify, which also writes the recurrent
    /// state after each token into its snapshot slot.
    private static let snapshotKernel = MLXFast.metalKernel(
        name: "gated_delta_decode_snapshots",
        inputNames: inputNames,
        outputNames: outputNames + ["snapshots"],
        source: source(snapshotWrite: snapshotWrite),
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
        guard recurrentState.ndim == HeadRowsAxis.count, inputs.qkv.ndim == Axis.channel + 1
        else { return nil }
        let valueHeads = recurrentState.dim(HeadRowsAxis.head)
        let valueHeadDim = recurrentState.dim(HeadRowsAxis.row)
        let keyHeadDim = recurrentState.dim(HeadRowsAxis.width)
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
        let batch = inputs.qkv.dim(GatedDeltaAxis.batch)
        let channels = inputs.qkv.dim(Axis.channel)
        let tokens = inputs.tokenCount
        let gates = [batch, tokens, shape.valueHeads]
        return (1 ... mostTokens).contains(tokens)
            && inputs.z.shape == [batch, tokens, shape.valueHeads, shape.valueHeadDim]
            && inputs.b.shape == gates && inputs.a.shape == gates
            && convState.shape == [batch, shape.taps - 1, channels]
            && recurrentState.dim(HeadRowsAxis.batch) == batch
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

    /// Tells if the kernel runs the tokens: the default device is the GPU, the
    /// run holds one to ``mostTokens`` tokens, and the shapes and the dtypes
    /// of the operands are the ones the kernel reads.
    ///
    /// - Parameters:
    ///   - inputs: The token inputs.
    ///   - convState: The conv state before the first token.
    ///   - recurrentState: The recurrent state before the first token.
    ///   - weights: The weights of the layer.
    /// - Returns: True when
    ///   ``step(inputs:convState:recurrentState:weights:takesSnapshots:)``
    ///   runs the tokens.
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

    /// One run of one gated delta net layer over one to ``mostTokens``
    /// tokens: a decode step, or the tokens of a speculative verify.
    ///
    /// - Parameters:
    ///   - inputs: The token inputs.
    ///   - convState: The conv state before the first token.
    ///   - recurrentState: The recurrent state before the first token, in
    ///     float32.
    ///   - weights: The weights of the layer.
    ///   - takesSnapshots: True to keep the states after each token, as a
    ///     verify needs them for a rollback.
    /// - Returns: The output of the heads, the states after the last token,
    ///   and the states after each token when the run takes snapshots.
    /// - Precondition: ``canRun(inputs:convState:recurrentState:weights:)``.
    package static func step(
        inputs: Inputs, convState: MLXArray, recurrentState: MLXArray, weights: Weights,
        takesSnapshots: Bool = false
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
        let batch = inputs.qkv.dim(GatedDeltaAxis.batch)
        let tokens = inputs.tokenCount
        var outputShapes = [
            [batch, tokens, shape.valueHeads * shape.valueHeadDim], recurrentState.shape,
            convState.shape,
        ]
        var outputDTypes = [inputs.qkv.dtype, recurrentState.dtype, convState.dtype]
        if takesSnapshots {
            outputShapes.append([tokens] + recurrentState.shape)
            outputDTypes.append(recurrentState.dtype)
        }
        let outputs = (takesSnapshots ? snapshotKernel : kernel)(
            [
                inputs.qkv, inputs.z, inputs.b, inputs.a, weights.aLog, weights.dtBias,
                weights.conv, convState, recurrentState, weights.norm, weights.parameters,
            ],
            template: [
                ("InT", inputs.qkv.dtype), ("StT", recurrentState.dtype),
                ("Dk", shape.keyHeadDim), ("Dv", shape.valueHeadDim), ("Hk", shape.keyHeads),
                ("Hv", shape.valueHeads), ("TAPS", shape.taps), ("SIMD_GROUPS", shape.simdGroups),
                ("TOKENS", tokens),
            ],
            grid: (metalSimdWidth, shape.simdGroups, batch * shape.valueHeads),
            threadGroup: (metalSimdWidth, shape.simdGroups, 1),
            outputShapes: outputShapes,
            outputDTypes: outputDTypes)
        let snapshots =
            takesSnapshots
            ? Snapshots(
                recurrent: outputs[Output.snapshots],
                convWindow: concatenated([convState, inputs.qkv], axis: GatedDeltaAxis.step))
            : nil
        return Step(
            output: outputs[Output.output], convState: outputs[Output.convState],
            recurrentState: outputs[Output.recurrentState], snapshots: snapshots)
    }
}

/// The snapshot slots as a compiled trace gives them back. A speculative
/// verify pass of a gated delta net layer keeps them so that a rollback can
/// restore the layer, from the kernel or from the op path, and
/// ``MambaCache`` keeps them as transient speculative data.
extension GatedDeltaDecodeKernel.Snapshots {

    /// The places of the slots in ``traceArrays``.
    private enum TraceArray {
        /// The recurrent snapshot slots.
        static let recurrent = 0
        /// The conv input window.
        static let conv = 1
        /// The number of arrays.
        static let count = 2
    }

    /// The number of arrays of ``traceArrays``.
    package static let traceArrayCount = TraceArray.count

    /// The arrays of the slots, as a compiled trace gives them back: the
    /// recurrent slots, then the conv input window.
    package var traceArrays: [MLXArray] {
        [recurrent, convWindow]
    }

    /// The slots from the arrays of ``traceArrays``.
    ///
    /// - Parameter traceArrays: The arrays of ``traceArrays``.
    package init(traceArrays: [MLXArray]) {
        precondition(
            traceArrays.count == TraceArray.count,
            "snapshot slots take \(TraceArray.count) arrays, not \(traceArrays.count)")
        self.init(
            recurrent: traceArrays[TraceArray.recurrent], convWindow: traceArrays[TraceArray.conv])
    }

    /// The gated delta rule of the op path over the tokens of a speculative
    /// verify, one token at a time, with the snapshot slots the kernel would
    /// keep: each token runs ``gatedDeltaUpdate(q:k:v:a:b:aLog:dtBias:state:mask:useKernel:)``
    /// from the state after the token before it, and its state is its slot.
    /// A verify of two tokens thus runs the two one-token updates that a
    /// split after its first token runs.
    ///
    /// - Parameters:
    ///   - q: The normed queries, shape `(batch, tokens, keyHeads, keyHeadDim)`.
    ///   - k: The normed keys, shape `(batch, tokens, keyHeads, keyHeadDim)`.
    ///   - v: The values, shape `(batch, tokens, valueHeads, valueHeadDim)`.
    ///   - a: The decay input, shape `(batch, tokens, valueHeads)`.
    ///   - b: The beta input, shape `(batch, tokens, valueHeads)`.
    ///   - aLog: The log of the decay rate of each value head.
    ///   - dtBias: The bias of the decay step of each value head.
    ///   - state: The recurrent state before the first token, or nil.
    ///   - mask: The mask of the tokens, shape `(batch, tokens)`, or nil.
    ///   - useKernel: False when the layer trains.
    ///   - convWindow: The conv input window of the tokens: the conv state
    ///     before the first token, then the conv input of each token.
    /// - Returns: The output of the tokens, the recurrent state after the
    ///   last token, and the slots.
    package static func gatedDeltaUpdate(
        q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray, aLog: MLXArray,
        dtBias: MLXArray, state: MLXArray?, mask: MLXArray?, useKernel: Bool,
        convWindow: MLXArray
    ) -> (output: MLXArray, state: MLXArray, snapshots: Self) {
        var current = state
        let steps = (0 ..< q.dim(GatedDeltaAxis.step)).map { token in
            let tokens = token ..< token + GatedDeltaDecodeKernel.decodeTokens
            let step = MLXLMCommon.gatedDeltaUpdate(
                q: q[0..., tokens], k: k[0..., tokens], v: v[0..., tokens], a: a[0..., tokens],
                b: b[0..., tokens], aLog: aLog, dtBias: dtBias, state: current,
                mask: mask.map { $0[0..., tokens] }, useKernel: useKernel)
            current = step.1
            return step
        }
        guard let last = steps.last else {
            preconditionFailure("a speculative verify holds at least one token")
        }
        return (
            concatenated(steps.map(\.0), axis: GatedDeltaAxis.step), last.1,
            Self(recurrent: stacked(steps.map(\.1)), convWindow: convWindow)
        )
    }
}

/// The gated RMSNorm of the output of a gated delta net layer, as
/// ``GatedDeltaDecodeKernel`` reads it.
package protocol GatedDeltaOutputNorm: Module {
    /// The weight of the norm, shape `(valueHeadDim)`.
    var weight: MLXArray { get }
    /// The epsilon of the norm.
    var eps: Float { get }
}

/// A gated delta net layer whose body on one decode token can run in
/// ``GatedDeltaDecodeKernel``: the Qwen3-Next layer and the Qwen 3.5 layers of
/// MLXLLM and MLXVLM. Each layer gives the modules the kernel reads. The
/// shared ``decodeKernelWeights`` collects their weights, and the shared
/// ``decodeKernelStep(qkv:z:b:a:convState:recState:)`` decides if the kernel
/// runs the token. The q and k RMSNorm of each layer is
/// ``GatedDeltaQueryKeyNorm``.
package protocol GatedDeltaDecodeLayer: Module {
    /// The type of the gated RMSNorm of the output.
    associatedtype OutputNorm: GatedDeltaOutputNorm

    /// The width of one key head.
    var headKDim: Int { get }
    /// The depthwise conv.
    var conv1d: Conv1d { get }
    /// The log of the decay rate of each value head, shape `(valueHeads)`.
    var aLog: MLXArray { get }
    /// The bias of the decay step of each value head, shape `(valueHeads)`.
    var dtBias: MLXArray { get }
    /// The gated RMSNorm of the output.
    var norm: OutputNorm { get }
}

extension GatedDeltaDecodeLayer {
    /// The weights of the layer that ``GatedDeltaDecodeKernel`` reads.
    package var decodeKernelWeights: GatedDeltaDecodeKernel.Weights {
        let scales = GatedDeltaQueryKeyNorm.scales(keyHeadDim: headKDim)
        return GatedDeltaDecodeKernel.Weights(
            conv: conv1d.weight, aLog: aLog, dtBias: dtBias, norm: norm.weight,
            normEps: norm.eps, queryKeyNormEps: GatedDeltaQueryKeyNorm.eps(keyHeadDim: headKDim),
            queryScale: scales.query, keyScale: scales.key)
    }

    /// The body of one decode token in one Metal kernel
    /// (``GatedDeltaDecodeKernel``), or nil when the kernel does not run the
    /// token: the layer trains (the kernel has no gradient), the recurrent
    /// state is not there yet, or the shapes and dtypes are not the ones the
    /// kernel reads. The output is the input of the output projection.
    ///
    /// - Parameters:
    ///   - qkv: The conv input of the token, shape `(batch, 1, convChannels)`.
    ///   - z: The gate of the output, shape `(batch, 1, valueHeads, valueHeadDim)`.
    ///   - b: The beta input, shape `(batch, 1, valueHeads)`.
    ///   - a: The decay input, shape `(batch, 1, valueHeads)`.
    ///   - convState: The conv state before the token.
    ///   - recState: The recurrent state before the token, or nil.
    /// - Returns: The output of the heads and the states after the token, or
    ///   nil.
    package func decodeKernelStep(
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray, convState: MLXArray,
        recState: MLXArray?
    ) -> GatedDeltaDecodeKernel.Step? {
        guard qkv.dim(GatedDeltaAxis.step) == GatedDeltaDecodeKernel.decodeTokens else {
            return nil
        }
        return kernelRun(
            inputs: GatedDeltaDecodeKernel.Inputs(qkv: qkv, z: z, b: b, a: a),
            convState: convState, recState: recState, takesSnapshots: false)
    }

    /// The body of the tokens of a speculative verify in one Metal kernel
    /// (``GatedDeltaDecodeKernel``), with the states after each token, or nil
    /// when the kernel does not run the tokens: the verify holds one token or
    /// more than ``GatedDeltaDecodeKernel/mostTokens`` tokens, the layer
    /// trains, the recurrent state is not there yet, or the shapes and dtypes
    /// are not the ones the kernel reads. The output is the input of the
    /// output projection.
    ///
    /// - Parameters:
    ///   - qkv: The conv input of the tokens, shape
    ///     `(batch, tokens, convChannels)`.
    ///   - z: The gate of the output, shape
    ///     `(batch, tokens, valueHeads, valueHeadDim)`.
    ///   - b: The beta input, shape `(batch, tokens, valueHeads)`.
    ///   - a: The decay input, shape `(batch, tokens, valueHeads)`.
    ///   - convState: The conv state before the first token.
    ///   - recState: The recurrent state before the first token, or nil.
    /// - Returns: The output of the heads, the states after the last token
    ///   and the states after each token, or nil.
    package func speculativeKernelStep(
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray, convState: MLXArray,
        recState: MLXArray?
    ) -> GatedDeltaDecodeKernel.Step? {
        guard qkv.dim(GatedDeltaAxis.step) > GatedDeltaDecodeKernel.decodeTokens else {
            return nil
        }
        return kernelRun(
            inputs: GatedDeltaDecodeKernel.Inputs(qkv: qkv, z: z, b: b, a: a),
            convState: convState, recState: recState, takesSnapshots: true)
    }

    /// The kernel run of the tokens, or nil when the layer trains, the
    /// recurrent state is not there yet, or the kernel does not run them.
    private func kernelRun(
        inputs: GatedDeltaDecodeKernel.Inputs, convState: MLXArray, recState: MLXArray?,
        takesSnapshots: Bool
    ) -> GatedDeltaDecodeKernel.Step? {
        guard !training, let recState else { return nil }
        let weights = decodeKernelWeights
        guard
            GatedDeltaDecodeKernel.canRun(
                inputs: inputs, convState: convState, recurrentState: recState, weights: weights)
        else { return nil }
        return GatedDeltaDecodeKernel.step(
            inputs: inputs, convState: convState, recurrentState: recState, weights: weights,
            takesSnapshots: takesSnapshots)
    }
}
