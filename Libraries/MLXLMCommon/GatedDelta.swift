//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

/// Fused form of the decay gate chain — elementwise, and MLX `compile`
/// preserves per-node dtype rounding (verified bitwise against the unfused
/// chain on the real decode/prefill shapes, bf16 and f16), so this is
/// bit-identical while cutting ~6 kernel launches per GDN layer per step.
private let computeGatedDeltaG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { aLog, a, dtBias in
    exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
}

// MARK: - Axes

/// The axes of the `(batch, steps, heads, width)` q, k, v and y tensors. The
/// token inputs of ``GatedDeltaDecodeKernel`` hold their sequences and their
/// tokens on the same axes. The recurrent state is a ``HeadRowsAxis`` tensor.
enum GatedDeltaAxis {
    /// The axis that holds the sequences.
    static let batch = 0
    /// The axis that holds the steps.
    static let step = 1
    /// The axis that holds the heads.
    static let head = 2
    /// The axis that holds the numbers of one head.
    static let width = 3
}

// MARK: - Query and key norm

/// The q and k RMSNorm of the Qwen3-Next and Qwen 3.5 gated delta net layers.
/// The op path of each layer and the weights of ``GatedDeltaDecodeKernel``
/// read these values, thus the two paths normalize q and k the same way.
package enum GatedDeltaQueryKeyNorm {
    /// The epsilon of the q and k RMSNorm.
    package static let eps: Float = 1e-6

    /// The power of the key head width that gives the k factor:
    /// `1 / sqrt(keyHeadDim)`.
    private static let keyScalePower: Float = -0.5

    /// The factors of the normalized q and k.
    ///
    /// - Parameter keyHeadDim: The width of one key head.
    /// - Returns: The q factor `1 / keyHeadDim` and the k factor
    ///   `1 / sqrt(keyHeadDim)`.
    package static func scales(keyHeadDim: Int) -> (query: Float, key: Float) {
        let keyScale = pow(Float(keyHeadDim), keyScalePower)
        return (keyScale * keyScale, keyScale)
    }

    /// The q and k of the op path of a layer: each row normalized, then
    /// multiplied by its factor of ``scales(keyHeadDim:)`` in `dtype`.
    ///
    /// - Parameters:
    ///   - query: The q heads, shape `(batch, steps, keyHeads, keyHeadDim)`.
    ///   - key: The k heads, with the shape of `query`.
    ///   - dtype: The dtype of the factors.
    /// - Returns: The normalized q and k.
    package static func normalized(query: MLXArray, key: MLXArray, dtype: DType) -> (
        query: MLXArray, key: MLXArray
    ) {
        let scales = scales(keyHeadDim: query.dim(GatedDeltaAxis.width))
        return (
            MLXArray(scales.query).asType(dtype)
                * MLXFast.rmsNorm(query, weight: MLXArray.mlxNone, eps: eps),
            MLXArray(scales.key).asType(dtype)
                * MLXFast.rmsNorm(key, weight: MLXArray.mlxNone, eps: eps)
        )
    }
}

/// The ranks of the gate and of the mask of one step of the ops path.
private enum GatedDeltaStepRank {
    /// One number for each sequence: `(batch)`.
    static let perSequence = 1
    /// One number for each head: `(batch, heads)`.
    static let perHead = 2
    /// One number for each number of a head: `(batch, heads, width)`.
    static let perHeadNumber = 3
}

// MARK: - Metal Kernel

/// The number of SIMD groups of one threadgroup of `gated_delta_step`. Each
/// SIMD group holds one value row of the recurrent state.
private let gatedDeltaStepSimdGroups = 4

/// The Metal helpers that the gated delta kernels share.
///
/// `gated_delta_row` runs one token of the gated delta rule on one row of the
/// recurrent state: the decay, the delta update, and the output of the row.
/// One lane holds `N` numbers of the row and the same `N` numbers of the key
/// and the query. The lanes of the SIMD group hold the whole row. A row has
/// `SIMD_WIDTH` lanes, which is ``metalSimdWidth``, thus each lane holds
/// `Dk / SIMD_WIDTH` numbers.
///
/// The first sum keeps Kahan compensation under the default fast math of
/// Metal, for parity with mlx-lm.
///
/// MLX writes the kernel signature directly after the header, thus the
/// header ends in a line break.
let gatedDeltaMetalHeader =
    metalSimdWidthDeclaration + """

        template <int N>
        inline float gated_delta_row(
            thread float (&state)[N], thread const float (&k)[N], thread const float (&q)[N],
            float g, float beta, float v) {
          float kv_mem = 0.0f;
          {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            float kv_compensation = 0.0f;
            for (int i = 0; i < N; ++i) {
              state[i] = state[i] * g;
              auto product = state[i] * k[i];
              auto corrected = product - kv_compensation;
              auto next_sum = kv_mem + corrected;
              kv_compensation = (next_sum - kv_mem) - corrected;
              kv_mem = next_sum;
            }
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v - kv_mem) * beta;

          float out = 0.0f;
          for (int i = 0; i < N; ++i) {
            state[i] = state[i] + k[i] * delta;
            out += state[i] * q[i];
          }
          return simd_sum(out);
        }

        """

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / SIMD_WIDTH;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float k_row[n_per_t];
                float q_row[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  k_row[i] = static_cast<float>(k_[s_idx]);
                  q_row[i] = static_cast<float>(q_[s_idx]);
                }
                float out = gated_delta_row(
                    state, k_row, q_row, g_[hv_idx], beta_[hv_idx], static_cast<float>(v_[dv_idx]));
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source,
        header: gatedDeltaMetalHeader
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(GatedDeltaAxis.batch)
    let T = k.dim(GatedDeltaAxis.step)
    let Hk = k.dim(GatedDeltaAxis.head)
    let Dk = k.dim(GatedDeltaAxis.width)
    let Hv = v.dim(GatedDeltaAxis.head)
    let Dv = v.dim(GatedDeltaAxis.width)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (metalSimdWidth, Dv, B * Hv),
        threadGroup: (metalSimdWidth, gatedDeltaStepSimdGroups, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == GatedDeltaStepRank.perHead {
        decay = expandedDimensions(g, axes: [HeadRowsAxis.row, HeadRowsAxis.width])
    } else if g.ndim == GatedDeltaStepRank.perHeadNumber {
        decay = expandedDimensions(g, axis: HeadRowsAxis.row)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: HeadRowsAxis.row)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state =
        state + expandedDimensions(k, axis: HeadRowsAxis.row)
        * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: HeadRowsAxis.row)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == GatedDeltaStepRank.perSequence {
            expandedMask = expandedDimensions(
                mask, axes: [HeadRowsAxis.head, HeadRowsAxis.row, HeadRowsAxis.width])
        } else if mask.ndim == GatedDeltaStepRank.perHead {
            expandedMask = expandedDimensions(
                mask, axes: [HeadRowsAxis.row, HeadRowsAxis.width])
        } else if mask.ndim == GatedDeltaStepRank.perHeadNumber {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

/// Steps per recompute chunk in `gatedDeltaOps`.
///
/// The ops path is the one that can be trained through (the fused kernel
/// has no gradient), and a recurrence run as plain ops keeps every step's
/// state for the backward pass: at Qwen3.5-27B's shapes (48 heads of
/// 128x128 fp32) that is about 4 MB a step, gigabytes per layer at a
/// thousand tokens. Each chunk is run through a custom function whose
/// backward recomputes the chunk instead, so only the chunk boundaries are
/// kept - what `mx.checkpoint` does for the Python trainer. Inference is
/// unaffected: the forward is the same ops in the same order.
let gatedDeltaRecomputeChunk = 16

private enum GatedDeltaRecompute {
    /// The places of the operands in the `inputs` of ``steps(_:masked:)``.
    private enum Input {
        /// The queries.
        static let q = 0
        /// The keys.
        static let k = 1
        /// The values.
        static let v = 2
        /// The decay gates.
        static let g = 3
        /// The beta gates.
        static let beta = 4
        /// The recurrent state.
        static let state = 5
        /// The mask, when there is one.
        static let mask = 6
    }

    /// `inputs` = [q, k, v, g, beta, state] plus the mask when there is one,
    /// each sliced to the chunk; returns [y, state].
    static func steps(_ inputs: [MLXArray], masked: Bool) -> [MLXArray] {
        let (q, k, v, g, beta) = (
            inputs[Input.q], inputs[Input.k], inputs[Input.v], inputs[Input.g], inputs[Input.beta]
        )
        var state = inputs[Input.state]
        let mask = masked ? inputs[Input.mask] : nil
        var ys = [MLXArray]()
        ys.reserveCapacity(q.dim(GatedDeltaAxis.step))
        for t in 0 ..< q.dim(GatedDeltaAxis.step) {
            let (y, newState) = gatedDeltaStepOps(
                q: q[0..., t],
                k: k[0..., t],
                v: v[0..., t],
                g: g[0..., t],
                beta: beta[0..., t],
                state: state,
                mask: mask.map { $0[0..., t] }
            )
            ys.append(y)
            state = newState
        }
        return [MLX.stacked(ys, axis: GatedDeltaAxis.step), state]
    }

    // The closure holds a locked state object; two callers serialise on it.
    nonisolated(unsafe) static let unmasked: ([MLXArray]) -> [MLXArray] = CustomFunction {
        Forward { steps($0, masked: false) }
        VJP { primals, cotangents in
            vjp({ steps($0, masked: false) }, primals: primals, cotangents: cotangents).1
        }
    }

    nonisolated(unsafe) static let masked: ([MLXArray]) -> [MLXArray] = CustomFunction {
        Forward { steps($0, masked: true) }
        VJP { primals, cotangents in
            vjp({ steps($0, masked: true) }, primals: primals, cotangents: cotangents).1
        }
    }
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(GatedDeltaAxis.batch)
    let T = q.dim(GatedDeltaAxis.step)
    let Hk = q.dim(GatedDeltaAxis.head)
    let Dk = q.dim(GatedDeltaAxis.width)
    let Hv = v.dim(GatedDeltaAxis.head)
    let Dv = v.dim(GatedDeltaAxis.width)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: GatedDeltaAxis.head)
        k = repeated(k, count: repeatFactor, axis: GatedDeltaAxis.head)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity((T + gatedDeltaRecomputeChunk - 1) / gatedDeltaRecomputeChunk)

    for start in stride(from: 0, to: T, by: gatedDeltaRecomputeChunk) {
        let steps = start ..< min(start + gatedDeltaRecomputeChunk, T)
        var inputs = [
            q[0..., steps], k[0..., steps], v[0..., steps], g[0..., steps], beta[0..., steps],
            state,
        ]
        let run: ([MLXArray]) -> [MLXArray]
        if let mask {
            inputs.append(mask[0..., steps])
            run = GatedDeltaRecompute.masked
        } else {
            run = GatedDeltaRecompute.unmasked
        }
        let out = run(inputs)
        ys.append(out[0])
        state = out[1]
    }

    let y = ys.count == 1 ? ys[0] : MLX.concatenated(ys, axis: GatedDeltaAxis.step)
    return (y, state)
}

// MARK: - Public API

public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    useKernel: Bool = true
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b).asType(.float32)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(GatedDeltaAxis.batch)
    let Dk = q.dim(GatedDeltaAxis.width)
    let Hv = v.dim(GatedDeltaAxis.head)
    let Dv = v.dim(GatedDeltaAxis.width)

    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }

    // The fused kernel distributes Dk over exactly `metalSimdWidth` simd lanes
    // (`n_per_t = Dk / SIMD_WIDTH`, integer-truncating). A key head dim that is
    // not a multiple of `metalSimdWidth` would silently drop its trailing
    // `Dk % metalSimdWidth` state dimensions and produce wrong output with no
    // error. Stock Qwen3.5 uses Dk = 192 (a multiple of `metalSimdWidth`), but
    // the value is config-driven, so route any other Dk to the ops fallback,
    // which handles an arbitrary key dimension correctly (slower, but never
    // truncating).
    //
    // `useKernel: false` is for training: the kernel is a custom Metal
    // kernel and has no gradient, so a model in training mode passes
    // `!training` here, as the Python model does with `use_kernel`.
    if useKernel, GatedDeltaKernelManager.shared.kernel != nil, Dk % metalSimdWidth == 0 {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}
