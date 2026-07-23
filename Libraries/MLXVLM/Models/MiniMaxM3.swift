//
//  MiniMaxM3.swift
//  mlx-swift-lm
//
//  MoE block + swigluoai activation building blocks for MiniMax-M3
//  (model_type `minimax_m3_vl`, arch `MiniMaxM3SparseForConditionalGeneration`).
//  No decoder/model yet -- see kanban ^xgvth41 for the full
//  `MiniMaxM3TextConfiguration` + dense-attention decoder.
//
//  Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py`
//  (`MiniMaxSwiGLUOAI`, `MiniMaxPackedSwitchGLU`, `MiniMaxSparseMoeBlock`,
//  `_minimax_moe_select`) and upstream mlx-lm PRs #1398 (`minimax_m3_vl.py`)
//  / #1401 (`minimax_m3`). Builds on the M2 pattern in
//  `Libraries/MLXLLM/Models/MiniMax.swift` (sigmoid scoring +
//  `e_score_correction_bias` routing), and reuses GPT-OSS's `swigluoai`
//  clamp shape (`Libraries/MLXLLM/Models/GPTOSS.swift`).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MiniMaxM3SwiGLUOAI

/// Clipped SwiGLU activation used by MiniMax-M3's dense MLPs, shared expert,
/// and routed experts (`hidden_act: "swigluoai"`).
///
/// Verified directly against mlx-vlm's `_swiglu_oai` / `MiniMaxSwiGLUOAI`
/// (`mlx_vlm/models/minimax_m3_vl/language.py`), which matches GPT-OSS's
/// `swiglu` (`GPTOSS.swift`) exactly: the gate branch is clipped only at its
/// *upper* bound (very negative gates are left alone -- the sigmoid
/// saturates them towards zero on its own), while the linear branch is
/// clipped symmetrically:
///
/// ```
/// x_glu = clip(x_glu, max: limit)
/// x_linear = clip(x_linear, min: -limit, max: limit)
/// return x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)
/// ```
///
/// `alpha`/`limit`/`beta` are config-driven (`swiglu_alpha`, `swiglu_limit`,
/// `swiglu_beta` in the checkpoint's `text_config`) rather than hardcoded --
/// for MiniMax-M3 they default to 1.702 / 7.0 / 1.0, same as GPT-OSS.
///
/// Note: an earlier draft of the M3 port (upstream mlx-lm PR #1398's
/// `minimax_m3_vl.py`) clipped the gate branch symmetrically, like the
/// linear branch. That draft was superseded by the asymmetric-clip version
/// implemented here, which is what mlx-vlm's `minimax_m3_vl` package (and the
/// published `mlx-community/MiniMax-M3-4bit` checkpoint) actually uses.
struct MiniMaxM3SwiGLUOAI {
    let alpha: Float
    let limit: Float
    let beta: Float

    init(alpha: Float = 1.702, limit: Float = 7.0, beta: Float = 1.0) {
        self.alpha = alpha
        self.limit = limit
        self.beta = beta
    }

    /// - Parameters:
    ///   - xLinear: the "up" projection branch (clipped symmetrically, `+ beta`).
    ///   - xGlu: the "gate" projection branch (clipped at its upper bound only).
    func callAsFunction(_ xLinear: MLXArray, _ xGlu: MLXArray) -> MLXArray {
        let glu = clip(xGlu, max: MLXArray(limit))
        let linear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))
        return glu * sigmoid(alpha * glu) * (linear + beta)
    }
}

// MARK: - MiniMaxM3MoEConfiguration

/// Minimal configuration for MiniMax-M3's MoE building blocks -- just the
/// fields `MiniMaxM3SparseMoeBlock` needs. Field names mirror the
/// checkpoint's `text_config` keys so they can be lifted directly into the
/// full `MiniMaxM3TextConfiguration` once the decoder lands (kanban
/// ^xgvth41); this task builds standalone compute blocks only, with no
/// decoder/model and no `Codable` config yet.
struct MiniMaxM3MoEConfiguration: Sendable {
    var hiddenSize: Int
    var intermediateSize: Int
    var numLocalExperts: Int
    var numExpertsPerTok: Int
    var routedScalingFactor: Float
    var swigluAlpha: Float
    var swigluLimit: Float
    var swigluBeta: Float

    init(
        hiddenSize: Int,
        intermediateSize: Int,
        numLocalExperts: Int,
        numExpertsPerTok: Int,
        routedScalingFactor: Float = 2.0,
        swigluAlpha: Float = 1.702,
        swigluLimit: Float = 7.0,
        swigluBeta: Float = 1.0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.routedScalingFactor = routedScalingFactor
        self.swigluAlpha = swigluAlpha
        self.swigluLimit = swigluLimit
        self.swigluBeta = swigluBeta
    }
}

// MARK: - MiniMaxM3SparseMoeBlock

/// MiniMax-M3's sparse MoE block: sigmoid-scoring, biased top-k selection
/// (M2's `MiniMaxSparseMoeBlock` pattern, `Libraries/MLXLLM/Models/
/// MiniMax.swift`), extended with M3's single shared expert and
/// `routed_scaling_factor`.
///
/// **The shared expert is packed, not a separate module.** Verified against
/// the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `model.safetensors.index.json` and safetensors shard headers: there is no
/// `shared_expert(s)`-prefixed weight anywhere in the checkpoint, and
/// `block_sparse_moe.switch_mlp.gate_up_proj` has a **129**-row leading
/// (expert) dimension for this checkpoint's 128 local experts -- confirmed
/// directly from the shard header (`gate_up_proj.weight` shape
/// `[129, 6144, 768]`, quantized). This is exactly mlx-vlm's
/// `pack_shared_expert` scheme (`MiniMaxPackedSwitchGLU` in
/// `minimax_m3_vl/language.py`), which upstream selects automatically
/// whenever `n_shared_experts == 1 && shared_intermediate_size ==
/// intermediate_size` (true for M3): the shared expert is expert row
/// `numLocalExperts`, *always* selected with weight `1.0` (i.e. unscaled by
/// `routed_scaling_factor`), and summed into the routed output via the same
/// weighted-sum reduction used for the routed experts.
///
/// Fused expert weights (`gate_up_proj`, pre-stacked and pre-quantized) are
/// consumed through `FusedGateUpSwitchGLU`'s two-arg activation seam, since
/// `MiniMaxM3SwiGLUOAI` needs the gate and up halves separately rather than
/// as a single `activation(gate) * up` product. The split order is
/// concatenated halves (`gate = first half, up = second half` of the fused
/// `2 * intermediateSize` column dimension) -- verified against mlx-vlm's
/// `MiniMaxPackedSwitchGLU.__call__`: `gate, up = mx.split(gate_up, 2,
/// axis=-1)`, then `self.activation(up, gate)`.
class MiniMaxM3SparseMoeBlock: Module {
    let numExpertsPerTok: Int
    let numLocalExperts: Int
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: FusedGateUpSwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: MiniMaxM3MoEConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok
        self.numLocalExperts = config.numLocalExperts
        self.routedScalingFactor = config.routedScalingFactor

        _gate.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: false)

        let swiglu = MiniMaxM3SwiGLUOAI(
            alpha: config.swigluAlpha, limit: config.swigluLimit, beta: config.swigluBeta)
        _switchMLP.wrappedValue = FusedGateUpSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numLocalExperts + 1,
            twoArgActivation: { gate, up in swiglu(up, gate) }
        )
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.numLocalExperts])
    }

    /// Top-k routing over the `numLocalExperts` *routed* experts only (the
    /// packed shared expert is appended separately in `callAsFunction`).
    ///
    /// Selection uses the *biased* scores (`sigmoid(gate(x)) +
    /// e_score_correction_bias`, computed in float32); routing weights are
    /// gathered from the *original unbiased* sigmoid scores at the selected
    /// indices, renormalized to sum to 1 (+ 1e-20 epsilon), then scaled by
    /// `routed_scaling_factor` -- matching mlx-vlm's `_minimax_moe_select`
    /// exactly.
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let gates = gate(x.asType(.float32))
        let scores = sigmoid(gates)
        let biasedScores = scores + eScoreCorrectionBias

        let k = numExpertsPerTok
        let indices = argPartition(-biasedScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        weights = weights * routedScalingFactor
        return (indices, weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (routedIndices, routedWeights) = route(x)
        let weights = routedWeights.asType(x.dtype)

        let sharedIndices = MLXArray.full(
            Array(routedIndices.shape.dropLast()) + [1],
            values: MLXArray(Int32(numLocalExperts)), dtype: routedIndices.dtype)
        let sharedWeights = MLXArray.ones(
            Array(weights.shape.dropLast()) + [1], dtype: weights.dtype)

        let indices = MLX.concatenated([routedIndices, sharedIndices], axis: -1)
        let allWeights = MLX.concatenated([weights, sharedWeights], axis: -1)

        let y = switchMLP(x, indices)
        return weightedExpertSum(y, allWeights)
    }
}
