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

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Shared defaults

/// Default SwiGLU-OAI gate-branch sigmoid steepness (`swiglu_alpha`) when the
/// checkpoint config omits it. Shared by `MiniMaxM3SwiGLUOAI`,
/// `MiniMaxM3MoEConfiguration`, and `MiniMaxM3TextConfiguration` so the value
/// only needs to change in one place.
public let defaultSwigluAlpha: Float = 1.702

/// Default SwiGLU-OAI clip limit (`swiglu_limit`) applied to both branches
/// when the checkpoint config omits it. See `defaultSwigluAlpha`.
public let defaultSwigluLimit: Float = 7.0

/// Default SwiGLU-OAI linear-branch additive bias (`swiglu_beta`) when the
/// checkpoint config omits it. See `defaultSwigluAlpha`.
public let defaultSwigluBeta: Float = 1.0

/// Default MoE routing-weight scaling factor (`routed_scaling_factor`) when
/// the checkpoint config omits it. Shared by `MiniMaxM3MoEConfiguration` and
/// `MiniMaxM3TextConfiguration`.
public let defaultRoutedScalingFactor: Float = 2.0

/// First MoE-scheduled layer index in the verified default schedule (layers
/// before this index are dense; this index and beyond are MoE). Shared by
/// both of `MiniMaxM3TextConfiguration`'s default `moeLayerFreq` derivations
/// (`init` and `init(from:)`).
let defaultMoeLayerStart = 3

/// Default per-layer dense/MoE schedule (`moe_layer_freq`) when the
/// checkpoint config omits it: layers before `defaultMoeLayerStart` are dense
/// (`0`), layers at or after it are MoE (`1`). Shared by
/// `MiniMaxM3TextConfiguration`'s `init` and `init(from:)` to avoid
/// duplicating the derivation.
private func defaultMoeLayerFreq(hiddenLayers: Int) -> [Int] {
    (0 ..< hiddenLayers).map { $0 < defaultMoeLayerStart ? 0 : 1 }
}

/// Default DSA indexer per-index-head dimension (`sparse_index_dim`),
/// verified against the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `config.json`. Shared by `MiniMaxM3SparseAttentionConfiguration`'s
/// memberwise `init` and its `init(from:)` decoder fallback.
public let defaultIndexDim = 128

/// Default DSA indexer head count (`sparse_num_index_heads`). See `defaultIndexDim`.
public let defaultNumIndexHeads = 4

/// Default DSA indexer top-k selected block count (`sparse_topk_blocks`). See `defaultIndexDim`.
public let defaultTopkBlocks = 16

/// Default DSA indexer block size, in tokens (`sparse_block_size`). See `defaultIndexDim`.
public let defaultBlockSize = 128

/// Default DSA indexer count of always-selected leading ("init") key blocks
/// (`sparse_init_block`). Shared by `MiniMaxM3SparseAttentionConfiguration`'s
/// memberwise `init` and its `init(from:)` decoder fallback.
public let defaultSparseInitBlocks = 0

/// Default DSA indexer count of always-selected trailing ("local") key
/// blocks ending at the query's own block (`sparse_local_block`). See
/// `defaultSparseInitBlocks`.
public let defaultSparseLocalBlocks = 1

/// Default DSA indexer intra-block score aggregation, `"max"` or `"lse"`
/// (`sparse_score_type`). See `defaultSparseInitBlocks`.
public let defaultSparseScoreType = "max"

/// Default checkpoint model type (`model_type`), verified against the
/// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`. Shared by
/// `MiniMaxM3TextConfiguration`'s memberwise `init` and its `init(from:)`
/// decoder fallback.
public let defaultModelType = "minimax_m3"

/// Default hidden (model) dimension (`hidden_size`). See `defaultModelType`.
public let defaultHiddenSize = 6144

/// Default decoder-layer count (`num_hidden_layers`). See `defaultModelType`.
public let defaultHiddenLayers = 60

/// Default number of query attention heads (`num_attention_heads`). See `defaultModelType`.
public let defaultAttentionHeads = 64

/// Default key/value head count for grouped-query attention
/// (`num_key_value_heads`). See `defaultModelType`.
public let defaultKvHeads = 4

/// Default per-head dimension (`head_dim`). See `defaultModelType`.
public let defaultHeadDim = 128

/// Default vocabulary size (`vocab_size`). See `defaultModelType`.
public let defaultVocabularySize = 200_064

/// Default maximum sequence length supported by RoPE
/// (`max_position_embeddings`). See `defaultModelType`.
public let defaultMaxPositionEmbeddings = 1_048_576

/// Default epsilon used by every RMSNorm in the model (`rms_norm_eps`). See `defaultModelType`.
public let defaultRmsNormEps: Float = 1e-6

/// Default RoPE base frequency (`rope_theta`). See `defaultModelType`.
public let defaultRopeTheta: Float = 5_000_000

/// Default fraction of `headDim` that rotates when `rotaryDim` isn't given
/// explicitly (`partial_rotary_factor`). See `defaultModelType`.
public let defaultPartialRotaryFactor: Float = 0.5

/// Default MLP activation function name (`hidden_act`). See `defaultModelType`.
public let defaultHiddenAct = "swigluoai"

/// Default QK-norm layout (`qk_norm_type`). Also referenced by
/// `MiniMaxM3TextConfiguration.init(from:)`'s decode-time validation, which
/// throws a `DecodingError` if a checkpoint ever requests a different
/// layout. See `defaultModelType`.
public let defaultQkNormType = "per_head"

/// Default dense (non-MoE) MLP intermediate size (`dense_intermediate_size`). See `defaultModelType`.
public let defaultDenseIntermediateSize = 12288

/// Default per-expert MoE intermediate size (`intermediate_size`). See `defaultModelType`.
public let defaultIntermediateSize = 3072

/// Default number of routed experts per MoE layer (`num_local_experts`). See
/// `defaultModelType`. (Distinct from `defaultIndexDim` despite sharing the
/// same numeric value.)
public let defaultNumLocalExperts = 128

/// Default number of routed experts selected per token (`num_experts_per_tok`). See `defaultModelType`.
public let defaultNumExpertsPerTok = 4

/// Default router scoring function name (`scoring_func`). See `defaultModelType`.
public let defaultScoringFunc = "sigmoid"

/// Default multi-token-prediction module count (`num_mtp_modules`). See `defaultModelType`.
public let defaultNumMTPModules = 7

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

    init(
        alpha: Float = defaultSwigluAlpha, limit: Float = defaultSwigluLimit,
        beta: Float = defaultSwigluBeta
    ) {
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
        routedScalingFactor: Float = defaultRoutedScalingFactor,
        swigluAlpha: Float = defaultSwigluAlpha,
        swigluLimit: Float = defaultSwigluLimit,
        swigluBeta: Float = defaultSwigluBeta
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
final class MiniMaxM3SparseMoeBlock: Module {
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

// MARK: - MiniMaxM3SparseAttentionConfiguration

/// MiniMax-M3's DSA (sparse attention indexer) hyperparameters, decoded from
/// `text_config.sparse_attention_config`.
///
/// Seven of the real `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `sparse_attention_config` fields are modeled: the four decoded since
/// kanban ^xgvth41 (`sparse_index_dim`/`sparse_num_index_heads`/
/// `sparse_topk_blocks`/`sparse_block_size`) plus three more this task
/// (^8dbc476) needs to actually run the indexer (`sparse_init_block`,
/// `sparse_local_block`, `sparse_score_type`). `use_sparse_attention`,
/// `sparse_disable_index_value`, and `sparse_attention_freq` remain
/// undecoded and unverified against real values -- this port instead reuses
/// the already-verified `moeLayerFreq` schedule
/// (`MiniMaxM3TextConfiguration.isMoELayer`) to decide which layers build the
/// indexer, rather than introduce a fourth unverified schedule field. They
/// decode as ordinary unknown keys (ignored).
public struct MiniMaxM3SparseAttentionConfiguration: Codable, Sendable {
    /// Per-index-head dimension used by the DSA indexer (`sparse_index_dim`).
    public var indexDim: Int
    /// Number of index heads used by the DSA indexer (`sparse_num_index_heads`).
    public var numIndexHeads: Int
    /// Number of top-scoring blocks the indexer selects per query (`sparse_topk_blocks`).
    public var topkBlocks: Int
    /// Block size, in tokens, used by the DSA indexer (`sparse_block_size`).
    public var blockSize: Int
    /// Number of leading key blocks always force-selected regardless of
    /// score (`sparse_init_block`).
    public var initBlocks: Int
    /// Number of trailing key blocks, ending at the query's own block,
    /// always force-selected regardless of score (`sparse_local_block`).
    public var localBlocks: Int
    /// Intra-block score aggregation: `"max"` (per-token max) or `"lse"`
    /// (log-sum-exp) (`sparse_score_type`). Aggregation across index heads is
    /// always max, regardless of this setting.
    public var scoreType: String

    enum CodingKeys: String, CodingKey {
        case indexDim = "sparse_index_dim"
        case numIndexHeads = "sparse_num_index_heads"
        case topkBlocks = "sparse_topk_blocks"
        case blockSize = "sparse_block_size"
        case initBlocks = "sparse_init_block"
        case localBlocks = "sparse_local_block"
        case scoreType = "sparse_score_type"
    }

    /// Creates a sparse-attention (DSA indexer) configuration, defaulting to
    /// the values verified against the `mlx-community/MiniMax-M3-4bit`
    /// checkpoint's `config.json`.
    public init(
        indexDim: Int = defaultIndexDim,
        numIndexHeads: Int = defaultNumIndexHeads,
        topkBlocks: Int = defaultTopkBlocks,
        blockSize: Int = defaultBlockSize,
        initBlocks: Int = defaultSparseInitBlocks,
        localBlocks: Int = defaultSparseLocalBlocks,
        scoreType: String = defaultSparseScoreType
    ) {
        self.indexDim = indexDim
        self.numIndexHeads = numIndexHeads
        self.topkBlocks = topkBlocks
        self.blockSize = blockSize
        self.initBlocks = initBlocks
        self.localBlocks = localBlocks
        self.scoreType = scoreType
    }

    /// Decodes a sparse-attention (DSA indexer) configuration from
    /// `text_config.sparse_attention_config`, defaulting any field the
    /// checkpoint omits to the value verified against the
    /// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        indexDim = try container.decodeIfPresent(Int.self, forKey: .indexDim) ?? defaultIndexDim
        numIndexHeads =
            try container.decodeIfPresent(Int.self, forKey: .numIndexHeads) ?? defaultNumIndexHeads
        topkBlocks = try container.decodeIfPresent(Int.self, forKey: .topkBlocks) ?? defaultTopkBlocks
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? defaultBlockSize
        initBlocks =
            try container.decodeIfPresent(Int.self, forKey: .initBlocks) ?? defaultSparseInitBlocks
        localBlocks =
            try container.decodeIfPresent(Int.self, forKey: .localBlocks) ?? defaultSparseLocalBlocks
        scoreType =
            try container.decodeIfPresent(String.self, forKey: .scoreType) ?? defaultSparseScoreType
    }
}

// MARK: - MiniMaxM3TextConfiguration

/// MiniMax-M3's text-model configuration.
///
/// Decodes **both** checkpoint shapes:
/// - VL-nested: `model_type: "minimax_m3_vl"`, fields under `text_config`
///   (the published `mlx-community/MiniMax-M3-4bit` checkpoint).
/// - Flat: `model_type: "minimax_m3"`, fields at the JSON root (upstream
///   mlx-lm PR #1401-style text-only conversions).
///
/// Follows `Gemma3TextConfiguration`'s (`Libraries/MLXLLM/Models/
/// Gemma3Text.swift`) fallback-to-`text_config` `init(from:)` pattern:
/// look for a nested `text_config` container first, and only decode from the
/// decoder's own root container when one isn't present. Unknown keys
/// (`vision_config`, `mtp`-related fields, `quantization`) are simply never
/// referenced by `CodingKeys` and so never fail decoding.
///
/// **`encode(to:)` is one-directional, not a round-trip inverse of
/// `init(from:)`.** The synthesized `Encodable` conformance always writes a
/// flat JSON object -- decoding a VL-nested checkpoint (fields under
/// `text_config`) and re-encoding the result does *not* reproduce the
/// original nesting. This mirrors `Gemma3TextConfiguration`'s identical
/// asymmetry and is harmless today: nothing in this repo round-trips these
/// configuration types through `JSONEncoder` (the one real `config.json`
/// rewriter, `ModelConversion.updateModelConfigWithQuantization`, patches a
/// raw `[String: JSONValue]` dictionary instead, specifically to preserve
/// shape fidelity for keys it doesn't model as Swift types). If a future
/// caller needs faithful VL-nested round-tripping, add a custom
/// `encode(to:)` that re-nests under `text_config` when `modelType` is the
/// VL variant -- don't assume the synthesized encoder already does this.
public struct MiniMaxM3TextConfiguration: Codable, Sendable {
    /// Checkpoint model type: `"minimax_m3_vl"` (VL-nested) or `"minimax_m3"` (flat) (`model_type`).
    public var modelType: String
    /// Hidden (model) dimension (`hidden_size`).
    public var hiddenSize: Int
    /// Number of decoder layers (`num_hidden_layers`).
    public var hiddenLayers: Int
    /// Number of query attention heads (`num_attention_heads`).
    public var attentionHeads: Int
    /// Number of key/value heads for grouped-query attention (`num_key_value_heads`).
    public var kvHeads: Int
    /// Per-head dimension (`head_dim`).
    public var headDim: Int
    /// Vocabulary size (`vocab_size`).
    public var vocabularySize: Int
    /// Maximum sequence length supported by RoPE (`max_position_embeddings`).
    public var maxPositionEmbeddings: Int
    /// Epsilon used by every RMSNorm in the model (`rms_norm_eps`).
    public var rmsNormEps: Float
    /// Whether RMSNorm uses Gemma-mode `(1 + weight)` scaling (`use_gemma_norm`).
    public var useGemmaNorm: Bool
    /// RoPE base frequency (`rope_theta`).
    public var ropeTheta: Float
    /// Number of head dimensions that actually rotate (`partialRotaryFactor * headDim`
    /// unless the checkpoint provides `rotary_dim` explicitly). 64 for the
    /// verified real config (128 head dims, factor 0.5).
    public var rotaryDim: Int
    /// Fraction of `headDim` that rotates when `rotaryDim` isn't given explicitly (`partial_rotary_factor`).
    public var partialRotaryFactor: Float
    /// Activation function name used by the dense/MoE MLPs (`hidden_act`).
    public var hiddenAct: String
    /// Whether per-head QK-norm is applied to queries and keys (`use_qk_norm`).
    public var useQkNorm: Bool
    /// QK-norm layout; only `"per_head"` is supported by this dense-attention stage (`qk_norm_type`).
    public var qkNormType: String
    /// Whether the output projection shares weights with the input embedding (`tie_word_embeddings`).
    public var tieWordEmbeddings: Bool
    /// Dense (non-MoE) MLP intermediate size, layers 0..<3 in the verified schedule.
    public var denseIntermediateSize: Int
    /// Per-expert MoE intermediate size.
    public var intermediateSize: Int
    /// Shared-expert MLP intermediate size (`shared_intermediate_size`).
    public var sharedIntermediateSize: Int
    /// Number of routed experts per MoE layer (`num_local_experts`).
    public var numLocalExperts: Int
    /// Number of routed experts selected per token (`num_experts_per_tok`).
    public var numExpertsPerTok: Int
    /// Number of shared experts always selected alongside the routed experts (`n_shared_experts`).
    public var nSharedExperts: Int
    /// Router scoring function name (`scoring_func`).
    public var scoringFunc: String
    /// Whether router selection is biased by `e_score_correction_bias` (`use_routing_bias`).
    public var useRoutingBias: Bool
    /// Scaling factor applied to routed-expert routing weights (`routed_scaling_factor`).
    public var routedScalingFactor: Float
    /// Per-layer dense/MoE schedule (`0` dense, `1` MoE). Verified real
    /// schedule: layers 0-2 dense, 3-59 MoE.
    public var moeLayerFreq: [Int]
    /// SwiGLU-OAI gate-branch sigmoid steepness (`swiglu_alpha`).
    public var swigluAlpha: Float
    /// SwiGLU-OAI clip limit applied to both branches (`swiglu_limit`).
    public var swigluLimit: Float
    /// SwiGLU-OAI linear-branch additive bias (`swiglu_beta`).
    public var swigluBeta: Float
    /// Number of multi-token-prediction modules present in the checkpoint (`num_mtp_modules`).
    public var numMTPModules: Int
    /// DSA (sparse attention indexer) hyperparameters (`sparse_attention_config`).
    public var sparseAttention: MiniMaxM3SparseAttentionConfiguration

    /// Whether `layerIndex` uses the MoE block (`true`) or the dense MLP (`false`).
    public func isMoELayer(_ layerIndex: Int) -> Bool {
        guard layerIndex >= 0, layerIndex < moeLayerFreq.count else { return false }
        return moeLayerFreq[layerIndex] == 1
    }

    /// Creates a text-model configuration directly from field values,
    /// defaulting each parameter to the value verified against the
    /// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`.
    public init(
        modelType: String = defaultModelType,
        hiddenSize: Int = defaultHiddenSize,
        hiddenLayers: Int = defaultHiddenLayers,
        attentionHeads: Int = defaultAttentionHeads,
        kvHeads: Int = defaultKvHeads,
        headDim: Int = defaultHeadDim,
        vocabularySize: Int = defaultVocabularySize,
        maxPositionEmbeddings: Int = defaultMaxPositionEmbeddings,
        rmsNormEps: Float = defaultRmsNormEps,
        useGemmaNorm: Bool = true,
        ropeTheta: Float = defaultRopeTheta,
        rotaryDim: Int? = nil,
        partialRotaryFactor: Float = defaultPartialRotaryFactor,
        hiddenAct: String = defaultHiddenAct,
        useQkNorm: Bool = true,
        qkNormType: String = defaultQkNormType,
        tieWordEmbeddings: Bool = false,
        denseIntermediateSize: Int = defaultDenseIntermediateSize,
        intermediateSize: Int = defaultIntermediateSize,
        sharedIntermediateSize: Int? = nil,
        numLocalExperts: Int = defaultNumLocalExperts,
        numExpertsPerTok: Int = defaultNumExpertsPerTok,
        nSharedExperts: Int = 1,
        scoringFunc: String = defaultScoringFunc,
        useRoutingBias: Bool = true,
        routedScalingFactor: Float = defaultRoutedScalingFactor,
        moeLayerFreq: [Int]? = nil,
        swigluAlpha: Float = defaultSwigluAlpha,
        swigluLimit: Float = defaultSwigluLimit,
        swigluBeta: Float = defaultSwigluBeta,
        numMTPModules: Int = defaultNumMTPModules,
        sparseAttention: MiniMaxM3SparseAttentionConfiguration = MiniMaxM3SparseAttentionConfiguration()
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.vocabularySize = vocabularySize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.useGemmaNorm = useGemmaNorm
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.rotaryDim = rotaryDim ?? Int(partialRotaryFactor * Float(headDim))
        self.hiddenAct = hiddenAct
        self.useQkNorm = useQkNorm
        self.qkNormType = qkNormType
        self.tieWordEmbeddings = tieWordEmbeddings
        self.denseIntermediateSize = denseIntermediateSize
        self.intermediateSize = intermediateSize
        self.sharedIntermediateSize = sharedIntermediateSize ?? intermediateSize
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.nSharedExperts = nSharedExperts
        self.scoringFunc = scoringFunc
        self.useRoutingBias = useRoutingBias
        self.routedScalingFactor = routedScalingFactor
        self.moeLayerFreq =
            moeLayerFreq ?? defaultMoeLayerFreq(hiddenLayers: hiddenLayers)
        self.swigluAlpha = swigluAlpha
        self.swigluLimit = swigluLimit
        self.swigluBeta = swigluBeta
        self.numMTPModules = numMTPModules
        self.sparseAttention = sparseAttention
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case useGemmaNorm = "use_gemma_norm"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case hiddenAct = "hidden_act"
        case useQkNorm = "use_qk_norm"
        case qkNormType = "qk_norm_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case denseIntermediateSize = "dense_intermediate_size"
        case intermediateSize = "intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nSharedExperts = "n_shared_experts"
        case scoringFunc = "scoring_func"
        case useRoutingBias = "use_routing_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case moeLayerFreq = "moe_layer_freq"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case swigluBeta = "swiglu_beta"
        case numMTPModules = "num_mtp_modules"
        case sparseAttention = "sparse_attention_config"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    /// Decodes a text-model configuration from either checkpoint shape:
    /// VL-nested, with fields under a `text_config` container, or flat, with
    /// fields at the decoder's own root -- falling back field-by-field to the
    /// verified defaults whenever a key is absent from either shape.
    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // VL checkpoints (and configs converted via mlx_lm.convert) nest the
        // text-model fields under `text_config`; flat `minimax_m3` configs
        // (upstream mlx-lm PR #1401-style text-only conversions) have them at
        // the root -- mirrors Gemma3TextConfiguration's fallback.
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? defaultModelType
        hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? defaultHiddenSize
        hiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? defaultHiddenLayers
        attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? defaultAttentionHeads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? defaultKvHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? defaultHeadDim
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? defaultVocabularySize
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? defaultMaxPositionEmbeddings
        rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? defaultRmsNormEps
        useGemmaNorm = try container.decodeIfPresent(Bool.self, forKey: .useGemmaNorm) ?? true
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? defaultRopeTheta
        partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? defaultPartialRotaryFactor
        rotaryDim =
            try container.decodeIfPresent(Int.self, forKey: .rotaryDim)
            ?? Int(partialRotaryFactor * Float(headDim))
        hiddenAct =
            try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? defaultHiddenAct
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        qkNormType =
            try container.decodeIfPresent(String.self, forKey: .qkNormType) ?? defaultQkNormType

        // This dense-attention-stage implementation only supports Gemma-mode
        // RMSNorm and per-head QK-norm -- both hold for the one verified real
        // checkpoint, but `use_gemma_norm`/`qk_norm_type` are external,
        // checkpoint-supplied data, so an unsupported value must fail
        // decoding with a typed error rather than crash the process later
        // (flagged by adversarial review of kanban ^xgvth41).
        guard useGemmaNorm else {
            throw DecodingError.dataCorruptedError(
                forKey: .useGemmaNorm, in: container,
                debugDescription:
                    "MiniMaxM3 only supports use_gemma_norm=true (dense-attention stage, ^xgvth41)"
            )
        }
        if useQkNorm {
            guard qkNormType == defaultQkNormType else {
                throw DecodingError.dataCorruptedError(
                    forKey: .qkNormType, in: container,
                    debugDescription:
                        "MiniMaxM3 only supports qk_norm_type=\"per_head\" (dense-attention stage, ^xgvth41)"
                )
            }
        }

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        denseIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize)
            ?? defaultDenseIntermediateSize
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? defaultIntermediateSize
        let decodedSharedIntermediateSize = try container.decodeIfPresent(
            Int.self, forKey: .sharedIntermediateSize)
        numLocalExperts =
            try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? defaultNumLocalExperts
        numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
            ?? defaultNumExpertsPerTok
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? defaultScoringFunc
        useRoutingBias = try container.decodeIfPresent(Bool.self, forKey: .useRoutingBias) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
            ?? defaultRoutedScalingFactor
        let decodedMoeLayerFreq = try container.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
        swigluAlpha =
            try container.decodeIfPresent(Float.self, forKey: .swigluAlpha) ?? defaultSwigluAlpha
        swigluLimit =
            try container.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? defaultSwigluLimit
        swigluBeta =
            try container.decodeIfPresent(Float.self, forKey: .swigluBeta) ?? defaultSwigluBeta
        numMTPModules =
            try container.decodeIfPresent(Int.self, forKey: .numMTPModules) ?? defaultNumMTPModules
        sparseAttention =
            try container.decodeIfPresent(
                MiniMaxM3SparseAttentionConfiguration.self, forKey: .sparseAttention)
            ?? MiniMaxM3SparseAttentionConfiguration()

        self.sharedIntermediateSize = decodedSharedIntermediateSize ?? intermediateSize
        self.moeLayerFreq =
            decodedMoeLayerFreq ?? defaultMoeLayerFreq(hiddenLayers: hiddenLayers)
    }
}

// MARK: - MiniMaxM3VisionConfiguration

/// Default vision-tower hidden dimension (`vision_config.hidden_size`),
/// verified against the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `config.json`. Shared by `MiniMaxM3VisionConfiguration`'s memberwise
/// `init` and its `init(from:)` decoder fallback.
public let defaultVisionHiddenSize = 1280

/// Default vision-tower MLP intermediate size (`vision_config.intermediate_size`). See `defaultVisionHiddenSize`.
public let defaultVisionIntermediateSize = 5120

/// Default vision-tower attention head count (`vision_config.num_attention_heads`). See `defaultVisionHiddenSize`.
public let defaultVisionAttentionHeads = 16

/// Default vision-tower encoder-layer count (`vision_config.num_hidden_layers`). See `defaultVisionHiddenSize`.
public let defaultVisionHiddenLayers = 32

/// Default vision-tower patch size, in pixels (`vision_config.patch_size`). See `defaultVisionHiddenSize`.
public let defaultVisionPatchSize = 14

/// Default vision-tower input channel count (`vision_config.num_channels`). See `defaultVisionHiddenSize`.
public let defaultVisionChannels = 3

/// Default vision-tower LayerNorm epsilon (`vision_config.layer_norm_eps`). See `defaultVisionHiddenSize`.
public let defaultVisionLayerNormEps: Float = 1e-5

/// Default vision-tower MLP activation function name (`vision_config.hidden_act`). See `defaultVisionHiddenSize`.
public let defaultVisionHiddenAct = "gelu"

/// Default vision-tower 3D-RoPE base frequency (`vision_config.rope_theta`). See `defaultVisionHiddenSize`.
public let defaultVisionRopeTheta: Float = 10_000

/// Default patch-merge spatial compression factor
/// (`vision_config.img_token_compression_config.spatial_merge_size`): a
/// `spatialMergeSize x spatialMergeSize` block of adjacent patches is
/// combined into one token by `patch_merge_mlp`. See `defaultVisionHiddenSize`.
public let defaultVisionSpatialMergeSize = 2

/// Default temporal patch size
/// (`vision_config.img_token_compression_config.temporal_patch_size`): the
/// processor replicates a still image this many times along the temporal
/// axis before patchifying, so every image grid has `t == 1`. See
/// `defaultVisionHiddenSize`.
public let defaultVisionTemporalPatchSize = 2

/// Default maximum frame count per video attention segment
/// (`vision_config.vision_segment_max_frames`); a video grid with more
/// frames than this is split into consecutive, independently-attended
/// segments. See `defaultVisionHiddenSize`.
public let defaultVisionSegmentMaxFrames = 4

/// MiniMax-M3's vision-tower configuration, decoded from `vision_config`.
///
/// Only the fields this port's vision tower actually consumes are modeled --
/// `model_type`, `projection_dim`, `position_embedding_type`, `rope_mode`,
/// `vocab_size`, `initializer_range`/`initializer_factor`, and
/// `attention_dropout` are checkpoint metadata this dense (inference-only)
/// port never reads, so they decode as ordinary unknown keys (ignored).
public struct MiniMaxM3VisionConfiguration: Decodable, Sendable {
    /// Vision-tower hidden dimension (`hidden_size`).
    public var hiddenSize: Int
    /// Vision-tower MLP intermediate size (`intermediate_size`).
    public var intermediateSize: Int
    /// Vision-tower attention head count (`num_attention_heads`).
    public var numAttentionHeads: Int
    /// Vision-tower encoder-layer count (`num_hidden_layers`).
    public var numHiddenLayers: Int
    /// Vision-tower patch size, in pixels (`patch_size`).
    public var patchSize: Int
    /// Vision-tower input channel count (`num_channels`).
    public var numChannels: Int
    /// Vision-tower LayerNorm epsilon (`layer_norm_eps`).
    public var layerNormEps: Float
    /// Vision-tower MLP activation function name (`hidden_act`).
    public var hiddenAct: String
    /// 3D-RoPE base frequency (`rope_theta`).
    public var ropeTheta: Float
    /// Patch-merge spatial compression factor
    /// (`img_token_compression_config.spatial_merge_size`).
    public var spatialMergeSize: Int
    /// Temporal patch size (`img_token_compression_config.temporal_patch_size`).
    public var temporalPatchSize: Int
    /// Maximum frame count per video attention segment (`vision_segment_max_frames`).
    public var segmentMaxFrames: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
        case ropeTheta = "rope_theta"
        case compression = "img_token_compression_config"
        case segmentMaxFrames = "vision_segment_max_frames"
    }

    enum CompressionCodingKeys: String, CodingKey {
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
    }

    /// Creates a vision-tower configuration directly from field values,
    /// defaulting each parameter to the value verified against the
    /// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`.
    public init(
        hiddenSize: Int = defaultVisionHiddenSize,
        intermediateSize: Int = defaultVisionIntermediateSize,
        numAttentionHeads: Int = defaultVisionAttentionHeads,
        numHiddenLayers: Int = defaultVisionHiddenLayers,
        patchSize: Int = defaultVisionPatchSize,
        numChannels: Int = defaultVisionChannels,
        layerNormEps: Float = defaultVisionLayerNormEps,
        hiddenAct: String = defaultVisionHiddenAct,
        ropeTheta: Float = defaultVisionRopeTheta,
        spatialMergeSize: Int = defaultVisionSpatialMergeSize,
        temporalPatchSize: Int = defaultVisionTemporalPatchSize,
        segmentMaxFrames: Int = defaultVisionSegmentMaxFrames
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.layerNormEps = layerNormEps
        self.hiddenAct = hiddenAct
        self.ropeTheta = ropeTheta
        self.spatialMergeSize = spatialMergeSize
        self.temporalPatchSize = temporalPatchSize
        self.segmentMaxFrames = segmentMaxFrames
    }

    /// Decodes a vision-tower configuration, defaulting any field the
    /// checkpoint omits (including the entire nested
    /// `img_token_compression_config` object) to the value verified against
    /// the `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? defaultVisionHiddenSize
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? defaultVisionIntermediateSize
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? defaultVisionAttentionHeads
        numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? defaultVisionHiddenLayers
        patchSize =
            try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? defaultVisionPatchSize
        numChannels =
            try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? defaultVisionChannels
        layerNormEps =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
            ?? defaultVisionLayerNormEps
        hiddenAct =
            try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? defaultVisionHiddenAct
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? defaultVisionRopeTheta
        segmentMaxFrames =
            try container.decodeIfPresent(Int.self, forKey: .segmentMaxFrames)
            ?? defaultVisionSegmentMaxFrames

        if container.contains(.compression) {
            let compression = try container.nestedContainer(
                keyedBy: CompressionCodingKeys.self, forKey: .compression)
            spatialMergeSize =
                try compression.decodeIfPresent(Int.self, forKey: .spatialMergeSize)
                ?? defaultVisionSpatialMergeSize
            temporalPatchSize =
                try compression.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
                ?? defaultVisionTemporalPatchSize
        } else {
            spatialMergeSize = defaultVisionSpatialMergeSize
            temporalPatchSize = defaultVisionTemporalPatchSize
        }
    }
}

// MARK: - MiniMaxM3Vision

/// MiniMax-M3's vision tower: a CLIP-style ViT with 3D (time/height/width)
/// RoPE instead of learned position embeddings, and no in-tower token
/// compression (patch-merge token compression happens at the
/// `MiniMaxM3Model` level, in `patch_merge_mlp`, after the
/// `multi_modal_projector` -- see `MiniMaxM3Model.mergeVisualTokens`).
///
/// Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/vision.py`
/// (`MiniMaxVisionEmbeddings`, `MiniMaxVisionAttention`, `MiniMaxVisionMLP`,
/// `MiniMaxVisionEncoderLayer`, `MiniMaxVisionEncoder`,
/// `MiniMaxVisionTransformer`, `VisionModel`), verified against the
/// `mlx-community/MiniMax-M3-4bit` checkpoint's `vision_config` in
/// `config.json`.
///
/// **Only the default feature-selection path is implemented.** The real
/// checkpoint's top-level `vision_feature_layer`/`vision_feature_select_strategy`
/// are `-1`/`"full"` -- the plain "run every encoder layer once, project the
/// final hidden state" path mlx-vlm's `Model._compute_visual_features` takes
/// when `use_hidden_states` is `false`. Multi-layer hidden-state selection
/// (`vision_feature_layer` as a list, or `"default"` strategy dropping the
/// CLS token) is not modeled since no verified checkpoint exercises it.
enum MiniMaxM3Vision {

    /// Rotates the second half of `x`'s last dimension into the first half,
    /// negated -- the standard RoPE half-rotation, mirroring mlx-vlm's
    /// `_rotate_half` (`vision.py`).
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    /// Applies 3D RoPE to `x` (shape `(sequence, heads, headDim)`) using
    /// precomputed `cos`/`sin` tables (shape `(sequence, rotaryDim)`),
    /// leaving any trailing dimensions beyond `rotaryDim` untouched --
    /// mirrors mlx-vlm's `_apply_vision_rope`. `rotaryDim` need not equal
    /// `headDim`: for the verified default config (`headDim` 80, 3 axes),
    /// only 78 of 80 head dimensions rotate.
    static func applyRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        guard cos.size > 0 else { return x }
        let rotaryDim = cos.dim(-1)
        let cosBroadcast = expandedDimensions(cos, axis: 1)
        let sinBroadcast = expandedDimensions(sin, axis: 1)
        let rotated = x[.ellipsis, 0 ..< rotaryDim]
        let passthrough = x[.ellipsis, rotaryDim...]
        let result = rotated * cosBroadcast + rotateHalf(rotated) * sinBroadcast
        return concatenated([result, passthrough], axis: -1).asType(x.dtype)
    }

    /// One axis's (time, height, or width) RoPE frequency table: `theta`-based
    /// inverse frequencies outer-producted against `coords` -- mirrors
    /// mlx-vlm's `_axis_freq`.
    static func axisFrequencies(_ coords: MLXArray, dimensions: Int, theta: Float) -> MLXArray {
        let inverseFrequency =
            1.0
            / pow(
                MLXArray(theta),
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                    / Float(dimensions))
        return outer(coords.asType(.float32), inverseFrequency)
    }

    /// Splits `grids` so no segment exceeds `maxFrames` temporal frames --
    /// mirrors mlx-vlm's `_segment_grid_thw`. Every image grid has `t == 1`
    /// (see `defaultVisionTemporalPatchSize`), so this only ever subdivides
    /// video grids; each independent segment gets its own non-causal
    /// attention block (see `cumulativeSequenceLengths`).
    static func segmentGrids(_ grids: [THW], maxFrames: Int) -> [THW] {
        var segments: [THW] = []
        for grid in grids {
            guard grid.t > maxFrames else {
                segments.append(grid)
                continue
            }
            var start = 0
            while start < grid.t {
                segments.append(THW(min(maxFrames, grid.t - start), grid.h, grid.w))
                start += maxFrames
            }
        }
        return segments
    }

    /// Builds the flattened, per-token 3D-RoPE `cos`/`sin` tables for every
    /// grid in `grids`, concatenated in order -- mirrors mlx-vlm's
    /// `MiniMaxVisionTransformer._rotary_pos_emb`. Token order within each
    /// grid matches `QwenVL.patchify`'s patch order exactly: `(t, h/merge,
    /// w/merge, mergeRow, mergeCol)`, flattened row-major.
    static func rotaryPositionEmbedding(
        grids: [THW], config: MiniMaxM3VisionConfiguration
    ) -> (cos: MLXArray, sin: MLXArray) {
        let mergeSize = config.spatialMergeSize
        let headDim = config.hiddenSize / config.numAttentionHeads
        let rotaryDims = 2 * (headDim / 2)
        let axisDim = 2 * ((rotaryDims / 3) / 2)

        var freqSegments: [MLXArray] = []
        for grid in segmentGrids(grids, maxFrames: config.segmentMaxFrames) {
            let mergedH = grid.h / mergeSize
            let mergedW = grid.w / mergeSize
            guard grid.t > 0, mergedH > 0, mergedW > 0 else { continue }

            let shape = [grid.t, mergedH, mergedW, mergeSize, mergeSize]
            let tIndex = broadcast(
                MLXArray(0 ..< grid.t).asType(.int32).reshaped([grid.t, 1, 1, 1, 1]), to: shape)
            let hBlock = MLXArray(0 ..< mergedH).asType(.int32).reshaped([1, mergedH, 1, 1, 1])
            let wBlock = MLXArray(0 ..< mergedW).asType(.int32).reshaped([1, 1, mergedW, 1, 1])
            let intraRow = MLXArray(0 ..< mergeSize).asType(.int32).reshaped([1, 1, 1, mergeSize, 1])
            let intraCol = MLXArray(0 ..< mergeSize).asType(.int32).reshaped([1, 1, 1, 1, mergeSize])
            let hIndex = broadcast(hBlock * Int32(mergeSize) + intraRow, to: shape)
            let wIndex = broadcast(wBlock * Int32(mergeSize) + intraCol, to: shape)

            let freqs = concatenated(
                [
                    axisFrequencies(tIndex.flattened(), dimensions: axisDim, theta: config.ropeTheta),
                    axisFrequencies(hIndex.flattened(), dimensions: axisDim, theta: config.ropeTheta),
                    axisFrequencies(wIndex.flattened(), dimensions: axisDim, theta: config.ropeTheta),
                ], axis: -1)
            freqSegments.append(concatenated([freqs, freqs], axis: -1))
        }

        guard !freqSegments.isEmpty else {
            return (MLXArray.zeros([0, 0]), MLXArray.zeros([0, 0]))
        }
        let allFrequencies = concatenated(freqSegments, axis: 0)
        return (MLX.cos(allFrequencies), MLX.sin(allFrequencies))
    }

    /// Cumulative per-segment token-count boundaries (`[0, len(segment0),
    /// len(segment0)+len(segment1), ...]`) used to build the block-diagonal
    /// attention mask -- mirrors mlx-vlm's `cu_seqlens` (built from
    /// `_segment_grid_thw`, matching `rotaryPositionEmbedding`'s segmentation
    /// exactly so RoPE and masking stay aligned).
    static func cumulativeSequenceLengths(grids: [THW], maxFrames: Int) -> [Int] {
        var boundaries = [0]
        var total = 0
        for grid in segmentGrids(grids, maxFrames: maxFrames) {
            total += grid.product
            boundaries.append(total)
        }
        return boundaries
    }

    /// `vision_tower.vision_model.embeddings`: a single non-overlapping
    /// patch-embedding projection (patch stride == patch size, so this is
    /// exactly a linear projection over each already-patchified token, not a
    /// true convolution) -- mirrors mlx-vlm's `MiniMaxVisionEmbeddings` /
    /// `MiniMaxVisionPatchEmbedding`. The checkpoint stores the projection
    /// weight in Conv3d layout (`hidden, channels, temporalPatch, patch,
    /// patch`); `MiniMaxM3Model.sanitize` flattens it to the 2D `Linear`
    /// layout this module expects.
    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        let patchDimensions: Int

        init(_ config: MiniMaxM3VisionConfiguration) {
            self.patchDimensions =
                config.numChannels * config.temporalPatchSize * config.patchSize * config.patchSize
            _patchEmbedding.wrappedValue = Linear(patchDimensions, config.hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            patchEmbedding(x.reshaped(-1, patchDimensions))
        }
    }

    /// `self_attn`: full (non-causal) multi-head attention over 3D-RoPE'd
    /// queries/keys, block-diagonal across `cuSeqlens` segments -- mirrors
    /// mlx-vlm's `MiniMaxVisionAttention`. Follows the same additive-mask
    /// construction as `Qwen3VLVision.Attention` (`Qwen3VL.swift`).
    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniMaxM3VisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            self.headDim = config.hiddenSize / config.numAttentionHeads
            self.scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var q = qProj(x).reshaped(sequenceLength, numHeads, headDim)
            var k = kProj(x).reshaped(sequenceLength, numHeads, headDim)
            let v = vProj(x).reshaped(sequenceLength, numHeads, headDim)

            q = MiniMaxM3Vision.applyRoPE(q, cos: cos, sin: sin)
            k = MiniMaxM3Vision.applyRoPE(k, cos: cos, sin: sin)

            let queries = q.transposed(1, 0, 2)[.newAxis]
            let keys = k.transposed(1, 0, 2)[.newAxis]
            let values = v.transposed(1, 0, 2)[.newAxis]

            var mask = MLXArray.ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
            mask = mask * MLXArray(-1e9, dtype: queries.dtype)
            for i in 1 ..< cuSeqlens.count {
                let start = cuSeqlens[i - 1]
                let end = cuSeqlens[i]
                mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask)
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return outProj(attended)
        }
    }

    /// Applies the config-selected MLP activation named by `hiddenAct`
    /// (`quick_gelu`, `silu`, or `gelu` -- the verified default) -- shared by
    /// `MLP.callAsFunction` and `MiniMaxM3Projector.callAsFunction`, both of
    /// which apply the same activation between a checkpoint's two-layer MLP
    /// projections.
    static func applyActivation(_ x: MLXArray, hiddenAct: String) -> MLXArray {
        switch hiddenAct {
        case "quick_gelu":
            return x * sigmoid(1.702 * x)
        case "silu":
            return silu(x)
        default:
            return gelu(x)
        }
    }

    /// `mlp`: two-layer MLP with a config-selected activation (`quick_gelu`,
    /// `silu`, or `gelu` -- the verified default) -- mirrors mlx-vlm's
    /// `MiniMaxVisionMLP`.
    final class MLP: Module {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let hiddenAct: String

        init(_ config: MiniMaxM3VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            self.hiddenAct = config.hiddenAct
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = applyActivation(fc1(x), hiddenAct: hiddenAct)
            return fc2(h)
        }
    }

    /// One pre-norm transformer block: `self_attn` then `mlp`, each with its
    /// own residual -- mirrors mlx-vlm's `MiniMaxVisionEncoderLayer`.
    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: MiniMaxM3VisionConfiguration) {
            _selfAttn.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, cuSeqlens: [Int], cos: MLXArray, sin: MLXArray) -> MLXArray {
            var h = x + selfAttn(layerNorm1(x), cuSeqlens: cuSeqlens, cos: cos, sin: sin)
            h = h + mlp(layerNorm2(h))
            return h
        }
    }

    /// `encoder`: a plain stack of `EncoderLayer`s -- mirrors mlx-vlm's
    /// `MiniMaxVisionEncoder`.
    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: MiniMaxM3VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
            super.init()
        }

        func callAsFunction(_ x: MLXArray, cuSeqlens: [Int], cos: MLXArray, sin: MLXArray) -> MLXArray {
            var h = x
            for layer in layers {
                h = layer(h, cuSeqlens: cuSeqlens, cos: cos, sin: sin)
            }
            return h
        }
    }

    /// `vision_tower.vision_model`: embeddings, a pre-encoder LayerNorm (key
    /// `pre_layrnorm` -- the checkpoint's own spelling, preserved verbatim so
    /// weight loading matches), and the encoder stack. No post-encoder norm
    /// and no in-tower merger: the final encoder layer's raw per-patch hidden
    /// states are returned directly, matching the verified checkpoint's
    /// `vision_feature_layer: -1, vision_feature_select_strategy: "full"` --
    /// mirrors mlx-vlm's `MiniMaxVisionTransformer`.
    final class Transformer: Module {
        let config: MiniMaxM3VisionConfiguration
        @ModuleInfo(key: "embeddings") var embeddings: Embeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
        @ModuleInfo(key: "encoder") var encoder: Encoder

        init(_ config: MiniMaxM3VisionConfiguration) {
            self.config = config
            _embeddings.wrappedValue = Embeddings(config)
            _preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _encoder.wrappedValue = Encoder(config)
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> MLXArray {
            var hidden = embeddings(pixelValues).reshaped(-1, config.hiddenSize)
            hidden = preLayerNorm(hidden)
            let (cosTable, sinTable) = MiniMaxM3Vision.rotaryPositionEmbedding(
                grids: gridTHW, config: config)
            let cuSeqlens = MiniMaxM3Vision.cumulativeSequenceLengths(
                grids: gridTHW, maxFrames: config.segmentMaxFrames)
            return encoder(hidden, cuSeqlens: cuSeqlens, cos: cosTable, sin: sinTable)
        }
    }

    /// `vision_tower`: the checkpoint's top-level wrapper around
    /// `vision_model` -- mirrors mlx-vlm's `VisionModel`.
    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var transformer: Transformer

        init(_ config: MiniMaxM3VisionConfiguration) {
            _transformer.wrappedValue = Transformer(config)
            super.init()
        }

        /// The dtype pixel values must be cast to before this vision tower
        /// consumes them, taken from the patch-embedding projection weight --
        /// mirrors mlx-vlm's `Model._compute_visual_features` dtype cast.
        var inputDtype: DType { transformer.embeddings.patchEmbedding.weight.dtype }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> MLXArray {
            transformer(pixelValues, gridTHW: gridTHW)
        }
    }
}

// MARK: - MiniMaxM3Projector

/// A two-layer MLP with a config-selected activation, used for both
/// `multi_modal_projector` (vision hidden size -> text hidden size) and
/// `patch_merge_mlp` (`spatialMergeSize^2`-concatenated projected tokens ->
/// text hidden size) -- mirrors mlx-vlm's `MiniMaxProjector`, which the
/// checkpoint's `Model` reuses for both roles with different dimensions/bias.
final class MiniMaxM3Projector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let hiddenAct: String

    init(inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int, bias: Bool, hiddenAct: String) {
        _linear1.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: bias)
        _linear2.wrappedValue = Linear(hiddenDimensions, outputDimensions, bias: bias)
        self.hiddenAct = hiddenAct
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = MiniMaxM3Vision.applyActivation(linear1(x), hiddenAct: hiddenAct)
        return linear2(h)
    }
}

// MARK: - MiniMaxM3Configuration

/// Default image-token id spliced into the prompt at each image's token
/// positions (`image_token_index`), verified against the
/// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`. Shared by
/// `MiniMaxM3Configuration`'s memberwise `init` and its `init(from:)` decoder
/// fallback.
public let defaultImageTokenIndex = 200_025

/// Default video-token id (`video_token_index`). See `defaultImageTokenIndex`.
public let defaultVideoTokenIndex = 200_026

/// Default `multi_modal_projector`/`patch_merge_mlp` activation function name
/// (`projector_hidden_act`). See `defaultImageTokenIndex`.
public let defaultProjectorHiddenAct = "gelu"

/// Default `multi_modal_projector` hidden dimension (`projector_hidden_size`). See `defaultImageTokenIndex`.
public let defaultProjectorHiddenSize = 6144

/// Top-level MiniMax-M3 VL configuration: `text_config` plus `vision_config`
/// (vision tower + merger + splicing, kanban ^9a2aw98).
public struct MiniMaxM3Configuration: Decodable, Sendable {
    /// Top-level checkpoint model type (`"minimax_m3_vl"`).
    public var modelType: String
    /// The text-model configuration used by the dense-attention language model.
    public var textConfiguration: MiniMaxM3TextConfiguration
    /// The vision-tower configuration.
    public var visionConfiguration: MiniMaxM3VisionConfiguration
    /// Token id marking an image's spliced-embedding positions (`image_token_index`).
    public var imageTokenIndex: Int
    /// Token id marking a video's spliced-embedding positions (`video_token_index`).
    public var videoTokenIndex: Int
    /// `multi_modal_projector`/`patch_merge_mlp` activation function name (`projector_hidden_act`).
    public var projectorHiddenAct: String
    /// `multi_modal_projector` hidden dimension (`projector_hidden_size`).
    public var projectorHiddenSize: Int
    /// Whether `multi_modal_projector`'s linear layers carry a bias (`multimodal_projector_bias`).
    public var multimodalProjectorBias: Bool
    /// Whether `patch_merge_mlp`'s linear layers carry a bias (`patch_merge_bias`).
    public var patchMergeBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_index"
        case videoTokenIndex = "video_token_index"
        case projectorHiddenAct = "projector_hidden_act"
        case projectorHiddenSize = "projector_hidden_size"
        case multimodalProjectorBias = "multimodal_projector_bias"
        case patchMergeBias = "patch_merge_bias"
    }

    /// Creates a top-level MiniMax-M3 configuration from its model type, text
    /// configuration, and vision configuration.
    public init(
        modelType: String = "minimax_m3_vl",
        textConfiguration: MiniMaxM3TextConfiguration = MiniMaxM3TextConfiguration(),
        visionConfiguration: MiniMaxM3VisionConfiguration = MiniMaxM3VisionConfiguration(),
        imageTokenIndex: Int = defaultImageTokenIndex,
        videoTokenIndex: Int = defaultVideoTokenIndex,
        projectorHiddenAct: String = defaultProjectorHiddenAct,
        projectorHiddenSize: Int = defaultProjectorHiddenSize,
        multimodalProjectorBias: Bool = true,
        patchMergeBias: Bool = true
    ) {
        self.modelType = modelType
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.imageTokenIndex = imageTokenIndex
        self.videoTokenIndex = videoTokenIndex
        self.projectorHiddenAct = projectorHiddenAct
        self.projectorHiddenSize = projectorHiddenSize
        self.multimodalProjectorBias = multimodalProjectorBias
        self.patchMergeBias = patchMergeBias
    }

    /// Decodes a top-level configuration, defaulting any field the checkpoint
    /// omits (including the entire `vision_config` object) to the value
    /// verified against the `mlx-community/MiniMax-M3-4bit` checkpoint's
    /// `config.json`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3_vl"
        textConfiguration =
            try container.decodeIfPresent(MiniMaxM3TextConfiguration.self, forKey: .textConfiguration)
            ?? MiniMaxM3TextConfiguration()
        visionConfiguration =
            try container.decodeIfPresent(
                MiniMaxM3VisionConfiguration.self, forKey: .visionConfiguration)
            ?? MiniMaxM3VisionConfiguration()
        imageTokenIndex =
            try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex)
            ?? defaultImageTokenIndex
        videoTokenIndex =
            try container.decodeIfPresent(Int.self, forKey: .videoTokenIndex)
            ?? defaultVideoTokenIndex
        projectorHiddenAct =
            try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct)
            ?? defaultProjectorHiddenAct
        projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize)
            ?? defaultProjectorHiddenSize
        multimodalProjectorBias =
            try container.decodeIfPresent(Bool.self, forKey: .multimodalProjectorBias) ?? true
        patchMergeBias = try container.decodeIfPresent(Bool.self, forKey: .patchMergeBias) ?? true
    }
}

// MARK: - MiniMaxM3KVCache

/// Sparse-attention-aware KV cache for MiniMax-M3's DSA layers (3-59):
/// wraps a plain `KVCacheSimple` for the regular grouped-query K/V, plus an
/// independently-tracked `indexKeys` buffer (the DSA indexer's own
/// single-head `index_k_proj` output) and `indexOffset`, mirroring
/// mlx-vlm's `MiniMaxM3KVCache` (`language.py`).
///
/// **Does not participate in `MLXFoundationModels` prompt-cache reuse.**
/// `PromptCache.isChunkable`/`isHybridMambaAttention` (in the
/// `MLXFoundationModels` package) recognize `KVCacheSimple` and hybrid
/// Mamba/attention cache mixes, but neither check matches this type -- so
/// `MLXLanguageModel.supportsPromptCacheReuse` correctly reports `false` for
/// any MiniMax-M3 session, and prompt prefill always runs from scratch
/// rather than reusing a cached prefix. This is a deliberate, documented
/// limitation of this task (^8dbc476) rather than a bug: teaching
/// `PromptCache` to recognize a cache type with two independent offsets
/// (`offset` and `indexOffset`) is separate follow-up work, not fixed here.
public final class MiniMaxM3KVCache: KVCache {
    /// Number of `state` array elements when no index-key history has been
    /// recorded yet (`keys`, `values`). See the `state` setter.
    private static let expectedStateComponentsWithoutIndex = 2
    /// Number of `state` array elements once index-key history is present
    /// (`keys`, `values`, `indexKeys`). See the `state` setter.
    private static let expectedStateComponentsWithIndex = 3

    private let kvCache = KVCacheSimple()
    private var indexKeys: MLXArray?
    /// Growth step size for `indexKeys`. Reads `kvCache.step` directly
    /// (rather than a separately-hardcoded literal) so the two buffers'
    /// growth chunking can never drift out of sync.
    private var indexStep: Int { kvCache.step }

    /// Number of tokens present in `indexKeys`. Tracked independently from
    /// `offset` -- mirroring the reference cache -- even though the two
    /// normally advance in lockstep during ordinary single-sequence
    /// generation (this cache has no independent trim path that could ever
    /// desync them; see `isTrimmable`).
    public private(set) var indexOffset = 0

    /// Number of tokens currently held in the regular (non-index) key/value
    /// cache -- the current KV cache offset, forwarded from the wrapped
    /// `KVCacheSimple`.
    public var offset: Int { kvCache.offset }

    /// Always `nil`: this cache, like the `KVCacheSimple` it wraps for the
    /// regular K/V buffers, has no maximum size limit.
    public var maxSize: Int? { nil }

    /// Registers `MiniMaxM3KVCache` with `KVCacheSerializationRegistry` so
    /// `savePromptCache`/`loadPromptCache` round-trip it correctly instead of
    /// silently misclassifying it as a plain `KVCacheSimple` (which only
    /// accepts a 2-array state and crashes once an `indexKeys` history is
    /// present). Triggered once, lazily, from `init()` below -- guaranteed to
    /// run before any instance could be saved.
    private static let registerSerialization: Void = {
        KVCacheSerializationRegistry.register(MiniMaxM3KVCache.self, className: "MiniMaxM3KVCache") {
            state, metaState in
            let cache = MiniMaxM3KVCache()
            cache.state = state
            cache.metaState = metaState
            return cache
        }
    }()

    /// Creates an empty sparse-attention KV cache.
    public init() {
        Self.registerSerialization
    }

    /// Appends `keys`/`values` to the regular (non-index) key/value cache
    /// and returns the full accumulated history, forwarding directly to the
    /// wrapped `KVCacheSimple`.
    ///
    /// - Parameters:
    ///   - keys: New-token keys, shaped `[batch, kvHeads, newTokens, headDim]`.
    ///   - values: New-token values, shaped like `keys`.
    /// - Returns: The full accumulated `(keys, values)` history.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kvCache.update(keys: keys, values: values)
    }

    /// Appends the indexer's new-token key projections to `indexKeys` and
    /// returns the full accumulated index-key history, mirroring mlx-vlm's
    /// `MiniMaxM3KVCache.update_index_and_fetch`.
    func updateIndexAndFetch(_ keys: MLXArray) -> MLXArray {
        let previous = indexOffset
        let incoming = keys.dim(2)

        let needsGrowth = indexKeys.map { previous + incoming > $0.dim(2) } ?? true
        if needsGrowth {
            let (B, heads, headDim) = (keys.dim(0), keys.dim(1), keys.dim(3))
            let steps = (indexStep + incoming - 1) / indexStep
            let newKeys = MLXArray.zeros(
                [B, heads, steps * indexStep, headDim], dtype: keys.dtype)
            if var current = indexKeys {
                if previous % indexStep != 0 {
                    current = current[.ellipsis, ..<previous, 0...]
                }
                indexKeys = concatenated([current, newKeys], axis: 2)
            } else {
                indexKeys = newKeys
            }
        }

        indexOffset += incoming
        indexKeys?[.ellipsis, previous ..< indexOffset, 0...] = keys
        guard let indexKeys else {
            fatalError("indexKeys must be initialized by the growth step above")
        }
        return indexKeys[.ellipsis, ..<indexOffset, 0...]
    }

    /// Returns the raw arrays backing this cache's persisted state: the
    /// wrapped `KVCacheSimple`'s inner state (keys, values), plus the raw,
    /// untrimmed `indexKeys` buffer if one has been allocated yet.
    public func innerState() -> [MLXArray] {
        kvCache.innerState() + (indexKeys.map { [$0] } ?? [])
    }

    /// The cache's serializable state: `[keys, values]` when no index-key
    /// history has been recorded yet, or `[keys, values, indexKeys]` once it
    /// has (with `indexKeys` trimmed to `indexOffset` on read). Setting
    /// reconstructs the wrapped `KVCacheSimple` and `indexKeys`/`indexOffset`
    /// from the supplied arrays; see `expectedStateComponentsWithoutIndex`/
    /// `expectedStateComponentsWithIndex` for the accepted array counts.
    public var state: [MLXArray] {
        get {
            var result = kvCache.state
            if let indexKeys {
                result.append(indexKeys[.ellipsis, ..<indexOffset, 0...])
            }
            return result
        }
        set {
            switch newValue.count {
            case Self.expectedStateComponentsWithoutIndex:
                kvCache.state = newValue
                indexKeys = nil
                indexOffset = 0
            case Self.expectedStateComponentsWithIndex:
                kvCache.state = [newValue[0], newValue[1]]
                indexKeys = newValue[2]
                guard let indexKeys else {
                    fatalError("indexKeys must be non-nil immediately after assignment above")
                }
                indexOffset = indexKeys.dim(2)
            default:
                fatalError(
                    "MiniMaxM3KVCache state must have \(Self.expectedStateComponentsWithoutIndex) (keys, values) or \(Self.expectedStateComponentsWithIndex) (+ indexKeys) arrays, got \(newValue.count)"
                )
            }
        }
    }

    /// Serialized `indexOffset`, carried alongside `state` for cache
    /// checkpointing/restoration: a single-element array holding the
    /// string-encoded token count of `indexKeys`. Defaults to `0` when
    /// absent or unparsable.
    public var metaState: [String] {
        get { [String(indexOffset)] }
        set { indexOffset = newValue.first.flatMap { Int($0) } ?? 0 }
    }

    /// Always `false`. mlx-vlm's reference cache supports trimming both
    /// buffers in lockstep, but this Swift port's first sparse-cache landing
    /// (^8dbc476) keeps trimming out of scope -- nothing in this repo's
    /// generation loop currently needs to trim a MiniMax-M3 session.
    public var isTrimmable: Bool { false }

    /// Trimming is not supported by this cache; always returns `0` (no
    /// tokens trimmed). See `isTrimmable`.
    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    /// Builds the attention mask mode for the next `n` query positions,
    /// forwarding directly to the wrapped `KVCacheSimple`'s causal-mask
    /// construction (this cache has no sliding-window-aware masking of its
    /// own -- see `MiniMaxM3Attention.resolveMask`'s doc comment for why a
    /// sliding-window cache instead falls back to dense masking).
    ///
    /// - Parameters:
    ///   - n: Number of new query positions to build a mask for.
    ///   - windowSize: Sliding-window size, or `nil` for unrestricted causal attention.
    ///   - returnArray: When `true`, forces a materialized mask array rather than a deferred mask mode.
    /// - Returns: The mask mode to pass to scaled-dot-product attention.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kvCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Creates a deep copy of this cache's current state (regular K/V plus
    /// `indexKeys`/`indexOffset`), independent of the original.
    public func copy() -> any KVCache {
        let new = MiniMaxM3KVCache()
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }
}

// MARK: - MiniMaxM3Indexer

/// MiniMax-M3's MSA (sparse attention) block selector: scores every causal
/// key block against the current query, force-selects the leading
/// `initBlocks` and trailing `localBlocks` blocks, and returns the top-
/// `topkBlocks` block indices per query -- mirrors mlx-vlm's
/// `MiniMaxM3Indexer` (`language.py`).
///
/// Not itself an `MLXNN.Module`, mirroring the Python reference exactly: the
/// learnable `index_q_proj`/`index_k_proj`/`index_q_norm`/`index_k_norm`
/// submodules live directly on the owning `MiniMaxM3Attention` (so the
/// checkpoint's flat `self_attn.index_*` weight keys are preserved -- a
/// nested `indexer` submodule would key-path them as
/// `self_attn.indexer.index_*` instead, which would not match the real
/// checkpoint), and this struct only borrows references to them plus the
/// attention's shared `rope` instance.
struct MiniMaxM3Indexer {
    let indexHeads: Int
    let indexDim: Int
    let blockSize: Int
    let topkBlocks: Int
    let initBlocks: Int
    let localBlocks: Int
    let scoreType: String
    let scale: Float

    let indexQProj: Linear
    let indexKProj: Linear
    let indexQNorm: Gemma.RMSNorm
    let indexKNorm: Gemma.RMSNorm
    let rope: RoPELayer

    /// Projects and RoPE-embeds this call's own index queries/keys, folds the
    /// new index keys into `cache`'s independent index-key history (if a
    /// cache is given), and either selects the top-`topkBlocks` key blocks
    /// per query or signals the dense fallback (`nil`) when the accumulated
    /// index-key history is short enough that every causal block would be
    /// selected anyway (`totalLen <= topkBlocks * blockSize` -- mirrors
    /// mlx-vlm's `MiniMaxM3Indexer.__call__`; for the verified default
    /// config this is 2048 tokens).
    ///
    /// - Returns: `(blockIndices, totalLen)` when block selection ran, or
    ///   `nil` for the dense fallback. `blockIndices` has shape
    ///   `(B, L, min(topkBlocks, numBlocks))`, ascending-sorted per query
    ///   with `-1` marking unused slots.
    func callAsFunction(
        _ x: MLXArray, offset: RoPEOffset?, cache: MiniMaxM3KVCache?, qStart: Int
    ) -> (blockIndices: MLXArray, totalLen: Int)? {
        let (B, L) = (x.dim(0), x.dim(1))

        var idxQ = indexQProj(x).reshaped(B, L, indexHeads, indexDim)
        var idxK = indexKProj(x).reshaped(B, L, 1, indexDim)
        idxQ = indexQNorm(idxQ).transposed(0, 2, 1, 3)
        idxK = indexKNorm(idxK).transposed(0, 2, 1, 3)
        idxQ = applyRotaryPosition(rope, to: idxQ, offset: offset)
        idxK = applyRotaryPosition(rope, to: idxK, offset: offset)

        if let cache {
            idxK = cache.updateIndexAndFetch(idxK)
        }

        let totalLen = idxK.dim(2)
        guard totalLen > blockSize * topkBlocks else { return nil }

        let blockIndices = selectBlocks(idxQueries: idxQ, idxKeys: idxK, qStart: qStart)
        return (blockIndices, totalLen)
    }

    /// Scores every causal key block against `idxQueries` (max- or
    /// lse-aggregated over each block's tokens, then max-aggregated across
    /// index heads), force-selects the leading `initBlocks` and trailing
    /// `localBlocks` blocks, and returns the ascending-sorted indices of the
    /// top `min(topkBlocks, numBlocks)` blocks per query (`-1` for unused
    /// slots) -- mirrors mlx-vlm's `MiniMaxM3Indexer.select_blocks`
    /// (mask-free path; this port has no batched left-padding to support).
    private func selectBlocks(idxQueries: MLXArray, idxKeys: MLXArray, qStart: Int) -> MLXArray {
        let (B, hIdx, L) = (idxQueries.dim(0), idxQueries.dim(1), idxQueries.dim(2))
        let totalLen = idxKeys.dim(2)
        let numBlocks = (totalLen + blockSize - 1) / blockSize
        let neg = MLXArray(-Float.infinity)

        var scores = matmul(
            idxQueries.asType(.float32), idxKeys.asType(.float32).swappedAxes(-1, -2))
        scores = scores * scale

        let qPositions = MLXArray(Int32(qStart) ..< Int32(qStart + L))
        let kPositions = MLXArray(Int32(0) ..< Int32(totalLen))
        let tokenCausal = kPositions.reshaped(1, totalLen) .<= qPositions.reshaped(L, 1)
        scores = which(tokenCausal, scores, neg)

        let pad = numBlocks * blockSize - totalLen
        if pad > 0 {
            scores = padded(
                scores,
                widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, pad))],
                value: neg)
        }

        let blocks = MLXArray(Int32(0) ..< Int32(numBlocks))
        let curBlock = qPositions.floorDivide(blockSize)
        let causalBlock = blocks.reshaped(1, numBlocks) .<= curBlock.reshaped(L, 1)

        let blockedScores = scores.reshaped(B, hIdx, L, numBlocks, blockSize)
        let intraBlock =
            scoreType == "lse"
            ? logSumExp(blockedScores, axis: -1)
            : max(blockedScores, axis: -1)
        var blockScores = max(intraBlock, axis: 1)
        blockScores = which(blockScores .== blockScores, blockScores, neg)  // NaN guard

        var selectedScores = which(causalBlock, blockScores, neg)

        if initBlocks > 0 {
            let initMask = (blocks.reshaped(1, numBlocks) .< initBlocks) & causalBlock
            selectedScores = which(initMask, Float(1e30), selectedScores)
        }
        if localBlocks > 0 {
            let localStart = maximum(curBlock - (localBlocks - 1), 0)
            let localMask =
                (blocks.reshaped(1, numBlocks) .>= localStart.reshaped(L, 1))
                & (blocks.reshaped(1, numBlocks) .<= curBlock.reshaped(L, 1))
                & causalBlock
            selectedScores = which(localMask, Float(1e29), selectedScores)
        }

        let topk = min(topkBlocks, numBlocks)
        let topkIdx = argPartition(-selectedScores, kth: topk - 1, axis: -1)[.ellipsis, ..<topk]
            .asType(.int32)
        let validBlocks = broadcast(causalBlock, to: [B, L, numBlocks])
        let topkValid = takeAlong(validBlocks, topkIdx, axis: -1)
        let invalid = MLXArray.full(topkIdx.shape, values: MLXArray(numBlocks), dtype: .int32)
        var blockIndices = which(topkValid, topkIdx, invalid)
        let order = argSort(blockIndices, axis: -1)
        blockIndices = takeAlong(blockIndices, order, axis: -1)
        blockIndices = which(blockIndices .== numBlocks, -1, blockIndices)
        return blockIndices
    }

    /// Expands `blockIndices` (per-query selected key blocks) into a
    /// per-token boolean attention mask over `keyLength` keys, ANDed with
    /// per-token causal masking -- mirrors mlx-vlm's
    /// `MiniMaxM3Indexer.build_block_mask` (mask-free path).
    func buildBlockMask(blockIndices: MLXArray, keyLength: Int, qStart: Int) -> MLXArray {
        let (B, L, _) = (blockIndices.dim(0), blockIndices.dim(1), blockIndices.dim(2))
        let numBlocks = (keyLength + blockSize - 1) / blockSize
        let blocks = MLXArray(Int32(0) ..< Int32(numBlocks))
        let blockKeep = (blockIndices[.ellipsis, .newAxis] .== blocks).any(axis: -2)

        let kPositions = MLXArray(Int32(0) ..< Int32(keyLength))
        let keyBlocks = broadcast(
            kPositions.floorDivide(blockSize).reshaped(1, 1, keyLength), to: [B, L, keyLength])
        let keyKeep = takeAlong(blockKeep, keyBlocks, axis: -1)

        let qPositions = MLXArray(Int32(qStart) ..< Int32(qStart + L))
        let causal = kPositions.reshaped(1, keyLength) .<= qPositions.reshaped(L, 1)

        return (keyKeep & causal).reshaped(B, 1, L, keyLength)
    }
}

// MARK: - MiniMaxM3Attention

/// MiniMax-M3's self-attention: GQA (64 query heads / 4 KV heads in the
/// verified config), partial RoPE (only the first `rotaryDim` of `headDim`
/// dims rotate -- `initializeRope` handles this natively since
/// `MLXFast.RoPE`/`RoPE` leave any dims beyond `dimensions` untouched), and
/// per-head Gemma-mode QK-norm applied *after* reshaping to
/// `(B, L, heads, headDim)` (unlike M2's flat, pre-reshape norm over
/// `heads * headDim`). `attention_output_gate` is `false` for M3 -- no output
/// gate module.
///
/// **MiniMax Sparse Attention (MSA), layers 3-59 (kanban ^8dbc476).** Layers
/// where `config.isMoELayer(layerIndex)` build a `MiniMaxM3Indexer` (its own
/// `index_q_proj`/`index_k_proj` projections, per-head RMSNorms, and the
/// attention's shared RoPE instance) that scores and selects the top-
/// `sparseAttention.topkBlocks` key blocks per query -- see
/// `MiniMaxM3Indexer`'s documentation for the selection algorithm and the
/// dense-fallback threshold. Layers 0..<3 have no indexer and always run
/// plain dense causal attention, mirroring the verified real checkpoint's
/// dense/MoE schedule.
final class MiniMaxM3Attention: Module {
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: Gemma.RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: Gemma.RMSNorm?

    @ModuleInfo var rope: RoPELayer

    @ModuleInfo(key: "index_q_proj") var indexQProj: Linear?
    @ModuleInfo(key: "index_k_proj") var indexKProj: Linear?
    @ModuleInfo(key: "index_q_norm") var indexQNorm: Gemma.RMSNorm?
    @ModuleInfo(key: "index_k_norm") var indexKNorm: Gemma.RMSNorm?

    /// The MSA block selector for this layer, or `nil` for dense-only layers
    /// (0..<3). See `MiniMaxM3Indexer`'s documentation for why it borrows
    /// this attention's own submodules rather than owning them itself.
    let indexer: MiniMaxM3Indexer?

    init(_ config: MiniMaxM3TextConfiguration, layerIndex: Int = 0) {
        self.numAttentionHeads = config.attentionHeads
        self.numKeyValueHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(config.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numAttentionHeads * headDim, config.hiddenSize, bias: false)

        if config.useQkNorm {
            _qNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        }

        let rope = initializeRope(
            dims: config.rotaryDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        self.rope = rope

        if config.isMoELayer(layerIndex) {
            let sparse = config.sparseAttention
            let indexQProj = Linear(
                config.hiddenSize, sparse.numIndexHeads * sparse.indexDim, bias: false)
            let indexKProj = Linear(config.hiddenSize, sparse.indexDim, bias: false)
            let indexQNorm = Gemma.RMSNorm(dimensions: sparse.indexDim, eps: config.rmsNormEps)
            let indexKNorm = Gemma.RMSNorm(dimensions: sparse.indexDim, eps: config.rmsNormEps)
            _indexQProj.wrappedValue = indexQProj
            _indexKProj.wrappedValue = indexKProj
            _indexQNorm.wrappedValue = indexQNorm
            _indexKNorm.wrappedValue = indexKNorm
            self.indexer = MiniMaxM3Indexer(
                indexHeads: sparse.numIndexHeads,
                indexDim: sparse.indexDim,
                blockSize: sparse.blockSize,
                topkBlocks: sparse.topkBlocks,
                initBlocks: sparse.initBlocks,
                localBlocks: sparse.localBlocks,
                scoreType: sparse.scoreType,
                scale: scale,
                indexQProj: indexQProj,
                indexKProj: indexKProj,
                indexQNorm: indexQNorm,
                indexKNorm: indexKNorm,
                rope: rope
            )
        } else {
            self.indexer = nil
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        let values = wv(x)

        // Reshape into per-head structure *before* QK-norm: M3's norm is a
        // per-head RMSNorm over `headDim`, applied on (B, L, heads, headDim),
        // not M2's flat norm over the whole `heads * headDim` projection.
        queries = queries.reshaped(B, L, numAttentionHeads, headDim)
        keys = keys.reshaped(B, L, numKeyValueHeads, headDim)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let effectiveMask = resolveMask(x: x, mask: mask, offset: offset, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: v,
            cache: cache,
            scale: scale,
            mask: effectiveMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// Resolves the attention mask for this call: the dense `mask` passed in
    /// by the caller, unchanged, for dense-only layers, a cache that can't
    /// track index keys (e.g. `RotatingKVCache`, when `maxKVSize` is set --
    /// mirrors mlx-vlm's `use_sparse_mask` gate), or the dense fallback; a
    /// block-sparse mask built from the indexer's top-k block selection
    /// otherwise.
    private func resolveMask(
        x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, offset: RoPEOffset?,
        cache: KVCache?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let indexer else { return mask }
        guard cache == nil || cache is MiniMaxM3KVCache else { return mask }

        let sparseCache = cache as? MiniMaxM3KVCache
        let qStart = sparseCache?.indexOffset ?? 0
        guard
            let (blockIndices, totalLen) = indexer(
                x, offset: offset, cache: sparseCache, qStart: qStart)
        else {
            return mask
        }
        return .array(
            indexer.buildBlockMask(blockIndices: blockIndices, keyLength: totalLen, qStart: qStart))
    }
}

// MARK: - MiniMaxM3MLP

/// Dense (non-MoE) MLP used by layers 0..<3 in the verified schedule, sized
/// by `denseIntermediateSize` (12,288) rather than the per-expert
/// `intermediateSize` (3,072) the MoE blocks use. Shares
/// `MiniMaxM3SwiGLUOAI`'s clipped activation with the MoE path.
final class MiniMaxM3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let activation: MiniMaxM3SwiGLUOAI

    init(_ config: MiniMaxM3TextConfiguration) {
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.denseIntermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.denseIntermediateSize, bias: false)
        _downProj.wrappedValue = Linear(
            config.denseIntermediateSize, config.hiddenSize, bias: false)
        self.activation = MiniMaxM3SwiGLUOAI(
            alpha: config.swigluAlpha, limit: config.swigluLimit, beta: config.swigluBeta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(upProj(x), gateProj(x)))
    }
}

// MARK: - MiniMaxM3DecoderLayer

/// A single decoder layer: dense `MiniMaxM3MLP` for layers 0..<3, MoE
/// `MiniMaxM3SparseMoeBlock` from layer 3 on, per `config.moeLayerFreq`.
///
/// Unlike `DeepSeekV3DecoderLayer` (`Libraries/MLXLLM/Models/DeepSeekV3.swift`),
/// which reuses a single `mlp`-keyed property for both variants, M3's two
/// variants are exposed through two separate optional `@ModuleInfo`-wrapped
/// properties (`denseMLP`/`blockSparseMoe`) -- verified directly against the
/// real checkpoint's safetensors index: dense layers ship `mlp.gate_proj/
/// up_proj/down_proj`, while MoE layers ship `block_sparse_moe.gate` +
/// `block_sparse_moe.switch_mlp.*`, genuinely different key prefixes rather
/// than one shared name.
final class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM3SparseMoeBlock?
    @ModuleInfo(key: "mlp") var denseMLP: MiniMaxM3MLP?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    init(_ config: MiniMaxM3TextConfiguration, layerIndex: Int) {
        _selfAttn.wrappedValue = MiniMaxM3Attention(config, layerIndex: layerIndex)

        if config.isMoELayer(layerIndex) {
            _blockSparseMoe.wrappedValue = MiniMaxM3SparseMoeBlock(
                MiniMaxM3MoEConfiguration(
                    hiddenSize: config.hiddenSize,
                    intermediateSize: config.intermediateSize,
                    numLocalExperts: config.numLocalExperts,
                    numExpertsPerTok: config.numExpertsPerTok,
                    routedScalingFactor: config.routedScalingFactor,
                    swigluAlpha: config.swigluAlpha,
                    swigluLimit: config.swigluLimit,
                    swigluBeta: config.swigluBeta
                ))
        } else {
            _denseMLP.wrappedValue = MiniMaxM3MLP(config)
        }

        _inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let mlpOutput: MLXArray
        if let blockSparseMoe {
            mlpOutput = blockSparseMoe(postAttentionLayerNorm(h))
        } else if let denseMLP {
            mlpOutput = denseMLP(postAttentionLayerNorm(h))
        } else {
            // Unreachable: `init` always sets exactly one of `blockSparseMoe`
            // / `denseMLP` based on `config.isMoELayer(layerIndex)`.
            fatalError(
                "MiniMaxM3DecoderLayer has neither blockSparseMoe nor denseMLP set -- invariant violated by init"
            )
        }
        return h + mlpOutput
    }
}

// MARK: - MiniMaxM3ModelInner

final class MiniMaxM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma.RMSNorm

    init(_ config: MiniMaxM3TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            MiniMaxM3DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Runs the decoder stack, embedding `inputs` via `embedTokens` unless
    /// `inputEmbeddings` is supplied -- vision-augmented prefill (kanban
    /// ^9a2aw98) passes already-spliced image embeddings here instead of
    /// letting this method derive them from raw token ids.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - MiniMaxM3LanguageModel

/// The `language_model.*`-keyed submodule: `model` (embeddings, decoder
/// layers, final norm) + `lm_head`. Wrapped by `MiniMaxM3Model` under the
/// `language_model` module key -- see that type's documentation for why the
/// prefix must be preserved.
final class MiniMaxM3LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") var model: MiniMaxM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: MiniMaxM3TextConfiguration) {
        // `useGemmaNorm`/`qkNormType` are not wired to any branch below --
        // every norm unconditionally uses Gemma-mode RMSNorm, and QK-norm
        // (when enabled) is always per-head. Both hold for the one verified
        // real checkpoint; unsupported values are rejected earlier, at decode
        // time, by `MiniMaxM3TextConfiguration.init(from:)` -- see its
        // `use_gemma_norm`/`qk_norm_type` validation -- since those fields are
        // external, checkpoint-supplied data and must fail with a typed
        // decoding error rather than crash a running process (flagged by
        // adversarial review of kanban ^xgvth41).
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = MiniMaxM3ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    /// Runs the language model, forwarding `inputEmbeddings` through to
    /// `MiniMaxM3ModelInner` unchanged -- see that method's documentation.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }
}

// MARK: - MiniMaxM3Model

/// MiniMax-M3's language model: dense attention on layers 0..<3, MSA
/// (MiniMax Sparse Attention) on layers 3-59 -- see `MiniMaxM3Attention` and
/// `MiniMaxM3Indexer` (kanban ^8dbc476). Registered with the VLM factory
/// under both `"minimax_m3_vl"` (VL-nested) and `"minimax_m3"` (flat) model
/// types (kanban ^wz8y8qq) and conforms to `VLMModel` via the
/// `prepare(_:cache:state:windowSize:)` implementation below -- but there is
/// still no vision tower here: `MiniMaxM3Processor` rejects image/video
/// input with a descriptive error, and real vision support (a vision tower
/// plus image-aware `prepare`) is a later task. This type was originally the
/// deliverable for kanban ^xgvth41: a model that compiles and runs
/// tiny-config forward passes with a KV cache.
///
/// **Preserves the `language_model.` module-path prefix.** The published VL
/// checkpoint's weights *and* its per-module quantization overrides are keyed
/// `language_model.model.layers.N...` (e.g. one MoE gate ships at 8-bit while
/// the checkpoint default is 4-bit) -- `loadWeights` resolves quantization by
/// module path (`PerLayerQuantization.quantization(layer:)`), so re-keying
/// weights away from this prefix would silently fall back to the wrong
/// quantization and fail the shape check at load. Follows the `Qwen35.swift`
/// `@ModuleInfo(key: "language_model")` wrapper precedent.
public final class MiniMaxM3Model: Module, BaseLanguageModel, KVCacheDimensionProvider {
    /// Module-path prefix applied to flat (`minimax_m3`) checkpoint keys and
    /// checked against already-prefixed (`minimax_m3_vl`) ones -- see
    /// `_filterUnusedWeights` and `_remapExpertWeights`.
    private static let languageModelPrefix = "language_model."

    @ModuleInfo(key: "language_model") var languageModel: MiniMaxM3LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: MiniMaxM3Vision.VisionModel?
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: MiniMaxM3Projector?
    @ModuleInfo(key: "patch_merge_mlp") var patchMergeMlp: MiniMaxM3Projector?

    /// The text-model configuration this instance was constructed from.
    public let configuration: MiniMaxM3TextConfiguration
    /// The vision-tower configuration, or `nil` for a text-only (`minimax_m3`
    /// flat, or VL-nested but vision-less) instance.
    public let visionConfiguration: MiniMaxM3VisionConfiguration?
    /// Token id marking an image's spliced-embedding positions
    /// (`defaultImageTokenIndex` for a text-only instance, unused).
    public let imageTokenIndex: Int
    /// Token id marking a video's spliced-embedding positions. See `imageTokenIndex`.
    public let videoTokenIndex: Int
    /// Per-layer key/value head counts, forwarded from the language model.
    public var kvHeads: [Int] { languageModel.kvHeads }
    /// Vocabulary size, forwarded from `configuration`.
    public var vocabularySize: Int { configuration.vocabularySize }

    /// Creates a dense-attention, text-only MiniMax-M3 language model (no
    /// vision tower) from its text configuration -- used by the flat
    /// (`minimax_m3`) registration.
    public init(_ config: MiniMaxM3TextConfiguration) {
        self.configuration = config
        self.visionConfiguration = nil
        self.imageTokenIndex = defaultImageTokenIndex
        self.videoTokenIndex = defaultVideoTokenIndex
        _languageModel.wrappedValue = MiniMaxM3LanguageModel(config)
        super.init()
    }

    /// Creates a vision-capable MiniMax-M3 model from its full top-level
    /// configuration -- used by the VL-nested (`minimax_m3_vl`) registration
    /// (kanban ^9a2aw98).
    public init(_ config: MiniMaxM3Configuration) {
        self.configuration = config.textConfiguration
        self.visionConfiguration = config.visionConfiguration
        self.imageTokenIndex = config.imageTokenIndex
        self.videoTokenIndex = config.videoTokenIndex
        _languageModel.wrappedValue = MiniMaxM3LanguageModel(config.textConfiguration)
        _visionTower.wrappedValue = MiniMaxM3Vision.VisionModel(config.visionConfiguration)
        _multiModalProjector.wrappedValue = MiniMaxM3Projector(
            inputDimensions: config.visionConfiguration.hiddenSize,
            hiddenDimensions: config.projectorHiddenSize,
            outputDimensions: config.textConfiguration.hiddenSize,
            bias: config.multimodalProjectorBias,
            hiddenAct: config.projectorHiddenAct)
        let mergeSize = config.visionConfiguration.spatialMergeSize
        _patchMergeMlp.wrappedValue = MiniMaxM3Projector(
            inputDimensions: config.textConfiguration.hiddenSize * mergeSize * mergeSize,
            hiddenDimensions: config.textConfiguration.hiddenSize,
            outputDimensions: config.textConfiguration.hiddenSize,
            bias: config.patchMergeBias,
            hiddenAct: config.projectorHiddenAct)
        super.init()
    }

    /// Runs the language model over `inputs`, reading from and updating `cache` in place if provided.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Runs the language model over already-computed input embeddings, used
    /// for vision-augmented prefill where image features have already been
    /// spliced into the embedding sequence -- see
    /// `prepare(_:cache:state:windowSize:)`.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray
    ) -> MLXArray {
        languageModel(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
    }

    /// Runs the vision tower and `multi_modal_projector`, then merges
    /// adjacent `spatialMergeSize x spatialMergeSize` projected patches into
    /// single tokens via `patch_merge_mlp` -- mirrors mlx-vlm's
    /// `Model._compute_visual_features`.
    private func computeVisualFeatures(pixelValues: MLXArray, gridTHW: [THW]) -> MLXArray {
        guard let visionTower, let multiModalProjector else {
            fatalError("computeVisualFeatures called on a MiniMaxM3Model with no vision tower")
        }
        let hidden = visionTower(pixelValues.asType(visionTower.inputDtype), gridTHW: gridTHW)
        return mergeVisualTokens(multiModalProjector(hidden), gridTHW: gridTHW)
    }

    /// Reduces `features` (one row per un-merged patch, concatenated across
    /// every grid in `gridTHW` in order) into one row per
    /// `spatialMergeSize x spatialMergeSize` block via `patch_merge_mlp` --
    /// mirrors mlx-vlm's `Model._merge_visual_tokens`. Patch order within
    /// each grid already groups each merge block contiguously (see
    /// `MiniMaxM3Vision.rotaryPositionEmbedding`'s documentation), so no
    /// transpose is needed before reshaping.
    private func mergeVisualTokens(_ features: MLXArray, gridTHW: [THW]) -> MLXArray {
        guard let patchMergeMlp, let visionConfiguration else {
            fatalError("mergeVisualTokens called on a MiniMaxM3Model with no vision tower")
        }
        let mergeSize = visionConfiguration.spatialMergeSize
        let featureDim = features.dim(-1)
        var outputs: [MLXArray] = []
        var offset = 0
        for grid in gridTHW {
            let length = grid.product
            let block = features[offset ..< offset + length, 0...]
            offset += length
            let reshaped = block.reshaped(
                grid.t, grid.h / mergeSize, grid.w / mergeSize, mergeSize, mergeSize, featureDim
            ).reshaped(-1, mergeSize * mergeSize * featureDim)
            outputs.append(patchMergeMlp(reshaped))
        }
        return concatenated(outputs, axis: 0)
    }

    /// `MiniMaxM3KVCache` per MSA layer (`configuration.isMoELayer`, layers
    /// 3-59 in the verified schedule), `KVCacheSimple` per dense layer
    /// (0..<3) -- or `RotatingKVCache` for every layer when
    /// `parameters.maxKVSize` is set, matching the pre-MSA behavior for that
    /// path (a sliding-window cache doesn't track index keys, so
    /// `MiniMaxM3Attention.resolveMask` treats it like "no cache" support and
    /// falls back to dense masking for those layers -- see its doc comment).
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        let numLayers = kvHeads.count
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
        }
        return (0 ..< numLayers).map { layerIndex -> KVCache in
            configuration.isMoELayer(layerIndex) ? MiniMaxM3KVCache() : KVCacheSimple()
        }
    }

    /// Sanitize and prepare weight dictionary for model loading: dequantize
    /// block-quantized (fp8/bf16 `weight_scale_inv`) weights, re-key flat
    /// `minimax_m3` checkpoints into the `language_model.`-prefixed layout,
    /// drop unused vision-tower/multi-modal-projector/MTP weights, and remap
    /// per-expert fallback weights (`w1`/`w2`/`w3`) into the fused
    /// `gate_up_proj`/`down_proj` layout `FusedGateUpSwitchGLU` expects. The
    /// three phases are independent transformations, applied in order -- see
    /// `_mergeQuantized`, `_filterUnusedWeights`, `_remapExpertWeights`, and
    /// `_remapVisionWeights`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let merged = _mergeQuantized(weights)
        let filtered = _filterUnusedWeights(merged)
        let experts = _remapExpertWeights(filtered)
        return _remapVisionWeights(experts)
    }

    /// `sanitize` step 1: merge fp8/bf16 block-quantized `weight_scale_inv`
    /// pairs (M2's dequant path, for bf16/fp8-original checkpoints) into
    /// their dequantized weight.
    private func _mergeQuantized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let dtype = weight.dtype
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var merged: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    merged[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if merged[key] == nil {
                merged[key] = value
            }
        }
        return merged
    }

    /// `sanitize` step 2: drop MTP weights (out of scope for this port), keep
    /// vision-tower / multi-modal-projector / patch-merge weights as-is (they
    /// are already top-level sibling keys matching this model's own module
    /// tree -- see `MiniMaxM3Model.init(_ config: MiniMaxM3Configuration)` --
    /// so unlike the language-model weights below, they need no
    /// `language_model.` prefixing), re-key flat (`minimax_m3`) checkpoints
    /// into the `language_model.`-prefixed layout the VL checkpoint already
    /// uses, and drop the tied `lm_head` weight when
    /// `configuration.tieWordEmbeddings` is set. The sparse-attention indexer
    /// weights (`self_attn.index_q_proj`/`index_k_proj`/`index_q_norm`/
    /// `index_k_norm`, confirmed present on every MoE-scheduled layer in the
    /// real checkpoint's safetensors index) are no longer stripped here --
    /// `MiniMaxM3Indexer` (kanban ^8dbc476) now builds and loads them like
    /// any other layer weight.
    ///
    /// Un-drops the vision/projector keys this method used to discard
    /// (kanban ^9a2aw98): a prior task (^wz8y8qq) verified they were absent
    /// from the one real checkpoint downloaded so far, but a model built with
    /// vision support (`MiniMaxM3Model.init(_ config: MiniMaxM3Configuration)`)
    /// now has real submodules to load them into.
    private func _filterUnusedWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var prefixed: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("mtp.") || key.contains(".mtp.") {
                continue
            }
            if key.hasPrefix("vision_tower.") || key.hasPrefix("multi_modal_projector.")
                || key.hasPrefix("patch_merge_mlp.")
            {
                prefixed[key] = value
                continue
            }

            let newKey =
                key.hasPrefix(Self.languageModelPrefix) ? key : Self.languageModelPrefix + key
            prefixed[newKey] = value
        }

        if configuration.tieWordEmbeddings {
            prefixed["\(Self.languageModelPrefix)lm_head.weight"] = nil
        }

        return prefixed
    }

    /// `sanitize` step 3: fallback remap for unconverted per-expert
    /// checkpoints (`w1`/`w2`/`w3`, M2's layout) into
    /// `FusedGateUpSwitchGLU`'s fused `gate_up_proj`/`down_proj` layout
    /// (gate = first half, up = second half of the fused column dimension,
    /// matching `MiniMaxM3SparseMoeBlock`'s split order). The published
    /// mlx-community checkpoint already ships fused, pre-stacked expert
    /// weights and never reaches this loop's body (the guard fails
    /// immediately). This fallback stacks only the `numLocalExperts`
    /// *routed* experts; it does not attempt to synthesize the packed
    /// shared-expert row, since no unconverted-per-expert M3 checkpoint
    /// exists yet to verify that layout against.
    private func _remapExpertWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        /// Per-expert gate-projection weight type (`w1`), used both to probe
        /// for an unconverted per-expert checkpoint and to collect its
        /// tensors below -- kept as a single constant so the probe and the
        /// collection can never drift to different key paths.
        let gateWeightType = "w1"
        var prefixed = weights

        /// Removes and collects the per-expert `weightType` tensors for
        /// `key` (e.g. `"weight"`/`"scales"`/`"biases"`) across all
        /// `configuration.numLocalExperts` routed experts under `prefix`, in
        /// expert-index order -- the shared shape for gate (`w1`), up
        /// (`w3`), and down (`w2`) projection collection below.
        func collectExpertWeights(_ key: String, weightType: String, prefix: String) -> [MLXArray] {
            (0 ..< configuration.numLocalExperts).map {
                prefixed.removeValue(forKey: "\(prefix).experts.\($0).\(weightType).\(key)")!
            }
        }

        for layerIndex in 0 ..< configuration.hiddenLayers
        where configuration.isMoELayer(layerIndex) {
            let prefix = "\(Self.languageModelPrefix)model.layers.\(layerIndex).block_sparse_moe"
            guard prefixed["\(prefix).experts.0.\(gateWeightType).weight"] != nil else { continue }

            for key in ["weight", "scales", "biases"] {
                guard prefixed["\(prefix).experts.0.\(gateWeightType).\(key)"] != nil else {
                    continue
                }

                let gate = collectExpertWeights(key, weightType: gateWeightType, prefix: prefix)
                let up = collectExpertWeights(key, weightType: "w3", prefix: prefix)
                let down = collectExpertWeights(key, weightType: "w2", prefix: prefix)

                prefixed["\(prefix).switch_mlp.gate_up_proj.\(key)"] =
                    MLX.concatenated([MLX.stacked(gate), MLX.stacked(up)], axis: 1)
                prefixed["\(prefix).switch_mlp.down_proj.\(key)"] = MLX.stacked(down)
            }
        }
        return prefixed
    }

    /// `sanitize` step 4: flattens the vision patch-embedding weight from its
    /// checkpoint Conv3d layout (`hidden, channels, temporalPatch, patch,
    /// patch`) into the 2D `Linear` layout `MiniMaxM3Vision.Embeddings`
    /// expects (`hidden, channels*temporalPatch*patch*patch`) -- see that
    /// type's documentation. A no-op when the key is absent (text-only
    /// instance) or already 2D (e.g. a pre-flattened conversion).
    private func _remapVisionWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let patchEmbeddingKey = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        guard let weight = weights[patchEmbeddingKey], weight.ndim == 5 else { return weights }

        var remapped = weights
        let hiddenSize = weight.dim(0)
        let patchDimensions = weight.dim(1) * weight.dim(2) * weight.dim(3) * weight.dim(4)
        remapped[patchEmbeddingKey] = weight.reshaped(hiddenSize, patchDimensions)
        return remapped
    }
}

/// LoRA support for MiniMax-M3: exposes the dense-attention decoder layers as
/// adaptation targets, matching the `LoRAModel` conformance pattern used by
/// the other decoder-only models in this package.
extension MiniMaxM3Model: LoRAModel {
    /// The decoder layers available for LoRA parameter adaptation.
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - MiniMaxM3Model + VLMModel

/// `VLMModel` conformance for MiniMax-M3 (kanban ^9a2aw98). A text-only
/// instance (`visionTower == nil`, e.g. the flat `minimax_m3` registration)
/// still only ever sees `input.image == nil` -- `MiniMaxM3Processor` only
/// attaches an image when the model it will run against might support one --
/// but `prepare` throws a descriptive error rather than crashing if that
/// invariant is ever violated.
extension MiniMaxM3Model: VLMModel {
    /// Prefills `input`'s tokens and returns either the remainder for the
    /// `TokenIterator` to consume one token at a time (text-only prompts,
    /// `windowSize`-chunked) or already-computed logits (image prompts,
    /// prefilled in one shot since vision-augmented input embeddings can't be
    /// re-derived from a token-id chunk alone) -- mirrors `Qwen3VL.prepare`'s
    /// split between the two paths (`Qwen3VL.swift`).
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws -> PrepareResult {
        guard let image = input.image else {
            let prefillStepSize = windowSize ?? 512
            var y = input.text

            try withPreparedCache(cache, lengths: y.sequenceLengths) {
                var state: LMOutput.State? = state
                while y.tokens.size > prefillStepSize {
                    try Task.checkCancellation()
                    let chunk = y[.newAxis, ..<prefillStepSize]
                    let output = self(chunk, cache: cache.isEmpty ? nil : cache, state: state)
                    state = output.state
                    asyncEval(cache)
                    y = y[prefillStepSize...]
                }
                eval(cache)
            }

            return .tokens(y)
        }

        guard visionTower != nil, multiModalProjector != nil, patchMergeMlp != nil else {
            throw VLMError.mediaNotSupported("image")
        }

        let inputIds = input.text.tokens
        let textEmbeds = languageModel.model.embedTokens(inputIds)
        let visualFeatures = computeVisualFeatures(
            pixelValues: image.pixels, gridTHW: image.frames ?? []
        ).asType(textEmbeds.dtype)
        let mergedEmbeds = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: textEmbeds, imageFeatures: visualFeatures,
            imageTokenId: imageTokenIndex, videoTokenId: videoTokenIndex)

        var logits = MLXArray(0)
        withPreparedCache(cache, lengths: [inputIds.size]) {
            logits = self(inputIds, cache: cache.isEmpty ? nil : cache, inputEmbeddings: mergedEmbeds)
            eval(cache)
        }

        return .logits(LMOutput(logits: logits))
    }
}

// MARK: - MiniMaxM3ProcessorConfiguration + MiniMaxM3Processor

/// Declared processor class name (`processor_class`) MiniMax-M3's checkpoint
/// declares, used when the config the checkpoint ships omits the key
/// entirely. Shared by `MiniMaxM3ProcessorConfiguration`'s memberwise `init`
/// and its `init(from:)` decoder fallback.
public let defaultMiniMaxM3ProcessorClass = "MiniMaxM3VLProcessor"

/// Default `image_mean` (`preprocessor_config.json`'s `image_processor.image_mean`,
/// the standard OpenAI CLIP normalization mean), used when the checkpoint's
/// config omits the `image_processor` object entirely. Shared by
/// `MiniMaxM3ProcessorConfiguration.ImageProcessorSettings`'s memberwise
/// `init` and its `init(from:)` decoder fallback.
public let defaultMiniMaxM3ImageMean: [CGFloat] = [0.481_454_66, 0.457_827_5, 0.408_210_73]

/// Default `image_std` (OpenAI CLIP normalization standard deviation). See `defaultMiniMaxM3ImageMean`.
public let defaultMiniMaxM3ImageStd: [CGFloat] = [0.268_629_54, 0.261_302_58, 0.275_777_11]

/// Default minimum resize pixel budget (`image_processor.min_pixels`), verified
/// against the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `preprocessor_config.json`. See `defaultMiniMaxM3ImageMean`.
public let defaultMiniMaxM3MinPixels = 3_136

/// Default maximum resize pixel budget (`image_processor.max_pixels`). See `defaultMiniMaxM3ImageMean`.
public let defaultMiniMaxM3MaxPixels = 451_584

/// Configuration for `MiniMaxM3Processor`.
///
/// `MiniMaxM3ProcessorConfiguration` is decoded from whichever file
/// `loadProcessorConfig` selects. The real `mlx-community/MiniMax-M3-4bit`
/// checkpoint's `preprocessor_config.json` omits `processor_class` entirely,
/// so `loadProcessorConfig` falls back to `processor_config.json` -- a
/// *composite* document nesting the image-processor fields under an
/// `image_processor` key rather than at the JSON root (unlike, e.g.,
/// `Qwen3VLProcessorConfiguration`, which decodes a flat
/// `preprocessor_config.json` directly). `init(from:)` supports both shapes,
/// nested-first, falling back to the decoder's own root container when
/// `image_processor` is absent -- mirrors
/// `MiniMaxM3TextConfiguration.init(from:)`'s VL-nested/flat fallback pattern.
public struct MiniMaxM3ProcessorConfiguration: Codable, Sendable {
    /// The image-preprocessing fields consumed by `MiniMaxM3Processor`,
    /// mirroring mlx-vlm's `MiniMaxM3VLImageProcessor` defaults.
    public struct ImageProcessorSettings: Codable, Sendable {
        /// Per-channel normalization mean (`image_mean`).
        public let imageMean: [CGFloat]
        /// Per-channel normalization standard deviation (`image_std`).
        public let imageStd: [CGFloat]
        /// Minimum resize pixel budget (`min_pixels`).
        public let minPixels: Int
        /// Maximum resize pixel budget (`max_pixels`).
        public let maxPixels: Int
        /// Patch size, in pixels (`patch_size`).
        public let patchSize: Int
        /// Temporal patch size (`temporal_patch_size`).
        public let temporalPatchSize: Int
        /// Spatial patch-merge factor (`merge_size`).
        public let mergeSize: Int

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case minPixels = "min_pixels"
            case maxPixels = "max_pixels"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
        }

        /// Creates image-processor settings directly from field values,
        /// defaulting each parameter to the value verified against the
        /// `mlx-community/MiniMax-M3-4bit` checkpoint's `preprocessor_config.json`.
        public init(
            imageMean: [CGFloat] = defaultMiniMaxM3ImageMean,
            imageStd: [CGFloat] = defaultMiniMaxM3ImageStd,
            minPixels: Int = defaultMiniMaxM3MinPixels,
            maxPixels: Int = defaultMiniMaxM3MaxPixels,
            patchSize: Int = defaultVisionPatchSize,
            temporalPatchSize: Int = defaultVisionTemporalPatchSize,
            mergeSize: Int = defaultVisionSpatialMergeSize
        ) {
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.minPixels = minPixels
            self.maxPixels = maxPixels
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.mergeSize = mergeSize
        }

        /// Decodes image-processor settings, defaulting any field the
        /// checkpoint omits to the value verified against the
        /// `mlx-community/MiniMax-M3-4bit` checkpoint's `preprocessor_config.json`.
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            imageMean =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageMean)
                ?? defaultMiniMaxM3ImageMean
            imageStd =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageStd)
                ?? defaultMiniMaxM3ImageStd
            minPixels =
                try container.decodeIfPresent(Int.self, forKey: .minPixels)
                ?? defaultMiniMaxM3MinPixels
            maxPixels =
                try container.decodeIfPresent(Int.self, forKey: .maxPixels)
                ?? defaultMiniMaxM3MaxPixels
            patchSize =
                try container.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? defaultVisionPatchSize
            temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
                ?? defaultVisionTemporalPatchSize
            mergeSize =
                try container.decodeIfPresent(Int.self, forKey: .mergeSize)
                ?? defaultVisionSpatialMergeSize
        }
    }

    /// Declared processor class name (`processor_class`), verified as
    /// `"MiniMaxM3VLProcessor"` against the checkpoint's `processor_config.json`.
    public let processorClass: String
    /// The image-preprocessing settings, nested under `image_processor` in
    /// the real checkpoint's `processor_config.json` (see this type's
    /// documentation), or decoded flat when that key is absent.
    public let imageProcessor: ImageProcessorSettings

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
    }

    /// Creates a processor configuration directly from its declared class
    /// name and image-processor settings.
    public init(
        processorClass: String = defaultMiniMaxM3ProcessorClass,
        imageProcessor: ImageProcessorSettings = ImageProcessorSettings()
    ) {
        self.processorClass = processorClass
        self.imageProcessor = imageProcessor
    }

    /// Decodes a processor configuration, defaulting `processorClass` when
    /// the checkpoint's config omits it, and decoding `imageProcessor` from a
    /// nested `image_processor` object when present or from the decoder's
    /// own root container otherwise (see this type's documentation).
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try container.decodeIfPresent(String.self, forKey: .processorClass)
            ?? defaultMiniMaxM3ProcessorClass
        if let nested = try container.decodeIfPresent(
            ImageProcessorSettings.self, forKey: .imageProcessor)
        {
            imageProcessor = nested
        } else {
            imageProcessor = try ImageProcessorSettings(from: decoder)
        }
    }
}

/// `MessageGenerator` for MiniMax-M3's chat template, used only when
/// `UserInput` carries at least one image (kanban ^9a2aw98).
///
/// MiniMax-M3's chat template (`processor_config.json`'s `chat_template`,
/// `visible_text` macro) renders each `{"type": "image", ...}` content block
/// as the literal `]<]image[>[` placeholder text (`MiniMaxM3Processor.imageToken`),
/// which `MiniMaxM3Processor.prepare(input:)` then expands into one repeated
/// copy per merged image token, wrapped in `]<]start of image[>[`/`]<]end of
/// image[>[` markup -- mirrors `Qwen3VLMessageGenerator`'s equivalent role
/// for Qwen's own `<|vision_start|>`/`<|image_pad|>`/`<|vision_end|>` markup
/// (`Qwen3VL.swift`). Text-only prompts keep using `DefaultMessageGenerator`
/// (plain string `content`, unaffected by this type) so ^wz8y8qq's verified
/// text-only tokenization stays byte-for-byte unchanged.
public struct MiniMaxM3MessageGenerator: MessageGenerator {
    /// Creates a message generator for MiniMax-M3's chat template.
    public init() {}

    /// Converts a chat message to MiniMax-M3's structured content-block
    /// format, expanding each attached image into an `{"type": "image"}`
    /// placeholder block ahead of the message's text content.
    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        let imageContent = message.images.map { _ in ["type": "image"] }
        let textContent = [["type": "text", "text": message.content]]
        var dictionary: MLXLMCommon.Message = [
            "role": message.role.rawValue,
            "content": imageContent + textContent,
        ]
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

/// Input processor for MiniMax-M3: text and image input, ported from
/// mlx-vlm's `processing_minimax_m3_vl.py`.
///
/// **Image path.** Resizes to a patch/merge-factor-aligned grid within the
/// checkpoint's min/max pixel budget (`QwenVL.targetSize`, the same
/// smart-resize algorithm as mlx-vlm's `smart_resize` -- both round to
/// `patchSize * mergeSize` multiples and rescale to fit a pixel-count
/// budget), patchifies via `QwenVL.patchify` (identical patch/merge-block
/// ordering to mlx-vlm's `_patchify`), and expands the chat template's single
/// `]<]image[>[` placeholder per image into `frame.product / mergeSize^2`
/// repeated copies wrapped in `]<]start of image[>[`/`]<]end of image[>[`
/// markup (mirrors `QwenVL.replacePaddingTokens`, adapted to MiniMax-M3's own
/// literal placeholder/markup strings instead of Qwen's
/// `<|vision_start|>`/`<|image_pad|>`/`<|vision_end|>`). `image_grid_pinpoints`
/// (`config.json`, spanning 336-2016px) is checkpoint metadata this
/// smart-resize-based processor never reads -- confirmed absent from
/// mlx-vlm's `processing_minimax_m3_vl.py`, which only consults
/// `min_pixels`/`max_pixels`.
///
/// **Video path remains a throwing stub.** Half of this repository's VLM
/// processors have no video support at all (`FastVLM`, `Idefics3`,
/// `LFM2VL`, `Mistral3`, `Paligemma`, `Pixtral`) -- matching that prevailing
/// capability level, `prepare(input:)` throws `VLMError.mediaNotSupported("video")`
/// for video input rather than silently dropping it.
public struct MiniMaxM3Processor: UserInputProcessor {
    /// MiniMax-M3's chat template renders each image reference as this
    /// literal placeholder string before `prepare(input:)` expands it --
    /// mlx-vlm's `MiniMaxM3VLProcessor.IMAGE_TOKEN`.
    static let imageToken = "]<]image[>["
    /// Markup MiniMax-M3's processor wraps each image's expanded placeholder
    /// run in -- mlx-vlm's `MiniMaxM3VLProcessor.VISION_START_TOKEN`.
    static let visionStartToken = "]<]start of image[>["
    /// See `visionStartToken` -- mlx-vlm's `MiniMaxM3VLProcessor.VISION_END_TOKEN`.
    static let visionEndToken = "]<]end of image[>["

    private let configuration: MiniMaxM3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    /// Creates a MiniMax-M3 input processor.
    public init(_ configuration: MiniMaxM3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    /// Resizes, normalizes, and patchifies one image, following the same
    /// sRGB-tone-curve-then-resample-then-normalize pipeline
    /// `Qwen3VLProcessor.preprocess(images:processing:)` uses (`Qwen3VL.swift`)
    /// -- see that method's documentation for why the tone-curve step must
    /// precede resampling.
    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let settings = configuration.imageProcessor
        let processed = MediaProcessing.apply(image, processing: processing)
        let extent = processed.extent.size
        let factor = settings.patchSize * settings.mergeSize

        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(extent.height), width: Int(extent.width), factor: factor,
            minPixels: settings.minPixels, maxPixels: settings.maxPixels)
        let targetSize = CGSize(width: resizedWidth, height: resizedHeight)

        let resampled = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(processed), to: targetSize)
        let normalized = MediaProcessing.asMLXArray(
            MediaProcessing.normalize(
                resampled,
                mean: (settings.imageMean[0], settings.imageMean[1], settings.imageMean[2]),
                std: (settings.imageStd[0], settings.imageStd[1], settings.imageStd[2])))

        return try QwenVL.patchify(
            images: [normalized], mergeSize: settings.mergeSize, patchSize: settings.patchSize,
            temporalPatchSize: settings.temporalPatchSize)
    }

    /// Expands each image's single `imageToken` placeholder occurrence into
    /// `frame.product / mergeSize^2` repeated copies wrapped in
    /// `visionStartToken`/`visionEndToken` -- mirrors mlx-vlm's
    /// `MiniMaxM3VLProcessor.replace_image_token` and this repository's own
    /// `QwenVL.replacePaddingTokens` (`QwenVL.swift`).
    ///
    /// - Throws: `VLMError.processing` if the number of placeholder
    ///   occurrences doesn't match the number of images.
    private func expandImagePlaceholders(in promptTokens: [Int], frames: [THW]) throws -> [Int] {
        let placeholderTokens = tokenizer.encode(text: Self.imageToken)
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of image placeholder tokens does not match number of images")
        }

        let mergeLength = configuration.imageProcessor.mergeSize * configuration.imageProcessor.mergeSize
        let replacementSequences = frames.map { frame -> [Int] in
            let count = frame.product / mergeLength
            let text =
                Self.visionStartToken + Array(repeating: Self.imageToken, count: count).joined()
                + Self.visionEndToken
            return tokenizer.encode(text: text)
        }

        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    /// Converts `input` into an `LMInput`, throwing when `input` carries a
    /// video attachment (see this type's documentation).
    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.mediaNotSupported("video")
        }

        guard !input.images.isEmpty else {
            // Exact pre-existing text-only path (^wz8y8qq) -- untouched, so
            // its verified tokenization stays byte-for-byte identical.
            let messages = DefaultMessageGenerator().generate(from: input)
            do {
                let promptTokens = try tokenizer.applyChatTemplate(
                    messages: messages, tools: input.tools,
                    additionalContext: input.additionalContext)
                return LMInput(tokens: MLXArray(promptTokens))
            } catch TokenizerError.missingChatTemplate {
                let prompt =
                    messages
                    .compactMap { $0["content"] as? String }
                    .joined(separator: "\n\n")
                return LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
            }
        }

        let messages = MiniMaxM3MessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        let processedImages = try input.images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let pixels = concatenated(processedImages.map { $0.0 })
        let frames = processedImages.map { $0.1 }

        let splicedTokens = try expandImagePlaceholders(in: promptTokens, frames: frames)
        let promptArray = MLXArray(splicedTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: pixels, frames: frames))
    }
}
