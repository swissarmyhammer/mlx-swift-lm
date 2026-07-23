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

/// Default checkpoint model type (`model_type`), verified against the
/// `mlx-community/MiniMax-M3-4bit` checkpoint's `config.json`. Shared by
/// `MiniMaxM3TextConfiguration`'s memberwise `init` and its `init(from:)`
/// decoder fallback.
public let defaultModelType = "minimax_m3"

/// Default decoder-layer count (`num_hidden_layers`). See `defaultModelType`.
public let defaultHiddenLayers = 60

/// Default key/value head count for grouped-query attention
/// (`num_key_value_heads`). See `defaultModelType`.
public let defaultKvHeads = 4

/// Default vocabulary size (`vocab_size`). See `defaultModelType`.
public let defaultVocabularySize = 200_064

/// Default maximum sequence length supported by RoPE
/// (`max_position_embeddings`). See `defaultModelType`.
public let defaultMaxPositionEmbeddings = 1_048_576

/// Default RoPE base frequency (`rope_theta`). See `defaultModelType`.
public let defaultRopeTheta: Float = 5_000_000

/// Default MLP activation function name (`hidden_act`). See `defaultModelType`.
public let defaultHiddenAct = "swigluoai"

/// Default QK-norm layout (`qk_norm_type`). Also referenced by
/// `MiniMaxM3LanguageModel`'s `init` precondition, which fails fast if a
/// checkpoint ever requests a different layout. See `defaultModelType`.
public let defaultQkNormType = "per_head"

/// Default dense (non-MoE) MLP intermediate size (`dense_intermediate_size`). See `defaultModelType`.
public let defaultDenseIntermediateSize = 12288

/// Default number of routed experts selected per token (`num_experts_per_tok`). See `defaultModelType`.
public let defaultNumExpertsPerTok = 4

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

// MARK: - MiniMaxM3SparseAttentionConfiguration

/// MiniMax-M3's DSA (sparse attention indexer) hyperparameters, decoded from
/// `text_config.sparse_attention_config`.
///
/// Only the four fields the future indexer (kanban ^8dbc476) will need are
/// modeled here; the real `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `config.json` nests several additional fields alongside these
/// (`use_sparse_attention`, `sparse_disable_index_value`,
/// `sparse_score_type`, `sparse_init_block`, `sparse_local_block`,
/// `sparse_attention_freq`) -- verified directly from the downloaded config,
/// which contradicts an earlier (incorrect) assumption that these were flat
/// top-level keys. They decode as ordinary unknown keys (ignored) since this
/// dense-only task does not build the indexer.
public struct MiniMaxM3SparseAttentionConfiguration: Codable, Sendable {
    /// Per-index-head dimension used by the DSA indexer (`sparse_index_dim`).
    public var indexDim: Int
    /// Number of index heads used by the DSA indexer (`sparse_num_index_heads`).
    public var numIndexHeads: Int
    /// Number of top-scoring blocks the indexer selects per query (`sparse_topk_blocks`).
    public var topkBlocks: Int
    /// Block size, in tokens, used by the DSA indexer (`sparse_block_size`).
    public var blockSize: Int

    enum CodingKeys: String, CodingKey {
        case indexDim = "sparse_index_dim"
        case numIndexHeads = "sparse_num_index_heads"
        case topkBlocks = "sparse_topk_blocks"
        case blockSize = "sparse_block_size"
    }

    /// Creates a sparse-attention (DSA indexer) configuration, defaulting to
    /// the values verified against the `mlx-community/MiniMax-M3-4bit`
    /// checkpoint's `config.json`.
    public init(
        indexDim: Int = defaultIndexDim,
        numIndexHeads: Int = defaultNumIndexHeads,
        topkBlocks: Int = defaultTopkBlocks,
        blockSize: Int = defaultBlockSize
    ) {
        self.indexDim = indexDim
        self.numIndexHeads = numIndexHeads
        self.topkBlocks = topkBlocks
        self.blockSize = blockSize
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
        hiddenSize: Int = 6144,
        hiddenLayers: Int = defaultHiddenLayers,
        attentionHeads: Int = 64,
        kvHeads: Int = defaultKvHeads,
        headDim: Int = 128,
        vocabularySize: Int = defaultVocabularySize,
        maxPositionEmbeddings: Int = defaultMaxPositionEmbeddings,
        rmsNormEps: Float = 1e-6,
        useGemmaNorm: Bool = true,
        ropeTheta: Float = defaultRopeTheta,
        rotaryDim: Int? = nil,
        partialRotaryFactor: Float = 0.5,
        hiddenAct: String = defaultHiddenAct,
        useQkNorm: Bool = true,
        qkNormType: String = defaultQkNormType,
        tieWordEmbeddings: Bool = false,
        denseIntermediateSize: Int = defaultDenseIntermediateSize,
        intermediateSize: Int = 3072,
        sharedIntermediateSize: Int? = nil,
        numLocalExperts: Int = 128,
        numExpertsPerTok: Int = defaultNumExpertsPerTok,
        nSharedExperts: Int = 1,
        scoringFunc: String = "sigmoid",
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
            moeLayerFreq ?? (0 ..< hiddenLayers).map { $0 < defaultMoeLayerStart ? 0 : 1 }
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
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        hiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? defaultHiddenLayers
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? defaultKvHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? defaultVocabularySize
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? defaultMaxPositionEmbeddings
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        useGemmaNorm = try container.decodeIfPresent(Bool.self, forKey: .useGemmaNorm) ?? true
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? defaultRopeTheta
        partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        rotaryDim =
            try container.decodeIfPresent(Int.self, forKey: .rotaryDim)
            ?? Int(partialRotaryFactor * Float(headDim))
        hiddenAct =
            try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? defaultHiddenAct
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        qkNormType =
            try container.decodeIfPresent(String.self, forKey: .qkNormType) ?? defaultQkNormType
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        denseIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize)
            ?? defaultDenseIntermediateSize
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        let decodedSharedIntermediateSize = try container.decodeIfPresent(
            Int.self, forKey: .sharedIntermediateSize)
        numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 128
        numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
            ?? defaultNumExpertsPerTok
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
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
            decodedMoeLayerFreq ?? (0 ..< hiddenLayers).map { $0 < defaultMoeLayerStart ? 0 : 1 }
    }
}

// MARK: - MiniMaxM3Configuration

/// Top-level MiniMax-M3 VL configuration: `text_config` (used by this dense
/// -attention-only task) plus `vision_config` (decoded upstream but unused
/// here -- vision support is a later task).
public struct MiniMaxM3Configuration: Codable, Sendable {
    /// Top-level checkpoint model type (`"minimax_m3_vl"`).
    public var modelType: String
    /// The text-model configuration used by the dense-attention language model.
    public var textConfiguration: MiniMaxM3TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
    }

    /// Creates a top-level MiniMax-M3 configuration from its model type and text configuration.
    public init(
        modelType: String = "minimax_m3_vl",
        textConfiguration: MiniMaxM3TextConfiguration = MiniMaxM3TextConfiguration()
    ) {
        self.modelType = modelType
        self.textConfiguration = textConfiguration
    }
}

// MARK: - MiniMaxM3Attention

/// MiniMax-M3's dense self-attention: GQA (64 query heads / 4 KV heads in the
/// verified config), partial RoPE (only the first `rotaryDim` of `headDim`
/// dims rotate -- `initializeRope` handles this natively since
/// `MLXFast.RoPE`/`RoPE` leave any dims beyond `dimensions` untouched), and
/// per-head Gemma-mode QK-norm applied *after* reshaping to
/// `(B, L, heads, headDim)` (unlike M2's flat, pre-reshape norm over
/// `heads * headDim`). `attention_output_gate` is `false` for M3 -- no output
/// gate module. Sparse (DSA) attention is a follow-up task (^8dbc476); this
/// is the dense fallback path used by every layer for now.
class MiniMaxM3Attention: Module {
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

    init(_ config: MiniMaxM3TextConfiguration) {
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

        self.rope = initializeRope(
            dims: config.rotaryDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

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

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MiniMaxM3MLP

/// Dense (non-MoE) MLP used by layers 0..<3 in the verified schedule, sized
/// by `denseIntermediateSize` (12,288) rather than the per-expert
/// `intermediateSize` (3,072) the MoE blocks use. Shares
/// `MiniMaxM3SwiGLUOAI`'s clipped activation with the MoE path.
class MiniMaxM3MLP: Module {
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
/// Unlike `DeepseekV3DecoderLayer` (`Libraries/MLXLLM/Models/DeepseekV3.swift`),
/// which reuses a single `mlp`-keyed property for both variants, M3's two
/// variants are exposed through two separate optional `@ModuleInfo`-wrapped
/// properties (`denseMLP`/`blockSparseMoe`) -- verified directly against the
/// real checkpoint's safetensors index: dense layers ship `mlp.gate_proj/
/// up_proj/down_proj`, while MoE layers ship `block_sparse_moe.gate` +
/// `block_sparse_moe.switch_mlp.*`, genuinely different key prefixes rather
/// than one shared name.
class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM3SparseMoeBlock?
    @ModuleInfo(key: "mlp") var denseMLP: MiniMaxM3MLP?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    init(_ config: MiniMaxM3TextConfiguration, layerIndex: Int) {
        _selfAttn.wrappedValue = MiniMaxM3Attention(config)

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
        } else {
            mlpOutput = denseMLP!(postAttentionLayerNorm(h))
        }
        return h + mlpOutput
    }
}

// MARK: - MiniMaxM3ModelInner

class MiniMaxM3ModelInner: Module {
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

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

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
        // `useGemmaNorm`/`qkNormType` decode faithfully but are not wired to
        // any branch below -- every norm unconditionally uses Gemma-mode
        // RMSNorm, and QK-norm (when enabled) is always per-head. Both hold
        // for the one verified real checkpoint; fail fast rather than
        // silently mis-computing numerics if a future checkpoint sets either
        // differently (flagged by adversarial review of kanban ^xgvth41).
        precondition(
            config.useGemmaNorm,
            "MiniMaxM3 only supports use_gemma_norm=true (dense-attention stage, ^xgvth41)")
        if config.useQkNorm {
            precondition(
                config.qkNormType == defaultQkNormType,
                "MiniMaxM3 only supports qk_norm_type=\"per_head\" (dense-attention stage, ^xgvth41)"
            )
        }

        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = MiniMaxM3ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }
}

// MARK: - MiniMaxM3Model

/// MiniMax-M3's dense-attention language model. **Not yet registered with the
/// VLM factory** -- there is no vision tower here, and this type does not
/// conform to `VLMModel` (which would require a real `prepare(_:cache:
/// state:windowSize:)` implementation); that lands with vision support in a
/// later task. This is the deliverable for kanban ^xgvth41: a model type that
/// compiles and runs tiny-config forward passes with a KV cache.
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
    @ModuleInfo(key: "language_model") var languageModel: MiniMaxM3LanguageModel

    /// The text-model configuration this instance was constructed from.
    public let configuration: MiniMaxM3TextConfiguration
    /// Per-layer key/value head counts, forwarded from the language model.
    public var kvHeads: [Int] { languageModel.kvHeads }
    /// Vocabulary size, forwarded from `configuration`.
    public var vocabularySize: Int { configuration.vocabularySize }

    /// Creates a dense-attention MiniMax-M3 language model from its text configuration.
    public init(_ config: MiniMaxM3TextConfiguration) {
        self.configuration = config
        _languageModel.wrappedValue = MiniMaxM3LanguageModel(config)
        super.init()
    }

    /// Runs the language model over `inputs`, reading from and updating `cache` in place if provided.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// One `KVCacheSimple` per decoder layer (or `RotatingKVCache` when
    /// `parameters.maxKVSize` is set), matching every layer's dense
    /// `MiniMaxM3Attention` -- there is no linear/SSM layer mix here to
    /// special-case, unlike e.g. `Qwen35`.
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        let numLayers = kvHeads.count
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
        }
        return (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    /// Sanitize and prepare weight dictionary for model loading: dequantize
    /// block-quantized (fp8/bf16 `weight_scale_inv`) weights, re-key flat
    /// `minimax_m3` checkpoints into the `language_model.`-prefixed layout,
    /// drop unused vision-tower/multi-modal-projector/MTP/sparse-attention-indexer
    /// weights, and remap per-expert fallback weights (`w1`/`w2`/`w3`) into the
    /// fused `gate_up_proj`/`down_proj` layout `FusedGateUpSwitchGLU` expects.
    /// The three phases are independent transformations, applied in order --
    /// see `_mergeQuantized`, `_filterUnusedWeights`, and `_remapExpertWeights`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let merged = _mergeQuantized(weights)
        let filtered = _filterUnusedWeights(merged)
        return _remapExpertWeights(filtered)
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

    /// `sanitize` step 2: drop vision-tower / multi-modal-projector / MTP
    /// weights (verified absent from the mlx-community/MiniMax-M3-4bit
    /// checkpoint; MTP task will revisit), strip the sparse-attention
    /// indexer weights (`index_q_proj`/`index_k_proj`/`index_q_norm`/
    /// `index_k_norm` -- confirmed present under `self_attn.` on every
    /// MoE-scheduled layer in the real checkpoint's safetensors index; this
    /// dense-only stage doesn't build the indexer, task ^8dbc476 does),
    /// re-key flat (`minimax_m3`) checkpoints into the `language_model.`
    /// -prefixed layout the VL checkpoint already uses, and drop the tied
    /// `lm_head` weight when `configuration.tieWordEmbeddings` is set.
    private func _filterUnusedWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var prefixed: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("vision_tower.") || key.hasPrefix("multi_modal_projector.")
                || key.hasPrefix("patch_merge_mlp.")
            {
                continue
            }
            if key.hasPrefix("mtp.") || key.contains(".mtp.") {
                continue
            }
            if key.contains("self_attn.index_") {
                continue
            }

            let newKey = key.hasPrefix("language_model.") ? key : "language_model.\(key)"
            prefixed[newKey] = value
        }

        if configuration.tieWordEmbeddings {
            prefixed["language_model.lm_head.weight"] = nil
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
        var prefixed = weights
        for layerIndex in 0 ..< configuration.hiddenLayers
        where configuration.isMoELayer(layerIndex) {
            let prefix = "language_model.model.layers.\(layerIndex).block_sparse_moe"
            guard prefixed["\(prefix).experts.0.w1.weight"] != nil else { continue }

            for key in ["weight", "scales", "biases"] {
                guard prefixed["\(prefix).experts.0.w1.\(key)"] != nil else { continue }

                let gate = (0 ..< configuration.numLocalExperts).map {
                    prefixed.removeValue(forKey: "\(prefix).experts.\($0).w1.\(key)")!
                }
                let up = (0 ..< configuration.numLocalExperts).map {
                    prefixed.removeValue(forKey: "\(prefix).experts.\($0).w3.\(key)")!
                }
                let down = (0 ..< configuration.numLocalExperts).map {
                    prefixed.removeValue(forKey: "\(prefix).experts.\($0).w2.\(key)")!
                }

                prefixed["\(prefix).switch_mlp.gate_up_proj.\(key)"] =
                    MLX.concatenated([MLX.stacked(gate), MLX.stacked(up)], axis: 1)
                prefixed["\(prefix).switch_mlp.down_proj.\(key)"] = MLX.stacked(down)
            }
        }
        return prefixed
    }
}

extension MiniMaxM3Model: LoRAModel {
    /// The decoder layers available for LoRA parameter adaptation.
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
