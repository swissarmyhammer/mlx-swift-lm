//
//  Flint.swift
//  mlx-swift-lm
//

// port of mlx-vlm mlx_vlm/models/flint/language.py (0.7.12), text only

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The configuration of a Recursive Flint checkpoint (`model_type` "flint").
public struct FlintConfiguration: Codable, Sendable {
    var modelType: String = "flint"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float = 1e-6
    var maxPositionEmbeddings: Int = 262_144
    var ropeTheta: Float = 10_000
    var ropeScaling: [String: StringOrNumber]? = nil
    var layerTypes: [String]
    var noRopeLayerIndices: [Int] = []
    /// The window of a `sliding_attention` layer, or `nil` when no layer slides.
    var slidingWindow: Int?
    var attentionBias: Bool = false
    var mlpOnlyLayers: [Int] = []
    var numExperts: Int
    var numExpertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int = 0
    var normTopkProb: Bool = true
    var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case mlpOnlyLayers = "mlp_only_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Keys that are checked or folded in while decoding, and not stored.
    private enum CheckedKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
        case useSlidingWindow = "use_sliding_window"
        case softmaxType = "softmax_type"
        case hiddenAct = "hidden_act"
        case routerScoreFunction = "router_score_function"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let checked = try decoder.container(keyedBy: CheckedKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "flint"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        // transformers 5 writes the RoPE settings under `rope_parameters`.
        if let parameters = try checked.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            if let theta = parameters["rope_theta"]?.asFloat() {
                self.ropeTheta = theta
            }
            self.ropeScaling = parameters
        }
        self.layerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.noRopeLayerIndices =
            try container.decodeIfPresent([Int].self, forKey: .noRopeLayerIndices) ?? []
        let useSlidingWindow =
            try checked.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        self.slidingWindow =
            useSlidingWindow
            ? (try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 513) : nil
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpOnlyLayers = try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types must match num_hidden_layers")
        }
        guard Set(layerTypes).isSubset(of: ["full_attention", "sliding_attention"]) else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "unsupported Flint attention layer type")
        }
        if let slidingWindow, slidingWindow < 1 {
            throw DecodingError.dataCorruptedError(
                forKey: .slidingWindow, in: container,
                debugDescription: "sliding_window must be positive")
        }
        let hiddenAct = try checked.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        let scoreFunction =
            try checked.decodeIfPresent(String.self, forKey: .routerScoreFunction) ?? "sigmoid"
        guard hiddenAct == "silu", scoreFunction == "sigmoid" else {
            throw DecodingError.dataCorruptedError(
                forKey: .routerScoreFunction, in: checked,
                debugDescription: "Flint requires SiLU experts and sigmoid routing")
        }
        let softmaxType =
            try checked.decodeIfPresent(String.self, forKey: .softmaxType) ?? "learnable"
        guard softmaxType == "learnable" else {
            throw DecodingError.dataCorruptedError(
                forKey: .softmaxType, in: checked,
                debugDescription: "Flint requires learned attention sinks")
        }
        guard (1 ... numExperts).contains(numExpertsPerToken) else {
            throw DecodingError.dataCorruptedError(
                forKey: .numExpertsPerToken, in: container,
                debugDescription: "num_experts_per_tok must be between 1 and num_experts")
        }
    }

    /// The window of layer `index`, or `nil` for a full-attention layer.
    func window(layer index: Int) -> Int? {
        layerTypes[index] == "sliding_attention" ? slidingWindow : nil
    }
}

extension FlintConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try validateRoPEConfiguration(ropeScaling, context: "FlintConfiguration.rope_scaling")
    }
}

// MARK: - Attention

class FlintAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    /// The learned attention sink of each query head (`softmax_type` "learnable").
    @ParameterInfo(key: "softmax_offset") var sinks: MLXArray

    /// `nil` for a layer of `no_rope_layer_indices`.
    let rope: RoPELayer?

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        _wq.wrappedValue = Linear(dim, attentionHeads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(attentionHeads * headDim, dim, bias: args.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _sinks.wrappedValue = zeros([attentionHeads])

        if args.noRopeLayerIndices.contains(layerIdx) {
            self.rope = nil
        } else {
            self.rope = initializeRope(
                dims: headDim, base: args.ropeTheta, traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, attentionHeads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask, sinks: sinks.asType(queries.dtype)
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

class FlintMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(compiledSiluProduct(gate(x), up(x)))
    }
}

/// Sigmoid-routed experts plus one shared expert. The expert bias steers the
/// choice of experts only; the weights of the chosen experts are their sigmoid
/// scores, normalized, in float32.
class FlintSparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP?

    init(_ args: FlintConfiguration) {
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _expertBias.wrappedValue = zeros([args.numExperts])
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        if args.sharedExpertIntermediateSize > 0 {
            _sharedExpert.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The projection runs in the activation dtype; sigmoid, selection and
        // normalization run in float32, as the reference does.
        let scores = sigmoid(gate(x).asType(.float32))
        let choice = scores + expertBias.asType(.float32)

        let indices: MLXArray
        var weights: MLXArray
        if supportsFusedRouterTopK(choice, k: topK) {
            (indices, weights) = fusedRouterTopK(
                selection: choice, values: scores, k: topK, normalize: false,
                order: .ascending)
        } else {
            let kth = choice.dim(-1) - topK
            indices = MLX.stopGradient(
                MLX.argPartition(choice, kth: kth, axis: -1)[.ellipsis, kth...])
            weights = MLX.takeAlong(scores, indices, axis: -1)
        }
        if normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }

        let routed = switchMLP(x, indices)
        var out = weightedExpertSum(routed.asType(.float32), weights).asType(x.dtype)
        if let sharedExpert {
            out = out + sharedExpert(x)
        }
        return out
    }
}

// MARK: - Decoder

class FlintDecoderLayer: Module {
    /// The sliding window of this layer, or `nil` for full attention.
    let window: Int?

    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    let mlp: UnaryLayer

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.window = args.window(layer: layerIdx)
        _selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.mlpOnlyLayers.contains(layerIdx) {
            self.mlp = FlintMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self.mlp = FlintSparseMoeBlock(args)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [FlintDecoderLayer]
    let norm: RMSNorm

    /// The first full-attention layer and the first sliding layer, whose caches
    /// shape the masks of every layer of their kind.
    let firstFullLayer: Int?
    let firstSlidingLayer: Int?

    init(_ args: FlintConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { FlintDecoderLayer(args, layerIdx: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.firstFullLayer = layers.firstIndex { $0.window == nil }
        self.firstSlidingLayer = layers.firstIndex { $0.window != nil }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = firstFullLayer.map {
            createAttentionMask(h: h, cache: cache?[$0])
        }
        let slidingMask = firstSlidingLayer.map {
            createAttentionMask(h: h, cache: cache?[$0], windowSize: layers[$0].window)
        }

        for (i, layer) in layers.enumerated() {
            let mask = layer.window == nil ? fullMask! : slidingMask!
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// The text model under the `language_model` key of a checkpoint.
public class FlintLanguageModel: Module {
    @ModuleInfo(key: "model") var model: FlintModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: FlintConfiguration) {
        _model.wrappedValue = FlintModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // The reference serves the output projection in float32.
        let hidden = model(inputs, cache: cache).asType(.float32)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}

// MARK: - Top-level Model

/// Recursive Flint: a 49-layer MoE (256 sigmoid-routed experts, top-6, one
/// shared expert) with interleaved sliding-window and full attention, learned
/// attention sinks, and no RoPE on some layers.
///
/// The module tree keeps the `language_model.` level of the mlx-vlm checkpoints
/// (for example `mlx-community/flint-oQ8e`), so that their per-tensor
/// quantization overrides name its layers. A checkpoint without that prefix
/// (the layout of mlx-lm) gets it in ``sanitize(weights:)``.
public class FlintModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: FlintConfiguration

    @ModuleInfo(key: "language_model") var languageModel: FlintLanguageModel

    public init(_ args: FlintConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        _languageModel.wrappedValue = FlintLanguageModel(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.layers.map { layer in
            try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: layer.window,
                usesSlidingWindow: layer.window != nil)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = "language_model."
        var weights = Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                (key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : key, value)
            })

        // Flint INT4 checkpoints store quantized weights as flat uint8; the
        // quantized layers read the packed uint32 layout.
        for (key, value) in weights where value.dtype == .uint8 && key.hasSuffix(".weight") {
            if weights[String(key.dropLast(".weight".count)) + ".scales"] != nil {
                weights[key] = value.view(dtype: .uint32)
            }
        }

        weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for (index, layer) in languageModel.model.layers.enumerated()
        where layer.mlp is FlintSparseMoeBlock {
            let mlp = "model.layers.\(index).mlp"
            // Hugging Face checkpoints hold one tensor per expert.
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let first = "\(mlp).experts.0.\(projection).\(suffix)"
                    guard weights[first] != nil else { continue }
                    let stacked = (0 ..< configuration.numExperts).map { expert in
                        weights.removeValue(
                            forKey: "\(mlp).experts.\(expert).\(projection).\(suffix)")!
                    }
                    weights["\(mlp).switch_mlp.\(projection).\(suffix)"] = MLX.stacked(stacked)
                }
            }
            for alias in ["e_score_correction_bias", "expert_bias"] {
                if let bias = weights.removeValue(forKey: "\(mlp).gate.\(alias)") {
                    weights["\(mlp).expert_bias"] = bias
                }
            }
            // The router correction bias is optional in the source checkpoint.
            if weights["\(mlp).gate.weight"] != nil, weights["\(mlp).expert_bias"] == nil {
                weights["\(mlp).expert_bias"] = zeros([configuration.numExperts])
            }
        }

        return Dictionary(uniqueKeysWithValues: weights.map { (prefix + $0.key, $0.value) })
    }
}

// MARK: - LoRA

extension FlintModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Chat conventions

extension FlintModel {
    /// Flint writes GLM-style calls:
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    /// Flint always thinks: the chat template opens no `<think>` block and has
    /// no switch, so the model opens the block itself. The template writes the
    /// `reasoning_content` of every past assistant turn back into its
    /// `<think>` block, thus a replayed history extends the tokens the model
    /// wrote and the prompt cache carries across turns.
    public var reasoningConfig: ReasoningConfig? { FlintModel.reasoning }

    static let reasoning: ReasoningConfig = {
        var config = ReasoningConfig.alwaysOnThinking
        config.implicitEndDelimiters = ["<tool_call>"]
        config.replaysReasoningIntoHistory = true
        return config
    }()
}
