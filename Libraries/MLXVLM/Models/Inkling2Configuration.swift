// Copyright © 2026 Apple Inc.
//
// The configuration of Inkling 2: the text stack, its loop, the vision stack and the model.

import Foundation

// MARK: - Configuration

/// The loop of the Inkling 2 text stack: a core of layers that the model runs several times (`loop_config`).
public struct Inkling2LoopConfiguration: Codable, Sendable {
    /// The first layer of the loop core.
    public let coreStart: Int
    /// The layer after the last layer of the loop core.
    public let coreEnd: Int
    /// The passes through the loop core.
    public let passes: Int
    /// The rank of the low-rank gates of the loop.
    public let gateRank: Int

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
    }

    /// The layers of the loop core.
    public var core: Range<Int> { coreStart ..< coreEnd }
}

/// The configuration of the Inkling 2 text stack (`text_config`).
public struct Inkling2TextConfiguration: Codable, Sendable {
    /// The width of the residual stream.
    public let hiddenSize: Int
    /// The number of distinct decoder layers.
    public let hiddenLayers: Int
    /// The rows of the embedding and of the output head, padding included.
    public let vocabularySize: Int
    /// The tokens of the tokenizer; the rows past it are padding.
    public let unpaddedVocabularySize: Int
    /// The query heads of a full-attention layer.
    public let attentionHeads: Int
    /// The key and value heads of a full-attention layer.
    public let kvHeads: Int
    /// The width of one head of a full-attention layer.
    public let headDim: Int
    /// The query heads of a sliding-window layer.
    public let slidingAttentionHeads: Int
    /// The key and value heads of a sliding-window layer.
    public let slidingKVHeads: Int
    /// The width of one head of a sliding-window layer.
    public let slidingHeadDim: Int
    /// The tokens a sliding-window layer attends to.
    public let slidingWindow: Int
    /// The layers that use sliding-window attention.
    public let localLayerIds: [Int]
    /// The width of the relative-position projection of each head.
    public let relativeDim: Int
    /// The key distances the relative-position logits of a full-attention layer cover.
    public let relativeExtent: Int
    /// The query position below which the full-attention layers do not scale their queries.
    public let logScalingFloor: Int
    /// The strength of the log-length query scaling of the full-attention layers.
    public let logScalingAlpha: Float
    /// The epsilon of the RMS norms.
    public let rmsNormEps: Float
    /// The L2 norm to which the query and key head vectors are clipped.
    public let qkClipThreshold: Float
    /// The L2 norm to which the relative-position head vectors are clipped.
    public let relativeClipThreshold: Float
    /// The taps of the short convolutions.
    public let convolutionKernelSize: Int
    /// The layers of one attention-residual block.
    public let layersPerBlock: Int
    /// The routed experts of each mixture-of-experts layer.
    public let experts: Int
    /// The experts that each token is routed to.
    public let expertsPerToken: Int
    /// The inner width of one expert.
    public let intermediateSize: Int
    /// The soft cap of the expert gate activations, or 0 for none.
    public let gateSoftcap: Float
    /// The soft cap of the expert up activations, or 0 for none.
    public let upSoftcap: Float
    /// The factor applied to the routing weights.
    public let routeScale: Float
    /// The soft cap of the output logits, or 0 for none.
    public let finalLogitSoftcap: Float
    /// The loop of the layer stack, or `nil` for a stack that runs each layer once.
    public let loop: Inkling2LoopConfiguration?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingAttentionHeads = "swa_num_attention_heads"
        case slidingKVHeads = "swa_num_key_value_heads"
        case slidingHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case relativeDim = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkClipThreshold = "qk_l2_clip_threshold"
        case relativeClipThreshold = "rel_l2_clip_threshold"
        case convolutionKernelSize = "sconv_kernel_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case experts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case gateSoftcap = "mlp_gate_softcap"
        case upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case finalLogitSoftcap = "final_logit_softcapping"
        case loop = "loop_config"
    }

    /// The taps of the short convolutions, for a `text_config` that leaves them out.
    private static let defaultConvolutionKernelSize = 4

    /// Decodes a `text_config`, with the defaults of Inkling 2 for the keys it leaves out.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        unpaddedVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize) ?? vocabularySize
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        slidingAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .slidingAttentionHeads) ?? attentionHeads
        slidingKVHeads = try c.decodeIfPresent(Int.self, forKey: .slidingKVHeads) ?? kvHeads
        slidingHeadDim = try c.decodeIfPresent(Int.self, forKey: .slidingHeadDim) ?? headDim
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        localLayerIds = try c.decodeIfPresent([Int].self, forKey: .localLayerIds) ?? []
        relativeDim = try c.decode(Int.self, forKey: .relativeDim)
        relativeExtent = try c.decode(Int.self, forKey: .relativeExtent)
        logScalingFloor = try c.decode(Int.self, forKey: .logScalingFloor)
        logScalingAlpha = try c.decode(Float.self, forKey: .logScalingAlpha)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? normEpsilon
        qkClipThreshold = try c.decode(Float.self, forKey: .qkClipThreshold)
        relativeClipThreshold = try c.decode(Float.self, forKey: .relativeClipThreshold)
        convolutionKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .convolutionKernelSize)
            ?? Self.defaultConvolutionKernelSize
        layersPerBlock = try c.decode(Int.self, forKey: .layersPerBlock)
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        gateSoftcap = try c.decodeIfPresent(Float.self, forKey: .gateSoftcap) ?? 0
        upSoftcap = try c.decodeIfPresent(Float.self, forKey: .upSoftcap) ?? 0
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 1
        finalLogitSoftcap = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 0
        loop = try c.decodeIfPresent(Inkling2LoopConfiguration.self, forKey: .loop)
    }

    /// The layer of each layer visit: the prelude, every pass of the loop core, then the coda.
    public var layerVisits: [Int] {
        guard let loop else { return Array(0 ..< hiddenLayers) }
        return Array(0 ..< loop.coreStart)
            + (0 ..< loop.passes).flatMap { _ in loop.core }
            + Array(loop.coreEnd ..< hiddenLayers)
    }

    func isSliding(_ layer: Int) -> Bool { localLayerIds.contains(layer) }
}

/// The configuration of the Inkling 2 vision tower (`vision_config`).
public struct Inkling2VisionConfiguration: Codable, Sendable {
    /// The width of the vision tower.
    public let hiddenSize: Int
    /// The layers of the vision tower.
    public let hiddenLayers: Int
    /// The attention heads of each vision layer.
    public let attentionHeads: Int
    /// The pixels on each side of one patch.
    public let patchSize: Int
    /// The patches on each side of the square that the adapter merges into one token.
    public let mergeSize: Int
    /// The cells on each side of the learned position-embedding grid.
    public let positionGridSize: Int
    /// The hidden width of the projector into the text stack.
    public let projectionHiddenSize: Int
    /// Whether the projector layers have biases.
    public let projectorBias: Bool
    /// The ratio of the MLP width to the hidden width.
    public let mlpRatio: Float
    /// The epsilon of the layer norms.
    public let layerNormEps: Float
    /// The base of the rotary position embedding.
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case patchSize = "native_patch_size"
        case mergeSize = "adapter_merge_size"
        case positionGridSize = "posemb_grid_size"
        case projectionHiddenSize = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case mlpRatio = "mlp_ratio"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    /// Decodes a `vision_config`, with the settings of the Inkling 2 vision tower for the keys it leaves out.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self.defaults
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? defaults.hiddenSize
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? defaults.hiddenLayers
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? defaults.attentionHeads
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? defaults.patchSize
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? defaults.mergeSize
        positionGridSize =
            try c.decodeIfPresent(Int.self, forKey: .positionGridSize)
            ?? defaults.positionGridSize
        projectionHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectionHiddenSize)
            ?? defaults.projectionHiddenSize
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? defaults.mlpRatio
        layerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? defaults.layerNormEps
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? defaults.ropeTheta
    }

    /// The settings of the Inkling 2 vision tower, for a `vision_config` that leaves one out.
    private struct Defaults {
        let hiddenSize = 1024
        let hiddenLayers = 23
        let attentionHeads = 16
        let patchSize = 20
        let mergeSize = 2
        let positionGridSize = 32
        let projectionHiddenSize = 4096
        let mlpRatio: Float = 4
        let layerNormEps: Float = 1e-5
        let ropeTheta: Float = 10_000
    }

    /// The settings of the Inkling 2 vision tower.
    private static let defaults = Defaults()

    var headDim: Int { hiddenSize / attentionHeads }
    var intermediateSize: Int { Int(Float(hiddenSize) * mlpRatio) }
    /// The pixels on each side of one image token.
    var blockSize: Int { patchSize * mergeSize }
}

/// The configuration of an Inkling 2 checkpoint (`config.json`).
public struct Inkling2Configuration: Codable, Sendable {
    /// The `model_type` of the checkpoint.
    public let modelType: String
    /// The configuration of the text stack.
    public let textConfiguration: Inkling2TextConfiguration
    /// The configuration of the vision tower, or `nil` for a text-only checkpoint.
    public let visionConfiguration: Inkling2VisionConfiguration?
    /// The token that stands for one image token in the prompt.
    public let imageTokenId: Int
    /// The MTP drafter the checkpoint ships in `mtp.safetensors`, or `nil` without one.
    public let mtpConfiguration: Inkling2MTPConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenId = "image_token_id"
        case mtpConfiguration = "mtp_config"
    }

    /// The image token of the Inkling 2 tokenizer, for a `config.json` that leaves it out.
    private static let defaultImageTokenId = 200_054

    /// Decodes a `config.json`, with the image token of the Inkling 2 tokenizer when it leaves one out.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        textConfiguration = try c.decode(Inkling2TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decodeIfPresent(
            Inkling2VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenId =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? Self.defaultImageTokenId
        mtpConfiguration = try c.decodeIfPresent(
            Inkling2MTPConfiguration.self, forKey: .mtpConfiguration)
    }
}
