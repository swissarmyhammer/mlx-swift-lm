// Copyright © 2026 Apple Inc.
//
// The configuration of Inkling 2: the text stack, its loop, the vision stack and the model.

import Foundation

// MARK: - Configuration

public struct Inkling2LoopConfiguration: Codable, Sendable {
    public let coreStart: Int
    public let coreEnd: Int
    public let passes: Int
    public let gateRank: Int

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
    }

    public var core: Range<Int> { coreStart ..< coreEnd }
}

public struct Inkling2TextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    public let unpaddedVocabularySize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let slidingAttentionHeads: Int
    public let slidingKVHeads: Int
    public let slidingHeadDim: Int
    public let slidingWindow: Int
    public let localLayerIds: [Int]
    public let relativeDim: Int
    public let relativeExtent: Int
    public let logScalingFloor: Int
    public let logScalingAlpha: Float
    public let rmsNormEps: Float
    public let qkClipThreshold: Float
    public let relativeClipThreshold: Float
    public let convolutionKernelSize: Int
    public let layersPerBlock: Int
    public let experts: Int
    public let expertsPerToken: Int
    public let intermediateSize: Int
    public let gateSoftcap: Float
    public let upSoftcap: Float
    public let routeScale: Float
    public let finalLogitSoftcap: Float
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

public struct Inkling2VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let patchSize: Int
    public let mergeSize: Int
    public let positionGridSize: Int
    public let projectionHiddenSize: Int
    public let projectorBias: Bool
    public let mlpRatio: Float
    public let layerNormEps: Float
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

public struct Inkling2Configuration: Codable, Sendable {
    public let modelType: String
    public let textConfiguration: Inkling2TextConfiguration
    public let visionConfiguration: Inkling2VisionConfiguration?
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
