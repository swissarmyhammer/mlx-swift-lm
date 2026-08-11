// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4Configuration.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// The four `dspark_*` keys are not in the file above. They come from the
// transcribed copy of that file in scouzi1966/mlx-swift-lm,
// Libraries/MLXLLM/Models/DeepseekV4Configuration.swift @ e1852869ce61ded0d23b76df3757e9b75c77c1f5,
// which gives the same Osaurus AI copyright line.

import MLXLMCommon

/// The configuration of a DeepSeek-V4 checkpoint.
///
/// DeepSeek-V4 is not DeepSeek-V3 with more layers. It adds manifold
/// hyper-connections, one latent key/value head with a head dimension of 512, a
/// grouped low-rank output projection, hash routing on the first layers, and a
/// compress ratio for each layer.
///
/// Every key of `config.json` is optional. A key that the file does not give
/// gets the DeepSeek-V4-Flash value, thus a short file decodes without an error.
public struct DeepseekV4Configuration: Codable, Sendable {

    // MARK: - Core transformer

    /// The number of tokens in the vocabulary.
    public var vocabSize: Int
    /// The width of the residual stream.
    public var hiddenSize: Int
    /// The number of decoder layers.
    public var numHiddenLayers: Int
    /// The number of query heads in each attention layer.
    public var numAttentionHeads: Int
    /// The number of latent key/value heads. DeepSeek-V4 keeps one head and
    /// sends it to every query head.
    public var numKeyValueHeads: Int
    /// The width of one attention head. DeepSeek-V4 does not divide this width
    /// into a no-position part and a rotary part.
    public var headDim: Int
    /// The number of dimensions at the end of a head that get the rotary
    /// position. The dimensions before them keep no position.
    public var qkRopeHeadDim: Int
    /// The rank of the low-rank query projection.
    public var qLoraRank: Int
    /// The epsilon of each RMS norm.
    public var rmsNormEps: Float
    /// The largest context length the checkpoint accepts.
    public var maxPositionEmbeddings: Int

    // MARK: - Grouped low-rank output projection

    /// The number of head groups the output projection reads. Each group gets
    /// its own low-rank matrix.
    public var oGroups: Int
    /// The rank of the low-rank output projection of one group.
    public var oLoraRank: Int

    // MARK: - Mixture of experts

    /// The number of routed experts in a layer.
    public var nRoutedExperts: Int
    /// The number of shared experts in a layer. A shared expert reads every
    /// token.
    public var numSharedExperts: Int
    /// The number of routed experts one token reads.
    public var numExpertsPerTok: Int
    /// The width of one expert.
    public var moeIntermediateSize: Int
    /// The number of first layers that use hash routing. A hash layer reads a
    /// table from token identifier to expert identifier, and does not use the
    /// top-k gate.
    public var numHashLayers: Int
    /// The name of the function that makes the gate scores.
    public var scoringFunction: String
    /// True when the gate divides the top-k weights by their sum.
    public var normalizeTopkProb: Bool
    /// The factor that multiplies the output of the routed experts.
    public var routedScalingFactor: Float
    /// The limit of the DeepSeek-V4 SwiGLU. The gate and the up value stay
    /// inside this limit, which stops the activation from growing too large.
    public var swigluLimit: Float

    // MARK: - Manifold hyper-connections

    /// The number of parallel copies of the residual stream in each layer.
    public var hcMult: Int
    /// The number of Sinkhorn steps that make the mixing matrix doubly
    /// stochastic.
    public var hcSinkhornIters: Int
    /// The epsilon of the Sinkhorn steps.
    public var hcEps: Float

    // MARK: - Rotary position

    /// The rope theta of a layer with a compress ratio of 0.
    public var ropeTheta: Float
    /// The rope theta of a layer with a compress ratio of more than 0.
    public var compressRopeTheta: Float
    /// The YaRN scaling of the rope, or `nil` when the file gives no scaling.
    public var ropeScaling: [String: StringOrNumber]?

    // MARK: - Sliding window and compressor

    /// The width of the sliding attention window.
    public var slidingWindow: Int
    /// The compress ratio of each layer. A ratio of 0 gives plain attention. A
    /// ratio of more than 0 adds a compressor for the global context.
    public var compressRatios: [Int]

    // MARK: - Indexer

    /// The number of indexer heads.
    public var indexNHeads: Int
    /// The width of one indexer head.
    public var indexHeadDim: Int
    /// The number of keys the indexer keeps for each query.
    public var indexTopK: Int

    // MARK: - Attention sink

    /// True when each attention layer adds a learned logit before the softmax.
    public var useAttnSink: Bool

    // MARK: - DSpark

    /// The block size of DSpark. This repository decodes the DSpark keys and
    /// does not use them.
    public var dsparkBlockSize: Int
    /// The identifier of the DSpark noise token. This repository decodes the
    /// DSpark keys and does not use them.
    public var dsparkNoiseTokenId: Int
    /// The layers DSpark reads. This repository decodes the DSpark keys and
    /// does not use them.
    public var dsparkTargetLayerIds: [Int]
    /// The rank of the DSpark Markov matrix. This repository decodes the DSpark
    /// keys and does not use them.
    public var dsparkMarkovRank: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case nRoutedExperts = "n_routed_experts"
        case numSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHashLayers = "num_hash_layers"
        case scoringFunction = "scoring_func"
        case normalizeTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case compressRopeTheta = "compress_rope_theta"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case useAttnSink = "use_attn_sink"
        case dsparkBlockSize = "dspark_block_size"
        case dsparkNoiseTokenId = "dspark_noise_token_id"
        case dsparkTargetLayerIds = "dspark_target_layer_ids"
        case dsparkMarkovRank = "dspark_markov_rank"
    }

    /// Reads a DeepSeek-V4 `config.json`.
    ///
    /// - Parameter decoder: The decoder that holds the `config.json` content.
    /// - Throws: A `DecodingError` when a key that the file gives has the wrong
    ///   type. A key that the file does not give gets its default value.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        func value<T: Decodable>(_ key: CodingKeys, or fallback: T) throws -> T {
            try container.decodeIfPresent(T.self, forKey: key) ?? fallback
        }

        self.vocabSize = try value(.vocabSize, or: Default.vocabSize)
        self.hiddenSize = try value(.hiddenSize, or: Default.hiddenSize)
        self.numHiddenLayers = try value(.numHiddenLayers, or: Default.numHiddenLayers)
        self.numAttentionHeads = try value(.numAttentionHeads, or: Default.numAttentionHeads)
        self.numKeyValueHeads = try value(.numKeyValueHeads, or: Default.numKeyValueHeads)
        self.headDim = try value(.headDim, or: Default.headDim)
        self.qkRopeHeadDim = try value(.qkRopeHeadDim, or: Default.qkRopeHeadDim)
        self.qLoraRank = try value(.qLoraRank, or: Default.qLoraRank)
        self.rmsNormEps = try value(.rmsNormEps, or: Default.rmsNormEps)
        self.maxPositionEmbeddings = try value(
            .maxPositionEmbeddings, or: Default.maxPositionEmbeddings)
        self.oGroups = try value(.oGroups, or: Default.oGroups)
        self.oLoraRank = try value(.oLoraRank, or: Default.oLoraRank)
        self.nRoutedExperts = try value(.nRoutedExperts, or: Default.nRoutedExperts)
        self.numSharedExperts = try value(.numSharedExperts, or: Default.numSharedExperts)
        self.numExpertsPerTok = try value(.numExpertsPerTok, or: Default.numExpertsPerTok)
        self.moeIntermediateSize = try value(.moeIntermediateSize, or: Default.moeIntermediateSize)
        self.numHashLayers = try value(.numHashLayers, or: Default.numHashLayers)
        self.scoringFunction = try value(.scoringFunction, or: Default.scoringFunction)
        self.normalizeTopkProb = try value(.normalizeTopkProb, or: Default.normalizeTopkProb)
        self.routedScalingFactor = try value(.routedScalingFactor, or: Default.routedScalingFactor)
        self.swigluLimit = try value(.swigluLimit, or: Default.swigluLimit)
        self.hcMult = try value(.hcMult, or: Default.hcMult)
        self.hcSinkhornIters = try value(.hcSinkhornIters, or: Default.hcSinkhornIters)
        self.hcEps = try value(.hcEps, or: Default.hcEps)
        self.ropeTheta = try value(.ropeTheta, or: Default.ropeTheta)
        self.compressRopeTheta = try value(.compressRopeTheta, or: Default.compressRopeTheta)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.slidingWindow = try value(.slidingWindow, or: Default.slidingWindow)
        self.compressRatios = try value(.compressRatios, or: Default.compressRatios)
        self.indexNHeads = try value(.indexNHeads, or: Default.indexNHeads)
        self.indexHeadDim = try value(.indexHeadDim, or: Default.indexHeadDim)
        self.indexTopK = try value(.indexTopK, or: Default.indexTopK)
        self.useAttnSink = try value(.useAttnSink, or: Default.useAttnSink)
        self.dsparkBlockSize = try value(.dsparkBlockSize, or: Default.dsparkBlockSize)
        self.dsparkNoiseTokenId = try value(.dsparkNoiseTokenId, or: Default.dsparkNoiseTokenId)
        self.dsparkTargetLayerIds = try value(
            .dsparkTargetLayerIds, or: Default.dsparkTargetLayerIds)
        self.dsparkMarkovRank = try value(.dsparkMarkovRank, or: Default.dsparkMarkovRank)
    }

    /// The value of each key for a file that does not give that key. The
    /// numbers are those of the DeepSeek-V4-Flash checkpoint.
    private enum Default {
        static let vocabSize = 129_280
        static let hiddenSize = 4096
        static let numHiddenLayers = 43
        static let numAttentionHeads = 64
        static let numKeyValueHeads = 1
        static let headDim = 512
        static let qkRopeHeadDim = 64
        static let qLoraRank = 1024
        static let rmsNormEps: Float = 1e-6
        static let maxPositionEmbeddings = 1_048_576
        static let oGroups = 8
        static let oLoraRank = 1024
        static let nRoutedExperts = 256
        static let numSharedExperts = 1
        static let numExpertsPerTok = 6
        static let moeIntermediateSize = 2048
        static let numHashLayers = 3
        static let scoringFunction = "sqrtsoftplus"
        static let normalizeTopkProb = true
        static let routedScalingFactor: Float = 1.5
        static let swigluLimit: Float = 10.0
        static let hcMult = 4
        static let hcSinkhornIters = 20
        static let hcEps: Float = 1e-6
        static let ropeTheta: Float = 10000.0
        static let compressRopeTheta: Float = 160_000.0
        static let slidingWindow = 128
        static let compressRatios: [Int] = []
        static let indexNHeads = 64
        static let indexHeadDim = 128
        static let indexTopK = 512
        static let useAttnSink = true
        static let dsparkBlockSize = 0
        static let dsparkNoiseTokenId = 0
        static let dsparkTargetLayerIds: [Int] = []
        static let dsparkMarkovRank = 256
    }
}

extension DeepseekV4Configuration {

    /// Tells whether a layer routes its tokens through the hash table.
    ///
    /// The first ``numHashLayers`` layers read a table from token identifier to
    /// expert identifier. They do not use the top-k gate.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: True when the layer uses hash routing.
    public func isHashLayer(_ layer: Int) -> Bool {
        layer < numHashLayers
    }

    /// Tells whether a layer has a compressor.
    ///
    /// A compressor sits on a layer whose compress ratio is more than 0. A
    /// layer that the ``compressRatios`` list does not reach has no compressor.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: True when the layer has a compressor.
    public func hasCompressor(layer: Int) -> Bool {
        guard layer < compressRatios.count else { return false }
        return compressRatios[layer] > 0
    }

    /// Gives the rope theta of one layer.
    ///
    /// A layer with a compressor uses ``compressRopeTheta``, which goes with
    /// the YaRN scaling. Every other layer uses ``ropeTheta``.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: The rope theta of that layer.
    public func ropeTheta(forLayer layer: Int) -> Float {
        hasCompressor(layer: layer) ? compressRopeTheta : ropeTheta
    }
}
