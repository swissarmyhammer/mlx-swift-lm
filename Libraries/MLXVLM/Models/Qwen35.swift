//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

private let precomputedPositionIdsKey = LMOutput.Key<MLXArray>(
    "qwen35.precomputedPositionIds")
private let ropeDeltasKey = LMOutput.Key<MLXArray>(
    "qwen35.ropeDeltas")

private func supportsExactCompiledDecode(_ dtype: DType) -> Bool {
    dtype == .float16 || dtype == .bfloat16
}

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        /// The values a `config.json` that leaves a field out gets.
        fileprivate enum Defaults {
            /// The width of the hidden states.
            static let hiddenSize = 4096
            /// The decoder layers.
            static let hiddenLayers = 32
            /// The width of the MLP.
            static let intermediateSize = 14_336
            /// The query heads of an attention layer.
            static let attentionHeads = 32
            /// The key and value heads of an attention layer.
            static let kvHeads = 8
            /// The value heads of a GDN layer.
            static let linearNumValueHeads = 64
            /// The key heads of a GDN layer.
            static let linearNumKeyHeads = 16
            /// The width of a GDN key head.
            static let linearKeyHeadDim = 192
            /// The width of a GDN value head.
            static let linearValueHeadDim = 128
            /// The kernel size of the GDN conv.
            static let linearConvKernelDim = 4
            /// The epsilon of the RMS norms.
            static let rmsNormEps: Float = 1e-6
            /// The tokens of the vocabulary.
            static let vocabularySize = 248_320
            /// The base of the rotary frequencies.
            static let ropeTheta: Float = 100_000.0
            /// The share of each head that M-RoPE rotates.
            static let partialRotaryFactor: Float = 0.25
            /// The longest sequence the positions were trained on.
            static let maxPositionEmbeddings = 131_072
            /// One full-attention layer for each of this many layers.
            static let fullAttentionInterval = 4
            /// The rotary frequencies of the temporal M-RoPE axis.
            static let temporalFrequencies = 11
            /// The rotary frequencies of the height M-RoPE axis.
            static let heightFrequencies = 11
            /// The rotary frequencies of the width M-RoPE axis.
            static let widthFrequencies = 10
            /// The rotary frequencies of the temporal, height and width
            /// M-RoPE axes.
            static let mropeSection = [temporalFrequencies, heightFrequencies, widthFrequencies]
        }

        public var modelType: String = ""
        public var hiddenSize: Int = Defaults.hiddenSize
        public var hiddenLayers: Int = Defaults.hiddenLayers
        public var intermediateSize: Int = Defaults.intermediateSize
        public var attentionHeads: Int = Defaults.attentionHeads
        public var kvHeads: Int = Defaults.kvHeads
        public var linearNumValueHeads: Int = Defaults.linearNumValueHeads
        public var linearNumKeyHeads: Int = Defaults.linearNumKeyHeads
        public var linearKeyHeadDim: Int = Defaults.linearKeyHeadDim
        public var linearValueHeadDim: Int = Defaults.linearValueHeadDim
        public var linearConvKernelDim: Int = Defaults.linearConvKernelDim
        public var rmsNormEps: Float = Defaults.rmsNormEps
        public var vocabularySize: Int = Defaults.vocabularySize
        public var ropeTheta: Float = Defaults.ropeTheta
        public var partialRotaryFactor: Float = Defaults.partialRotaryFactor
        public var maxPositionEmbeddings: Int = Defaults.maxPositionEmbeddings
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = Defaults.fullAttentionInterval
        public var layerTypes: [String]?
        var resolvedLayerTypes: [String] = []
        public var mtpNumHiddenLayers: Int = 0
        public var mtpUseDedicatedEmbeddings: Bool = false

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case layerTypes = "layer_types"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize =
                try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? Defaults.hiddenSize
            self.hiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? Defaults.hiddenLayers
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? Defaults.intermediateSize
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? Defaults.attentionHeads
            self.kvHeads =
                try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? Defaults.kvHeads
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads)
                ?? Defaults.linearNumValueHeads
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads)
                ?? Defaults.linearNumKeyHeads
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim)
                ?? Defaults.linearKeyHeadDim
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim)
                ?? Defaults.linearValueHeadDim
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim)
                ?? Defaults.linearConvKernelDim
            self.rmsNormEps =
                try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? Defaults.rmsNormEps
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? Defaults.vocabularySize
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? Defaults.maxPositionEmbeddings
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval)
                ?? Defaults.fullAttentionInterval
            self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            let layerSchedule = try HybridAttentionSchedule(
                hiddenLayerCount: hiddenLayers,
                fullAttentionInterval: fullAttentionInterval,
                explicitLayerTypes: layerTypes
            )
            self.resolvedLayerTypes = layerSchedule.layerTypes
            self.mtpNumHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
            self.mtpUseDedicatedEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .mtpUseDedicatedEmbeddings)
                ?? false

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints(Defaults.mropeSection),
                "rope_theta": .float(Defaults.ropeTheta),
                "partial_rotary_factor": .float(Defaults.partialRotaryFactor),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? Defaults.ropeTheta
                self.partialRotaryFactor =
                    decodedRope["partial_rotary_factor"]?.asFloat()
                    ?? Defaults.partialRotaryFactor
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = Defaults.ropeTheta
                self.partialRotaryFactor = Defaults.partialRotaryFactor
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    /// The ids of the special tokens a `config.json` that leaves them out gets.
    private enum DefaultTokenId {
        /// The placeholder of an image patch.
        static let image = 248_056
        /// The placeholder of a video patch.
        static let video = 248_057
        /// The token that opens an image or a video.
        static let visionStart = 248_045
        /// The token that closes an image or a video.
        static let visionEnd = 248_046
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? DefaultTokenId.image }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? DefaultTokenId.video }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? DefaultTokenId.visionStart }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? DefaultTokenId.visionEnd }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

/// The text decoder of the Qwen 3.5 vision-language model, and the layers it
/// is built from.
public enum Qwen35Language {

    /// The M-RoPE rotation of the tokens of a step: the cosines and the sines
    /// of their positions.
    typealias RotaryValues = (cos: MLXArray, sin: MLXArray)

    /// The axis of the sequences of a batch of hidden states or token ids.
    private static let batchAxis = 0

    /// The axis of the tokens of a sequence of hidden states or token ids.
    private static let tokenAxis = 1

    /// The sequences of the one batch a compiled trace runs.
    private static let compiledBatchSize = 1

    /// The rank of an array of shape `(batch, tokens)`: positions that every
    /// M-RoPE axis shares, or a padding mask.
    private static let batchTokensRank = 2

    /// The dimensions of a head that one rotary frequency turns: a pair.
    private static let dimensionsPerFrequency = 2

    /// The axes of the M-RoPE positions, `(axes, batch, tokens)`, and of the
    /// sections of the rotary frequencies. The frequencies interleave the
    /// axes: frequency `i` belongs to axis `i % count` within its section.
    private enum MRopeAxis {
        /// The height axis.
        static let height = 1
        /// The width axis.
        static let width = 2
        /// The temporal, height and width axes.
        static let count = 3
    }

    /// The places of the two states of a GDN layer in its `MambaCache`.
    private enum LinearCacheSlot {
        /// The conv state.
        static let convState = 0
        /// The recurrent state.
        static let recurrentState = 1
    }

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeIndices: MLXArray
        let attentionScaling: Float

        init(
            dim: Int, base: Float, mropeSection: [Int],
            ropeParameters: [String: StringOrNumber]? = nil, maxPositionEmbeddings: Int = 131_072
        ) {
            let safeDim = max(1, dim)
            var freq = MLXArray(
                stride(from: 0, to: safeDim, by: Qwen35Language.dimensionsPerFrequency)
            ).asType(.float32)
            freq = freq / Float(safeDim)

            let ropeType = ropeParameters?["type"] ?? ropeParameters?["rope_type"]
            if let ropeParameters, case .string("yarn") = ropeType {
                let yarn = Qwen35Language.yarnParameters(
                    dim: safeDim, base: base, maxPositionEmbeddings: maxPositionEmbeddings,
                    ropeParameters)
                self.invFreq = MLXArray(yarn.invFreq)
                self.attentionScaling = yarn.attentionFactor
            } else {
                self.invFreq = 1.0 / pow(MLXArray(base), freq)
                self.attentionScaling = 1
            }

            let sections =
                mropeSection.count >= MRopeAxis.count
                ? mropeSection : Qwen35Configuration.TextConfiguration.Defaults.mropeSection
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for axis in [MRopeAxis.height, MRopeAxis.width] {
                let end = min(sections[axis] * MRopeAxis.count, indices.count)
                for index in stride(from: axis, to: end, by: MRopeAxis.count) {
                    indices[index] = Int32(axis)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == Qwen35Language.batchTokensRank {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [MRopeAxis.count, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            var cosEmb = cos(emb)
            var sinEmb = sin(emb)
            // YaRN scales only the rotary slice of q and k, as in HF.
            if attentionScaling != 1 {
                cosEmb = cosEmb * attentionScaling
                sinEmb = sinEmb * attentionScaling
            }
            return (cosEmb.asType(x.dtype), sinEmb.asType(x.dtype))
        }
    }

    /// The YaRN constants of HF `_compute_yarn_parameters`.
    private enum Yarn {
        /// The rotations of the fastest frequency that YaRN keeps whole,
        /// when the rope parameters leave out `beta_fast`.
        static let betaFast: Float = 32
        /// The slope of the attention factor over the log of the scale.
        static let mscaleSlope = 0.1
        /// The width a correction range of zero widens to, so the ramp never
        /// divides by zero.
        static let degenerateRangeWidth = 0.001
        /// One full turn, in radians: two half turns.
        static let fullTurn = Double.pi + Double.pi
    }

    /// HF `_compute_yarn_parameters`: blended inverse frequencies and the attention factor.
    static func yarnParameters(
        dim: Int, base: Float, maxPositionEmbeddings: Int,
        _ ropeParameters: [String: StringOrNumber]
    ) -> (invFreq: [Float], attentionFactor: Float) {
        let base = Double(base)
        let factor = Double(ropeParameters["factor"]?.asFloat() ?? 1)
        let originalMax = Double(
            ropeParameters["original_max_position_embeddings"]?.asFloat()
                ?? Float(maxPositionEmbeddings))
        let betaFast = Double(ropeParameters["beta_fast"]?.asFloat() ?? Yarn.betaFast)
        let betaSlow = Double(ropeParameters["beta_slow"]?.asFloat() ?? 1)

        func mscale(_ scale: Double, _ mscale: Double = 1) -> Double {
            scale <= 1 ? 1 : Yarn.mscaleSlope * mscale * Foundation.log(scale) + 1
        }

        let attentionFactor: Double
        if let value = ropeParameters["attention_factor"]?.asFloat() {
            attentionFactor = Double(value)
        } else if let m = ropeParameters["mscale"]?.asFloat(),
            let mAll = ropeParameters["mscale_all_dim"]?.asFloat(), m != 0, mAll != 0
        {
            attentionFactor = mscale(factor, Double(m)) / mscale(factor, Double(mAll))
        } else {
            attentionFactor = mscale(factor)
        }

        func correctionDim(_ rotations: Double) -> Double {
            Double(dim) * Foundation.log(originalMax / (rotations * Yarn.fullTurn))
                / (Double(dimensionsPerFrequency) * Foundation.log(base))
        }
        var low = correctionDim(betaFast)
        var high = correctionDim(betaSlow)
        if case .bool(false) = ropeParameters["truncate"] {
            // HF keeps the fractional correction range when truncate is false.
        } else {
            low = low.rounded(.down)
            high = high.rounded(.up)
        }
        low = max(low, 0)
        high = min(high, Double(dim - 1))
        if low == high { high += Yarn.degenerateRangeWidth }

        let invFreq = (0 ..< dim / dimensionsPerFrequency).map { i -> Float in
            let posFreq = Foundation.pow(
                base, Double(dimensionsPerFrequency * i) / Double(dim))
            let ramp = min(max((Double(i) - low) / (high - low), 0), 1)
            let extrapolation = 1 - ramp
            return Float(
                (1 / (factor * posFreq)) * (1 - extrapolation) + (1 / posFreq) * extrapolation)
        }
        return (invFreq, Float(attentionFactor))
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    /// Tells if an attention mask mode holds no mask: a single-token step,
    /// which the compiled decode traces run.
    ///
    /// - Parameter mask: The attention mask mode of a call.
    /// - Returns: True for `.none`.
    static func isUnmasked(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        guard case .none = mask else { return false }
        return true
    }

    package final class RMSNormGated: Module, GatedDeltaOutputNorm {
        @ParameterInfo(key: "weight") package var weight: MLXArray
        package let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    package final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        /// The kernel of a decode token, or `nil` when it does not reproduce
        /// the norms and the rotation of this layer.
        private let decodeHeads: Qwen35DecodeHeadsKernel?

        /// The query, key and value projections as one matmul, once the
        /// loader prepares them.
        private let fusedQKV = FusedQuantizedProjectionGroup(keys: ["q_proj", "k_proj", "v_proj"])

        /// The place of the query projection among the q/k/v outputs.
        private static let queriesPart = 0
        /// The place of the key projection among the q/k/v outputs.
        private static let keysPart = 1
        /// The place of the value projection among the q/k/v outputs.
        private static let valuesPart = 2

        /// The halves of each head of the `q_proj` output: the query, then the
        /// output gate.
        private static let gatedQueryParts = 2
        /// The place of the query among the halves of a `q_proj` head.
        private static let queryPart = 0
        /// The place of the output gate among the halves of a `q_proj` head.
        private static let gatePart = 1

        /// The power of the head width that scales the attention scores.
        private static let scaleExponent: Float = -0.5

        /// The axes of the token-major layout `(batch, tokens, heads, headDim)`.
        private enum TokenMajorAxis {
            /// The sequences of the batch.
            static let batch = 0
            /// The tokens of a sequence.
            static let tokens = 1
            /// The heads of a token.
            static let heads = 2
            /// The numbers of a head.
            static let headDim = 3
        }

        /// The permutation between the token-major layout
        /// `(batch, tokens, heads, headDim)` and the head-major layout
        /// `(batch, heads, tokens, headDim)`; it is its own inverse.
        private static let headTokenSwap = [
            TokenMajorAxis.batch, TokenMajorAxis.heads, TokenMajorAxis.tokens,
            TokenMajorAxis.headDim,
        ]

        /// The axis of the tokens in the head-major layout.
        private static let headMajorTokenAxis = 2

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), Self.scaleExponent)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * Self.gatedQueryParts,
                bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope =
                args.ropeParameters?["mrope_section"]?.asInts()
                ?? Qwen35Configuration.TextConfiguration.Defaults.mropeSection
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope,
                ropeParameters: args.ropeParameters,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
            self.decodeHeads = Qwen35DecodeHeadsKernel(
                queryHeads: numAttentionHeads, kvHeads: numKeyValueHeads, headDim: headDim,
                rotaryDim: rotaryDim, queryEps: args.rmsNormEps, keyEps: args.rmsNormEps)
            super.init()
        }

        @discardableResult
        package override func update(
            parameters: ModuleParameters, verify: VerifyUpdate,
            path: [String] = [], modulePath: [String] = []
        ) throws -> Self {
            try fusedQKV.invalidating(afterUpdating: parameters) {
                try super.update(
                    parameters: parameters, verify: verify, path: path, modulePath: modulePath)
            }
        }

        package override func updateModule(key: String, _ value: Any) throws {
            try fusedQKV.invalidating(afterReplacing: key) {
                try super.updateModule(key: key, value)
            }
        }

        /// Whether one matmul computes the query, key and value projections.
        var hasFusedProjection: Bool { fusedQKV.isPrepared }

        /// Fuses the query, key and value projections; see
        /// ``FusedQuantizedProjectionGroup``. The model loader calls this
        /// before it publishes the model; a forward never does.
        ///
        /// - Returns: Whether one matmul computes the three projections.
        @discardableResult
        func prepareFusedProjection() throws -> Bool {
            try fusedQKV.prepare([qProj, kProj, vProj], in: self)
        }

        /// The fused projection as compile state of `trace`; see
        /// ``FusedQuantizedProjectionGroup/traceState(of:)``.
        func fusedProjectionTraceState(of trace: any CompiledTraceInvalidating) -> [Module] {
            fusedQKV.traceState(of: trace)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let projection = projectPreNorm(x)
            let attention = attend(
                queries: projection.queries,
                keys: projection.keys,
                values: projection.values,
                mask: mask,
                cache: cache,
                positionIds: positionIds)
            return mergeHeadsAndProject(attention: attention, gate: projection.gate)
        }

        /// The projections of `x` up to the query and key norms, which run
        /// outside the compiled traces with M-RoPE (see
        /// ``rotatedHeads(queries:keys:rotation:)``).
        ///
        /// - Parameter x: The normed layer input, shape `(batch, tokens, hiddenSize)`.
        /// - Returns: The `q_proj` output, which holds the query and the output
        ///   gate of each head, shape `(batch, tokens, heads * 2 * headDim)`;
        ///   the output gate, shape `(batch, tokens, heads * headDim)`; the
        ///   `k_proj` output, shape `(batch, tokens, kvHeads * headDim)`; and
        ///   the values, shape `(batch, kvHeads, tokens, headDim)`.
        func projectPreNorm(
            _ x: MLXArray
        ) -> (queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray) {
            let B = x.dim(0)
            let L = x.dim(1)

            let projected = fusedQKV(x, [qProj, kProj, vProj])
            let queries = projected[Self.queriesPart]
            let gate = gatedQueryHalves(queries)[Self.gatePart].reshaped(B, L, -1)
            let values = projected[Self.valuesPart].reshaped(B, L, numKeyValueHeads, -1)
                .transposed(axes: Self.headTokenSwap)

            return (queries, gate, projected[Self.keysPart], values)
        }

        /// The query and the output gate of each head of a `q_proj` output,
        /// each `(batch, tokens, heads, headDim)`.
        private func gatedQueryHalves(_ queries: MLXArray) -> [MLXArray] {
            queries.reshaped(queries.dim(0), queries.dim(1), numAttentionHeads, -1)
                .split(parts: Self.gatedQueryParts, axis: -1)
        }

        /// The op path of the query and key norms and the head-major layout.
        ///
        /// - Parameters:
        ///   - queries: The `q_proj` output from ``projectPreNorm(_:)``.
        ///   - keys: The `k_proj` output from ``projectPreNorm(_:)``.
        /// - Returns: The normed queries, shape `(batch, heads, tokens, headDim)`,
        ///   and the normed keys, shape `(batch, kvHeads, tokens, headDim)`.
        func normedHeads(
            queries: MLXArray, keys: MLXArray
        ) -> (queries: MLXArray, keys: MLXArray) {
            let (B, L) = (keys.dim(0), keys.dim(1))
            return (
                qNorm(gatedQueryHalves(queries)[Self.queryPart])
                    .transposed(axes: Self.headTokenSwap),
                kNorm(keys.reshaped(B, L, numKeyValueHeads, -1))
                    .transposed(axes: Self.headTokenSwap)
            )
        }

        /// The normed and rotated heads the attention reads. A decode token
        /// that ``Qwen35DecodeHeadsKernel`` reads takes the kernel, with the
        /// bits of the op path in one dispatch; every other block takes the op
        /// path: ``normedHeads(queries:keys:)``, then
        /// ``Qwen35Language/applyMultimodalRotaryPosEmb(q:k:cos:sin:)``.
        ///
        /// - Parameters:
        ///   - queries: The `q_proj` output from ``projectPreNorm(_:)``.
        ///   - keys: The `k_proj` output from ``projectPreNorm(_:)``.
        ///   - rotation: The M-RoPE rotation of the tokens.
        /// - Returns: The queries, shape `(batch, heads, tokens, headDim)`, and
        ///   the keys, shape `(batch, kvHeads, tokens, headDim)`.
        func rotatedHeads(
            queries: MLXArray, keys: MLXArray, rotation: RotaryValues
        ) -> (queries: MLXArray, keys: MLXArray) {
            if let decodeHeads,
                let heads = decodeHeads.heads(
                    queryProjection: queries, keyProjection: keys, queryWeight: qNorm.weight,
                    keyWeight: kNorm.weight, rotation: rotation)
            {
                return heads
            }
            let normed = normedHeads(queries: queries, keys: keys)
            let (rotatedQueries, rotatedKeys) = applyMultimodalRotaryPosEmb(
                q: normed.queries, k: normed.keys, cos: rotation.cos, sin: rotation.sin)
            return (rotatedQueries, rotatedKeys)
        }

        /// Dynamic M-RoPE, cache mutation, and SDPA stay eager. The attention
        /// takes the route of the text model,
        /// ``PrefillAttentionKernel/attentionWithCacheUpdate(queries:keys:values:cache:scale:mask:)``:
        /// over a plain key cache, the causal mask mode or no mask reaches the
        /// prefill, verify and decode attention kernels, and a mask array
        /// keeps the SDPA of MLX.
        ///
        /// The attention layers of a model share their rotation tables, so a
        /// step can compute the rotation of its tokens once and give it to
        /// each layer.
        ///
        /// - Parameters:
        ///   - queryProjection: The `q_proj` output from ``projectPreNorm(_:)``,
        ///     shape `(batch, tokens, heads * 2 * headDim)`.
        ///   - keyProjection: The `k_proj` output from ``projectPreNorm(_:)``,
        ///     shape `(batch, tokens, kvHeads * headDim)`.
        ///   - values: The new values, shape `(batch, kvHeads, tokens, headDim)`.
        ///   - mask: The attention mask of the call.
        ///   - cache: The cache of this layer, or `nil`.
        ///   - providedPositionIds: The M-RoPE positions of the tokens, or
        ///     `nil` for the positions after the cache.
        ///   - rotary: The rotation of `providedPositionIds`, or `nil` to
        ///     compute it here.
        /// - Returns: The attention output, shape `(batch, heads, tokens, headDim)`.
        func attend(
            queries queryProjection: MLXArray,
            keys keyProjection: MLXArray,
            values: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIds providedPositionIds: MLXArray?,
            rotary: RotaryValues? = nil
        ) -> MLXArray {
            let B = keyProjection.dim(0)
            let L = keyProjection.dim(1)

            let positionIds =
                providedPositionIds
                ?? Self.positionsAfterCache(batch: B, tokens: L, cache: cache)

            let (queries, keys) = rotatedHeads(
                queries: queryProjection, keys: keyProjection,
                rotation: rotary ?? rotaryEmbedding(x: values, positionIds: positionIds))

            // The mask is sized for the keys the cache presents; slicing it to
            // an offset-derived length would misalign it after eviction.
            let attentionMask = mask

            return PrefillAttentionKernel.attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: attentionMask)
        }

        /// The M-RoPE positions of the tokens after the cache, the same on
        /// every axis.
        ///
        /// - Parameters:
        ///   - batch: The sequences of the batch.
        ///   - tokens: The tokens of each sequence.
        ///   - cache: The cache of the layer, or `nil` for positions from zero.
        /// - Returns: The positions, shape `(axes, batch, tokens)`.
        private static func positionsAfterCache(
            batch: Int, tokens: Int, cache: KVCache?
        ) -> MLXArray {
            let offset = cache?.offset ?? 0
            var base = MLXArray(stride(from: offset, to: offset + tokens, by: 1)).asType(.int32)
            base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
            return tiled(base[.newAxis, 0..., 0...], repetitions: [MRopeAxis.count, 1, 1])
        }

        func mergeHeadsAndProject(attention: MLXArray, gate: MLXArray) -> MLXArray {
            let output =
                attention
                .transposed(axes: Self.headTokenSwap)
                .reshaped(attention.dim(0), attention.dim(Self.headMajorTokenAxis), -1)
            return MultiRowQuantizedMatvec.projection(output * sigmoid(gate), by: oProj)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        /// The gate and up projections as one matmul, once the loader
        /// prepares them.
        private let fusedGateUp = FusedQuantizedProjectionGroup(keys: ["gate_proj", "up_proj"])

        /// The place of the gate projection among the gate/up outputs.
        private static let gatePart = 0
        /// The place of the up projection among the gate/up outputs.
        private static let upPart = 1

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        @discardableResult
        override func update(
            parameters: ModuleParameters, verify: VerifyUpdate,
            path: [String] = [], modulePath: [String] = []
        ) throws -> Self {
            try fusedGateUp.invalidating(afterUpdating: parameters) {
                try super.update(
                    parameters: parameters, verify: verify, path: path, modulePath: modulePath)
            }
        }

        override func updateModule(key: String, _ value: Any) throws {
            try fusedGateUp.invalidating(afterReplacing: key) {
                try super.updateModule(key: key, value)
            }
        }

        /// Whether one matmul computes the gate and up projections.
        var hasFusedProjection: Bool { fusedGateUp.isPrepared }

        /// Fuses the gate and up projections; see
        /// ``FusedQuantizedProjectionGroup``. The model loader calls this
        /// before it publishes the model; a forward never does.
        ///
        /// - Returns: Whether one matmul computes the two projections.
        @discardableResult
        func prepareFusedProjection() throws -> Bool {
            try fusedGateUp.prepare([gateProj, upProj], in: self)
        }

        /// The fused projection as compile state of `trace`; see
        /// ``FusedQuantizedProjectionGroup/traceState(of:)``.
        func fusedProjectionTraceState(of trace: any CompiledTraceInvalidating) -> [Module] {
            fusedGateUp.traceState(of: trace)
        }

        /// The MLP of `x`. Each projection runs through
        /// ``MultiRowQuantizedMatvec/projection(_:by:)``, so inside a
        /// speculative verify each token row keeps the bits of a decode step.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = fusedGateUp(x, [gateProj, upProj])
            return MultiRowQuantizedMatvec.projection(
                silu(projected[Self.gatePart]) * projected[Self.upPart], by: downProj)
        }
    }

    open class GatedDeltaNet: Module, GatedDeltaDecodeLayer {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        package let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") package var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        // Inference-only physical projection. The four registered modules
        // remain as views so checkpoint and adapter paths stay unchanged.
        private let fusedInputProjection = FusedQuantizedLinearProjectionCache()
        var fusedInputProjectionEnabled = qwen35FourGDNEnabled

        private struct ProjectionTrace {
            weak var value: (any CompiledTraceInvalidating)?
        }
        // Submodule updates must also drop traces owned by parent decoders.
        private let fusedProjectionTracesLock = NSLock()
        private var fusedProjectionTraces: [ObjectIdentifier: ProjectionTrace] = [:]

        @ParameterInfo(key: "dt_bias") package var dtBias: MLXArray
        @ParameterInfo(key: "A_log") package var aLog: MLXArray

        @ModuleInfo(key: "norm") package var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        /// The parts of the mixed qkv that each take `keyDim` channels: the
        /// query and the key.
        private static let keyDimParts = 2

        /// The places of the query, the key and the value among the parts
        /// of the convolved qkv.
        private enum MixedQKVPart {
            /// The query.
            static let query = 0
            /// The key.
            static let key = 1
            /// The value.
            static let value = 2
        }

        /// The places of the four input projections among the views of the
        /// fused input projection.
        private enum InputProjectionPart {
            /// `in_proj_qkv`.
            static let qkv = 0
            /// `in_proj_z`.
            static let z = 1
            /// `in_proj_b`.
            static let b = 2
            /// `in_proj_a`.
            static let a = 3
        }

        /// The upper bound of the uniform draw of `A` that `A_log` starts
        /// from before a checkpoint replaces it.
        private static let initialDecayHigh = 16

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * Self.keyDimParts + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, convDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: Self.initialDecayHigh, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        @discardableResult
        open override func update(
            parameters: ModuleParameters, verify: VerifyUpdate,
            path: [String] = [], modulePath: [String] = []
        ) throws -> Self {
            let inputProjectionPrefixes = [
                "in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a.",
            ]
            let replacesInputProjection = parameters.flattened().contains { key, _ in
                inputProjectionPrefixes.contains(where: key.hasPrefix)
            }
            defer {
                if replacesInputProjection {
                    invalidateFusedInputProjection()
                }
            }
            return try super.update(
                parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        }

        open override func updateModule(key: String, _ value: Any) throws {
            let replacesInputProjection =
                key == "in_proj_qkv" || key == "in_proj_z"
                || key == "in_proj_b" || key == "in_proj_a"
            defer {
                if replacesInputProjection {
                    invalidateFusedInputProjection()
                }
            }
            try super.updateModule(key: key, value)
        }

        var hasFusedInputProjection: Bool { fusedInputProjection.isPrepared }

        func registerFusedProjectionTrace(_ trace: any CompiledTraceInvalidating) {
            fusedProjectionTracesLock.withLock {
                fusedProjectionTraces[ObjectIdentifier(trace)] = ProjectionTrace(value: trace)
            }
        }

        private func invalidateFusedInputProjection() {
            fusedInputProjection.invalidate()
            let traces = fusedProjectionTracesLock.withLock {
                let traces = fusedProjectionTraces.values.compactMap(\.value)
                fusedProjectionTraces.removeAll()
                return traces
            }
            // Trace setup takes its own lock first, so invalidate after unlocking.
            for trace in traces {
                trace.invalidate()
            }
        }

        /// The fused projection as compile state for a trace that runs this
        /// layer. It is not a registered child because the four source
        /// projections retain the checkpoint topology.
        var fusedProjectionTraceState: [Module] {
            fusedInputProjection.fused.map { [$0] } ?? []
        }

        @discardableResult
        func prepareFusedInputProjection() throws -> Bool {
            try fusedInputProjection.prepare(
                enabled: fusedInputProjectionEnabled,
                linears: [
                    inProjQKV, inProjZ, inProjB, inProjA,
                ]
            ) { sourceViews in
                try update(
                    modules: ModuleChildren(values: [
                        "in_proj_qkv": .value(sourceViews[InputProjectionPart.qkv]),
                        "in_proj_z": .value(sourceViews[InputProjectionPart.z]),
                        "in_proj_b": .value(sourceViews[InputProjectionPart.b]),
                        "in_proj_a": .value(sourceViews[InputProjectionPart.a]),
                    ]), verify: [])
            }
        }

        func projectInputs(_ inputs: MLXArray, batch: Int, sequence: Int) -> (
            qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
        ) {
            guard fusedInputProjectionEnabled, let fusedInProj = fusedInputProjection.fused else {
                return (
                    MultiRowQuantizedMatvec.projection(inputs, by: inProjQKV),
                    MultiRowQuantizedMatvec.projection(inputs, by: inProjZ).reshaped(
                        batch, sequence, numVHeads, headVDim),
                    MultiRowQuantizedMatvec.projection(inputs, by: inProjB),
                    MultiRowQuantizedMatvec.projection(inputs, by: inProjA)
                )
            }

            let projected = MultiRowQuantizedMatvec.projection(inputs, by: fusedInProj)
            let qkvEnd = convDim
            let zEnd = qkvEnd + valueDim
            let bEnd = zEnd + numVHeads
            let aEnd = bEnd + numVHeads
            return (
                projected[0..., 0..., ..<qkvEnd],
                projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                    batch, sequence, numVHeads, headVDim),
                projected[0..., 0..., zEnd ..< bEnd],
                projected[0..., 0..., bEnd ..< aEnd]
            )
        }

        open func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil,
            keepsSnapshots: Bool = false
        ) -> MLXArray {
            let convState =
                cache?[LinearCacheSlot.convState]
                ?? zeroStates(batch: inputs.dim(0), dtype: inputs.dtype).conv
            let result = forward(
                inputs, convState: convState, recState: cache?[LinearCacheSlot.recurrentState],
                mask: mask,
                keepsSnapshots: keepsSnapshots)
            if let cache {
                if convKernelSize > 1 {
                    cache[LinearCacheSlot.convState] = result.convState
                }
                cache[LinearCacheSlot.recurrentState] = result.recurrentState
                if let snapshots = result.snapshots {
                    cache.saveSpeculativeSnapshots(snapshots)
                }
                cache.advance(inputs.dim(1))
            }
            return result.output
        }

        func zeroStates(batch: Int, dtype: DType) -> (conv: MLXArray, rec: MLXArray) {
            (
                MLXArray.zeros(
                    [batch, max(0, convKernelSize - 1), convDim], dtype: dtype),
                MLXArray.zeros(
                    [batch, numVHeads, headVDim, headKDim], dtype: .float32)
            )
        }

        /// The GDN body with cache state passed explicitly for compiled decode.
        ///
        /// A verify pass (`keepsSnapshots` over two tokens or more) keeps the
        /// states after each token: one kernel runs a verify of up to
        /// ``GatedDeltaDecodeKernel/mostTokens`` tokens, and the op path runs
        /// any other verify one token at a time.
        func forward(
            _ x: MLXArray,
            convState: MLXArray,
            recState: MLXArray?,
            mask: MLXArray?,
            keepsSnapshots: Bool = false
        ) -> (
            output: MLXArray,
            convState: MLXArray,
            recurrentState: MLXArray,
            snapshots: GatedDeltaDecodeKernel.Snapshots?
        ) {
            let B = x.dim(0)
            let S = x.dim(1)

            var (mixedQKV, z, b, a) = projectInputs(x, batch: B, sequence: S)
            let keepsSnapshots = keepsSnapshots && S > GatedDeltaDecodeKernel.decodeTokens
            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            } else if let step = decodeKernelStep(
                qkv: mixedQKV, z: z, b: b, a: a, convState: convState, recState: recState)
            {
                return (projectOutput(step.output), step.convState, step.recurrentState, nil)
            } else if keepsSnapshots,
                let step = speculativeKernelStep(
                    qkv: mixedQKV, z: z, b: b, a: a, convState: convState, recState: recState),
                let snapshots = step.snapshots
            {
                return (
                    projectOutput(step.output), step.convState, step.recurrentState, snapshots
                )
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            let newConvState: MLXArray
            if convKernelSize > 1 {
                newConvState = contiguous(
                    convInput[0..., (-(convKernelSize - 1))..., 0...])
            } else {
                newConvState = MLXArray.zeros([B, 0, convDim], dtype: mixedQKV.dtype)
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(
                convOut, indices: [keyDim, Self.keyDimParts * keyDim], axis: -1)
            let q = split[MixedQKVPart.query].reshaped(B, S, numKHeads, headKDim)
            let k = split[MixedQKVPart.key].reshaped(B, S, numKHeads, headKDim)
            let v = split[MixedQKVPart.value].reshaped(B, S, numVHeads, headVDim)

            let (qNormed, kNormed) = GatedDeltaQueryKeyNorm.normalized(
                query: q, key: k, dtype: q.dtype)

            let out: MLXArray
            let newRecState: MLXArray
            let snapshots: GatedDeltaDecodeKernel.Snapshots?
            if keepsSnapshots {
                (out, newRecState, snapshots) = GatedDeltaDecodeKernel.Snapshots.gatedDeltaUpdate(
                    q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                    state: recState, mask: mask, useKernel: !training, convWindow: convInput)
            } else {
                (out, newRecState) = gatedDeltaUpdate(
                    q: qNormed,
                    k: kNormed,
                    v: v,
                    a: a,
                    b: b,
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recState,
                    mask: mask,
                    useKernel: !training)
                snapshots = nil
            }

            let gated = norm(out, gate: z)
            return (projectOutput(gated.reshaped(B, S, -1)), newConvState, newRecState, snapshots)
        }

        /// The output projection of the gated delta net, through
        /// ``MultiRowQuantizedMatvec/projection(_:by:)``: inside a
        /// speculative verify each token row keeps the bits of a decode step.
        ///
        /// - Parameter x: The gated output, shape `(batch, tokens, valueDim)`.
        /// - Returns: The layer output, shape `(batch, tokens, hiddenSize)`.
        func projectOutput(_ x: MLXArray) -> MLXArray {
            MultiRowQuantizedMatvec.projection(x, by: outProj)
        }
    }

    open class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        open func callAsFunction(_ x: MLXArray) -> MLXArray {
            guard x.dim(Qwen35Language.tokenAxis) == GatedDeltaDecodeKernel.decodeTokens,
                !training, supportsExactCompiledDecode(x.dtype),
                supportsCompiledDecode
            else {
                return forward(x)
            }
            return compiledForward(self, x)
        }

        private let compiledForward = CompiledTrace<SparseMoeBlock> { block, arguments in
            [block.forward(arguments[SparseMoeBlock.inputArgument])]
        }

        /// The argument of the compiled forward that holds the block input.
        private static let inputArgument = 0

        var compiledDecodeIsCompiled: Bool { compiledForward.isCompiled }

        var supportsCompiledDecode: Bool {
            type(of: self) == SparseMoeBlock.self && type(of: switchMLP) == SwitchGLU.self
        }

        /// The eager body, also inlined by an enclosing decoder trace.
        func forward(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let (inds, scores) = moeRouterTopK(
                gates, k: topK, normalize: normTopkProb)

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores)

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    open class DecoderLayer: Module, FusedProjectionDecodeLayer {
        let isLinear: Bool
        let allowsCompiledDecode: Bool

        /// The attention; non-nil exactly for a full-attention layer.
        @ModuleInfo(key: "self_attn") package var selfAttn: Attention?
        /// The gated delta net; non-nil exactly for a linear layer.
        @ModuleInfo(key: "linear_attn") package var linearAttn: GatedDeltaNet?

        /// The RMSNorm before the gated delta net or the attention.
        @ModuleInfo(key: "input_layernorm") package var inputLayerNorm: RMSNorm
        /// The RMSNorm before the feed-forward block.
        @ModuleInfo(key: "post_attention_layernorm") package var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer

        public init(
            _ args: Qwen35Configuration.TextConfiguration, layerIdx: Int,
            forceFullAttention: Bool = false
        ) {
            self.allowsCompiledDecode = !forceFullAttention
            self.isLinear =
                forceFullAttention
                ? false
                : args.resolvedLayerTypes[layerIdx] == HybridAttentionSchedule.linearAttention

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        open func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            keepsSnapshots: Bool = false
        ) -> MLXArray {
            forward(
                x,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: cache,
                positionIds: positionIds,
                keepsSnapshots: keepsSnapshots,
                allowCompiledDecode: allowsCompiledDecode)
        }

        func forward(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            keepsSnapshots: Bool,
            allowCompiledDecode: Bool
        ) -> MLXArray {
            let canCompile =
                allowCompiledDecode && allowsCompiledDecode && !training
                && supportsExactCompiledDecode(x.dtype)
                && x.dim(Qwen35Language.batchAxis) == Qwen35Language.compiledBatchSize
                && x.dim(Qwen35Language.tokenAxis) == GatedDeltaDecodeKernel.decodeTokens
                && Qwen35Language.isUnmasked(attentionMask) && ssmMask == nil && !keepsSnapshots
                && supportsCompiledDecode

            if canCompile, isLinear, let mambaCache = cache as? MambaCache,
                let convState = mambaCache[LinearCacheSlot.convState],
                let recState = mambaCache[LinearCacheSlot.recurrentState]
            {
                return decodeLinearLayer(
                    x, convState: convState, recState: recState, cache: mambaCache)
            }

            let r: MLXArray
            if isLinear {
                r = gatedDeltaNet(
                    inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                    keepsSnapshots: keepsSnapshots)
            } else {
                r = fullAttention(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h
                + mlpForward(
                    postAttentionLayerNorm(h), allowCompiledDecode: canCompile)
        }

        private let compiledLinearLayer: CompiledTrace<DecoderLayer> = CompiledTrace(
            layerTrace: \.compiledLinearLayer,
            body: { layer, arguments in
                let result = layer.linearLayerBody(
                    x: arguments[LinearTraceSlot.hidden],
                    convState: arguments[LinearTraceSlot.convState],
                    recState: arguments[LinearTraceSlot.recurrentState])
                return [result.output, result.convState, result.recurrentState]
            })

        var compiledGDNIsCompiled: Bool { compiledLinearLayer.isCompiled }

        /// Fuses the projections of this layer that read one input: the four
        /// input projections of the gated delta net, the query, key and value
        /// projections of the attention, and the gate and up projections of a
        /// dense MLP. The model loader calls this before it publishes the
        /// model.
        func prepareFusedProjections() throws {
            _ = try linearAttn?.prepareFusedInputProjection()
            _ = try selfAttn?.prepareFusedProjection()
            _ = try (mlp as? MLP)?.prepareFusedProjection()
        }

        /// The fused projections of this layer as compile state of `trace`,
        /// which each one drops when its fusion turns off.
        ///
        /// - Parameter trace: The trace that runs this layer.
        /// - Returns: The fused projections the layer holds.
        package func fusedProjectionTraceState(of trace: any CompiledTraceInvalidating) -> [Module]
        {
            linearAttn?.registerFusedProjectionTrace(trace)
            return (linearAttn?.fusedProjectionTraceState ?? [])
                + (selfAttn?.fusedProjectionTraceState(of: trace) ?? [])
                + ((mlp as? MLP)?.fusedProjectionTraceState(of: trace) ?? [])
        }

        var supportsCompiledDecode: Bool {
            guard type(of: self) == DecoderLayer.self, !training else { return false }
            if isLinear {
                guard let linearAttn, type(of: linearAttn) == GatedDeltaNet.self,
                    !linearAttn.training
                else {
                    return false
                }
            } else if selfAttn == nil {
                return false
            }

            if let moe = mlp as? SparseMoeBlock {
                return moe.supportsCompiledDecode
            }
            return mlp is MLP
        }

        /// GDN decode layer as one traced function, from the states the cache
        /// holds; it writes the states it gives back into the cache.
        private func decodeLinearLayer(
            _ x: MLXArray, convState: MLXArray, recState: MLXArray, cache: MambaCache
        ) -> MLXArray {
            let outputs = compiledLinearLayer(self, [x, convState, recState])
            cache[LinearCacheSlot.convState] = outputs[LinearTraceSlot.convState]
            cache[LinearCacheSlot.recurrentState] = outputs[LinearTraceSlot.recurrentState]
            cache.advance(GatedDeltaDecodeKernel.decodeTokens)
            return outputs[LinearTraceSlot.hidden]
        }

        func attentionCacheStep(
            queries: MLXArray,
            keys: MLXArray,
            values: MLXArray,
            cache: KVCache,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            positionIds: MLXArray?,
            rotary: RotaryValues? = nil
        ) -> MLXArray {
            fullAttention.attend(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                cache: cache,
                positionIds: positionIds,
                rotary: rotary)
        }

        // MARK: - Layer blocks

        /// The gated delta net with its states passed in and out; the shared
        /// ``HybridDecodeLayer/linearLayerBody(x:convState:recState:keepsSnapshots:)``
        /// wraps it in the layer's norms and residuals.
        package func gatedDeltaMixer(
            _ x: MLXArray, convState: MLXArray, recState: MLXArray, keepsSnapshots: Bool
        ) -> GatedDeltaLayerStep {
            let step = gatedDeltaNet.forward(
                x, convState: convState, recState: recState, mask: nil,
                keepsSnapshots: keepsSnapshots)
            return GatedDeltaLayerStep(
                output: step.output, convState: step.convState,
                recurrentState: step.recurrentState, snapshots: step.snapshots)
        }

        /// The projections of the attention before its query and key norms,
        /// which ``Attention/attend(queries:keys:values:mask:cache:positionIds:rotary:)``
        /// runs outside the compiled traces with M-RoPE; see
        /// ``Attention/projectPreNorm(_:)``.
        package func projectAttention(_ x: MLXArray) -> (
            queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray
        ) {
            fullAttention.projectPreNorm(x)
        }

        package func mergeAttention(_ attention: MLXArray, gate: MLXArray) -> MLXArray {
            fullAttention.mergeHeadsAndProject(attention: attention, gate: gate)
        }

        /// The MLP, or the MoE block in its form without its own compiled
        /// trace, so it can live inside a decode trace.
        package func feedForward(_ x: MLXArray) -> MLXArray {
            mlpForward(x, allowCompiledDecode: false)
        }

        /// The feed-forward block; a MoE block of the exact base type runs
        /// its own compiled trace when `allowCompiledDecode` is true.
        private func mlpForward(_ x: MLXArray, allowCompiledDecode: Bool) -> MLXArray {
            if let moe = mlp as? SparseMoeBlock, type(of: moe) == SparseMoeBlock.self {
                return allowCompiledDecode ? moe(x) : moe.forward(x)
            }
            return mlp(x)
        }
    }

    open class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int?
        let faIdx: Int?

        /// The traced pieces of a decode step and the runner over them.
        private let decodeSchedule: CompiledDecodeSchedule
        private let compiledSegments: CompiledDecodeSegmentCache<Model>

        /// The same segments for a speculative verify, one schedule for each
        /// ``CompiledVerifyShape``: the segments run every token of the verify,
        /// and each GDN layer also gives back the states it keeps for a
        /// rollback.
        private let compiledVerifySegments:
            KeyedCompiledDecodeSegmentCaches<Model, CompiledVerifyShape>

        /// The argument of a segment that carries the hidden states.
        private static let carryArgument = 0

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            let layers = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _layers.wrappedValue = layers
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.decodeSchedule = CompiledDecodeSchedule(layers: layers)
            self.ssmIdx = decodeSchedule.firstLinearLayer
            self.faIdx = decodeSchedule.firstAttentionLayer

            let segmentCount = decodeSchedule.segments.count
            self.compiledSegments = CompiledDecodeSegmentCache(
                count: segmentCount,
                state: Model.segmentTraceState(of: { $0.compiledSegments }),
                body: { model, index, arguments in
                    model.segmentBody(at: index, arguments, verify: nil)
                })
            self.compiledVerifySegments = KeyedCompiledDecodeSegmentCaches(
                count: segmentCount,
                state: Model.segmentTraceState(of: { $0.compiledVerifySegments }),
                body: { model, shape, index, arguments in
                    model.segmentBody(at: index, arguments, verify: shape)
                })
            super.init()
        }

        /// The hidden states of the tokens, with optional final normalization.
        ///
        /// A decode step with the final norm, and a speculative verify of
        /// ``CompiledVerifyShape/tokenRange`` tokens that keeps its snapshots,
        /// run through a compiled schedule; every other call takes the general
        /// path of the layers.
        ///
        /// - Parameters:
        ///   - inputs: The token ids, shape `(batch, tokens)`.
        ///   - inputsEmbeds: The embedded tokens, in place of the embedding of
        ///     `inputs`.
        ///   - cache: The cache of each layer, or nil.
        ///   - positionIds: The M-RoPE positions of the tokens, or nil for the
        ///     positions after the cache.
        ///   - applyFinalNorm: True to apply the final norm to the hidden states.
        ///   - keepsSnapshots: For a verify, true to have the GDN layers keep
        ///     the states after each token for a rollback.
        /// - Returns: The hidden states, shape `(batch, tokens, hiddenSize)`.
        open func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            applyFinalNorm: Bool = true,
            keepsSnapshots: Bool = false
        ) -> MLXArray {
            forward(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds,
                applyFinalNorm: applyFinalNorm,
                keepsSnapshots: keepsSnapshots,
                useCompiledDecode: true)
        }

        /// ``callAsFunction(_:inputsEmbeds:cache:positionIds:applyFinalNorm:keepsSnapshots:)``,
        /// where `useCompiledDecode` false takes the general path of the layers
        /// for every call, as a test of the compiled schedules compares the two.
        func forward(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            applyFinalNorm: Bool = true,
            keepsSnapshots: Bool = false,
            useCompiledDecode: Bool = true
        ) -> MLXArray {
            if useCompiledDecode, type(of: self) == Model.self, !training,
                inputsEmbeds == nil,
                inputs.dim(Qwen35Language.batchAxis) == Qwen35Language.compiledBatchSize,
                let cache,
                let hiddenStates = compiledStep(
                    inputs, cache, positionIds: positionIds, applyFinalNorm: applyFinalNorm,
                    keepsSnapshots: keepsSnapshots)
            {
                return hiddenStates
            }

            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMask =
                faIdx.map {
                    createAttentionMask(h: hiddenStates, cache: cacheArray?[$0])
                } ?? .none
            let ssmMask = ssmIdx.flatMap {
                createSSMMask(h: hiddenStates, cache: cacheArray?[$0] as? MambaCache)
            }

            let allowLayerCompilation =
                useCompiledDecode && type(of: self) == Model.self && !training
                && inputsEmbeds == nil && applyFinalNorm
                && supportsExactCompiledDecode(hiddenStates.dtype)
                && hiddenStates.dim(Qwen35Language.batchAxis) == Qwen35Language.compiledBatchSize
                && hiddenStates.dim(Qwen35Language.tokenAxis) == GatedDeltaDecodeKernel.decodeTokens
                && Qwen35Language.isUnmasked(faMask) && ssmMask == nil && !keepsSnapshots
                && layers.enumerated().allSatisfy { index, layer in
                    guard layer.supportsCompiledDecode else { return false }
                    if layer.isLinear {
                        guard let mambaCache = cacheArray?[index] as? MambaCache else {
                            return false
                        }
                        return mambaCache[LinearCacheSlot.convState] != nil
                            && mambaCache[LinearCacheSlot.recurrentState] != nil
                    }
                    guard let kvCache = cacheArray?[index] else { return false }
                    return kvCache is QuantizedKVCacheProtocol
                        || kvCache is TurboQuantKVCache
                }

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                if type(of: layer) == DecoderLayer.self {
                    hiddenStates = layer.forward(
                        hiddenStates,
                        attentionMask: faMask,
                        ssmMask: layerSSMMask,
                        cache: cacheArray?[index],
                        positionIds: positionIds,
                        keepsSnapshots: keepsSnapshots,
                        allowCompiledDecode: allowLayerCompilation)
                } else {
                    hiddenStates = layer(
                        hiddenStates,
                        attentionMask: faMask,
                        ssmMask: layerSSMMask,
                        cache: cacheArray?[index],
                        positionIds: positionIds,
                        keepsSnapshots: keepsSnapshots)
                }
            }

            return applyFinalNorm ? norm(hiddenStates) : hiddenStates
        }

        var compiledDecodeSegmentCount: Int { compiledSegments.compiledCount }

        /// The compiled segment traces of every verify shape a step has run.
        var compiledVerifySegmentCount: Int { compiledVerifySegments.compiledCount }

        /// Fuses the projections of each layer that read one input; see
        /// ``DecoderLayer/prepareFusedProjections()``.
        func prepareFusedProjections() throws {
            for layer in layers {
                try layer.prepareFusedProjections()
            }
        }

        var compiledGDNTraceCount: Int {
            layers.filter(\.compiledGDNIsCompiled).count
        }

        var compiledMoETraceCount: Int {
            layers.compactMap { $0.mlp as? SparseMoeBlock }
                .filter(\.compiledDecodeIsCompiled).count
        }

        /// The compile state provider of the segment traces `traces` gives:
        /// ``segmentTraceState(at:of:)`` for each segment, with those traces.
        ///
        /// - Parameter traces: The traces of a model that read the state.
        /// - Returns: The state provider of those traces.
        private static func segmentTraceState(
            of traces: @escaping @Sendable (Model) -> any CompiledTraceInvalidating
        ) -> CompiledDecodeSegmentCache<Model>.StateProvider {
            { model, index in model.segmentTraceState(at: index, of: traces(model)) }
        }

        /// The compile state of segment `index`: everything its body reads,
        /// the layers it runs (with the fused projections of each layer) and
        /// the final norm it ends with. Each fused projection drops `trace`
        /// when it goes stale.
        ///
        /// - Parameters:
        ///   - index: The segment.
        ///   - trace: The traces that read this state.
        /// - Returns: The modules whose weights the segment reads.
        private func segmentTraceState(
            at index: Int, of trace: any CompiledTraceInvalidating
        ) -> [Module] {
            let segments = decodeSchedule.segments
            var modules = segments[index].layerIndices.flatMap {
                layers[$0].layerTraceState(of: trace)
            }
            if index == segments.indices.last {
                modules.append(norm)
            }
            return modules
        }

        /// Runs segment `index` of a step through the shared
        /// ``CompiledDecodeSegment/body(_:arguments:layers:finalNorm:verify:)``.
        /// The embedding runs outside the trace, so every segment starts from
        /// its first argument. A decode step ends with the final norm; a verify
        /// leaves it to its caller, which emits the rows before it.
        private func segmentBody(
            at index: Int, _ arguments: [MLXArray], verify: CompiledVerifyShape?
        ) -> [MLXArray] {
            decodeSchedule.segments[index].body(
                arguments[Self.carryArgument], arguments: arguments, layers: layers,
                finalNorm: norm, verify: verify)
        }

        /// One step through a compiled schedule, or nil when the step takes the
        /// general path. The schedules read a decode step with the final norm
        /// and a speculative verify of ``CompiledVerifyShape/tokenRange`` tokens
        /// that keeps its snapshots.
        private func compiledStep(
            _ inputs: MLXArray, _ cache: [KVCache?], positionIds: MLXArray?,
            applyFinalNorm: Bool, keepsSnapshots: Bool
        ) -> MLXArray? {
            let tokens = inputs.dim(Qwen35Language.tokenAxis)
            if tokens == GatedDeltaDecodeKernel.decodeTokens {
                guard applyFinalNorm, !keepsSnapshots else { return nil }
                return decodeStep(inputs, cache, positionIds: positionIds)
            }
            guard keepsSnapshots, let shape = CompiledVerifyShape(tokens: tokens),
                let hiddenStates = verifyStep(inputs, cache, positionIds: positionIds, shape: shape)
            else { return nil }
            return applyFinalNorm ? norm(hiddenStates) : hiddenStates
        }

        /// The embedded tokens of a compiled step, or nil when the step takes
        /// the general path: a dtype the traces do not hold exactly, or a layer
        /// that does not support compiled decode.
        private func compiledStepInput(_ inputs: MLXArray) -> MLXArray? {
            let embedded = embedTokens(inputs)
            guard supportsExactCompiledDecode(embedded.dtype),
                layers.allSatisfy(\.supportsCompiledDecode)
            else { return nil }
            return embedded
        }

        /// One decode step through the compiled segments, or nil when the step
        /// takes the general path: ``compiledStepInput(_:)`` or the schedule
        /// refuses it (``CompiledDecodeSchedule/decode(_:cache:segments:of:attention:)``).
        private func decodeStep(
            _ inputs: MLXArray, _ cache: [KVCache?], positionIds: MLXArray?
        ) -> MLXArray? {
            guard let embedded = compiledStepInput(inputs) else { return nil }
            return decodeSchedule.decode(
                embedded, cache: cache, segments: compiledSegments, of: self,
                attention: sharedRotationAttention(mask: .none, positionIds: positionIds))
        }

        /// One speculative verify through the compiled segments of its shape, or
        /// nil when the step takes the general path: ``compiledStepInput(_:)`` or
        /// ``CompiledDecodeSchedule/caches(_:over:)`` refuses it. The hidden rows
        /// come back before the final norm.
        ///
        /// Each attention layer runs its queries over the cache with the mask of
        /// the general path, so M-RoPE, the KV write and the attention kernels
        /// stay outside the traces; see ``sharedRotationAttention(mask:positionIds:)``.
        private func verifyStep(
            _ inputs: MLXArray, _ cache: [KVCache?], positionIds: MLXArray?,
            shape: CompiledVerifyShape
        ) -> MLXArray? {
            guard let embedded = compiledStepInput(inputs),
                let caches = decodeSchedule.caches(cache, over: embedded)
            else { return nil }
            return decodeSchedule.run(
                embedded, caches: caches, verify: shape,
                segment: { index, arguments in
                    compiledVerifySegments(self, key: shape, at: index, arguments)
                },
                attention: sharedRotationAttention(
                    mask: decodeSchedule.attentionMask(embedded, cache), positionIds: positionIds))
        }

        /// The attention of each full-attention layer of one compiled step,
        /// through
        /// ``DecoderLayer/attentionCacheStep(queries:keys:values:cache:mask:positionIds:rotary:)``.
        /// The attention layers of a model share their rotation tables and
        /// rotate the same tokens, so the first layer of the step computes the
        /// M-RoPE rotation of `positionIds` and every later layer reuses it.
        /// Without `positionIds` each layer rotates the positions after its
        /// cache.
        ///
        /// The step holds the layers of the model when the step starts, and
        /// no reference to the model, so it never reads a model that is gone.
        ///
        /// - Parameters:
        ///   - mask: The attention mask of the step.
        ///   - positionIds: The M-RoPE positions of the tokens of the step, or
        ///     `nil` for the positions after the cache.
        /// - Returns: The attention step of the schedule, for one step only.
        private func sharedRotationAttention(
            mask: MLXFast.ScaledDotProductAttentionMaskMode, positionIds: MLXArray?
        ) -> (CompiledDecodeSchedule.AttentionStep) -> MLXArray {
            let layers = layers
            var rotary: RotaryValues?
            return { step in
                let layer = layers[step.layer]
                rotary =
                    rotary
                    ?? positionIds.map {
                        layer.fullAttention.rotaryEmbedding(x: step.values, positionIds: $0)
                    }
                return layer.attentionCacheStep(
                    queries: step.queries, keys: step.keys, values: step.values,
                    cache: step.cache, mask: mask, positionIds: positionIds, rotary: rotary)
            }
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            state: LMOutput.State?,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil,
            lastTokenOnly: Bool = false
        ) -> LMOutput {
            var state = state ?? .init()

            // Ensure inputs is 2D [batch, seq]. Text-only callers (e.g.
            // WiredMemoryUtils, TokenIterator) may pass 1D token arrays.
            let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs

            if pixelValues != nil {
                state[precomputedPositionIdsKey] = nil
                state[ropeDeltasKey] = nil
            }
            let precomputedPositionIds = state[precomputedPositionIdsKey]
            let ropeDeltas = state[ropeDeltasKey]

            let cacheAnchorIndex = model.faIdx ?? model.ssmIdx
            var cacheOffset = 0
            var hasCacheAnchor = false
            if let cache, let cacheAnchorIndex, cache.indices.contains(cacheAnchorIndex),
                let cacheAnchor = cache[cacheAnchorIndex]
            {
                cacheOffset = cacheAnchor.offset
                hasCacheAnchor = true
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil
                && (ropeMask == nil || ropeMask?.ndim == Qwen35Language.batchTokensRank)
            {
                if (hasCacheAnchor && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    if let precomputedPositionIds {
                        let seqLength = inputs.dim(1)
                        positionIds =
                            precomputedPositionIds[
                                0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                    } else {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        state[precomputedPositionIdsKey] = computed
                        state[ropeDeltasKey] = deltas
                    }
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    } else if delta.dim(0) < batchSize {
                        delta = repeated(delta, count: batchSize, axis: 0)
                    } else if delta.dim(0) > batchSize {
                        delta = delta[0 ..< batchSize]
                    }

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [MRopeAxis.count, batchSize, seqLength])
                }
            }

            let emitDrafterState = state[mtpEmitFlagKey] ?? false
            let keepsSnapshots = state[mtpVerifyKeepsSnapshotsKey] ?? false
            let (out, hiddenStates) = MultiRowQuantizedMatvec.run(
                asSpeculativeVerify: keepsSnapshots
            ) {
                let preNormHidden = model(
                    inputs,
                    inputsEmbeds: inputsEmbeds,
                    cache: cache,
                    positionIds: positionIds,
                    applyFinalNorm: !emitDrafterState,
                    keepsSnapshots: keepsSnapshots
                )
                let hiddenStates =
                    emitDrafterState ? model.norm(preNormHidden) : preNormHidden
                let projectionInput =
                    lastTokenOnly && !emitDrafterState
                    ? quantizedVocabularyProjectionInput(
                        hiddenStates, projection: lmHead ?? model.embedTokens)
                    : hiddenStates
                return (projectVocabulary(projectionInput), hiddenStates)
            }

            if emitDrafterState {
                state[mtpLastHiddenStatesKey] = hiddenStates
                state[mtpSharedKVStatesKey] = qwen35VLMSharedKVState(
                    cache: cache, fullAttentionIndex: model.faIdx)
                state[mtpSharedKVOffsetsKey] = qwen35VLMSharedKVOffsets(
                    cache: cache, fullAttentionIndex: model.faIdx)
                state[mtpSharedKVSourceIndicesKey] =
                    model.faIdx.map {
                        ["full_attention": $0]
                    } ?? [:]
                state[mtpPositionDeltasKey] = state[ropeDeltasKey]
            }

            return LMOutput(logits: out, state: state)
        }

        /// The vocabulary projection: the LM head, through
        /// ``MultiRowQuantizedMatvec/projection(_:by:)``, or the tied
        /// embedding.
        ///
        /// - Parameter hidden: The normed hidden states.
        /// - Returns: The logits.
        private func projectVocabulary(_ hidden: MLXArray) -> MLXArray {
            guard let lmHead else { return model.embedTokens.asLinear(hidden) }
            return MultiRowQuantizedMatvec.projection(hidden, by: lmHead)
        }

        func makeCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                return try makeAttentionKVCache(parameters: parameters)
            }
        }

        func prepare() throws {
            try model.prepareFusedProjections()
        }
    }
}

/// The places of the keys and the values among the state arrays of a
/// full-attention cache.
private enum KVCacheStatePart {
    /// The keys.
    static let keys = 0
    /// The values.
    static let values = 1
    /// The arrays of the state.
    static let count = 2
}

private func qwen35VLMSharedKVState(
    cache: [KVCache?]?,
    fullAttentionIndex: Int?
) -> [String: (MLXArray, MLXArray)] {
    guard let cache, let fullAttentionIndex,
        fullAttentionIndex < cache.count,
        let faCache = cache[fullAttentionIndex]
    else {
        return [:]
    }
    let state = faCache.state
    guard state.count == KVCacheStatePart.count else {
        return [:]
    }
    return ["full_attention": (state[KVCacheStatePart.keys], state[KVCacheStatePart.values])]
}

private func qwen35VLMSharedKVOffsets(
    cache: [KVCache?]?,
    fullAttentionIndex: Int?
) -> [String: Int]? {
    guard let cache, let fullAttentionIndex,
        fullAttentionIndex < cache.count,
        let faCache = cache[fullAttentionIndex]
    else {
        return nil
    }
    return ["full_attention": faCache.offset]
}

// MARK: - Model

/// The vision tower at the path it has in ``Qwen35`` when it loads separately.
private final class Qwen35Vision: Module {
    @ModuleInfo(key: "vision_tower") var model: Qwen3VLVision.VisionModel

    init(_ config: Qwen35Configuration) {
        _model.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        super.init()
    }
}

public class Qwen35: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel?
    @ModuleInfo(key: "language_model") var languageModel: Qwen35Language.LanguageModel

    public let config: Qwen35Configuration
    private let onDemandVision: OnDemandComponent<Qwen35Vision>

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        onDemandVision = OnDemandComponent(.vision) { Qwen35Vision(config) }
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        _ = try parameters?.kvCachePlan()
        return try languageModel.makeCache(parameters: parameters)
    }

    public func prepare() throws {
        try languageModel.prepare()
    }

    private func vision() throws -> Qwen3VLVision.VisionModel {
        if let visionModel { return visionModel }
        return try onDemandVision.load().model
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    /// Vision tower + image→token merge over the input, shared by `prepare`
    /// and `prepareContinuation` (mirrors `Qwen25VL.inputEmbeddings`).
    private func visionInputEmbeddings(
        _ input: LMInput
    ) throws -> (
        pixelValues: MLXArray?, imageFrames: [THW]?, videoFrames: [THW]?,
        inputEmbeddings: MLXArray?
    ) {
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels)
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels)
            videoFrames = video.frames
        }

        var inputEmbeddings: MLXArray?

        if !pixelParts.isEmpty,
            let frames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let vision = try vision()
            let typedPixels = concatenated(pixelParts).asType(vision.patchEmbed.proj.weight.dtype)
            pixelValues = typedPixels
            let inputIds = input.text.tokens
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = vision(typedPixels, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        }

        return (pixelValues, imageFrames, videoFrames, inputEmbeddings)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let inputIds2D = inputIds.ndim == 1 ? inputIds[.newAxis, 0...] : inputIds
        let cacheOffset = anchorCacheOffset(cache)
        // Resolved before the routing decision so a warm cache fails closed on either path.
        let positionOffset = try QwenVL.continuationAnchor(
            model: "Qwen35", key: ropeDeltasKey, cacheOffset: cacheOffset,
            batchSize: inputIds2D.dim(0), state: state)

        // Windowed (chunked) prefill — the remaining #344 deferred item for
        // Qwen3.5 — with the same default as the sibling chunked prefills
        // (Gemma3/LLMModel: `prefill.resolvedStepSize()`). The windowed forward also
        // owns every warm continuation (multi-turn chat, tool restart,
        // restored prompt cache): a cold cache is just a continuation
        // anchored at offset 0, and a warm one anchors M-RoPE positions at
        // the cache offset plus the rope delta carried in `state` — never
        // back at zero. The windowed forward is single-sequence; batched
        // inputs keep the single-shot path below.
        let window = prefill.resolvedStepSize()
        if inputIds2D.dim(0) == 1, inputIds2D.dim(-1) > 0,
            cacheOffset > 0 || inputIds2D.dim(-1) > window
        {
            return try prepareContinuation(
                input, inputIds: inputIds2D, cache: cache, cacheOffset: cacheOffset,
                positionOffset: positionOffset, state: state, prefill: prefill)
        }

        let (pixelValues, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        let typedCache = castCache(cache)
        let output = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            languageModel(
                inputIds,
                inputsEmbeds: inputEmbeddings,
                cache: typedCache,
                state: state,
                mask: input.text.mask,
                positionIds: nil,
                pixelValues: pixelValues,
                imageGridTHW: imageFrames,
                videoGridTHW: videoFrames,
                lastTokenOnly: true
            )
        }

        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(output)
    }

    /// Cached token count from a full-attention layer, or a linear layer when no full attention exists.
    private func anchorCacheOffset(_ cache: [any KVCache]) -> Int {
        guard let index = languageModel.model.faIdx ?? languageModel.model.ssmIdx,
            cache.indices.contains(index)
        else {
            return 0
        }
        return cache[index].offset
    }

    /// Warm, windowed continuation through an image-bearing remainder — the
    /// windowed forward that `prepare` also delegates to for long prompts.
    ///
    /// It runs the vision tower and the image→token merge **once** over the
    /// remainder, computes the new image's M-RoPE positions **once** from the
    /// seeded **Position Anchor** (so the image's diverging t/h/w indices
    /// start at the anchor, not zero), then drives the language-model forward
    /// in chunks from the prefill parameters. The full-attention scratch is bounded to
    /// `[heads, chunk, L]` instead of `[heads, L, L]`, so it cannot crash on
    /// a long prefix or a large image.
    ///
    /// `cache` must already be warmed to the restore offset `P` (a fresh cache
    /// means `P = 0`, a crash-safe cold prefill). `state` carries the anchor's
    /// rope delta in the `qwen35.ropeDeltas` slot (zero / absent for an
    /// image-free prefix). The returned state carries the rope delta the
    /// post-image text tail resumes with (`getRopeIndex` delta − `P`), so a
    /// caller threading state end-to-end continues that tail with the ordinary
    /// flat-continuation branch. When the caller requests MTP state, each
    /// chunk's target hidden rows go to the state's
    /// ``MLXLMCommon/mtpPrefillChunkHandlerKey`` handler, with the rope delta
    /// the drafter positions the remainder at, and the returned state carries
    /// the rows of the final forward only. Decode stays caller-owned.
    private func prepareContinuation(
        _ input: LMInput,
        inputIds: MLXArray,
        cache: [any KVCache],
        cacheOffset: Int,
        positionOffset: Int,
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let remainderLength = inputIds.dim(-1)
        precondition(remainderLength > 0, "prepareContinuation needs a non-empty remainder")

        // Vision tower + image→token merge — once, over the whole remainder;
        // chunks slice the merged embeddings below.
        let (_, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        // Offset-aware M-RoPE positions for the whole remainder — once. The new
        // image's t/h/w indices diverge from the anchor; the returned delta is
        // in the same offset frame.
        let (positionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: input.text.mask,
            positionOffset: positionOffset
        )

        // Chunk the forward. Each window forwards `chunk` query tokens against
        // the growing cache, so the full-attention scratch stays `[heads,
        // chunk, L]`. Chunk logits are dropped un-evaluated (only the cache is
        // realized between windows) and the logits the iterator samples come
        // from the reserved final position, so `lm_head` materializes
        // `[1, 1, vocab]`, never `[1, L, vocab]`. `asyncEval` bounds the
        // un-evaluated graph while letting the GPU run window i as the CPU
        // builds window i+1 (same shape as the sibling chunked prefills).
        let typedCache = castCache(cache)
        let emitDrafterState = state?[mtpEmitFlagKey] ?? false

        // Seed the post-image text tail's anchor. The vendor's flat-continuation
        // branch positions tail token j at `tailCacheOffset + ropeDeltas + j`;
        // after this remainder `tailCacheOffset = P + remainderLength`, so the
        // delta the tail needs is the offset-frame `getRopeIndex` delta minus
        // `P` (which `getRopeIndex` implicitly counted into `remainderLength`).
        let resumeDelta = ropeDeltas - MLXArray(Int32(cacheOffset))

        /// One forward over `range`, slicing positions and embeddings in lockstep.
        func forward(_ range: Range<Int>) -> LMOutput {
            languageModel(
                inputIds[0..., range],
                inputsEmbeds: inputEmbeddings.map { $0[0..., range, 0...] },
                cache: typedCache,
                state: state,
                mask: nil,
                positionIds: positionIds[0..., 0..., range],
                // Never the pixels: a non-nil value here clears the carried
                // anchor and restarts positions at zero.
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil
            )
        }

        let processed = try prefill.forEachChunk(total: remainderLength) { range in
            let output = forward(range)
            state?.feedMTPPrefillChunk(
                tokens: inputIds[0..., range], output: output, positionDeltas: resumeDelta)
            if let typedCache {
                asyncEval(typedCache)
            }
        }
        if processed > 0, let typedCache {
            eval(typedCache)
        }

        let lastOutput = forward(processed ..< remainderLength)
        prefill.progress?(remainderLength, remainderLength)

        var resumeState = lastOutput.state ?? state ?? LMOutput.State()
        resumeState[precomputedPositionIdsKey] = nil
        resumeState[ropeDeltasKey] = resumeDelta
        if emitDrafterState {
            resumeState[mtpPositionDeltasKey] = resumeDelta
        }

        return .logits(LMOutput(logits: lastOutput.logits, state: resumeState))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        precondition(
            anchorCacheOffset(cache ?? []) == 0 || state?[ropeDeltasKey] != nil,
            "Qwen35 cannot continue a warm prompt cache without \(ropeDeltasKey.id)")
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            input.tokens,
            inputsEmbeds: nil,
            cache: typedCache,
            state: state,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result
    }

    public var excludedCheckpointNamespaces: [String] {
        Qwen35CheckpointPolicy.excludedTargetNamespaces
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try Qwen35CheckpointPolicy.prepareTarget(
            checkpoint, layout: .vision,
            tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)
        checkpoint.weights = try sanitize(
            weights: checkpoint.weights, metadata: checkpoint.metadata)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            // Converted checkpoints can keep the MTP head; the drafter loads it, not this model.
            return Qwen35CheckpointPolicy.targetWeights(weights)
        }
        return try sanitize(weights: weights)
    }

    /// The axis of the kernel of a conv1d weight in a raw checkpoint,
    /// `(out, in, kernel)`.
    private static let checkpointConvKernelAxis = 2

    /// The axis of the kernel of a conv1d weight in MLX, `(out, kernel, in)`.
    private static let convKernelAxis = 1

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        // Whether the checkpoint stores RMSNorm weights in the raw (un-shifted)
        // convention that needs the `+1` offset. Mirrors the MLXLLM Qwen35 gate
        // so the VLM and text paths agree: a pre-converted MLX checkpoint
        // (conv1d already sanitized) has already been shifted at
        // conversion time and must NOT be shifted again — doing so double-shifts
        // every layernorm and produces garbage tokens. Computed on the incoming
        // weights, before the `mtp.` filter below.
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        // MTP tensors are not proof of a raw checkpoint: a converted checkpoint can
        // keep them (the framework uses them for speculative decoding), and shifting
        // its already-shifted norms a second time produces garbage tokens. The conv1d
        // layout is the reliable signal on its own.
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        let checkpoint = try Qwen35CheckpointPolicy.prepareTarget(
            .init(weights: weights), layout: .vision,
            tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)
        let weights = checkpoint.weights

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var value = originalValue

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(
                    source: Self.checkpointConvKernelAxis, destination: Self.convKernelAxis)
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1
            {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return Qwen3VLVision.VisionModel.sanitize(
            weights: sanitized, config: config.visionConfiguration)
    }
}

extension Qwen35: ModelComponentsProviding {
    public var modelComponents: [ModelComponent: [CheckpointComponent]] {
        Qwen35CheckpointPolicy.components
    }

    public var onDemandComponents: [AnyOnDemandComponent] { [onDemandVision] }
}

extension Qwen35: SpeculativeCacheRewindModel {
    /// The drafts of the longest verify whose snapshot slots a GDN layer keeps.
    public var maximumNativeTargetCacheRewind: Int { GatedDeltaDecodeKernel.mostRewoundTokens }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

// MARK: - Chat conventions

// `Qwen35MoE` subclasses `Qwen35` and inherits both declarations.
extension Qwen35 {
    /// The XML tool-call format of the Qwen 3.5 chat template.
    public var toolCallFormat: ToolCallFormat? { .qwen35 }

    /// The Qwen 3.5 reasoning protocol, which replays the reasoning of a past
    /// turn into the history render, thus the prompt cache extends across
    /// tool rounds.
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.qwen35 }
}
