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

/// A Flint attention layer: query, key and value projections, then attention
/// with per-head RMS norms, optional RoPE and learned sinks.
///
/// ``FlintAttention`` keeps the three projections of the checkpoint;
/// ``FlintFusedAttention`` computes them with one matmul.
protocol FlintAttentionLayer: Module {
    var scale: Float { get }
    var wo: Linear { get }
    var qNorm: RMSNorm { get }
    var kNorm: RMSNorm { get }
    /// The learned attention sink of each query head.
    var sinks: MLXArray { get }
    /// `nil` for a layer of `no_rope_layer_indices`.
    var rope: RoPELayer? { get }

    /// The queries, keys and values of `x`, each `[B, L, heads, headDim]`.
    func projectHeads(_ x: MLXArray) -> (queries: MLXArray, keys: MLXArray, values: MLXArray)

    /// The normed and rotated queries and keys of `x` and its values, each
    /// `[B, heads, L, headDim]`.
    ///
    /// - Parameters:
    ///   - x: The layer input, shape `[B, L, hiddenSize]`.
    ///   - offset: The position of the first token of `x`; `nil` reads 0.
    /// - Returns: The heads the attention reads.
    func attentionInputs(_ x: MLXArray, offset: RoPEOffset?) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    )
}

/// The axes of a `[B, L, heads, headDim]` attention tensor.
private enum FlintHeadAxis {
    /// The axis that holds the batch.
    static let batch = 0
    /// The axis that holds the tokens.
    static let token = 1
    /// The axis that holds the heads.
    static let head = 2
    /// The axis that holds the numbers of one head.
    static let width = 3

    /// The axis order that swaps the token axis and the head axis. The swap is
    /// its own inverse, thus the same order takes the attention output back
    /// again.
    static let headMajor = [batch, head, token, width]
}

/// The axes of an attention mask array, counted from its end: a mask of
/// `[..., queries, keys]`.
private enum FlintMaskAxis {
    /// The axis that holds the queries.
    static let query = -2
    /// The axis that holds the keys.
    static let key = -1
}

/// The attention mask of one Flint layer.
enum FlintAttentionMask {
    /// A mask the SDPA of MLX reads.
    case sdpa(MLXFast.ScaledDotProductAttentionMaskMode)
    /// The bands of a sliding layer whose queries fill several blocks. The
    /// layer gives the keys its cache presents to the bands in place of the
    /// array mask of its window.
    case bands(SlidingWindowBands)
}

extension FlintAttentionLayer {
    func callAsFunction(_ x: MLXArray, mask: FlintAttentionMask, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let heads = attentionInputs(x, offset: cache?.ropeOffset)

        // The sinks load in the dtype of the queries (see `FlintModel.sanitize`),
        // so the cast returns them as they are and adds no kernel. It casts only
        // when the parameters change dtype after load.
        let output = attention(
            queries: heads.queries, keys: heads.keys, values: heads.values, mask: mask,
            cache: cache, sinks: sinks.asType(heads.queries.dtype)
        )
        .transposed(axes: FlintHeadAxis.headMajor)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// The op path of ``attentionInputs(_:offset:)``: the heads of
    /// ``projectHeads(_:)``, normed and rotated.
    func attentionInputs(_ x: MLXArray, offset: RoPEOffset?) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        rotatedHeads(projectHeads(x), offset: offset)
    }

    /// The query and key norms, the head-major layout and the RoPE of the
    /// projected heads.
    ///
    /// - Parameters:
    ///   - heads: The projected heads, each `[B, L, heads, headDim]`.
    ///   - offset: The position of the first token; `nil` reads 0.
    /// - Returns: The heads the attention reads, each `[B, heads, L, headDim]`.
    func rotatedHeads(
        _ heads: (queries: MLXArray, keys: MLXArray, values: MLXArray), offset: RoPEOffset?
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        var queries = qNorm(heads.queries).transposed(axes: FlintHeadAxis.headMajor)
        var keys = kNorm(heads.keys).transposed(axes: FlintHeadAxis.headMajor)
        let values = heads.values.transposed(axes: FlintHeadAxis.headMajor)

        if let rope {
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }
        return (queries, keys, values)
    }

    /// Writes the keys and the values into `cache`, then attends over every
    /// key the cache presents under `mask`.
    ///
    /// The bands come only with no cache or a ``RotatingKVCache`` that keeps
    /// no prefix rows (see ``FlintModelInner/attentionMasks(h:cache:)``),
    /// whose update presents the keys the bands were made for.
    private func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, mask: FlintAttentionMask,
        cache: KVCache?, sinks: MLXArray
    ) -> MLXArray {
        switch mask {
        case .sdpa(let mode):
            return attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache,
                scale: scale, mask: mode, sinks: sinks)
        case .bands(let bands):
            let (presentedKeys, presentedValues) =
                cache?.update(keys: keys, values: values) ?? (keys, values)
            return bands.attention(
                queries: queries, keys: presentedKeys, values: presentedValues, scale: scale,
                sinks: sinks)
        }
    }
}

class FlintAttention: Module, FlintAttentionLayer {
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

    /// The rotation ``FlintDecodeHeadsKernel`` gives the heads of this layer,
    /// or `nil` when the kernel does not reproduce the RoPE of the layer.
    let headRotation: FlintHeadRotation?

    /// The position scale of a RoPE without `rope_scaling`.
    private static let unscaledPositions: Float = 1

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
            self.headRotation = .unrotated
        } else if args.ropeScaling == nil {
            self.rope = RoPE(
                dimensions: headDim, traditional: false, base: args.ropeTheta,
                scale: Self.unscaledPositions)
            self.headRotation = .rotary(base: args.ropeTheta, scale: Self.unscaledPositions)
        } else {
            self.rope = initializeRope(
                dims: headDim, base: args.ropeTheta, traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
            self.headRotation = nil
        }
    }

    /// The ``FlintFusedAttention`` of this layer when its projections under
    /// `path` join in `checkpoint`, or `nil` when they keep their own layers.
    func fused(at path: String, in checkpoint: inout ModelCheckpoint) -> FlintFusedAttention? {
        let joined = FlintProjectionFusion.fuse(
            ["q_proj", "k_proj", "v_proj"], under: path, into: "qkv_proj",
            outputAxis: FlintProjectionFusion.linearOutputAxis, in: &checkpoint)
        return joined ? FlintFusedAttention(fusing: self) : nil
    }

    func projectHeads(_ x: MLXArray) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        return (
            wq(x).reshaped(B, L, attentionHeads, headDim),
            wk(x).reshaped(B, L, kvHeads, headDim),
            wv(x).reshaped(B, L, kvHeads, headDim)
        )
    }
}

/// A ``FlintAttention`` whose query, key and value projections load as one
/// `qkv_proj`, whose rows are the query rows, then the key rows, then the value
/// rows.
final class FlintFusedAttention: Module, FlintAttentionLayer {
    /// The index of the query heads among the split parts of `qkv_proj`.
    private static let queriesPart = 0
    /// The index of the key heads among the split parts of `qkv_proj`.
    private static let keysPart = 1
    /// The index of the value heads among the split parts of `qkv_proj`.
    private static let valuesPart = 2

    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var sinks: MLXArray

    let rope: RoPELayer?

    /// The kernel of a decode token, or `nil` when it does not reproduce the
    /// heads of this layer.
    private let decodeHeads: FlintDecodeHeadsKernel?

    /// Takes over the output projection, norms, sinks and RoPE of `attention`,
    /// and makes a `qkv_proj` as wide as its three projections together.
    init(fusing attention: FlintAttention) {
        self.attentionHeads = attention.attentionHeads
        self.kvHeads = attention.kvHeads
        self.headDim = attention.headDim
        self.scale = attention.scale
        self.rope = attention.rope
        self.decodeHeads = attention.headRotation.flatMap {
            FlintDecodeHeadsKernel(
                queryHeads: attention.attentionHeads, kvHeads: attention.kvHeads,
                headDim: attention.headDim, queryEps: attention.qNorm.eps,
                keyEps: attention.kNorm.eps, rotation: $0)
        }

        let projections = [attention.wq, attention.wk, attention.wv]
        _wqkv.wrappedValue = Linear(
            attention.wq.shape.1, projections.map(\.shape.0).reduce(0, +),
            bias: attention.wq.bias != nil)
        _wo.wrappedValue = attention.wo
        _qNorm.wrappedValue = attention.qNorm
        _kNorm.wrappedValue = attention.kNorm
        _sinks.wrappedValue = attention.sinks
    }

    /// Splits the heads of the one projection. At decode (`L` = 1) each part is
    /// a contiguous run of heads, so the norms that follow read it in place.
    func projectHeads(_ x: MLXArray) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        splitHeads(wqkv(x))
    }

    /// The heads of one `qkv_proj` output, each `[B, L, heads, headDim]`.
    private func splitHeads(_ projection: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        let (B, L) = (projection.dim(0), projection.dim(1))
        let heads = projection.reshaped(B, L, -1, headDim)
        let parts = split(
            heads, indices: [attentionHeads, attentionHeads + kvHeads], axis: FlintHeadAxis.head)
        return (parts[Self.queriesPart], parts[Self.keysPart], parts[Self.valuesPart])
    }

    /// The heads the attention reads. A decode token whose layout and RoPE
    /// ``FlintDecodeHeadsKernel`` reads takes the kernel, with the bits of
    /// the op path in one dispatch; every other block takes the op path.
    func attentionInputs(_ x: MLXArray, offset: RoPEOffset?) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        let projection = wqkv(x)
        if let decodeHeads, let position = Self.position(of: offset),
            let heads = decodeHeads.heads(
                of: projection, queryWeight: qNorm.weight, keyWeight: kNorm.weight,
                offset: position)
        {
            return heads
        }
        return rotatedHeads(splitHeads(projection), offset: offset)
    }

    /// The one position the kernel reads: 0 with no offset, the scalar
    /// offset, and `nil` for an offset of each sequence.
    private static func position(of offset: RoPEOffset?) -> Int? {
        switch offset {
        case nil: 0
        case .scalar(let position): position
        case .batch: nil
        }
    }
}

// MARK: - MLP

/// The feed-forward block of a Flint decoder layer: a dense ``FlintMLP`` or
/// ``FlintFusedMLP``, or a ``FlintSparseMoeBlock``.
protocol FlintFeedForward: UnaryLayer {
    /// Adds the output of the block to the residual stream.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hiddenSize)`.
    ///   - residual: The residual stream the output joins, of the shape of `x`.
    /// - Returns: `residual` plus the block output.
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray
}

extension FlintFeedForward {
    /// Adds the block output to `residual` with one add of its own; a block
    /// that can fuse the add into its last kernel implements this itself.
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        residual + self(x)
    }
}

class FlintMLP: Module, FlintFeedForward {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    /// The ``FlintFusedMLP`` of this MLP when its projections under `path` join
    /// in `checkpoint`, or `nil` when they keep their own layers.
    func fused(at path: String, in checkpoint: inout ModelCheckpoint) -> FlintFusedMLP? {
        let joined = FlintProjectionFusion.fuse(
            ["gate_proj", "up_proj"], under: path, into: "gate_up_proj",
            outputAxis: FlintProjectionFusion.linearOutputAxis, in: &checkpoint)
        return joined ? FlintFusedMLP(fusing: self) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(compiledSiluProduct(gate(x), up(x)))
    }
}

/// A ``FlintMLP`` whose gate and up projections load as one `gate_up_proj`,
/// the gate rows first.
final class FlintFusedMLP: Module, FlintFeedForward {
    /// The number of projections, gate and up, that `gate_up_proj` holds.
    private static let parts = 2

    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    /// Takes over the down projection of `mlp`, and makes a `gate_up_proj` as
    /// wide as its gate and up projections together.
    init(fusing mlp: FlintMLP) {
        let (hiddenDimensions, dimensions) = mlp.gate.shape
        _gateUp.wrappedValue = Linear(dimensions, Self.parts * hiddenDimensions, bias: false)
        _down.wrappedValue = mlp.down
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(gateUp(x), parts: Self.parts, axis: -1)
        return down(compiledSiluProduct(parts[0], parts[1]))
    }
}

/// The routed experts of a Flint MoE block: ``SwitchGLU`` keeps the gate and up
/// projections of the checkpoint, and ``FusedGateUpSwitchGLU`` computes them
/// with one gathered matmul.
protocol FlintRoutedExperts: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}

extension SwitchGLU: FlintRoutedExperts {}
extension FusedGateUpSwitchGLU: FlintRoutedExperts {}

/// Sigmoid-routed experts plus one shared expert. The expert bias steers the
/// choice of experts only; the weights of the chosen experts are their sigmoid
/// scores, normalized, in float32.
///
/// A decode step of one token runs through one compiled trace of the block,
/// which reads the compiled router scores and the compiled combine of
/// ``FlintCompiledDecode`` and adds the residual stream inside the combine. A
/// longer block reads the op path of ``FlintExpertRouting``. The two give the
/// same bits.
class FlintSparseMoeBlock: Module, FlintFeedForward {
    /// The routing constants of the checkpoint, and the op path of the routing.
    let routing: FlintExpertRouting

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: any FlintRoutedExperts
    /// A ``FlintMLP``, or a ``FlintFusedMLP`` once its projections fuse.
    @ModuleInfo(key: "shared_expert") var sharedExpert: UnaryLayer?

    /// The compiled combine of a block with a shared expert, which a decode
    /// step reads.
    private let compiledCombine:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray

    /// The compiled combine of a block without a shared expert, which a decode
    /// step reads.
    private let compiledRoutedCombine:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray

    /// The compiled trace of one decode step, which reads the weights of
    /// this block as inputs. Its arguments are `[x, residual]`, and its one
    /// result is `residual` plus the block output.
    ///
    /// One trace replays the whole decode step of the block: the gate, the
    /// router kernel, the experts and the combine. The CPU then builds one
    /// primitive for the block where the op path builds one for each step.
    let compiledDecodeStep = CompiledTrace<FlintSparseMoeBlock> { block, arguments in
        [block.decodeStep(arguments[0], residual: arguments[1])]
    }

    init(_ args: FlintConfiguration) {
        let routing = FlintExpertRouting(
            topK: args.numExpertsPerToken, normalizesWeights: args.normTopkProb)
        self.routing = routing
        self.compiledCombine = FlintCompiledDecode.combine(routing)
        self.compiledRoutedCombine = FlintCompiledDecode.combineRoutedExperts(routing)

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

    /// The op path of one block of tokens.
    ///
    /// - Parameter x: The block input, shape `(batch, tokens, hiddenSize)`.
    /// - Returns: The block output, of the shape and dtype of `x`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let router = FlintExpertRouting.scores(logits: gate(x), bias: expertBias)
        let selected = routing.selected(scores: router.scores, choice: router.choice)
        return routing.combined(
            x, routed: switchMLP(x, selected.indices), weights: selected.weights,
            shared: sharedExpert?(x))
    }

    /// Adds the output of one block of tokens to the residual stream.
    ///
    /// A block of one token runs through ``compiledDecodeStep``, and a longer
    /// block reads the op path. The two give the same bits.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hiddenSize)`.
    ///   - residual: The residual stream the output joins, of the shape of `x`.
    /// - Returns: `residual` plus the block output.
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        guard FlintCompiledDecode.isDecodeStep(x) else {
            return residual + self(x)
        }
        return compiledDecodeStep(self, [x, residual])[0]
    }

    /// The body of ``compiledDecodeStep``: the decode step of the block, with
    /// the compiled router scores and the compiled combine, which adds the
    /// residual stream.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, 1, hiddenSize)`.
    ///   - residual: The residual stream the output joins, of the shape of `x`.
    /// - Returns: `residual` plus the block output.
    private func decodeStep(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        let (scores, choice) = FlintCompiledDecode.routerScores(gate(x), expertBias)
        let selected = routing.selected(scores: scores, choice: choice)
        let routed = switchMLP(x, selected.indices)
        guard let sharedExpert else {
            return compiledRoutedCombine(x, routed, selected.weights, residual)
        }
        return compiledCombine(x, routed, selected.weights, sharedExpert(x), residual)
    }

    /// Fuses the routed gate and up projections, and those of the shared
    /// expert, wherever their parts under `path` join in `checkpoint`.
    func fuseProjections(
        at path: String, in checkpoint: inout ModelCheckpoint, configuration: FlintConfiguration
    ) throws {
        if switchMLP is SwitchGLU,
            FlintProjectionFusion.fuse(
                ["gate_proj", "up_proj"], under: "\(path).switch_mlp", into: "gate_up_proj",
                outputAxis: FlintProjectionFusion.switchOutputAxis, in: &checkpoint)
        {
            try replaceChild(
                "switch_mlp",
                with: FusedGateUpSwitchGLU(
                    inputDims: configuration.hiddenSize,
                    hiddenDims: configuration.moeIntermediateSize,
                    numExperts: configuration.numExperts))
        }
        if let shared = sharedExpert as? FlintMLP,
            let fused = shared.fused(at: "\(path).shared_expert", in: &checkpoint)
        {
            try replaceChild("shared_expert", with: fused)
        }
    }
}

/// The routing constants of a Flint checkpoint, and the op path of the
/// routing and of the combine of a ``FlintSparseMoeBlock``.
///
/// The routing lives in a value type rather than in the block, because
/// ``FlintCompiledDecode`` compiles it: a compiled body can capture a
/// `Sendable` value, and it cannot capture a module.
struct FlintExpertRouting: Sendable {
    /// The number of routed experts one token reads.
    let topK: Int

    /// True when the selected weights divide by their own sum
    /// (`norm_topk_prob`).
    let normalizesWeights: Bool

    /// The epsilon the normalization adds, so that selected weights that add
    /// up to zero cannot divide by zero.
    private static let normalizeEpsilon: Float = 1e-20

    /// The axis of a `(batch, tokens, experts)` score tensor that holds one
    /// score for each routed expert. It is the last axis.
    private static let expertAxis = -1

    /// Scores every routed expert.
    ///
    /// The projection runs in the activation dtype; the sigmoid and the
    /// selection run in float32, as the reference does. The bias loads in
    /// float32 (see ``FlintModel/sanitize(weights:)``); a bias of another
    /// dtype promotes to float32 in the add.
    ///
    /// - Parameters:
    ///   - logits: The gate projection of the block, shape
    ///     `(batch, tokens, experts)`.
    ///   - bias: The expert bias, one value for each routed expert.
    /// - Returns: The sigmoid scores, which weigh the selected experts, and
    ///   the biased scores, which select them. Both are float32.
    static func scores(logits: MLXArray, bias: MLXArray) -> (scores: MLXArray, choice: MLXArray) {
        let scores = sigmoid(logits.asType(.float32))
        return (scores, scores + bias)
    }

    /// Selects the ``topK`` experts of the highest biased scores, and gathers
    /// their unbiased scores, in ascending order of the biased score.
    ///
    /// A single row reads the fused router kernel, and any other block reads
    /// the generic MLX chain. The two give the same bits.
    ///
    /// - Parameters:
    ///   - scores: The sigmoid scores, shape `(batch, tokens, experts)`.
    ///   - choice: The biased scores, of the shape of `scores`.
    /// - Returns: The selected experts and their scores, each of shape
    ///   `(batch, tokens, topK)`. The scores are not normalized yet.
    func selected(scores: MLXArray, choice: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        guard supportsFusedRouterTopK(choice, k: topK) else {
            let kth = choice.dim(Self.expertAxis) - topK
            let indices = MLX.stopGradient(
                MLX.argPartition(choice, kth: kth, axis: Self.expertAxis)[.ellipsis, kth...])
            return (indices, MLX.takeAlong(scores, indices, axis: Self.expertAxis))
        }
        let fused = fusedRouterTopK(
            selection: choice, values: scores, k: topK, normalize: false, order: .ascending)
        return (fused.indices, fused.scores)
    }

    /// Divides the selected weights by their own sum when the checkpoint
    /// normalizes them, and leaves them as they are otherwise.
    ///
    /// - Parameter weights: The selected weights, shape `(batch, tokens, topK)`.
    /// - Returns: The weights the combine reads, of the shape of `weights`.
    func normalized(_ weights: MLXArray) -> MLXArray {
        guard normalizesWeights else {
            return weights
        }
        return weights
            / (weights.sum(axis: Self.expertAxis, keepDims: true) + Self.normalizeEpsilon)
    }

    /// Weighs the outputs of the selected experts, adds them up, and adds the
    /// output of the shared expert.
    ///
    /// The weights are float32, so the compiled sum promotes the routed
    /// outputs to float32 inside its fused multiply, with no cast kernel of
    /// its own. The sum then casts back to the dtype of the block input.
    ///
    /// - Parameters:
    ///   - x: The block input. Only its dtype reaches the result.
    ///   - routed: The output of each selected expert, shape
    ///     `(batch, tokens, topK, hiddenSize)`.
    ///   - weights: The selected weights, shape `(batch, tokens, topK)`, not
    ///     normalized yet.
    ///   - shared: The output of the shared expert, or `nil` for a block that
    ///     holds none.
    /// - Returns: The block output, shape `(batch, tokens, hiddenSize)`, in the
    ///   dtype of `x` (or the dtype of its sum with `shared`).
    func combined(
        _ x: MLXArray, routed: MLXArray, weights: MLXArray, shared: MLXArray?
    ) -> MLXArray {
        let summed = weightedExpertSum(routed, normalized(weights)).asType(x.dtype)
        guard let shared else {
            return summed
        }
        return summed + shared
    }
}

// MARK: - Projection fusion

/// Joins projections that read the same input, at load, so that one matmul
/// computes the rows of all of them.
///
/// Decode of a quantized Flint is bound by the number of kernels per token, not
/// by bandwidth. A quantized matmul computes each output row from its own
/// weights, scales and biases, thus the rows of the joined matmul are the rows
/// of the separate ones, bit for bit.
///
/// The parts join only when each one is quantized with the same bits, group
/// size and mode and carries the same tensors; other quantizations of Flint keep
/// the separate layers. An unquantized checkpoint keeps its layout, because a
/// conversion saves the module tree it loads.
enum FlintProjectionFusion {
    /// The output axis of a `Linear` weight, `[out, in]`, and of its scales,
    /// biases and bias.
    static let linearOutputAxis = 0
    /// The output axis of a `SwitchLinear` weight, `[experts, out, in]`, and of
    /// its scales, biases and bias.
    static let switchOutputAxis = 1

    /// The tensors a quantized layer can carry.
    private static let tensorNames = ["weight", "scales", "biases", "bias"]

    /// Joins the layers `parts` under `parent` into the one layer `fused`,
    /// along `outputAxis`, in the order of `parts`, and moves their quantization
    /// to it.
    ///
    /// - Parameters:
    ///   - parts: the names of the layers to join, under `parent`.
    ///   - parent: the module path of the module that holds the layers.
    ///   - fused: the name of the joined layer, under `parent`.
    ///   - outputAxis: the axis of the output rows of each tensor.
    ///   - checkpoint: the prepared checkpoint, with its tensor names.
    /// - Returns: whether the layers joined. When they do not, `checkpoint` is
    ///   unchanged.
    static func fuse(
        _ parts: [String], under parent: String, into fused: String, outputAxis: Int,
        in checkpoint: inout ModelCheckpoint
    ) -> Bool {
        let paths = parts.map { "\(parent).\($0)" }
        guard var plan = checkpoint.perLayerQuantization,
            let quantization = sharedQuantization(of: paths, in: plan),
            let names = sharedTensorNames(of: paths, in: checkpoint.weights)
        else { return false }

        let fusedPath = "\(parent).\(fused)"
        for name in names {
            let tensors = paths.compactMap {
                checkpoint.weights.removeValue(forKey: "\($0).\(name)")
            }
            checkpoint.weights["\(fusedPath).\(name)"] = concatenated(tensors, axis: outputAxis)
        }
        for path in paths {
            plan.perLayerQuantization[path] = nil
        }
        plan.perLayerQuantization[fusedPath] = .quantize(quantization)
        checkpoint.perLayerQuantization = plan
        return true
    }

    /// The quantization of every layer of `paths`, or `nil` when one layer is
    /// not quantized or two differ in bits, group size or mode.
    private static func sharedQuantization(
        of paths: [String], in plan: BaseConfiguration.PerLayerQuantization
    ) -> BaseConfiguration.Quantization? {
        let settings = paths.map { plan.quantization(layer: $0) }
        guard let first = settings.first, let quantization = first,
            settings.allSatisfy({ $0.map { $0.asTuple == quantization.asTuple } ?? false })
        else { return nil }
        return quantization
    }

    /// The tensor names that every layer of `paths` carries, or `nil` when the
    /// layers carry different tensors or are not quantized.
    private static func sharedTensorNames(
        of paths: [String], in weights: [String: MLXArray]
    ) -> [String]? {
        let present = paths.map { path in
            tensorNames.filter { weights["\(path).\($0)"] != nil }
        }
        guard let names = present.first, names.contains("scales"),
            present.allSatisfy({ $0 == names })
        else { return nil }
        return names
    }
}

extension Module {
    /// Replaces the child module under `key` with `module`.
    fileprivate func replaceChild(_ key: String, with module: Module) throws {
        try update(modules: ModuleChildren(values: [key: .value(module)]), verify: .all)
    }
}

// MARK: - Decoder

class FlintDecoderLayer: Module {
    /// The sliding window of this layer, or `nil` for full attention.
    let window: Int?

    @ModuleInfo(key: "self_attn") var selfAttn: any FlintAttentionLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    /// A dense ``FlintMLP`` or ``FlintFusedMLP``, or a ``FlintSparseMoeBlock``.
    @ModuleInfo var mlp: any FlintFeedForward

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.window = args.window(layer: layerIdx)
        _selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.mlpOnlyLayers.contains(layerIdx) {
            _mlp.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            _mlp.wrappedValue = FlintSparseMoeBlock(args)
        }
    }

    /// Fuses the projections of this layer wherever their parts under `path`
    /// join in `checkpoint`; see ``FlintProjectionFusion``.
    func fuseProjections(
        at path: String, in checkpoint: inout ModelCheckpoint, configuration: FlintConfiguration
    ) throws {
        if let attention = selfAttn as? FlintAttention,
            let fused = attention.fused(at: "\(path).self_attn", in: &checkpoint)
        {
            try replaceChild("self_attn", with: fused)
        }
        if let dense = mlp as? FlintMLP, let fused = dense.fused(at: "\(path).mlp", in: &checkpoint)
        {
            try replaceChild("mlp", with: fused)
        } else if let sparse = mlp as? FlintSparseMoeBlock {
            try sparse.fuseProjections(
                at: "\(path).mlp", in: &checkpoint, configuration: configuration)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: FlintAttentionMask, cache: KVCache?) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return mlp(postAttentionLayerNorm(h), residual: h)
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
        let masks = attentionMasks(h: h, cache: cache)
        for (i, layer) in layers.enumerated() {
            guard let mask = layer.window == nil ? masks.full : masks.sliding else {
                preconditionFailure("Each layer kind has the mask of its first layer")
            }
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    /// The mask of every full-attention layer and of every sliding layer for
    /// the hidden states `h`, each `nil` when the model has no layer of its
    /// kind.
    ///
    /// The caches of one kind hold the same rows, so the cache of the first
    /// layer of a kind shapes the mask of each layer of that kind.
    func attentionMasks(
        h: MLXArray, cache: [KVCache]?
    ) -> (full: FlintAttentionMask?, sliding: FlintAttentionMask?) {
        let full = firstFullLayer.map {
            FlintAttentionMask.sdpa(createAttentionMask(h: h, cache: cache?[$0]))
        }
        let sliding = firstSlidingLayer.flatMap { index in
            layers[index].window.map { slidingMask(h: h, cache: cache?[index], window: $0) }
        }
        return (full, sliding)
    }

    /// The mask of a sliding layer of `window` keys: the bands, when the
    /// queries fill several blocks of the window and the cache presents the
    /// keys of the array mask; otherwise the mask of the SDPA of MLX.
    ///
    /// A ``RotatingKVCache`` without kept prefix rows (each Flint sliding
    /// cache; see ``makeSlidingWindowKVCache(parameters:window:)``) and no
    /// cache both mask a multi-token write with
    /// `createCausalMask(n: L, offset: T - L, windowSize: window)` over the
    /// `T` keys it presents, `[L, T]`. The bands give that attention, so they
    /// take the shape of the array mask.
    private func slidingMask(h: MLXArray, cache: KVCache?, window: Int) -> FlintAttentionMask {
        let mode = createAttentionMask(h: h, cache: cache, windowSize: window)
        guard case .array(let mask) = mode, Self.presentsWindowedKeys(cache),
            let bands = SlidingWindowBands(
                queryCount: mask.dim(FlintMaskAxis.query), keyCount: mask.dim(FlintMaskAxis.key),
                window: window)
        else { return .sdpa(mode) }
        return .bands(bands)
    }

    /// Whether a multi-token write into `cache` attends over the keys of the
    /// windowed causal mask: with no cache, or with a ``RotatingKVCache``
    /// that keeps no prefix rows, whose mask keeps no prefix keys either.
    private static func presentsWindowedKeys(_ cache: KVCache?) -> Bool {
        guard let cache else { return true }
        return (cache as? RotatingKVCache)?.preservedPrefixTokens == 0
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

    /// The prefill step when the caller sets none: the prefill chunk of mlx-vlm
    /// and mlx-lm. A forward of more tokens keeps the GPU busier than the
    /// generic step of ``PrefillParameters/defaultStepSize``.
    public let defaultPrefillStepSize = 2048

    /// Prefills every prompt token but the last into `cache`, and hands the
    /// last token to the iterator, whose forward computes the logits.
    ///
    /// The chunks run through the text model without the output projection, in
    /// steps of at most ``defaultPrefillStepSize`` tokens unless `prefill` sets
    /// its own. The generic prepare hands a prompt that fits in one step to the
    /// iterator whole, and that forward computes the float32 logits of every
    /// prompt row, of which the iterator reads only the last.
    ///
    /// - Parameters:
    ///   - input: The prompt.
    ///   - cache: The cache of each layer, which the prefill fills.
    ///   - state: Not read; Flint keeps no state beside its cache.
    ///   - prefill: The step size, the chunking and the progress observer.
    /// - Returns: The tokens the iterator still has to read: the last prompt
    ///   token, or the whole prompt under ``PrefillParameters/Chunking/unchunked``.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let text = input.text
        let layerCaches = cache.isEmpty ? nil : cache
        let processed = try withPreparedCache(cache, lengths: text.sequenceLengths) {
            // asyncEval lets the CPU build the graph of the next chunk while
            // the GPU evaluates this one; one eval after the loop syncs.
            let processed = try prefill.forEachChunk(
                total: text.tokens.size, defaultStepSize: defaultPrefillStepSize
            ) { range in
                _ = languageModel.model(text[.newAxis, range].tokens, cache: layerCaches)
                asyncEval(cache)
            }
            if processed > 0 {
                eval(cache)
            }
            return processed
        }
        return .tokens(text[processed...])
    }

    /// The module path of the text model in an mlx-vlm checkpoint.
    static let checkpointPrefix = "language_model."

    /// Sanitizes the checkpoint, then fuses the projections of a quantized one
    /// (see ``FlintProjectionFusion``).
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        checkpoint.weights = try sanitize(
            weights: checkpoint.weights, metadata: checkpoint.metadata)
        guard checkpoint.perLayerQuantization != nil else { return checkpoint }
        for (index, layer) in languageModel.model.layers.enumerated() {
            try layer.fuseProjections(
                at: "\(Self.checkpointPrefix)model.layers.\(index)", in: &checkpoint,
                configuration: configuration)
        }
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = Self.checkpointPrefix
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
        storeInForwardDtypes(&weights)

        return Dictionary(uniqueKeysWithValues: weights.map { (prefix + $0.key, $0.value) })
    }

    /// Stores the parameters that a forward pass casts in the dtype that it
    /// reads them in, so that the casts run once, here: the expert bias in the
    /// float32 of the router scores, and the sinks in the dtype of the queries,
    /// which is the dtype of the query norm. These are the casts the forward
    /// pass made, thus the logits do not change.
    ///
    /// - Parameter weights: the checkpoint tensors, without the
    ///   ``checkpointPrefix``.
    private func storeInForwardDtypes(_ weights: inout [String: MLXArray]) {
        for index in languageModel.model.layers.indices {
            let layer = "model.layers.\(index)"
            let biasKey = "\(layer).mlp.expert_bias"
            if let bias = weights[biasKey] {
                weights[biasKey] = bias.asType(.float32)
            }
            let sinksKey = "\(layer).self_attn.softmax_offset"
            if let sinks = weights[sinksKey],
                let queryNorm = weights["\(layer).self_attn.q_norm.weight"]
            {
                weights[sinksKey] = sinks.asType(queryNorm.dtype)
            }
        }
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
    ///
    /// Each assistant message holds one `<think>` block, and `</think>` is a
    /// plain token. After a status line Flint can sample a second `</think>`
    /// where it should end its turn with `<|user|>`, and then write the next
    /// user turn as text. Generation masks `</think>` once the block closes.
    public var reasoningConfig: ReasoningConfig? { FlintModel.reasoning }

    static let reasoning: ReasoningConfig = {
        var config = ReasoningConfig.alwaysOnThinking
        config.implicitEndDelimiters = ["<tool_call>"]
        config.replaysReasoningIntoHistory = true
        config.masksEndDelimiterAfterClose = true
        return config
    }()
}
