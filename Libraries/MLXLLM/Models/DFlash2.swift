// Copyright © 2026 Apple Inc.

// DFlash2 block drafter (https://inco.ai/blog/dflash2/), a port of the reference
// `dflash/model_mlx.py` (https://github.com/z-lab/dflash).
//
// The drafter embeds an `[anchor, MASK, ...]` block, runs it through a few
// bidirectional layers that attend to a sliding window of projected target hidden
// states, keeps the top-K candidates of each position, and walks one path through
// them with a pairwise selector. The embedding and the LM head are the target's.
//
// It runs as an MTP drafter: `MTPSpeculativeTokenIterator` verifies its blocks in
// place, and its per-stream state is the context window, one ring of keys and
// values for each drafter layer.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The `dflash_config` object of a DFlash2 checkpoint.
public struct DFlash2SpecConfiguration: Codable, Sendable {

    /// The tokens of a verify the checkpoint was trained for, the anchor included.
    public var blockSize = 16

    /// The channels of one group of the dynamic conv.
    public var convGroupSize = 16

    /// The taps of the dynamic conv.
    public var convKernelSize = 2

    /// The token id that fills the block after the anchor.
    public var maskTokenId = 0

    /// The rank of the selector's bilinear pair score.
    public var selectorRank = 256

    /// The candidates the selector keeps at each block position.
    public var selectorTopK = 16

    /// The target layers the drafter reads, in feature-concatenation order.
    public var targetLayerIds: [Int] = []

    /// The factor of the block embedding.
    public var inputEmbeddingScale: Float = 1

    /// The factor of the draft logits.
    public var outputMultiplier: Float = 1

    /// The soft cap of the draft logits, or `nil` for none.
    public var finalLogitSoftcapping: Float?

    /// The mlx-community layout's sliding window switch, `use_swa`.
    public var useSlidingWindow: Bool?

    /// The mlx-community layout's sliding window, `swa_window_size`.
    public var slidingWindowSize: Int?

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case maskTokenId = "mask_token_id"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useSlidingWindow = "use_swa"
        case slidingWindowSize = "swa_window_size"
    }

    /// The defaults of the reference implementation.
    public init() {}

    /// Decodes a `dflash_config` object; a missing key keeps its default.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? blockSize
        convGroupSize =
            try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? convGroupSize
        convKernelSize =
            try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? convKernelSize
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? maskTokenId
        selectorRank =
            try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? selectorRank
        selectorTopK =
            try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? selectorTopK
        targetLayerIds =
            try container.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? targetLayerIds
        inputEmbeddingScale =
            try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale)
            ?? inputEmbeddingScale
        outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? outputMultiplier
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
        slidingWindowSize = try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
    }
}

/// Configuration of a DFlash2 draft checkpoint (`model_type: qwen3`,
/// `architectures: [DFlash2DraftModel]`).
///
/// Reads both published layouts: `layer_types` with `sliding_window` (z-lab,
/// inco.ai), and `dflash_config.use_swa` with a top-level `block_size`
/// (mlx-community). Every layer must be non-causal sliding-window attention.
public struct DFlash2Configuration: Codable, Sendable {

    /// The width of the hidden state.
    public var hiddenSize = 5120

    /// The decoder layers.
    public var hiddenLayers = 5

    /// The query heads of each layer.
    public var attentionHeads = 32

    /// The key and value heads of each layer.
    public var kvHeads = 8

    /// The width of each head.
    public var headDim = 128

    /// The width of the MLP.
    public var intermediateSize = 17408

    /// The tokens of the vocabulary.
    public var vocabularySize = 248320

    /// The epsilon of each RMS norm.
    public var rmsNormEps: Float = 1e-6

    /// The longest position the rope serves.
    public var maxPositionEmbeddings = 262144

    /// The sliding window of each layer, the block's own anchor included.
    public var slidingWindow = 2048

    /// The base of the rope.
    public var ropeTheta: Float = 10_000_000

    /// The rope scaling parameters, or `nil` for none.
    public var ropeScaling: [String: StringOrNumber]? = nil

    /// The depth of the target the drafter was distilled for; mlx-community omits it.
    public var numTargetLayers: Int?

    /// The `dflash_config` object.
    public var dflash: DFlash2SpecConfiguration = .init()

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case dflash = "dflash_config"
    }

    /// The layer type of every DFlash2 layer.
    private static let slidingAttention = "sliding_attention"

    /// The smallest sliding window: the anchor and one context row.
    private static let smallestWindow = 2

    /// The defaults of the Qwen3.8-27B drafter.
    public init() {}

    /// Decodes a drafter `config.json`.
    ///
    /// - Throws: A `DecodingError` when a layer is causal or does not slide, or the drafter
    ///   reads no target layer.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try decodeShape(container)
        try decodeRope(container)
        try decodeWindow(container)
        guard try container.decodeIfPresent(Bool.self, forKey: .isCausal) == false else {
            throw DecodingError.dataCorruptedError(
                forKey: .isCausal, in: container,
                debugDescription: "DFlash2 needs a non-causal (is_causal: false) drafter")
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids is empty")
        }
    }

    /// Decodes the widths, the depths and the `dflash_config` object.
    private mutating func decodeShape(_ container: KeyedDecodingContainer<CodingKeys>) throws {
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        hiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / attentionHeads)
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? intermediateSize
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? maxPositionEmbeddings
        numTargetLayers = try container.decodeIfPresent(Int.self, forKey: .numTargetLayers)
        dflash =
            try container.decodeIfPresent(DFlash2SpecConfiguration.self, forKey: .dflash)
            ?? dflash
        let nested = try? container.nestedContainer(
            keyedBy: DFlash2SpecConfiguration.CodingKeys.self, forKey: .dflash)
        if try nested?.decodeIfPresent(Int.self, forKey: .blockSize) == nil,
            let blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
        {
            dflash.blockSize = blockSize
        }
    }

    /// Decodes the rope base and scaling from `rope_parameters` or `rope_scaling`.
    private mutating func decodeRope(_ container: KeyedDecodingContainer<CodingKeys>) throws {
        var ropeParameters =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters?["type"] == nil, let ropeType = ropeParameters?["rope_type"] {
            ropeParameters?["type"] = ropeType
        }
        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? ropeTheta
        ropeScaling = ropeParameters
    }

    /// Decodes the sliding window from `layer_types` with `sliding_window`, or from
    /// `dflash_config.use_swa` with `swa_window_size`.
    private mutating func decodeWindow(_ container: KeyedDecodingContainer<CodingKeys>) throws {
        let window: Int?
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            let allSliding =
                layerTypes.count == hiddenLayers
                && layerTypes.allSatisfy { $0 == Self.slidingAttention }
            window =
                allSliding ? try container.decodeIfPresent(Int.self, forKey: .slidingWindow) : nil
        } else {
            window = dflash.useSlidingWindow == true ? dflash.slidingWindowSize : nil
        }
        guard let window, window >= Self.smallestWindow else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "DFlash2 needs sliding-window attention in every layer")
        }
        slidingWindow = window
    }

    /// Encodes the values that decide the drafter's layers, for its identity.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encodeIfPresent(numTargetLayers, forKey: .numTargetLayers)
        try container.encode(dflash, forKey: .dflash)
    }

    /// The context rows a block can see: the window less the block's own anchor.
    public var contextWindow: Int { slidingWindow - 1 }
}

// MARK: - Axes

/// The axes of the drafter's tensors.
private enum DFlash2Axis {
    /// The batch axis of every tensor of the drafter.
    static let batch = 0

    /// The token axis of hidden states, `[batch, tokens, width]`.
    static let token = 1

    /// The width axis of hidden states, `[batch, tokens, width]`.
    static let width = 2

    /// The row axis of heads-first keys and values, `[batch, heads, rows, headDim]`.
    static let row = 2

    /// The head axis of tokens-first projections, `[batch, tokens, heads, headDim]`.
    static let head = 2

    /// The head-dimension axis of projections, `[batch, tokens, heads, headDim]`.
    static let headDim = 3

    /// The permutation between `[batch, tokens, heads, headDim]` and heads first; it is its
    /// own inverse.
    static let headsFirst = [batch, head, token, headDim]
}

// MARK: - Projections

/// The projections of the drafter: through ``MultiRowQuantizedMatvec`` for the 3 to 8
/// rows of a block or a commit when the layer is quantized, else through the layer.
private enum DFlash2Projection {

    /// The projection of `x` by `layer`.
    static func project(_ x: MLXArray, by layer: Linear) -> MLXArray {
        MultiRowQuantizedMatvec.canRun(x, layer)
            ? MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer) : layer(x)
    }
}

// MARK: - Two-tap dynamic depthwise convolution

/// `Conv(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` on `groupSize`-channel groups:
/// a learned base kernel plus a per-group correction projected from the input.
/// `prepare` applies slot 0 and returns the dynamic kernel; `finish` applies slot 1.
final class DFlash2DynamicConv: Module {

    /// The two kernel slots: one for `prepare`, one for `finish`.
    static let kernelSlots = 2

    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        precondition(hiddenSize % groupSize == 0, "conv groups must divide hidden size")
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([Self.kernelSlots, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, Self.kernelSlots * kernelSize * groups, bias: false)
        super.init()
    }

    /// The conv of `hidden` `[B, L, H]` with the dynamic kernel `[B, L, K, groups]` and the
    /// base kernel `[K, H]`. The first position reads zeros before it.
    private static func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let (batch, length, width) = (
            hidden.dim(DFlash2Axis.batch), hidden.dim(DFlash2Axis.token),
            hidden.dim(DFlash2Axis.width)
        )
        let groups = width / groupSize
        let blocks = hidden.reshaped(batch, length, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for tap in 0 ..< base.dim(0) {
            let values =
                tap == 0
                ? blocks
                : concatenated(
                    [MLXArray.zeros(like: blocks[0..., ..<tap]), blocks[0..., ..<(length - tap)]],
                    axis: 1)
            let baseKernel = base[tap].asType(hidden.dtype).reshaped(1, 1, groups, groupSize)
            output = output + baseKernel * values
            output = output + dynamic[0..., 0..., tap, 0..., .newAxis] * values
        }
        return output.reshaped(hidden.shape)
    }

    /// Applies slot 0 to `hidden` and returns the kernel of slot 1.
    func prepare(_ hidden: MLXArray) -> (convolved: MLXArray, kernel: MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = DFlash2Projection.project(hidden, by: kernelProjection).reshaped(
            hidden.dim(DFlash2Axis.batch), hidden.dim(DFlash2Axis.token), Self.kernelSlots,
            kernelSize, groups)
        return (
            Self.convolve(
                hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    /// Applies slot 1 with the kernel `prepare` returned.
    func finish(_ hidden: MLXArray, kernel: MLXArray) -> MLXArray {
        Self.convolve(hidden, dynamic: kernel, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - Attention

/// Block queries attend to the context keys and values plus the block's own.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ configuration: DFlash2Configuration) {
        numHeads = configuration.attentionHeads
        numKVHeads = configuration.kvHeads
        headDim = configuration.headDim
        scale = 1 / Float(configuration.headDim).squareRoot()
        _qProj.wrappedValue = Linear(configuration.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(configuration.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(configuration.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, configuration.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
        super.init()
    }

    /// The rotated keys and the values of `x`, heads first.
    func keysAndValues(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let (batch, length) = (x.dim(DFlash2Axis.batch), x.dim(DFlash2Axis.token))
        let keys = kNorm(
            DFlash2Projection.project(x, by: kProj).reshaped(batch, length, numKVHeads, headDim)
        ).transposed(axes: DFlash2Axis.headsFirst)
        let values = DFlash2Projection.project(x, by: vProj)
            .reshaped(batch, length, numKVHeads, headDim).transposed(axes: DFlash2Axis.headsFirst)
        return (applyRotaryPosition(rope, to: keys, offset: .batch(position)), values)
    }

    /// The rotated block queries and keys plus the values, heads first.
    func projectBlock(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (batch, length) = (x.dim(DFlash2Axis.batch), x.dim(DFlash2Axis.token))
        let queries = qNorm(
            DFlash2Projection.project(x, by: qProj).reshaped(batch, length, numHeads, headDim)
        ).transposed(axes: DFlash2Axis.headsFirst)
        let (keys, values) = keysAndValues(x, rope: rope, position: position)
        return (applyRotaryPosition(rope, to: queries, offset: .batch(position)), keys, values)
    }

    /// The output projection of the attention `[B, heads, L, headDim]`.
    func project(_ attention: MLXArray) -> MLXArray {
        let (batch, length) = (attention.dim(DFlash2Axis.batch), attention.dim(DFlash2Axis.row))
        return DFlash2Projection.project(
            attention.transposed(axes: DFlash2Axis.headsFirst)
                .reshaped(batch, length, numHeads * headDim),
            by: oProj)
    }
}

// MARK: - Decoder layer

/// The gated SiLU MLP of a drafter layer.
final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ configuration: DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(
            configuration.intermediateSize, configuration.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated =
            silu(DFlash2Projection.project(x, by: gateProj))
            * DFlash2Projection.project(x, by: upProj)
        return DFlash2Projection.project(gated, by: downProj)
    }
}

/// Pre-norm layer with a dynamic conv pair around each sublayer, split at the
/// attention so the attention over the context runs between two traces.
final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2DynamicConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2DynamicConv

    init(_ configuration: DFlash2Configuration) {
        _selfAttn.wrappedValue = DFlash2Attention(configuration)
        _mlp.wrappedValue = DFlash2MLP(configuration)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.dflash.convKernelSize,
            groupSize: configuration.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.dflash.convKernelSize,
            groupSize: configuration.dflash.convGroupSize)
        super.init()
    }

    /// The input norm, the conv prepare and the block projections.
    func preAttention(
        x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (kernel: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        let (queries, keys, values) = selfAttn.projectBlock(h, rope: rope, position: position)
        return (kernel, queries, keys, values)
    }

    /// The output projection, the conv finish, both residual adds and the MLP.
    func postAttention(x: MLXArray, attention: MLXArray, kernel: MLXArray) -> MLXArray {
        let h = x + attentionConv.finish(selfAttn.project(attention), kernel: kernel)
        let (m, mlpKernel) = mlpConv.prepare(postAttentionLayerNorm(h))
        return h + mlpConv.finish(mlp(m), kernel: mlpKernel)
    }
}

// MARK: - Candidate selector

/// Scores adjacent pairs of top-K candidates with a low-rank bilinear match
/// gated by the successor's hidden state,
/// `S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>`, then walks the best path greedily.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    /// The walk is about 60 small ops; one trace for each block width.
    private let compiledWalks = KeyedCompiledDecodeSegmentCaches<DFlash2CandidateSelector, Int>(
        count: 1, state: { selector, _ in [selector] },
        body: { selector, _, _, arguments in
            [
                selector.walk(
                    hidden: arguments[WalkArgument.hidden],
                    logits: arguments[WalkArgument.logits],
                    anchor: arguments[WalkArgument.anchor])
            ]
        })

    /// The places of the arguments of a walk trace.
    private enum WalkArgument {
        static let hidden = 0
        static let logits = 1
        static let anchor = 2
    }

    /// The axes of the pair scores, `[batch, positions, candidates, rank]`.
    private enum CandidateAxis {
        static let batch = 0
        static let position = 1
        static let candidate = 2
        static let rank = 3
    }

    /// The permutation of `[batch, positions, candidates, rank]` that puts the candidates last.
    private static let candidatesLast = [
        CandidateAxis.batch, CandidateAxis.position, CandidateAxis.rank, CandidateAxis.candidate,
    ]

    /// Whether ``select(hidden:logits:anchor:)`` runs the walk through its traces.
    var compiledTracesEnabled = true

    init(_ configuration: DFlash2Configuration) {
        topK = configuration.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.dflash.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.dflash.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.dflash.selectorRank, bias: false)
        super.init()
    }

    /// The compiled walk traces.
    var compiledTraceCount: Int { compiledWalks.compiledCount }

    /// The drafted tokens of a block.
    ///
    /// - Parameters:
    ///   - hidden: The normed block hidden states past the anchor, `[1, L, H]`.
    ///   - logits: The matching draft logits, `[1, L, V]`.
    ///   - anchor: The verified token before the block, `[1]`.
    /// - Returns: The drafted token ids, `[1, L]`.
    func select(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
        guard compiledTracesEnabled else {
            return walk(hidden: hidden, logits: logits, anchor: anchor)
        }
        return compiledWalks(self, key: logits.dim(1), at: 0, [hidden, logits, anchor])[0]
    }

    /// The greedy walk over the top-K candidates of each position.
    private func walk(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
        let length = logits.dim(1)
        let kth = logits.dim(-1) - topK
        let candidates = argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = DFlash2Projection.project(hidden, by: hiddenProjection)
        let successors = successorCodebook(candidates)
        let predecessors = predecessorCodebook(candidates)
        // Boundary 0 pairs the anchor with position 0; boundary t pairs the
        // candidates of t - 1 with those of t, gated by position t's hidden.
        let anchorEdges =
            ((predecessorCodebook(anchor) * projected[0..., 0])[0..., .newAxis, 0...]
            * successors[0..., 0, 0..., 0...]).sum(axis: -1)
        let gated =
            predecessors[0..., 0 ..< (length - 1), 0..., 0...]
            * projected[0..., 1..., .newAxis, 0...]
        let edges = matmul(
            gated, successors[0..., 1..., 0..., 0...].transposed(axes: Self.candidatesLast))

        var path: [MLXArray] = []
        var scores = unary[0..., 0] + anchorEdges
        for position in 0 ..< length {
            let chosen = argMax(scores, axis: -1)
            path.append(
                takeAlong(candidates[0..., position], chosen[.newAxis, 0...], axis: -1)[0..., 0])
            if position + 1 < length {
                scores =
                    unary[0..., position + 1]
                    + takeAlong(
                        edges[0..., position], chosen[0..., .newAxis, .newAxis], axis: 1
                    )[0..., 0, 0...]
            }
        }
        return stacked(path, axis: 1).asType(.int32)
    }
}

// MARK: - Draft model

/// The DFlash2 drafter, an MTP drafter of the target layers it reads.
///
/// A proposal runs as compiled segments split at each layer's attention, one schedule for
/// each block width. The context of a stream is its ``MTPDrafterState``: one
/// ``RotatingKVCache`` of the context keys and values for each layer, whose entry `e` is the
/// target hidden row of position `e`. The MTP iterator appends the rows of each prefill chunk
/// and each verified round, and every rewind of the state stays exact through the rings'
/// rewind reserve.
public final class DFlash2DraftModel: Module, ResumableMTPDrafterModel {

    /// The quantization the KaedeTai 4-bit drafters use: affine, 4 bits, groups of 64.
    public static let fourBit = BaseConfiguration.Quantization(
        groupSize: fourBitGroupSize, bits: fourBitBits)

    /// The weights of one quantization group of ``fourBit``.
    private static let fourBitGroupSize = 64

    /// The bits of one weight of ``fourBit``.
    private static let fourBitBits = 4

    /// The context arrays of each layer: its keys, then its values.
    private static let contextArraysPerLayer = 2

    /// The rows of a context projection that run through a trace: the rows of a verified
    /// round. A prompt chunk projects eagerly; a trace for each chunk length would only churn.
    private static let mostTracedContextRows = GatedDeltaDecodeKernel.mostTokens

    /// The configuration of the checkpoint.
    public let configuration: DFlash2Configuration

    /// The quantization the drafter applies to its linear layers at load, or `nil` to keep
    /// the checkpoint's precision.
    public let quantization: BaseConfiguration.Quantization?

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    let rope: RoPELayer

    /// Whether a proposal runs through its compiled traces; tests compare the two.
    var compiledTracesEnabled = true {
        didSet { candidateSelector.compiledTracesEnabled = compiledTracesEnabled }
    }

    /// The segments of a proposal, one schedule for each block width: segment `i` closes layer
    /// `i - 1` and opens layer `i`, and the last applies the final norm.
    private let segmentTraces: KeyedCompiledDecodeSegmentCaches<DFlash2DraftModel, Int>

    /// The context projection of a verified round, one trace for each row count.
    private let contextTraces: KeyedCompiledDecodeSegmentCaches<DFlash2DraftModel, Int>

    /// The identity of the drafter: its configuration and its quantization.
    public let drafterID: MTPDrafterID

    /// Makes a drafter of `configuration`.
    ///
    /// - Parameters:
    ///   - configuration: The checkpoint configuration.
    ///   - quantization: The quantization to apply to the linear layers at load, as
    ///     ``fourBit``, or `nil` to keep the checkpoint's precision.
    public init(
        _ configuration: DFlash2Configuration,
        quantization: BaseConfiguration.Quantization? = nil
    ) {
        self.configuration = configuration
        self.quantization = quantization
        self.drafterID = MTPDrafterID(
            drafterType: DFlash2DraftModel.self,
            configuration: Identity(configuration: configuration, quantization: quantization))
        let concatenatedWidth = configuration.dflash.targetLayerIds.count * configuration.hiddenSize
        _fc.wrappedValue = Linear(concatenatedWidth, configuration.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            DFlash2DecoderLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(configuration)
        rope = initializeRope(
            dims: configuration.headDim,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: configuration.ropeScaling,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings)
        segmentTraces = KeyedCompiledDecodeSegmentCaches(
            count: configuration.hiddenLayers + 1,
            state: { model, index in model.segmentModules(at: index) },
            body: { model, _, index, arguments in model.segmentBody(at: index, arguments) })
        contextTraces = KeyedCompiledDecodeSegmentCaches(
            count: 1,
            state: { model, _ in [model.fc, model.hiddenNorm] + model.layers.map { $0 as Module } },
            body: { model, _, _, arguments in model.contextBody(arguments) })
        super.init()
    }

    /// The values a continuation of the drafter must match: the configuration and the
    /// quantization, since a 4-bit drafter writes other context rows than a bf16 one.
    private struct Identity: Encodable {
        let configuration: DFlash2Configuration
        let quantization: BaseConfiguration.Quantization?
    }

    // MARK: Drafter surface

    /// The tokens of a verify: the trained block, capped by the longest verify the target
    /// rewinds in place.
    public var maximumBlockSize: Int? {
        min(configuration.dflash.blockSize, GatedDeltaDecodeKernel.mostTokens)
    }

    /// The drafter reads hidden rows, not the target's keys and values.
    public var requiresSharedTargetKV: Bool { false }

    /// The drafter reads the prompt's hidden rows before its first proposal.
    public var requiresPromptPrefill: Bool { true }

    /// Sampled streams verify the greedy drafts by sample-and-match.
    public var requiresGreedySampling: Bool { false }

    /// The target layers the drafter reads.
    public var targetLayerIds: [Int]? { configuration.dflash.targetLayerIds }

    /// The width of the target rows the drafter reads: the outputs of its target layers,
    /// side by side.
    public var hiddenSize: Int? { targetHiddenWidth }

    /// The width of the target rows the drafter reads.
    private var targetHiddenWidth: Int {
        configuration.dflash.targetLayerIds.count * configuration.hiddenSize
    }

    /// The context rows a block can see.
    public var contextWindow: Int { configuration.contextWindow }

    /// The target shape the drafter was distilled for.
    public var targetGeometry: DFlash2Geometry {
        DFlash2Geometry(
            layerCount: configuration.numTargetLayers, hiddenSize: configuration.hiddenSize,
            vocabularySize: configuration.vocabularySize)
    }

    /// The compiled traces a proposal has run.
    var compiledTraceCount: Int {
        segmentTraces.compiledCount + contextTraces.compiledCount
            + candidateSelector.compiledTraceCount
    }

    /// Checks that `target` is a DFlash2 target of the shape the drafter was distilled for.
    public func validateCompatibility(with target: any LanguageModel) throws {
        guard let target = target as? any DFlash2TargetModel else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "DFlash2DraftModel", expected: "a DFlash2 target",
                actual: String(describing: type(of: target)))
        }
        guard
            targetGeometry.accepts(
                target.dflash2Geometry, layerIds: configuration.dflash.targetLayerIds)
        else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "DFlash2DraftModel", expected: "a target of \(targetGeometry)",
                actual: "a target of \(target.dflash2Geometry)")
        }
    }

    /// A state with an empty context ring for each layer, whose rewind reserve keeps each
    /// rewind of the state exact after the ring wraps.
    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        let reserve = ShiftedMTPDrafter.rewindReserve(
            maximumBlockSize: maximumBlockSize ?? configuration.dflash.blockSize)
        return MTPDrafterState(
            cache: layers.map { _ in
                RotatingKVCache(maxSize: contextWindow, rewindCapacity: reserve)
            })
    }

    // MARK: Context

    /// Appends the target hidden rows of `tokens` to the context: entry `nextPosition + i`
    /// takes row `i`, the target hidden row of that position.
    ///
    /// - Returns: `false`, leaving the state unchanged, when the rows are too few or of
    ///   another width, or the state is not aligned.
    public func appendDrafterState(
        target _: any LanguageModel,
        tokens: MLXArray,
        targetHidden: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> Bool {
        let count = normalizedMTPTokenBatch(tokens).dim(-1)
        guard count > 0, targetHidden.dim(1) >= count, targetHidden.dim(-1) == targetHiddenWidth,
            state.proposalAppended == 0, state.isAligned, state.cache.count == layers.count
        else { return false }
        let rows = targetHidden.dim(1) == count ? targetHidden : targetHidden[0..., ..<count, 0...]
        let projected = projectContext(rows, position: MLXArray([Int32(state.nextPosition)]))
        for (index, cache) in state.cache.enumerated() {
            let first = Self.contextArraysPerLayer * index
            _ = cache.update(keys: projected[first], values: projected[first + 1])
        }
        state.nextPosition += count
        return state.isAligned
    }

    /// The context keys and values of each layer for `targetHidden` at `position`.
    private func projectContext(_ targetHidden: MLXArray, position: MLXArray) -> [MLXArray] {
        let rows = targetHidden.dim(1)
        guard compiledTracesEnabled, rows <= Self.mostTracedContextRows else {
            return contextBody([targetHidden, position])
        }
        return contextTraces(self, key: rows, at: 0, [targetHidden, position])
    }

    /// The places of the arguments of a context trace.
    private enum ContextArgument {
        static let targetHidden = 0
        static let position = 1
    }

    /// `[targetHidden, position]` to `[keys, values]` for each layer.
    private func contextBody(_ arguments: [MLXArray]) -> [MLXArray] {
        let context = hiddenNorm(
            DFlash2Projection.project(arguments[ContextArgument.targetHidden], by: fc))
        return layers.flatMap { layer in
            let (keys, values) = layer.selfAttn.keysAndValues(
                context, rope: rope, position: arguments[ContextArgument.position])
            return [keys, values]
        }
    }

    /// The newest context keys and values of `cache`, in position order, or `nil` before the
    /// first row.
    private func contextRows(_ cache: KVCache) -> (keys: MLXArray, values: MLXArray)? {
        guard let ring = cache as? RotatingKVCache, ring.offset > 0 else { return nil }
        return ring.logicalView(tail: contextWindow)
    }

    /// The mask of a block of `width` queries over `contextRows` context rows that end right
    /// before the block, then the block itself.
    ///
    /// Query `i` sees the context rows less than `window` positions before it, and the whole
    /// block. While the oldest row is inside every query's window the mask is `.none`.
    ///
    /// - Parameters:
    ///   - contextRows: The context rows, the newest right before the block.
    ///   - width: The block width.
    ///   - window: The sliding window.
    /// - Returns: The mask mode.
    static func visibilityMask(
        contextRows: Int, width: Int, window: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Query `i` sees context row `j` when `j >= contextRows + i - window + 1`.
        let firstHidden = contextRows + width - 1 - window
        guard firstHidden >= 0 else { return .none }
        let columns = MLXArray(Int32(0) ..< Int32(contextRows + width))
        let rows = MLXArray(Int32(0) ..< Int32(width))
        let firstVisible = rows + Int32(contextRows + 1 - window)
        let visible =
            (columns.expandedDimensions(axis: 0) .>= firstVisible.expandedDimensions(axis: 1))
            | (columns .>= Int32(contextRows)).expandedDimensions(axis: 0)
        return .array(visible)
    }

    // MARK: Proposals

    /// Drafts from an empty context.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    /// Drafts `blockSize - 1` tokens after `lastToken`, at `queryOffset`, in one pass over the
    /// block `[lastToken, MASK, ...]` and the context of `state`. The state does not change.
    ///
    /// - Returns: The drafts, shape `[1, blockSize - 1]`.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?, queryOffset: Int,
        blockSize: Int, state: inout MTPDrafterState, sampler: any LogitSampler
    ) -> MLXArray {
        guard let target = target as? any DFlash2TargetModel else {
            preconditionFailure("DFlash2DraftModel requires a DFlash2 target")
        }
        let anchor = normalizedMTPColumn(lastToken).asType(.int32)
        precondition(anchor.dim(0) == 1, "DFlash2 drafts one stream at a time")
        let masks = MLXArray.full(
            [1, blockSize - 1], values: MLXArray(Int32(configuration.dflash.maskTokenId)),
            dtype: .int32)
        let block = concatenated([anchor, masks], axis: 1)
        let hidden = blockHidden(
            target.dflash2Embedding(block) * configuration.dflash.inputEmbeddingScale,
            position: MLXArray([Int32(queryOffset)]), context: state.cache.map(contextRows))
        let logits = sampler.suppressing(draftLogits(hidden, target: target))
        return candidateSelector.select(hidden: hidden, logits: logits, anchor: anchor[0..., 0])
    }

    /// The normed hidden states of the block past its anchor, `[1, width - 1, H]`.
    private func blockHidden(
        _ embedded: MLXArray, position: MLXArray,
        context: [(keys: MLXArray, values: MLXArray)?]
    ) -> MLXArray {
        let width = embedded.dim(1)
        let contextCount = context.first.flatMap { $0?.keys.dim(DFlash2Axis.row) } ?? 0
        let mask = Self.visibilityMask(
            contextRows: contextCount, width: width, window: configuration.slidingWindow)
        var x = embedded
        var pending: [MLXArray] = []
        for index in 0 ... layers.count {
            let arguments = [x] + pending + [position]
            let outputs =
                compiledTracesEnabled
                ? segmentTraces(self, key: width, at: index, arguments)
                : segmentBody(at: index, arguments)
            x = outputs[SegmentOutput.hidden]
            guard index < layers.count else { break }
            let rows = context[index]
            let attention = MLXFast.scaledDotProductAttention(
                queries: outputs[SegmentOutput.queries],
                keys: rows.map {
                    concatenated([$0.keys, outputs[SegmentOutput.keys]], axis: DFlash2Axis.row)
                }
                    ?? outputs[SegmentOutput.keys],
                values: rows.map {
                    concatenated(
                        [$0.values, outputs[SegmentOutput.values]], axis: DFlash2Axis.row)
                } ?? outputs[SegmentOutput.values],
                scale: layers[index].selfAttn.scale, mask: mask)
            pending = [attention, outputs[SegmentOutput.kernel]]
        }
        return x[0..., 1..., 0...]
    }

    /// The draft logits of the block's hidden states through the target's head.
    private func draftLogits(_ hidden: MLXArray, target: any DFlash2TargetModel) -> MLXArray {
        var logits =
            target.dflash2Head.map { DFlash2Projection.project(hidden, by: $0) }
            ?? target.dflash2Embedding.asLinear(hidden)
        logits = logits * configuration.dflash.outputMultiplier
        if let cap = configuration.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    // MARK: Compiled segments

    /// The places of the outputs of a segment that opens a layer.
    private enum SegmentOutput {
        static let hidden = 0
        static let kernel = 1
        static let queries = 2
        static let keys = 3
        static let values = 4
    }

    /// The places of the arguments of a segment that closes a layer; the position is last.
    private enum SegmentArgument {
        static let hidden = 0
        static let attention = 1
        static let kernel = 2
    }

    /// The modules segment `index` reads: the layer it closes, and the layer it opens or the
    /// final norm.
    private func segmentModules(at index: Int) -> [Module] {
        let closes: [Module] = index > 0 ? [layers[index - 1]] : []
        let opens: [Module] = index < layers.count ? [layers[index]] : [norm]
        return closes + opens
    }

    /// Segment `index`: closes layer `index - 1` and opens layer `index`; the last segment
    /// applies the final norm. In: `[x, position]` for segment 0, else
    /// `[x, attention, kernel, position]`. Out: `[x, kernel, queries, keys, values]`, or
    /// `[normed]` for the last.
    private func segmentBody(at index: Int, _ arguments: [MLXArray]) -> [MLXArray] {
        var x = arguments[SegmentArgument.hidden]
        if index > 0 {
            x = layers[index - 1].postAttention(
                x: x, attention: arguments[SegmentArgument.attention],
                kernel: arguments[SegmentArgument.kernel])
        }
        guard index < layers.count else { return [norm(x)] }
        let (kernel, queries, keys, values) = layers[index].preAttention(
            x: x, rope: rope, position: arguments[arguments.count - 1])
        return [x, kernel, queries, keys, values]
    }

    // MARK: Checkpoint

    /// Names the selector codebooks as `Embedding` weights, and quantizes the linear layers
    /// with ``quantization`` when the checkpoint is not quantized already.
    ///
    /// The quantized tensors replace the full-precision ones before the loader materializes
    /// the checkpoint, so the full-precision weights never stand in memory all at once.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        checkpoint.weights = sanitize(weights: checkpoint.weights)
        guard let quantization, checkpoint.perLayerQuantization == nil else { return checkpoint }
        for (path, module) in leafModules().flattened() where module is Linear {
            let key = "\(path).weight"
            guard let weight = checkpoint.weights[key],
                weight.dim(-1) % quantization.groupSize == 0
            else { continue }
            let (packed, scales, biases) = MLX.quantized(
                weight, groupSize: quantization.groupSize, bits: quantization.bits,
                mode: quantization.mode)
            checkpoint.weights[key] = packed
            checkpoint.weights["\(path).scales"] = scales
            checkpoint.weights["\(path).biases"] = biases
        }
        checkpoint.perLayerQuantization = .init(
            quantization: quantization, perLayerQuantization: [:])
        return checkpoint
    }

    /// The checkpoint stores the selector codebooks as bare tensors; `Embedding` expects a
    /// `.weight` suffix.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: bare) {
                weights["\(bare).weight"] = value
            }
        }
        return weights
    }
}
