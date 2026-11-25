// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import os

// The DSpark block drafter that Inkling 2 checkpoints ship in `mtp.safetensors`, after the oMLX
// reference (`omlx/patches/inkling2/dspark.py` and the batched forward of
// `omlx/speculative/dflash/drafter.py`).
//
// Each draft layer mixes the target's feature taps with its own softmax weights and projects the
// mix to context keys and values. A block holds the anchor token, the last token the target
// decided, and a learned mask embedding at each later position. It attends the newest
// `sliding_window_size - 1` context rows and itself in both directions, and a recurrent Markov
// head turns the target's vocabulary projection of each mask position into a proposal, left to
// right.

// MARK: - Configuration

/// The MTP drafter of an Inkling 2 checkpoint: the `mtp_config` of its `config.json`.
public struct Inkling2MTPConfiguration: Codable, Sendable {
    /// The width of the drafter's hidden states, the target's hidden size.
    public let hiddenSize: Int
    /// The draft layers.
    public let hiddenLayers: Int
    /// The query heads of each draft layer.
    public let attentionHeads: Int
    /// The key and value heads of each draft layer.
    public let kvHeads: Int
    /// The width of each head.
    public let headDim: Int
    /// The width of each draft layer's gated MLP.
    public let intermediateSize: Int
    /// The epsilon of every RMS norm.
    public let rmsNormEps: Float
    /// The base of the rotary position embedding of the context and the block.
    public let ropeTheta: Float
    /// The context the drafter attends: the newest `slidingWindow - 1` rows, in every layer.
    public let slidingWindow: Int
    /// The block of a round: the anchor token and `blockSize - 1` proposals.
    public let blockSize: Int
    /// The width of the Markov head's recurrent state and token embedding.
    public let markovRank: Int
    /// The Markov head; this port implements `rnn_hidden`.
    public let markovHeadType: String
    /// The feature taps of the target, counted in layer visits; the number of visits taps the
    /// final hidden state.
    public let targetLayerIds: [Int]
    /// Whether the query projections have a bias; the port supports none.
    public let queryBias: Bool
    /// Whether the output projections have a bias; the port supports none.
    public let outputBias: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window_size"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case targetLayerIds = "target_layer_ids"
        case queryBias = "q_bias"
        case outputBias = "o_bias"
    }

    /// The norm epsilon of a configuration that names none.
    private static let defaultRMSNormEps: Float = 1e-6

    /// The RoPE base of a configuration that names none.
    private static let defaultRopeTheta: Float = 1_000_000

    /// Decodes `mtp_config`, with the defaults of the reference for the norm epsilon, the RoPE
    /// base and the projection biases.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps =
            try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? Self.defaultRMSNormEps
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? Self.defaultRopeTheta
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        markovRank = try c.decode(Int.self, forKey: .markovRank)
        markovHeadType = try c.decode(String.self, forKey: .markovHeadType)
        targetLayerIds = try c.decode([Int].self, forKey: .targetLayerIds)
        queryBias = try c.decodeIfPresent(Bool.self, forKey: .queryBias) ?? false
        outputBias = try c.decodeIfPresent(Bool.self, forKey: .outputBias) ?? false
    }
}

/// The configuration of an ``Inkling2DraftModel``: the `config.json` of the Inkling 2
/// checkpoint that ships the drafter, whose text configuration gives the vocabulary and the
/// target geometry.
public struct Inkling2DraftConfiguration: Codable, Sendable {
    /// The text stack of the target.
    public let text: Inkling2TextConfiguration
    /// The drafter.
    public let draft: Inkling2MTPConfiguration

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case draft = "mtp_config"
    }

    /// The Markov head this port implements.
    static let markovHeadType = "rnn_hidden"

    /// The smallest block: the anchor and one proposal.
    static let minimumBlockSize = 2

    /// The width of one row of target taps: the hidden states of every tap side by side.
    var tapWidth: Int { draft.targetLayerIds.count * text.hiddenSize }
}

extension Inkling2DraftConfiguration: ModelConfigurationValidating {
    /// Checks that the drafter is the rnn_hidden DSpark drafter without projection biases, that
    /// its dimensions agree, and that it taps the target's own layer visits.
    public func validateModelConfiguration() throws {
        let visits = text.layerVisits.count
        guard draft.markovHeadType == Self.markovHeadType, !draft.queryBias, !draft.outputBias
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark requires the rnn_hidden Markov head and no projection biases.")
        }
        guard draft.hiddenSize == text.hiddenSize, draft.hiddenLayers > 0,
            draft.attentionHeads > 0, draft.kvHeads > 0,
            draft.attentionHeads.isMultiple(of: draft.kvHeads), draft.headDim > 0,
            draft.intermediateSize > 0, draft.markovRank > 0,
            draft.blockSize >= Self.minimumBlockSize, draft.slidingWindow > 1
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark dimensions must be positive and match the target hidden size.")
        }
        guard !draft.targetLayerIds.isEmpty,
            Set(draft.targetLayerIds).count == draft.targetLayerIds.count,
            draft.targetLayerIds.allSatisfy({ (0 ... visits).contains($0) })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark taps must be unique layer visits of the target or its final state."
            )
        }
    }
}

// MARK: - Modules

/// The axis of the taps of `[B, L, T, D]` tap rows.
private let tapAxis = 2

/// The softmax-weighted sum of the target taps of each row, RMS-normalized in float32.
///
/// - Parameters:
///   - taps: the target taps, `[B, L, T * D]`, one hidden state per tap side by side.
///   - weights: the softmax weights of the taps, `[T]`.
///   - normWeight: the weight of the RMS norm, `[D]`.
///   - eps: the epsilon of the RMS norm.
/// - Returns: the mixed rows, `[B, L, D]`, in the dtype of `taps`.
private func mixedTaps(_ taps: MLXArray, weights: MLXArray, normWeight: MLXArray, eps: Float)
    -> MLXArray
{
    let (b, l, width) = (taps.dim(0), taps.dim(1), taps.dim(-1))
    let count = weights.dim(0)
    let rows = taps.reshaped(b, l, count, width / count).asType(.float32)
    let mixed = (rows * weights.asType(.float32).expandedDimensions(axis: -1)).sum(axis: tapAxis)
    return MLXFast.rmsNorm(mixed, weight: normWeight.asType(.float32), eps: eps)
        .asType(taps.dtype)
}

/// The exponent of the attention scale, `headDim^(-1/2)`.
private let attentionScaleExponent: Float = -0.5

private final class Inkling2DraftAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "k_proj_ctx") var contextKeyProjection: Linear
    @ModuleInfo(key: "v_proj_ctx") var contextValueProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var contextKeyNorm: RMSNorm

    let heads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPE

    init(_ config: Inkling2MTPConfiguration) {
        let (dims, headDim) = (config.hiddenSize, config.headDim)
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        scale = pow(Float(headDim), attentionScaleExponent)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        _queryProjection.wrappedValue = Linear(dims, heads * headDim, bias: false)
        _keyProjection.wrappedValue = Linear(dims, kvHeads * headDim, bias: false)
        _valueProjection.wrappedValue = Linear(dims, kvHeads * headDim, bias: false)
        _outputProjection.wrappedValue = Linear(heads * headDim, dims, bias: false)
        _contextKeyProjection.wrappedValue = Linear(dims, kvHeads * headDim, bias: false)
        _contextValueProjection.wrappedValue = Linear(dims, kvHeads * headDim, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _contextKeyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `[B, L, H * D]` as `[B, H, L, D]`.
    private func heads(_ x: MLXArray, count: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), count, -1).swappedAxes(1, headPositionAxis)
    }

    /// The context keys, rotated to their positions, and values of mixed tap rows.
    ///
    /// - Parameters:
    ///   - mixed: the mixed tap rows of this layer, `[B, L, D]`.
    ///   - position: the absolute position of the first row.
    /// - Returns: keys and values, `[B, kvHeads, L, headDim]`.
    func contextKeysValues(_ mixed: MLXArray, position: Int) -> (MLXArray, MLXArray) {
        let keys = contextKeyNorm(heads(contextKeyProjection(mixed), count: kvHeads))
        let values = heads(contextValueProjection(mixed), count: kvHeads)
        return (rope(keys, offset: position), values)
    }

    /// The block attends the context and itself in both directions.
    ///
    /// - Parameters:
    ///   - x: the normalized block, `[B, L, D]`.
    ///   - position: the absolute position of the anchor.
    ///   - context: the context keys and values, or `nil` without context.
    func callAsFunction(_ x: MLXArray, position: Int, context: (MLXArray, MLXArray)?)
        -> MLXArray
    {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = rope(queryNorm(heads(queryProjection(x), count: heads)), offset: position)
        var keys = rope(keyNorm(heads(keyProjection(x), count: kvHeads)), offset: position)
        var values = heads(valueProjection(x), count: kvHeads)
        if let context {
            keys = concatenated([context.0.asType(keys.dtype), keys], axis: headPositionAxis)
            values = concatenated(
                [context.1.asType(values.dtype), values], axis: headPositionAxis)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        return outputProjection(attended.swappedAxes(1, headPositionAxis).reshaped(b, l, -1))
    }
}

private final class Inkling2DraftMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: Inkling2MTPConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class Inkling2DraftLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2DraftAttention
    @ModuleInfo var mlp: Inkling2DraftMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    /// The norm of this layer's mix of the target taps.
    @ModuleInfo(key: "dflare_norm") var contextNorm: RMSNorm

    init(_ config: Inkling2MTPConfiguration) {
        _attention.wrappedValue = Inkling2DraftAttention(config)
        _mlp.wrappedValue = Inkling2DraftMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _contextNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ h: MLXArray, position: Int, context: (MLXArray, MLXArray)?)
        -> MLXArray
    {
        let attended = h + attention(inputNorm(h), position: position, context: context)
        return attended + mlp(postAttentionNorm(attended))
    }
}

/// The recurrent (`rnn_hidden`) Markov head: a bias on each proposal's logits from a gated state,
/// the previous token, the draft hidden state of the position and the newest context row.
private final class Inkling2DraftMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousTokenEmbedding: Embedding
    @ModuleInfo(key: "rnn_proj") var recurrentProjection: Linear
    @ModuleInfo(key: "hidden_out") var hiddenProjection: Linear
    /// The norm of the head's mix of the target taps.
    @ModuleInfo(key: "dflare_norm") var contextNorm: RMSNorm

    let rank: Int

    /// The parts of the recurrent projection's output, `rank` wide each, in order.
    private enum Output: Int, CaseIterable {
        case gate, candidate, latent
    }

    /// The inputs of the recurrent projection `rank` wide: the state and the previous token.
    private static let rankWideInputs = 2

    /// The inputs of the recurrent projection as wide as the hidden state: the draft hidden
    /// state and the context row.
    private static let hiddenWideInputs = 2

    init(_ config: Inkling2DraftConfiguration) {
        let (rank, dims) = (config.draft.markovRank, config.draft.hiddenSize)
        self.rank = rank
        _previousTokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.text.vocabularySize, dimensions: rank)
        _recurrentProjection.wrappedValue = Linear(
            Self.rankWideInputs * rank + Self.hiddenWideInputs * dims,
            Output.allCases.count * rank, bias: true)
        _hiddenProjection.wrappedValue = Linear(rank, dims, bias: true)
        _contextNorm.wrappedValue = RMSNorm(dimensions: dims, eps: config.draft.rmsNormEps)
        super.init()
    }

    /// One step of the recurrence.
    ///
    /// - Parameters:
    ///   - state: the recurrent state, `[B, rank]`.
    ///   - previous: the previous token, `[B]`.
    ///   - hidden: the draft hidden state of the position, `[B, D]`.
    ///   - context: the head's mix of the newest context row, `[B, D]`.
    /// - Returns: the next state and the latent the vocabulary bias projects, `[B, rank]` each.
    func step(state: MLXArray, previous: MLXArray, hidden: MLXArray, context: MLXArray)
        -> (state: MLXArray, latent: MLXArray)
    {
        let dtype = state.dtype
        let inputs = concatenated(
            [
                state, previousTokenEmbedding(previous).asType(dtype), hidden.asType(dtype),
                context.asType(dtype),
            ], axis: -1)
        let projected = recurrentProjection(inputs).asType(.float32)
        func part(_ output: Output) -> MLXArray {
            projected[0..., (output.rawValue * rank) ..< ((output.rawValue + 1) * rank)]
        }
        let gate = sigmoid(part(.gate)).asType(dtype)
        let kept = (gate * state).asType(dtype)
        let candidate = tanh(part(.candidate)).asType(dtype)
        let blend = ((1 - gate) * candidate).asType(dtype)
        let latent = tanh(part(.latent)).asType(dtype)
        return (kept + blend, latent)
    }
}

/// The target's vocabulary projection folded into the Markov head's latent projection, so that
/// a step reads a `[V, rank]` matrix rather than the whole vocabulary projection.
///
/// The fold depends on the target, thus the drafter computes it on the first block it drafts for
/// a target and keeps it for that target. One lock holds the fold, so that streams that share the
/// drafter compute it once.
private final class Inkling2DraftFold: Sendable {
    /// The fold of one target.
    struct Projection {
        /// The target's vocabulary projection the fold belongs to.
        weak var head: Linear?
        /// `head @ hidden_out.weight`, `[V, rank]`.
        let weight: MLXArray
        /// `head @ hidden_out.bias`, `[V]`.
        let bias: MLXArray
    }

    /// The fold of the last target, which only the lock reads and writes.
    private let projection = OSAllocatedUnfairLock<Projection?>(uncheckedState: nil)

    /// The fold for `head`, computed by `make` when the drafter holds none for it.
    func projection(for head: Linear, make: () -> (MLXArray, MLXArray)) -> Projection {
        projection.withLockUnchecked { projection in
            if let projection, projection.head === head {
                return projection
            }
            let (weight, bias) = make()
            let made = Projection(head: head, weight: weight, bias: bias)
            projection = made
            return made
        }
    }
}

// MARK: - Drafter

/// The DSpark block drafter of Inkling 2 Mini and Small (`mtp.safetensors`).
///
/// The drafter reads the target's feature taps, ``targetLayerIds``, which count layer visits of
/// the looped stack. Its state holds, for each draft layer, the context keys and values of the
/// newest `sliding_window_size - 1` positions in a rotating ring, and one ring more for the
/// Markov head's mix of the newest position. Each entry of the state is one target position, so
/// the state rewinds and resumes as `ResumableMTPDrafterModel` describes. Each proposal can
/// take a logit processor, as `StatefulMTPDistributionDrafterModel` describes, so guided
/// generation drafts under its grammar mask.
///
/// The token embeddings and the vocabulary projection are the target's.
public final class Inkling2DraftModel: Module, ResumableMTPDrafterModel,
    StatefulMTPDistributionDrafterModel
{
    /// The configuration the drafter was built from.
    public let configuration: Inkling2DraftConfiguration

    @ModuleInfo(key: "layers") fileprivate var layers: [Inkling2DraftLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: Inkling2DraftMarkovHead
    /// The embedding of each position of a block after the anchor.
    @ParameterInfo(key: "mask_embed") var maskEmbedding: MLXArray
    /// The tap weights of each draft layer and of the Markov head, before the softmax.
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray

    private let fold = Inkling2DraftFold()

    /// The rows of the vocabulary projection the fold computes at a time.
    private static let foldRowsPerChunk = 16384

    /// Creates a drafter with zero weights, which the loader replaces.
    ///
    /// - Parameter configuration: the `config.json` of the checkpoint that ships the drafter.
    public init(_ configuration: Inkling2DraftConfiguration) {
        self.configuration = configuration
        let draft = configuration.draft
        _layers.wrappedValue = (0 ..< draft.hiddenLayers).map { _ in Inkling2DraftLayer(draft) }
        _norm.wrappedValue = RMSNorm(dimensions: draft.hiddenSize, eps: draft.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DraftMarkovHead(configuration)
        _maskEmbedding.wrappedValue = MLXArray.zeros([draft.hiddenSize])
        _fuseWeight.wrappedValue = MLXArray.zeros([
            draft.hiddenLayers + 1, draft.targetLayerIds.count,
        ])
        super.init()
    }

    /// The block of a round: the bonus token and the proposals.
    public var maximumBlockSize: Int? { configuration.draft.blockSize }
    /// The block Inkling 2 Mini decodes fastest with on the M5: the bonus token and two drafts.
    /// A verify row costs the target about a fifth of a decode step in expert reads, thus the
    /// third draft, which the target accepts less often, costs more than it saves.
    public var preferredBlockSize: Int? { Self.fastestBlockSize }
    /// See ``preferredBlockSize``.
    private static let fastestBlockSize = 3
    /// The drafter reads the target's taps, not its keys and values.
    public var requiresSharedTargetKV: Bool { false }
    /// The drafter's context rings take every prompt position.
    public var requiresPromptPrefill: Bool { true }
    /// A sampled stream speculates too: it verifies argmax drafts by sample-and-match, or
    /// verifies sampled drafts by rejection sampling against their reported distribution.
    public var requiresGreedySampling: Bool { false }
    /// The layer visits the target taps for the drafter, and the final hidden state.
    public var targetLayerIds: [Int]? { configuration.draft.targetLayerIds }
    /// The width of one row of target taps.
    public var hiddenSize: Int? { configuration.tapWidth }
    /// The type and the configuration, since a Mini and a Small drafter build other caches.
    public var drafterID: MTPDrafterID {
        MTPDrafterID(drafterType: Self.self, configuration: configuration)
    }

    /// The context rows each draft layer attends.
    private var contextLength: Int { configuration.draft.slidingWindow - 1 }

    /// The context rings of the draft layers, then the ring of the Markov head's mix.
    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        // A finalized round rewinds its discarded tokens and the boundary, at most a block.
        let rewind = configuration.draft.blockSize
        let rings: [KVCache] =
            layers.map { _ in RotatingKVCache(maxSize: contextLength, rewindCapacity: rewind) }
            + [RotatingKVCache(maxSize: 1, rewindCapacity: rewind)]
        return MTPDrafterState(cache: rings)
    }

    /// Appends one context row for each target position of `targetHidden`; the tokens play no
    /// part, since the drafter reads the target taps alone.
    public func appendDrafterState(
        target: any LanguageModel, tokens: MLXArray, targetHidden: MLXArray,
        positionDeltas _: MLXArray?, state: inout MTPDrafterState, sampler _: any LogitSampler
    ) -> Bool {
        let count = normalizedMTPTokenBatch(tokens).dim(-1)
        guard count > 0, target is Inkling2, targetHidden.dim(1) >= count,
            targetHidden.dim(-1) == configuration.tapWidth, state.isAligned,
            state.proposalAppended == 0, state.cache.count == layers.count + 1
        else { return false }

        let rows = targetHidden[0..., ..<count, 0...]
        let weights = tapWeights
        let eps = configuration.draft.rmsNormEps
        for (index, layer) in layers.enumerated() {
            let mixed = mixedTaps(
                rows, weights: weights[index], normWeight: layer.contextNorm.weight, eps: eps)
            let (keys, values) = layer.attention.contextKeysValues(
                mixed, position: state.nextPosition)
            _ = state.cache[index].update(keys: keys, values: values)
        }
        let markovRows = mixedTaps(
            rows, weights: weights[layers.count], normWeight: markovHead.contextNorm.weight,
            eps: eps
        ).expandedDimensions(axis: 1)
        _ = state.cache[layers.count].update(keys: markovRows, values: markovRows)
        state.nextPosition += count
        return true
    }

    /// Starts a new state at `position` over the suffix rows: each row pairs with the target
    /// position it was computed at, from `position` on. A new state rather than a rewound one,
    /// since a ring past its window cannot rewind to empty.
    public func coldStartDrafterState(
        target: any LanguageModel, suffixTokens: MLXArray, suffixTargetHidden: MLXArray,
        firstBonus: MLXArray, position: Int, positionDeltas: MLXArray?,
        state: inout MTPDrafterState, sampler: any LogitSampler
    ) -> Bool {
        let suffix = normalizedMTPTokenBatch(suffixTokens)
        guard suffix.dim(-1) > 0, position >= 0 else { return false }
        var fresh = makeState(parameters: nil)
        fresh.cacheStartPosition = position
        fresh.nextPosition = position
        fresh.minimumDraftProbability = state.minimumDraftProbability
        guard
            appendDrafterState(
                target: target, tokens: suffix, targetHidden: suffixTargetHidden,
                positionDeltas: positionDeltas, state: &fresh, sampler: sampler)
        else { return false }
        state = fresh
        return true
    }

    /// Drafts from a fresh state. A fresh state holds no context row, and drafting needs one,
    /// so a caller drafts through the stateful form with a primed state.
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

    /// Drafts `blockSize - 1` tokens after `lastToken`, which stands at the position after the
    /// newest context row. The state does not change: the block is transient.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockWithLogits(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, processor: nil, sampler: sampler
        ).tokens
    }

    /// Drafts `blockSize - 1` tokens after `lastToken` as the token-only form does, with each
    /// step's logits shaped by `processor` before `sampler` draws from them. The processor
    /// learns each draft before it shapes the next step, so a grammar mask follows the drafts.
    /// The state does not change: the block is transient.
    public func draftBlockWithLogits(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        processor: (any LogitProcessor)?, sampler: any LogitSampler
    ) -> MTPDraftBlockOutput {
        let proposals = blockSize - 1
        precondition(
            (1 ..< configuration.draft.blockSize).contains(proposals),
            "Inkling 2 DSpark drafts from 1 to \(configuration.draft.blockSize - 1) tokens")
        let target = inklingTarget(target)
        let anchor = normalizedMTPColumn(lastToken).asType(.int32)
        let draftHidden = blockHidden(target: target, anchor: anchor, state: state)
        return proposeTokens(
            target: target, anchor: anchor, draftHidden: draftHidden[0..., ..<proposals, 0...],
            context: markovContext(state), processor: processor, sampler: sampler)
    }

    /// The normalized draft hidden states of the positions after the anchor, `[B, block - 1, D]`.
    private func blockHidden(target: Inkling2, anchor: MLXArray, state: MTPDrafterState)
        -> MLXArray
    {
        let anchorEmbedding = target.tokenEmbeddings(anchor)
        let (b, dims) = (anchor.dim(0), configuration.draft.hiddenSize)
        let masks = broadcast(
            maskEmbedding.asType(anchorEmbedding.dtype),
            to: [b, configuration.draft.blockSize - 1, dims])
        var hidden = concatenated([anchorEmbedding, masks], axis: 1)
        for (index, layer) in layers.enumerated() {
            let context = (state.cache[index] as? RotatingKVCache)?.logicalView(
                tail: contextLength)
            hidden = layer(hidden, position: state.nextPosition, context: context)
        }
        return norm(hidden)[0..., 1..., 0...]
    }

    /// Proposes one token per draft hidden state, left to right: each step adds the Markov
    /// head's bias, from the token before it, to the target's logits of the position, and
    /// `processor` shapes the result before `sampler` draws the proposal.
    private func proposeTokens(
        target: Inkling2, anchor: MLXArray, draftHidden: MLXArray, context: MLXArray,
        processor: (any LogitProcessor)?, sampler: any LogitSampler
    ) -> MTPDraftBlockOutput {
        let text = configuration.text
        let vocabulary = text.unpaddedVocabularySize
        let base = target.vocabularyProjection(draftHidden)[.ellipsis, ..<vocabulary]
        let folded = foldedProjection(target.vocabularyProjection)
        var state = MLXArray.zeros(
            [anchor.dim(0), markovHead.rank], dtype: draftHidden.dtype)
        var previous = anchor[0..., 0]
        var processor = processor
        var tokens: [MLXArray] = []
        var processedLogits: [MLXArray] = []
        for step in 0 ..< draftHidden.dim(1) {
            let latent: MLXArray
            (state, latent) = markovHead.step(
                state: state, previous: previous, hidden: draftHidden[0..., step, 0...],
                context: context)
            let bias = matmul(latent, folded.weight.transposed()) + folded.bias
            var logits = base[0..., step, 0...].asType(.float32) + bias.asType(.float32)
            if text.finalLogitSoftcap > 0 {
                logits = tanh(logits / text.finalLogitSoftcap) * text.finalLogitSoftcap
            }
            logits = processor?.process(logits: logits) ?? logits
            let next = normalizedMTPColumn(sampler.sample(logits: logits))
            processor?.didSample(token: next)
            tokens.append(next)
            processedLogits.append(logits.expandedDimensions(axis: 1))
            previous = next[0..., 0]
        }
        return MTPDraftBlockOutput(
            tokens: concatenated(tokens, axis: 1),
            processedLogits: concatenated(processedLogits, axis: 1))
    }

    /// The Markov head's mix of the newest context row, `[B, D]`.
    private func markovContext(_ state: MTPDrafterState) -> MLXArray {
        guard let ring = state.cache.last as? RotatingKVCache,
            let (rows, _) = ring.logicalView(tail: 1)
        else {
            preconditionFailure("Inkling 2 DSpark drafts after at least one context row")
        }
        return rows[0..., 0, -1, 0...]
    }

    /// The softmax weights of the taps, `[layers + 1, taps]`: one row for each draft layer, then
    /// the row of the Markov head.
    private var tapWeights: MLXArray {
        softmax(fuseWeight.asType(.float32), axis: -1).asType(fuseWeight.dtype)
    }

    /// The fold of the target's vocabulary projection, over the unpadded vocabulary.
    private func foldedProjection(_ head: Linear) -> Inkling2DraftFold.Projection {
        fold.projection(for: head) {
            let vocabulary = configuration.text.unpaddedVocabularySize
            let projection = markovHead.hiddenProjection
            // `[D, rank]`: the rows of the vocabulary projection multiply it directly.
            let weight = projection.weight.asType(.float32)
            let bias =
                projection.bias.map { $0.asType(.float32) } ?? MLXArray.zeros([weight.dim(0)])
            let dtype = projection.weight.dtype
            let chunks = stride(from: 0, to: vocabulary, by: Self.foldRowsPerChunk).map { start in
                let rows = denseRows(
                    of: head, start ..< Swift.min(start + Self.foldRowsPerChunk, vocabulary))
                let chunk = (matmul(rows, weight).asType(dtype), matmul(rows, bias).asType(dtype))
                eval(chunk.0, chunk.1)
                return chunk
            }
            return (concatenated(chunks.map(\.0)), concatenated(chunks.map(\.1)))
        }
    }

    /// Rows of the target's vocabulary projection in float32, dequantized when quantized.
    private func denseRows(of head: Linear, _ rows: Range<Int>) -> MLXArray {
        guard let quantized = head as? QuantizedLinear else {
            return head.weight[rows].asType(.float32)
        }
        return dequantized(
            quantized.weight[rows], scales: quantized.scales[rows],
            biases: quantized.biases?[rows], groupSize: quantized.groupSize,
            bits: quantized.bits, mode: quantized.mode
        ).asType(.float32)
    }

    private func inklingTarget(_ target: any LanguageModel) -> Inkling2 {
        guard let target = target as? Inkling2 else {
            preconditionFailure(
                "Inkling2DraftModel requires an Inkling 2 target, got \(type(of: target))")
        }
        return target
    }

    /// Throws when ``isCompatible(with:)`` does not hold for `target`.
    public func validateCompatibility(with target: any LanguageModel) throws {
        try requireCompatibleTarget(
            target,
            expected: "an Inkling 2 target with the drafter's hidden size, vocabulary and layer "
                + "visits")
    }

    /// Whether `target` is the Inkling 2 model the drafter was trained on: the same hidden size,
    /// vocabulary and layer visits.
    public func isCompatible(with target: any LanguageModel) -> Bool {
        guard let target = target as? Inkling2 else { return false }
        let text = target.config.textConfiguration
        return text.hiddenSize == configuration.text.hiddenSize
            && text.vocabularySize == configuration.text.vocabularySize
            && text.unpaddedVocabularySize == configuration.text.unpaddedVocabularySize
            && text.layerVisits == configuration.text.layerVisits
    }

    /// `mtp.safetensors` holds the drafter's own parameter names.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

extension Inkling2DraftModel: AdditionalWeightFilesProviding {
    /// The drafter's weights, beside the target's in the same checkpoint.
    public var additionalWeightFiles: [String] { [Inkling2DraftRegistration.weightFile] }

    /// The target's weights, which the target model loads: the drafter reads only its own.
    public var excludedCheckpointNamespaces: [String] { ["model", "lm_head", "vision_tower"] }
}

// MARK: - Registration

/// Registers the DSpark drafter of Inkling 2 checkpoints with the shared drafter factory: the
/// `config.json` of a checkpoint whose `mtp_config` names the rnn_hidden Markov head.
public enum Inkling2DraftRegistration {
    /// The file that holds the drafter's weights in an Inkling 2 checkpoint.
    public static let weightFile = "mtp.safetensors"

    /// The model types of Inkling 2 checkpoints.
    static let modelTypes = ["inkling2", "inkling2_loop"]

    /// Registers the drafter for both Inkling 2 model types with the shared drafter registry.
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerDrafter(
            modelTypes: modelTypes, configuration: Inkling2DraftConfiguration.self,
            matches: { $0.draft.markovHeadType == Inkling2DraftConfiguration.markovHeadType },
            create: { Inkling2DraftModel($0) })
    }
}

extension MTPDrafterRegistration {
    /// The DSpark drafters of Inkling 2 checkpoints, for an ``Inkling2`` target.
    public static let inkling2 = MTPDrafterRegistration(
        registerDrafterTypes: Inkling2DraftRegistration.register,
        draftsForTarget: { $0 is Inkling2 })
}
