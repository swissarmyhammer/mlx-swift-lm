// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Text model

/// The query, key, value and MLP message queries of each loop core layer, `[layers, 4, D]`.
private final class Inkling2MessageInput: Module {
    @ParameterInfo var query: MLXArray

    init(dimensions: Int, layers: Int) {
        self._query.wrappedValue = MLXArray.zeros(
            [layers, LayerMessage.count, dimensions], dtype: .float32)
        super.init()
    }
}

private final class Inkling2TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockAttentionResidual
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    let config: Inkling2TextConfiguration
    let visits: [Int]

    /// The layer visits of a verify pass after which the part built so far goes to the GPU.
    static let sequentialSubmitInterval = 4

    init(_ config: Inkling2TextConfiguration) {
        self.config = config
        self.visits = config.layerVisits
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Inkling2DecoderLayer(config, layer: $0)
        }
        _finalInput.wrappedValue = Inkling2BlockAttentionResidual(dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if let loop = config.loop {
            // The query, key, value and MLP message queries of each core layer.
            _messageInput.wrappedValue = Inkling2MessageInput(
                dimensions: config.hiddenSize, layers: loop.core.count)
        }
        super.init()
    }

    func embed(_ inputs: MLXArray) -> MLXArray {
        embedNorm(embedTokens(inputs))
    }

    /// The final hidden states before the norm's output, for the rows of `outputRows`.
    func callAsFunction(
        _ embeddings: MLXArray, cache: [KVCache]?, outputRows: Range<Int>? = nil
    ) -> MLXArray {
        forward(embeddings, cache: cache, outputRows: outputRows, taps: []).hidden
    }

    /// The final hidden states for the rows of `outputRows`, and the hidden states of every row
    /// at each tap of `taps`, in its order.
    ///
    /// Tap `i` below the number of layer visits is the partial block sum that enters visit `i`;
    /// the tap equal to the number of visits is the final hidden state. Taps count layer visits,
    /// so a looped checkpoint taps each pass of its core on its own.
    ///
    /// - Parameters:
    ///   - embeddings: the input embeddings, `[B, L, D]`.
    ///   - cache: the cache of each layer visit, or `nil`.
    ///   - outputRows: the rows of the final hidden states, or `nil` for every row. A tap of the
    ///     final hidden state needs every row.
    ///   - taps: the taps to capture.
    /// - Returns: the final hidden states and one `[B, L, D]` array for each tap.
    func forward(
        _ embeddings: MLXArray, cache: [KVCache]?, outputRows: Range<Int>?, taps: [Int]
    ) -> (hidden: MLXArray, taps: [MLXArray]) {
        let finalTap = visits.count
        precondition(
            outputRows == nil || !taps.contains(finalTap),
            "Inkling 2 taps the final hidden state of every row")
        let tapped = Set(taps)
        var captured: [Int: MLXArray] = [:]
        var hidden = embeddings
        let caches = cache?.map { $0 as? CacheList }
        let masks = Inkling2DecodeMasks()
        var convolutionMask: MLXArray?
        if let first = caches?.first??[1] as? ArraysCache {
            convolutionMask = first.makeMask(N: hidden.dim(1))
        }
        // A few tokens of one sequence, a speculative verify pass, run through the decode
        // kernels as rows; each row computes what a decode step at its position computes.
        let sequential = runsSequentialDecode(
            embeddings, caches: caches, convolutionMask: convolutionMask)
        if sequential {
            hidden = hidden.swappedAxes(0, 1)
        }

        var visit = 0
        var blocks: MLXArray?
        /// The blocks after the layers run so far, which open with the first layer.
        func completedBlocks() -> MLXArray {
            guard let blocks else { preconditionFailure("The first layer opens the first block") }
            return blocks
        }
        /// The messages of a loop pass read the previous pass's blocks after the prelude and its
        /// partial block, with the `[layers, 4, D]` queries of the core layers.
        typealias MessageSources = (blocks: MLXArray, partial: MLXArray, queries: MLXArray)
        func run(_ layerRange: Range<Int>, messages: MessageSources?) {
            // One token reads every core layer's messages at once; a chunk reads them per layer.
            let bank = messages.flatMap { sources in
                hidden.dim(1) == 1
                    ? inkling2DepthReadout(
                        blocks: sources.blocks, partial: sources.partial,
                        queries: sources.queries.reshaped(-1, config.hiddenSize),
                        queriesPerGroup: LayerMessage.count) : nil
            }
            for (index, layer) in layerRange.enumerated() {
                var layerMessages = bank.map { ($0, index) }
                if let messages, bank == nil {
                    let readout = inkling2DepthReadout(
                        blocks: messages.blocks, partial: messages.partial,
                        queries: messages.queries[index], queriesPerGroup: LayerMessage.count)
                    layerMessages = (readout, 0)
                }
                if tapped.contains(visit) {
                    captured[visit] = hidden
                }
                (hidden, blocks) = layers[layer](
                    hidden, blocks: blocks, cache: caches?[visit], convolutionMask: convolutionMask,
                    messages: layerMessages, masks: masks, sequential: sequential)
                visit += 1
                if sequential, visit % Self.sequentialSubmitInterval == 0 {
                    // A verify pass waits on the host before it is built, so the GPU would idle
                    // while the CPU builds it. Submitting each part as it is built overlaps them.
                    asyncEval(hidden, completedBlocks())
                }
            }
        }

        if let loop = config.loop, let messageInput {
            run(0 ..< loop.coreStart, messages: nil)
            let preludeHidden = hidden
            let preludeBlocks = completedBlocks()
            let preludeCount = preludeBlocks.dim(0)
            for pass in 0 ..< loop.passes {
                var messages: MessageSources?
                if pass > 0 {
                    messages = (
                        completedBlocks()[(preludeCount + 1)...],
                        unweightedRMSNorm(hidden.asType(.float32)), messageInput.query
                    )
                    hidden = preludeHidden
                    blocks = preludeBlocks
                }
                run(loop.core, messages: messages)
            }
            run(loop.coreEnd ..< config.hiddenLayers, messages: nil)
        } else {
            run(0 ..< config.hiddenLayers, messages: nil)
        }

        var finalBlocks = completedBlocks()
        if sequential {
            hidden = hidden.swappedAxes(0, 1)
            // `[K, L, 1, D]` blocks of the rows as the `[K, 1, L, D]` blocks of one sequence.
            finalBlocks = finalBlocks.swappedAxes(1, headPositionAxis)
            captured = captured.mapValues { $0.swappedAxes(0, 1) }
        }
        if let outputRows {
            hidden = hidden[0..., outputRows, 0...]
            finalBlocks = finalBlocks[0..., 0..., outputRows, 0...]
        }
        let normed = unweightedRMSNorm(hidden.asType(.float32))
        let output = finalInput(finalBlocks, normed)[.ellipsis, 0, 0...]
        let final = norm(output).asType(hidden.dtype)
        captured[finalTap] = final
        return (final, taps.compactMap { captured[$0] })
    }

    /// Whether the tokens of `embeddings` run through the decode kernels as rows of one
    /// sequence: a batch of one with 2 to ``Inkling2/speculativeVerifyPositions`` tokens, unmasked,
    /// whose every layer visit decodes.
    private func runsSequentialDecode(
        _ embeddings: MLXArray, caches: [CacheList?]?, convolutionMask: MLXArray?
    ) -> Bool {
        let length = embeddings.dim(1)
        guard embeddings.dim(0) == 1, length > 1, length <= Inkling2.speculativeVerifyPositions,
            let caches, caches.count == visits.count
        else { return false }
        return caches.allSatisfy { cache in
            cache.map {
                Inkling2DecoderLayer.decodes(
                    embeddings, cache: $0, convolutionMask: convolutionMask)
            } ?? false
        }
    }
}

// MARK: - Model

/// The vision stack, with the parameter path it has in ``Inkling2``.
private final class Inkling2Vision: Module {
    @ModuleInfo(key: "vision_tower") var tower: Inkling2VisionModel

    init(_ vision: Inkling2VisionConfiguration, textHiddenSize: Int) {
        _tower.wrappedValue = Inkling2VisionModel(vision, textHiddenSize: textHiddenSize)
        super.init()
    }
}

/// Inkling 2 Mini and Inkling 2 Small: a multimodal mixture-of-experts model whose text stack
/// can run its middle layers several times per token.
public final class Inkling2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Inkling2VisionModel?
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    public let config: Inkling2Configuration

    /// Long chunks give each of the 128 experts enough rows to fill its matmul tiles. 4096
    /// prefills 4-7% faster than 2048 on an M5 Ultra for about 1 GiB more peak memory.
    static let prefillStepSize = 4096

    /// The most tokens of a speculative verify pass whose convolution windows each layer visit
    /// keeps, thus the pass rewinds in place: the bonus token and up to 15 drafts.
    static let speculativeVerifyPositions = 16

    /// Loads the vision tower on the first image when the factory loads the model without it.
    private let onDemandVision: OnDemandComponent<Inkling2Vision>?

    public var vocabularySize: Int { config.textConfiguration.unpaddedVocabularySize }

    /// Thinking, answers and tool calls arrive in their own message frames.
    public var toolCallFormat: ToolCallFormat? { .inkling }

    public var reasoningConfig: ReasoningConfig? { InklingReasoningProtocol.inkling2 }

    /// The key and value heads of each layer visit: after ``shard(group:)``, the share of this
    /// process.
    public var kvHeads: [Int] {
        config.textConfiguration.layerVisits.map { languageModel.layers[$0].attention.kvHeads }
    }

    public init(_ config: Inkling2Configuration) {
        self.config = config
        let text = config.textConfiguration
        _languageModel.wrappedValue = Inkling2TextModel(text)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
        if let vision = config.visionConfiguration {
            _visionTower.wrappedValue = Inkling2VisionModel(
                vision, textHiddenSize: text.hiddenSize)
            onDemandVision = OnDemandComponent(.vision) {
                Inkling2Vision(vision, textHiddenSize: text.hiddenSize)
            }
        } else {
            onDemandVision = nil
        }
        super.init()
    }

    /// Each layer visit has its own attention cache and the states of its four convolutions.
    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let text = config.textConfiguration
        return try text.layerVisits.map { layer in
            let attention: KVCache =
                text.isSliding(layer)
                ? try makeSlidingWindowKVCache(parameters: parameters, window: text.slidingWindow)
                : try makeAttentionKVCache(parameters: parameters)
            return CacheList(
                attention,
                SpeculativeConvolutionCache(
                    size: ConvolutionSlot.count,
                    maximumVerifyPositions: Self.speculativeVerifyPositions))
        }
    }

    private func vision() throws -> Inkling2VisionModel {
        if let visionTower { return visionTower }
        guard let onDemandVision else { throw VLMError.imageProcessingFailure("no vision tower") }
        return try onDemandVision.load().tower
    }

    /// Token embeddings with the image features spliced in, `[1, L, D]`.
    private func inputEmbeddings(_ tokens: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let embeddings = languageModel.embed(tokens)
        guard let image, let grids = image.frames else { return embeddings }

        let tower = try vision()
        let features = try tower.encode(pixels: image.pixels, grids: grids)
            .asType(embeddings.dtype)
        let positions = tokens.reshaped(-1).asArray(Int.self).enumerated()
            .filter { $0.element == config.imageTokenId }
            .map { Int32($0.offset) }
        guard positions.count == features.dim(0) else {
            throw VLMError.processing(
                "Image features and image tokens do not match: tokens=\(positions.count), "
                    + "features=\(features.dim(0))")
        }
        embeddings[0..., MLXArray(positions), 0...] = features.expandedDimensions(axis: 0)
        return embeddings
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        let text = config.textConfiguration
        // The rows of a verify pass take one matvec each, with the bits of a decode step.
        let rows = hidden.size / hidden.dim(-1)
        let rowwise = rows > 1 && rows <= Self.speculativeVerifyPositions
        var logits =
            (rowwise
            ? rowwiseProjection(hidden.reshaped(rows, 1, -1), by: lmHead)
                .reshaped(Array(hidden.shape.dropLast()) + [-1])
            : lmHead(hidden)).asType(.float32)
        if text.finalLogitSoftcap > 0 {
            logits = tanh(logits / text.finalLogitSoftcap) * text.finalLogitSoftcap
        }
        if text.unpaddedVocabularySize < text.vocabularySize {
            logits = logits[.ellipsis, ..<text.unpaddedVocabularySize]
        }
        return logits
    }

    /// Prefills the prompt in chunks and returns the logits of its last position.
    ///
    /// When `state` asks for MTP drafter state, each chunk but the last hands the drafter taps
    /// of its rows to the state's chunk handler, and the returned state carries the taps of the
    /// last chunk's rows (see ``callAsFunction(_:cache:state:)``).
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let embeddings = try inputEmbeddings(tokens, image: input.image)
        let taps = drafterTaps(state)
        let feed = taps == nil ? nil : state?.mtpPrefillChunkHandler

        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(
            total: total, defaultStepSize: Self.prefillStepSize
        ) { range in
            let chunk = embeddings[0..., range, 0...]
            if let taps, let feed {
                let rows = languageModel.forward(chunk, cache: cache, outputRows: nil, taps: taps)
                feed(tokens[0..., range], concatenatedTaps(rows.taps), nil)
            } else {
                _ = languageModel(chunk, cache: cache, outputRows: 0 ..< 1)
            }
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let remaining = embeddings[0..., processed..., 0...]
        let last = remaining.dim(1)
        defer { prefill.progress?(total, total) }
        guard let taps else {
            let hidden = languageModel(remaining, cache: cache, outputRows: (last - 1) ..< last)
            return .logits(LMOutput(logits: logits(hidden)))
        }
        let (hidden, rows) = languageModel.forward(
            remaining, cache: cache, outputRows: nil, taps: taps)
        return .logits(
            LMOutput(
                logits: logits(hidden[0..., (last - 1)..., 0...]),
                state: drafterOutputState(state, taps: rows)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(languageModel(languageModel.embed(inputs), cache: cache))
    }

    /// The logits of every position and, when `state` asks for MTP drafter state, the drafter
    /// taps of every position in `MLXLMCommon.mtpLastHiddenStatesKey`: the hidden states at
    /// the taps of `MLXLMCommon.mtpTargetLayerIdsKey`, or else at the taps of the checkpoint's
    /// own drafter, concatenated in that order.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let taps = drafterTaps(state) else {
            return LMOutput(logits: self(input.tokens, cache: cache))
        }
        var tokens = input.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let (hidden, rows) = languageModel.forward(
            languageModel.embed(tokens), cache: cache, outputRows: nil, taps: taps)
        return LMOutput(logits: logits(hidden), state: drafterOutputState(state, taps: rows))
    }

    /// The taps a forward pass captures for an MTP drafter, or `nil` when `state` does not ask
    /// for drafter state. Without named taps these are the taps of the checkpoint's drafter, and
    /// without a drafter the final hidden state.
    private func drafterTaps(_ state: LMOutput.State?) -> [Int]? {
        guard state?[mtpEmitFlagKey] == true else { return nil }
        let finalTap = config.textConfiguration.layerVisits.count
        let taps =
            state?[mtpTargetLayerIdsKey] ?? config.mtpConfiguration?.targetLayerIds ?? [finalTap]
        precondition(
            Set(taps).count == taps.count && taps.allSatisfy { (0 ... finalTap).contains($0) },
            "Inkling 2 drafter taps must be unique layer visits or the final hidden state")
        return taps
    }

    /// `state` with the drafter taps of a forward pass and the empty shared K/V of a drafter
    /// that reads hidden states only.
    private func drafterOutputState(_ state: LMOutput.State?, taps rows: [MLXArray])
        -> LMOutput.State
    {
        var result = state ?? LMOutput.State()
        result[mtpLastHiddenStatesKey] = concatenatedTaps(rows)
        result[mtpSharedKVStatesKey] = [:]
        result[mtpSharedKVOffsetsKey] = [:]
        result[mtpSharedKVSourceIndicesKey] = [:]
        return result
    }

    /// The hidden states of the taps of one forward pass side by side, `[B, L, taps * D]`.
    private func concatenatedTaps(_ rows: [MLXArray]) -> MLXArray {
        rows.count == 1 ? rows[0] : concatenated(rows, axis: -1)
    }

    /// The normalized embeddings of `tokens`, the input of the first layer visit, which an MTP
    /// drafter embeds its anchor token with.
    func tokenEmbeddings(_ tokens: MLXArray) -> MLXArray {
        languageModel.embed(tokens)
    }

    /// The vocabulary projection before the softcap, which an MTP drafter shares.
    var vocabularyProjection: Linear { lmHead }

    public func prepare() throws {
        for layer in languageModel.layers {
            try layer.attention.prepareFusedProjections()
            try layer.prepareMessageControllers()
            layer.prepareAttentionNorms()
        }
    }

    /// Maps an oMLX checkpoint onto the Hugging Face layout (`Inkling2OMLXLayout`), strips the
    /// `model.` scope and sets each expert layer to its checkpoint's quantization.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var source = checkpoint.weights
        if Inkling2OMLXLayout.holdsOMLXLayout(source) {
            source = Inkling2OMLXLayout.remap(source, text: config.textConfiguration)
            checkpoint.perLayerQuantization = checkpoint.perLayerQuantization.map {
                Inkling2OMLXLayout.remap($0)
            }
        }
        var weights = [String: MLXArray](minimumCapacity: source.count)
        for (key, value) in source {
            var key = key
            if key.hasPrefix("model.") { key = String(key.dropFirst("model.".count)) }
            weights[key] = value
        }

        let text = config.textConfiguration
        for layer in 0 ..< text.hiddenLayers {
            let path = "language_model.layers.\(layer).mlp.experts"
            guard
                let gateUp = Inkling2ExpertFormat(
                    path: "\(path).gate_up_proj", weights: weights, inputDims: text.hiddenSize),
                let down = Inkling2ExpertFormat(
                    path: "\(path).down_proj", weights: weights,
                    inputDims: text.intermediateSize)
            else { continue }
            let mlp = languageModel.layers[layer].mlp
            if mlp.experts.gateUpProj.format != gateUp || mlp.experts.downProj.format != down {
                let experts = Inkling2Experts(text, gateUp: gateUp, down: down)
                try mlp.update(
                    modules: ModuleChildren(values: ["experts": .value(experts)]), verify: .none)
            }
        }
        checkpoint.weights = weights
        return checkpoint
    }
}

extension Inkling2: SpeculativeCacheRewindModel {
    /// A verify pass keeps the convolution windows of up to `speculativeVerifyPositions` tokens
    /// and rewinds every token but its first, the bonus token.
    public var maximumNativeTargetCacheRewind: Int { Self.speculativeVerifyPositions - 1 }
}

extension Inkling2: TensorParallelShardable {
    /// Splits the attention heads and the inner width of every expert of the text stack
    /// across `group`. The embedding, the LM head, the router, the loop gates and the vision
    /// tower stay whole. The projections fuse again over the shards, and every compiled trace
    /// drops.
    ///
    /// - Parameters:
    ///   - group: The group to split across.
    ///   - hooks: The calls around the split of each distinct text layer.
    public func shardLayers(group: MLXDistributed.Group, hooks: TensorParallelLayerHooks) throws {
        guard try group.size > 1 else { return }
        try languageModel.shardLayers(group: group, hooks: hooks)
        try prepare()
        invalidateCompiledTraces()
    }
}

extension Inkling2TextModel: TensorParallelShardable {
    /// Keeps the share of one process of `group` of every distinct layer. The layers of the
    /// loop core run in every pass, so each splits once. ``Inkling2/shardLayers(group:hooks:)``
    /// then fuses the projections again.
    ///
    /// - Parameters:
    ///   - group: The group to split across.
    ///   - hooks: The calls around the split of each distinct layer.
    func shardLayers(group: MLXDistributed.Group, hooks: TensorParallelLayerHooks) throws {
        let plan = try Inkling2TensorParallelPlan(configuration: config, groupSize: try group.size)
        guard plan.groupSize > 1 else { return }
        try TensorParallel.shardLayers(
            layers, hooks: hooks, makeShards: { try $0.makeShards(plan, group: group) },
            shard: { try $0.shard(plan, with: $1, group: group) })
    }
}

extension Inkling2: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}

extension Inkling2: ModelComponentsProviding {
    public var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [.vision: .modules("vision_tower")]
    }

    public var onDemandComponents: [AnyOnDemandComponent] {
        onDemandVision.map { [$0] } ?? []
    }
}
