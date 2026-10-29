// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35MTPPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1
        let layerTypes = Array(
            repeating: HybridAttentionSchedule.fullAttention, count: mtpArgs.hiddenLayers)
        mtpArgs.layerTypes = layerTypes
        mtpArgs.resolvedLayerTypes = layerTypes

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// The cache of each predictor layer in a stream of `parameters`, as
    /// ``ShiftedMTPDrafter/makeCache(layerCount:parameters:maximumBlockSize:)`` builds it.
    ///
    /// The drafter cache follows the main full-attention caches one position behind:
    ///
    /// - A rotating main cache (`maxKVSize`) gives a ring of the same window, with a rewind
    ///   reserve for the drafts of the longest round, so the drafter stays bounded and rewinds
    ///   its rejected drafts exactly after the ring wraps.
    /// - A quantized main cache (`kvBits`, a quantizing `KVCachePlan`) leaves this cache in
    ///   full precision (bf16 weights give bf16 rows). It is one layer: about
    ///   2 × 4 KV heads × 256 × 2 B = 4 KB for each token, small next to the main cache.
    ///
    /// - Parameter parameters: The generation parameters of the stream, or `nil` for none.
    /// - Returns: One cache for each predictor layer.
    func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        ShiftedMTPDrafter.makeCache(
            layerCount: layers.count, parameters: parameters,
            maximumBlockSize: GatedDeltaDecodeKernel.mostTokens)
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        hiddenStates previousHidden: MLXArray,
        cache: [KVCache],
        positionOffset: Int
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            let faMask = createAttentionMask(h: hiddenStates, cache: layerCache)
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: nil,
                cache: layerCache,
                positionOffset: positionOffset)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, ResumableMTPDrafterModel,
    StatefulMTPDistributionDrafterModel
{
    public let configuration: Qwen35TextConfiguration

    /// The longest verify the hybrid target rewinds in place: the most tokens its gated delta
    /// net kernel and its compiled verify schedule hold. A longer round would verify through a
    /// staged round, which runs neither.
    public let maximumBlockSize: Int? = GatedDeltaDecodeKernel.mostTokens
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = false
    private let checkpointPolicy: Qwen35CheckpointPolicy

    /// The type name and a hash of ``configuration``, which decides the drafter cache layers.
    public let drafterID: MTPDrafterID

    /// The hidden size of ``configuration``: the width of the target hidden rows and of the
    /// predictor's seed hidden row.
    public var hiddenSize: Int? { configuration.hiddenSize }

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public convenience init(
        _ configuration: Qwen35TextConfiguration,
        preconvertedNorms: Bool = false
    ) {
        self.init(
            configuration,
            checkpointPolicy: .init(
                modelType: configuration.modelType,
                preconvertedNorms: preconvertedNorms))
    }

    private init(
        _ configuration: Qwen35TextConfiguration, checkpointPolicy: Qwen35CheckpointPolicy
    ) {
        self.configuration = configuration
        self.checkpointPolicy = checkpointPolicy
        self.drafterID = MTPDrafterID(
            drafterType: Qwen35MTPDraftModel.self, configuration: configuration)
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false
    ) {
        self.init(
            configuration.textConfig,
            checkpointPolicy: .init(
                modelType: configuration.modelType,
                preconvertedNorms: preconvertedNorms))
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        let targetConfiguration: Qwen35TextConfiguration
        switch target {
        case let target as Qwen35Model:
            targetConfiguration = target.languageModel.configuration
        case let target as Qwen35TextModel:
            targetConfiguration = target.configuration
        default:
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Qwen35MTPDraftModel",
                expected: "a Qwen3.5 text target",
                actual: String(describing: type(of: target))
            )
        }

        guard
            targetConfiguration.hiddenSize == configuration.hiddenSize,
            targetConfiguration.vocabularySize == configuration.vocabularySize
        else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Qwen35MTPDraftModel",
                expected: "a Qwen3.5 text target with hidden size \(configuration.hiddenSize) "
                    + "and vocabulary size \(configuration.vocabularySize)",
                actual: "\(type(of: target)) with hidden size \(targetConfiguration.hiddenSize) "
                    + "and vocabulary size \(targetConfiguration.vocabularySize)"
            )
        }
    }

    /// A drafter state whose cache follows the main caches of `parameters`: see
    /// ``Qwen35MTPPredictor/newCache(parameters:)``.
    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache(parameters: parameters))
    }

    public func appendDrafterState(
        target: any LanguageModel,
        tokens: MLXArray,
        targetHidden: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool {
        let tokens = normalizedMTPTokenBatch(tokens)
        let count = tokens.dim(-1)
        guard count > 0, targetHidden.dim(1) >= count, state.proposalAppended == 0,
            state.isAligned
        else { return false }

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(tokens),
            hiddenStates: targetHidden.dim(1) == count
                ? targetHidden : targetHidden[0..., ..<count, 0...],
            cache: state.cache, positionOffset: state.nextPosition)
        state.nextPosition += count
        let seedHidden = mtpHidden[0..., (-1)..., 0...]
        state.seedHidden = seedHidden
        state.seedToken = sampleMTPSeed(
            hidden: seedHidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
            sampler: sampler)
        return state.isAligned
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    /// Drafts the seed the last commit left, then chains the predictor on its own hidden rows for
    /// the rest of the round; with no seed, chains every draft from `lastToken`.
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        shiftedDrafter(target).draftTokens(
            lastToken: lastToken, lastHidden: lastHidden, queryOffset: queryOffset,
            blockSize: blockSize, sampler: sampler, state: &state)
    }

    /// Draws the drafts of a round with their processed logits: the seed the last commit left,
    /// drawn again from its hidden row, then the drafts chained after it, or else a chain from
    /// `lastToken`.
    public func draftBlockWithLogits(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        processor: (any LogitProcessor)?,
        sampler: any LogitSampler
    ) -> MTPDraftBlockOutput {
        shiftedDrafter(target).draftsWithLogits(
            lastToken: lastToken, lastHidden: lastHidden, queryOffset: queryOffset,
            blockSize: blockSize, processor: processor, sampler: sampler, state: &state)
    }

    /// The predictor of this drafter over the embedding and the LM head of `target`.
    private func shiftedDrafter(_ target: any LanguageModel) -> ShiftedMTPDrafter {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        return ShiftedMTPDrafter(
            targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
            inputEmbedding: mtp.embedTokens ?? targetEmbedTokens
        ) { [mtp] inputsEmbeds, hiddenStates, cache, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds, hiddenStates: hiddenStates, cache: cache,
                positionOffset: positionOffset)
        }
    }

    public var excludedCheckpointNamespaces: [String] {
        checkpointPolicy.excludedCheckpointNamespaces
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try checkpointPolicy.prepare(
            checkpoint, mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try sanitize(weights: weights, metadata: [:])
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws -> [String:
        MLXArray]
    {
        try prepareCheckpoint(.init(weights: weights, metadata: metadata)).weights
    }

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        if let model = target as? Qwen35Model {
            return (model.languageModel.model.embedTokens, model.languageModel.lmHead)
        }
        if let model = target as? Qwen35TextModel {
            return (model.model.embedTokens, model.lmHead)
        }
        fatalError(
            "Qwen35MTPDraftModel requires a Qwen35 target, got \(type(of: target))")
    }
}

extension Qwen35MTPDraftModel: ModelConversionMetadataProvider {
    public var modelConversionMetadata: [String: String] {
        [Qwen35CheckpointPolicy.normMetadataKey: "scale"]
    }
}
