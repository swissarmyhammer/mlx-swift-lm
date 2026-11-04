// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

/// MTP speculative decoding for an ``MLXLanguageModel``: the drafter checkpoint, how it loads,
/// the memory policy that admits it, and the shape of each round.
///
/// The model loads the drafter one time, on the first generation or warm-up that needs it, and
/// keeps it beside its weights in the shared model cache; ``MLXLanguageModel/evict()`` drops
/// both. Text, reasoning, allowed-tool and think-then-call turns then speculate whenever the
/// drafter can draft for the loaded model; a drafter that cannot draft for it leaves each turn
/// to plain decode. Schema and required-tool output speculate too: each draft takes the grammar
/// mask, and the output is the output of plain guided decoding.
///
/// A ``memoryPolicy`` weighs the drafter beside the loaded model on each call, as the memory
/// policy of a `ChatSession` does: with ``drafterBytes`` before the drafter loads, and without
/// it on the parameters of the loaded drafter. A drafter the policy does not admit leaves the
/// call to plain decode, which reports
/// `SpeculativeDecodingFallbackReason.memoryBudgetExceeded`, or, under the `.fail` action, the
/// call throws `SpeculativeDecodingMemoryError`. With an estimate, neither loads the drafter.
///
/// ```swift
/// let model = MLXLanguageModel(
///     configuration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-4bit"),
///     capabilities: [.reasoning, .toolCalling],
///     speculativeDecoding: try MLXSpeculativeDecoding(
///         drafterConfiguration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-MTP-4bit"),
///         from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
///         registration: .qwen35Text,
///         drafterBytes: drafterBytes, memoryPolicy: .recommendedWorkingSet),
///     weightsLocation: weightsLocation,
///     load: load)
/// ```
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
public struct MLXSpeculativeDecoding: Sendable {

    /// Loads the drafter container of a drafter configuration.
    ///
    /// The closure must honor cancellation while it is suspended: an eviction of the model
    /// cancels a drafter load in flight.
    public typealias DrafterLoader =
        @Sendable (_ configuration: ModelConfiguration) async throws -> MTPDrafterContainer

    /// The drafter checkpoint.
    public let drafterConfiguration: ModelConfiguration

    /// The tokens of each round: up to `blockSize - 1` drafts and the bonus token. The drafter
    /// caps it at its ``MTPDrafterModel/maximumBlockSize``.
    public let blockSize: Int

    /// The drafter top-1 probability below which a chaining drafter (Qwen MTP) stops a round's
    /// chain early; 0 drafts the whole block every round.
    public let minimumDraftProbability: Float

    /// The estimated resident parameter bytes of the drafter, which ``memoryPolicy`` weighs
    /// before the drafter loads, or `nil` to weigh the parameters of the loaded drafter.
    public let drafterBytes: Int?

    /// The memory policy that admits the drafter beside the loaded model on each call, or `nil`
    /// to speculate whatever the drafter weighs.
    public let memoryPolicy: SpeculativeDecodingMemoryPolicy?

    /// The loader of the drafter that drafts for a target, as the target loaded.
    private let pickDrafterLoader: @Sendable (any LanguageModel) -> DrafterLoader

    /// The identity of the drafter in the model cache, beside the identity of the model.
    var drafterID: String {
        MLXLanguageModel.cacheIdentity(of: drafterConfiguration)
    }

    /// Creates MTP speculative decoding with a drafter that `loadDrafter` loads.
    ///
    /// - Parameters:
    ///   - drafterConfiguration: the drafter checkpoint, as a model id or a local directory.
    ///   - blockSize: the tokens of each round.
    ///   - minimumDraftProbability: the confidence floor of a chaining drafter, from 0 to 1.
    ///   - drafterBytes: the estimated resident parameter bytes of the drafter, which
    ///     `memoryPolicy` weighs before the drafter loads, or `nil`. A negative estimate counts
    ///     as zero.
    ///   - memoryPolicy: the memory policy that admits the drafter on each call, or `nil`.
    ///   - loadDrafter: loads the drafter container of `drafterConfiguration`.
    /// - Throws: ``SpeculativeDecodingConfigurationError/invalidMTPBlockSize(_:)`` when
    ///   `blockSize` is smaller than 2, and
    ///   ``SpeculativeDecodingConfigurationError/invalidMTPMinimumDraftProbability(_:)`` when
    ///   `minimumDraftProbability` is not from 0 to 1.
    public init(
        drafterConfiguration: ModelConfiguration,
        blockSize: Int = SpeculativeDecodingConfig.defaultMTPBlockSize,
        minimumDraftProbability: Float =
            SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability,
        drafterBytes: Int? = nil,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        loadDrafter: @escaping DrafterLoader
    ) throws {
        try self.init(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability,
            drafterBytes: drafterBytes, memoryPolicy: memoryPolicy,
            pickDrafterLoader: { _ in loadDrafter })
    }

    /// Creates MTP speculative decoding with a drafter that the factory of `registration`
    /// loads from `drafterConfiguration`.
    ///
    /// The load first registers the drafter model types of `registration` with the registry
    /// of its factory. Pass `.qwen35Text` of MLXLLM for a Qwen3.5-family text target,
    /// `.qwen35VLM` of MLXVLM for a target that `VLMModelFactory` loaded, or
    /// `.matchingTarget(.qwen35Text, .qwen35VLM)` to load the drafter of the loaded target
    /// however it loaded: `loadModelContainer` loads a Qwen3.5-family target as a
    /// vision-language model whenever MLXVLM is linked. A drafter that cannot draft for the
    /// loaded target leaves each turn to plain decode, and the completion information and the
    /// log state why.
    ///
    /// - Parameters:
    ///   - drafterConfiguration: the drafter checkpoint, as a model id or a local directory.
    ///   - downloader: the downloader that fetches a remote checkpoint.
    ///   - tokenizerLoader: the tokenizer loader the factory takes; an MTP drafter borrows the
    ///     tokenizer of its target, so the load reads no tokenizer.
    ///   - registration: the factory that loads the drafter, and the registration of its
    ///     drafter model types that runs before the load.
    ///   - blockSize: the tokens of each round.
    ///   - minimumDraftProbability: the confidence floor of a chaining drafter, from 0 to 1.
    ///   - drafterBytes: the estimated resident parameter bytes of the drafter, which
    ///     `memoryPolicy` weighs before the drafter loads, or `nil`. A negative estimate counts
    ///     as zero.
    ///   - memoryPolicy: the memory policy that admits the drafter on each call, or `nil`.
    /// - Throws: the errors of
    ///   ``init(drafterConfiguration:blockSize:minimumDraftProbability:drafterBytes:memoryPolicy:loadDrafter:)``.
    public init(
        drafterConfiguration: ModelConfiguration,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        registration: MTPDrafterRegistration,
        blockSize: Int = SpeculativeDecodingConfig.defaultMTPBlockSize,
        minimumDraftProbability: Float =
            SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability,
        drafterBytes: Int? = nil,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) throws {
        try self.init(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability,
            drafterBytes: drafterBytes, memoryPolicy: memoryPolicy,
            pickDrafterLoader: { target in
                registration.drafterLoader(
                    for: target, from: downloader, using: tokenizerLoader)
            })
    }

    /// Creates MTP speculative decoding whose target picks the loader of its drafter, after
    /// checking the block size and the confidence floor.
    private init(
        drafterConfiguration: ModelConfiguration, blockSize: Int, minimumDraftProbability: Float,
        drafterBytes: Int?, memoryPolicy: SpeculativeDecodingMemoryPolicy?,
        pickDrafterLoader: @escaping @Sendable (any LanguageModel) -> DrafterLoader
    ) throws {
        try SpeculativeDecodingConfig.validate(mtpBlockSize: blockSize)
        try SpeculativeDecodingConfig.validate(minimumDraftProbability: minimumDraftProbability)
        self.drafterConfiguration = drafterConfiguration
        self.blockSize = blockSize
        self.minimumDraftProbability = minimumDraftProbability
        self.drafterBytes = drafterBytes.map { max(0, $0) }
        self.memoryPolicy = memoryPolicy
        self.pickDrafterLoader = pickDrafterLoader
    }

    /// The loader of the drafter that drafts for `target`, as `target` loaded.
    ///
    /// - Parameter target: the loaded model the drafter speculates for.
    /// - Returns: the loader.
    func drafterLoader(for target: any LanguageModel) -> DrafterLoader {
        pickDrafterLoader(target)
    }

    /// Whether ``memoryPolicy`` admits the drafter beside `target` before the drafter loads.
    ///
    /// The policy weighs the parameters of `target` and ``drafterBytes``. With no policy, or
    /// no estimate to weigh, the drafter may load, and
    /// ``admitsLoadedDrafter(_:beside:)`` decides on the loaded drafter.
    ///
    /// - Parameter target: the loaded model the drafter speculates for.
    /// - Returns: `false` when the policy does not admit the estimate, else `true`.
    /// - Throws: `SpeculativeDecodingMemoryError` when the policy does not admit the estimate
    ///   and its action is `.fail`.
    func admitsDrafterBeforeLoading(beside target: any LanguageModel) throws -> Bool {
        guard let memoryPolicy, let drafterBytes else { return true }
        return try memoryPolicy.admitsSpeculation(
            mainModelBytes: SpeculativeDecodingMemoryPolicy.modelWeightBytes(target),
            draftModelBytes: drafterBytes)
    }

    /// Whether ``memoryPolicy`` admits the loaded `drafter` beside `target`, when it had no
    /// estimate to weigh before the load.
    ///
    /// - Parameters:
    ///   - drafter: the loaded drafter.
    ///   - target: the loaded model the drafter speculates for.
    /// - Returns: `false` when the policy weighed the parameters of both models and does not
    ///   admit them, else `true`.
    /// - Throws: `SpeculativeDecodingMemoryError` when the policy does not admit the drafter and
    ///   its action is `.fail`.
    func admitsLoadedDrafter(
        _ drafter: any MTPDrafterModel, beside target: any LanguageModel
    ) throws -> Bool {
        guard let memoryPolicy, drafterBytes == nil else { return true }
        return try memoryPolicy.admitsSpeculation(
            mainModelBytes: SpeculativeDecodingMemoryPolicy.modelWeightBytes(target),
            draftModelBytes: SpeculativeDecodingMemoryPolicy.modelWeightBytes(drafter))
    }

    /// The speculation of a generation with the drafter of `container`.
    ///
    /// The drafter model leaves its container here, one time for a response, so that the
    /// generations of the response run it inside the target's `ModelContainer.perform` with
    /// no second container lock.
    ///
    /// - Parameter container: the loaded drafter.
    /// - Returns: the speculation.
    func speculation(with container: MTPDrafterContainer) async -> MTPSpeculation {
        let drafter = await container.perform { SendableBox($0.model) }.consume()
        return MTPSpeculation(
            drafter: drafter, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
