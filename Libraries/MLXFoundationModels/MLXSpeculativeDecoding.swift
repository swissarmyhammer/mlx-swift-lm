// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

/// MTP speculative decoding for an ``MLXLanguageModel``: the drafter checkpoint, how it loads,
/// and the shape of each round.
///
/// The model loads the drafter one time, on the first generation or warm-up that needs it, and
/// keeps it beside its weights in the shared model cache; ``MLXLanguageModel/evict()`` drops
/// both. Text, reasoning, allowed-tool and think-then-call turns then speculate whenever the
/// drafter can draft for the loaded model; a drafter that cannot draft for it leaves each turn
/// to plain decode. Schema and required-tool output speculate too: each draft takes the grammar
/// mask, and the output is the output of plain guided decoding.
///
/// ```swift
/// let model = MLXLanguageModel(
///     configuration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-4bit"),
///     capabilities: [.reasoning, .toolCalling],
///     speculativeDecoding: try MLXSpeculativeDecoding(
///         drafterConfiguration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-MTP-4bit"),
///         from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
///         registration: .qwen35Text),
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

    /// Loads the drafter.
    let loadDrafter: DrafterLoader

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
        loadDrafter: @escaping DrafterLoader
    ) throws {
        try SpeculativeDecodingConfig.validate(mtpBlockSize: blockSize)
        try SpeculativeDecodingConfig.validate(minimumDraftProbability: minimumDraftProbability)
        self.drafterConfiguration = drafterConfiguration
        self.blockSize = blockSize
        self.minimumDraftProbability = minimumDraftProbability
        self.loadDrafter = loadDrafter
    }

    /// Creates MTP speculative decoding with a drafter that the factory of `registration`
    /// loads from `drafterConfiguration`.
    ///
    /// The load first registers the drafter model types of `registration` with the registry
    /// of its factory. Pass `.qwen35Text` of MLXLLM for a Qwen3.5-family text target, or
    /// `.qwen35VLM` of MLXVLM for a target that `VLMModelFactory` loaded.
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
    /// - Throws: the errors of
    ///   ``init(drafterConfiguration:blockSize:minimumDraftProbability:loadDrafter:)``.
    public init(
        drafterConfiguration: ModelConfiguration,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        registration: MTPDrafterRegistration,
        blockSize: Int = SpeculativeDecodingConfig.defaultMTPBlockSize,
        minimumDraftProbability: Float =
            SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability
    ) throws {
        try self.init(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability
        ) { configuration in
            try await registration.loadContainer(
                from: downloader, using: tokenizerLoader, configuration: configuration)
        }
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
