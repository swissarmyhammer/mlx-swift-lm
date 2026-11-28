// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Synchronization

@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension Qwen35MTPFixture {

    /// The drafter configuration the speculative decoding of the fixture names.
    static let drafterConfiguration = ModelConfiguration(id: "probe/qwen35-mtp-fixture-drafter")

    /// A FoundationModels language model whose load gives the container of the fixture's
    /// target, so that an executor test can run a `LanguageModelSession` on it.
    ///
    /// The model takes a fresh identity, thus the process-wide model cache keeps it apart
    /// from every other test.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the model available, from
    ///     `makeScriptedWeightsDirectory()`.
    ///   - capabilities: The capabilities the model declares.
    ///   - resolver: The resolver that patches the configuration of each call.
    ///   - eosTokenIds: The tokens that end a generation.
    ///   - tokenizer: The tokenizer that renders the prompts and decodes the replies, or `nil`
    ///     for the fixture's own.
    ///   - toolCallFormat: The response protocol of the target, whose cache-reuse rules the
    ///     executor consults, or `nil` for none.
    ///   - speculativeDecoding: The MTP drafter of the model, or `nil` for plain decode.
    ///   - kvCache: The KV-cache configuration of every generation, or `nil` for full precision.
    /// - Returns: The model.
    func makeLanguageModel(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        resolver: any ModelConfigurationResolver = DefaultConfigurationResolver(),
        eosTokenIds: Set<Int> = [], tokenizer: (any Tokenizer)? = nil,
        toolCallFormat: ToolCallFormat? = nil, speculativeDecoding: MLXSpeculativeDecoding? = nil,
        kvCache: KVCacheConfiguration? = nil
    ) -> MLXLanguageModel {
        Self.makeLanguageModel(
            serving: makeLanguageContext(
                eosTokenIds: eosTokenIds, tokenizer: tokenizer, toolCallFormat: toolCallFormat),
            weights: weights, capabilities: capabilities, resolver: resolver,
            speculativeDecoding: speculativeDecoding, kvCache: kvCache)
    }

    /// A FoundationModels language model whose target runs in an in-process worker, behind a
    /// `ClusterLanguageModel`, with the tokenizer and the configuration of
    /// ``makeLanguageModel(weights:capabilities:resolver:eosTokenIds:tokenizer:toolCallFormat:speculativeDecoding:)``.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the model available.
    ///   - capabilities: The capabilities the model declares.
    ///   - tokenizer: The tokenizer that renders the prompts and decodes the replies, or `nil`
    ///     for the fixture's own.
    ///   - speculativeDecoding: The MTP drafter of the model, or `nil` for plain decode.
    ///   - drafting: Whether rank 0 of the worker holds the drafter of the fixture.
    ///   - followerCount: The ranks other than 0 the worker stands for.
    ///   - kvCache: The KV-cache configuration of every generation, or `nil` for full precision.
    /// - Returns: The model, and the worker that runs its target.
    /// - Throws: The error of the worker handshake.
    func makeClusteredLanguageModel(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        tokenizer: (any Tokenizer)? = nil, speculativeDecoding: MLXSpeculativeDecoding? = nil,
        drafting: Bool = false, followerCount: Int = 0, kvCache: KVCacheConfiguration? = nil
    ) throws -> (model: MLXLanguageModel, worker: InProcessRemoteModelWorker) {
        let worker = makeWorker(drafting: drafting, followerCount: followerCount)
        var context = makeLanguageContext(
            eosTokenIds: [], tokenizer: tokenizer, toolCallFormat: nil)
        context.model = try ClusterLanguageModel.connect(over: worker)
        let model = Self.makeLanguageModel(
            serving: context, weights: weights, capabilities: capabilities,
            resolver: DefaultConfigurationResolver(), speculativeDecoding: speculativeDecoding,
            kvCache: kvCache)
        return (model, worker)
    }

    /// A FoundationModels language model over `target` in the place of the target of the
    /// fixture, with the tokenizer and the input processor of the fixture, thus any model of
    /// the vocabulary of the fixture, local or clustered, runs behind the executor.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the model available.
    ///   - target: The model that each generation runs.
    /// - Returns: The model.
    func makeLanguageModel(weights: URL, target: any MLXLMCommon.LanguageModel) -> MLXLanguageModel
    {
        var context = makeLanguageContext(eosTokenIds: [], tokenizer: nil, toolCallFormat: nil)
        context.model = target
        return Self.makeLanguageModel(
            serving: context, weights: weights, capabilities: [],
            resolver: DefaultConfigurationResolver(), speculativeDecoding: nil)
    }

    /// The model context of the fixture with `tokenizer` and `toolCallFormat`.
    private func makeLanguageContext(
        eosTokenIds: Set<Int>, tokenizer: (any Tokenizer)?, toolCallFormat: ToolCallFormat?
    ) -> ModelContext {
        var context = makeModelContext(eosTokenIds: eosTokenIds)
        context.configuration.toolCallFormat = toolCallFormat
        if let tokenizer {
            context.tokenizer = tokenizer
            context.processor = ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: context.configuration)
        }
        return context
    }

    /// A FoundationModels language model under a fresh identity, whose load gives a container
    /// of `context`.
    private static func makeLanguageModel(
        serving context: ModelContext, weights: URL,
        capabilities: [LanguageModelCapabilities.Capability],
        resolver: any ModelConfigurationResolver, speculativeDecoding: MLXSpeculativeDecoding?,
        kvCache: KVCacheConfiguration? = nil
    ) -> MLXLanguageModel {
        let container = ModelContainer(context: context)
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: "probe/qwen35-mtp-fixture-\(UUID().uuidString)"),
            capabilities: capabilities,
            configurationResolver: resolver,
            speculativeDecoding: speculativeDecoding,
            kvCache: kvCache,
            weightsLocation: { _ in weights },
            load: { _, _ in container })
    }

    /// The target of one fixture as a FoundationModels language model, once plain and once with
    /// the MTP drafter of the fixture.
    struct LanguageModels {

        /// The fixture whose target both models load.
        let fixture: Qwen35MTPFixture

        /// The model that decodes without a drafter.
        let plain: MLXLanguageModel

        /// The model that speculates with the drafter of the fixture.
        let speculative: MLXLanguageModel
    }

    /// Makes a new fixture and its plain and speculative language models, which share every
    /// argument but the speculative decoding.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the models available.
    ///   - capabilities: The capabilities each model declares.
    ///   - resolver: The resolver that patches the configuration of each call.
    ///   - eosTokenIds: The tokens that end a generation.
    ///   - tokenizer: The tokenizer that renders the prompts and decodes the replies, or `nil`
    ///     for the fixture's own.
    /// - Returns: The fixture and its two models.
    /// - Throws: The error of the fixture or of the speculative decoding.
    static func makeLanguageModels(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        resolver: any ModelConfigurationResolver = DefaultConfigurationResolver(),
        eosTokenIds: Set<Int> = [], tokenizer: (any Tokenizer)? = nil
    ) throws -> LanguageModels {
        let fixture = try Qwen35MTPFixture()
        return LanguageModels(
            fixture: fixture,
            plain: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, resolver: resolver,
                eosTokenIds: eosTokenIds, tokenizer: tokenizer),
            speculative: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, resolver: resolver,
                eosTokenIds: eosTokenIds, tokenizer: tokenizer,
                speculativeDecoding: try makeSpeculativeDecoding()))
    }

    /// Speculative decoding with the drafter of a fixture, in rounds of ``blockSize``.
    ///
    /// Each load makes a new fixture and hands out its drafter, thus `log` sees every load the
    /// model cache asks for.
    ///
    /// - Parameters:
    ///   - log: Records each load, or `nil`.
    ///   - drafterBytes: The byte estimate the memory policy weighs before the drafter loads,
    ///     or `nil` for none.
    ///   - memoryPolicy: The memory policy that admits the drafter, or `nil` for none.
    ///   - makeDrafter: Makes the drafter of one load. The default makes the drafter of a new
    ///     fixture.
    /// - Returns: The speculative decoding.
    /// - Throws: The error of the configuration when the drafter refuses the block size.
    static func makeSpeculativeDecoding(
        log: DrafterLoadLog? = nil, drafterBytes: Int? = nil,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        drafter makeDrafter: @escaping @Sendable () throws -> any MTPDrafterModel = {
            try Qwen35MTPFixture().drafter
        }
    ) throws -> MLXSpeculativeDecoding {
        try MLXSpeculativeDecoding(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize,
            drafterBytes: drafterBytes, memoryPolicy: memoryPolicy
        ) { configuration in
            let container = MTPDrafterContainer(
                context: MTPDrafterContext(configuration: configuration, model: try makeDrafter()))
            log?.recordLoad(of: container)
            return container
        }
    }

    /// Speculative decoding whose drafter is the vision-language drafter, which cannot draft for
    /// the text target of the fixture.
    ///
    /// - Returns: The speculative decoding.
    /// - Throws: The error of the configuration when the drafter refuses the block size.
    static func makeIncompatibleSpeculativeDecoding() throws -> MLXSpeculativeDecoding {
        try makeSpeculativeDecoding(drafter: { try Qwen35VLMMTPFixture().drafter })
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension Qwen35VLMMTPFixture {

    /// A FoundationModels language model over the fixture's target, which reads images, with
    /// its MTP drafter or without it.
    ///
    /// The fixture must read images, as `init(seed:readsImages:)` makes it.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the model available.
    ///   - speculates: Whether the model speculates with the fixture's drafter.
    /// - Returns: The model, which declares `.vision`.
    /// - Throws: The error of the configuration when the drafter refuses the block size.
    func makeLanguageModel(weights: URL, speculates: Bool) throws -> MLXLanguageModel {
        let container = makeModelContainer()
        let speculativeDecoding =
            speculates
            ? try Qwen35MTPFixture.makeSpeculativeDecoding(drafter: {
                try Qwen35VLMMTPFixture(readsImages: true).drafter
            }) : nil
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: "probe/qwen35-vlm-mtp-\(UUID().uuidString)"),
            capabilities: [.vision],
            speculativeDecoding: speculativeDecoding,
            weightsLocation: { _ in weights },
            load: { _, _ in container })
    }
}

/// The loads of an MTP drafter, and whether the drafter each load gave is still alive.
///
/// The model cache loads on its own task and the test reads on its task, thus a `Mutex` holds
/// the loads.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class DrafterLoadLog: Sendable {
    private let loads = Mutex<[WeakDrafter]>([])

    /// A loaded drafter container, held weakly so the log keeps none alive.
    private struct WeakDrafter: Sendable {
        weak var container: MTPDrafterContainer?
    }

    /// Records one load that gave `container`.
    ///
    /// - Parameter container: The container the load gave.
    func recordLoad(of container: MTPDrafterContainer) {
        loads.withLock { $0.append(WeakDrafter(container: container)) }
    }

    /// The number of loads so far.
    var loadCount: Int {
        loads.withLock { $0.count }
    }

    /// The number of loaded drafters something still holds.
    var liveDrafterCount: Int {
        loads.withLock { $0.filter { $0.container != nil }.count }
    }
}

#endif
