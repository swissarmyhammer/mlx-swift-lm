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
    /// - Returns: The model.
    func makeLanguageModel(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        resolver: any ModelConfigurationResolver = DefaultConfigurationResolver(),
        eosTokenIds: Set<Int> = [], tokenizer: (any Tokenizer)? = nil,
        toolCallFormat: ToolCallFormat? = nil, speculativeDecoding: MLXSpeculativeDecoding? = nil
    ) -> MLXLanguageModel {
        var context = makeModelContext(eosTokenIds: eosTokenIds)
        context.configuration.toolCallFormat = toolCallFormat
        if let tokenizer {
            context.tokenizer = tokenizer
            context.processor = ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: context.configuration)
        }
        let container = ModelContainer(context: context)
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: "probe/qwen35-mtp-fixture-\(UUID().uuidString)"),
            capabilities: capabilities,
            configurationResolver: resolver,
            speculativeDecoding: speculativeDecoding,
            weightsLocation: { _ in weights },
            load: { _, _ in container })
    }

    /// Speculative decoding with the drafter of a fixture, in rounds of ``blockSize``.
    ///
    /// Each load makes a new fixture and hands out its drafter, thus `log` sees every load the
    /// model cache asks for.
    ///
    /// - Parameter log: Records each load, or `nil`.
    /// - Returns: The speculative decoding.
    /// - Throws: The error of the configuration when the drafter refuses the block size.
    static func makeSpeculativeDecoding(
        log: DrafterLoadLog? = nil
    ) throws -> MLXSpeculativeDecoding {
        try MLXSpeculativeDecoding(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize
        ) { configuration in
            let container = MTPDrafterContainer(
                context: MTPDrafterContext(
                    configuration: configuration, model: try Qwen35MTPFixture().drafter))
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
        try MLXSpeculativeDecoding(
            drafterConfiguration: drafterConfiguration, blockSize: blockSize
        ) { configuration in
            MTPDrafterContainer(
                context: MTPDrafterContext(
                    configuration: configuration, model: try Qwen35VLMMTPFixture().drafter))
        }
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
            ? try MLXSpeculativeDecoding(
                drafterConfiguration: Qwen35MTPFixture.drafterConfiguration,
                blockSize: Qwen35MTPFixture.blockSize
            ) { configuration in
                MTPDrafterContainer(
                    context: MTPDrafterContext(
                        configuration: configuration,
                        model: try Qwen35VLMMTPFixture(readsImages: true).drafter))
            } : nil
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
