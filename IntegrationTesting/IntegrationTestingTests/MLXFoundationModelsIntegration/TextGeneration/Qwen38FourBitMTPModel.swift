// Copyright © 2026 Apple Inc.
//
// The Qwen3.8-27B-4bit target and its 4-bit MTP head, as the FoundationModels MTP suites load
// them.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

@testable import MLXFoundationModels

/// Qwen3.8-27B-4bit as a FoundationModels language model, with its 4-bit MTP head or without it.
enum Qwen38FourBitMTPModel {

    /// The id of the target.
    static let targetID = "mlx-community/Qwen3.8-27B-4bit"

    /// The id of the 4-bit MTP head of the target.
    static let drafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"

    /// Whether the target and its drafter are local; the suites download nothing.
    static let isLocal =
        hfSnapshotDir(modelId: targetID) != nil && hfSnapshotDir(modelId: drafterID) != nil

    /// The largest gap of plain decode's two best logits at which MTP may write the second one.
    static let nearTieLogitGap: Float = 1e-3

    /// The two best logits of one row.
    private static let topCandidateCount = 2

    /// Where MTP first parts from plain decode, and the two best logits of an uncached forward
    /// pass of the target over the tokens both decodes share.
    ///
    /// A greedy MTP round verifies its drafts in one batched forward pass, and plain decode
    /// feeds one token at a time, thus the two can round a near tie apart.
    struct Divergence: CustomStringConvertible {

        /// The index of the first token where the two parts.
        let step: Int

        /// The token MTP wrote there, or `nil` when MTP wrote no token there.
        let mtpToken: Int?

        /// The two best tokens of the forward pass, best first.
        let topTokens: [Int]

        /// The logits of ``topTokens``.
        let topLogits: [Float]

        /// Whether MTP wrote the second best token, within ``nearTieLogitGap`` of the best.
        var mtpWroteANearTie: Bool {
            mtpToken == topTokens[1] && topLogits[0] - topLogits[1] < nearTieLogitGap
        }

        /// Whether MTP wrote the best token of the uncached forward pass, thus plain decode's
        /// cached forward passes rounded the other way.
        var mtpWroteTheArgmax: Bool { mtpToken == topTokens[0] }

        var description: String {
            """
            MTP parts from plain decode at token \(step): MTP \(mtpToken ?? -1), plain best \
            \(topTokens) with logits \(topLogits)
            """
        }
    }

    /// Where `mtp` first parts from `plain`.
    ///
    /// - Parameters:
    ///   - mtp: The tokens MTP wrote after `context`.
    ///   - plain: The tokens plain decode wrote after `context`.
    ///   - context: The tokens both decodes followed.
    /// - Returns: The divergence, or `nil` when the two are the same.
    /// - Throws: The error of the model load.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    static func divergence(mtp: [Int], plain: [Int], context: [Int]) async throws -> Divergence? {
        guard mtp != plain else { return nil }
        let comparedCount = Swift.min(mtp.count, plain.count)
        let step = (0 ..< comparedCount).first { mtp[$0] != plain[$0] } ?? comparedCount
        let container = try await make(speculates: false).loadContainer()
        let (topTokens, topLogits) = await container.perform { modelContext in
            let tokens = context + plain.prefix(step)
            let logits = modelContext.model(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis, 0...]),
                cache: nil, state: nil
            ).logits[0, -1].asType(.float32)
            let order = argSort(-logits)[..<topCandidateCount]
            return (order.asArray(Int.self), logits[order].asArray(Float.self))
        }
        return Divergence(
            step: step, mtpToken: step < mtp.count ? mtp[step] : nil, topTokens: topTokens,
            topLogits: topLogits)
    }

    /// The two best tokens of one row of logits, best first, with their log probabilities: the
    /// gap of two log probabilities of one row is the gap of their logits.
    ///
    /// - Parameter logits: The logits of one row, over the vocabulary.
    /// - Returns: The two candidates.
    static func topCandidates(of logits: MLXArray) -> [GenerateTokenLogProbability] {
        let row = logits.asType(.float32)
        let logProbabilities = row - logSumExp(row)
        let order = argSort(-logProbabilities)[..<topCandidateCount]
        return zip(order.asArray(Int.self), logProbabilities[order].asArray(Float.self))
            .map { GenerateTokenLogProbability(token: $0, logProbability: $1) }
    }

    /// Drops the reasoning protocol, so that the executor renders the plain prompt a suite can
    /// render again to read the logits of plain decode.
    struct PlainPromptResolver: ModelConfigurationResolver {
        func resolve(
            _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
        ) -> ModelConfiguration {
            var resolved = configuration
            resolved.reasoningConfig = nil
            return resolved
        }
    }

    /// How the target loads, and thus which MTP head drafts for it.
    enum LoadPath: String, CaseIterable, Sendable {

        /// `LLMModelFactory` loads the text target, and the text MTP head drafts for it.
        case text

        /// `loadModelContainer` loads the target as the vision-language `Qwen35`, as the
        /// FoundationModels router of the ACP agent loads it, and
        /// `.matchingTarget(.qwen35Text, .qwen35VLM)` picks the MTP head of the loaded target.
        case router

        /// The registration of the MTP head that drafts for the target of this path.
        var drafterRegistration: MTPDrafterRegistration {
            switch self {
            case .text: .qwen35Text
            case .router: .matchingTarget(.qwen35Text, .qwen35VLM)
            }
        }

        /// The loader of the target of this path.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        var containerLoader: MLXLanguageModel.ContainerLoader {
            switch self {
            case .text:
                { configuration, progress in
                    try await LLMModelFactory.shared.loadContainer(
                        from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                        configuration: configuration, progressHandler: progress)
                }
            case .router: testLoad()
            }
        }
    }

    /// The target, with the MTP head or without it. Every model of the target shares the one
    /// loaded container, whatever its load path: a suite that loads the target two ways must
    /// evict the container of one path before it loads the other.
    ///
    /// The text path is the default, as the MTP benchmarks load it: the text MTP head drafts
    /// for the text target, and `loadModelContainer` loads the multimodal one.
    ///
    /// A model that reasons keeps the reasoning protocol of the target and declares `.reasoning`,
    /// thus each turn writes its reasoning apart from its response, and a later turn renders that
    /// reasoning inside the think block of its turn. A model that does not reason drops the
    /// protocol: the executor renders the plain prompt, and the model writes its reasoning as its
    /// response.
    ///
    /// - Parameters:
    ///   - speculates: Whether the model drafts with the MTP head.
    ///   - reasons: Whether the model keeps the reasoning protocol of the target.
    ///   - path: How the target loads.
    /// - Returns: The model.
    /// - Throws: The error of the speculative decoding configuration.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    static func make(
        speculates: Bool, reasons: Bool = false, path: LoadPath = .text
    ) throws -> MLXLanguageModel {
        let resolver: any ModelConfigurationResolver =
            reasons ? DefaultConfigurationResolver() : PlainPromptResolver()
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: targetID),
            capabilities: reasons ? [.reasoning] : [],
            configurationResolver: resolver,
            speculativeDecoding: speculates
                ? try MLXSpeculativeDecoding(
                    drafterConfiguration: ModelConfiguration(id: drafterID),
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    registration: path.drafterRegistration)
                : nil,
            weightsLocation: testWeightsLocation(modelID:),
            load: path.containerLoader)
    }

    /// The target of one load path with its MTP head, loaded as the FoundationModels executor
    /// loads them, for a benchmark that drives the target and the drafter itself.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    struct LoadedWithDrafter {

        /// The model, which a benchmark evicts when it is done.
        let model: MLXLanguageModel

        /// The speculative decoding of the model: its block size and its confidence floor.
        let speculation: MLXSpeculativeDecoding

        /// The container of the target.
        let container: ModelContainer

        /// The MTP head of the target.
        let drafter: PerformBox<any MTPDrafterModel>
    }

    /// Loads the target of `path` that reasons, and the MTP head that drafts for it.
    ///
    /// - Parameter path: How the target loads.
    /// - Returns: The model, its container and its drafter.
    /// - Throws: The error of the speculative decoding configuration or of a load.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    static func loadWithDrafter(path: LoadPath) async throws -> LoadedWithDrafter {
        let model = try make(speculates: true, reasons: true, path: path)
        let speculation = try #require(model.speculativeDecoding)
        let container = try await model.loadContainer()
        let loadDrafter = await container.perform { speculation.drafterLoader(for: $0.model) }
        let drafterContainer = try await loadDrafter(speculation.drafterConfiguration)
        let drafter = await drafterContainer.perform { PerformBox($0.model) }
        return LoadedWithDrafter(
            model: model, speculation: speculation, container: container, drafter: drafter)
    }
}

/// A value a benchmark hands across a `perform` boundary.
///
/// The benchmark reads the boxed value only inside one `ModelContainer.perform` of the target at
/// a time, and its suite runs serially, thus no two tasks touch the value at once.
final class PerformBox<Value>: @unchecked Sendable {

    /// The boxed value.
    let value: Value

    /// Boxes `value`.
    init(_ value: Value) { self.value = value }
}

#endif
