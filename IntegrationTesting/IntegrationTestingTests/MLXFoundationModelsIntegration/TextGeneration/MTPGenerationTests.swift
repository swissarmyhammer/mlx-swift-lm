// Copyright © 2026 Apple Inc.
//
// A greedy FoundationModels session on Qwen3.8-27B-4bit, with its 4-bit MTP head and without
// it. Run it in Release, or the throughput measures the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPGenerationTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import BenchmarkHelpers
import Foundation
import FoundationModels
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Synchronization
import os
import Testing
import Tokenizers

@testable import MLXFoundationModels

/// The id of the target.
private let qwen38FourBitID = "mlx-community/Qwen3.8-27B-4bit"

/// The id of the 4-bit MTP head of the target.
private let qwen38FourBitMTPDrafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"

/// Whether the target and its drafter are local; the suite downloads nothing.
private let qwen38FourBitMTPIsLocal =
    hfSnapshotDir(modelId: qwen38FourBitID) != nil
    && hfSnapshotDir(modelId: qwen38FourBitMTPDrafterID) != nil

/// A FoundationModels session with MTP writes the tokens a session without it writes, and
/// decodes faster.
@Suite(.serialized, .timeLimit(.minutes(30)))
struct MTPGenerationTests {

    /// The tokens each turn generates.
    private static let generatedTokenCount = 256

    /// The tokens of the warm-up turn that compiles the kernels of each model.
    private static let warmUpTokenCount = 8

    /// The largest gap of plain decode's two best logits at which MTP may write the second one.
    private static let nearTieLogitGap: Float = 1e-3

    /// The two best logits of one row.
    private static let topCandidateCount = 2

    /// The prompt whose decode tok/s the test holds above plain decode.
    private static let mathPrompt = MTPBenchmarkPrompt.defaults.first { $0.name == "math" }

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP GENERATION:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPGeneration")

    /// What one turn wrote, and the completion of its generation.
    private struct Turn {
        let text: String
        let completion: GenerateCompletionInfo
    }

    /// Records the completion events of a turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class CompletionLog: Sendable {
        private let completions = Mutex<[GenerateCompletionInfo]>([])

        /// Adds the completion of `event`, when it is one.
        func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
            if case .completion(let info) = event {
                completions.withLock { $0.append(info) }
            }
        }

        /// The last completion so far.
        var last: GenerateCompletionInfo? { completions.withLock { $0.last } }
    }

    /// Drops the reasoning protocol, so that the executor renders the plain prompt this test
    /// renders again to read the logits of plain decode.
    private struct PlainPromptResolver: ModelConfigurationResolver {
        func resolve(
            _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
        ) -> ModelConfiguration {
            var resolved = configuration
            resolved.reasoningConfig = nil
            return resolved
        }
    }

    /// The target, with the MTP head or without it. Both share the one loaded container.
    ///
    /// `LLMModelFactory` loads the target, as the MTP benchmarks do: the text MTP head drafts
    /// for the text target, and `loadModelContainer` would load the multimodal one.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModel(speculates: Bool) throws -> MLXLanguageModel {
        MLXLanguageModel(
            configuration: ModelConfiguration(id: qwen38FourBitID),
            capabilities: [],
            configurationResolver: PlainPromptResolver(),
            speculativeDecoding: speculates
                ? try MLXSpeculativeDecoding(
                    drafterConfiguration: ModelConfiguration(id: qwen38FourBitMTPDrafterID),
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    registration: .qwen35Text)
                : nil,
            weightsLocation: testWeightsLocation(modelID:),
            load: { configuration, progress in
                try await LLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    configuration: configuration, progressHandler: progress)
            })
    }

    /// Runs one uncached greedy turn of a new session of `model`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func respond(
        _ model: MLXLanguageModel, to prompt: String, tokens: Int
    ) async throws -> Turn {
        let log = CompletionLog()
        let observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void = {
            log.record($0)
        }
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)
        let text = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
            try await MLXLanguageModel.Executor.$generationObserver.withValue(observer) {
                try await session.respond(
                    to: prompt,
                    options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: tokens)
                ).content
            }
        }
        return Turn(text: text, completion: try #require(log.last))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: qwen38FourBitMTPIsLocal))
    func greedySessionWritesThePlainTokensFaster() async throws {
        let prompt = try #require(Self.mathPrompt).text
        let plainModel = try makeModel(speculates: false)
        let mtpModel = try makeModel(speculates: true)
        _ = try await respond(plainModel, to: prompt, tokens: Self.warmUpTokenCount)
        _ = try await respond(mtpModel, to: prompt, tokens: Self.warmUpTokenCount)

        let plain = try await respond(plainModel, to: prompt, tokens: Self.generatedTokenCount)
        let mtp = try await respond(mtpModel, to: prompt, tokens: Self.generatedTokenCount)
        let measurement = """
            \(Self.measurementPrefix) math: MTP \(mtp.completion.tokensPerSecond) tok/s \
            (accepted \(mtp.completion.acceptedDraftTokens ?? 0) of \
            \(mtp.completion.proposedDraftTokens ?? 0)), plain \
            \(plain.completion.tokensPerSecond) tok/s
            """
        Self.measurementLog.info("\(measurement, privacy: .public)")

        #expect((mtp.completion.proposedDraftTokens ?? 0) > 0)
        #expect(mtp.completion.passthroughReason == nil)
        try await expectPlainTokens(mtp: mtp.text, plain: plain.text, prompt: prompt)
        #expect(mtp.completion.tokensPerSecond > plain.completion.tokensPerSecond)
        await mtpModel.evict()
    }

    /// Expects `mtp` to write the tokens of `plain`, or to part from them only where plain
    /// decode's two best logits are within ``nearTieLogitGap`` and MTP took the second.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectPlainTokens(mtp: String, plain: String, prompt: String) async throws {
        guard mtp != plain else { return }
        let container = try await makeModel(speculates: false).loadContainer()
        let tokenizer = await container.tokenizer
        let mtpTokens = tokenizer.encode(text: mtp, addSpecialTokens: false)
        let plainTokens = tokenizer.encode(text: plain, addSpecialTokens: false)
        let comparedCount = Swift.min(mtpTokens.count, plainTokens.count)
        let step =
            (0 ..< comparedCount).first { mtpTokens[$0] != plainTokens[$0] } ?? comparedCount
        let candidates = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)]))
            let tokens =
                input.text.tokens.reshaped(-1).asArray(Int.self)
                + plainTokens.prefix(step)
            let logits = context.model(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis, 0...]),
                cache: nil, state: nil
            ).logits[0, -1].asType(.float32)
            let order = argSort(-logits)[..<Self.topCandidateCount]
            return (order.asArray(Int.self), logits[order].asArray(Float.self))
        }
        let (topTokens, topLogits) = candidates
        #expect(
            step < mtpTokens.count && mtpTokens[step] == topTokens[1]
                && topLogits[0] - topLogits[1] < Self.nearTieLogitGap,
            """
            MTP parts from plain decode at token \(step): MTP \
            \(step < mtpTokens.count ? mtpTokens[step] : -1), plain best \(topTokens) with \
            logits \(topLogits)
            """)
    }
}

#endif
