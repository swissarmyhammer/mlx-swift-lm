// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of LFM2.5: the dense 2.6B model, the sparse 8B-A1B
// model, and the DSpark block drafter on the dense model.
//
// These tests came from `MLXLMTests` (upstream merge 6dc1e4e7), where an
// environment variable that pointed at a downloaded checkpoint gated each one.
// Here each test downloads its pinned checkpoint through the Hugging Face
// downloader (or reuses the cached copy), thus none of them skips.
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`. Use
// `xcodebuild build-for-testing -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
// as the compile evidence for any change to it.

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

@testable import MLXLLM

// MARK: - Pinned checkpoints

/// The bf16 variant of the official dense LFM2.5 2.6B model. The repository
/// holds every precision variant, each in its own subdirectory.
private let lfm25DenseBF16 = ModelConfiguration(
    id: "LiquidAI/LFM2.5-2.6B-MLX",
    revision: "b41f2b65685e95418f1ac809bb022d4f79e1ab27",
    modelSubdirectory: "bf16")

/// The official sparse LFM2.5 8B-A1B model at the revision whose conversion
/// keeps the router gate of layer 2 at 8 bits while the other modules are at
/// 4 bits. A later revision (760bf10c) stores the router gate in float32, thus
/// this revision is the one that exercises mixed-precision routing.
private let lfm25MoEMixedPrecision = ModelConfiguration(
    id: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
    revision: "d16665ca1d23bb47a6641a4b6a96076bd6b5a7b8")

/// The DSpark block drafter that pairs with the dense 2.6B target.
private let lfm25DenseDSpark = ModelConfiguration(
    id: "LiquidAI/LFM2.5-2.6B-DSpark",
    revision: "458cedab07d0f7b2b05700c77e1aa463d43d6f04")

/// The user turn of the DSpark tests. The target's chat template renders it,
/// so the tests exercise the distribution the draft checkpoints were trained
/// on.
private let dsparkQuestion = "What is the capital of France?"

/// The proposal block of the released DSpark drafters is nine tokens. The
/// `blockSize` of the iterator includes the verifier bonus token.
private let lfm25DSparkBlockSize = 10

/// The `temperature` of `generation_config.json` in the pinned dense checkpoint.
private let lfm25DenseTemperature: Float = 0.2

/// The vocabulary of the pinned sparse checkpoint.
private let lfm25MoEVocabularySize = 128_000

/// The decoder layers of the pinned sparse checkpoint, one cache each.
private let lfm25MoELayerCount = 24

/// The bits of the 8-bit router override of the pinned sparse checkpoint.
private let lfm25MoERouterBits = 8

/// The bits of every other quantized module of the pinned sparse checkpoint.
private let lfm25MoEModuleBits = 4

/// The quantization group size of the pinned sparse checkpoint.
private let lfm25MoEGroupSize = 64

/// The tokens of the short forward pass of the load tests.
private let shortPromptLength = 3

/// The token ids of the short forward pass of the load tests: the first
/// ``shortPromptLength`` non-zero ids.
private let shortPromptTokens = (1 ... shortPromptLength).map { Int32($0) }

/// A batch of one sequence of `tokens`, as a model takes it.
private func batch(_ tokens: [Int32]) -> MLXArray {
    MLXArray(tokens).reshaped(1, tokens.count)
}

/// Drains `iterator` and returns the tokens it emitted. `iterator` keeps its
/// final state, thus a caller can read its draft counts afterwards.
private func drain<Iterator: TokenIteratorProtocol>(_ iterator: inout Iterator) -> [Int] {
    var tokens: [Int] = []
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}

// MARK: - Benchmark shape

/// The shape of a forward throughput or DSpark benchmark.
private enum BenchmarkShape {
    /// The tokens of the warm-up prompt.
    static let warmUpPromptTokens = 16

    /// The single-token decode steps of the warm-up.
    static let warmUpDecodeSteps = 4

    /// The token id that fills each benchmark prompt.
    static let promptToken: Int32 = 42

    /// The token id of each benchmark decode step.
    static let decodeToken: Int32 = 43

    /// The tokens of the timed prompt.
    static let promptTokens = 256

    /// The timed single-token decode steps.
    static let decodeTokens = 32

    /// The bytes of one GiB.
    static let bytesPerGiB = 1_073_741_824.0

    /// The tokens of the DSpark greedy parity check.
    static let dsparkParityTokens = 32

    /// The tokens of one untimed DSpark warm-up sample.
    static let dsparkWarmUpTokens = 16

    /// The tokens of one timed DSpark sample.
    static let dsparkMeasuredTokens = 128

    /// The timed DSpark samples of each path; the median counts.
    static let dsparkRepetitions = 3
}

// MARK: - Forward throughput

/// The forward throughput of a model, without tokenizer or UI work.
private struct ForwardThroughput {
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let peakGiB: Double
}

/// Measures the prefill of a ``BenchmarkShape/promptTokens``-token prompt and
/// ``BenchmarkShape/decodeTokens`` single-token decode steps, after a warm-up
/// outside the timed samples.
private func measureForwardThroughput(_ model: any LanguageModel) throws -> ForwardThroughput {
    /// A prompt of `count` filler tokens.
    func prompt(count: Int) -> MLXArray {
        batch([Int32](repeating: BenchmarkShape.promptToken, count: count))
    }
    let decodeStep = batch([BenchmarkShape.decodeToken])

    // Warm Metal kernels and allocator state outside the timed samples.
    let warmCache = try model.newCache(parameters: nil)
    eval(model(prompt(count: BenchmarkShape.warmUpPromptTokens), cache: warmCache))
    for _ in 0 ..< BenchmarkShape.warmUpDecodeSteps {
        eval(model(decodeStep, cache: warmCache))
    }

    Memory.peakMemory = 0
    let promptTokens = BenchmarkShape.promptTokens
    let decodeTokens = BenchmarkShape.decodeTokens
    let cache = try model.newCache(parameters: nil)
    let timedPrompt = prompt(count: promptTokens)

    let prefillMilliseconds = GPUWallTime.milliseconds { eval(model(timedPrompt, cache: cache)) }
    let decodeMilliseconds = GPUWallTime.milliseconds {
        for _ in 0 ..< decodeTokens {
            eval(model(decodeStep, cache: cache))
        }
    }

    return ForwardThroughput(
        prefillTokensPerSecond: GPUWallTime.ratePerSecond(
            promptTokens, milliseconds: prefillMilliseconds),
        decodeTokensPerSecond: GPUWallTime.ratePerSecond(
            decodeTokens, milliseconds: decodeMilliseconds),
        peakGiB: Double(Memory.peakMemory) / BenchmarkShape.bytesPerGiB)
}

/// Expects a finite, positive rate.
private func expectPositiveRate(
    _ rate: Double, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(rate.isFinite && rate > 0, sourceLocation: sourceLocation)
}

/// Measures the forward throughput of `model`, prints it under `label`, and
/// expects a finite, positive prefill and decode rate.
private func benchmarkForward(
    _ model: any LanguageModel, label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let throughput = try measureForwardThroughput(model)
    // swiftlint:disable:next no_direct_standard_out_logs — a benchmark; its output is the measurement.
    print(
        String(
            format: "%@: prefill %.2f tok/s, decode %.2f tok/s, peak %.2f GiB",
            label, throughput.prefillTokensPerSecond, throughput.decodeTokensPerSecond,
            throughput.peakGiB))
    expectPositiveRate(throughput.prefillTokensPerSecond, sourceLocation: sourceLocation)
    expectPositiveRate(throughput.decodeTokensPerSecond, sourceLocation: sourceLocation)
}

// MARK: - Tests

@Suite(.serialized)
struct LFM2CheckpointIntegrationTests {
    private let downloader = #hubDownloader()
    private let tokenizerLoader = #huggingFaceTokenizerLoader()

    /// Downloads the checkpoint of `configuration`, or reuses the cached copy,
    /// and returns the directory that holds its `config.json`.
    private func checkpointDirectory(_ configuration: ModelConfiguration) async throws -> URL {
        try await resolve(
            configuration: configuration, from: downloader, useLatest: false,
            progressHandler: { _ in }
        ).modelDirectory
    }

    /// Loads the dense checkpoint through the factory, which verifies every
    /// real tensor name and shape, then runs one forward pass.
    @Test
    func denseCheckpointLoadsThroughFactory() async throws {
        let directory = try await checkpointDirectory(lfm25DenseBF16)
        let generationData = try Data(
            contentsOf: directory.appending(component: "generation_config.json"))
        let generationConfig = try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: generationData)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: tokenizerLoader)
        let model = try #require(context.model as? LFM2Model)
        let config = model.configuration

        let tokens = batch(shortPromptTokens)
        let logits = model(tokens, cache: try model.newCache(parameters: nil))
        eval(logits)
        #expect(logits.shape == [1, shortPromptTokens.count, config.vocabularySize])
        #expect(context.configuration.toolCallFormat == .lfm2)
        #expect(context.configuration.reasoningConfig?.promptStrategy == .alwaysOn)
        let loadedGenerationConfig = try #require(context.configuration.generationConfig)
        #expect(generationConfig.temperature == lfm25DenseTemperature)
        #expect(loadedGenerationConfig.temperature == generationConfig.temperature)
    }

    /// Measures the forward throughput of the dense bf16 checkpoint.
    @Test
    func denseCheckpointPerformance() async throws {
        let directory = try await checkpointDirectory(lfm25DenseBF16)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM2Configuration.self, from: data)
        let model = LFM2Model(config)
        try await loadWeights(modelDirectory: directory, model: model)

        try benchmarkForward(model, label: "LFM2.5-2.6B bf16")
    }

    /// Loads the mixed-precision sparse checkpoint through the factory, which
    /// verifies every checkpoint tensor, then asserts that the 8-bit router
    /// override of the checkpoint coexists with its 4-bit modules.
    @Test
    func moeCheckpointLoadsWithMixedPrecisionRouting() async throws {
        let directory = try await checkpointDirectory(lfm25MoEMixedPrecision)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: tokenizerLoader)
        let model = try #require(context.model as? LFM2MoEModel)
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let router = try #require(
            modules["model.layers.2.feed_forward.gate"] as? QuantizedLinear)
        let convolution = try #require(
            modules["model.layers.0.conv.in_proj"] as? QuantizedLinear)

        #expect(router.bits == lfm25MoERouterBits)
        #expect(router.groupSize == lfm25MoEGroupSize)
        #expect(convolution.bits == lfm25MoEModuleBits)
        #expect(convolution.groupSize == lfm25MoEGroupSize)
        #expect(context.configuration.toolCallFormat == .lfm2)
        #expect(context.configuration.reasoningConfig?.promptStrategy == .alwaysOn)
        #expect(context.configuration.reasoningConfig?.isSpecialToken == false)

        let cache = try model.newCache(parameters: nil)
        let logits = model(
            batch(shortPromptTokens), cache: cache)
        eval(logits)
        #expect(logits.shape == [1, shortPromptTokens.count, lfm25MoEVocabularySize])
        #expect(cache.count == lfm25MoELayerCount)
    }

    /// Measures the forward throughput of the mixed-precision sparse checkpoint.
    @Test
    func moeCheckpointPerformance() async throws {
        let directory = try await checkpointDirectory(lfm25MoEMixedPrecision)
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: configData)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let model = LFM2MoEModel(config)
        try await loadWeights(
            modelDirectory: directory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        try benchmarkForward(model, label: "LFM2.5-8B-A1B mixed 4/8-bit")
    }

    /// Loads the dense target and its DSpark drafter, then checks that greedy
    /// speculative decoding emits the baseline greedy stream and accepts
    /// draft tokens.
    @Test
    func dsparkCheckpointLoadsAndGenerates() async throws {
        let (target, drafter, input) = try await loadDSparkPair()
        let parameters = GenerateParameters(
            maxTokens: BenchmarkShape.dsparkParityTokens, temperature: 0)
        var baseline = try TokenIterator(
            input: input, model: target.model, parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target.model, drafter: drafter,
            parameters: parameters, blockSize: lfm25DSparkBlockSize)

        let expected = drain(&baseline)
        let tokens = drain(&iterator)

        // swiftlint:disable:next no_direct_standard_out_logs — a benchmark; its output is the measurement.
        print(
            "LFM2 DSpark acceptance: \(iterator.acceptedCount)/\(iterator.proposedCount) "
                + "draft tokens")
        // The bf16 target has no quantized verifier kernels, thus the
        // speculative stream equals the baseline stream token for token.
        #expect(tokens == expected)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        #expect(iterator.acceptedCount > 0)
    }

    /// Times iterator initialization (including prompt prefill) separately
    /// from token consumption, and alternates baseline and speculative samples
    /// to reduce thermal-ordering bias.
    @Test
    func dsparkCheckpointPerformance() async throws {
        let (target, drafter, input) = try await loadDSparkPair()

        /// One timed sample of a decode path.
        struct Sample {
            /// The milliseconds of iterator initialization, including prompt prefill.
            let prefillMilliseconds: Double
            /// The milliseconds of token consumption.
            let decodeMilliseconds: Double
            /// The draft tokens the verifier accepted; 0 for the baseline.
            let accepted: Int
            /// The draft tokens the drafter proposed; 0 for the baseline.
            let proposed: Int
        }

        /// Times one decode path: iterator initialization (including prompt
        /// prefill) as prefill, then the drain of `tokens` tokens as decode.
        func sample<Iterator: TokenIteratorProtocol>(
            tokens: Int,
            makeIterator: (GenerateParameters) throws -> Iterator,
            drafts: (Iterator) -> (accepted: Int, proposed: Int)
        ) throws -> Sample {
            let parameters = GenerateParameters(maxTokens: tokens, temperature: 0)
            let (initialized, prefillMilliseconds) = try GPUWallTime.measure {
                try makeIterator(parameters)
            }
            var iterator = initialized
            let (count, decodeMilliseconds) = GPUWallTime.measure { drain(&iterator).count }
            #expect(count == tokens)
            let (accepted, proposed) = drafts(iterator)
            return Sample(
                prefillMilliseconds: prefillMilliseconds, decodeMilliseconds: decodeMilliseconds,
                accepted: accepted, proposed: proposed)
        }

        func baseline(tokens: Int) throws -> Sample {
            try sample(
                tokens: tokens,
                makeIterator: {
                    try TokenIterator(input: input, model: target.model, parameters: $0)
                },
                // The baseline drafts nothing, thus it proposes and accepts no tokens.
                drafts: { _ in (accepted: 0, proposed: 0) })
        }

        func speculative(tokens: Int) throws -> Sample {
            try sample(
                tokens: tokens,
                makeIterator: {
                    try MTPSpeculativeTokenIterator(
                        input: input, mainModel: target.model, drafter: drafter,
                        parameters: $0, blockSize: lfm25DSparkBlockSize)
                },
                drafts: { iterator in
                    #expect(iterator.passthroughReason == nil)
                    return (accepted: iterator.acceptedCount, proposed: iterator.proposedCount)
                })
        }

        // Compile kernels and realize lazy checkpoint tensors before sampling.
        _ = try baseline(tokens: BenchmarkShape.dsparkWarmUpTokens)
        _ = try speculative(tokens: BenchmarkShape.dsparkWarmUpTokens)
        Memory.clearCache()

        let measuredTokens = BenchmarkShape.dsparkMeasuredTokens
        let repetitions = BenchmarkShape.dsparkRepetitions
        // Each repetition times the baseline, then the speculative path.
        let samples = try (0 ..< repetitions).map { _ in
            (
                baseline: try baseline(tokens: measuredTokens),
                speculative: try speculative(tokens: measuredTokens)
            )
        }
        let baselineSamples = samples.map(\.baseline)
        let speculativeSamples = samples.map(\.speculative)

        let baselineMilliseconds = SampleMedian.of(baselineSamples.map(\.decodeMilliseconds))
        let speculativeMilliseconds = SampleMedian.of(
            speculativeSamples.map(\.decodeMilliseconds))
        let baselinePrefill = SampleMedian.of(baselineSamples.map(\.prefillMilliseconds))
        let speculativePrefill = SampleMedian.of(speculativeSamples.map(\.prefillMilliseconds))
        let representative = try #require(
            speculativeSamples.min {
                abs($0.decodeMilliseconds - speculativeMilliseconds)
                    < abs($1.decodeMilliseconds - speculativeMilliseconds)
            })
        let baselineTokensPerSecond = GPUWallTime.ratePerSecond(
            measuredTokens, milliseconds: baselineMilliseconds)
        let speculativeTokensPerSecond = GPUWallTime.ratePerSecond(
            measuredTokens, milliseconds: speculativeMilliseconds)
        // swiftlint:disable:next no_direct_standard_out_logs — a benchmark; its output is the measurement.
        print(
            String(
                format:
                    "LFM2 DSpark benchmark (block %d, median of %d): baseline %.2f tok/s, DSpark %.2f tok/s, speedup %.2fx, prefill %.1f ms -> %.1f ms, acceptance %d/%d",
                lfm25DSparkBlockSize, repetitions, baselineTokensPerSecond,
                speculativeTokensPerSecond,
                speculativeTokensPerSecond / baselineTokensPerSecond, baselinePrefill,
                speculativePrefill,
                representative.accepted, representative.proposed))

        expectPositiveRate(baselineTokensPerSecond)
        expectPositiveRate(speculativeTokensPerSecond)
        #expect(representative.proposed > 0)
    }

    /// Loads the dense bf16 target and the DSpark drafter that pairs with it,
    /// and renders ``dsparkQuestion`` through the target's chat template.
    private func loadDSparkPair() async throws -> (
        target: ModelContext, drafter: LFM2DSparkDraftModel, input: LMInput
    ) {
        await LFM2DSparkRegistration.register()
        let targetDirectory = try await checkpointDirectory(lfm25DenseBF16)
        let target = try await LLMModelFactory.shared.load(
            from: targetDirectory, using: tokenizerLoader)
        let draftContext = try await MTPDrafterModelFactory.shared.load(
            from: downloader, using: tokenizerLoader, configuration: lfm25DenseDSpark)
        let drafter = try #require(draftContext.model as? LFM2DSparkDraftModel)
        let input = try await target.processor.prepare(
            input: UserInput(chat: [.user(dsparkQuestion)]))
        return (target, drafter, input)
    }
}
