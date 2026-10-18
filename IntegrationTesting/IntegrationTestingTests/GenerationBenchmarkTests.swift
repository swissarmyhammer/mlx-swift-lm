// Copyright © 2026 Apple Inc.
//
// Throughput and system prompt cache benchmarks on recently tuned checkpoints,
// measured with `BenchmarkHelpers`. The suite downloads nothing.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/GenerationBenchmarkTests`

import BenchmarkHelpers
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

/// A checkpoint under benchmark, loaded at most once per test process.
private struct BenchmarkCheckpoint: Sendable {
    let id: String
    let load: Task<ModelContainer, Error>?

    init(id: String, factory: any ModelFactory) {
        self.id = id
        self.load =
            hfSnapshotDir(modelId: id) == nil
            ? nil
            : Task {
                try await factory.loadContainer(
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    configuration: ModelConfiguration(id: id)
                ) { _ in }
            }
    }

    func containerOrSkip(testName: String) async throws -> ModelContainer? {
        guard let load else {
            print("Skipping \(testName): no local copy of \(id)")
            return nil
        }
        return try await load.value
    }
}

private let qwen38 = BenchmarkCheckpoint(
    id: "mlx-community/Qwen3.8-27B-mxfp4", factory: LLMModelFactory.shared)

private let qwen38FourBit = BenchmarkCheckpoint(
    id: "mlx-community/Qwen3.8-27B-4bit", factory: LLMModelFactory.shared)

private let museGlimmer = BenchmarkCheckpoint(
    id: "mlx-community/Muse-Glimmer-30B-4bit", factory: VLMModelFactory.shared)

@Suite(.serialized, .timeLimit(.minutes(60)))
struct GenerationBenchmarkTests {

    /// Muse Glimmer reasons before it answers and has no switch to turn that off. A
    /// turn with no answer text is not recorded, so give it room to answer.
    private static let museGlimmerMaxTokens = 512
    private static let museGlimmerContext: [String: any Sendable] = ["reasoning_strength": "low"]

    private func generation(_ checkpoint: BenchmarkCheckpoint, testName: String) async throws {
        guard let container = try await checkpoint.containerOrSkip(testName: testName)
        else { return }
        let stats = try await benchmarkLLMGeneration(container: container)
        stats.printSummary(label: checkpoint.id)
    }

    private func systemPromptCache(
        _ checkpoint: BenchmarkCheckpoint, maxTokens: Int = 16,
        kvCache: KVCacheConfiguration? = nil,
        additionalContext: [String: any Sendable]? = nil, testName: String
    ) async throws {
        guard let container = try await checkpoint.containerOrSkip(testName: testName)
        else { return }
        let stats = try await benchmarkSystemPromptCache(
            container: container, maxTokens: maxTokens, kvCache: kvCache,
            additionalContext: additionalContext)
        stats.printSummary(label: checkpoint.id)
        #expect(stats.followUp.allSatisfy { $0.cachedPromptTokenCount > 0 })
        #expect(stats.fork.allSatisfy { $0.cachedPromptTokenCount > 0 })
    }

    @Test
    func qwen38Generation() async throws {
        try await generation(qwen38, testName: "qwen38Generation")
    }

    @Test
    func qwen38SystemPromptCache() async throws {
        try await systemPromptCache(qwen38, testName: "qwen38SystemPromptCache")
    }

    @Test
    func qwen38FourBitGeneration() async throws {
        try await generation(qwen38FourBit, testName: "qwen38FourBitGeneration")
    }

    @Test
    func qwen38FourBitSystemPromptCache() async throws {
        try await systemPromptCache(qwen38FourBit, testName: "qwen38FourBitSystemPromptCache")
    }

    @Test
    func museGlimmerGeneration() async throws {
        try await generation(museGlimmer, testName: "museGlimmerGeneration")
    }

    @Test
    func museGlimmerSystemPromptCache() async throws {
        try await systemPromptCache(
            museGlimmer, maxTokens: Self.museGlimmerMaxTokens,
            additionalContext: Self.museGlimmerContext,
            testName: "museGlimmerSystemPromptCache")
    }

    /// The sliding layers keep 512 rows past the window, so a fork can rewind them.
    @Test
    func museGlimmerSystemPromptCacheWithRewindReserve() async throws {
        try await systemPromptCache(
            museGlimmer, maxTokens: Self.museGlimmerMaxTokens,
            kvCache: KVCacheConfiguration(rewind: try .init(maxTokens: 512)),
            additionalContext: Self.museGlimmerContext,
            testName: "museGlimmerSystemPromptCacheWithRewindReserve")
    }

    /// Plain against MTP decoding of Qwen3.8-27B-4bit with its 4-bit MTP head.
    @Test
    func qwen38FourBitMTPGeneration() async throws {
        let targetID = "mlx-community/Qwen3.8-27B-4bit"
        let drafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"
        guard hfSnapshotDir(modelId: targetID) != nil, hfSnapshotDir(modelId: drafterID) != nil
        else {
            print(
                "Skipping qwen38FourBitMTPGeneration: no local copy of \(targetID) or \(drafterID)")
            return
        }
        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: targetID))
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: drafterID))
        let results = try await benchmarkMTPGeneration(context: context, drafter: drafter.model)
        #expect(results.allSatisfy { $0.acceptanceRate > 0 })
    }

    private static let dflash2TargetID = "mlx-community/Qwen3.8-27B-4bit"
    private static let dflash2DrafterID = "incoai/Qwen3.8-27B-DFlash2"

    /// The target and the DFlash2 drafter, or nil when either has no local copy.
    private func dflash2Pair(testName: String) async throws -> (
        context: ModelContext, drafter: any DFlash2DrafterModel
    )? {
        guard hfSnapshotDir(modelId: Self.dflash2TargetID) != nil,
            hfSnapshotDir(modelId: Self.dflash2DrafterID) != nil
        else {
            print("Skipping \(testName): no local copy of the target or the DFlash2 drafter")
            return nil
        }
        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: Self.dflash2TargetID))
        await DFlash2Registration.register()
        let drafter = try await DFlash2DrafterModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: Self.dflash2DrafterID))
        return (context, drafter.model)
    }

    /// Plain against DFlash2 decoding, greedy and at Qwen3.8's recommended sampling.
    @Test
    func qwen38FourBitDFlash2Generation() async throws {
        guard let pair = try await dflash2Pair(testName: "qwen38FourBitDFlash2Generation")
        else { return }
        let greedy = try await benchmarkDFlash2Generation(
            context: pair.context, drafter: pair.drafter)
        let sampled = try await benchmarkDFlash2Generation(
            context: pair.context, drafter: pair.drafter,
            parameters: GenerateParameters(
                maxTokens: 192, temperature: 1.0, topP: 0.95, topK: 20))
        #expect((greedy + sampled).allSatisfy { $0.acceptanceRate > 0 })
    }

    @Test
    func qwen38FourBitDFlash2WarmPrefix() async throws {
        guard let pair = try await dflash2Pair(testName: "qwen38FourBitDFlash2WarmPrefix")
        else { return }
        _ = try await benchmarkDFlash2WarmPrefix(context: pair.context, drafter: pair.drafter)
    }

    /// A three-turn ChatSession on Qwen3.8-27B-4bit, plain and with DFlash2: the
    /// same text and cache reuse per turn, and the decode speed of each.
    @Test
    func qwen38FourBitDFlash2ChatSession() async throws {
        guard hfSnapshotDir(modelId: Self.dflash2DrafterID) != nil,
            let container = try await qwen38FourBit.containerOrSkip(
                testName: "qwen38FourBitDFlash2ChatSession")
        else { return }
        await DFlash2Registration.register()
        let drafter = try await DFlash2DrafterModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: Self.dflash2DrafterID))

        // Warm up both paths once.
        _ = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: drafter, prompts: [.defaults[0]])
        _ = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: nil, prompts: [.defaults[0]])

        let plain = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: nil)
        let drafted = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: drafter)
        for (p, d) in zip(plain, drafted) {
            let shared = zip(p.text, d.text).prefix { $0 == $1 }.count
            print(
                "ChatSession \(p.prompt): \(String(format: "%.2f", d.tokensPerSecond / p.tokensPerSecond))x, "
                    + "same text: \(p.text == d.text), first \(shared) of \(p.text.count) characters match"
            )
        }
        // A greedy verify pass can break a near-tie the other way, so the histories, and
        // the cached counts after them, agree only while the texts do.
        #expect(drafted[0].cachedPromptTokenCount == plain[0].cachedPromptTokenCount)
        #expect(drafted.dropFirst().allSatisfy { $0.cachedPromptTokenCount > 0 })
        #expect(drafted.allSatisfy { $0.proposedDraftTokens > 0 })

        let sampling = GenerateParameters(maxTokens: 192, temperature: 1.0, topP: 0.95, topK: 20)
        let plainSampled = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: nil, parameters: sampling)
        let draftedSampled = try await benchmarkChatSessionDecode(
            container: container, dflash2Drafter: drafter, parameters: sampling)
        for (p, d) in zip(plainSampled, draftedSampled) {
            print(
                "ChatSession sampled \(p.prompt): "
                    + "\(String(format: "%.2f", d.tokensPerSecond / p.tokensPerSecond))x")
        }

        let cache = try await benchmarkSystemPromptCache(
            container: container, dflash2Drafter: drafter)
        cache.printSummary(label: "Qwen3.8-27B-4bit with DFlash2")
        #expect(cache.fork.allSatisfy { $0.cachedPromptTokenCount > 0 })
        #expect(cache.followUp.allSatisfy { $0.cachedPromptTokenCount > 0 })
    }
}
