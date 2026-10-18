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
}
