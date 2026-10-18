// Copyright © 2026 Apple Inc.
//
// Throughput and system prompt cache benchmarks on the Qwen 3.8 27B checkpoint,
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
import Testing
import Tokenizers

private let qwen38ModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// One shared load, so the 14 GB checkpoint loads once per test process.
private let qwen38Load: Task<ModelContainer, Error>? = {
    guard hfSnapshotDir(modelId: qwen38ModelID) != nil else { return nil }
    return Task {
        try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: qwen38ModelID)
        ) { _ in }
    }
}()

private func qwen38ContainerOrSkip(testName: String) async throws -> ModelContainer? {
    guard let load = qwen38Load else {
        print("Skipping \(testName): no local copy of \(qwen38ModelID)")
        return nil
    }
    return try await load.value
}

@Suite(.serialized, .timeLimit(.minutes(60)))
struct GenerationBenchmarkTests {

    @Test
    func qwen38Generation() async throws {
        guard let container = try await qwen38ContainerOrSkip(testName: "qwen38Generation")
        else { return }
        let stats = try await benchmarkLLMGeneration(container: container)
        stats.printSummary(label: qwen38ModelID)
    }

    @Test
    func qwen38SystemPromptCache() async throws {
        guard let container = try await qwen38ContainerOrSkip(testName: "qwen38SystemPromptCache")
        else { return }
        let stats = try await benchmarkSystemPromptCache(container: container)
        stats.printSummary(label: qwen38ModelID)
        #expect(stats.followUp.allSatisfy { $0.cachedPromptTokenCount > 0 })
        #expect(stats.fork.allSatisfy { $0.cachedPromptTokenCount > 0 })
    }
}
