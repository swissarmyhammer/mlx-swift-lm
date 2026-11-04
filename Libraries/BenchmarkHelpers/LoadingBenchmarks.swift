// Model loading and download cache benchmarks.

import Foundation
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

extension Benchmark {
    /// The revision ``downloadCacheHit(from:modelId:runs:)`` downloads.
    private static let downloadRevision = "main"

    /// Benchmark LLM model loading. Performs a warm-up run, then measures `runs` timed loads.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the model
    ///   - tokenizerLoader: the loader of the tokenizer of the model
    ///   - modelId: the model to load
    ///   - runs: the timed loads
    /// - Returns: the time of each load, in milliseconds
    public static func llmLoading(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        modelId: String = BenchmarkDefaults.languageModelID,
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> BenchmarkStats {
        let configuration = ModelConfiguration(id: modelId)
        return try await measureLoads("LLM", runs: runs) {
            _ = try await LLMModelFactory.shared.load(
                from: downloader, using: tokenizerLoader, configuration: configuration
            ) { _ in }
        }
    }

    /// Benchmark VLM model loading. Performs a warm-up run, then measures `runs` timed loads.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the model
    ///   - tokenizerLoader: the loader of the tokenizer of the model
    ///   - modelId: the model to load
    ///   - runs: the timed loads
    /// - Returns: the time of each load, in milliseconds
    public static func vlmLoading(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        modelId: String = BenchmarkDefaults.visionLanguageModelID,
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> BenchmarkStats {
        let configuration = ModelConfiguration(id: modelId)
        return try await measureLoads("VLM", runs: runs) {
            _ = try await VLMModelFactory.shared.load(
                from: downloader, using: tokenizerLoader, configuration: configuration
            ) { _ in }
        }
    }

    /// Benchmark embedding model loading. Performs a warm-up run, then measures
    /// `runs` timed loads.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the model
    ///   - tokenizerLoader: the loader of the tokenizer of the model
    ///   - configuration: the model to load
    ///   - runs: the timed loads
    /// - Returns: the time of each load, in milliseconds
    public static func embeddingLoading(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration = .init(id: BenchmarkDefaults.embeddingModelID),
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> BenchmarkStats {
        try await measureLoads("Embedding", runs: runs) {
            _ = try await EmbedderModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader, configuration: configuration
            ) { _ in }
        }
    }

    /// Measures `load` as ``measure(_:runs:repetitions:clearsCache:operation:)``
    /// does, clearing the MLX buffer cache after each load so the next load
    /// allocates its own buffers.
    private static func measureLoads(
        _ kind: String, runs: Int, load: () async throws -> Void
    ) async throws -> BenchmarkStats {
        try await measure("\(kind) load", runs: runs, clearsCache: true) {
            try await load()
            return ""
        }
    }

    /// Benchmark download cache hit performance. Ensures the model is cached with
    /// a warm-up download, then measures repeated cache lookups.
    ///
    /// - Parameters:
    ///   - downloader: the downloader under measurement
    ///   - modelId: the model whose files the downloader looks up
    ///   - runs: the timed lookups
    /// - Returns: the time of each lookup, in milliseconds
    public static func downloadCacheHit(
        from downloader: any Downloader,
        modelId: String = BenchmarkDefaults.languageModelID,
        runs: Int = BenchmarkDefaults.downloadRuns
    ) async throws -> BenchmarkStats {
        try await measure("Download cache hit", runs: runs) {
            _ = try await downloader.download(
                id: modelId, revision: downloadRevision, matching: modelDownloadPatterns,
                useLatest: false, progressHandler: { _ in })
            return ""
        }
    }
}
