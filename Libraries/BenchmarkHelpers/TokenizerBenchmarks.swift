// Tokenizer loading, tokenization and decoding benchmarks, which read only the
// tokenizer files of a model.

import Foundation
import MLXLMCommon

extension TokenizerSource {
    /// The tokenizer source at the same place as the model `identifier`.
    fileprivate init(_ identifier: ModelConfiguration.Identifier) {
        switch identifier {
        case .id(let id, let revision):
            self = .id(id, revision: revision)
        case .directory(let directory):
            self = .directory(directory)
        }
    }
}

extension Benchmark {
    /// The local directory of the tokenizer of `configuration`, downloaded when
    /// it is remote. A configuration without a tokenizer source reads the
    /// tokenizer beside the model.
    private static func tokenizerDirectory(
        from downloader: any Downloader,
        configuration: ModelConfiguration,
        useLatest: Bool
    ) async throws -> URL {
        switch configuration.tokenizerSource ?? TokenizerSource(configuration.id) {
        case .id(let id, let revision):
            return try await downloader.download(
                id: id,
                revision: revision,
                matching: tokenizerDownloadPatterns,
                useLatest: useLatest,
                progressHandler: { _ in }
            )
        case .directory(let directory):
            return directory
        }
    }

    /// The tokenizer of `configuration`, loaded from its local directory.
    private static func loadTokenizer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool
    ) async throws -> any Tokenizer {
        let directory = try await tokenizerDirectory(
            from: downloader, configuration: configuration, useLatest: useLatest)
        return try await tokenizerLoader.load(from: directory)
    }

    /// Benchmark tokenizer loading without downloading model weights or
    /// initializing a model.
    ///
    /// The tokenizer files download once, then each run loads them again.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the tokenizer files
    ///   - tokenizerLoader: the loader under measurement
    ///   - configuration: the model whose tokenizer loads
    ///   - useLatest: whether the downloader checks for a newer revision
    ///   - runs: the timed loads
    /// - Returns: the time of each load, in milliseconds
    public static func tokenizerLoading(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration = .init(id: BenchmarkDefaults.languageModelID),
        useLatest: Bool = false,
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> BenchmarkStats {
        let directory = try await tokenizerDirectory(
            from: downloader, configuration: configuration, useLatest: useLatest)
        return try await measure("Tokenizer load", runs: runs) {
            _ = try await tokenizerLoader.load(from: directory)
            return ""
        }
    }

    /// Benchmark tokenization on a preloaded tokenizer without initializing a model.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the tokenizer files
    ///   - tokenizerLoader: the loader of the tokenizer under measurement
    ///   - configuration: the model whose tokenizer encodes
    ///   - text: the text each run encodes
    ///   - addSpecialTokens: whether the encoding adds the special tokens
    ///   - useLatest: whether the downloader checks for a newer revision
    ///   - runs: the timed encodes
    /// - Returns: the time of each encode, in milliseconds
    public static func tokenization(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration = .init(id: BenchmarkDefaults.languageModelID),
        text: String = BenchmarkDefaults.sampleText,
        addSpecialTokens: Bool = true,
        useLatest: Bool = false,
        runs: Int = BenchmarkDefaults.tokenizationRuns
    ) async throws -> BenchmarkStats {
        let tokenizer = try await loadTokenizer(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest)
        return try await measure("Tokenization", runs: runs) {
            let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
            return " (\(tokenIds.count) tokens)"
        }
    }

    /// Benchmark decoding on a preloaded tokenizer without initializing a model.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the tokenizer files
    ///   - tokenizerLoader: the loader of the tokenizer under measurement
    ///   - configuration: the model whose tokenizer decodes
    ///   - text: the text whose tokens each run decodes
    ///   - addSpecialTokens: whether the encoding of `text` adds the special tokens
    ///   - skipSpecialTokens: whether the decode drops the special tokens
    ///   - useLatest: whether the downloader checks for a newer revision
    ///   - runs: the timed runs
    ///   - decodesPerRun: the decodes of each run, averaged into its time
    /// - Returns: the mean time of one decode in each run, in milliseconds
    public static func decoding(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration = .init(id: BenchmarkDefaults.languageModelID),
        text: String = BenchmarkDefaults.sampleText,
        addSpecialTokens: Bool = true,
        skipSpecialTokens: Bool = false,
        useLatest: Bool = false,
        runs: Int = BenchmarkDefaults.decodingRuns,
        decodesPerRun: Int = BenchmarkDefaults.decodesPerRun
    ) async throws -> BenchmarkStats {
        let tokenizer = try await loadTokenizer(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest)
        let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        return try await measure("Decoding", runs: runs, repetitions: decodesPerRun) {
            var decoded = ""
            for _ in 0 ..< decodesPerRun {
                decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
            }
            return " avg over \(decodesPerRun)x (\(decoded.count) chars)"
        }
    }
}
