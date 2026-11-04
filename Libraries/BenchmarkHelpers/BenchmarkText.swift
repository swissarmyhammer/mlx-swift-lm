// The public-domain text the benchmarks tokenize, decode and prompt with, and
// the default settings of the benchmark runners.

import Foundation
import MLXLMCommon

// MARK: - Defaults

/// The default settings of the benchmark runners.
public enum BenchmarkDefaults {
    /// The source of the benchmark text.
    public static let textSource = BenchmarkTextSource.prideAndPrejudice
    /// The characters of benchmark text a tokenization benchmark encodes.
    public static let tokenizationTextCharacterCount = 20_000
    /// The characters of benchmark text a decoding benchmark decodes.
    public static let decodingTextCharacterCount = 200_000
    /// The timed runs of a loading benchmark.
    public static let loadingRuns = 7
    /// The timed runs of a download benchmark.
    public static let downloadRuns = 7
    /// The timed runs of a tokenization benchmark.
    public static let tokenizationRuns = 25
    /// The timed runs of a decoding benchmark.
    public static let decodingRuns = 25
    /// The decodes of each timed decoding run, averaged into its time.
    public static let decodesPerRun = 10
    /// The small language model the tokenizer, loading, download and generation
    /// benchmarks measure by default.
    public static let languageModelID = "mlx-community/Qwen3-0.6B-4bit"
    /// The vision language model the VLM loading benchmark loads by default.
    public static let visionLanguageModelID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    /// The embedding model the embedding loading benchmark loads by default.
    public static let embeddingModelID = "mlx-community/Qwen3-Embedding-0.6B-8bit"
    /// The short text the tokenization and decoding benchmarks use by default.
    public static let sampleText = "The quick brown fox jumps over the lazy dog."
}

// MARK: - Benchmark Text

/// A public-domain text, fetched once and cached in the temporary directory.
public struct BenchmarkTextSource: Sendable {
    /// The name of the cached copy, without its extension.
    public let name: String
    /// Where the text is fetched from.
    public let url: URL
    /// The text the benchmark text starts at, which skips a header before it;
    /// `nil` keeps the whole text.
    public let contentStartMarker: String?

    /// Creates a source named `name` fetched from `url`.
    ///
    /// - Parameters:
    ///   - name: the name of the cached copy, without its extension
    ///   - url: where the text is fetched from
    ///   - contentStartMarker: the text the benchmark text starts at, or `nil`
    ///     to keep the whole text
    public init(name: String, url: URL, contentStartMarker: String? = nil) {
        self.name = name
        self.url = url
        self.contentStartMarker = contentStartMarker
    }

    /// The Project Gutenberg text of *Pride and Prejudice*, from its first sentence.
    public static let prideAndPrejudice: BenchmarkTextSource = {
        guard let url = URL(string: "https://www.gutenberg.org/ebooks/1342.txt.utf-8") else {
            preconditionFailure("the Project Gutenberg address is a valid URL")
        }
        return BenchmarkTextSource(
            name: "pride-and-prejudice", url: url,
            contentStartMarker: "It is a truth universally acknowledged")
    }()

    /// Where the fetched text is cached.
    fileprivate var cacheURL: URL {
        FileManager.default.temporaryDirectory
            .appending(component: "BenchmarkHelpers", directoryHint: .isDirectory)
            .appending(component: "\(name).txt")
    }

    /// `text` with its line endings as `\n`, from ``contentStartMarker`` on.
    fileprivate func benchmarkText(from text: String) throws -> String {
        let normalized =
            text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        guard let marker = contentStartMarker else {
            return normalized
        }
        guard let markerRange = normalized.range(of: marker) else {
            throw BenchmarkTextError.contentStartMarkerNotFound(marker)
        }
        return String(normalized[markerRange.lowerBound...])
    }
}

/// Why the benchmark text could not be read.
public enum BenchmarkTextError: LocalizedError {
    /// The server did not answer with a successful HTTP response.
    case invalidResponse(URL)
    /// The fetched text is not UTF-8.
    case decodeFailed(URL)
    /// The text has no ``BenchmarkTextSource/contentStartMarker``.
    case contentStartMarkerNotFound(String)

    /// A description of the failure for the benchmark output.
    public var errorDescription: String? {
        switch self {
        case .invalidResponse(let url):
            return "Unexpected response while fetching benchmark text from \(url.absoluteString)."
        case .decodeFailed(let url):
            return "Failed to decode benchmark text from \(url.absoluteString) as UTF-8."
        case .contentStartMarkerNotFound(let marker):
            return "Benchmark text start marker not found: '\(marker)'."
        }
    }
}

extension Benchmark {
    /// The HTTP status codes of a successful response.
    private static let successfulStatusCodes = 200 ..< 300

    /// Load benchmark text from a remote public-domain source and cache it
    /// locally in the temporary directory.
    ///
    /// - Parameters:
    ///   - source: the text to load
    ///   - characterCount: the characters of the text to return, from its start;
    ///     at least one
    /// - Returns: the first `characterCount` characters of the text
    /// - Throws: a ``BenchmarkTextError``, or the error of the fetch or the cache
    public static func loadText(
        source: BenchmarkTextSource = BenchmarkDefaults.textSource,
        characterCount: Int = BenchmarkDefaults.tokenizationTextCharacterCount
    ) async throws -> String {
        precondition(characterCount > 0, "characterCount must be greater than zero")
        let text = try await fetchText(source: source)
        return String(text.prefix(characterCount))
    }

    /// The text of `source`, from its cached copy when there is one.
    private static func fetchText(source: BenchmarkTextSource) async throws -> String {
        let cacheURL = source.cacheURL
        let fileManager = FileManager.default

        if fileManager.fileExists(atPath: cacheURL.path) {
            let cached = try String(contentsOf: cacheURL, encoding: .utf8)
            return try source.benchmarkText(from: cached)
        }

        let (data, response) = try await URLSession.shared.data(from: source.url)
        guard let httpResponse = response as? HTTPURLResponse,
            successfulStatusCodes.contains(httpResponse.statusCode)
        else {
            throw BenchmarkTextError.invalidResponse(source.url)
        }
        guard let downloaded = String(data: data, encoding: .utf8) else {
            throw BenchmarkTextError.decodeFailed(source.url)
        }

        try fileManager.createDirectory(
            at: cacheURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try downloaded.write(to: cacheURL, atomically: true, encoding: .utf8)

        return try source.benchmarkText(from: downloaded)
    }
}
