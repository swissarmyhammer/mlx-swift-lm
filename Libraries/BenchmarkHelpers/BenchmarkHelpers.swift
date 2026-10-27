// Shared benchmark logic for measuring model loading, tokenizer performance,
// and download performance.
// Integration packages inject their own Downloader and TokenizerLoader.

import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

// MARK: - No-Op Tokenizer

/// A tokenizer loader that returns a stub tokenizer. Useful for benchmarking
/// model loading in downloader integration packages that don't provide a
/// real tokenizer.
public struct NoOpTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any Tokenizer {
        NoOpTokenizer()
    }
}

private struct NoOpTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
}

// MARK: - Stats

public struct BenchmarkStats: Sendable {
    public let mean: Double
    public let median: Double
    public let stdDev: Double
    public let min: Double
    public let max: Double

    public init(times: [Double]) {
        precondition(!times.isEmpty, "BenchmarkStats requires at least one timing measurement")
        let sorted = times.sorted()
        self.min = sorted.first!
        self.max = sorted.last!
        let mean = times.reduce(0, +) / Double(times.count)
        self.mean = mean
        self.median = sorted[sorted.count / 2]

        let squaredDiffs = times.map { ($0 - mean) * ($0 - mean) }
        self.stdDev = sqrt(squaredDiffs.reduce(0, +) / Double(times.count))
    }

    public func printSummary(label: String) {
        print("\(label) results:")
        print("  Mean:   \(String(format: "%.1f", mean))ms")
        print("  Median: \(String(format: "%.1f", median))ms")
        print("  StdDev: \(String(format: "%.1f", stdDev))ms")
        print("  Range:  \(String(format: "%.1f", min))-\(String(format: "%.1f", max))ms")
    }
}

// MARK: - Benchmark Text

public enum BenchmarkDefaults {
    public static let textSource = BenchmarkTextSource.prideAndPrejudice
    public static let tokenizationTextCharacterCount = 20_000
    public static let decodingTextCharacterCount = 200_000
    public static let loadingRuns = 7
    public static let downloadRuns = 7
    public static let tokenizationRuns = 25
    public static let decodingRuns = 25
    public static let decodesPerRun = 10
}

public struct BenchmarkTextSource: Sendable {
    public let name: String
    public let url: URL
    public let contentStartMarker: String?

    public init(name: String, url: URL, contentStartMarker: String? = nil) {
        self.name = name
        self.url = url
        self.contentStartMarker = contentStartMarker
    }

    public static let prideAndPrejudice = BenchmarkTextSource(
        name: "pride-and-prejudice",
        url: URL(string: "https://www.gutenberg.org/ebooks/1342.txt.utf-8")!,
        contentStartMarker: "It is a truth universally acknowledged"
    )
}

public enum BenchmarkTextError: LocalizedError {
    case invalidResponse(URL)
    case decodeFailed(URL)
    case contentStartMarkerNotFound(String)

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

private func benchmarkTextCacheURL(for source: BenchmarkTextSource) -> URL {
    FileManager.default.temporaryDirectory
        .appending(component: "BenchmarkHelpers", directoryHint: .isDirectory)
        .appending(component: "\(source.name).txt")
}

private func normalizeBenchmarkText(_ text: String) -> String {
    text.replacingOccurrences(of: "\r\n", with: "\n")
        .replacingOccurrences(of: "\r", with: "\n")
}

private func trimmedBenchmarkText(_ text: String, source: BenchmarkTextSource) throws -> String {
    guard let marker = source.contentStartMarker else {
        return text
    }
    guard let markerRange = text.range(of: marker) else {
        throw BenchmarkTextError.contentStartMarkerNotFound(marker)
    }
    return String(text[markerRange.lowerBound...])
}

private func fetchBenchmarkText(source: BenchmarkTextSource) async throws -> String {
    let cacheURL = benchmarkTextCacheURL(for: source)
    let fileManager = FileManager.default

    if fileManager.fileExists(atPath: cacheURL.path) {
        let cached = try String(contentsOf: cacheURL, encoding: .utf8)
        return try trimmedBenchmarkText(normalizeBenchmarkText(cached), source: source)
    }

    let (data, response) = try await URLSession.shared.data(from: source.url)
    guard let httpResponse = response as? HTTPURLResponse,
        (200 ..< 300).contains(httpResponse.statusCode)
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

    return try trimmedBenchmarkText(normalizeBenchmarkText(downloaded), source: source)
}

/// Load benchmark text from a remote public-domain source and cache it locally in the temporary directory.
public func loadBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource,
    characterCount: Int = 20_000
) async throws -> String {
    precondition(characterCount > 0, "characterCount must be greater than zero")
    let text = try await fetchBenchmarkText(source: source)
    return String(text.prefix(characterCount))
}

public func loadTokenizationBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource
) async throws -> String {
    try await loadBenchmarkText(
        source: source,
        characterCount: BenchmarkDefaults.tokenizationTextCharacterCount
    )
}

public func loadDecodingBenchmarkText(
    source: BenchmarkTextSource = BenchmarkDefaults.textSource
) async throws -> String {
    try await loadBenchmarkText(
        source: source,
        characterCount: BenchmarkDefaults.decodingTextCharacterCount
    )
}

private func resolveTokenizerDirectory(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration,
    useLatest: Bool
) async throws -> URL {
    switch configuration.tokenizerSource {
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
    case nil:
        switch configuration.id {
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
}

private func loadTokenizerForBenchmark(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration,
    useLatest: Bool
) async throws -> any Tokenizer {
    let tokenizerDirectory = try await resolveTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )
    return try await tokenizerLoader.load(from: tokenizerDirectory)
}

// MARK: - Benchmark Runners

/// Benchmark tokenizer loading without downloading model weights or initializing a model.
public func benchmarkTokenizerLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    let tokenizerDirectory = try await resolveTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )

    _ = try await tokenizerLoader.load(from: tokenizerDirectory)

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await tokenizerLoader.load(from: tokenizerDirectory)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Tokenizer load run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}

/// Benchmark tokenization on a preloaded tokenizer without initializing a model.
public func benchmarkTokenization(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    text: String = "The quick brown fox jumps over the lazy dog.",
    addSpecialTokens: Bool = true,
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.tokenizationRuns
) async throws -> BenchmarkStats {
    let tokenizer = try await loadTokenizerForBenchmark(
        from: downloader,
        using: tokenizerLoader,
        configuration: configuration,
        useLatest: useLatest
    )

    _ = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print(
            "Tokenization run \(i): \(String(format: "%.1f", elapsed))ms (\(tokenIds.count) tokens)"
        )
    }

    return BenchmarkStats(times: times)
}

/// Benchmark decoding on a preloaded tokenizer without initializing a model.
public func benchmarkDecoding(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: MLXLMCommon.ModelConfiguration = .init(id: "mlx-community/Qwen3-0.6B-4bit"),
    text: String = "The quick brown fox jumps over the lazy dog.",
    addSpecialTokens: Bool = true,
    skipSpecialTokens: Bool = false,
    useLatest: Bool = false,
    runs: Int = BenchmarkDefaults.decodingRuns,
    decodesPerRun: Int = BenchmarkDefaults.decodesPerRun
) async throws -> BenchmarkStats {
    precondition(decodesPerRun > 0, "decodesPerRun must be greater than zero")

    let tokenizer = try await loadTokenizerForBenchmark(
        from: downloader,
        using: tokenizerLoader,
        configuration: configuration,
        useLatest: useLatest
    )
    let tokenIds = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)

    _ = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)

    var times: [Double] = []
    for i in 1 ... runs {
        var decoded = ""
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< decodesPerRun {
            decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(decodesPerRun)
        times.append(elapsed)
        print(
            "Decoding run \(i): \(String(format: "%.1f", elapsed))ms avg over \(decodesPerRun)x "
                + "(\(decoded.count) chars)"
        )
    }

    return BenchmarkStats(times: times)
}

/// Benchmark LLM model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkLLMLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    let config = MLXLMCommon.ModelConfiguration(id: modelId)

    _ = try await LLMModelFactory.shared.load(
        from: downloader, using: tokenizerLoader, configuration: config
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await LLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader, configuration: config
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("LLM load run \(i): \(String(format: "%.1f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

/// Benchmark VLM model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkVLMLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    let config = MLXLMCommon.ModelConfiguration(id: modelId)

    _ = try await VLMModelFactory.shared.load(
        from: downloader, using: tokenizerLoader, configuration: config
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await VLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader, configuration: config
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("VLM load run \(i): \(String(format: "%.1f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

/// Benchmark embedding model loading. Performs a warm-up run, then measures `runs` timed loads.
public func benchmarkEmbeddingLoading(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration = .init(
        id: "mlx-community/Qwen3-Embedding-0.6B-8bit"),
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> BenchmarkStats {
    _ = try await EmbedderModelFactory.shared.loadContainer(
        from: downloader, using: tokenizerLoader, configuration: configuration
    ) { _ in }
    Memory.clearCache()

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await EmbedderModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration
        ) { _ in }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Embedding load run \(i): \(String(format: "%.1f", elapsed))ms")
        Memory.clearCache()
    }

    return BenchmarkStats(times: times)
}

// MARK: - Download Benchmarks

/// Benchmark download cache hit performance. Ensures the model is cached with a warm-up
/// download, then measures repeated cache lookups.
public func benchmarkDownloadCacheHit(
    from downloader: any Downloader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    runs: Int = BenchmarkDefaults.downloadRuns
) async throws -> BenchmarkStats {
    let patterns = modelDownloadPatterns

    // Warm-up: ensure the model is cached
    _ = try await downloader.download(
        id: modelId, revision: "main", matching: patterns,
        useLatest: false, progressHandler: { _ in })

    var times: [Double] = []
    for i in 1 ... runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await downloader.download(
            id: modelId, revision: "main", matching: patterns,
            useLatest: false, progressHandler: { _ in })
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Download cache hit run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}

// MARK: - LLM Generation Benchmarks

/// Combined prefill and decode timing for an LLM generation run.
public struct LLMGenerationStats: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    /// Prefill latency per run, in milliseconds.
    public let promptTimeStats: BenchmarkStats
    /// Decode latency per run, in milliseconds (covers all generated tokens).
    public let generateTimeStats: BenchmarkStats

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTimeStats: BenchmarkStats,
        generateTimeStats: BenchmarkStats
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTimeStats = promptTimeStats
        self.generateTimeStats = generateTimeStats
    }

    /// Median prefill throughput in tokens per second.
    public var prefillTokensPerSecond: Double {
        Double(promptTokenCount) / (promptTimeStats.median / 1000.0)
    }

    /// Median decode throughput in tokens per second.
    public var decodeTokensPerSecond: Double {
        Double(generationTokenCount) / (generateTimeStats.median / 1000.0)
    }

    public func printSummary(label: String) {
        print("\(label) results:")
        print(
            "  Prefill: \(promptTokenCount) tokens, "
                + "\(String(format: "%.2f", prefillTokensPerSecond)) tok/s "
                + "(median \(String(format: "%.1f", promptTimeStats.median))ms, "
                + "stddev \(String(format: "%.1f", promptTimeStats.stdDev))ms)"
        )
        print(
            "  Decode:  \(generationTokenCount) tokens, "
                + "\(String(format: "%.2f", decodeTokensPerSecond)) tok/s "
                + "(median \(String(format: "%.1f", generateTimeStats.median))ms, "
                + "stddev \(String(format: "%.1f", generateTimeStats.stdDev))ms)"
        )
    }
}

public enum LLMGenerationBenchmarkError: LocalizedError {
    case missingCompletionInfo

    public var errorDescription: String? {
        switch self {
        case .missingCompletionInfo:
            return
                "LLM generation stream completed without emitting GenerateCompletionInfo."
        }
    }
}

/// Benchmark prefill and decode performance for an LLM. Performs a warm-up run
/// (loading + first generation) to reach steady-state, then measures `runs` timed
/// trials and reports prefill/decode throughput.
///
/// Uses ``BenchmarkDefaults/textSource`` for the prompt by default; pass an
/// explicit `prompt` to override. `temperature` defaults to 0 so trials are
/// deterministic and tps comparisons are reproducible across runs.
public func benchmarkLLMGeneration(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    modelId: String = "mlx-community/Qwen3-0.6B-4bit",
    prompt: String? = nil,
    promptCharacterCount: Int = 4_000,
    maxTokens: Int = 64,
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> LLMGenerationStats {
    let configuration = MLXLMCommon.ModelConfiguration(id: modelId)
    let container = try await LLMModelFactory.shared.loadContainer(
        from: downloader, using: tokenizerLoader, configuration: configuration
    ) { _ in }
    return try await benchmarkLLMGeneration(
        container: container, prompt: prompt, promptCharacterCount: promptCharacterCount,
        maxTokens: maxTokens, runs: runs)
}

/// Benchmark prefill and decode performance for an already loaded model.
public func benchmarkLLMGeneration(
    container: ModelContainer,
    prompt: String? = nil,
    promptCharacterCount: Int = 4_000,
    maxTokens: Int = 64,
    runs: Int = BenchmarkDefaults.loadingRuns
) async throws -> LLMGenerationStats {
    let benchmarkPrompt: String
    if let prompt = prompt {
        benchmarkPrompt = prompt
    } else {
        benchmarkPrompt = try await loadBenchmarkText(characterCount: promptCharacterCount)
    }

    let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

    // Warm-up: prime caches and weights. UserInput is `sending`, so build a
    // fresh instance for each call.
    do {
        let warmInput = try await container.prepare(input: UserInput(prompt: benchmarkPrompt))
        let warmStream = try await container.generate(
            input: warmInput, parameters: parameters)
        for await _ in warmStream {}
    }

    var promptTimes: [Double] = []
    var generateTimes: [Double] = []
    var promptTokenCount = 0
    var generationTokenCount = 0

    for i in 1 ... runs {
        let input = try await container.prepare(input: UserInput(prompt: benchmarkPrompt))
        let stream = try await container.generate(input: input, parameters: parameters)
        var info: GenerateCompletionInfo?
        for await item in stream {
            if case .info(let i) = item { info = i }
        }
        guard let completion = info else {
            throw LLMGenerationBenchmarkError.missingCompletionInfo
        }
        promptTokenCount = completion.promptTokenCount
        generationTokenCount = completion.generationTokenCount
        let promptMs = completion.promptTime * 1000
        let generateMs = completion.generateTime * 1000
        promptTimes.append(promptMs)
        generateTimes.append(generateMs)
        print(
            "Generation run \(i): "
                + "prefill \(completion.promptTokenCount) tok / "
                + "\(String(format: "%.1f", promptMs))ms "
                + "(\(String(format: "%.2f", completion.promptTokensPerSecond)) tok/s), "
                + "decode \(completion.generationTokenCount) tok / "
                + "\(String(format: "%.1f", generateMs))ms "
                + "(\(String(format: "%.2f", completion.tokensPerSecond)) tok/s)"
        )
    }

    return LLMGenerationStats(
        promptTokenCount: promptTokenCount,
        generationTokenCount: generationTokenCount,
        promptTimeStats: BenchmarkStats(times: promptTimes),
        generateTimeStats: BenchmarkStats(times: generateTimes)
    )
}

// MARK: - System Prompt Cache Benchmarks

/// The system prompt of a prompt cache benchmark: a preamble followed by benchmark text.
enum BenchmarkSystemPrompt {
    /// The preamble of ``benchmarkSystemPromptCache(container:systemPrompt:systemPromptCharacterCount:coldQuestion:warmQuestion:maxTokens:kvCache:additionalContext:runs:)``.
    static let textQuestionsPreamble = "Answer questions about this text."

    /// The preamble of ``benchmarkMTPSystemPromptCache(container:drafter:scenarios:systemPrompt:systemPromptCharacterCount:blockSize:)``,
    /// whose turns ask math questions.
    static let mathTutorPreamble =
        "You are a careful math tutor. Answer the questions concisely and show brief working. "
        + "The text below is background reading only."

    /// Returns `systemPrompt` when given, else `preamble` followed by the first
    /// `characterCount` characters of the benchmark text.
    static func load(
        _ systemPrompt: String?, preamble: String, characterCount: Int
    ) async throws -> String {
        if let systemPrompt {
            return systemPrompt
        }
        let text = try await loadBenchmarkText(characterCount: characterCount)
        return preamble + "\n\n" + text
    }
}

/// One turn of a system prompt cache run.
public struct PromptCacheTurn: Sendable {
    /// Prompt tokens that were prefilled.
    public let promptTokenCount: Int
    /// Prompt tokens served from the reused cache prefix.
    public let cachedPromptTokenCount: Int
    /// Prefill time reported by the generation, in milliseconds.
    public let promptTimeMs: Double
    /// Wall-clock time from the call to the first generated chunk, in milliseconds.
    public let firstChunkMs: Double
}

/// Timings for sessions that share a cached system prompt.
public struct PromptCacheStats: Sendable {
    /// First turn of a new session: the whole prompt is prefilled.
    public let cold: [PromptCacheTurn]
    /// First turn of a session forked with an empty history: only the system prompt can match.
    public let fork: [PromptCacheTurn]
    /// Second turn of the cold session: the system prompt and the first turn can match.
    public let followUp: [PromptCacheTurn]

    public init(cold: [PromptCacheTurn], fork: [PromptCacheTurn], followUp: [PromptCacheTurn]) {
        self.cold = cold
        self.fork = fork
        self.followUp = followUp
    }

    public func printSummary(label: String) {
        print("\(label) system prompt cache results:")
        let coldMedian = BenchmarkStats(times: cold.map(\.promptTimeMs)).median
        for (name, turns) in [("Cold", cold), ("Fork", fork), ("Follow-up", followUp)] {
            let prefill = BenchmarkStats(times: turns.map(\.promptTimeMs))
            let first = BenchmarkStats(times: turns.map(\.firstChunkMs))
            let turn = turns.last!
            let total = turn.promptTokenCount + turn.cachedPromptTokenCount
            let efficiency = total > 0 ? Double(turn.cachedPromptTokenCount) / Double(total) : 0
            print(
                "  \(name): prefilled \(turn.promptTokenCount) tok, cached \(turn.cachedPromptTokenCount) tok "
                    + "(\(String(format: "%.1f", efficiency * 100))%), "
                    + "prefill median \(String(format: "%.1f", prefill.median))ms "
                    + "(stddev \(String(format: "%.1f", prefill.stdDev))ms, "
                    + "\(String(format: "%.1f", coldMedian / prefill.median))x vs cold), "
                    + "first chunk median \(String(format: "%.1f", first.median))ms"
            )
        }
    }
}

/// Benchmark the reuse of a cached system prompt. Each run prefills the system
/// prompt cold in a new ``ChatSession``. It then asks a second question two ways:
/// in a fork of the session with an empty history, and as a follow-up turn of the
/// session itself.
///
/// The questions differ, so a fork can match only the system prompt.
public func benchmarkSystemPromptCache(
    container: ModelContainer,
    systemPrompt: String? = nil,
    systemPromptCharacterCount: Int = 16_000,
    coldQuestion: String = "In one sentence, who is the main character?",
    warmQuestion: String = "In one sentence, where does the story begin?",
    maxTokens: Int = 16,
    kvCache: KVCacheConfiguration? = nil,
    additionalContext: [String: any Sendable]? = nil,
    runs: Int = 5
) async throws -> PromptCacheStats {
    let instructions = try await BenchmarkSystemPrompt.load(
        systemPrompt, preamble: BenchmarkSystemPrompt.textQuestionsPreamble,
        characterCount: systemPromptCharacterCount)
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    parameters.kvCache = kvCache

    func turn(_ session: ChatSession, _ question: String) async throws -> PromptCacheTurn {
        let start = CFAbsoluteTimeGetCurrent()
        var firstChunk: Double?
        var info: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: question) {
            if case .info(let i) = item {
                info = i
            } else if firstChunk == nil {
                firstChunk = (CFAbsoluteTimeGetCurrent() - start) * 1000
            }
        }
        guard let info else { throw LLMGenerationBenchmarkError.missingCompletionInfo }
        if firstChunk == nil {
            print("Warning: the turn produced no text, so the session does not record it")
        }
        return PromptCacheTurn(
            promptTokenCount: info.promptTokenCount,
            cachedPromptTokenCount: info.cachedPromptTokenCount,
            promptTimeMs: info.promptTime * 1000,
            firstChunkMs: firstChunk ?? (CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    func run() async throws -> (PromptCacheTurn, PromptCacheTurn, PromptCacheTurn) {
        let session = ChatSession(
            container, instructions: instructions, generateParameters: parameters,
            additionalContext: additionalContext)
        let cold = try await turn(session, coldQuestion)
        let fork = try await turn(await session.fork(history: []), warmQuestion)
        let followUp = try await turn(session, warmQuestion)
        return (cold, fork, followUp)
    }

    // Warm-up run, not recorded.
    _ = try await run()

    var cold: [PromptCacheTurn] = []
    var fork: [PromptCacheTurn] = []
    var followUp: [PromptCacheTurn] = []
    for i in 1 ... runs {
        let (c, f, u) = try await run()
        cold.append(c)
        fork.append(f)
        followUp.append(u)
        print(
            "Prompt cache run \(i): "
                + "cold \(c.promptTokenCount) tok / \(String(format: "%.1f", c.promptTimeMs))ms, "
                + "fork \(f.promptTokenCount) tok (cached \(f.cachedPromptTokenCount)) / "
                + "\(String(format: "%.1f", f.promptTimeMs))ms, "
                + "follow-up \(u.promptTokenCount) tok (cached \(u.cachedPromptTokenCount)) / "
                + "\(String(format: "%.1f", u.promptTimeMs))ms"
        )
    }

    return PromptCacheStats(cold: cold, fork: fork, followUp: followUp)
}

// MARK: - MTP Speculative Decoding Benchmarks

/// A prompt for the MTP benchmark, named by the kind of text it asks for.
public struct MTPBenchmarkPrompt: Sendable {
    public let name: String
    public let text: String

    public init(name: String, text: String) {
        self.name = name
        self.text = text
    }

    /// Math, code and prose, the categories the DFlash2 measurements on Qwen3.8 use.
    public static let defaults = [
        MTPBenchmarkPrompt(
            name: "math",
            text: "A train leaves at 9:40 and arrives at 13:05. It stops twice for 12 minutes. "
                + "How long was it moving? Then: a shop sells 3 pens for $4. How much do 18 pens cost?"
        ),
        MTPBenchmarkPrompt(
            name: "code",
            text: "Write a Swift function that returns the longest common subsequence of two "
                + "strings, with a few unit tests."),
        MTPBenchmarkPrompt(
            name: "prose", text: "Write a short travel blog post about a weekend in Hawaii."),
    ]
}

/// Plain and MTP decode timings for one prompt.
public struct MTPGenerationResult: Sendable {
    public let prompt: String
    public let generatedTokenCount: Int
    /// Median plain decode throughput, in tokens per second.
    public let baselineTokensPerSecond: Double
    /// Median MTP decode throughput, in tokens per second.
    public let mtpTokensPerSecond: Double
    /// Accepted drafts divided by proposed drafts, from the last MTP run.
    public let acceptanceRate: Double
    /// Tokens emitted per target verify pass, from the last MTP run. 1 is plain decoding.
    public let acceptanceLength: Double
    /// Whether the MTP tokens equal the plain greedy tokens.
    public let matchesBaseline: Bool

    public var speedup: Double { mtpTokensPerSecond / baselineTokensPerSecond }
}

/// Benchmark MTP speculative decoding against plain greedy decoding of the same
/// model. Each prompt runs once to warm up, then `runs` timed times each way.
public func benchmarkMTPGeneration(
    context: ModelContext,
    drafter: any MTPDrafterModel,
    prompts: [MTPBenchmarkPrompt] = MTPBenchmarkPrompt.defaults,
    maxTokens: Int = 192,
    blockSize: Int = 2,
    runs: Int = 3
) async throws -> [MTPGenerationResult] {
    let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

    func collect(_ stream: AsyncStream<TokenGeneration>) async throws -> (
        tokens: [Int], info: GenerateCompletionInfo
    ) {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let completion): info = completion
            }
        }
        guard let info else { throw LLMGenerationBenchmarkError.missingCompletionInfo }
        return (tokens, info)
    }

    var results: [MTPGenerationResult] = []
    for prompt in prompts {
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user(prompt.text)]))
        func plain() async throws -> (tokens: [Int], info: GenerateCompletionInfo) {
            try await collect(
                generateTokens(input: input, parameters: parameters, context: context))
        }
        func mtp() async throws -> (tokens: [Int], info: GenerateCompletionInfo) {
            try await collect(
                generateTokens(
                    input: input, parameters: parameters, context: context,
                    mtpDrafter: drafter, blockSize: blockSize))
        }

        _ = try await plain()
        _ = try await mtp()

        var plainRates: [Double] = []
        var mtpRates: [Double] = []
        var baseline: [Int] = []
        var last: (tokens: [Int], info: GenerateCompletionInfo)?
        for _ in 0 ..< runs {
            let p = try await plain()
            baseline = p.tokens
            plainRates.append(p.info.tokensPerSecond)
            let m = try await mtp()
            mtpRates.append(m.info.tokensPerSecond)
            last = m
        }
        guard let last else { continue }

        let proposed = last.info.proposedDraftTokens ?? 0
        let accepted = last.info.acceptedDraftTokens ?? 0
        let rounds = last.info.speculativeDecodingTelemetry?.roundCount ?? 0
        let result = MTPGenerationResult(
            prompt: prompt.name,
            generatedTokenCount: last.info.generationTokenCount,
            baselineTokensPerSecond: BenchmarkStats(times: plainRates).median,
            mtpTokensPerSecond: BenchmarkStats(times: mtpRates).median,
            acceptanceRate: proposed > 0 ? Double(accepted) / Double(proposed) : 0,
            acceptanceLength: rounds > 0
                ? Double(last.info.generationTokenCount) / Double(rounds) : 1,
            matchesBaseline: last.tokens == baseline)
        results.append(result)
        print(
            "MTP \(prompt.name): \(result.generatedTokenCount) tok, "
                + "plain \(String(format: "%.1f", result.baselineTokensPerSecond)) tok/s, "
                + "MTP \(String(format: "%.1f", result.mtpTokensPerSecond)) tok/s "
                + "(\(String(format: "%.2f", result.speedup))x), "
                + "acceptance \(String(format: "%.1f", result.acceptanceRate * 100))%, "
                + "\(String(format: "%.2f", result.acceptanceLength)) tok/verify, "
                + "matches plain: \(result.matchesBaseline)"
                + (last.info.passthroughReason.map { ", passthrough: \($0)" } ?? "")
        )
    }
    return results
}

// MARK: - MTP Chat Session Benchmarks

/// Writes benchmark result lines to standard output, where `xcodebuild` and
/// `swift test` show them beside the test results.
enum BenchmarkLog {
    /// Writes one result line.
    static func line(_ text: String) {
        // swiftlint:disable:next no_direct_standard_out_logs  benchmark lines are read from the test output, which a logger does not reach
        print(text)
    }
}

/// A turn that goes back to the end of an earlier reply of a chat and asks a
/// different question there, as an app does when its user edits a message and
/// regenerates the reply.
public struct MTPChatEdit: Sendable {
    /// The turns the edit keeps: it goes back to the end of the reply of turn
    /// `keptTurnCount - 1` and drops every turn after it.
    public let keptTurnCount: Int
    /// The prompt asked in place of the turns the edit drops.
    public let prompt: String

    /// Creates an edit that keeps the first `keptTurnCount` turns, then asks `prompt`.
    public init(keptTurnCount: Int, prompt: String) {
        self.keptTurnCount = keptTurnCount
        self.prompt = prompt
    }
}

/// A scripted chat that ``benchmarkMTPSystemPromptCache(container:drafter:scenarios:systemPrompt:systemPromptCharacterCount:blockSize:)``
/// runs in a ``ChatSession`` with MTP speculative decoding and in one without speculation.
public struct MTPChatScenario: Sendable {
    /// The name of the chat in the result lines.
    public let name: String
    /// The thinking switch of both sessions, ``ChatSession/thinkingEnabled``.
    public let thinking: Bool
    /// The user prompts, one for each turn, asked in order in one session.
    public let prompts: [String]
    /// A last turn that goes back to an earlier reply, or `nil` for none.
    public let edit: MTPChatEdit?
    /// Tokens each turn may generate.
    public let maxTokens: Int

    /// Creates a chat of `prompts`, with `edit` as its last turn when given.
    public init(
        name: String, thinking: Bool, prompts: [String], edit: MTPChatEdit? = nil, maxTokens: Int
    ) {
        self.name = name
        self.thinking = thinking
        self.prompts = prompts
        self.edit = edit
        self.maxTokens = maxTokens
    }

    /// Tokens a turn with thinking off may generate.
    public static let answerTokenBudget = 192
    /// Tokens a turn with thinking on may generate: the reasoning comes before the answer.
    public static let reasoningTokenBudget = 1_024
    /// Tokens a warm-up turn may generate.
    static let warmUpTokenBudget = 8

    /// Math follow-up turns with thinking off: each turn extends the turn before it.
    public static let followUps = MTPChatScenario(
        name: "follow-ups", thinking: false,
        prompts: [
            "A train leaves at 9:40 and arrives at 13:05. It stops twice for 12 minutes. "
                + "How long was it moving?",
            "A shop sells 3 pens for $4. How much do 18 pens cost? Show the working.",
            "Add 15% sales tax to that price. What is the total?",
        ],
        maxTokens: answerTokenBudget)

    /// Math follow-up turns with thinking on. The Qwen3.8 template drops the
    /// reasoning of earlier replies when it renders the history again, so each
    /// follow-up differs from the cached tokens of the turn before it.
    public static let thinkingFollowUps = MTPChatScenario(
        name: "thinking", thinking: true,
        prompts: [
            "What is 17 * 23? Show the working briefly.",
            "Divide that result by 7 and round to two decimals.",
            "Is the original product a prime number? Answer in one sentence.",
        ],
        maxTokens: reasoningTokenBudget)

    /// Two math turns, then an edit that goes back to the end of the first reply
    /// and asks a different second question.
    public static let editAndRegenerate = MTPChatScenario(
        name: "edit", thinking: false,
        prompts: [
            "A rectangle is 12 cm by 7 cm. What are its area and perimeter?",
            "Double both sides. What are the new area and perimeter?",
        ],
        edit: MTPChatEdit(
            keptTurnCount: 1,
            prompt: "Halve both sides instead. What are the new area and perimeter?"),
        maxTokens: answerTokenBudget)

    /// Follow-ups, follow-ups with thinking on, and an edit.
    public static let defaults = [followUps, thinkingFollowUps, editAndRegenerate]

    /// A one-turn chat that compiles the kernels of a decode path before the timed chats.
    static let warmUp = MTPChatScenario(
        name: "warm-up", thinking: false, prompts: ["Reply with one word: ready."],
        maxTokens: warmUpTokenBudget)
}

/// What one turn of an ``MTPChatScenario`` reported in its `.info` event.
public struct MTPChatTurn: Sendable {
    /// The position of the turn in its chat. An edit comes after every prompt.
    public let index: Int
    /// Whether the turn is the ``MTPChatScenario/edit`` of its chat.
    public let isEdit: Bool
    /// Prompt tokens the turn prefilled.
    public let promptTokenCount: Int
    /// Prompt tokens the reused cache served, ``GenerateCompletionInfo/cachedPromptTokenCount``.
    public let cachedPromptTokenCount: Int
    /// Prefill time, in milliseconds.
    public let promptTimeMs: Double
    /// Tokens the turn generated.
    public let generationTokenCount: Int
    /// Decode throughput, in tokens per second.
    public let decodeTokensPerSecond: Double
    /// Why the session ran the turn without speculation, from the `.info` event.
    public let fallbackReason: SpeculativeDecodingFallbackReason?
    /// Why the MTP iterator stopped drafting, from the `.info` event.
    public let passthroughReason: String?
    /// Tokens the drafter proposed, or `nil` when no MTP iterator ran.
    public let proposedDraftTokenCount: Int?
    /// Proposed tokens the target accepted, or `nil` when no MTP iterator ran.
    public let acceptedDraftTokenCount: Int?
    /// The generated text.
    public let text: String
    /// The generated tokens, with the candidates of each step when the session
    /// reported them.
    public let tokens: [MTPChatToken]

    init(
        index: Int, isEdit: Bool, info: GenerateCompletionInfo, text: String,
        tokens: [MTPChatToken]
    ) {
        self.index = index
        self.isEdit = isEdit
        self.tokens = tokens
        self.promptTokenCount = info.promptTokenCount
        self.cachedPromptTokenCount = info.cachedPromptTokenCount
        self.promptTimeMs = info.promptTime * 1000
        self.generationTokenCount = info.generationTokenCount
        self.decodeTokensPerSecond = info.tokensPerSecond
        self.fallbackReason = info.speculativeDecodingFallbackReason
        self.passthroughReason = info.passthroughReason
        self.proposedDraftTokenCount = info.proposedDraftTokens
        self.acceptedDraftTokenCount = info.acceptedDraftTokens
        self.text = text
    }

    /// The rendered prompt: the cached prefix and the prefilled tokens.
    public var totalPromptTokenCount: Int { promptTokenCount + cachedPromptTokenCount }

    /// Whether the drafter proposed tokens in this turn.
    public var speculated: Bool { (proposedDraftTokenCount ?? 0) > 0 }

    /// Prompt tokens the stateful MTP drafter prefilled, or `nil` for a session
    /// without speculation.
    ///
    /// No event reports this count. The MTP iterator feeds the drafter the
    /// tokens it feeds the target: the whole prompt on a cold cache, the suffix
    /// past the cached prefix on a warm one. So the drafter prefilled
    /// ``promptTokenCount`` when it drafted, and nothing when the session fell
    /// back or the iterator passed through before its first draft.
    public var drafterPrefillTokenCount: Int? {
        if fallbackReason != nil {
            return 0
        }
        guard proposedDraftTokenCount != nil else { return nil }
        return speculated ? promptTokenCount : 0
    }

    /// Accepted divided by proposed drafts, or `nil` when no draft was proposed.
    public var acceptanceRate: Double? {
        guard let proposed = proposedDraftTokenCount, proposed > 0 else { return nil }
        return Double(acceptedDraftTokenCount ?? 0) / Double(proposed)
    }

    /// Whether speculation ran, and why it did not when it was configured.
    public var speculationDescription: String {
        if let fallbackReason {
            return "fell back (\(fallbackReason))"
        }
        guard proposedDraftTokenCount != nil else { return "off" }
        let state = speculated ? "MTP" : "MTP without drafts"
        return passthroughReason.map { "\(state), passthrough: \($0)" } ?? state
    }

    /// One result line for the turn of `scenario` in the `session` named.
    func summaryLine(scenario: String, session: String) -> String {
        let turn = isEdit ? "turn \(index) (edit)" : "turn \(index)"
        let drafter = drafterPrefillTokenCount.map { "\($0) tok" } ?? "-"
        let acceptance = acceptanceRate.map { String(format: "%.1f%%", $0 * 100) } ?? "-"
        return "[\(scenario)] \(session) \(turn): prompt \(totalPromptTokenCount) tok, "
            + "cached \(cachedPromptTokenCount) tok, prefilled \(promptTokenCount) tok in "
            + "\(String(format: "%.1f", promptTimeMs))ms, drafter prefilled \(drafter), "
            + "speculation \(speculationDescription), decode \(generationTokenCount) tok at "
            + "\(String(format: "%.1f", decodeTokensPerSecond)) tok/s, acceptance \(acceptance)"
    }
}

/// The turns of one ``MTPChatScenario`` in the MTP session and in the plain session.
public struct MTPChatScenarioResult: Sendable {
    /// The chat that ran.
    public let scenario: MTPChatScenario
    /// The turns of the session with MTP speculative decoding.
    public let mtp: [MTPChatTurn]
    /// The same turns in the session without speculation.
    public let plain: [MTPChatTurn]

    /// Characters of text a result line quotes.
    static let excerptLength = 100

    /// Writes one line for each turn of each session, the start of the text of
    /// the turn, and where the MTP text leaves the plain text when they differ.
    public func printSummary() {
        for (mtpTurn, plainTurn) in zip(mtp, plain) {
            let speedup = mtpTurn.decodeTokensPerSecond / plainTurn.decodeTokensPerSecond
            let match = MTPChatTokenMatch(mtp: mtpTurn.tokens, plain: plainTurn.tokens)
            BenchmarkLog.line(
                mtpTurn.summaryLine(scenario: scenario.name, session: "MTP  ")
                    + ", text equal to plain: \(mtpTurn.text == plainTurn.text), "
                    + "tokens match plain: \(match.summary), "
                    + "decode \(String(format: "%.2f", speedup))x plain")
            BenchmarkLog.line(plainTurn.summaryLine(scenario: scenario.name, session: "plain"))
            BenchmarkLog.line(
                "[\(scenario.name)] turn \(mtpTurn.index) text starts "
                    + "<<<\(Self.excerpt(of: plainTurn.text[...]))>>>")
            if mtpTurn.text != plainTurn.text {
                BenchmarkLog.line(Self.divergenceLine(mtpTurn, plainTurn, scenario: scenario.name))
            }
        }
    }

    /// Where the text of `mtpTurn` leaves the text of `plainTurn`, and what each says there.
    private static func divergenceLine(
        _ mtpTurn: MTPChatTurn, _ plainTurn: MTPChatTurn, scenario: String
    ) -> String {
        let shared = zip(mtpTurn.text, plainTurn.text).prefix { $0 == $1 }.count
        return "[\(scenario)] turn \(mtpTurn.index) MTP text leaves plain text after "
            + "\(shared) of \(plainTurn.text.count) characters: MTP <<<"
            + excerpt(of: mtpTurn.text.dropFirst(shared)) + ">>> plain <<<"
            + excerpt(of: plainTurn.text.dropFirst(shared)) + ">>>"
    }

    /// The first ``excerptLength`` characters of `text`, on one line.
    private static func excerpt(of text: Substring) -> String {
        String(text.prefix(excerptLength)).replacingOccurrences(of: "\n", with: "\\n")
    }
}

/// Runs the turns of an ``MTPChatScenario`` in one ``ChatSession``.
private struct MTPChatSessionRunner {
    /// Messages one turn adds to the history: the prompt and the reply.
    static let messagesPerTurn = 2

    let container: ModelContainer
    let instructions: String
    let speculativeDecoding: SpeculativeDecodingConfig?

    /// The candidates each turn reports for each step: the two best for a plain
    /// session, so a parting from it can be told a near-tie, and none for a
    /// speculative session, which cannot report them.
    var candidateCount: Int? { speculativeDecoding == nil ? 2 : nil }

    /// Runs the prompts of `scenario` in a new session, then its edit in a fork
    /// whose history ends at the reply the edit goes back to.
    func run(_ scenario: MTPChatScenario) async throws -> [MTPChatTurn] {
        let session = ChatSession(
            container, instructions: instructions, speculativeDecoding: speculativeDecoding,
            generateParameters: GenerateParameters(maxTokens: scenario.maxTokens, temperature: 0))
        session.thinkingEnabled = scenario.thinking
        var turns: [MTPChatTurn] = []
        var history: [Chat.Message] = []
        for (index, prompt) in scenario.prompts.enumerated() {
            let turn = try await respond(in: session, to: prompt, index: index, isEdit: false)
            turns.append(turn)
            history += [.user(prompt), .assistant(turn.text)]
        }
        guard let edit = scenario.edit else { return turns }
        let branch = await session.fork(
            history: Array(history.prefix(edit.keptTurnCount * Self.messagesPerTurn)))
        turns.append(
            try await respond(in: branch, to: edit.prompt, index: turns.count, isEdit: true))
        return turns
    }

    /// Asks `prompt` in `session` and collects the text, the tokens and the
    /// `.info` event.
    private func respond(
        in session: ChatSession, to prompt: String, index: Int, isEdit: Bool
    ) async throws -> MTPChatTurn {
        var text = ""
        var tokens: [MTPChatToken] = []
        var info: GenerateCompletionInfo?
        let recorder = MTPChatTokenRecorder(candidateCount: candidateCount)
        for try await event in session.observe(to: prompt, using: recorder) {
            switch event {
            case .token(let token): tokens.append(token)
            case .generation(.chunk(let chunk)): text += chunk
            case .generation(.info(let completion)): info = completion
            case .generation(.toolCall), .generation(.rejectedToolCall): break
            }
        }
        guard let info else { throw LLMGenerationBenchmarkError.missingCompletionInfo }
        return MTPChatTurn(index: index, isEdit: isEdit, info: info, text: text, tokens: tokens)
    }
}

/// Benchmark MTP speculative decoding in a multi-turn ``ChatSession`` that reuses
/// its prompt cache, beside a session without speculation over the same turns.
///
/// Both sessions share a system prompt of `systemPromptCharacterCount` benchmark
/// characters and decode greedily, so their texts can be compared. Each
/// ``MTPChatScenario`` runs once in a new session of each kind, after an
/// unrecorded warm-up turn of each kind. Every turn writes one line for each
/// session: the prompt tokens, the cached and prefilled tokens, the prefill
/// time, the drafter prefill, whether MTP ran or why it fell back, the decode
/// throughput and the acceptance.
public func benchmarkMTPSystemPromptCache(
    container: ModelContainer,
    drafter: MTPDrafterContainer,
    scenarios: [MTPChatScenario] = MTPChatScenario.defaults,
    systemPrompt: String? = nil,
    systemPromptCharacterCount: Int = 8_000,
    blockSize: Int = 2
) async throws -> [MTPChatScenarioResult] {
    let instructions = try await BenchmarkSystemPrompt.load(
        systemPrompt, preamble: BenchmarkSystemPrompt.mathTutorPreamble,
        characterCount: systemPromptCharacterCount)
    let mtp = MTPChatSessionRunner(
        container: container, instructions: instructions,
        speculativeDecoding: try SpeculativeDecodingConfig(
            mtpDrafter: drafter, blockSize: blockSize))
    let plain = MTPChatSessionRunner(
        container: container, instructions: instructions, speculativeDecoding: nil)

    _ = try await plain.run(.warmUp)
    _ = try await mtp.run(.warmUp)

    var results: [MTPChatScenarioResult] = []
    for scenario in scenarios {
        let result = MTPChatScenarioResult(
            scenario: scenario, mtp: try await mtp.run(scenario),
            plain: try await plain.run(scenario))
        result.printSummary()
        results.append(result)
    }
    return results
}
