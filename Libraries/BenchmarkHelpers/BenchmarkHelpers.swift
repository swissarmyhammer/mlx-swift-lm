// Shared benchmark logic for measuring model loading, tokenizer performance,
// and download performance.
// Integration packages inject their own Downloader and TokenizerLoader.

import Foundation
import MLX
import MLXLMCommon
import os

// MARK: - Namespace

/// The namespace of the public benchmark runners.
///
/// Each runner writes one `BenchmarkLog` line for each timed run and returns its
/// measurements.
public enum Benchmark {
    /// Milliseconds in one second, to report a time in seconds as milliseconds.
    static let millisecondsPerSecond = 1000.0

    /// The milliseconds since `start`, a `CFAbsoluteTimeGetCurrent()` reading.
    static func millisecondsSince(_ start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * millisecondsPerSecond
    }

    /// Runs `operation` once to warm up, then `runs` timed times, and writes one
    /// line for each timed run.
    ///
    /// - Parameters:
    ///   - label: the name of the run in its line, such as `"LLM load"`
    ///   - runs: the timed runs; at least one
    ///   - repetitions: the times `operation` repeats its work, so each time is
    ///     the time of one repetition
    ///   - clearsCache: whether to clear the MLX buffer cache after the warm-up
    ///     and after each timed run, so a load does not reuse the buffers of the
    ///     load before it
    ///   - operation: the measured work; it returns the detail its line ends with
    /// - Returns: the time of each timed run, in milliseconds
    static func measure(
        _ label: String,
        runs: Int,
        repetitions: Int = 1,
        clearsCache: Bool = false,
        operation: () async throws -> String
    ) async throws -> BenchmarkStats {
        precondition(runs > 0, "runs must be greater than zero")
        precondition(repetitions > 0, "repetitions must be greater than zero")
        _ = try await operation()
        if clearsCache { Memory.clearCache() }

        var times: [Double] = []
        for run in 1 ... runs {
            let start = CFAbsoluteTimeGetCurrent()
            let detail = try await operation()
            let elapsed = millisecondsSince(start) / Double(repetitions)
            times.append(elapsed)
            BenchmarkLog.line("\(label) run \(run): \(BenchmarkLog.decimal(elapsed))ms\(detail)")
            if clearsCache { Memory.clearCache() }
        }
        return BenchmarkStats(times: times)
    }
}

// MARK: - Log

/// Writes benchmark result lines to the unified log, as the other benchmarks of
/// the package do.
///
/// `xcodebuild test` shows the lines beside the test results, tagged
/// `[BenchmarkHelpers]`. Outside a test run, read them with
/// `/usr/bin/log stream --level info --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "BenchmarkHelpers"'`.
enum BenchmarkLog {
    /// The log every benchmark line goes to.
    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "BenchmarkHelpers")

    /// Writes one result line.
    static func line(_ text: String) {
        logger.notice("\(text, privacy: .public)")
    }

    /// Writes one line about a measurement the benchmark could not take as asked.
    static func warning(_ text: String) {
        logger.warning("\(text, privacy: .public)")
    }

    /// `value` with one decimal, as a result line writes a time or a rate.
    static func decimal(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    /// `value` with two decimals, as a result line writes a throughput or a ratio.
    static func hundredths(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

// MARK: - No-Op Tokenizer

/// A tokenizer loader that returns a stub tokenizer. Useful for benchmarking
/// model loading in downloader integration packages that don't provide a
/// real tokenizer.
public struct NoOpTokenizerLoader: TokenizerLoader {
    /// Creates the loader.
    public init() {}

    /// Returns a tokenizer that encodes nothing and decodes to empty text.
    ///
    /// - Parameter directory: the tokenizer directory, which the stub does not read.
    /// - Returns: the stub tokenizer.
    public func load(from directory: URL) async throws -> any Tokenizer {
        NoOpTokenizer()
    }
}

/// A tokenizer that encodes nothing and decodes to empty text.
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

/// Summary statistics of the timed runs of a benchmark.
public struct BenchmarkStats: Sendable {
    /// The divisor that takes the middle position of the sorted times.
    private static let middleDivisor = 2

    /// The mean time.
    public let mean: Double
    /// The middle time of the sorted times; the later one of the two middle times
    /// for an even count.
    public let median: Double
    /// The population standard deviation of the times.
    public let stdDev: Double
    /// The shortest time.
    public let min: Double
    /// The longest time.
    public let max: Double

    /// Summarizes `times`, which holds at least one measurement.
    ///
    /// - Parameter times: the time of each run, in any one unit.
    public init(times: [Double]) {
        let sorted = times.sorted()
        guard let shortest = sorted.first, let longest = sorted.last else {
            preconditionFailure("BenchmarkStats requires at least one timing measurement")
        }
        self.min = shortest
        self.max = longest
        let mean = times.reduce(0, +) / Double(times.count)
        self.mean = mean
        self.median = sorted[sorted.count / Self.middleDivisor]

        let squaredDiffs = times.map { ($0 - mean) * ($0 - mean) }
        self.stdDev = sqrt(squaredDiffs.reduce(0, +) / Double(times.count))
    }

    /// Writes the statistics, in milliseconds, under `label`.
    ///
    /// - Parameter label: the name of the benchmark in the first line.
    public func logSummary(label: String) {
        BenchmarkLog.line("\(label) results:")
        BenchmarkLog.line("  Mean:   \(BenchmarkLog.decimal(mean))ms")
        BenchmarkLog.line("  Median: \(BenchmarkLog.decimal(median))ms")
        BenchmarkLog.line("  StdDev: \(BenchmarkLog.decimal(stdDev))ms")
        BenchmarkLog.line("  Range:  \(BenchmarkLog.decimal(min))-\(BenchmarkLog.decimal(max))ms")
    }
}
