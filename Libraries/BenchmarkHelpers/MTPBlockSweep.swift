// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

// MARK: - One MTP generation result

extension SpeculativeDecodingTelemetry {

    /// The acceptance of each draft depth: element `k` is
    /// ``acceptanceRate(atDepth:)`` of depth `k + 1`.
    var depthAcceptanceRates: [Double] {
        draftedRoundCountsByDepth.indices.compactMap { acceptanceRate(atDepth: $0 + 1) }
    }
}

extension MTPGenerationResult {

    /// The acceptance of each draft depth as a result line shows it, `d1 82% d2 61% …`, or `-`
    /// when no round drafted.
    var depthAcceptanceSummary: String {
        guard !depthAcceptanceRates.isEmpty else { return "-" }
        return depthAcceptanceRates.enumerated()
            .map { "d\($0.offset + 1) \(String(format: "%.0f", $0.element * 100))%" }
            .joined(separator: " ")
    }

    /// The result line of one prompt.
    ///
    /// - Parameter passthroughReason: Why the last MTP run stopped speculating, or `nil`.
    /// - Returns: The line.
    func summaryLine(passthroughReason: String?) -> String {
        "MTP \(prompt): \(generatedTokenCount) tok, "
            + "plain \(String(format: "%.1f", baselineTokensPerSecond)) tok/s, "
            + "MTP \(String(format: "%.1f", mtpTokensPerSecond)) tok/s "
            + "(\(String(format: "%.2f", speedup))x), "
            + "acceptance \(String(format: "%.1f", acceptanceRate * 100))% "
            + "[\(depthAcceptanceSummary)], "
            + "\(String(format: "%.2f", acceptanceLength)) tok/verify, "
            + "tokens \(tokenMatch?.summary ?? (matchesBaseline ? "exact" : "sampled"))"
            + (passthroughReason.map { ", passthrough: \($0)" } ?? "")
    }
}

/// The plain greedy tokens after `input`, each with the two most likely candidates of its
/// step, so that ``MTPChatTokenMatch`` can tell a near-tie from a divergence.
///
/// The decode runs outside the timed runs: asking for the candidates reads log probabilities
/// beside the logits and does not change the tokens.
///
/// - Parameters:
///   - input: The prompt.
///   - context: The model.
///   - parameters: The greedy parameters of the timed runs.
///   - count: The tokens the timed plain runs generated.
/// - Returns: The first `count` plain tokens with their candidates.
/// - Throws: The error of the token iterator.
func plainTokensWithCandidates(
    input: LMInput, context: ModelContext, parameters: GenerateParameters, count: Int
) throws -> [MTPChatToken] {
    var candidateParameters = parameters
    candidateParameters.logProbabilities = MTPBlockSweep.reportedCandidateCount
    var iterator = try TokenIterator(
        input: input, model: context.model, parameters: candidateParameters)
    var tokens: [MTPChatToken] = []
    while tokens.count < count, let token = iterator.next() {
        tokens.append(
            MTPChatToken(
                token: token,
                candidates: iterator.lastLogProbabilities?.materialize().topLogProbabilities
                    ?? []))
    }
    return tokens
}

// MARK: - Block size and confidence floor sweep

/// One setting of the rounds of MTP speculative decoding: the tokens of each round and the
/// drafter top-1 probability below which a chained round stops drafting.
public struct MTPBlockSetting: Sendable, Hashable, CustomStringConvertible {

    /// The tokens of each round: the drafts and the bonus token.
    public let blockSize: Int

    /// The drafter top-1 probability below which a chained round stops drafting.
    public let minimumDraftProbability: Float

    /// - Parameters:
    ///   - blockSize: The tokens of each round.
    ///   - minimumDraftProbability: The confidence floor of a chained round.
    public init(blockSize: Int, minimumDraftProbability: Float) {
        self.blockSize = blockSize
        self.minimumDraftProbability = minimumDraftProbability
    }

    /// The setting a ``SpeculativeDecodingConfig`` takes when it names none.
    public static let shipped = MTPBlockSetting(
        blockSize: SpeculativeDecodingConfig.defaultMTPBlockSize,
        minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)

    /// One draft for each round and no floor: the rounds before chained drafts.
    public static let singleDraft = MTPBlockSetting(
        blockSize: MTPDraftBlockOutput.minimumChainedMTPBlockSize, minimumDraftProbability: 0)

    /// The block sizes a sweep measures: one draft, up to the most drafts Qwen3.8's hybrid
    /// target rewinds in place.
    public static let sweptBlockSizes = Array(
        MTPDraftBlockOutput.minimumChainedMTPBlockSize ... GatedDeltaDecodeKernel.mostTokens)

    /// The confidence floor llama.cpp users report fastest at five drafts
    /// (`--spec-draft-p-min 0.75`).
    public static let reportedFastestMinimumDraftProbability: Float = 0.75

    /// The confidence floors a sweep measures: none, and
    /// ``reportedFastestMinimumDraftProbability``.
    public static let sweptMinimumDraftProbabilities: [Float] = [
        0, reportedFastestMinimumDraftProbability,
    ]

    /// Every swept block size with every swept floor, except the floors of ``singleDraft``,
    /// whose one draft no floor stops.
    public static let sweep: [MTPBlockSetting] = sweptBlockSizes.flatMap { blockSize in
        sweptMinimumDraftProbabilities
            .filter { blockSize > singleDraft.blockSize || $0 == 0 }
            .map { MTPBlockSetting(blockSize: blockSize, minimumDraftProbability: $0) }
    }

    public var description: String {
        "blockSize \(blockSize) pMin \(String(format: "%.2f", minimumDraftProbability))"
    }
}

/// The measurements of one ``MTPBlockSetting``: one ``MTPGenerationResult`` for each prompt.
public struct MTPBlockSweepRow: Sendable {

    /// The setting the row measured.
    public let setting: MTPBlockSetting

    /// One result for each prompt.
    public let results: [MTPGenerationResult]

    /// The mean of the MTP throughputs of the prompts, in tokens per second: the combined
    /// speed of the setting.
    public var meanMTPTokensPerSecond: Double {
        results.map(\.mtpTokensPerSecond).reduce(0, +) / Double(Swift.max(results.count, 1))
    }

    /// The result of the prompt named `prompt`, or `nil`.
    public func result(for prompt: String) -> MTPGenerationResult? {
        results.first { $0.prompt == prompt }
    }
}

/// The measurements of a sweep over ``MTPBlockSetting``s, and the machine load around them.
public struct MTPBlockSweep: Sendable {

    /// The candidates the plain decode reports for each step: the two best tokens, whose gap
    /// tells a near-tie.
    static let reportedCandidateCount = 2

    /// One row for each setting, in the order measured.
    public let rows: [MTPBlockSweepRow]

    /// The one-minute load average before the sweep, or `nil` when the system does not report
    /// it.
    public let loadAverageBefore: Double?

    /// The one-minute load average after the sweep, or `nil`.
    public let loadAverageAfter: Double?

    /// The row of the highest ``MTPBlockSweepRow/meanMTPTokensPerSecond``.
    public var fastest: MTPBlockSweepRow? {
        rows.max { $0.meanMTPTokensPerSecond < $1.meanMTPTokensPerSecond }
    }

    /// The fastest row among those ``isNoSlowerThanSingleDraft(_:)`` admits: the setting a
    /// default may take, since it is the fastest combined and costs no prompt more than noise.
    public var fastestNoSlowerThanSingleDraft: MTPBlockSweepRow? {
        rows.filter(isNoSlowerThanSingleDraft)
            .max { $0.meanMTPTokensPerSecond < $1.meanMTPTokensPerSecond }
    }

    /// Whether `row` decodes each prompt at least as fast as ``MTPBlockSetting/singleDraft``,
    /// within ``noiseFraction``. `false` when the sweep did not measure one draft for each round.
    public func isNoSlowerThanSingleDraft(_ row: MTPBlockSweepRow) -> Bool {
        guard let singleDraft = self.row(for: .singleDraft) else { return false }
        return row.results.allSatisfy { result in
            singleDraft.result(for: result.prompt).map {
                result.mtpTokensPerSecond >= $0.mtpTokensPerSecond * (1 - noiseFraction)
            } ?? false
        }
    }

    /// The row of `setting`, or `nil` when the sweep did not measure it.
    public func row(for setting: MTPBlockSetting) -> MTPBlockSweepRow? {
        rows.first { $0.setting == setting }
    }

    /// The run-to-run noise of the measurements, as a fraction of a throughput: for each
    /// prompt, the spread of the plain throughput over the rows divided by its median, the
    /// largest over the prompts.
    ///
    /// Every row decodes each prompt plainly too, and plain decode does not change between
    /// rows, so its spread is what the machine alone moves a throughput by.
    public var noiseFraction: Double {
        let prompts = Set(rows.flatMap { $0.results.map(\.prompt) })
        return prompts.map { prompt in
            let plain = rows.compactMap { $0.result(for: prompt)?.baselineTokensPerSecond }
            guard !plain.isEmpty else { return 0 }
            let stats = BenchmarkStats(times: plain)
            return (stats.max - stats.min) / stats.median
        }.max() ?? 0
    }

    /// The lines of the sweep's table: one for each row, the fastest row, the noise, and the
    /// load around the sweep.
    public func summaryLines() -> [String] {
        let rowLines = rows.map { row in
            let prompts = row.results.map {
                "\($0.prompt) \(String(format: "%.1f", $0.mtpTokensPerSecond)) "
                    + "(plain \(String(format: "%.1f", $0.baselineTokensPerSecond)), "
                    + "\($0.depthAcceptanceSummary))"
            }
            return
                "MTP sweep \(row.setting): mean \(String(format: "%.1f", row.meanMTPTokensPerSecond)) "
                + "tok/s; " + prompts.joined(separator: "; ")
        }
        let load = [loadAverageBefore, loadAverageAfter]
            .map { $0.map { String(format: "%.2f", $0) } ?? "-" }
        return rowLines + [
            "MTP sweep fastest: \(fastest.map { "\($0.setting)" } ?? "-"); "
                + "fastest no slower than one draft on any prompt: "
                + "\(fastestNoSlowerThanSingleDraft.map { "\($0.setting)" } ?? "-"); "
                + "noise \(String(format: "%.1f", noiseFraction * 100))%; "
                + "load average \(load[0]) before, \(load[1]) after; "
                + "thermal state \(ProcessInfo.processInfo.thermalState.rawValue)"
        ]
    }

    /// The one-minute load average of the system, or `nil` when it does not report it.
    static func currentLoadAverage() -> Double? {
        var loads = [Double](repeating: 0, count: 1)
        return getloadavg(&loads, Int32(loads.count)) == loads.count ? loads[0] : nil
    }
}

/// Benchmarks MTP speculative decoding of each setting of `settings` against plain decoding,
/// greedy, with ``benchmarkMTPGeneration(context:drafter:prompts:maxTokens:blockSize:minimumDraftProbability:runs:sampling:)``,
/// and writes the table of the sweep.
///
/// - Parameters:
///   - context: The model.
///   - drafter: The MTP drafter of the model.
///   - settings: The settings to measure, in order.
///   - prompts: The prompts, each a result of each row.
///   - maxTokens: The tokens each decode generates at most.
///   - runs: The timed runs of each decode.
/// - Returns: The rows of the sweep and the load around it.
/// - Throws: The error of a decode.
public func benchmarkMTPBlockSweep(
    context: ModelContext,
    drafter: any MTPDrafterModel,
    settings: [MTPBlockSetting] = MTPBlockSetting.sweep,
    prompts: [MTPBenchmarkPrompt] = MTPBenchmarkPrompt.defaults,
    maxTokens: Int = 192,
    runs: Int = 3
) async throws -> MTPBlockSweep {
    let loadAverageBefore = MTPBlockSweep.currentLoadAverage()
    var rows: [MTPBlockSweepRow] = []
    for setting in settings {
        BenchmarkLog.line("MTP sweep \(setting):")
        rows.append(
            MTPBlockSweepRow(
                setting: setting,
                results: try await benchmarkMTPGeneration(
                    context: context, drafter: drafter, prompts: prompts, maxTokens: maxTokens,
                    blockSize: setting.blockSize,
                    minimumDraftProbability: setting.minimumDraftProbability, runs: runs)))
    }
    let sweep = MTPBlockSweep(
        rows: rows, loadAverageBefore: loadAverageBefore,
        loadAverageAfter: MTPBlockSweep.currentLoadAverage())
    sweep.summaryLines().forEach(BenchmarkLog.line)
    return sweep
}
