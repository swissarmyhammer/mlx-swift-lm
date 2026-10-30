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
        guard !depthAcceptanceRates.isEmpty else { return MTPBlockSweep.absentValue }
        return depthAcceptanceRates.enumerated()
            .map { "d\($0.offset + 1) \(BenchmarkLog.whole($0.element * 100))%" }
            .joined(separator: " ")
    }

    /// The result line of one prompt.
    ///
    /// - Parameter passthroughReason: Why the last MTP run stopped speculating, or `nil`.
    /// - Returns: The line.
    func summaryLine(passthroughReason: String?) -> String {
        "MTP \(prompt): \(generatedTokenCount) tok, "
            + "plain \(BenchmarkLog.decimal(baselineTokensPerSecond)) tok/s, "
            + "MTP \(BenchmarkLog.decimal(mtpTokensPerSecond)) tok/s "
            + "(\(BenchmarkLog.hundredths(speedup))x), "
            + "acceptance \(BenchmarkLog.decimal(acceptanceRate * 100))% "
            + "[\(depthAcceptanceSummary)], "
            + "\(BenchmarkLog.hundredths(acceptanceLength)) tok/verify, "
            + "tokens \(tokenMatch?.summary ?? (matchesBaseline ? "exact" : "sampled"))"
            + (passthroughReason.map { ", passthrough: \($0)" } ?? "")
    }
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

    /// The setting a `SpeculativeDecodingConfig` takes when it names none.
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

    /// The setting as a result line names it, `blockSize 4 pMin 0.75`.
    public var description: String {
        "blockSize \(blockSize) pMin \(BenchmarkLog.hundredths(Double(minimumDraftProbability)))"
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

    /// The line of the row in the sweep's table: the mean throughput, then each prompt's MTP
    /// and plain throughputs and the acceptance of each draft depth.
    func summaryLine() -> String {
        let prompts = results.map {
            "\($0.prompt) \(BenchmarkLog.decimal($0.mtpTokensPerSecond)) "
                + "(plain \(BenchmarkLog.decimal($0.baselineTokensPerSecond)), "
                + "\($0.depthAcceptanceSummary))"
        }
        return "MTP sweep \(setting): mean \(BenchmarkLog.decimal(meanMTPTokensPerSecond)) tok/s; "
            + prompts.joined(separator: "; ")
    }
}

/// The measurements of a sweep over ``MTPBlockSetting``s, and the machine load around them.
public struct MTPBlockSweep: Sendable {

    /// The candidates the plain decode reports for each step: the two best tokens, whose gap
    /// tells a near-tie.
    static let reportedCandidateCount = 2

    /// What a result line writes in place of a value the sweep did not measure.
    static let absentValue = "-"

    /// One row for each setting, in the order measured.
    public let rows: [MTPBlockSweepRow]

    /// The one-minute load average before the sweep, or `nil` when the system does not report
    /// it.
    public let loadAverageBefore: Double?

    /// The one-minute load average after the sweep, or `nil`.
    public let loadAverageAfter: Double?

    /// The row of the highest ``MTPBlockSweepRow/meanMTPTokensPerSecond``.
    public var fastest: MTPBlockSweepRow? {
        Self.fastest(among: rows)
    }

    /// The fastest row among those ``isNoSlowerThanSingleDraft(_:)`` admits: the setting a
    /// default may take, since it is the fastest combined and costs no prompt more than noise.
    public var fastestNoSlowerThanSingleDraft: MTPBlockSweepRow? {
        Self.fastest(among: rows.filter(isNoSlowerThanSingleDraft))
    }

    /// The row of `candidates` of the highest ``MTPBlockSweepRow/meanMTPTokensPerSecond``, or
    /// `nil` when there is none.
    private static func fastest(among candidates: [MTPBlockSweepRow]) -> MTPBlockSweepRow? {
        candidates.max { $0.meanMTPTokensPerSecond < $1.meanMTPTokensPerSecond }
    }

    /// Whether `row` decodes each prompt at least as fast as ``MTPBlockSetting/singleDraft``,
    /// within ``noiseAllowance``. `false` when the sweep did not measure one draft for each
    /// round.
    public func isNoSlowerThanSingleDraft(_ row: MTPBlockSweepRow) -> Bool {
        guard let singleDraft = self.row(for: .singleDraft) else { return false }
        return row.results.allSatisfy { result in
            singleDraft.result(for: result.prompt).map {
                result.mtpTokensPerSecond >= $0.mtpTokensPerSecond * (1 - noiseAllowance)
            } ?? false
        }
    }

    /// The least noise a comparison of two MTP throughputs of one sweep allows, as a fraction
    /// of a throughput.
    ///
    /// Two quiet-load sweeps (load average below 4) of Qwen3.8-27B-4bit on an M5 Ultra
    /// measured each setting and prompt twice. Net of the shift common to every row of the
    /// two sweeps, the MTP throughputs of one setting and prompt differed by a standard
    /// deviation of 0.37%, and by 1.1% at most. Each such gap is the difference of two
    /// measurements, as a comparison of two rows of one sweep is, so a gap within this floor
    /// is noise even when the plain throughputs of the sweep (``noiseFraction``) spread less.
    public static let quietLoadNoiseFloor = 0.011

    /// The noise a comparison of two throughputs of the sweep allows: ``noiseFraction``, and
    /// at least ``quietLoadNoiseFloor``.
    public var noiseAllowance: Double {
        Swift.max(noiseFraction, Self.quietLoadNoiseFloor)
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
        rows.map { $0.summaryLine() } + [closingLine()]
    }

    /// The last line of the sweep's table: the fastest rows, the noise and its allowance, and
    /// the load and thermal state around the sweep.
    private func closingLine() -> String {
        "MTP sweep fastest: \(Self.settingName(of: fastest)); "
            + "fastest no slower than one draft on any prompt: "
            + "\(Self.settingName(of: fastestNoSlowerThanSingleDraft)); "
            + "noise \(BenchmarkLog.decimal(noiseFraction * 100))%, "
            + "allowance \(BenchmarkLog.decimal(noiseAllowance * 100))%; "
            + "load average \(Self.loadName(loadAverageBefore)) before, "
            + "\(Self.loadName(loadAverageAfter)) after; "
            + "thermal state \(ProcessInfo.processInfo.thermalState.rawValue)"
    }

    /// The setting of `row` as a result line names it, or ``absentValue``.
    private static func settingName(of row: MTPBlockSweepRow?) -> String {
        row.map { "\($0.setting)" } ?? absentValue
    }

    /// `loadAverage` as a result line writes it, or ``absentValue``.
    private static func loadName(_ loadAverage: Double?) -> String {
        loadAverage.map(BenchmarkLog.hundredths) ?? absentValue
    }

    /// The one-minute load average of the system, or `nil` when it does not report it.
    static func currentLoadAverage() -> Double? {
        var loads = [Double](repeating: 0, count: 1)
        return getloadavg(&loads, Int32(loads.count)) == loads.count ? loads[0] : nil
    }
}

// MARK: - Runner

extension Benchmark {
    /// Benchmarks MTP speculative decoding of each setting of `settings` against plain
    /// decoding, greedy, with
    /// ``mtpGeneration(context:drafter:prompts:maxTokens:blockSize:minimumDraftProbability:runs:sampling:)``,
    /// and writes the table of the sweep.
    ///
    /// - Parameters:
    ///   - context: The model.
    ///   - drafter: The MTP drafter of the model.
    ///   - settings: The settings to measure, in order.
    ///   - prompts: The prompts, each a result of each row.
    ///   - maxTokens: The tokens each decode generates at most.
    ///   - runs: The timed runs of each decode; at least one.
    /// - Returns: The rows of the sweep and the load around it.
    /// - Throws: The error of a decode.
    public static func mtpBlockSweep(
        context: ModelContext,
        drafter: any MTPDrafterModel,
        settings: [MTPBlockSetting] = MTPBlockSetting.sweep,
        prompts: [MTPBenchmarkPrompt] = MTPBenchmarkPrompt.defaults,
        maxTokens: Int = defaultMTPMaxTokens,
        runs: Int = defaultMTPRuns
    ) async throws -> MTPBlockSweep {
        let loadAverageBefore = MTPBlockSweep.currentLoadAverage()
        var rows: [MTPBlockSweepRow] = []
        for setting in settings {
            BenchmarkLog.line("MTP sweep \(setting):")
            rows.append(
                MTPBlockSweepRow(
                    setting: setting,
                    results: try await mtpGeneration(
                        context: context, drafter: drafter, prompts: prompts,
                        maxTokens: maxTokens, blockSize: setting.blockSize,
                        minimumDraftProbability: setting.minimumDraftProbability, runs: runs)))
        }
        let sweep = MTPBlockSweep(
            rows: rows, loadAverageBefore: loadAverageBefore,
            loadAverageAfter: MTPBlockSweep.currentLoadAverage())
        sweep.summaryLines().forEach(BenchmarkLog.line)
        return sweep
    }

    /// The plain greedy tokens after `input`, each with the two most likely candidates of its
    /// step, so that ``MTPChatTokenMatch`` can tell a near-tie from a divergence.
    ///
    /// The decode runs outside the timed runs: asking for the candidates reads log
    /// probabilities beside the logits and does not change the tokens.
    ///
    /// - Parameters:
    ///   - input: The prompt.
    ///   - context: The model.
    ///   - parameters: The greedy parameters of the timed runs.
    ///   - count: The tokens the timed plain runs generated.
    /// - Returns: The first `count` plain tokens with their candidates.
    /// - Throws: The error of the token iterator.
    static func plainTokensWithCandidates(
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
}
