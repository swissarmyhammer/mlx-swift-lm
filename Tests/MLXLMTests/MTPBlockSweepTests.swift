// Copyright © 2026 Apple Inc.

import MLXLMCommon
import Testing

@testable import BenchmarkHelpers

/// The result lines and the row choice of the MTP block size sweep, read from fixed
/// measurements rather than from a model.
struct MTPBlockSweepTests {

    private static let plainRate = 50.0
    private static let generatedTokens = 100
    private static let chainedSetting = MTPBlockSetting(
        blockSize: 4,
        minimumDraftProbability: MTPBlockSetting.reportedFastestMinimumDraftProbability)
    private static let floorlessSetting = MTPBlockSetting(blockSize: 3, minimumDraftProbability: 0)

    /// A greedy result of `prompt` at `mtpRate` tokens per second, against ``plainRate``.
    private static func result(
        _ prompt: String, mtpRate: Double, depthAcceptanceRates: [Double] = [0.824]
    ) -> MTPGenerationResult {
        MTPGenerationResult(
            prompt: prompt, generatedTokenCount: generatedTokens,
            baselineTokensPerSecond: plainRate, mtpTokensPerSecond: mtpRate,
            acceptanceRate: 0.825, acceptanceLength: 2.25, matchesBaseline: true, tokenMatch: nil,
            depthAcceptanceRates: depthAcceptanceRates)
    }

    /// A sweep whose fastest row is slower than one draft on the code prompt.
    private static func sweep() -> MTPBlockSweep {
        MTPBlockSweep(
            rows: [
                MTPBlockSweepRow(
                    setting: .singleDraft,
                    results: [result("math", mtpRate: 60), result("code", mtpRate: 60)]),
                MTPBlockSweepRow(
                    setting: chainedSetting,
                    results: [result("math", mtpRate: 90), result("code", mtpRate: 50)]),
                MTPBlockSweepRow(
                    setting: floorlessSetting,
                    results: [result("math", mtpRate: 70), result("code", mtpRate: 62)]),
            ],
            loadAverageBefore: 1.25, loadAverageAfter: nil)
    }

    @Test func wholeRoundsToTheNearestWholeNumber() {
        #expect(BenchmarkLog.whole(82.4) == "82")
        #expect(BenchmarkLog.whole(60.6) == "61")
    }

    @Test func depthAcceptanceSummaryWritesEachDepthAsAWholePercent() {
        let summary = Self.result("math", mtpRate: 60, depthAcceptanceRates: [0.824, 0.606])
            .depthAcceptanceSummary
        #expect(summary == "d1 82% d2 61%")
    }

    @Test func depthAcceptanceSummaryOfARunWithoutDraftsIsADash() {
        let summary = Self.result("math", mtpRate: 60, depthAcceptanceRates: [])
            .depthAcceptanceSummary
        #expect(summary == "-")
    }

    @Test func resultLineWritesTheThroughputsTheAcceptanceAndTheTokens() {
        let line = Self.result("math", mtpRate: 75).summaryLine(passthroughReason: "budget")
        #expect(
            line
                == "MTP math: 100 tok, plain 50.0 tok/s, MTP 75.0 tok/s (1.50x), "
                + "acceptance 82.5% [d1 82%], 2.25 tok/verify, tokens exact, passthrough: budget")
    }

    @Test func settingDescriptionWritesTheFloorWithTwoDecimals() {
        #expect(Self.chainedSetting.description == "blockSize 4 pMin 0.75")
    }

    @Test func fastestIsTheRowOfTheHighestMeanThroughput() {
        #expect(Self.sweep().fastest?.setting == Self.chainedSetting)
    }

    @Test func fastestNoSlowerThanSingleDraftSkipsARowSlowerOnOnePrompt() {
        #expect(Self.sweep().fastestNoSlowerThanSingleDraft?.setting == Self.floorlessSetting)
    }

    @Test func rowLineWritesTheMeanAndEachPrompt() {
        let lines = Self.sweep().summaryLines()
        #expect(lines.count == Self.sweep().rows.count + 1)
        #expect(
            lines.first
                == "MTP sweep blockSize 2 pMin 0.00: mean 60.0 tok/s; "
                + "math 60.0 (plain 50.0, d1 82%); code 60.0 (plain 50.0, d1 82%)")
    }

    @Test func closingLineWritesTheFastestRowsTheNoiseAndTheLoad() throws {
        let closing = try #require(Self.sweep().summaryLines().last)
        #expect(
            closing.hasPrefix(
                "MTP sweep fastest: blockSize 4 pMin 0.75; "
                    + "fastest no slower than one draft on any prompt: blockSize 3 pMin 0.00; "
                    + "noise 0.0%, allowance 1.1%; load average 1.25 before, - after; "
                    + "thermal state "))
    }
}
