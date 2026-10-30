// Copyright © 2026 Apple Inc.

import MLXLMCommon
import Testing

@testable import BenchmarkHelpers

/// The noise a block sweep allows when it compares two throughputs: the noise the sweep
/// measured, never less than the noise of repeated quiet-load sweeps.
struct MTPBlockSweepNoiseAllowanceTests {

    private static let plainRate = 50.0
    private static let singleDraftRate = 100.0
    private static let prompt = "prose"
    private static let chainedSetting = MTPBlockSetting(
        blockSize: 4,
        minimumDraftProbability: MTPBlockSetting.reportedFastestMinimumDraftProbability)

    /// Half the quiet-load floor: a gap the floor admits.
    private static let gapWithinFloor = MTPBlockSweep.quietLoadNoiseFloor / 2

    /// Twice the quiet-load floor: a gap the floor rejects.
    private static let gapBeyondFloor = MTPBlockSweep.quietLoadNoiseFloor * 2

    /// The share by which the plain throughput of the chained row in ``noisySweep()`` exceeds
    /// that of the one-draft row. The spread over the larger rate, the median of two, is then
    /// a fifth: 20% noise, more than the quiet-load floor.
    private static let noisyPlainShare = 0.25

    private static let generatedTokens = 100
    private static let acceptance = 0.8
    private static let tokensPerVerify = 2.0

    /// A greedy result of ``prompt`` at `mtpRate` tokens per second, against `plainRate`.
    private static func result(mtpRate: Double, plainRate: Double = plainRate)
        -> MTPGenerationResult
    {
        MTPGenerationResult(
            prompt: prompt, generatedTokenCount: generatedTokens,
            baselineTokensPerSecond: plainRate, mtpTokensPerSecond: mtpRate,
            acceptanceRate: acceptance, acceptanceLength: tokensPerVerify,
            matchesBaseline: true, tokenMatch: nil, depthAcceptanceRates: [acceptance])
    }

    /// A sweep of one draft at ``singleDraftRate`` against ``plainRate``, and the chained
    /// setting at `chainedRate` against `chainedPlainRate`.
    private static func sweep(chainedRate: Double, chainedPlainRate: Double = plainRate)
        -> MTPBlockSweep
    {
        MTPBlockSweep(
            rows: [
                MTPBlockSweepRow(
                    setting: .singleDraft, results: [result(mtpRate: singleDraftRate)]),
                MTPBlockSweepRow(
                    setting: chainedSetting,
                    results: [result(mtpRate: chainedRate, plainRate: chainedPlainRate)]),
            ],
            loadAverageBefore: nil, loadAverageAfter: nil)
    }

    /// A sweep whose plain throughputs spread by ``noisyPlainShare``.
    private static func noisySweep() -> MTPBlockSweep {
        sweep(chainedRate: singleDraftRate, chainedPlainRate: plainRate * (1 + noisyPlainShare))
    }

    @Test func allowanceIsTheQuietLoadFloorWhenTheSweepMeasuredLessNoise() {
        #expect(Self.sweep(chainedRate: Self.singleDraftRate).noiseAllowance
            == MTPBlockSweep.quietLoadNoiseFloor)
    }

    @Test func allowanceIsTheMeasuredNoiseWhenItExceedsTheQuietLoadFloor() {
        let sweep = Self.noisySweep()
        #expect(sweep.noiseAllowance == sweep.noiseFraction)
    }

    @Test func rowSlowerWithinTheQuietLoadFloorIsNoSlowerThanSingleDraft() throws {
        let sweep = Self.sweep(chainedRate: Self.singleDraftRate * (1 - Self.gapWithinFloor))
        let row = try #require(sweep.row(for: Self.chainedSetting))
        #expect(sweep.isNoSlowerThanSingleDraft(row))
    }

    @Test func rowSlowerBeyondTheQuietLoadFloorIsSlowerThanSingleDraft() throws {
        let sweep = Self.sweep(chainedRate: Self.singleDraftRate * (1 - Self.gapBeyondFloor))
        let row = try #require(sweep.row(for: Self.chainedSetting))
        #expect(!sweep.isNoSlowerThanSingleDraft(row))
    }

    @Test func closingLineWritesTheNoiseAndTheAllowance() throws {
        let closing = try #require(Self.noisySweep().summaryLines().last)
        #expect(closing.contains("noise 20.0%, allowance 20.0%; "))
    }
}
