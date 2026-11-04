// Copyright © 2026 Apple Inc.

import MLXLMCommon
import Testing

@testable import BenchmarkHelpers

/// The shared pieces of the benchmark runners: the timed-run loop, the statistics,
/// the draft acceptance rate and the chat turn transcript.
struct BenchmarkHelpersTests {

    private static let timedRuns = 3
    private static let proposedDrafts = 8
    private static let acceptedDrafts = 6
    private static let generatedTokens = 5

    @Test func measureRunsTheOperationOnceToWarmUpThenOnceForEachTimedRun() async throws {
        var calls = 0
        let stats = try await Benchmark.measure("unit", runs: Self.timedRuns) {
            calls += 1
            return ""
        }
        #expect(calls == Self.timedRuns + 1)
        #expect(stats.min <= stats.median && stats.median <= stats.max)
    }

    @Test func statsTakeTheMiddleOfTheSortedTimes() {
        let stats = BenchmarkStats(times: [3, 1, 2])
        #expect(stats.min == 1)
        #expect(stats.median == 2)
        #expect(stats.max == 3)
        #expect(stats.mean == 2)
    }

    @Test func draftAcceptanceIsNilWhenNothingWasProposed() {
        #expect(DraftAcceptance.rate(accepted: 0, proposed: 0) == nil)
    }

    @Test func draftAcceptanceDividesAcceptedByProposed() {
        let rate = DraftAcceptance.rate(
            accepted: Self.acceptedDrafts, proposed: Self.proposedDrafts)
        #expect(rate == Double(Self.acceptedDrafts) / Double(Self.proposedDrafts))
    }

    @Test func transcriptCollectsTheTextAndTheInfoOfATurn() async throws {
        let info = GenerateCompletionInfo(
            promptTokenCount: 0, generationTokenCount: Self.generatedTokens, promptTime: 0,
            generationTime: 0, stopReason: .stop)
        let stream = AsyncThrowingStream<Generation, Error> { continuation in
            continuation.yield(.chunk("Hello, "))
            continuation.yield(.chunk("world"))
            continuation.yield(.info(info))
            continuation.finish()
        }
        let transcript = try await ChatTurnTranscript.collect(stream)
        #expect(transcript.text == "Hello, world")
        #expect(transcript.info.generationTokenCount == Self.generatedTokens)
        #expect(transcript.firstChunkMs != nil)
    }

    @Test func transcriptOfATurnWithoutInfoThrows() async {
        let stream = AsyncThrowingStream<Generation, Error> { continuation in
            continuation.yield(.chunk("no info"))
            continuation.finish()
        }
        await #expect(throws: LLMGenerationBenchmarkError.self) {
            _ = try await ChatTurnTranscript.collect(stream)
        }
    }
}
