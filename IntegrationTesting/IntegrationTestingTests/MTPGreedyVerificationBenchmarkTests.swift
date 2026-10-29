// Copyright © 2026 Apple Inc.
//
// Times the batched greedy MTP verification against the sequential one, on verify logits as
// wide as the Qwen3.5 vocabulary. The suite loads no model.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests`

import BenchmarkHelpers
import Foundation
import MLX
import Testing
import os

@testable import MLXLMCommon

/// The log every measurement line of this suite goes to.
private let measurementLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "MTPGreedyVerificationBenchmark")

/// The verify blocks the benchmark times: the drafts of a round and its bonus row. The raw
/// value is the rows of the block.
private enum BenchmarkBlockSize: Int, CaseIterable {
    case oneDraft = 2
    case threeDrafts = 4
    case sevenDrafts = 8
}

/// The divisor that takes half of the drafts of a block, the middle accepted count.
private let halfOfDraftsDivisor = 2

/// The tokens of the Qwen3.5 vocabulary, the width of each verify row.
private let qwen35VocabularySize = 151_936

/// The token every verify row predicts, and every accepted draft.
private let predictedToken = 7

/// The draft that stands where the walk must reject.
private let rejectedDraft: Int32 = 9

/// The logit of the predicted token, and of every other token.
private let predictedLogit: Float = 1
private let otherLogit: Float = -1

/// The calls of each path before the timed calls, so that each kernel is compiled.
private let warmupIterations = 20

/// The calls one sample times.
private let timedIterations = 100

/// The samples of each path; odd, so the median is one sample.
private let repetitions = 5

private let microsecondsPerSecond = 1e6
private let attosecondsPerMicrosecond = 1e12

/// One way to verify a block of drafts.
private enum VerificationPath: CaseIterable {
    case sequential
    case batched

    /// The emitted tokens of this path for `drafts` against `logits`.
    func verify(logits: MLXArray, drafts: MLXArray) -> [Int] {
        switch self {
        case .sequential:
            SequentialGreedyMTPVerification.verify(logits: logits, draftTokens: drafts).tokens
        case .batched:
            batchedGreedyMTPVerification(logits: logits, draftTokens: drafts).tokens
        }
    }

    /// The mean microseconds of one call over `iterations` calls.
    func microsecondsPerCall(logits: MLXArray, drafts: MLXArray, iterations: Int) -> Double {
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0 ..< iterations {
            _ = verify(logits: logits, drafts: drafts)
        }
        let elapsed = (clock.now - start).components
        let microseconds =
            Double(elapsed.seconds) * microsecondsPerSecond
            + Double(elapsed.attoseconds) / attosecondsPerMicrosecond
        return microseconds / Double(iterations)
    }
}

/// Times ``batchedGreedyMTPVerification(logits:draftTokens:)`` against
/// ``SequentialGreedyMTPVerification`` for each block size, with no draft accepted, half of
/// them accepted, and every draft accepted.
@Suite(.serialized)
struct MTPGreedyVerificationBenchmarkTests {

    @Test func batchedVerificationCostByBlockSize() {
        for blockSize in BenchmarkBlockSize.allCases.map(\.rawValue) {
            let logits = makeLogits(blockSize: blockSize)
            let draftCount = blockSize - 1
            for accepted in [0, draftCount / halfOfDraftsDivisor, draftCount] {
                let drafts = makeDrafts(blockSize: blockSize, accepted: accepted)
                let batched = batchedGreedyMTPVerification(logits: logits, draftTokens: drafts)
                let reference = SequentialGreedyMTPVerification.verify(
                    logits: logits, draftTokens: drafts)
                #expect(batched.accepted == accepted)
                #expect(batched.tokens == reference.tokens)

                let medians = medianMicroseconds(logits: logits, drafts: drafts)
                let measurement = String(
                    format:
                        "[MTPGREEDYBENCH] block=%d accepted=%d sequential=%.1f us batched=%.1f us speedup=%.2fx",
                    blockSize, accepted, medians.sequential, medians.batched,
                    medians.sequential / medians.batched)
                measurementLog.info("\(measurement, privacy: .public)")
            }
        }
    }

    /// Verify logits of `blockSize` rows, each predicting ``predictedToken``.
    private func makeLogits(blockSize: Int) -> MLXArray {
        var data = [Float](repeating: otherLogit, count: blockSize * qwen35VocabularySize)
        for row in 0 ..< blockSize {
            data[row * qwen35VocabularySize + predictedToken] = predictedLogit
        }
        let logits = MLXArray(data, [blockSize, qwen35VocabularySize])
        eval(logits)
        return logits
    }

    /// The drafts of a `blockSize` block whose walk accepts the first `accepted` of them.
    private func makeDrafts(blockSize: Int, accepted: Int) -> MLXArray {
        let values = (0 ..< (blockSize - 1)).map {
            $0 == accepted ? rejectedDraft : Int32(predictedToken)
        }
        let drafts = MLXArray(values)
        eval(drafts)
        return drafts
    }

    /// The median microseconds of one call of each path. The two paths swap their order from
    /// one repetition to the next, so neither always runs on a warmer GPU.
    private func medianMicroseconds(
        logits: MLXArray, drafts: MLXArray
    ) -> (sequential: Double, batched: Double) {
        for path in VerificationPath.allCases {
            _ = path.microsecondsPerCall(
                logits: logits, drafts: drafts, iterations: warmupIterations)
        }
        let samples = (0 ..< repetitions).flatMap { repetition in
            let order = VerificationPath.allCases
            return (repetition.isMultiple(of: 2) ? order : order.reversed()).map { path in
                (
                    path: path,
                    microseconds: path.microsecondsPerCall(
                        logits: logits, drafts: drafts, iterations: timedIterations)
                )
            }
        }
        func median(_ path: VerificationPath) -> Double {
            BenchmarkStats(times: samples.filter { $0.path == path }.map(\.microseconds)).median
        }
        return (median(.sequential), median(.batched))
    }
}
