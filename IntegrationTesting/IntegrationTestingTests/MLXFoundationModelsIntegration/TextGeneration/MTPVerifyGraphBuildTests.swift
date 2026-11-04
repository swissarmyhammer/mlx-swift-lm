// Copyright © 2026 Apple Inc.
//
// What the verify forward of Qwen3.8-27B-4bit costs on the router's load path at 32k and 100k tokens
// of context: the host time that builds its graph, while the GPU idles, against the whole forward.
// The rates of MTP and plain decode at the same contexts go with it. Each test is short enough for
// several runs of each arm of a comparison. Run them in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPVerifyGraphBuildTests -only-testing:'IntegrationTestingTests/MTPSWEBenchScaleTests/routerDecodeRateAtLongContexts()'`
//
// Every measurement line goes to the unified log under the `MTP VERIFY BUILD:` and
// `MTP SWE-BENCH SCALE:` prefixes:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND (category == "MTPVerifyBuild" OR category == "MTPSWEBenchScale")'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// The time limit of each test, in minutes: a prefill of 100k tokens and a few hundred forwards.
private let verifyBuildTimeLimitMinutes = 30

/// The MLX language model, apart from the FoundationModels one of the same name.
private typealias TargetModel = MLXLMCommon.LanguageModel

/// The median cost of the verify forward of one row count, in milliseconds.
private struct VerifyBuildCost: Sendable {

    /// The rows of the forward: the bonus token and its drafts.
    let rows: Int

    /// The host time from the call of the target until it gives back the lazy output.
    let graphBuild: Double

    /// The time of the whole forward, the graph build and the GPU work.
    let forward: Double
}

/// Times the host graph build of each verify forward of a round on the router's load path.
///
/// The cache holds the context, as the iterator's prefill leaves it, and each forward appends to
/// it, as a round without a rollback does. The forwards start with the GPU idle, as a verify does
/// after the last confidence read of the drafter chain, and the forwards cycle through the row
/// counts of a round.
@Suite(.serialized, .timeLimit(.minutes(verifyBuildTimeLimitMinutes)))
struct MTPVerifyGraphBuildTests {

    /// The context lengths of the measurement, in tokens.
    private static let contextTokenCounts = [32_768, 100_000]

    /// The tokens of each chunk of the prefill.
    private static let prefillChunkTokenCount = 2_048

    /// The most history rounds the context may draw on.
    private static let maximumHistoryRoundCount = 512

    /// The forwards of each row count that do not count: the first ones trace and allocate.
    private static let warmUpForwards = 2

    /// The timed forwards of each row count.
    private static let timedForwards = 48

    /// The fewest rows of a verify: the bonus token and one draft.
    private static let fewestRows = 2

    /// The token of each row of a timed verify: its value does not change the work.
    private static let verifyToken: Int32 = 0

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1_000.0

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP VERIFY BUILD:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPVerifyBuild")

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: Qwen38FourBitMTPModel.isLocal))
    func verifyGraphBuildOnTheRouterPath() async throws {
        await releaseAllGPUMemory()
        let loaded = try await Qwen38FourBitMTPModel.loadWithDrafter(path: .router)
        let tokenizer = await loaded.container.tokenizer
        let blockSize = loaded.speculation.blockSize
        let historyTokens = (0 ..< Self.maximumHistoryRoundCount).flatMap {
            tokenizer.encode(text: SWEBenchTranscript.exchangeText($0), addSpecialTokens: false)
        }
        for contextTokenCount in Self.contextTokenCounts {
            let context = Array(historyTokens.prefix(contextTokenCount))
            try #require(context.count == contextTokenCount, "the history is too short")
            let costs = try await loaded.perform { modelContext, _ in
                try Self.measure(modelContext.model, context: context, blockSize: blockSize)
            }
            for cost in costs {
                Self.log(
                    """
                    context \(contextTokenCount) tokens, \(cost.rows) rows: graph build \
                    \(String(format: "%.3f", cost.graphBuild)) ms, forward \
                    \(String(format: "%.2f", cost.forward)) ms
                    """)
            }
            #expect(costs.allSatisfy { $0.graphBuild > 0 }, "\(contextTokenCount)")
        }
        await loaded.model.evict()
        await releaseAllGPUMemory()
    }

    /// Prefills `context` into a cache of `target`, then times the verify forwards of a round.
    ///
    /// - Parameters:
    ///   - target: The target.
    ///   - context: The tokens of the context.
    ///   - blockSize: The tokens of a round: the bonus token and up to `blockSize - 1` drafts.
    /// - Returns: The cost of the forward of each row count.
    /// - Throws: When the target has no state to give.
    private static func measure(
        _ target: any TargetModel, context: [Int], blockSize: Int
    ) throws -> [VerifyBuildCost] {
        let cache = try target.newCache(parameters: nil)
        var state = try prefill(target, context: context, cache: cache)
        state[mtpVerifyKeepsSnapshotsKey] = true
        let rowCounts = Array(fewestRows ... blockSize)
        let samples = sampleVerifies(target, rowCounts: rowCounts, cache: cache, state: state)
        return zip(rowCounts, samples).map { rows, samples in
            VerifyBuildCost(
                rows: rows, graphBuild: SampleMedian.of(samples.map(\.graphBuild)),
                forward: SampleMedian.of(samples.map(\.forward)))
        }
    }

    /// Prefills `context` into `cache` in chunks, with the drafter state asked of the target, as
    /// the iterator's prefill does.
    ///
    /// - Parameters:
    ///   - target: The target.
    ///   - context: The tokens of the context.
    ///   - cache: The cache of the target.
    /// - Returns: The state the last chunk gave.
    /// - Throws: When the target has no state to give.
    private static func prefill(
        _ target: any TargetModel, context: [Int], cache: [KVCache]
    ) throws -> LMOutput.State {
        var state = LMOutput.State()
        for start in stride(
            from: context.startIndex, to: context.endIndex, by: prefillChunkTokenCount)
        {
            state[mtpEmitFlagKey] = true
            let chunk = context[start ..< min(start + prefillChunkTokenCount, context.endIndex)]
            let output = target(
                LMInput.Text(tokens: MLXArray(chunk.map(Int32.init))[.newAxis]), cache: cache,
                state: state)
            eval(output.logits, cache.flatMap(\.state))
            state = try #require(output.state)
        }
        return state
    }

    /// Runs ``warmUpForwards`` untimed verifies of each row count, then ``timedForwards`` timed
    /// rounds that cycle through the row counts.
    ///
    /// - Parameters:
    ///   - target: The target.
    ///   - rowCounts: The rows of each verify of a round.
    ///   - cache: The cache of the target, which each verify appends to.
    ///   - state: The state of a verify.
    /// - Returns: For each row count, the samples of its timed verifies.
    private static func sampleVerifies(
        _ target: any TargetModel, rowCounts: [Int], cache: [KVCache], state: LMOutput.State
    ) -> [[(graphBuild: Double, forward: Double)]] {
        for rows in rowCounts {
            for _ in 0 ..< warmUpForwards {
                _ = timedVerify(target, rows: rows, cache: cache, state: state)
            }
        }
        var samples = rowCounts.map { _ in [(graphBuild: Double, forward: Double)]() }
        for _ in 0 ..< timedForwards {
            for (index, rows) in rowCounts.enumerated() {
                samples[index].append(timedVerify(target, rows: rows, cache: cache, state: state))
            }
        }
        return samples
    }

    /// One verify forward of `rows` tokens, from an idle GPU.
    ///
    /// - Returns: The milliseconds of its graph build, and of the whole forward.
    private static func timedVerify(
        _ target: any TargetModel, rows: Int, cache: [KVCache], state: LMOutput.State
    ) -> (graphBuild: Double, forward: Double) {
        let tokens = MLXArray(Array(repeating: verifyToken, count: rows))[.newAxis]
        eval(tokens)
        Stream.gpu.synchronize()
        let start = Date()
        let output = target(LMInput.Text(tokens: tokens), cache: cache, state: state)
        let built = Date()
        eval([output.logits] + (output.state?[mtpLastHiddenStatesKey].map { [$0] } ?? []))
        Stream.gpu.synchronize()
        return (
            graphBuild: built.timeIntervalSince(start) * millisecondsPerSecond,
            forward: Date().timeIntervalSince(start) * millisecondsPerSecond
        )
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info(
            "\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

extension MTPSWEBenchScaleTests {

    /// The context lengths of the short router rate, in tokens of content.
    private static let routerLongContextTokenCounts = [32_768, 100_000]

    /// Decodes at each context of ``routerLongContextTokenCounts`` with MTP and plain decode on
    /// the router's load path. Each sweep point holds what
    /// ``measureSweepPoint(_:runners:countTokens:)`` holds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: Qwen38FourBitMTPModel.isLocal))
    func routerDecodeRateAtLongContexts() async throws {
        let (runners, countTokens) = try await makeSweepRunners(on: .router)
        for contextTokenCount in Self.routerLongContextTokenCounts {
            _ = try await measureSweepPoint(
                contextTokenCount, runners: runners, countTokens: countTokens)
        }
        await releaseAllGPUMemory()
    }
}

#endif
