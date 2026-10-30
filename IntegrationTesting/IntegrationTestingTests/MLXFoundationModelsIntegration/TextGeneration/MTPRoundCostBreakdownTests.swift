// Copyright © 2026 Apple Inc.
//
// What one MTP round of Qwen3.8-27B-4bit with its 4-bit MTP head costs at 32k and at 100k tokens
// of context, part by part, on the router's load path and on the text path: the verify forward of
// the target at each row count, each chained drafter step, one host sync, and the one-token decode
// forward of plain decode at the same context. The ACP agent's SWE-bench run saw MTP fall from
// 62.7 tok/s below 20k tokens to 37.5 tok/s at 80k-103k, about plain decode's 38. Two suspects:
// the drafter's own attention runs over the whole context on each chained step, and the
// multi-row verify takes a slower attention path than the one-query vector kernel. Run it in
// Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPRoundCostBreakdownTests`
//
// Every measurement line goes to the unified log under the `MTP ROUND COST:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPRoundCost"'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// The time limit of each test, in minutes: two prefills of 32k and 100k tokens, with the
/// drafter fed chunk by chunk, and a few hundred timed forwards.
private let roundCostTimeLimitMinutes = 60

/// The MLX language model, apart from the FoundationModels one of the same name.
private typealias TargetModel = MLXLMCommon.LanguageModel

/// A value the benchmark hands across a `perform` boundary.
///
/// The benchmark reads the boxed drafter only inside one `ModelContainer.perform` of the target at
/// a time, and the test runs serially, thus no two tasks touch the value at once.
private final class PerformBox<Value>: @unchecked Sendable {
    let value: Value
    init(_ value: Value) { self.value = value }
}

/// The cost of the parts of one MTP round at one context length, in milliseconds.
private struct RoundCost: Sendable {

    /// The tokens of context the target and the drafter hold.
    let contextTokenCount: Int

    /// The entries the drafter's own cache holds: its chained steps attend over them.
    let drafterEntryCount: Int

    /// Whether the verify forward keeps the recurrent snapshots of an in-place rewind.
    let keepsSnapshots: Bool

    /// The one-token decode forward of plain decode.
    let plainDecode: Double

    /// The verify forward of the target, at one row and up to the block size.
    let verify: [Double]

    /// A drafter proposal of a block of two tokens and up to the block size: one draft, then one
    /// more chained step for each block size after it.
    let draft: [Double]

    /// One host sync: a small evaluation read back to the CPU.
    let hostSync: Double

    /// The cost of each chained drafter step after the first draft.
    var chainedSteps: [Double] { zip(draft.dropFirst(), draft).map { $0 - $1 } }

    /// The parts in a measurement line.
    var summary: String {
        let format = { (values: [Double]) in
            values.map { String(format: "%.2f", $0) }.joined(separator: ", ")
        }
        return """
            context \(contextTokenCount) tokens, drafter cache \(drafterEntryCount) entries, \
            verify keeps snapshots \(keepsSnapshots): plain one-token decode \
            \(String(format: "%.2f", plainDecode)) ms; verify forward by rows 1... [\(format(verify))] \
            ms; drafter proposal by block size 2... [\(format(draft))] ms, chained steps \
            [\(format(chainedSteps))] ms; host sync \(String(format: "%.3f", hostSync)) ms
            """
    }
}

/// Times the parts of one MTP round of Qwen3.8-27B-4bit with its MTP head at 32k and 100k tokens.
///
/// The parts run outside the MTP iterator, each with a GPU sync before and after it, on a target
/// cache and a drafter cache that hold the same context: the target prefills it in chunks and
/// feeds each chunk's hidden rows to the drafter, as the iterator's prefill does. Each part is
/// the median of ``timedRepeats`` runs. The verify forward and the plain forward append to the
/// target cache, and a drafter proposal is removed after it, thus the context grows by a few
/// tokens over the runs, which is noise next to 32k. Comparing the sum of the parts with the
/// milliseconds per round of `MTPSWEBenchScaleTests`' sweep at the same context shows what the
/// iterator adds.
@Suite(.serialized, .timeLimit(.minutes(roundCostTimeLimitMinutes)))
struct MTPRoundCostBreakdownTests {

    /// The context lengths of the breakdown, in tokens.
    private static let contextTokenCounts = [32_768, 100_000]

    /// The tokens of each chunk of the prefill.
    private static let prefillChunkTokenCount = 2_048

    /// The most history rounds the context may draw on.
    private static let maximumHistoryRoundCount = 512

    /// The runs of each timed part.
    private static let timedRepeats = 8

    /// The divisor that picks the middle run of the sorted runs.
    private static let medianDivisor = 2

    /// The smallest block of a drafter proposal: the bonus token and one draft.
    private static let smallestBlockSize = 2

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1_000.0

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP ROUND COST:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPRoundCost")

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: Qwen38FourBitMTPModel.LoadPath.allCases)
    func roundCostAtThirtyTwoAndOneHundredThousandTokens(
        path: Qwen38FourBitMTPModel.LoadPath
    ) async throws {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        // The guard off: the breakdown measures the raw cost of each part.
        let mtpModel = try Qwen38FourBitMTPModel.make(
            speculates: true, reasons: true, path: path, adaptivePolicy: .disabled)
        let speculation = try #require(mtpModel.speculativeDecoding)
        let container = try await mtpModel.loadContainer()
        let loadDrafter = await container.perform { speculation.drafterLoader(for: $0.model) }
        let drafterContainer = try await loadDrafter(speculation.drafterConfiguration)
        let drafter = await drafterContainer.perform { PerformBox($0.model) }
        let tokenizer = await container.tokenizer
        let historyTokens = (0 ..< Self.maximumHistoryRoundCount).flatMap {
            tokenizer.encode(text: SWEBenchTranscript.exchangeText($0), addSpecialTokens: false)
        }
        for contextTokenCount in Self.contextTokenCounts {
            let context = Array(historyTokens.prefix(contextTokenCount))
            try #require(context.count == contextTokenCount, "\(path): the history is too short")
            let cost = try await container.perform { modelContext in
                try Self.measureRound(
                    target: modelContext.model, drafter: drafter.value, context: context,
                    blockSize: speculation.blockSize)
            }
            Self.log("\(path) \(cost.summary)")
            #expect(cost.drafterEntryCount == contextTokenCount, "\(path) \(contextTokenCount)")
            #expect(cost.verify.allSatisfy { $0 > 0 } && cost.draft.allSatisfy { $0 > 0 })
        }
        await mtpModel.evict()
        await releaseAllGPUMemory()
    }

    /// Prefills `context` into a target cache and a drafter cache, then times each part of a round.
    ///
    /// - Parameters:
    ///   - target: The target.
    ///   - drafter: The MTP head of the target.
    ///   - context: The tokens of the context.
    ///   - blockSize: The tokens of a round: the bonus token and up to `blockSize - 1` drafts.
    /// - Returns: The cost of each part.
    /// - Throws: When the drafter cannot hold a resumable state, or a drafter append fails.
    private static func measureRound(
        target: any TargetModel, drafter: any MTPDrafterModel, context: [Int], blockSize: Int
    ) throws -> RoundCost {
        let drafter = try #require(
            drafter as? any ResumableMTPDrafterModel, "the drafter must keep a resumable state")
        let cache = try target.newCache(parameters: nil)
        var drafterState = drafter.makeState(parameters: nil)
        // Zero drafts the whole block on every proposal, so each chained step runs.
        drafterState.minimumDraftProbability = 0
        let prefilled = try prefill(
            target: target, drafter: drafter, context: context, cache: cache,
            drafterState: &drafterState)
        let rewindDepth = (target as? any SpeculativeCacheRewindModel)?
            .maximumNativeTargetCacheRewind ?? 0
        let keepsSnapshots = rewindDepth >= blockSize - 1
        let blockSizes = Array(smallestBlockSize ... blockSize)
        return RoundCost(
            contextTokenCount: context.count, drafterEntryCount: drafterState.entryCount,
            keepsSnapshots: keepsSnapshots,
            plainDecode: median {
                forwardMilliseconds(
                    target, tokens: [prefilled.bonus], cache: cache,
                    state: plainState(prefilled.state))
            },
            verify: (1 ... blockSize).map { rows in
                median {
                    forwardMilliseconds(
                        target, tokens: Array(repeating: prefilled.bonus, count: rows),
                        cache: cache,
                        state: verifyState(prefilled.state, keepsSnapshots: keepsSnapshots))
                }
            },
            draft: blockSizes.map { size in
                median {
                    draftMilliseconds(
                        drafter, target: target, prefilled: prefilled, cache: cache,
                        blockSize: size, state: &drafterState)
                }
            },
            hostSync: median { milliseconds { (MLXArray(Int32(1)) + 1).item(Int32.self) } })
    }

    /// What the prefill leaves for a round: the bonus token, the last hidden row and the state.
    private struct Prefilled {
        let bonus: Int
        let lastHidden: MLXArray
        let state: LMOutput.State?
    }

    /// Prefills `context` into `cache` in chunks and appends each chunk to the drafter, each
    /// token paired with the hidden row of the position before it.
    private static func prefill(
        target: any TargetModel, drafter: any ResumableMTPDrafterModel, context: [Int],
        cache: [KVCache], drafterState: inout MTPDrafterState
    ) throws -> Prefilled {
        var state: LMOutput.State?
        var bonus = 0
        var lastHidden = MLXArray(Int32(0))
        for start in stride(from: 0, to: context.count, by: prefillChunkTokenCount) {
            let end = min(start + prefillChunkTokenCount, context.count)
            var chunkState = state ?? LMOutput.State()
            chunkState[mtpEmitFlagKey] = true
            let output = target(
                LMInput.Text(tokens: MLXArray(context[start ..< end].map(Int32.init))[.newAxis]),
                cache: cache, state: chunkState)
            let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
            bonus = end < context.count ? context[end] : argMax(output.logits[0, -1]).item(Int.self)
            let shifted = Array(context[(start + 1) ..< end]) + [bonus]
            let appended = drafter.appendDrafterState(
                target: target, tokens: MLXArray(shifted.map(Int32.init))[.newAxis],
                targetHidden: hidden, positionDeltas: output.state?[mtpPositionDeltasKey],
                state: &drafterState, sampler: ArgMaxSampler())
            try #require(appended, "the drafter must take chunk \(start) ..< \(end)")
            eval(cache.flatMap(\.state) + drafterState.cache.flatMap(\.state))
            lastHidden = hidden[0..., (-1)..., 0...]
            state = output.state
        }
        return Prefilled(bonus: bonus, lastHidden: lastHidden, state: state)
    }

    /// The state of plain decode's forward: no drafter state asked of the target.
    private static func plainState(_ state: LMOutput.State?) -> LMOutput.State? {
        var plain = state
        plain?[mtpEmitFlagKey] = nil
        plain?[mtpVerifyKeepsSnapshotsKey] = nil
        return plain
    }

    /// The state of a verify forward: the drafter state asked of the target, and the recurrent
    /// snapshots of an in-place rewind when the iterator would keep them.
    private static func verifyState(
        _ state: LMOutput.State?, keepsSnapshots: Bool
    ) -> LMOutput.State {
        var verify = state ?? LMOutput.State()
        verify[mtpEmitFlagKey] = true
        verify[mtpVerifyKeepsSnapshotsKey] = keepsSnapshots ? true : nil
        return verify
    }

    /// The milliseconds of one forward of `tokens`, its logits and its emitted state evaluated.
    private static func forwardMilliseconds(
        _ target: any TargetModel, tokens: [Int], cache: [KVCache], state: LMOutput.State?
    ) -> Double {
        milliseconds {
            let output = target(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis]),
                cache: cache, state: state)
            eval([output.logits] + (output.state?[mtpLastHiddenStatesKey].map { [$0] } ?? []))
        }
    }

    /// The milliseconds of one drafter proposal of a block of `blockSize` tokens, which the
    /// state then removes again.
    private static func draftMilliseconds(
        _ drafter: any ResumableMTPDrafterModel, target: any TargetModel, prefilled: Prefilled,
        cache: [KVCache], blockSize: Int, state: inout MTPDrafterState
    ) -> Double {
        let queryOffset = cache.map(\.offset).max() ?? 0
        let elapsed = milliseconds {
            eval(
                drafter.draftBlock(
                    target: target, lastToken: MLXArray([Int32(prefilled.bonus)]),
                    lastHidden: prefilled.lastHidden, sharedKV: [:],
                    positionDeltas: prefilled.state?[mtpPositionDeltasKey],
                    queryOffset: queryOffset, blockSize: blockSize, state: &state,
                    sampler: ArgMaxSampler()))
        }
        _ = state.removeLastEntries(state.proposalAppended)
        return elapsed
    }

    /// The wall milliseconds of `body`, with a GPU sync before and after it.
    private static func milliseconds<Result>(_ body: () -> Result) -> Double {
        Stream.gpu.synchronize()
        let start = Date()
        _ = body()
        Stream.gpu.synchronize()
        return Date().timeIntervalSince(start) * millisecondsPerSecond
    }

    /// The median of ``timedRepeats`` runs of `sample`.
    private static func median(_ sample: () -> Double) -> Double {
        let samples = (0 ..< timedRepeats).map { _ in sample() }.sorted()
        return samples[samples.count / medianDivisor]
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
