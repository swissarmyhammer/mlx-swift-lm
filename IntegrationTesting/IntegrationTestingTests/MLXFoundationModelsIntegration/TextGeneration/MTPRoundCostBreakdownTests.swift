// Copyright © 2026 Apple Inc.
//
// What one MTP round of Qwen3.8-27B-4bit with its 4-bit MTP head costs at 32k, 64k and 100k
// tokens of context, part by part, on the router's load path and on the text path: the verify
// forward of the target at each row count, each chained drafter step with each drafter attention
// window of the sweep, one host sync, and the one-token decode forward of plain decode at the same
// context. The ACP agent's SWE-bench run saw MTP fall from 62.7 tok/s below 20k tokens to 37.5
// tok/s at 80k-103k, about plain decode's 38. Two suspects: the drafter's own attention runs over
// the whole context on each chained step, and the multi-row verify takes a slower attention path
// than the one-query vector kernel. Run it in Release:
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

/// The time limit of each test, in minutes: three prefills of 32k, 64k and 100k tokens, with each
/// drafter state fed chunk by chunk, and a few hundred timed forwards.
private let roundCostTimeLimitMinutes = 90

/// The MLX language model, apart from the FoundationModels one of the same name.
private typealias TargetModel = MLXLMCommon.LanguageModel

/// The cost of a drafter proposal with one drafter attention window, in milliseconds.
private struct DrafterWindowCost: Sendable {

    /// The window of the drafter, or `nil` for the whole context.
    let window: Int?

    /// The entries the drafter state holds: its chained steps attend over the newest `window`
    /// of them.
    let entryCount: Int

    /// A drafter proposal of a block of two tokens and up to the block size: one draft, then one
    /// more chained step for each block size after it.
    let draft: [Double]

    /// The cost of each chained drafter step after the first draft.
    var chainedSteps: [Double] { zip(draft.dropFirst(), draft).map { $0 - $1 } }
}

/// The cost of the parts of one MTP round at one context length, in milliseconds.
private struct RoundCost: Sendable {

    /// The tokens of context the target and the drafter hold.
    let contextTokenCount: Int

    /// Whether the verify forward keeps the recurrent snapshots of an in-place rewind.
    let keepsSnapshots: Bool

    /// The one-token decode forward of plain decode.
    let plainDecode: Double

    /// The verify forward of the target, at one row and up to the block size.
    let verify: [Double]

    /// The drafter proposals with each drafter window of the sweep.
    let drafts: [DrafterWindowCost]

    /// One host sync: a small evaluation read back to the CPU.
    let hostSync: Double

    /// The parts in measurement lines: one for the target, then one for each drafter window.
    var summary: [String] {
        let format = { (values: [Double]) in
            values.map { String(format: "%.2f", $0) }.joined(separator: ", ")
        }
        let target = """
            context \(contextTokenCount) tokens, verify keeps snapshots \(keepsSnapshots): \
            plain one-token decode \(String(format: "%.2f", plainDecode)) ms; verify forward by \
            rows 1... [\(format(verify))] ms; host sync \(String(format: "%.3f", hostSync)) ms
            """
        return [target]
            + drafts.map { drafterCost in
                """
                context \(contextTokenCount) tokens, drafter window \
                \(DrafterWindowSweep.label(of: drafterCost.window)), drafter state \
                \(drafterCost.entryCount) entries: drafter proposal by block size 2... \
                [\(format(drafterCost.draft))] ms, chained steps \
                [\(format(drafterCost.chainedSteps))] ms
                """
            }
    }
}

/// Times the parts of one MTP round of Qwen3.8-27B-4bit with its MTP head at 32k, 64k and 100k
/// tokens, with a drafter of each window of ``DrafterWindowSweep/windowTokenCounts``.
///
/// The parts run outside the MTP iterator, each with a GPU sync before and after it, on a target
/// cache and drafter caches that hold the same context: the target prefills it in chunks and
/// feeds each chunk's hidden rows to each drafter state, as the iterator's prefill does. Each
/// part is the median of ``timedRepeats`` runs. The verify forward and the plain forward append
/// to the target cache, and a drafter proposal is removed after it, thus the context grows by a
/// few tokens over the runs, which is noise next to 32k. Comparing the sum of the parts with the
/// milliseconds per round of `MTPSWEBenchScaleTests`' sweep at the same context shows what the
/// iterator adds.
@Suite(.serialized, .timeLimit(.minutes(roundCostTimeLimitMinutes)))
struct MTPRoundCostBreakdownTests {

    /// The context lengths of the breakdown, in tokens.
    private static let contextTokenCounts = DrafterWindowSweep.contextTokenCounts

    /// The tokens of each chunk of the prefill.
    private static let prefillChunkTokenCount = 2_048

    /// The most history rounds the context may draw on.
    private static let maximumHistoryRoundCount = 512

    /// The runs of each timed part.
    private static let timedRepeats = 8

    /// The smallest block of a drafter proposal: the bonus token and one draft.
    private static let smallestBlockSize = 2

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP ROUND COST:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPRoundCost")

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: Qwen38FourBitMTPModel.LoadPath.allCases)
    func roundCostFromThirtyTwoToOneHundredThousandTokens(
        path: Qwen38FourBitMTPModel.LoadPath
    ) async throws {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        let loaded = try await Qwen38FourBitMTPModel.loadWithDrafter(path: path)
        let tokenizer = await loaded.container.tokenizer
        let blockSize = loaded.speculation.blockSize
        let historyTokens = (0 ..< Self.maximumHistoryRoundCount).flatMap {
            tokenizer.encode(text: SWEBenchTranscript.exchangeText($0), addSpecialTokens: false)
        }
        for contextTokenCount in Self.contextTokenCounts {
            let context = Array(historyTokens.prefix(contextTokenCount))
            try #require(context.count == contextTokenCount, "\(path): the history is too short")
            let cost = try await loaded.perform { modelContext, drafter in
                try Self.measureRound(
                    target: modelContext.model, drafter: drafter, context: context,
                    blockSize: blockSize)
            }
            for line in cost.summary {
                Self.log("\(path) \(line)")
            }
            #expect(cost.verify.allSatisfy { $0 > 0 }, "\(path) \(contextTokenCount)")
            for drafterCost in cost.drafts {
                #expect(
                    drafterCost.entryCount == contextTokenCount, "\(path) \(contextTokenCount)")
                #expect(drafterCost.draft.allSatisfy { $0 > 0 }, "\(path) \(contextTokenCount)")
            }
        }
        await loaded.model.evict()
        await releaseAllGPUMemory()
    }

    /// Prefills `context` into a target cache and a drafter state of each drafter window of the
    /// sweep, then times each part of a round.
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
        let windows = DrafterWindowSweep.windowTokenCounts
        var drafterStates = try windows.map { window in
            var state = drafter.makeState(
                parameters: try DrafterWindowSweep.parameters(window: window))
            // Zero drafts the whole block on every proposal, so each chained step runs.
            state.minimumDraftProbability = 0
            return state
        }
        let prefilled = try prefill(
            target: target, drafter: drafter, context: context, cache: cache,
            drafterStates: &drafterStates)
        let rewindDepth = (target as? any SpeculativeCacheRewindModel)?
            .maximumNativeTargetCacheRewind ?? 0
        let keepsSnapshots = rewindDepth >= blockSize - 1
        let blockSizes = Array(smallestBlockSize ... blockSize)
        return RoundCost(
            contextTokenCount: context.count, keepsSnapshots: keepsSnapshots,
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
            drafts: windows.indices.map { index in
                DrafterWindowCost(
                    window: windows[index], entryCount: drafterStates[index].entryCount,
                    draft: blockSizes.map { size in
                        median {
                            draftMilliseconds(
                                drafter, target: target, prefilled: prefilled, cache: cache,
                                blockSize: size, state: &drafterStates[index])
                        }
                    })
            },
            hostSync: median { milliseconds { (MLXArray(Int32(1)) + 1).item(Int32.self) } })
    }

    /// What the prefill leaves for a round: the bonus token, the last hidden row and the state.
    private struct Prefilled {
        let bonus: Int
        let lastHidden: MLXArray
        let state: LMOutput.State?
    }

    /// Prefills `context` into `cache` in chunks and appends each chunk to each drafter state,
    /// each token paired with the hidden row of the position before it.
    private static func prefill(
        target: any TargetModel, drafter: any ResumableMTPDrafterModel, context: [Int],
        cache: [KVCache], drafterStates: inout [MTPDrafterState]
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
            let shifted = MLXArray((Array(context[(start + 1) ..< end]) + [bonus]).map(Int32.init))
            for index in drafterStates.indices {
                let appended = drafter.appendDrafterState(
                    target: target, tokens: shifted[.newAxis], targetHidden: hidden,
                    positionDeltas: output.state?[mtpPositionDeltasKey],
                    state: &drafterStates[index], sampler: ArgMaxSampler())
                try #require(appended, "the drafter must take chunk \(start) ..< \(end)")
            }
            eval(cache.flatMap(\.state) + drafterStates.flatMap { $0.cache.flatMap(\.state) })
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
        GPUWallTime.milliseconds(body)
    }

    /// The median of ``timedRepeats`` runs of `sample`.
    private static func median(_ sample: () -> Double) -> Double {
        GPUWallTime.median(repeats: timedRepeats, sample)
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
