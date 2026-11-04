// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing
import os

@testable import MLXLLM
@testable import MLXLMCommon

/// Times the host synchronizations a chained Qwen MTP round makes for its confidence floor, on
/// the tiny fixture drafter, on a quiet host and beside a synthetic CPU load.
///
/// A chain with a floor reads the drafter's top-1 probability on the host before it may draft
/// on. Each such read waits for the GPU and leaves it idle while the CPU builds the next step,
/// and a loaded host stretches that wait. The tiny drafter's forward costs next to nothing, so
/// what the gated chain spends beyond the ungated one is the cost of its reads. Each chain
/// reports its milliseconds on standard error; the test asserts the drafts each one makes.
@Suite(.serialized)
struct MTPHostSyncBenchmarkTests {

    /// The tokens of each round: the seed and three chained drafts.
    private static let blockSize = 5

    /// A floor every drafter step reaches, so a gated chain reads its confidence at every step
    /// and still drafts the whole round.
    private static let passingFloor = Float.leastNonzeroMagnitude

    /// The chains each timing averages.
    private static let timedChains = 60

    /// The chains each timing runs first, untimed.
    private static let warmupChains = 5

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1_000.0

    /// A processor that leaves the logits as they are, so a chain drafts through the processor
    /// path: the path a guided chain takes, which reads its confidence at every step.
    private struct IdentityProcessor: LogitProcessor {
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray { logits }
        mutating func didSample(token: MLXArray) {}
    }

    /// How one timed chain drafts.
    private enum ChainKind: String, CaseIterable {
        /// Tokens only, without a floor: no host read before the drafts.
        case ungated
        /// Tokens only, with ``passingFloor``.
        case gated
        /// Through ``IdentityProcessor``, with ``passingFloor``.
        case processorGated
    }

    @Test("a floor every step reaches drafts the ungated chain on every path, timed")
    func gatedChainsDraftTheUngatedChain() throws {
        let fixture = try Qwen35MTPFixture()
        var quiet: [ChainKind: (tokens: [Int], milliseconds: Double)] = [:]
        for kind in ChainKind.allCases {
            quiet[kind] = try Self.time(kind, fixture: fixture)
        }
        let load = BusyLoad(threads: ProcessInfo.processInfo.activeProcessorCount)
        var loaded: [ChainKind: Double] = [:]
        for kind in ChainKind.allCases {
            loaded[kind] = try Self.time(kind, fixture: fixture).milliseconds
        }
        load.stop()

        let ungated = try #require(quiet[.ungated]).tokens
        #expect(ungated.count == Self.blockSize - 1)
        for kind in ChainKind.allCases {
            #expect(try #require(quiet[kind]).tokens == ungated, "\(kind.rawValue)")
        }
        let report = ChainKind.allCases.map { kind in
            String(
                format: "%@ quiet %.3f ms, loaded %.3f ms", kind.rawValue,
                quiet[kind]?.milliseconds ?? .nan, loaded[kind] ?? .nan)
        }
        FileHandle.standardError.write(
            Data("[MTPHostSyncBenchmark] \(report.joined(separator: "; "))\n".utf8))
    }

    /// Floors across the top-1 probabilities of the fixture drafter's steps, about 0.036 to
    /// 0.075 over its 64 tokens, so the chains they gate stop at different depths.
    private static let spanningFloors: [Float] = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08]

    @Test("a tokens-only chain drafts what a processor chain drafts at each floor")
    func tokensOnlyChainMatchesTheProcessorChainAtEachFloor() throws {
        let fixture = try Qwen35MTPFixture()
        var draftCounts: Set<Int> = []

        for floor in Self.spanningFloors {
            var tokensState = QwenMTPDistributionDraftTests.preparedState(fixture)
            var processorState = QwenMTPDistributionDraftTests.preparedState(fixture)
            tokensState.minimumDraftProbability = floor
            processorState.minimumDraftProbability = floor
            let seed = (
                tokensState.seedToken, tokensState.seedHidden, tokensState.seedTopProbability
            )

            let tokens = try Self.chain(.gated, fixture: fixture, state: &tokensState, seed: seed)
            let processorTokens = try Self.chain(
                .processorGated, fixture: fixture, state: &processorState, seed: seed)

            #expect(tokens.tokens == processorTokens.tokens, "floor \(floor)")
            #expect(tokens.appended == processorTokens.appended, "floor \(floor)")
            draftCounts.insert(tokens.tokens.count)
        }

        #expect(draftCounts.count > 1)
    }

    /// Drafts ``warmupChains`` and then ``timedChains`` rounds of `kind` from the same prepared
    /// drafter state.
    ///
    /// - Returns: The drafts of the first round and the mean milliseconds of a timed round.
    private static func time(
        _ kind: ChainKind, fixture: Qwen35MTPFixture
    ) throws -> (tokens: [Int], milliseconds: Double) {
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)
        state.minimumDraftProbability = kind == .ungated ? 0 : passingFloor
        let seed = (state.seedToken, state.seedHidden, state.seedTopProbability)
        let first = try chain(kind, fixture: fixture, state: &state, seed: seed).tokens
        for _ in 0 ..< warmupChains {
            _ = try chain(kind, fixture: fixture, state: &state, seed: seed)
        }
        let clock = ContinuousClock()
        let elapsed = try clock.measure {
            for _ in 0 ..< timedChains {
                _ = try chain(kind, fixture: fixture, state: &state, seed: seed)
            }
        }
        let seconds =
            Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        return (first, seconds * millisecondsPerSecond / Double(timedChains))
    }

    /// Drafts one round of `kind` from `state`, waits for its drafts as a verify would, and
    /// leaves `state` as it found it.
    ///
    /// - Returns: The drafts and the tentative entries the round appended.
    private static func chain(
        _ kind: ChainKind, fixture: Qwen35MTPFixture, state: inout MTPDrafterState,
        seed: (MLXArray?, MLXArray?, MLXArray?)
    ) throws -> (tokens: [Int], appended: Int) {
        let bonus = MLXArray([QwenMTPDistributionDraftTests.firstBonus])
        let hidden = MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize])
        let offset = state.nextPosition
        let drafts: MLXArray
        switch kind {
        case .ungated, .gated:
            drafts = fixture.drafter.draftBlock(
                target: fixture.target, lastToken: bonus, lastHidden: hidden, sharedKV: [:],
                positionDeltas: nil, queryOffset: offset, blockSize: blockSize, state: &state,
                sampler: ArgMaxSampler())
        case .processorGated:
            drafts =
                fixture.drafter.draftBlockWithLogits(
                    target: fixture.target, lastToken: bonus, lastHidden: hidden, sharedKV: [:],
                    positionDeltas: nil, queryOffset: offset, blockSize: blockSize, state: &state,
                    processor: IdentityProcessor(), sampler: ArgMaxSampler()
                ).tokens
        }
        eval(drafts)
        let tokens = drafts.asArray(Int.self)
        let appended = state.proposalAppended
        let removed = state.removeLastEntries(appended)
        #expect(removed)
        (state.seedToken, state.seedHidden, state.seedTopProbability) = seed
        return (tokens, appended)
    }
}

/// Busy threads that keep every core of the host loaded until ``stop()``.
private final class BusyLoad: Sendable {

    /// Whether the threads keep spinning.
    private let running = OSAllocatedUnfairLock(initialState: true)

    /// Starts `threads` spinning threads.
    init(threads: Int) {
        for _ in 0 ..< threads {
            Thread { [self] in
                while running.withLock({ $0 }) {}
            }.start()
        }
    }

    /// Stops every thread.
    func stop() {
        running.withLock { $0 = false }
    }
}
