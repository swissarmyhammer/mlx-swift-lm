// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// The adaptive guard replayed over the round and plain-step costs a real Qwen3.8 decode
/// measured with the guard off, on an M5 Ultra, at 2k to 100k tokens of context, on the text
/// and the router load paths.
///
/// At every point the measured rounds out-decode plain decode, by 1.19x to 1.75x, so a guard
/// that backs off there loses throughput. Real rounds do not all cost the same: a stretch of
/// rejected drafts emits one token a round, and a host stall stretches a round. The replay
/// keeps each point's measured means and gives every cycle of rounds one burst of rejections
/// followed by one stalled round.
@Suite
struct MTPAdaptiveReplayTests {

    /// The guard-off costs of one point of the sweep.
    struct Point: CustomTestStringConvertible, Sendable {

        /// The load path and the context of the point.
        let name: String

        /// The milliseconds of one round.
        let roundMilliseconds: Double

        /// The tokens one round emitted: the accepted drafts and the bonus token.
        let tokensPerRound: Double

        /// The milliseconds of one plain decode step.
        let plainMilliseconds: Double

        var testDescription: String { name }
    }

    /// The sweep's points: guard-off milliseconds per round and tokens per verify, and plain
    /// milliseconds per token.
    static let points = [
        Point(name: "text 2k", roundMilliseconds: 33.2, tokensPerRound: 2.92, plainMilliseconds: 18.1),
        Point(name: "text 8k", roundMilliseconds: 39.4, tokensPerRound: 3.75, plainMilliseconds: 18.4),
        Point(name: "text 32k", roundMilliseconds: 45.4, tokensPerRound: 3.39, plainMilliseconds: 20.9),
        Point(name: "text 64k", roundMilliseconds: 57.5, tokensPerRound: 3.01, plainMilliseconds: 26.1),
        Point(name: "text 100k", roundMilliseconds: 74.7, tokensPerRound: 3.26, plainMilliseconds: 31.0),
        Point(name: "router 2k", roundMilliseconds: 36.4, tokensPerRound: 2.78, plainMilliseconds: 18.1),
        Point(name: "router 8k", roundMilliseconds: 49.3, tokensPerRound: 3.76, plainMilliseconds: 18.7),
        Point(name: "router 32k", roundMilliseconds: 43.8, tokensPerRound: 2.43, plainMilliseconds: 21.4),
        Point(name: "router 64k", roundMilliseconds: 78.9, tokensPerRound: 3.84, plainMilliseconds: 26.8),
        Point(name: "router 100k", roundMilliseconds: 104.3, tokensPerRound: 3.95, plainMilliseconds: 33.0),
    ]

    /// The configured block size of the sweep.
    private static let blockSize = 5

    /// The tokens of each replayed decode, the sweep's budget.
    private static let decodeTokens = 1024

    /// The rounds of one replay cycle.
    private static let cycleRounds = 20

    /// The rounds of the rejection burst that opens each cycle, one token each.
    private static let burstRounds = 4

    /// The cost of the stalled round after the burst, as a multiple of the mean round.
    private static let stallFactor = 3.0

    /// The cycles whose mean costs the replay check reads.
    private static let checkedCycles = 50

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1000.0

    /// One round of the replay: the tokens it emitted and its seconds.
    struct Round {
        let tokens: Int
        let seconds: Double
    }

    /// The rounds of `point`, in order: each cycle opens with the burst, then the stalled round,
    /// then rounds at the mean cost. Across the replay the rounds keep the point's mean tokens
    /// and mean seconds.
    struct Replay {
        let point: Point

        /// The tokens of each round outside the bursts, so a cycle emits the point's mean.
        private var otherRoundTokens: Double {
            (Double(cycleRounds) * point.tokensPerRound - Double(burstRounds))
                / Double(cycleRounds - burstRounds)
        }

        /// The seconds of a round at the mean cost, so a cycle with its stall keeps the mean.
        private var regularSeconds: Double {
            point.roundMilliseconds / millisecondsPerSecond
                * (Double(cycleRounds) - stallFactor) / Double(cycleRounds - 1)
        }

        /// The `index`-th round of the replay.
        func round(_ index: Int) -> Round {
            let position = index % cycleRounds
            guard position >= burstRounds else {
                return Round(tokens: 1, seconds: regularSeconds)
            }
            let other = (index / cycleRounds) * (cycleRounds - burstRounds) + position - burstRounds
            let tokens = Int((Double(other + 1) * otherRoundTokens).rounded(.down))
                - Int((Double(other) * otherRoundTokens).rounded(.down))
            let seconds =
                position == burstRounds
                ? stallFactor * point.roundMilliseconds / millisecondsPerSecond : regularSeconds
            return Round(tokens: tokens, seconds: seconds)
        }
    }

    /// Replays one decode of `point` through a controller of the shipped settings.
    private func replay(_ point: Point) -> MTPAdaptiveController {
        var controller = MTPAdaptiveController(policy: .enabled, blockSize: Self.blockSize)
        let replay = Replay(point: point)
        var emitted = 0
        var rounds = 0
        while emitted < Self.decodeTokens {
            if controller.nextStep == .decodePlainly {
                controller.recordPlainStep(
                    tokens: 1, seconds: point.plainMilliseconds / Self.millisecondsPerSecond)
                emitted += 1
            } else {
                let round = replay.round(rounds)
                rounds += 1
                let tokens = min(round.tokens, controller.blockSize)
                controller.recordRound(tokens: tokens, seconds: round.seconds)
                emitted += tokens
            }
        }
        return controller
    }

    @Test("the replay keeps each point's measured round and plain costs", arguments: points)
    func replayKeepsTheMeasuredCosts(point: Point) {
        let replay = Replay(point: point)
        let rounds = (0 ..< Self.cycleRounds * Self.checkedCycles).map(replay.round)
        let tokens = Double(rounds.map(\.tokens).reduce(0, +)) / Double(rounds.count)
        let milliseconds =
            rounds.map(\.seconds).reduce(0, +) / Double(rounds.count) * Self.millisecondsPerSecond
        let tolerance = 0.01

        #expect(abs(tokens - point.tokensPerRound) < tolerance)
        #expect(abs(milliseconds - point.roundMilliseconds) < tolerance)
        #expect(rounds.allSatisfy { (1 ... Self.blockSize).contains($0.tokens) })
        #expect(point.tokensPerRound / point.roundMilliseconds > 1 / point.plainMilliseconds)
    }

    @Test(
        "rounds whose measured rate beats plain decode keep the block size and never pause",
        arguments: points)
    func measuredWinningRoundsNeverBackOff(point: Point) {
        let controller = replay(point)

        #expect(controller.blockSizeChangeCount == 0)
        #expect(controller.pauseCount == 0)
        #expect(controller.blockSize == Self.blockSize)
    }
}
