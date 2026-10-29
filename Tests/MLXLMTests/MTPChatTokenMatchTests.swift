// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import MLXLMCommon
import Testing

/// The near-tie tolerance the MTP chat benchmark compares MTP tokens with.
struct MTPChatTokenMatchTests {

    /// A plain token whose two best candidates are `token` and `runnerUp`, with
    /// `gap` between their log probabilities.
    private static func plain(_ token: Int, runnerUp: Int, gap: Float) -> MTPChatToken {
        MTPChatToken(
            token: token,
            candidates: [
                GenerateTokenLogProbability(token: token, logProbability: -1),
                GenerateTokenLogProbability(token: runnerUp, logProbability: -1 - gap),
            ])
    }

    private static let firstToken = 10
    private static let plainChoice = 20
    private static let runnerUp = 30
    private static let otherToken = 40
    private static let oneBF16Step: Float = 0.125

    @Test func equalTokensMatchExactly() {
        let plain = [Self.plain(Self.firstToken, runnerUp: Self.runnerUp, gap: 1)]
        let mtp = [MTPChatToken(token: Self.firstToken)]
        #expect(MTPChatTokenMatch(mtp: mtp, plain: plain) == .exact)
    }

    @Test func takingPlainsRunnerUpAtANearTieHolds() {
        let plain = [
            Self.plain(Self.firstToken, runnerUp: Self.runnerUp, gap: 1),
            Self.plain(Self.plainChoice, runnerUp: Self.runnerUp, gap: Self.oneBF16Step),
        ]
        let mtp = [MTPChatToken(token: Self.firstToken), MTPChatToken(token: Self.runnerUp)]
        let match = MTPChatTokenMatch(mtp: mtp, plain: plain)
        #expect(match == .nearTie(step: 1, gap: Self.oneBF16Step))
        #expect(match.holds)
    }

    @Test func aGapWiderThanTheToleranceDiverges() {
        let gap = MTPChatTokenMatch.nearTieLogitTolerance + Self.oneBF16Step
        let plain = [Self.plain(Self.plainChoice, runnerUp: Self.runnerUp, gap: gap)]
        let mtp = [MTPChatToken(token: Self.runnerUp)]
        let match = MTPChatTokenMatch(mtp: mtp, plain: plain)
        #expect(match == .diverged(step: 0, gap: gap))
        #expect(!match.holds)
    }

    @Test func aTokenOtherThanTheRunnerUpDiverges() {
        let plain = [Self.plain(Self.plainChoice, runnerUp: Self.runnerUp, gap: Self.oneBF16Step)]
        let mtp = [MTPChatToken(token: Self.otherToken)]
        #expect(!MTPChatTokenMatch(mtp: mtp, plain: plain).holds)
    }

    @Test func aStepWithoutPlainCandidatesDiverges() {
        let plain = [MTPChatToken(token: Self.plainChoice)]
        let mtp = [MTPChatToken(token: Self.runnerUp)]
        #expect(MTPChatTokenMatch(mtp: mtp, plain: plain) == .diverged(step: 0, gap: nil))
    }

    @Test func anMTPTurnThatStopsEarlierDiverges() {
        let plain = [
            Self.plain(Self.firstToken, runnerUp: Self.runnerUp, gap: 1),
            Self.plain(Self.plainChoice, runnerUp: Self.runnerUp, gap: Self.oneBF16Step),
        ]
        let mtp = [MTPChatToken(token: Self.firstToken)]
        #expect(!MTPChatTokenMatch(mtp: mtp, plain: plain).holds)
    }
}
