// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import MLXLMCommon
import Testing

/// The near-tie rule of ``MTPChatTokenMatch`` when the reported candidates rank the two
/// diverging tokens in either order: plain's candidates may come from a replay, not from the
/// decode that wrote plain's token, so either token may stand first at a tie.
struct MTPChatTokenTieTests {

    private static let plainToken = 58614
    private static let mtpToken = 16570
    private static let unreportedToken = 40
    private static let bestLogProbability: Float = -1
    private static let oneBF16Step: Float = 0.125

    /// The single-token turns of plain and MTP at a step where plain's reported candidates
    /// rank `first` above `second` by `gap`.
    private static func match(first: Int, second: Int, gap: Float, mtp: Int = mtpToken)
        -> MTPChatTokenMatch
    {
        let candidates = [
            GenerateTokenLogProbability(token: first, logProbability: bestLogProbability),
            GenerateTokenLogProbability(token: second, logProbability: bestLogProbability - gap),
        ]
        return MTPChatTokenMatch(
            mtp: [MTPChatToken(token: mtp)],
            plain: [MTPChatToken(token: plainToken, candidates: candidates)])
    }

    @Test func anExactTieRankingPlainsTokenFirstIsANearTie() {
        let match = Self.match(first: Self.plainToken, second: Self.mtpToken, gap: 0)
        #expect(match == .nearTie(step: 0, gap: 0))
    }

    @Test func anExactTieRankingMTPsTokenFirstIsANearTie() {
        let match = Self.match(first: Self.mtpToken, second: Self.plainToken, gap: 0)
        #expect(match == .nearTie(step: 0, gap: 0))
    }

    @Test func aNearTieRankingMTPsTokenFirstIsANearTie() {
        let match = Self.match(first: Self.mtpToken, second: Self.plainToken, gap: Self.oneBF16Step)
        #expect(match == .nearTie(step: 0, gap: Self.oneBF16Step))
    }

    @Test func aGapWiderThanTheToleranceRankingMTPsTokenFirstDiverges() {
        let gap = MTPChatTokenMatch.nearTieLogitTolerance + Self.oneBF16Step
        let match = Self.match(first: Self.mtpToken, second: Self.plainToken, gap: gap)
        #expect(match == .diverged(step: 0, gap: gap))
    }

    @Test func aTokenOutsideTheReportedCandidatesDivergesWithAnUnknownGap() {
        let match = Self.match(
            first: Self.plainToken, second: Self.mtpToken, gap: 0, mtp: Self.unreportedToken)
        #expect(match == .diverged(step: 0, gap: nil))
    }
}
