// Copyright © 2026 Apple Inc.
//
// The near-tie rule of ``AttentionKernelTolerance``: a verify row must pick
// plain decode's greedy token, or, when the two best logits of plain decode
// stand in a near tie, the other one of those two. Any other token fails.

import MLX
import Testing

struct AttentionKernelToleranceTests {

    /// The logits of plain decode over one row: tokens 0 and 1 stand in a
    /// near tie, token 1 first, and tokens 2 and 3 stand far below them.
    private static let nearTieDecodeRow: [Float] = [1.0, 1.0005, 0.0, 0.5]

    /// The second-best token of ``nearTieDecodeRow``.
    private static let secondBestToken = 0

    /// A token of ``nearTieDecodeRow`` outside its two best ones.
    private static let thirdBestToken = 3

    /// The logits of one row, shape `(1, 1, vocabulary)`.
    private static func logits(_ row: [Float]) -> MLXArray {
        MLXArray(row, [1, 1, row.count])
    }

    /// The logits of one row whose greedy token is `token`, over the
    /// vocabulary of ``nearTieDecodeRow``.
    private static func logitsPicking(_ token: Int) -> MLXArray {
        var row = [Float](repeating: 0, count: nearTieDecodeRow.count)
        row[token] = 1
        return logits(row)
    }

    /// Under a near tie, a verify row that picks the second-best token of
    /// plain decode passes.
    @Test func aNearTieAcceptsTheSecondBestToken() {
        AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie(
            verify: Self.logitsPicking(Self.secondBestToken),
            decode: Self.logits(Self.nearTieDecodeRow))
    }

    /// Under a near tie, a verify row that picks a token outside the two
    /// best tokens of plain decode fails.
    @Test func aNearTieRejectsATokenOutsideTheTwoBest() {
        withKnownIssue("a token outside decode's two best must fail the check") {
            AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie(
                verify: Self.logitsPicking(Self.thirdBestToken),
                decode: Self.logits(Self.nearTieDecodeRow))
        }
    }
}
