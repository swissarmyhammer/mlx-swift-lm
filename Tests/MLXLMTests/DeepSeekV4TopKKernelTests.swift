// Copyright © 2026 Apple Inc.
//
// The contract of ``DeepSeekV4TopKKernel``.
//
// The indexer picks its chunks through this kernel and not through
// `argPartition`. The kernel must pick the same set as `argPartition` when no
// two scores tie, it must give the picks in ascending index order, and it
// must rank a hidden chunk (`-1e30` or `-inf`) and a NaN score as
// `argPartition` ranks them.

import Foundation
import MLX
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4TopKKernelTests {

    /// One shape of the set-parity test.
    struct ParityCase: CustomTestStringConvertible, Sendable {
        /// The number of rows.
        let rows: Int
        /// The number of scores in one row.
        let width: Int
        /// The number of picks for each row.
        let count: Int

        /// The text the test report shows for this case.
        var testDescription: String { "\(rows) rows of \(width), \(count) picks" }
    }

    /// The shapes of the set-parity test: the indexer budget over a 32k
    /// pool, a width that is not a multiple of the threadgroup width, one
    /// score more than the budget, and a small row with a small budget.
    private static let parityCases = [
        ParityCase(rows: 3, width: 8192, count: 512),
        ParityCase(rows: 2, width: 1000, count: 512),
        ParityCase(rows: 2, width: 513, count: 512),
        ParityCase(rows: 4, width: 37, count: 5),
    ]

    /// The seed of the random order of the scores of the parity test.
    private static let paritySeed: UInt64 = 20_261_014

    /// The step between two neighbor scores of the parity test. It is not a
    /// power of two, thus the scores fill more than one exponent and the
    /// low bits of the float32 keys.
    private static let scoreStep: Float = 0.37

    /// Builds rows of distinct scores in a random order.
    ///
    /// Each row holds a random order of the numbers `0 ..< width`, shifted
    /// down by half the width and scaled by ``scoreStep``. Thus no two scores
    /// of one row tie, and a row holds negative and positive scores.
    ///
    /// - Parameter parityCase: The shape of the rows.
    /// - Returns: The scores, shape `(rows, width)`, in float32.
    private static func distinctScores(_ parityCase: ParityCase) -> MLXArray {
        let noise = MLXRandom.uniform(
            0 ..< 1, [parityCase.rows, parityCase.width], key: MLXRandom.key(paritySeed))
        let order = argSort(noise, axis: -1).asType(.float32)
        return (order - Float(parityCase.width) / 2) * scoreStep
    }

    /// The picks of `argPartition` for each row, each row sorted.
    ///
    /// - Parameters:
    ///   - scores: The scores, shape `(rows, width)`.
    ///   - count: The number of picks for each row.
    /// - Returns: The sorted picks, shape `(rows, count)`.
    private static func partitionPicks(_ scores: MLXArray, count: Int) -> MLXArray {
        sorted(argPartition(-scores, kth: count - 1, axis: -1)[.ellipsis, 0 ..< count], axis: -1)
    }

    @Test(arguments: parityCases)
    func theKernelPicksTheSetOfArgPartitionWhenNoScoresTie(parityCase: ParityCase) {
        let scores = Self.distinctScores(parityCase)

        let picks = DeepSeekV4TopKKernel.topIndices(scores, count: parityCase.count)

        #expect(picks.shape == [parityCase.rows, parityCase.count])
        #expect(picks.dtype == .uint32)
        #expect(
            DeepSeekV4TestArrays.integers(sorted(picks, axis: -1))
                == DeepSeekV4TestArrays.integers(
                    Self.partitionPicks(scores, count: parityCase.count)))
    }

    @Test(arguments: parityCases)
    func theKernelGivesThePicksInAscendingIndexOrder(parityCase: ParityCase) {
        let picks = DeepSeekV4TopKKernel.topIndices(
            Self.distinctScores(parityCase), count: parityCase.count)

        #expect(
            DeepSeekV4TestArrays.integers(picks)
                == DeepSeekV4TestArrays.integers(sorted(picks, axis: -1)))
    }

    /// The score of a hidden chunk, as the indexer writes it.
    private static let hiddenScore: Float = -1e30

    /// One row with fewer readable scores than the budget: two numbers, two
    /// hidden chunks, one `-inf` and three NaN scores.
    private static let fewReadableScores: [Float] = [
        -.infinity, 2.5, hiddenScore, .nan, -0.75, .nan, hiddenScore, .nan,
    ]

    /// The budget of the row of ``fewReadableScores``: more than its
    /// numbers, and fewer than its numbers and its hidden scores together.
    private static let fewReadableCount = 5

    /// The picks of the row of ``fewReadableScores``: every score that is not
    /// NaN, in ascending index order.
    private static let fewReadablePicks: [Int32] = [0, 1, 2, 4, 6]

    @Test func aNaNScoreRanksBelowAHiddenAndAMinusInfinityScore() {
        let scores = MLXArray(Self.fewReadableScores).reshaped(1, -1)

        let picks = DeepSeekV4TopKKernel.topIndices(scores, count: Self.fewReadableCount)

        #expect(DeepSeekV4TestArrays.integers(picks) == Self.fewReadablePicks)
        #expect(
            DeepSeekV4TestArrays.integers(picks)
                == DeepSeekV4TestArrays.integers(
                    Self.partitionPicks(scores, count: Self.fewReadableCount)))
    }

    /// One row in which four scores tie at the value the budget cuts.
    private static let tiedScores: [Float] = [0, 5, 1, 5, 5, 2, 5, 0, 0, 0]

    /// The budget of the row of ``tiedScores``: fewer than its tied scores.
    private static let tiedCount = 3

    /// The picks of the row of ``tiedScores``: the three tied scores of the
    /// lowest indices.
    private static let tiedPicks: [Int32] = [1, 3, 4]

    @Test func tiedScoresAtTheCutGiveTheLowestIndices() {
        let picks = DeepSeekV4TopKKernel.topIndices(
            MLXArray(Self.tiedScores).reshaped(1, -1), count: Self.tiedCount)

        #expect(DeepSeekV4TestArrays.integers(picks) == Self.tiedPicks)
    }

    /// The width of the wide row of tied scores. It is more than the
    /// threadgroup width of the kernel, thus one thread reads more than one
    /// tied score of the row.
    private static let wideTieWidth = 4096

    /// The indices of the wide row that hold a score above the tie. The
    /// first two stand in the run of the first thread, between its tied
    /// scores.
    private static let wideTieHighIndices = [1, 3, 4000]

    /// The budget of the wide row: the three high scores and one tied score.
    private static let wideTieCount = 4

    /// The picks of the wide row: the tied score of the lowest index and the
    /// three high scores.
    private static let wideTiePicks: [Int32] = [0, 1, 3, 4000]

    @Test func tiedScoresInTheRunOfOneThreadGiveTheLowestIndices() {
        var row = [Float](repeating: 0, count: Self.wideTieWidth)
        for index in Self.wideTieHighIndices {
            row[index] = 1
        }

        let picks = DeepSeekV4TopKKernel.topIndices(
            MLXArray(row).reshaped(1, -1), count: Self.wideTieCount)

        #expect(DeepSeekV4TestArrays.integers(picks) == Self.wideTiePicks)
    }

    /// One row that holds a negative zero at a lower index than a positive
    /// zero. The bits of the positive zero give the larger key, thus a
    /// kernel that does not fold the two zeros together picks the positive
    /// zero.
    private static let signedZeroScores: [Float] = [-1, -0.0, 0, -2]

    /// The picks of the row of ``signedZeroScores``: the two zeros tie, as
    /// `argPartition` ranks them, thus the lower index wins.
    private static let signedZeroPicks: [Int32] = [1]

    @Test func aNegativeZeroTiesWithAPositiveZero() {
        let picks = DeepSeekV4TopKKernel.topIndices(
            MLXArray(Self.signedZeroScores).reshaped(1, -1), count: Self.signedZeroPicks.count)

        #expect(DeepSeekV4TestArrays.integers(picks) == Self.signedZeroPicks)
    }
}
