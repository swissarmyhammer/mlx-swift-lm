// Copyright © 2026 Apple Inc.
//
// The contract of ``DeepSeekV4IndexerScoreKernel``.
//
// On the GPU the indexer scores its pooled chunks through this kernel and
// not through the op path: a matmul for each head, then
// ``DeepSeekV4Indexer/headWeightedSum(_:headWeights:scale:)``. The kernel adds
// the heads inside one dispatch and never writes the per-head scores, thus
// its sums run in another order, and the two paths are not bit-exact. Three
// rules hold:
//
//  1. Each score that its query may read stays inside a small tolerance of
//     the op path.
//  2. Each chunk that reaches past its query takes the masked score exactly.
//  3. The top-k selection of the indexer budget does not change.
//
// The visibility rule is stated again here in plain Swift, thus a wrong rule
// in the module cannot make the test agree with it.

import Foundation
import MLX
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4IndexerScoreKernelTests {

    /// One shape of the tolerance test.
    struct ScoreCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of indexer heads.
        let heads: Int
        /// The width of one indexer head.
        let headDim: Int
        /// The number of query tokens in the block.
        let tokens: Int
        /// The number of pooled chunks.
        let chunks: Int
        /// The absolute position of the first token of the block.
        let offset: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(heads) heads of \(headDim), \(tokens) tokens at \(offset), \(chunks) chunks"
        }
    }

    /// The shapes of the tolerance test.
    ///
    /// The first case is a prefill block of the published indexer: 64 heads
    /// of 128. Its chunk count is not a multiple of the chunk tile of the
    /// kernel, and its first queries may not read the last chunks. The second
    /// case is the synthetic indexer of the other indexer tests: 2 heads of 4
    /// and two sequences, thus the kernel pads the heads and the head width.
    /// The third case is a decode step of the published indexer, with every
    /// chunk behind its token. The fourth case is a decode step of the
    /// synthetic indexer: its head weights hold 2 values, and MLX gives an
    /// input of fewer than 8 values in the `constant` address space.
    private static let scoreCases = [
        ScoreCase(batch: 1, heads: 64, headDim: 128, tokens: 40, chunks: 1100, offset: 4370),
        ScoreCase(batch: 2, heads: 2, headDim: 4, tokens: 9, chunks: 70, offset: 250),
        ScoreCase(batch: 1, heads: 64, headDim: 128, tokens: 1, chunks: 300, offset: 1300),
        ScoreCase(batch: 1, heads: 2, headDim: 4, tokens: 1, chunks: 5, offset: 40),
    ]

    /// The number of raw positions one pooled chunk covers: the compress
    /// ratio of a layer that holds an indexer.
    private static let chunkWidth = 4

    /// The score of a chunk that its query may not read, as the indexer
    /// writes it.
    private static let maskedScore: Float = -1e30

    /// The largest difference between a kernel score and an op-path score,
    /// as a fraction of the largest op-path score of the case.
    ///
    /// The two paths add 128 products and 64 heads in float32, each in its
    /// own order. Each order is off by a few units in the last place, which
    /// is about 1e-7 of the score. The prototype measured 3e-7 at the
    /// published shape.
    private static let relativeTolerance: Float = 1e-5

    /// The seed of the random queries. The keys and the head weights take
    /// the next two seeds.
    private static let operandSeed: UInt64 = 20_261_014

    /// The axis a `(batch, heads, tokens, width)` tensor holds its heads on.
    private static let headAxis = 1

    /// The axis a stack of matrices holds its rows on.
    private static let matrixRowAxis = -2

    /// The last axis of a tensor.
    private static let lastAxis = -1

    /// The operands of one score call, in float32.
    private struct Operands {
        /// The rotated queries, shape `(batch, heads, tokens, headDim)`.
        let queries: MLXArray
        /// The pooled keys, shape `(batch, chunks, headDim)`.
        let keys: MLXArray
        /// The weight of each head for each query, shape
        /// `(batch, tokens, heads)`.
        let headWeights: MLXArray
        /// The factor each per-head score takes.
        let scale: Float
    }

    /// Builds repeatable random operands at the size of the checkpoint
    /// arithmetic: the head weights take `1 / sqrt(heads)` and the scores
    /// take `1 / sqrt(headDim)`, as the indexer applies them.
    ///
    /// - Parameter scoreCase: The shape of the operands.
    /// - Returns: The operands.
    private static func operands(_ scoreCase: ScoreCase) -> Operands {
        Operands(
            queries: MLXRandom.normal(
                [scoreCase.batch, scoreCase.heads, scoreCase.tokens, scoreCase.headDim],
                key: MLXRandom.key(operandSeed)),
            keys: MLXRandom.normal(
                [scoreCase.batch, scoreCase.chunks, scoreCase.headDim],
                key: MLXRandom.key(operandSeed + 1)),
            headWeights: MLXRandom.normal(
                [scoreCase.batch, scoreCase.tokens, scoreCase.heads],
                key: MLXRandom.key(operandSeed + 2))
                / Float(scoreCase.heads).squareRoot(),
            scale: 1 / Float(scoreCase.headDim).squareRoot())
    }

    /// The scores of the kernel.
    ///
    /// - Parameters:
    ///   - operands: The operands.
    ///   - offset: The absolute position of the first token of the block.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`.
    private static func kernelScores(_ operands: Operands, offset: Int) -> MLXArray {
        DeepSeekV4IndexerScoreKernel.scores(
            queries: operands.queries, keys: operands.keys, headWeights: operands.headWeights,
            scale: operands.scale,
            visibility: .init(offset: offset, chunkWidth: chunkWidth, maskedScore: maskedScore))
    }

    /// The scores of the op path, with no mask: one matmul for each head,
    /// then the weighted sum of the indexer.
    ///
    /// - Parameter operands: The operands.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`.
    private static func opPathScores(_ operands: Operands) -> MLXArray {
        let keys = operands.keys.expandedDimensions(axis: headAxis)
            .swappedAxes(lastAxis, matrixRowAxis)
        let headWeights = operands.headWeights.swappedAxes(lastAxis, matrixRowAxis)
            .expandedDimensions(axis: lastAxis)
        return DeepSeekV4Indexer.headWeightedSum(
            operands.queries.matmul(keys), headWeights: headWeights,
            scale: MLXArray(operands.scale))
    }

    /// Tells if a query may read a chunk: the whole chunk must stand behind
    /// the query.
    ///
    /// - Parameters:
    ///   - chunk: The chunk index.
    ///   - position: The absolute position of the query.
    /// - Returns: True when the query may read the chunk.
    private static func isVisible(chunk: Int, position: Int) -> Bool {
        (chunk + 1) * chunkWidth <= position + 1
    }

    @Test(arguments: scoreCases)
    func theKernelScoresStayInsideTheToleranceOfTheOpPath(scoreCase: ScoreCase) {
        let operands = Self.operands(scoreCase)
        let kernel = Self.kernelScores(operands, offset: scoreCase.offset)
        let reference = DeepSeekV4TestArrays.floats(Self.opPathScores(operands))
        let scores = DeepSeekV4TestArrays.floats(kernel)
        let largest = reference.map(abs).max() ?? 0
        var visibleCount = 0
        var worst: Float = 0

        for row in 0 ..< scoreCase.batch * scoreCase.tokens {
            let position = scoreCase.offset + row % scoreCase.tokens
            for chunk in 0 ..< scoreCase.chunks
            where Self.isVisible(chunk: chunk, position: position) {
                let index = row * scoreCase.chunks + chunk
                worst = max(worst, abs(scores[index] - reference[index]))
                visibleCount += 1
            }
        }

        #expect(kernel.shape == [scoreCase.batch, scoreCase.tokens, scoreCase.chunks])
        #expect(kernel.dtype == .float32)
        #expect(visibleCount > 0, "no score was visible, thus the test proved nothing")
        #expect(largest > 0, "every op-path score was zero, thus the test proved nothing")
        #expect(worst <= Self.relativeTolerance * largest, "worst \(worst) of largest \(largest)")
    }

    @Test(arguments: scoreCases)
    func aChunkPastItsQueryTakesTheMaskedScoreExactly(scoreCase: ScoreCase) {
        let scores = DeepSeekV4TestArrays.floats(
            Self.kernelScores(Self.operands(scoreCase), offset: scoreCase.offset))
        var hidden = 0
        var maskedVisible = 0
        var unmaskedHidden = 0

        for row in 0 ..< scoreCase.batch * scoreCase.tokens {
            let position = scoreCase.offset + row % scoreCase.tokens
            for chunk in 0 ..< scoreCase.chunks {
                let isMasked = scores[row * scoreCase.chunks + chunk] == Self.maskedScore
                if Self.isVisible(chunk: chunk, position: position) {
                    maskedVisible += isMasked ? 1 : 0
                } else {
                    hidden += 1
                    unmaskedHidden += isMasked ? 0 : 1
                }
            }
        }

        #expect(maskedVisible == 0)
        #expect(unmaskedHidden == 0, "of \(hidden) hidden chunks")
    }

    /// The shape of the selection test: a prefill block of the published
    /// indexer over a pool of four times the budget, every chunk behind each
    /// query.
    private static let selectionCase = ScoreCase(
        batch: 1, heads: 64, headDim: 128, tokens: 64, chunks: 2048, offset: 8192)

    /// The number of chunks the published indexer keeps for each query, from
    /// `index_topk`.
    private static let publishedTopK = 512

    /// The number of query rows whose selected set differs between two score
    /// tensors.
    ///
    /// - Parameters:
    ///   - first: Scores, shape `(batch, tokens, chunks)`.
    ///   - second: Scores of the same shape.
    /// - Returns: The number of rows whose top ``publishedTopK`` set differs.
    private static func changedSelections(_ first: MLXArray, _ second: MLXArray) -> Int {
        let firstPicks = DeepSeekV4TopKKernel.topIndices(first, count: publishedTopK)
        let secondPicks = DeepSeekV4TopKKernel.topIndices(second, count: publishedTopK)
        return MLX.any(firstPicks .!= secondPicks, axis: lastAxis).asType(.int32).sum()
            .item(Int.self)
    }

    /// The positive control of the selection test: the counter must see the
    /// one row whose best chunk falls to the masked score.
    @Test func theCounterSeesOneChangedSelection() {
        let reference = Self.opPathScores(Self.operands(Self.selectionCase))
        let chunks = Self.selectionCase.chunks
        let tokens = Self.selectionCase.tokens
        let best = argMax(reference, axis: Self.lastAxis, keepDims: true).asType(.int32)
        let chunkIndices = MLXArray(Int32(0) ..< Int32(chunks)).reshaped(1, 1, chunks)
        let isFirstRow = MLXArray(Int32(0) ..< Int32(tokens)).reshaped(1, tokens, 1) .== 0
        let lowered = MLX.where(
            (chunkIndices .== best) .&& isFirstRow, MLXArray(Self.maskedScore), reference)

        #expect(Self.changedSelections(reference, lowered) == 1)
    }

    @Test func theKernelChangesNoSelectionOfThePublishedBudget() {
        let operands = Self.operands(Self.selectionCase)

        let changed = Self.changedSelections(
            Self.kernelScores(operands, offset: Self.selectionCase.offset),
            Self.opPathScores(operands))

        #expect(changed == 0, "\(changed) of \(Self.selectionCase.tokens) rows changed")
    }

    /// The width of the published indexer heads.
    private static let publishedHeadDim = 128

    /// A head width that is not a whole number of four-float vectors.
    private static let unalignedHeadDim = 6

    @Test func theKernelTakesOnlyAHeadWidthOfWholeVectors() {
        #expect(DeepSeekV4IndexerScoreKernel.canRun(headDim: Self.publishedHeadDim))
        #expect(!DeepSeekV4IndexerScoreKernel.canRun(headDim: Self.unalignedHeadDim))
    }
}
