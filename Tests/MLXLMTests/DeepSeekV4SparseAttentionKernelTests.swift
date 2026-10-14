// Copyright © 2026 Apple Inc.
//
// The contract of ``DeepSeekV4SparseAttentionKernel``.
//
// On the GPU each DeepSeek-V4 prefill block reads its keys through this
// kernel and not through the op path: a masked SDPA over the window keys
// joined to the pooled chunks. The kernel reads only the keys each query may
// see and adds them with an online softmax, thus its sums run in another
// order, and the two paths are not bit-exact. The kernel output must stay
// inside a small tolerance of the masked SDPA of MLX.
//
// The visibility rules are stated again here in plain Swift: the sliding
// window of `createCausalMask`, and the block-causal rule of a pooled chunk.
// Thus a wrong rule in the module cannot make the test agree with it.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4SparseAttentionKernelTests {

    /// The pooled rows one case reads beside the window.
    enum PoolKind: Sendable {
        /// No pool: a layer with no compressor.
        case none
        /// Every chunk that stands wholly behind the query: a layer with no
        /// indexer, or a pool of `index_topk` chunks or fewer.
        case everyVisibleChunk
        /// The given number of chunks for each query, picked at random: a
        /// layer whose indexer ranks a pool of more than `index_topk` chunks.
        case picked(Int)
    }

    /// One shape of the tolerance test.
    struct AttentionCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of query heads.
        let heads: Int
        /// The width of one head.
        let headDim: Int
        /// The number of query tokens in the block.
        let tokens: Int
        /// The sliding window: the most recent keys one query reads.
        let window: Int
        /// The window keys that stand ahead of the block: the rows of the
        /// earlier blocks that the cache gives back.
        let earlierKeys: Int
        /// The absolute position of the first token of the block.
        let offset: Int
        /// The number of raw positions one pooled chunk covers.
        let chunkWidth: Int
        /// The pooled rows the block reads.
        let pool: PoolKind
        /// True when the softmax reads a learned sink for each head.
        let hasSinks: Bool

        /// The number of pooled chunks: each chunk that ends at or before
        /// the last token of the block.
        var chunks: Int { (offset + tokens) / chunkWidth }

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(heads) heads of \(headDim), \(tokens) tokens at \(offset) after "
                + "\(earlierKeys) keys, window \(window), \(chunks) chunks of \(chunkWidth), "
                + "\(pool), sinks \(hasSinks)"
        }
    }

    /// The shapes of the tolerance test.
    ///
    /// 1. The first block of a ranked layer. Query 0 sees no chunk, the
    ///    early queries get picks they may not read, and the later queries
    ///    see only part of the block through the window. The 20 heads of 80
    ///    numbers are no whole tile of the kernel, thus it pads both.
    /// 2. A later block of a ranked layer at the published head shape. Its
    ///    first query reads window rows of the earlier block, and each query
    ///    reads more keys than one key tile of the kernel holds.
    /// 3. A later block of a layer that reads every visible chunk. The last
    ///    chunk stands behind only the last query.
    /// 4. A layer with no compressor and no sink. The two heads give a sink
    ///    input of fewer than 8 values, which MLX gives in the `constant`
    ///    address space.
    /// 5. A decode step of a ranked layer: one query over a full window.
    /// 6. A decode step of a small layer that reads every visible chunk.
    static let attentionCases = [
        AttentionCase(
            batch: 2, heads: 20, headDim: 80, tokens: 9, window: 4, earlierKeys: 0, offset: 0,
            chunkWidth: 2, pool: .picked(3), hasSinks: true),
        AttentionCase(
            batch: 1, heads: 64, headDim: 512, tokens: 6, window: 16, earlierKeys: 15, offset: 200,
            chunkWidth: 4, pool: .picked(40), hasSinks: true),
        AttentionCase(
            batch: 1, heads: 64, headDim: 512, tokens: 5, window: 16, earlierKeys: 15, offset: 635,
            chunkWidth: 128, pool: .everyVisibleChunk, hasSinks: true),
        AttentionCase(
            batch: 2, heads: 2, headDim: 16, tokens: 7, window: 4, earlierKeys: 3, offset: 3,
            chunkWidth: 1, pool: .none, hasSinks: false),
        AttentionCase(
            batch: 1, heads: 64, headDim: 512, tokens: 1, window: 16, earlierKeys: 15, offset: 300,
            chunkWidth: 4, pool: .picked(20), hasSinks: true),
        AttentionCase(
            batch: 1, heads: 2, headDim: 16, tokens: 1, window: 4, earlierKeys: 3, offset: 40,
            chunkWidth: 4, pool: .everyVisibleChunk, hasSinks: true),
    ]

    /// The cases whose masks hide keys: each prefill block. A decode step
    /// sees its whole window and each chunk it reads.
    static let maskedCases = Array(attentionCases.prefix(4))

    /// The head count and the head width of the published checkpoint.
    private enum Published {
        static let heads = 64
        static let headDim = 512
        /// The tokens of one prefill step of the bench.
        static let prefillTokens = 2048
        /// The tokens of the shortest block whose 16-head threadgroups fill
        /// the GPU: 4 threadgroups for each token.
        static let shortestFusedTokens = 64
    }

    /// A head width that is not a whole number of column pairs.
    private static let unpairedHeadDim = 511

    /// The queries of one call of the published head shape.
    private static func publishedQueries(
        tokens: Int, headDim: Int = Published.headDim
    ) -> MLXArray {
        MLXArray.zeros([1, Published.heads, tokens, headDim], dtype: .bfloat16)
    }

    /// The kernel reads a block whose threadgroups fill the GPU. A decode
    /// step, and any block shorter than the shortest such block, keep the
    /// op path, as does a head width the paired loads cannot read.
    @Test func theKernelReadsOnlyBlocksThatFillTheGPU() {
        #expect(
            DeepSeekV4SparseAttentionKernel.canRun(
                queries: Self.publishedQueries(tokens: Published.prefillTokens)))
        #expect(
            DeepSeekV4SparseAttentionKernel.canRun(
                queries: Self.publishedQueries(tokens: Published.shortestFusedTokens)))
        #expect(
            !DeepSeekV4SparseAttentionKernel.canRun(
                queries: Self.publishedQueries(tokens: Published.shortestFusedTokens - 1)))
        #expect(!DeepSeekV4SparseAttentionKernel.canRun(queries: Self.publishedQueries(tokens: 1)))
        #expect(
            !DeepSeekV4SparseAttentionKernel.canRun(
                queries: Self.publishedQueries(
                    tokens: Published.prefillTokens, headDim: Self.unpairedHeadDim)))
    }

    /// The dtypes of the tolerance test: the float32 of the synthetic tests
    /// and the bfloat16 of the published checkpoint.
    static let dtypes: [DType] = [.float32, .bfloat16]

    /// The largest gap allowed between the kernel and the op path, absolute
    /// or relative, whichever is larger, for float32 inputs. Both paths add
    /// in float32, each in its own order.
    private static let float32Tolerance: Float = 1e-4

    /// The largest gap for bfloat16 inputs. The op path reads the same
    /// bfloat16 inputs in float32. The kernel adds in float32 and rounds its
    /// output to bfloat16, whose step is 2^-8 of the value.
    private static let bfloat16Tolerance: Float = 1e-2

    /// The seed of each random operand.
    private enum Seed {
        static let queries: UInt64 = 20_261_014
        static let windowKeys: UInt64 = 20_261_015
        static let pool: UInt64 = 20_261_016
        static let sinks: UInt64 = 20_261_017
        static let picks: UInt64 = 20_261_018
    }

    /// The axis a `(batch, heads, tokens, width)` tensor holds its heads on.
    private static let headAxis = 1

    /// The axis a `(batch, heads, keys, width)` tensor holds its keys on.
    private static let keyAxis = 2

    /// The last axis of a tensor.
    private static let lastAxis = -1

    /// The operands of one attention call.
    struct Operands {
        /// The rotated queries, shape `(batch, heads, tokens, headDim)`.
        let queries: MLXArray
        /// The window keys, which are also the values, shape
        /// `(batch, 1, earlierKeys + tokens, headDim)`.
        let windowKeys: MLXArray
        /// The pooled chunks, shape `(batch, chunks, headDim)`.
        let pool: MLXArray
        /// The chunks each query picked, shape `(batch, tokens, picks)`, or
        /// `nil` when the case picks none.
        let picked: MLXArray?
        /// The learned sink of each head, or `nil`.
        let sinks: MLXArray?
        /// The factor the scores take before the softmax.
        let scale: Float
    }

    /// Builds repeatable random operands for one case in one dtype.
    ///
    /// The picks of each query are distinct, as a top-k gives them, and they
    /// stand in random order.
    static func operands(_ attentionCase: AttentionCase, dtype: DType) -> Operands {
        let windowLength = attentionCase.earlierKeys + attentionCase.tokens
        var picked: MLXArray?
        if case .picked(let count) = attentionCase.pool {
            let order = argSort(
                MLXRandom.uniform(
                    0 ..< 1,
                    [attentionCase.batch, attentionCase.tokens, attentionCase.chunks],
                    key: MLXRandom.key(Seed.picks)),
                axis: lastAxis)
            picked = order[.ellipsis, 0 ..< count].asType(.int32)
        }
        return Operands(
            queries: MLXRandom.normal(
                [
                    attentionCase.batch, attentionCase.heads, attentionCase.tokens,
                    attentionCase.headDim,
                ],
                key: MLXRandom.key(Seed.queries)
            ).asType(dtype),
            windowKeys: MLXRandom.normal(
                [attentionCase.batch, 1, windowLength, attentionCase.headDim],
                key: MLXRandom.key(Seed.windowKeys)
            ).asType(dtype),
            pool: MLXRandom.normal(
                [attentionCase.batch, attentionCase.chunks, attentionCase.headDim],
                key: MLXRandom.key(Seed.pool)
            ).asType(dtype),
            picked: picked,
            sinks: attentionCase.hasSinks
                ? MLXRandom.normal([attentionCase.heads], key: MLXRandom.key(Seed.sinks))
                    .asType(dtype) : nil,
            scale: 1 / Float(attentionCase.headDim).squareRoot())
    }

    /// The output of the kernel.
    static func kernelOutput(_ attentionCase: AttentionCase, _ operands: Operands) -> MLXArray {
        var pooled: DeepSeekV4SparseAttentionKernel.PooledRows?
        switch attentionCase.pool {
        case .none:
            pooled = nil
        case .everyVisibleChunk, .picked:
            pooled = .init(
                pool: operands.pool, picked: operands.picked, offset: attentionCase.offset,
                chunkWidth: attentionCase.chunkWidth)
        }
        return DeepSeekV4SparseAttentionKernel.attention(
            queries: operands.queries,
            window: .init(
                keys: operands.windowKeys, values: operands.windowKeys,
                window: attentionCase.window),
            pooled: pooled, scale: operands.scale, sinks: operands.sinks)
    }

    /// Tells if a query may read a pooled chunk: the whole chunk must stand
    /// behind the query.
    private static func chunkIsVisible(_ chunk: Int, position: Int, chunkWidth: Int) -> Bool {
        (chunk + 1) * chunkWidth <= position + 1
    }

    /// Tells if a query may read a window key: the key stands at or before
    /// the query, and inside its window. Key `j` of the window stands at row
    /// `j` of `earlierKeys + tokens` rows, and query `t` at row
    /// `earlierKeys + t`.
    private static func windowKeyIsVisible(
        _ key: Int, query: Int, _ attentionCase: AttentionCase
    ) -> Bool {
        let row = attentionCase.earlierKeys + query
        return key <= row && key > row - attentionCase.window
    }

    /// The visibility of each joined key `[window | chunks]` for each query,
    /// shape `(batch, 1, tokens, windowKeys + chunks)`.
    ///
    /// - Parameters:
    ///   - attentionCase: The case.
    ///   - picked: The picks of the case, read back, or `nil`.
    ///   - revealed: True to show every chunk and every window key of the
    ///     block, which the positive control reads.
    static func joinedVisibility(
        _ attentionCase: AttentionCase, picked: [Int32]?, revealed: Bool
    ) -> MLXArray {
        let windowLength = attentionCase.earlierKeys + attentionCase.tokens
        let chunkCount = attentionCase.pool.readsChunks ? attentionCase.chunks : 0
        let flags = (0 ..< attentionCase.batch).flatMap { sequence in
            (0 ..< attentionCase.tokens).flatMap { query in
                let windowFlags = (0 ..< windowLength).map {
                    revealed || windowKeyIsVisible($0, query: query, attentionCase)
                }
                let chunkFlags = (0 ..< chunkCount).map { chunk in
                    revealed
                        || chunkIsReadable(
                            chunk, sequence: sequence, query: query, attentionCase, picked: picked)
                }
                return windowFlags + chunkFlags
            }
        }
        return MLXArray(
            flags, [attentionCase.batch, 1, attentionCase.tokens, windowLength + chunkCount])
    }

    /// Tells if one query reads one chunk: the chunk stands behind it, and a
    /// ranked case picked it.
    private static func chunkIsReadable(
        _ chunk: Int, sequence: Int, query: Int, _ attentionCase: AttentionCase, picked: [Int32]?
    ) -> Bool {
        let visible = chunkIsVisible(
            chunk, position: attentionCase.offset + query, chunkWidth: attentionCase.chunkWidth)
        guard case .picked(let count) = attentionCase.pool, let picked else {
            return visible
        }
        let first = (sequence * attentionCase.tokens + query) * count
        return visible && picked[first ..< first + count].contains(Int32(chunk))
    }

    /// The output of the op path: one masked SDPA over the window keys
    /// joined to every pooled chunk, in float32.
    static func opPathOutput(
        _ attentionCase: AttentionCase, _ operands: Operands, revealed: Bool = false
    ) -> MLXArray {
        let picked = operands.picked.map(DeepSeekV4TestArrays.integers)
        let windowKeys = operands.windowKeys.asType(.float32)
        let keys =
            attentionCase.pool.readsChunks
            ? concatenated(
                [windowKeys, operands.pool.asType(.float32).expandedDimensions(axis: headAxis)],
                axis: keyAxis)
            : windowKeys
        return MLXFast.scaledDotProductAttention(
            queries: operands.queries.asType(.float32), keys: keys, values: keys,
            scale: operands.scale,
            mask: .array(joinedVisibility(attentionCase, picked: picked, revealed: revealed)),
            sinks: operands.sinks?.asType(.float32))
    }

    /// The tolerance of one dtype.
    static func tolerance(_ dtype: DType) -> Float {
        dtype == .float32 ? float32Tolerance : bfloat16Tolerance
    }

    /// The largest gap between two lists, each gap measured against the
    /// tolerance: absolute below 1, relative above.
    static func worstGap(_ got: [Float], _ expected: [Float]) -> Float {
        zip(got, expected).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 0
    }

    @Test(arguments: attentionCases, dtypes)
    func theKernelStaysInsideTheToleranceOfTheOpPath(attentionCase: AttentionCase, dtype: DType) {
        let operands = Self.operands(attentionCase, dtype: dtype)

        let kernel = Self.kernelOutput(attentionCase, operands)
        let reference = DeepSeekV4TestArrays.floats(Self.opPathOutput(attentionCase, operands))
        let gap = Self.worstGap(DeepSeekV4TestArrays.floats(kernel), reference)

        #expect(kernel.shape == operands.queries.shape)
        #expect(kernel.dtype == dtype)
        #expect(reference.contains { $0 != 0 }, "every op-path value was zero")
        #expect(gap <= Self.tolerance(dtype), "worst gap \(gap)")
    }

    /// The positive control of the tolerance test: a kernel that read the
    /// keys its masks hide would break the tolerance.
    @Test(arguments: maskedCases)
    func theKeysTheMasksHideMoveTheOutputPastTheTolerance(attentionCase: AttentionCase) {
        let operands = Self.operands(attentionCase, dtype: .float32)

        let hidden = DeepSeekV4TestArrays.floats(Self.opPathOutput(attentionCase, operands))
        let revealed = DeepSeekV4TestArrays.floats(
            Self.opPathOutput(attentionCase, operands, revealed: true))

        #expect(Self.worstGap(revealed, hidden) > Self.bfloat16Tolerance)
    }

    /// The number of chunks that stand wholly behind one query of a case.
    private static func visibleChunkCount(query: Int, _ attentionCase: AttentionCase) -> Int {
        (0 ..< attentionCase.chunks).filter {
            chunkIsVisible(
                $0, position: attentionCase.offset + query, chunkWidth: attentionCase.chunkWidth)
        }.count
    }

    /// The first ranked case holds what its description promises: picks
    /// that the kernel must hide, and a first query that sees no chunk. The
    /// second query sees a chunk, which shows the count can see one.
    @Test func theFirstRankedBlockHoldsHiddenPicksAndAQueryThatSeesNoChunk() throws {
        let attentionCase = Self.attentionCases[0]
        let operands = Self.operands(attentionCase, dtype: .float32)
        let picked = DeepSeekV4TestArrays.integers(try #require(operands.picked))
        let count = try #require(attentionCase.pool.pickCount)

        let hiddenPicks = picked.enumerated().filter { index, chunk in
            let query = (index / count) % attentionCase.tokens
            return !Self.chunkIsVisible(
                Int(chunk), position: attentionCase.offset + query,
                chunkWidth: attentionCase.chunkWidth)
        }.count

        #expect(hiddenPicks > 0)
        #expect(Self.visibleChunkCount(query: 0, attentionCase) == 0)
        #expect(Self.visibleChunkCount(query: 1, attentionCase) > 0)
    }
}

extension DeepSeekV4SparseAttentionKernelTests.PoolKind {
    /// True when the case reads pooled chunks at all.
    var readsChunks: Bool {
        guard case .none = self else { return true }
        return false
    }

    /// The number of chunks each query picks, or `nil` when the case picks
    /// none.
    var pickCount: Int? {
        guard case .picked(let count) = self else { return nil }
        return count
    }
}
