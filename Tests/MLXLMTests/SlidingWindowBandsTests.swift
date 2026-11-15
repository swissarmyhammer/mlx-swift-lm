// Copyright © 2026 Apple Inc.
//
// The contract of ``SlidingWindowBands``.
//
// A sliding-window layer that reads several queries at once attends with the
// array mask of `createCausalMask(n:offset:windowSize:)` over every key the
// cache presents. The bands give each block of query rows only the keys of
// its own window, so the SDPA of MLX reads fewer scores. The bands must give
// the output of the array mask, with and without attention sinks, inside the
// tolerance of each dtype: the online softmax of each path visits the keys in
// its own tiles.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct SlidingWindowBandsTests {

    /// The sliding window of the cases: a block of four query rows.
    private static let window = 9

    /// The number of query heads of each case.
    private static let queryHeads = 4

    /// The number of key and value heads of each case.
    private static let kvHeads = 2

    /// The width of one head: the head width of Flint.
    private static let headDim = 128

    /// One case of the band tests: the shape of the operands, and the keys the
    /// cache holds ahead of the queries.
    struct BandCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of query heads.
        let queryHeads: Int
        /// The number of query tokens.
        let queries: Int
        /// The number of keys the cache presents ahead of the queries.
        let cachedKeys: Int

        /// The shape of the operands.
        var attentionCase: AttentionKernelCase {
            AttentionKernelCase(
                batch: batch, queryHeads: queryHeads, kvHeads: SlidingWindowBandsTests.kvHeads,
                headDim: SlidingWindowBandsTests.headDim, queries: queries,
                keys: cachedKeys + queries, capacity: cachedKeys + queries)
        }

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(queries) queries after \(cachedKeys) cached keys"
        }
    }

    /// The shapes of the band tests.
    ///
    /// 1. No cached key, and query rows that fill each block.
    /// 2. Fewer cached keys than the window holds, and a last block that the
    ///    query rows do not fill.
    /// 3. A full window of cached keys, as a wrapped rotating cache presents.
    /// 4. Two sequences of 3 query heads for each KV head.
    static let bandCases = [
        BandCase(batch: 1, queryHeads: queryHeads, queries: 16, cachedKeys: 0),
        BandCase(batch: 1, queryHeads: queryHeads, queries: 13, cachedKeys: 3),
        BandCase(batch: 1, queryHeads: queryHeads, queries: 21, cachedKeys: window - 1),
        BandCase(batch: 2, queryHeads: 6, queries: 10, cachedKeys: window - 1),
    ]

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_115_1, keys: 20_261_115_2, values: 20_261_115_3)

    /// The seed of the attention sinks.
    private static let sinkSeed: UInt64 = 20_261_115_4

    /// The attention sinks of `operands`: one per query head, in their dtype.
    private static func sinks(of operands: AttentionKernelOperands) -> MLXArray {
        MLXRandom.normal(
            [operands.queries.dim(AttentionKernelOperands.headAxis)],
            key: MLXRandom.key(sinkSeed)
        ).asType(operands.queries.dtype)
    }

    /// The output of the array mask over `operands`: the attention the bands
    /// replace.
    private static func arrayMaskOutput(
        _ operands: AttentionKernelOperands, sinks: MLXArray?
    ) -> MLXArray {
        let queries = operands.queries.dim(AttentionKernelOperands.rowAxis)
        let keys = operands.keys.dim(AttentionKernelOperands.rowAxis)
        let mask = createCausalMask(n: queries, offset: keys - queries, windowSize: window)
        return MLXFast.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .array(mask), sinks: sinks)
    }

    /// The output of the bands over `operands`.
    private static func bandOutput(
        _ operands: AttentionKernelOperands, sinks: MLXArray?
    ) throws -> MLXArray {
        let bands = try #require(
            SlidingWindowBands(
                queryCount: operands.queries.dim(AttentionKernelOperands.rowAxis),
                keyCount: operands.keys.dim(AttentionKernelOperands.rowAxis), window: window))
        return bands.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, sinks: sinks)
    }

    /// The worst gap of the bands against the array mask over one case.
    private static func worstGap(
        _ bandCase: BandCase, dtype: DType, withSinks: Bool
    ) throws -> Float {
        let operands = random.operands(bandCase.attentionCase, dtype: dtype)
        let sinks = withSinks ? sinks(of: operands) : nil
        return AttentionKernelTolerance.worstGap(
            try bandOutput(operands, sinks: sinks), arrayMaskOutput(operands, sinks: sinks))
    }

    /// The bands give the output of the array mask, with the sinks, in each
    /// dtype.
    @Test(arguments: bandCases, AttentionKernelTolerance.dtypes)
    func theBandsGiveTheArrayMaskOutputWithSinks(bandCase: BandCase, dtype: DType) throws {
        let tolerance = try #require(AttentionKernelTolerance.of(dtype))
        #expect(try Self.worstGap(bandCase, dtype: dtype, withSinks: true) <= tolerance)
    }

    /// The bands give the output of the array mask without sinks.
    @Test(arguments: bandCases)
    func theBandsGiveTheArrayMaskOutputWithoutSinks(bandCase: BandCase) throws {
        #expect(
            try Self.worstGap(bandCase, dtype: .float32, withSinks: false)
                <= AttentionKernelTolerance.float32)
    }

    /// The bands give their output in the shape of the queries.
    @Test(arguments: bandCases)
    func theBandsGiveTheShapeOfTheQueries(bandCase: BandCase) throws {
        let operands = Self.random.operands(bandCase.attentionCase, dtype: .float32)
        #expect(try Self.bandOutput(operands, sinks: nil).shape == operands.queries.shape)
    }

    /// Queries that fit in one block take no bands: one block reads every
    /// key the array mask reads.
    @Test func oneBlockOfQueriesTakesNoBands() {
        let blockSize = SlidingWindowBands.blockSize(window: Self.window)
        #expect(
            SlidingWindowBands(
                queryCount: blockSize, keyCount: blockSize + Self.window, window: Self.window)
                == nil)
    }

    /// A window of one key still takes blocks of one query row.
    @Test func aWindowOfOneKeyTakesBlocksOfOneRow() {
        #expect(SlidingWindowBands.blockSize(window: 1) == 1)
    }
}
