import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// A write of several rows into a ring that keeps a prefix attends as writes of one row do.
///
/// A ring built from `maxKVSize` keeps its first tokens. A write of one row keeps them in the
/// ring, so its query sees the kept prefix and the newest rows of the window. A write of several
/// rows (a chunked prefill, a speculative verify) must give each of its queries that same view,
/// or the tokens it yields differ from those of plain decode.
@Suite
struct RotatingKVCachePrefixMaskTests {

    /// The attention window of each ring.
    private static let window = 8

    /// The rows each ring keeps at its head.
    private static let keep = 2

    /// The attention heads of each write.
    private static let heads = 2

    /// The width of each attention head.
    private static let headDimension = 4

    /// One write of several rows after `history` rows written one at a time.
    struct Scenario: CustomTestStringConvertible, Sendable {
        /// The rows written one at a time before the write.
        let history: Int
        /// The rows of the write.
        let queries: Int
        /// The rewind reserve of the ring.
        let rewindCapacity: Int

        var testDescription: String {
            "history \(history), queries \(queries), rewind reserve \(rewindCapacity)"
        }
    }

    /// The writes under test: a prefill longer than the window, a write that crosses the window,
    /// and verifies past the wrap, with and without a rewind reserve.
    static let scenarios = [
        Scenario(history: 0, queries: 20, rewindCapacity: 0),
        Scenario(history: 6, queries: 5, rewindCapacity: 0),
        Scenario(history: 20, queries: 5, rewindCapacity: 0),
        Scenario(history: 20, queries: 5, rewindCapacity: 10),
    ]

    /// Writes `range` of `keys` and `values` into `cache` one row at a time, and attends each
    /// query of `range` through the presentation and the mask of its own write.
    private func attendOneRowAtATime(
        _ cache: RotatingKVCache, queries: MLXArray, keys: MLXArray, values: MLXArray,
        range: Range<Int>
    ) -> [MLXArray] {
        range.map { row in
            let mask = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
            let (cachedKeys, cachedValues) = cache.update(
                keys: keys[0..., 0..., row ..< (row + 1), 0...],
                values: values[0..., 0..., row ..< (row + 1), 0...])
            return MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., row ..< (row + 1), 0...], keys: cachedKeys,
                values: cachedValues, scale: 1, mask: mask)
        }
    }

    @Test(
        "each query of a write of several rows into a ring that keeps a prefix attends as a write of one row",
        arguments: scenarios)
    func multiRowWriteAttendsAsSingleRowWrites(scenario: Scenario) {
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let total = scenario.history + scenario.queries
            let shape = [1, Self.heads, total, Self.headDimension]
            let keys = MLXRandom.normal(shape)
            let values = MLXRandom.normal(shape)
            let queries = MLXRandom.normal(shape)
            let history = 0 ..< scenario.history
            let written = scenario.history ..< total

            let reference = RotatingKVCache(
                maxSize: Self.window, keep: Self.keep, rewindCapacity: scenario.rewindCapacity)
            _ = attendOneRowAtATime(
                reference, queries: queries, keys: keys, values: values, range: history)
            let expected = concatenated(
                attendOneRowAtATime(
                    reference, queries: queries, keys: keys, values: values, range: written),
                axis: 2)

            let cache = RotatingKVCache(
                maxSize: Self.window, keep: Self.keep, rewindCapacity: scenario.rewindCapacity)
            _ = attendOneRowAtATime(
                cache, queries: queries, keys: keys, values: values, range: history)
            let mask = cache.makeMask(n: scenario.queries, windowSize: nil, returnArray: false)
            let (cachedKeys, cachedValues) = cache.update(
                keys: keys[0..., 0..., written, 0...], values: values[0..., 0..., written, 0...])
            let actual = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., written, 0...], keys: cachedKeys,
                values: cachedValues, scale: 1, mask: mask)

            #expect(actual.shape == expected.shape)
            #expect(
                allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self),
                "the write of several rows diverged from writes of one row")
        }
    }
}
