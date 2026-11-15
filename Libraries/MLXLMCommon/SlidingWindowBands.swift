// Copyright © 2026 Apple Inc.
//
// Sliding-window attention of a multi-token write, by bands of query rows.
//
// A sliding-window layer that reads `L` queries over the `T` keys its cache
// presents attends with the array mask of
// `createCausalMask(n: L, offset: T - L, windowSize: W)`. The SDPA of MLX
// skips no key tile under an array mask, thus it scores every query against
// all `T` keys although each query reads only `W` of them. At a Flint prefill
// chunk (`L` 2048, `T` 2560, `W` 513) that is five times the scores the
// window needs.
//
// The bands cut the queries into blocks of `B` rows. The rows of one block
// read only the `B + W - 1` keys that end at its last row, so each block
// takes those keys as a sequence of its own on the batch axis, and one SDPA
// call reads every block under one small mask of `[blocks, 1, B, B + W - 1]`.
// The keys pad with zeros where a block reaches past the first or the last
// key, and the mask hides the padding. The attention sinks join the softmax
// of each query row as they do under the array mask.

import MLX

/// Sliding-window causal attention of a multi-token write, by bands of query
/// rows: each block of query rows attends only to the keys of its own window,
/// in one SDPA call.
///
/// The bands give the attention of the array mask
/// `createCausalMask(n: queryCount, offset: keyCount - queryCount,
/// windowSize: window)` over the keys, with the same sinks. The two paths add
/// the keys of one query in other tiles, so they agree inside the rounding of
/// their dtype, not bit for bit.
package struct SlidingWindowBands {

    /// The fewest blocks that take bands. One block reads every key the
    /// array mask reads, so it gains nothing.
    private static let minimumBlockCount = 2

    /// The number of query rows of one block for a window of `window` keys:
    /// half the window, and at least one row.
    ///
    /// A block of `B` rows scores `B + window - 1` keys for each query, and
    /// each block adds the fixed cost of its own key tiles. Measured on an
    /// Apple M5 at the Flint window of 513 keys, a prefill chunk of 2048
    /// queries over 2560 keys ran fastest at blocks of 256 rows (0.85 ms
    /// against 1.46 ms at 128 rows, 1.00 ms at 512 and 3.42 ms under the
    /// array mask).
    ///
    /// - Parameter window: The number of keys each query reads.
    /// - Returns: The number of query rows of one block.
    package static func blockSize(window: Int) -> Int {
        max(1, window / blocksPerWindow)
    }

    /// The number of blocks of query rows in one window of keys.
    private static let blocksPerWindow = 2

    /// The number of queries of the write.
    package let queryCount: Int
    /// The number of keys the cache presents: the cached keys ahead of the
    /// queries, then the keys of the queries.
    package let keyCount: Int
    /// The number of keys each query reads: itself and the `window - 1` keys
    /// before it.
    package let window: Int
    /// The number of query rows of one block.
    package let blockSize: Int
    /// The number of blocks; the last one pads its rows past ``queryCount``.
    private let blockCount: Int
    /// The number of zero keys ahead of the first key, so that the first
    /// block finds a whole window ahead of its rows.
    private let leadingPadding: Int
    /// The number of zero keys after the last key, so that the last block
    /// finds the keys of its padded rows.
    private let trailingPadding: Int
    /// Where each query row of a block may attend: `[blockCount, 1,
    /// blockSize, span]`.
    private let mask: MLXArray

    /// The number of keys one block reads.
    private var span: Int { blockSize + window - 1 }

    /// The bands of `queryCount` queries over `keyCount` keys, or `nil` when
    /// the queries fit in fewer than two blocks, where the array mask costs
    /// no more.
    ///
    /// - Parameters:
    ///   - queryCount: The number of queries of the write.
    ///   - keyCount: The number of keys the cache presents, the keys of the
    ///     queries included; at least `queryCount`.
    ///   - window: The number of keys each query reads; at least one.
    package init?(queryCount: Int, keyCount: Int, window: Int) {
        precondition(window > 0, "A sliding window reads at least one key")
        precondition(keyCount >= queryCount, "The keys of the queries are cached keys")
        let blockSize = Self.blockSize(window: window)
        let blockCount = (queryCount + blockSize - 1) / blockSize
        guard blockCount >= Self.minimumBlockCount else { return nil }

        let cachedKeys = keyCount - queryCount
        self.queryCount = queryCount
        self.keyCount = keyCount
        self.window = window
        self.blockSize = blockSize
        self.blockCount = blockCount
        self.leadingPadding = max(0, window - 1 - cachedKeys)
        self.trailingPadding = blockCount * blockSize - queryCount
        self.mask = Self.makeMask(
            blockCount: blockCount, blockSize: blockSize, window: window,
            firstKey: cachedKeys - (window - 1), keyCount: keyCount)
    }

    /// The mask of every block: query row `q` of block `b` reads key `k` of
    /// the block when `q <= k < q + window`, and when `k` stands for a key of
    /// the cache and not for padding.
    ///
    /// - Parameters:
    ///   - blockCount: The number of blocks.
    ///   - blockSize: The number of query rows of one block.
    ///   - window: The number of keys each query reads.
    ///   - firstKey: The index, among the keys of the cache, of the first key
    ///     of the first block; negative where the block starts in padding.
    ///   - keyCount: The number of keys of the cache.
    /// - Returns: The mask, `[blockCount, 1, blockSize, blockSize + window - 1]`.
    private static func makeMask(
        blockCount: Int, blockSize: Int, window: Int, firstKey: Int, keyCount: Int
    ) -> MLXArray {
        let span = blockSize + window - 1
        let key = MLXArray.arange(0, span, dtype: .int32).reshaped(1, 1, span)
        let row = MLXArray.arange(0, blockSize, dtype: .int32).reshaped(1, blockSize, 1)
        let block = MLXArray.arange(0, blockCount, dtype: .int32).reshaped(blockCount, 1, 1)
        let cacheKey = block * Int32(blockSize) + key + Int32(firstKey)
        let inWindow = (key .>= row) & (key .< row + Int32(window))
        let inCache = (cacheKey .>= 0) & (cacheKey .< Int32(keyCount))
        return expandedDimensions(inWindow & inCache, axis: HeadRowsAxis.head)
    }

    /// The attention of the queries over the keys, by bands.
    ///
    /// - Parameters:
    ///   - queries: The queries, `[batch, heads, queryCount, width]`.
    ///   - keys: The keys the cache presents, `[batch, kvHeads, keyCount,
    ///     width]`.
    ///   - values: The values of the keys, `[batch, kvHeads, keyCount,
    ///     valueWidth]`.
    ///   - scale: The factor the scores take before the softmax.
    ///   - sinks: The learned sink logit of each query head, or `nil`.
    /// - Returns: The attention output, `[batch, heads, queryCount,
    ///   valueWidth]`.
    package func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            queries.dim(HeadRowsAxis.row) == queryCount && keys.dim(HeadRowsAxis.row) == keyCount,
            "The bands were made for \(queryCount) queries over \(keyCount) keys")
        let batch = queries.dim(HeadRowsAxis.batch)
        let blockMask =
            batch == 1 ? mask : tiled(mask, repetitions: [batch, 1, 1, 1])
        let output = MLXFast.scaledDotProductAttention(
            queries: blockQueries(queries), keys: bandRows(of: keys),
            values: bandRows(of: values), scale: scale, mask: .array(blockMask), sinks: sinks)
        return rowsInOrder(of: output, batch: batch)
    }

    /// The queries with the rows of each block as a sequence of their own:
    /// `[batch * blockCount, heads, blockSize, width]`, the padded rows last.
    private func blockQueries(_ queries: MLXArray) -> MLXArray {
        let (batch, heads, width) = (
            queries.dim(HeadRowsAxis.batch), queries.dim(HeadRowsAxis.head),
            queries.dim(HeadRowsAxis.width)
        )
        return padRows(queries, leading: 0, trailing: trailingPadding)
            .reshaped(batch, heads, blockCount, blockSize, width)
            .transposed(axes: BandAxis.swapHeadAndBlock)
            .reshaped(batch * blockCount, heads, blockSize, width)
    }

    /// The keys or the values each block reads, as a sequence of their own:
    /// `[batch * blockCount, kvHeads, span, width]`.
    private func bandRows(of rows: MLXArray) -> MLXArray {
        let (batch, heads, width) = (
            rows.dim(HeadRowsAxis.batch), rows.dim(HeadRowsAxis.head),
            rows.dim(HeadRowsAxis.width)
        )
        let padded = padRows(rows, leading: leadingPadding, trailing: trailingPadding)
        let firstRow = keyCount - queryCount - (window - 1) + leadingPadding
        let bands = (0 ..< blockCount).map { block in
            let start = firstRow + block * blockSize
            return padded[.ellipsis, start ..< (start + span), 0...]
        }
        return stacked(bands, axis: BandAxis.block)
            .reshaped(batch * blockCount, heads, span, width)
    }

    /// The output of every block back in the order of the queries, without
    /// the padded rows: `[batch, heads, queryCount, width]`.
    private func rowsInOrder(of output: MLXArray, batch: Int) -> MLXArray {
        let (heads, width) = (output.dim(HeadRowsAxis.head), output.dim(HeadRowsAxis.width))
        return output.reshaped(batch, blockCount, heads, blockSize, width)
            .transposed(axes: BandAxis.swapHeadAndBlock)
            .reshaped(batch, heads, blockCount * blockSize, width)[.ellipsis, ..<queryCount, 0...]
    }

    /// `rows` with `leading` zero rows ahead of them and `trailing` zero rows
    /// after them; `rows` itself when both are zero.
    private func padRows(_ rows: MLXArray, leading: Int, trailing: Int) -> MLXArray {
        guard leading > 0 || trailing > 0 else { return rows }
        var widths = [IntOrPair](repeating: 0, count: HeadRowsAxis.count)
        widths[HeadRowsAxis.row] = IntOrPair((leading, trailing))
        return padded(rows, widths: widths)
    }
}

/// The axes of a tensor whose rows stand in blocks: `[batch, heads, block,
/// row, width]`, or `[batch, block, heads, row, width]` after the swap.
private enum BandAxis {
    /// The axis that holds the blocks, beside the batch axis.
    static let block = 1

    /// The axis order that swaps the head axis and the block axis. The swap is
    /// its own inverse, thus the same order takes the blocks back again.
    static let swapHeadAndBlock = [0, 2, 1, 3, 4]
}
