// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from scouzi1966/mlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4Compressor.swift @ 07e1b806cc7e7291d05e1d3a95e3a04b3139e531
// which gives the Osaurus AI copyright line above. Manual transcription of the
// `DeepseekV4Cache` part alone, lines 287 to 858; no git ancestry.
//
// The pooled cache DeepSeek-V4 sparse attention needs. A decode step carries
// one token, thus a stateless compressor pools nothing and the global context
// would go away at the first decode step. This file keeps the pooled chunks,
// and the raw rows of an incomplete chunk, across calls and across turns.
//
// Five details do not come from that file.
//
//  1. **The incomplete tail is kept as RAW rows, not as projected rows.** The
//     file above keeps the `wkv` and `wgate` output of the tokens that have
//     not filled a chunk yet, and it holds a second pooling path
//     (`accumulateOverlapWindows`) that joins them. This file keeps the block
//     input itself and re-pools it through
//     ``DeepSeekV4Compressor/callAsFunction(_:rope:offset:)``, which is the
//     one pooling path this port has and the one 12 tests already hold. The
//     cost is that at most `2 * compress_ratio` rows go through the two
//     projections again for each call.
//  2. **An overlapping layer keeps one WHOLE chunk of rows as well.** Chunk
//     `c` of a ratio-4 layer reads the tokens of chunk `c - 1` as well as its
//     own. The stateless pooling gives its FIRST chunk a padded left half,
//     thus this cache starts each call one whole chunk early and drops that
//     first pooled row. Chunk `c` then reads real tokens on both halves
//     whatever the call boundary is.
//  3. **The pooled rows are recomputed, never trimmed proportionally.** The
//     `trim(_:)` of the file above drops `max(1, n / compress_ratio)` trailing
//     pooled rows and clears its buffers, which leaves the pool one row short
//     or one row long of the rewound position. This file keeps every raw row
//     while a rewind is still legal -- which is while `RotatingKVCache` still
//     answers `isTrimmable`, thus over the first `sliding_window` positions --
//     and rebuilds the pool from those rows. Past that point it holds the
//     minimal carry and answers `isTrimmable` false, thus
//     `RewindToCommonPrefixRule` never asks for a rewind this cache cannot
//     make exactly.
//  4. **No pool quantization.** The file above encodes each pooled row to
//     UInt8 behind a `DSV4_POOL_QUANT` environment switch. This port keeps the
//     pool in the activation dtype. The switch is a memory trade of its own
//     and it changes the numbers.
//  5. **Any batch size.** The file above opens its pooling path with
//     `precondition(projectedKV.dim(0) == 1)`, because its pool windows belong
//     to one request. Nothing here belongs to a request, thus this file reads
//     any batch.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - The pooled chunks of one compressor

/// The chunks one DeepSeek-V4 compressor pooled, and the raw rows it has not
/// pooled into a whole chunk yet.
///
/// The pooled chunks are keyed to ABSOLUTE positions: chunk `c` covers the raw
/// positions `c * chunkWidth` through `(c + 1) * chunkWidth - 1`, whatever
/// call gave those tokens. A prompt arrives in prefill chunks whose boundaries
/// are not multiples of the compress ratio, and a decode step then adds one
/// token at a time, thus the cache holds the rows of the run in progress and
/// pools them when the run ends.
final class DeepSeekV4ChunkCache {

    /// The number of raw positions one pooled chunk covers, which is the
    /// compress ratio of this layer.
    let chunkWidth: Int

    /// The number of whole chunks of raw rows the next call must read again.
    ///
    /// An overlapping layer pools chunk `c` from the tokens of chunk `c - 1`
    /// as well as its own, thus it keeps one whole chunk and drops the first
    /// pooled row of each call. A layer that does not overlap keeps none.
    let retainedChunkCount: Int

    /// The number of leading positions whose raw rows stay whole.
    ///
    /// It is the sliding window, which is also the range in which the
    /// `RotatingKVCache` beside this cache still rewinds. Inside that range a
    /// rewind rebuilds the pool exactly; past it this cache holds the minimal
    /// carry and ``holdsEveryRewindableRow`` answers false.
    let rewindableTokenCount: Int

    /// The chunks this cache holds, shape `(batch, chunks, headDim)`, or `nil`
    /// before the first call.
    private(set) var chunks: MLXArray?

    /// The raw rows the next call must read again, shape
    /// `(batch, rows, hidden)`, or `nil` when there are none.
    private(set) var carry: MLXArray?

    /// The absolute position of the first row of ``carry``. It is always a
    /// multiple of ``chunkWidth``.
    private(set) var carryStart = 0

    /// The number of whole chunks of rows an overlapping layer keeps.
    private static let overlapRetainedChunkCount = 1

    /// The number of whole chunks of rows a layer that does not overlap keeps.
    private static let plainRetainedChunkCount = 0

    /// The axis a `(batch, tokens, width)` tensor holds its tokens on. A pool
    /// tensor and a carry tensor share that layout.
    private static let tokenAxis = 1

    /// The number of arrays this cache writes into a serialized state: the
    /// pooled chunks, the raw rows, and the position those rows start at.
    static let stateSlotCount = 3

    /// The place of the pooled chunks inside a serialized state.
    private static let chunksSlot = 0

    /// The place of the raw rows inside a serialized state.
    private static let carrySlot = 1

    /// The place of the carry position inside a serialized state.
    private static let carryStartSlot = 2

    /// The state slots a branch that is not there writes, so that the slot
    /// count of a serialized state never changes.
    static var absentState: [MLXArray] {
        [MLXArray.zeros([0]), MLXArray.zeros([0]), MLXArray([Int32(0)])]
    }

    /// Builds an empty pooled cache.
    ///
    /// - Parameters:
    ///   - chunkWidth: The number of raw positions one pooled chunk covers.
    ///   - retainedChunkCount: The number of whole chunks of raw rows the next
    ///     call must read again.
    ///   - rewindableTokenCount: The number of leading positions whose raw
    ///     rows stay whole.
    init(chunkWidth: Int, retainedChunkCount: Int, rewindableTokenCount: Int) {
        precondition(chunkWidth > 0, "a pooled cache needs a compress ratio of more than 0")
        self.chunkWidth = chunkWidth
        self.retainedChunkCount = retainedChunkCount
        self.rewindableTokenCount = rewindableTokenCount
    }

    /// Builds the empty pooled cache of one layer of a checkpoint.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer.
    convenience init(configuration: DeepSeekV4Configuration, layer: Int) {
        let ratio = configuration.compressRatio(ofLayer: layer)
        precondition(
            configuration.hasCompressor(layer: layer),
            "layer \(layer) has no compressor, thus its compress ratio is 0")
        self.init(
            chunkWidth: ratio,
            retainedChunkCount: ratio == DeepSeekV4Configuration.indexerCompressRatio
                ? Self.overlapRetainedChunkCount : Self.plainRetainedChunkCount,
            rewindableTokenCount: configuration.slidingWindow)
    }

    /// The number of chunks this cache holds.
    var chunkCount: Int {
        chunks?.dim(Self.tokenAxis) ?? 0
    }

    /// True while every raw row of the history is still here, thus while a
    /// rewind to any earlier position rebuilds the pool exactly.
    var holdsEveryRewindableRow: Bool {
        carryStart == 0
    }

    /// The number of rows ``carry`` holds.
    private var carryRowCount: Int {
        carry?.dim(Self.tokenAxis) ?? 0
    }

    /// Pools one block of tokens and answers every chunk this cache holds.
    ///
    /// The block joins the rows the last call kept, the join goes through the
    /// compressor, and the chunks that stand before those rows stay as they
    /// were. The answer is thus the same whatever the block was cut into.
    ///
    /// - Parameters:
    ///   - block: The block input, shape `(batch, tokens, hidden)`.
    ///   - compressor: The compressor of this branch.
    ///   - rope: The rotary position of this layer.
    ///   - offset: The absolute position of the first token of the block.
    /// - Returns: The chunks, shape `(batch, chunks, headDim)`.
    func pooled(
        _ block: MLXArray, through compressor: DeepSeekV4Compressor,
        rope: DeepSeekV4RoPE, offset: Int
    ) -> MLXArray {
        precondition(
            carryStart + carryRowCount == offset,
            "the cache stands at position \(carryStart + carryRowCount) and the block "
                + "starts at position \(offset)")

        let rows = carry.map { concatenated([$0, block], axis: Self.tokenAxis) } ?? block
        let fresh = compressor(rows, rope: rope, offset: carryStart)
        let joined = joined(fresh: fresh, droppingLeading: chunkCount - carryStart / chunkWidth)
        chunks = joined

        let nextStart = nextCarryStart(
            chunkCount: joined.dim(Self.tokenAxis),
            endPosition: offset + block.dim(Self.tokenAxis))
        carry = trailingRows(of: rows, from: nextStart - carryStart)
        carryStart = nextStart
        return joined
    }

    /// Takes this cache back to an earlier absolute position.
    ///
    /// - Parameter position: The number of positions the cache must hold after
    ///   the rewind.
    /// - Returns: True when the rewind happened. It answers false when the raw
    ///   rows of the rewound region are gone, and it then changes nothing.
    @discardableResult
    func rewind(to position: Int) -> Bool {
        guard carryStart <= position else { return false }
        let keptCount = min(chunkCount, position / chunkWidth)
        chunks = chunks.map { $0[0..., 0 ..< keptCount, 0...] }
        carry = leadingCarry(rowCount: position - carryStart)
        return true
    }

    /// The arrays this cache writes into a serialized state.
    ///
    /// The carry position rides in an array rather than in the meta state, so
    /// that a reader may set the state and the meta state in either order.
    var state: [MLXArray] {
        [Self.serializable(chunks), Self.serializable(carry), MLXArray([Int32(carryStart)])]
    }

    /// Reads a serialized state back.
    ///
    /// - Parameter state: The arrays ``state`` wrote, in that order.
    func restore(state: [MLXArray]) {
        precondition(
            state.count == Self.stateSlotCount,
            "a pooled cache state holds \(Self.stateSlotCount) arrays, and this one holds "
                + "\(state.count)")
        self.chunks = Self.nullable(state[Self.chunksSlot])
        self.carry = Self.nullable(state[Self.carrySlot])
        self.carryStart = Int(state[Self.carryStartSlot].asArray(Int32.self)[0])
    }

    /// The arrays a forward pass must evaluate to keep the lazy graph bounded.
    ///
    /// - Returns: The pooled chunks and the raw rows this cache holds.
    func innerState() -> [MLXArray] {
        [chunks, carry].compactMap { $0 }
    }

    /// Builds an independent copy of this cache.
    ///
    /// A pooled tensor is replaced rather than written in place, thus a view
    /// of it is a copy for every purpose this cache has.
    ///
    /// - Returns: The copy.
    func copy() -> DeepSeekV4ChunkCache {
        let duplicate = DeepSeekV4ChunkCache(
            chunkWidth: chunkWidth, retainedChunkCount: retainedChunkCount,
            rewindableTokenCount: rewindableTokenCount)
        duplicate.chunks = chunks?[.ellipsis]
        duplicate.carry = carry?[.ellipsis]
        duplicate.carryStart = carryStart
        return duplicate
    }

    /// Joins the chunks this cache already holds to the chunks this call
    /// pooled.
    ///
    /// The pooling starts at ``carryStart``, thus its leading rows re-pool
    /// chunks the cache already holds. Those rows are DROPPED rather than
    /// kept: on an overlapping layer the first row of any pooling takes a
    /// padded left half, which is right only for chunk 0. The stored row of
    /// the same chunk read the real tokens of the chunk before it.
    ///
    /// - Parameters:
    ///   - fresh: The chunks the compressor pooled from ``carryStart`` onward.
    ///   - dropCount: The number of leading fresh chunks the cache already
    ///     holds.
    /// - Returns: Every chunk, in absolute chunk order.
    private func joined(fresh: MLXArray, droppingLeading dropCount: Int) -> MLXArray {
        precondition(
            dropCount >= 0 && dropCount <= fresh.dim(Self.tokenAxis),
            "a pooling from position \(carryStart) answered \(fresh.dim(Self.tokenAxis)) "
                + "chunks, and \(dropCount) of them are already here")
        let added = dropCount > 0 ? fresh[0..., dropCount..., 0...] : fresh
        guard let chunks, chunkCount > 0 else { return added }
        return concatenated([chunks, added], axis: Self.tokenAxis)
    }

    /// The absolute position the raw rows of the next call must start at.
    ///
    /// While the history is short enough for a rewind, that position is 0 and
    /// every raw row stays. Past that the cache keeps ``retainedChunkCount``
    /// whole chunks and the run in progress.
    ///
    /// - Parameters:
    ///   - chunkCount: The number of chunks the cache holds after this call.
    ///   - endPosition: The absolute position after the last token of this
    ///     call.
    /// - Returns: The position, which is a multiple of ``chunkWidth``.
    private func nextCarryStart(chunkCount: Int, endPosition: Int) -> Int {
        guard endPosition > rewindableTokenCount else { return 0 }
        return max(0, chunkCount - retainedChunkCount) * chunkWidth
    }

    /// The trailing rows of a joined block, from one index onward.
    ///
    /// - Parameters:
    ///   - rows: The joined block, shape `(batch, rows, hidden)`.
    ///   - index: The first row to keep.
    /// - Returns: The rows, or `nil` when none are left.
    private func trailingRows(of block: MLXArray, from index: Int) -> MLXArray? {
        guard index < block.dim(Self.tokenAxis) else { return nil }
        return block[0..., index..., 0...]
    }

    /// The leading rows of ``carry``.
    ///
    /// - Parameter rowCount: The number of rows to keep.
    /// - Returns: The rows, or `nil` when none are left.
    private func leadingCarry(rowCount: Int) -> MLXArray? {
        guard rowCount > 0, let carry else { return nil }
        return carry[0..., 0 ..< rowCount, 0...]
    }

    /// The array a serialized state carries for one slot.
    ///
    /// - Parameter array: The array, or `nil`.
    /// - Returns: The array, or an empty one-axis array standing for `nil`.
    private static func serializable(_ array: MLXArray?) -> MLXArray {
        array ?? MLXArray.zeros([0])
    }

    /// The array one slot of a serialized state stood for.
    ///
    /// An empty pool and no pool hold the same chunks, thus both read back as
    /// `nil`.
    ///
    /// - Parameter array: The array a serialized state carried.
    /// - Returns: The array, or `nil` when it holds no value.
    private static func nullable(_ array: MLXArray) -> MLXArray? {
        array.size > 0 ? array : nil
    }
}

// MARK: - The cache of one compressed layer

/// The key/value cache of one DeepSeek-V4 attention layer whose compress ratio
/// is more than 0.
///
/// The layer reads two things: a sliding window of the most recent keys, and
/// the pooled chunks of everything before them. This cache holds both. The
/// window is a plain `RotatingKVCache`, and the chunks are one
/// ``DeepSeekV4ChunkCache`` for the compressor of the attention and, on a
/// layer that holds an indexer, a second one for the compressor inside it.
///
/// Every `KVCache` requirement that speaks of keys and values goes straight to
/// the window, thus the generation loop, the prompt-cache reuse rules and the
/// mask builders read this cache as they read any sliding-window cache.
public final class DeepSeekV4Cache: KVCache {

    /// The sliding window of the most recent keys.
    let local: RotatingKVCache

    /// The chunks the compressor of the attention pooled.
    let attentionChunks: DeepSeekV4ChunkCache

    /// The chunks the compressor inside the indexer pooled, or `nil` on a
    /// layer that holds no indexer.
    let indexerChunks: DeepSeekV4ChunkCache?

    /// The number of branches whose state this cache writes: the compressor of
    /// the attention, and the compressor inside the indexer.
    private static let branchCount = 2

    /// The number of arrays this cache adds to the state of its window: one
    /// slot group for each branch.
    private static let stateSlotCount = branchCount * DeepSeekV4ChunkCache.stateSlotCount

    /// Builds the empty cache of one layer of a checkpoint.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer.
    public init(configuration: DeepSeekV4Configuration, layer: Int) {
        self.local = RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
        self.attentionChunks = DeepSeekV4ChunkCache(configuration: configuration, layer: layer)
        self.indexerChunks =
            configuration.hasIndexer(layer: layer)
            ? DeepSeekV4ChunkCache(configuration: configuration, layer: layer) : nil
    }

    /// Builds a cache that holds the given window and branches.
    ///
    /// - Parameters:
    ///   - local: The sliding window.
    ///   - attentionChunks: The chunks of the compressor of the attention.
    ///   - indexerChunks: The chunks of the compressor inside the indexer.
    private init(
        local: RotatingKVCache,
        attentionChunks: DeepSeekV4ChunkCache,
        indexerChunks: DeepSeekV4ChunkCache?
    ) {
        self.local = local
        self.attentionChunks = attentionChunks
        self.indexerChunks = indexerChunks
    }

    /// The number of positions this cache has read.
    public var offset: Int { local.offset }

    /// The number of keys the sliding window holds.
    public var maxSize: Int? { local.maxSize }

    /// Adds the keys of one block to the sliding window.
    ///
    /// - Parameters:
    ///   - keys: The keys of this block.
    ///   - values: The values of this block, which DeepSeek-V4 keeps in the
    ///     one latent head that also carries the keys.
    /// - Returns: Every key and value the window holds.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        local.update(keys: keys, values: values)
    }

    /// The arrays a forward pass must evaluate to keep the lazy graph bounded.
    ///
    /// - Returns: The window, the pooled chunks and the raw rows of each
    ///   branch.
    public func innerState() -> [MLXArray] {
        local.innerState() + attentionChunks.innerState()
            + (indexerChunks?.innerState() ?? [])
    }

    /// The serialized state: the state of the window, then the slots of each
    /// branch.
    ///
    /// The branch slots sit at the END, because the window writes no array at
    /// all before its first write and two after it.
    public var state: [MLXArray] {
        get {
            local.state + attentionChunks.state
                + (indexerChunks?.state ?? DeepSeekV4ChunkCache.absentState)
        }
        set {
            precondition(
                newValue.count >= Self.stateSlotCount,
                "a DeepSeek-V4 cache state ends in \(Self.stateSlotCount) branch arrays, and "
                    + "this one holds \(newValue.count) arrays")
            let split = newValue.count - Self.stateSlotCount
            let slots = DeepSeekV4ChunkCache.stateSlotCount
            local.state = Array(newValue[0 ..< split])
            attentionChunks.restore(state: Array(newValue[split ..< (split + slots)]))
            indexerChunks?.restore(state: Array(newValue[(split + slots)...]))
        }
    }

    /// The serialized meta state, which is the meta state of the window.
    ///
    /// The branches keep their own numbers in ``state`` rather than here, thus
    /// a reader may set the two in either order.
    public var metaState: [String] {
        get { local.metaState }
        set { local.metaState = newValue }
    }

    /// True when this cache can go back to an earlier position.
    ///
    /// The window rewinds only before it has wrapped, and the branches rewind
    /// only while they still hold every raw row. The two ranges are the same
    /// range, thus a caller that reads this property never asks for a rewind
    /// the pool cannot make exactly.
    public var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    /// True when this cache can still go back after `positions` more tokens.
    ///
    /// - Parameter positions: The number of positions that would be added.
    /// - Returns: True when a later rewind stays valid.
    public func isTrimmable(after positions: Int) -> Bool {
        local.isTrimmable(after: positions) && attentionChunks.holdsEveryRewindableRow
            && (indexerChunks?.holdsEveryRewindableRow ?? true)
    }

    /// Takes this cache back by `n` positions.
    ///
    /// - Parameter n: The number of positions to drop.
    /// - Returns: The number of positions dropped.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = local.trim(n)
        guard trimmed > 0 else { return trimmed }
        attentionChunks.rewind(to: local.offset)
        indexerChunks?.rewind(to: local.offset)
        return trimmed
    }

    /// The attention mask of one block, which is the mask of the sliding
    /// window. The chunk half of the mask belongs to the attention layer,
    /// because only it knows how many chunks the block pooled.
    ///
    /// - Parameters:
    ///   - n: The number of tokens in this block.
    ///   - windowSize: The sliding window, or `nil` for the window of this
    ///     cache.
    ///   - returnArray: True to force an array mask rather than a symbolic
    ///     one.
    /// - Returns: The mask.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        local.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Builds an independent copy of this cache.
    ///
    /// - Returns: The copy.
    public func copy() -> any KVCache {
        guard let window = local.copy() as? RotatingKVCache else {
            preconditionFailure("a rotating window must copy into a rotating window")
        }
        return DeepSeekV4Cache(
            local: window,
            attentionChunks: attentionChunks.copy(),
            indexerChunks: indexerChunks?.copy())
    }
}
