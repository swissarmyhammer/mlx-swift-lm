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
// and the rows of an incomplete chunk, across calls and across turns.
//
// Five details do not come from that file.
//
//  1. **One pooling path.** The file above keeps the `wkv` and `wgate` output
//     of the tokens that have not filled a chunk yet, and it joins them
//     through a second pooling path of its own (`accumulateOverlapWindows`).
//     This file gives the rows to
//     ``DeepSeekV4Compressor/pooled(_:rope:offset:dtype:)``, which is the one
//     pooling path this port has and the one the compressor tests hold. A call
//     pools only the chunks that it ends, thus a decode step that ends no
//     chunk does no pooling at all (task `^cm8fq2q`). A call that ends no
//     chunk does no projection either: it keeps the raw rows of its block,
//     and the call that ends the chunk projects the whole run one time
//     (task `^4n7mtxa`, after the osaurus vmlx-swift commit fd93476a).
//  2. **An overlapping layer keeps one WHOLE chunk of rows as well.** Chunk
//     `c` of a ratio-4 layer reads the tokens of chunk `c - 1` as well as its
//     own. The pooling gives its FIRST chunk a padded left half, thus this
//     cache pools from one whole chunk early and drops that first pooled row.
//     Chunk `c` then reads real tokens on both halves whatever the call
//     boundary is. mlx-lm PR 1195 keeps the same chunk (`state_kv[:, :ratio]`).
//     mlx-vlm (`PoolingCache.accumulate_windows`) does not, thus there a
//     ratio-4 chunk takes a padded left half at each decode window.
//  3. **The pooled rows are recomputed, never trimmed proportionally.** The
//     `trim(_:)` of the file above drops `max(1, n / compress_ratio)` trailing
//     pooled rows and clears its buffers, which leaves the pool one row short
//     or one row long of the rewound position. This file keeps every projected
//     row while a rewind is still legal -- which is while `RotatingKVCache`
//     still answers `isTrimmable`, thus over the first `sliding_window`
//     positions -- and pools the rewound chunks again from those rows. Past
//     that point it holds the minimal carry and answers `isTrimmable` false,
//     thus `RewindToCommonPrefixRule` never asks for a rewind this cache cannot
//     make exactly.
//  4. **No pool quantization.** The file above encodes each pooled row to
//     UInt8 behind a `DSV4_POOL_QUANT` environment switch. This port keeps the
//     pool of the attention in the activation dtype, and the pool of the
//     indexer in the float32 dtype its scores use (task `^atj552p`). The
//     switch is a memory trade of its own and it changes the numbers.
//  5. **Any batch size.** The file above opens its pooling path with
//     `precondition(projectedKV.dim(0) == 1)`, because its pool windows belong
//     to one request. Nothing here belongs to a request, thus this file reads
//     any batch.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - A buffer of rows that grows in steps

/// A buffer of `(batch, rows, width)` rows that grows in steps, as
/// `KVCacheSimple` grows its keys.
///
/// A write goes into free rows of the buffer that is already there, thus MLX
/// can write in place when nothing else holds the buffer. A concatenation runs
/// only when the buffer is full, which is once for each ``allocationStep``
/// rows.
///
/// The buffer is a class because a write changes the one `MLXArray` it holds
/// in place. No view this buffer gives out is that array: each view is a new
/// slice, thus a write never changes an array a caller holds.
final class DeepSeekV4RowBuffer {

    /// The number of rows a buffer grows by when its caller names no step.
    static let defaultAllocationStep = 256

    /// The number of rows the buffer grows by.
    let allocationStep: Int

    /// The axis a `(batch, rows, width)` tensor holds its batch on.
    private static let batchAxis = 0

    /// The axis a `(batch, rows, width)` tensor holds its rows on.
    private static let rowAxis = 1

    /// The axis a `(batch, rows, width)` tensor holds its width on.
    private static let widthAxis = 2

    /// The buffer, shape `(batch, capacity, width)`, or `nil` when it holds no
    /// row. The rows past ``count`` hold no value.
    private(set) var storage: MLXArray?

    /// The number of rows the buffer holds.
    private(set) var count = 0

    /// Builds an empty buffer.
    ///
    /// - Parameter allocationStep: The number of rows the buffer grows by.
    init(allocationStep: Int = defaultAllocationStep) {
        precondition(allocationStep > 0, "a row buffer must grow by at least one row")
        self.allocationStep = allocationStep
    }

    /// Builds a buffer that holds the given rows and no free row.
    ///
    /// - Parameters:
    ///   - rows: The rows, shape `(batch, rows, width)`, or `nil` for an empty
    ///     buffer.
    ///   - allocationStep: The number of rows the buffer grows by.
    convenience init(rows: MLXArray?, allocationStep: Int = defaultAllocationStep) {
        self.init(allocationStep: allocationStep)
        guard let rows, rows.dim(Self.rowAxis) > 0 else { return }
        self.storage = rows[.ellipsis]
        self.count = rows.dim(Self.rowAxis)
    }

    /// A new view of the rows the buffer holds, or `nil` when it holds none.
    var rows: MLXArray? {
        guard let storage, count > 0 else { return nil }
        return storage[0..., 0 ..< count, 0...]
    }

    /// Writes rows after the rows the buffer holds.
    ///
    /// - Parameter added: The rows, shape `(batch, rows, width)`.
    func append(_ added: MLXArray) {
        let newCount = count + added.dim(Self.rowAxis)
        let buffer = reserved(toHold: newCount, like: added)
        buffer[0..., count ..< newCount, 0...] = added
        storage = buffer
        count = newCount
    }

    /// Keeps the leading rows and forgets the others.
    ///
    /// - Parameter newCount: The number of leading rows to keep.
    func truncate(to newCount: Int) {
        precondition(
            newCount >= 0 && newCount <= count,
            "a buffer of \(count) rows cannot keep \(newCount) rows")
        count = newCount
        if count == 0 {
            storage = nil
        }
    }

    /// Forgets every row and keeps the storage, thus the next write fills the
    /// free rows with no new allocation.
    ///
    /// A storage that grew past one ``allocationStep`` goes, thus a long block
    /// leaves no large buffer behind.
    func removeAll() {
        count = 0
        if let storage, storage.dim(Self.rowAxis) > allocationStep {
            self.storage = nil
        }
    }

    /// Forgets the leading rows and moves the others to the front.
    ///
    /// A buffer that grew past one step during a long prefill block shrinks
    /// back to one step here, thus a long prompt leaves no large buffer behind.
    ///
    /// - Parameter dropCount: The number of leading rows to forget.
    func dropLeading(_ dropCount: Int) {
        precondition(
            dropCount >= 0 && dropCount <= count,
            "a buffer of \(count) rows cannot forget \(dropCount) rows")
        let keptCount = count - dropCount
        guard dropCount > 0 else { return }
        guard let storage, keptCount > 0 else {
            truncate(to: 0)
            return
        }
        let kept = storage[0..., dropCount ..< count, 0...]
        if storage.dim(Self.rowAxis) > allocationStep {
            truncate(to: 0)
            append(kept)
            return
        }
        storage[0..., 0 ..< keptCount, 0...] = kept
        count = keptCount
    }

    /// Builds an independent copy of this buffer.
    ///
    /// - Returns: The copy.
    func copy() -> DeepSeekV4RowBuffer {
        let duplicate = DeepSeekV4RowBuffer(allocationStep: allocationStep)
        duplicate.storage = storage?[.ellipsis]
        duplicate.count = count
        return duplicate
    }

    /// The buffer that has room for `needed` rows.
    ///
    /// - Parameters:
    ///   - needed: The number of rows the buffer must hold.
    ///   - added: The rows that will be written, which give the batch, the
    ///     width and the dtype of a new buffer.
    /// - Returns: The buffer, which is ``storage`` when it has room.
    private func reserved(toHold needed: Int, like added: MLXArray) -> MLXArray {
        if let storage, needed <= storage.dim(Self.rowAxis) {
            return storage
        }
        let missing = needed - count
        let stepCount = (missing + allocationStep - 1) / allocationStep
        let free = MLXArray.zeros(
            [
                added.dim(Self.batchAxis), stepCount * allocationStep,
                added.dim(Self.widthAxis),
            ], dtype: added.dtype)
        guard let held = rows else { return free }
        return concatenated([held, free], axis: Self.rowAxis)
    }
}

// MARK: - The pooled chunks of one compressor

/// The chunks one DeepSeek-V4 compressor pooled, and the rows it has not
/// pooled into a whole chunk yet.
///
/// The pooled chunks are keyed to ABSOLUTE positions: chunk `c` covers the raw
/// positions `c * chunkWidth` through `(c + 1) * chunkWidth - 1`, whatever
/// call gave those tokens. A prompt arrives in prefill chunks whose boundaries
/// are not multiples of the compress ratio, and a decode step then adds one
/// token at a time, thus the cache holds the rows of the run in progress and
/// pools them when the run ends.
///
/// A block that ends no run is NOT projected: the cache keeps its raw input
/// rows, and the block that ends the run projects them together with its own
/// tokens. Most decode steps thus read no `wkv` or `wgate` weight at all.
final class DeepSeekV4ChunkCache {

    /// The number of raw positions one pooled chunk covers, which is the
    /// compress ratio of this layer.
    let chunkWidth: Int

    /// The number of whole chunks of rows the next pooling must read again.
    ///
    /// An overlapping layer pools chunk `c` from the tokens of chunk `c - 1`
    /// as well as its own, thus it keeps one whole chunk and drops the first
    /// pooled row of each pooling. A layer that does not overlap keeps none.
    let retainedChunkCount: Int

    /// The number of leading positions whose projected rows stay whole.
    ///
    /// It is the sliding window, which is also the range in which the
    /// `RotatingKVCache` beside this cache still rewinds. Inside that range a
    /// rewind pools the chunks again exactly; past it this cache holds the
    /// minimal carry and ``holdsEveryRewindableRow`` answers false.
    let rewindableTokenCount: Int

    /// The dtype the pool keeps its chunks in, or `nil` to keep each chunk in
    /// the dtype of the block that pooled it.
    ///
    /// The branch of the indexer keeps ``DeepSeekV4Indexer/scoreDType``. The
    /// cast of a new chunk runs one time, when the chunk arrives, and a decode
    /// step then reads the whole pool with no cast. A cast from bfloat16 to
    /// float32 is exact, thus the chunks keep the bits of the cast that each
    /// step ran before.
    let poolDType: DType?

    /// The pooled chunks, in a buffer that grows in steps.
    private var pool = DeepSeekV4RowBuffer()

    /// The view of every pooled chunk that ``pooled(_:through:rope:offset:)``
    /// gives back, or `nil` while the cache holds no chunk.
    ///
    /// The view changes only when the chunk count changes, thus a call that
    /// ends no chunk gives back this same array and adds no operation.
    private var heldChunks: MLXArray?

    /// The `wkv` rows of the positions from ``carryStart`` onward.
    private var projectedValues = DeepSeekV4RowBuffer()

    /// The `wgate` rows of the positions from ``carryStart`` onward.
    private var projectedGates = DeepSeekV4RowBuffer()

    /// The raw block rows of the positions from ``projectedEnd`` onward,
    /// which no block has projected yet because no block has ended their run.
    ///
    /// The buffer grows by ``chunkWidth`` rows, because it never holds a whole
    /// run: the block that ends a run projects these rows and empties it.
    private var unprojectedRows: DeepSeekV4RowBuffer

    /// The absolute position of the first projected row. It is always a
    /// multiple of ``chunkWidth``.
    private(set) var carryStart = 0

    /// The number of whole chunks of rows an overlapping layer keeps.
    private static let overlapRetainedChunkCount = 1

    /// The number of whole chunks of rows a layer that does not overlap keeps.
    private static let plainRetainedChunkCount = 0

    /// The axis a `(batch, tokens, width)` tensor holds its tokens on. A pool
    /// tensor and a row tensor share that layout.
    private static let tokenAxis = 1

    /// The number of arrays this cache writes into a serialized state: the
    /// pooled chunks, the projected values, the projected gate logits, the
    /// unprojected rows, and the position the projected rows start at.
    static let stateSlotCount = 5

    /// The place of the pooled chunks inside a serialized state.
    private static let chunksSlot = 0

    /// The place of the projected values inside a serialized state.
    private static let valuesSlot = 1

    /// The place of the projected gate logits inside a serialized state.
    private static let gatesSlot = 2

    /// The place of the unprojected rows inside a serialized state.
    private static let unprojectedSlot = 3

    /// The place of the carry position inside a serialized state.
    private static let carryStartSlot = 4

    /// The state slots a branch that is not there writes, so that the slot
    /// count of a serialized state never changes.
    static var absentState: [MLXArray] {
        [
            noValuePlaceholder, noValuePlaceholder, noValuePlaceholder, noValuePlaceholder,
            MLXArray([Int32(0)]),
        ]
    }

    /// The array that stands for "no value" in a chunks or rows slot of a serialized state.
    ///
    /// The safetensors writer refuses an array that holds no element, thus a prompt cache file
    /// cannot hold an empty array. The placeholder holds one element, and its rank of 1 tells
    /// it apart from a `(batch, rows, width)` tensor.
    private static var noValuePlaceholder: MLXArray {
        MLXArray.zeros([1])
    }

    /// The largest element count of a one-axis array that stands for "no value": the empty
    /// array that earlier states held, or ``noValuePlaceholder``.
    private static let noValueMaximumSize = 1

    /// Builds an empty pooled cache.
    ///
    /// - Parameters:
    ///   - chunkWidth: The number of raw positions one pooled chunk covers.
    ///   - retainedChunkCount: The number of whole chunks of rows the next
    ///     pooling must read again.
    ///   - rewindableTokenCount: The number of leading positions whose
    ///     projected rows stay whole.
    ///   - poolDType: The dtype the pool keeps its chunks in, or `nil` to
    ///     keep the dtype of each block.
    init(
        chunkWidth: Int, retainedChunkCount: Int, rewindableTokenCount: Int, poolDType: DType?
    ) {
        precondition(chunkWidth > 0, "a pooled cache needs a compress ratio of more than 0")
        self.chunkWidth = chunkWidth
        self.retainedChunkCount = retainedChunkCount
        self.rewindableTokenCount = rewindableTokenCount
        self.poolDType = poolDType
        self.unprojectedRows = DeepSeekV4RowBuffer(allocationStep: chunkWidth)
    }

    /// Builds the empty pooled cache of one layer of a checkpoint.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer.
    ///   - poolDType: The dtype the pool keeps its chunks in, or `nil` to
    ///     keep the dtype of each block.
    convenience init(
        configuration: DeepSeekV4Configuration, layer: Int, poolDType: DType? = nil
    ) {
        let ratio = configuration.compressRatio(ofLayer: layer)
        precondition(
            configuration.hasCompressor(layer: layer),
            "layer \(layer) has no compressor, thus its compress ratio is 0")
        self.init(
            chunkWidth: ratio,
            retainedChunkCount: ratio == DeepSeekV4Configuration.indexerCompressRatio
                ? Self.overlapRetainedChunkCount : Self.plainRetainedChunkCount,
            rewindableTokenCount: configuration.slidingWindow, poolDType: poolDType)
    }

    /// The number of chunks this cache holds.
    var chunkCount: Int {
        pool.count
    }

    /// Every chunk this cache holds, shape `(batch, chunks, headDim)`, or
    /// `nil` when it holds none.
    var chunks: MLXArray? {
        heldChunks
    }

    /// The projected rows of the positions from ``carryStart`` up to the end
    /// of the last block that ended a chunk, or `nil` when the cache holds
    /// none. The rows of a run that no block has ended are not projected yet.
    var projectedRows: DeepSeekV4Compressor.ProjectedRows? {
        guard let values = projectedValues.rows, let gates = projectedGates.rows else {
            return nil
        }
        return DeepSeekV4Compressor.ProjectedRows(values: values, gates: gates)
    }

    /// True while every projected row of the history is still here, thus while
    /// a rewind to any earlier position pools the chunks again exactly.
    var holdsEveryRewindableRow: Bool {
        carryStart == 0
    }

    /// The absolute position after the last projected row, which is where the
    /// unprojected rows start.
    var projectedEnd: Int {
        carryStart + projectedValues.count
    }

    /// Pools one block of tokens and answers every chunk this cache holds.
    ///
    /// A block that ends a run is projected together with the unprojected
    /// rows the last calls kept, and joins the projected rows. Only the chunks
    /// that this block ends are pooled, and the chunks that stand before them
    /// stay as they were. The answer is thus the same whatever the block was
    /// cut into. A block that ends no run does no projection and no pooling:
    /// the cache keeps its raw rows.
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
        let position = projectedEnd + unprojectedRows.count
        precondition(
            position == offset,
            "the cache stands at position \(position) and the block starts at position \(offset)")

        let endPosition = offset + block.dim(Self.tokenAxis)
        let endedChunkCount = endPosition / chunkWidth
        if endedChunkCount > chunkCount {
            projectRun(endingWith: block, through: compressor)
            poolChunks(
                endingBefore: endedChunkCount, through: compressor, rope: rope,
                dtype: block.dtype)
        } else {
            unprojectedRows.append(block)
        }
        dropRows(before: nextCarryStart(chunkCount: chunkCount, endPosition: endPosition))
        return heldChunks
            ?? zeros([block.dim(0), 0, compressor.headDim], dtype: poolDType ?? block.dtype)
    }

    /// Takes this cache back to an earlier absolute position.
    ///
    /// - Parameter position: The number of positions the cache must hold after
    ///   the rewind.
    /// - Returns: True when the rewind happened. It answers false when the
    ///   rows of the rewound region are gone, and it then changes nothing.
    @discardableResult
    func rewind(to position: Int) -> Bool {
        guard carryStart <= position else { return false }
        let keptCount = min(chunkCount, position / chunkWidth)
        if keptCount < chunkCount {
            heldChunks = nil
            pool.truncate(to: keptCount)
            heldChunks = pool.rows
        }
        let projectedEnd = self.projectedEnd
        if position >= projectedEnd {
            unprojectedRows.truncate(to: position - projectedEnd)
        } else {
            projectedValues.truncate(to: position - carryStart)
            projectedGates.truncate(to: position - carryStart)
            unprojectedRows.removeAll()
        }
        return true
    }

    /// The arrays this cache writes into a serialized state.
    ///
    /// The carry position rides in an array rather than in the meta state, so
    /// that a reader may set the state and the meta state in either order.
    var state: [MLXArray] {
        [
            heldChunks ?? Self.noValuePlaceholder,
            projectedValues.rows ?? Self.noValuePlaceholder,
            projectedGates.rows ?? Self.noValuePlaceholder,
            unprojectedRows.rows ?? Self.noValuePlaceholder,
            MLXArray([Int32(carryStart)]),
        ]
    }

    /// Checks a serialized state before ``restore(state:)`` reads it.
    ///
    /// ``restore(state:)`` stops the process on a wrong array count and reads the carry
    /// position as one `Int32`, thus a prompt cache file must pass this check first.
    ///
    /// - Parameter state: The arrays of one branch, as ``state`` wrote them.
    /// - Throws: `KVCacheError` when the arrays are not a state of this cache.
    func validateRestore(state: [MLXArray]) throws {
        guard state.count == Self.stateSlotCount,
            Self.isRowTensor(state[Self.chunksSlot]) || Self.isNoValue(state[Self.chunksSlot]),
            Self.areProjectedRows(values: state[Self.valuesSlot], gates: state[Self.gatesSlot]),
            isUnprojectedRun(state[Self.unprojectedSlot])
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid DeepSeek-V4 pooled-cache arrays.")
        }
        let carryStartArray = state[Self.carryStartSlot]
        guard carryStartArray.ndim == 1, carryStartArray.size == 1,
            carryStartArray.dtype == .int32
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: the DeepSeek-V4 carry position is not one Int32.")
        }
        let carryStart = Int(carryStartArray.asArray(Int32.self)[0])
        guard carryStart >= 0, carryStart.isMultiple(of: chunkWidth) else {
            throw KVCacheError(
                message:
                    "Corrupt prompt cache: the DeepSeek-V4 carry position \(carryStart) is not a chunk boundary."
            )
        }
    }

    /// True when a serialized branch state is the state of a branch that is not there, as
    /// ``absentState`` writes it.
    ///
    /// - Parameter state: The arrays of one branch.
    /// - Returns: True when the chunks and the rows stand for no value and the carry position
    ///   is 0.
    static func isAbsentState(_ state: [MLXArray]) -> Bool {
        state.count == stateSlotCount && isNoValue(state[chunksSlot])
            && isNoValue(state[valuesSlot]) && isNoValue(state[gatesSlot])
            && isNoValue(state[unprojectedSlot])
            && state[carryStartSlot].size == 1
            && state[carryStartSlot].dtype == .int32
            && state[carryStartSlot].asArray(Int32.self)[0] == 0
    }

    /// True when the two row slots of a serialized state belong together: both
    /// stand for no value, or both are row tensors of one shape.
    ///
    /// - Parameters:
    ///   - values: The array of the projected values slot.
    ///   - gates: The array of the projected gate logits slot.
    /// - Returns: True when the two slots are a valid pair.
    private static func areProjectedRows(values: MLXArray, gates: MLXArray) -> Bool {
        if isNoValue(values) && isNoValue(gates) {
            return true
        }
        return isRowTensor(values) && isRowTensor(gates) && values.shape == gates.shape
    }

    /// True when the unprojected rows slot of a serialized state stands for no
    /// value, or holds the rows of a run that has not ended: fewer rows than
    /// one chunk covers.
    ///
    /// - Parameter rows: The array of the unprojected rows slot.
    /// - Returns: True when the slot is valid for this cache.
    private func isUnprojectedRun(_ rows: MLXArray) -> Bool {
        Self.isNoValue(rows) || (Self.isRowTensor(rows) && rows.dim(Self.tokenAxis) < chunkWidth)
    }

    /// True when a chunks or rows slot holds a `(batch, rows, width)` tensor.
    ///
    /// - Parameter array: The array of the slot.
    /// - Returns: True when the array has the rank of a pool tensor or a row tensor.
    private static func isRowTensor(_ array: MLXArray) -> Bool {
        array.ndim == rowTensorRank
    }

    /// True when a chunks or rows slot holds the one-axis array that stands for no value.
    ///
    /// - Parameter array: The array of the slot.
    /// - Returns: True for ``noValuePlaceholder`` and for an empty one-axis array.
    private static func isNoValue(_ array: MLXArray) -> Bool {
        array.ndim == 1 && array.size <= noValueMaximumSize
    }

    /// The rank of a pool tensor and of a row tensor: `(batch, rows, width)`.
    private static let rowTensorRank = 3

    /// Reads a serialized state back.
    ///
    /// - Parameter state: The arrays ``state`` wrote, in that order.
    func restore(state: [MLXArray]) {
        precondition(
            state.count == Self.stateSlotCount,
            "a pooled cache state holds \(Self.stateSlotCount) arrays, and this one holds "
                + "\(state.count)")
        pool = DeepSeekV4RowBuffer(rows: Self.nullable(state[Self.chunksSlot]))
        heldChunks = pool.rows
        projectedValues = DeepSeekV4RowBuffer(rows: Self.nullable(state[Self.valuesSlot]))
        projectedGates = DeepSeekV4RowBuffer(rows: Self.nullable(state[Self.gatesSlot]))
        unprojectedRows = DeepSeekV4RowBuffer(
            rows: Self.nullable(state[Self.unprojectedSlot]), allocationStep: chunkWidth)
        carryStart = Int(state[Self.carryStartSlot].asArray(Int32.self)[0])
    }

    /// The arrays a forward pass must evaluate to keep the lazy graph bounded.
    ///
    /// - Returns: The buffers of the pooled chunks, of the projected rows and
    ///   of the unprojected rows, at their full allocated size.
    func innerState() -> [MLXArray] {
        [pool.storage, projectedValues.storage, projectedGates.storage, unprojectedRows.storage]
            .compactMap { $0 }
    }

    /// The arrays the generation loop evaluates with each token.
    ///
    /// The list is ``innerState()`` and the view of every pooled chunk that
    /// the next call reads. The view is a lazy slice of the pool, thus it
    /// must be evaluated on its own: an evaluation of the pool does not
    /// evaluate a slice of it. Each array is one this cache already holds,
    /// thus a read adds no operation.
    var evaluationState: [MLXArray] {
        innerState() + [heldChunks].compactMap { $0 }
    }

    /// Builds an independent copy of this cache.
    ///
    /// - Returns: The copy.
    func copy() -> DeepSeekV4ChunkCache {
        let duplicate = DeepSeekV4ChunkCache(
            chunkWidth: chunkWidth, retainedChunkCount: retainedChunkCount,
            rewindableTokenCount: rewindableTokenCount, poolDType: poolDType)
        duplicate.pool = pool.copy()
        duplicate.heldChunks = duplicate.pool.rows
        duplicate.projectedValues = projectedValues.copy()
        duplicate.projectedGates = projectedGates.copy()
        duplicate.unprojectedRows = unprojectedRows.copy()
        duplicate.carryStart = carryStart
        return duplicate
    }

    /// Projects the run that a block ends: the unprojected rows the last
    /// calls kept, then the block. The projected rows join the rows this
    /// cache holds, and no unprojected row stays.
    ///
    /// The tokens of the block past the last run it ends are projected too,
    /// thus the projected rows always reach the end of the block.
    ///
    /// - Parameters:
    ///   - block: The block input that ends the run, shape
    ///     `(batch, tokens, hidden)`.
    ///   - compressor: The compressor of this branch.
    private func projectRun(endingWith block: MLXArray, through compressor: DeepSeekV4Compressor) {
        let run =
            unprojectedRows.rows.map { concatenated([$0, block], axis: Self.tokenAxis) } ?? block
        let projected = compressor.project(run)
        projectedValues.append(projected.values)
        projectedGates.append(projected.gates)
        unprojectedRows.removeAll()
    }

    /// Pools the chunks that the last block ended and writes them after the
    /// chunks this cache already holds.
    ///
    /// The pooling starts one whole chunk early on an overlapping layer, thus
    /// its leading row pools a chunk the cache already holds. That row is
    /// DROPPED rather than kept: the first row of any pooling takes a padded
    /// left half, which is right only for chunk 0. The stored row of the same
    /// chunk read the real tokens of the chunk before it.
    ///
    /// - Parameters:
    ///   - endedChunkCount: The number of chunks that have ended after the
    ///     last block.
    ///   - compressor: The compressor of this branch.
    ///   - rope: The rotary position of this layer.
    ///   - dtype: The dtype the compressor pools the chunks in. The pool then
    ///     keeps them in ``poolDType`` when it is set.
    private func poolChunks(
        endingBefore endedChunkCount: Int, through compressor: DeepSeekV4Compressor,
        rope: DeepSeekV4RoPE, dtype: DType
    ) {
        guard let rows = projectedRows else {
            preconditionFailure("a block that ends a chunk leaves projected rows behind")
        }
        let firstChunk = max(carryStart / chunkWidth, chunkCount - retainedChunkCount)
        let firstRow = firstChunk * chunkWidth - carryStart
        let endRow = endedChunkCount * chunkWidth - carryStart
        let window = DeepSeekV4Compressor.ProjectedRows(
            values: rows.values[0..., firstRow ..< endRow, 0...],
            gates: rows.gates[0..., firstRow ..< endRow, 0...])
        let fresh = compressor.pooled(
            window, rope: rope, offset: firstChunk * chunkWidth, dtype: dtype)

        let dropCount = chunkCount - firstChunk
        let added = dropCount > 0 ? fresh[0..., dropCount..., 0...] : fresh
        // The view must go before the write, so that MLX can write the pool in
        // place rather than copy it.
        heldChunks = nil
        pool.append(poolDType.map { added.asType($0) } ?? added)
        heldChunks = pool.rows
    }

    /// Forgets the projected rows before an absolute position.
    ///
    /// - Parameter newStart: The absolute position the kept rows start at.
    private func dropRows(before newStart: Int) {
        guard newStart > carryStart else { return }
        let dropCount = newStart - carryStart
        projectedValues.dropLeading(dropCount)
        projectedGates.dropLeading(dropCount)
        carryStart = newStart
    }

    /// The absolute position the projected rows of the next call must start
    /// at.
    ///
    /// While the history is short enough for a rewind, that position is 0 and
    /// every projected row stays. Past that the cache keeps
    /// ``retainedChunkCount`` whole chunks and the run in progress.
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

    /// The array one slot of a serialized state stood for.
    ///
    /// An empty pool and no pool hold the same chunks, thus both read back as
    /// `nil`. So does the one-axis array that stands for no value.
    ///
    /// - Parameter array: The array a serialized state carried.
    /// - Returns: The array, or `nil` when it holds no value.
    private static func nullable(_ array: MLXArray) -> MLXArray? {
        isRowTensor(array) && array.size > 0 ? array : nil
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
            ? DeepSeekV4ChunkCache(
                configuration: configuration, layer: layer,
                poolDType: DeepSeekV4Indexer.scoreDType)
            : nil
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
    /// - Returns: The window, the pooled chunks and the projected rows of each
    ///   branch.
    public func innerState() -> [MLXArray] {
        local.innerState() + attentionChunks.innerState()
            + (indexerChunks?.innerState() ?? [])
    }

    /// The arrays the generation loop evaluates with each token: the buffers
    /// of the window, and the evaluation state of each branch.
    ///
    /// ``state`` writes a serialized form. It slices the window and the
    /// projected rows, and it makes a placeholder array for each slot that
    /// holds no value, thus it adds new operations on each call. This list
    /// holds only arrays the cache already holds. It still holds every lazy
    /// array the cache keeps, thus an evaluation at the end of prefill leaves
    /// no array that names the stream of the prefill thread.
    public var evaluationState: [MLXArray] {
        local.innerState() + attentionChunks.evaluationState
            + (indexerChunks?.evaluationState ?? [])
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
            let parts = Self.stateParts(newValue)
            // A window that has not written a key saves no array, and its setter needs two.
            if !parts.window.isEmpty {
                local.state = parts.window
            }
            attentionChunks.restore(state: parts.attention)
            indexerChunks?.restore(state: parts.indexer)
        }
    }

    /// The serialized meta state: the meta state of the window, then
    /// ``stateFormat``.
    ///
    /// The branches keep their own numbers in ``state`` rather than here, thus
    /// a reader may set the two in either order.
    public var metaState: [String] {
        get { local.metaState + [Self.stateFormat] }
        set {
            precondition(
                newValue.last == Self.stateFormat,
                "a DeepSeek-V4 meta state ends in the format \(Self.stateFormat)")
            local.metaState = Array(newValue.dropLast())
        }
    }

    /// The name of the format of ``state``, which ``metaState`` writes last.
    ///
    /// The first format kept the RAW block rows of an incomplete chunk in
    /// place of the projected rows, and wrote no format name, thus it does
    /// not restore. The second format,
    /// `DeepSeekV4Cache.projectedRows.v2`, kept the projected rows of each
    /// token (task `^cm8fq2q`) and an indexer pool in the activation dtype.
    /// The third format, `DeepSeekV4Cache.scoreDTypeIndexerPool.v3`, kept the
    /// indexer pool in ``DeepSeekV4Indexer/scoreDType`` (task `^atj552p`).
    /// This format keeps the projected rows up to the last ended chunk and
    /// the raw rows of the run in progress, in a slot of their own (task
    /// `^4n7mtxa`), thus a branch has five slots.
    static let stateFormat = "DeepSeekV4Cache.unprojectedRun.v4"

    /// The error message of a prompt cache whose meta state does not end in
    /// ``stateFormat``.
    static let earlierFormatMessage =
        "The prompt cache does not hold the DeepSeek-V4 cache format \(stateFormat). A prompt "
        + "cache of an earlier format cannot be restored: build the prompt cache again."

    /// True when this cache can go back to an earlier position.
    ///
    /// The window rewinds only before it has wrapped, and the branches rewind
    /// only while they still hold every projected row. The two ranges are the same
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

// MARK: - Restore into the cache of the model

extension DeepSeekV4Cache: PromptCacheRestorable {
    /// The class name that `savePromptCache` writes for this cache.
    public static let promptCacheClassName = "DeepSeekV4Cache"

    /// Checks a saved state and meta state before any setter reads them.
    ///
    /// The meta state must end in ``stateFormat``, the window part must fit a
    /// `RotatingKVCache`, each branch part must be a pooled-cache state, and a layer with no
    /// indexer accepts only the absent branch state.
    ///
    /// - Parameters:
    ///   - state: The saved arrays: the window arrays, then the slots of each branch.
    ///   - metaState: The saved meta state: the meta state of the window, then the format.
    /// - Throws: `KVCacheError` when the values are not a state of this cache.
    public func validatePromptCacheRestore(state: [MLXArray], metaState: [String]) throws {
        guard metaState.last == Self.stateFormat else {
            throw KVCacheError(message: Self.earlierFormatMessage)
        }
        guard state.count >= Self.stateSlotCount else {
            throw KVCacheError(
                message:
                    "Corrupt prompt cache: a DeepSeek-V4 state ends in \(Self.stateSlotCount) branch arrays, and this one holds \(state.count) arrays."
            )
        }
        let parts = Self.stateParts(state)
        try KVCacheSerializationRegistry.validate(
            state: parts.window, metaState: Array(metaState.dropLast()), for: local)
        try attentionChunks.validateRestore(state: parts.attention)
        if let indexerChunks {
            try indexerChunks.validateRestore(state: parts.indexer)
        } else if !DeepSeekV4ChunkCache.isAbsentState(parts.indexer) {
            throw KVCacheError(
                message:
                    "The prompt cache holds indexer data for a DeepSeek-V4 layer that has no indexer."
            )
        }
    }

    /// Writes a checked saved state and meta state into this cache.
    ///
    /// - Parameters:
    ///   - state: The saved arrays: the window arrays, then the slots of each branch.
    ///   - metaState: The saved meta state: the meta state of the window, then the format.
    public func restorePromptCache(state: [MLXArray], metaState: [String]) {
        self.state = state
        self.metaState = metaState
    }

    /// Splits a saved state into the window arrays and the slots of each branch.
    ///
    /// - Parameter state: The saved arrays. It holds at least the branch slots.
    /// - Returns: The window arrays, the attention branch slots and the indexer branch slots.
    private static func stateParts(
        _ state: [MLXArray]
    ) -> (window: [MLXArray], attention: [MLXArray], indexer: [MLXArray]) {
        let split = state.count - stateSlotCount
        let slots = DeepSeekV4ChunkCache.stateSlotCount
        return (
            Array(state[0 ..< split]), Array(state[split ..< (split + slots)]),
            Array(state[(split + slots)...])
        )
    }
}
