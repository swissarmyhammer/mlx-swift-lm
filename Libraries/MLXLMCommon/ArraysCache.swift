// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "ArraysCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "ArraysCache", state: state, metaState: metaState)
    }

    fileprivate var cache: [MLXArray?]
    internal var leftPadding: MLXArray?
    internal var lengths: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    public override func copy() -> any KVCache {
        let new = ArraysCache(size: cache.count)
        copyContents(to: new)
        return new
    }

    internal func copyContents(to new: ArraysCache) {
        new.cache = cache.map { $0?[.ellipsis] }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        new.lengths = self.lengths
    }

    internal var batchSize: Int {
        cache.lazy.compactMap { $0?.dim(0) }.first ?? leftPadding?.size ?? lengths?.size ?? 1
    }

    /// In-place filter to keep just the given indices in the cache
    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = leftPadding?[batchIndices]
        lengths = lengths?[batchIndices]
    }

    /// In-place extend this cache with the other cache
    public func extend(other: ArraysCache) {
        let aBatch = batchSize
        let bBatch = other.batchSize

        func concatenate(_ a: MLXArray?, _ b: MLXArray?) -> MLXArray? {
            guard let example = a ?? b else {
                return nil
            }

            let suffixShape = Array(example.shape.dropFirst())
            let dtype = example.dtype
            let lhs = a ?? MLXArray.zeros([aBatch] + suffixShape, dtype: dtype)
            let rhs = b ?? MLXArray.zeros([bBatch] + suffixShape, dtype: dtype)
            return MLX.concatenated([lhs, rhs])
        }

        cache = zip(cache, other.cache).map { c, o in
            concatenate(c, o)
        }
        leftPadding = concatenate(leftPadding, other.leftPadding)
        lengths = concatenate(lengths, other.lengths)
    }

    public override func prepare(lengths: [Int]?) {
        self.lengths = lengths.map { MLXArray($0) }
    }

    public override func prepare(lengths: MLXArray?) {
        self.lengths = lengths
    }

    public override func finalize() {
        lengths = nil
        leftPadding = nil
    }

    /// Moves this cache past `N` tokens that the layer fed through it.
    ///
    /// This decreases `lengths` and `leftPadding` by `N`, as `ArraysCache.advance` of
    /// mlx-lm does, AND increases `offset` by `N`. The offset is the number of tokens that
    /// the cache holds. The prompt cache keeps a cache only when its offset is the length of
    /// the token ledger, thus a recurrent cache whose offset stays at 0 starts each round
    /// cold. Upstream moves only `lengths` and `leftPadding`, because upstream has no such
    /// check. A layer calls this one time for each forward pass, and it must not also move
    /// `offset` itself.
    ///
    /// - Parameter N: the number of tokens the layer fed.
    public func advance(_ N: Int) {
        if let currentLengths = lengths {
            lengths = currentLengths - N
        }
        if let currentLeftPadding = leftPadding {
            leftPadding = currentLeftPadding - N
        }
        offset += N
    }

    public var currentLengths: MLXArray? {
        lengths
    }

    internal var leftPaddingValues: [Int]? {
        guard let leftPadding else { return nil }
        return leftPadding.asArray(Int.self)
    }

    internal var lengthsValues: [Int]? {
        guard let lengths else { return nil }
        return lengths.asArray(Int.self)
    }

    internal var presentSlotIndices: [Int] {
        cache.enumerated().compactMap { (i, v) in v != nil ? i : nil }
    }

    internal var slotCount: Int { cache.count }

    /// Create attention mask based on left padding or prepared sequence lengths
    public func makeMask(N: Int) -> MLXArray? {
        let positions = MLXArray(0 ..< N)
        if let leftPadding {
            return positions .>= leftPadding[0..., .newAxis]
        } else if let lengths {
            return positions .< lengths[0..., .newAxis]
        } else {
            return nil
        }
    }

    // MARK: - Serialization

    /// metaState format: [slotCount, presentSlots, leftPadding?, lengths?]
    /// Legacy format (BaseKVCache default): [""]
    public override var metaState: [String] {
        get {
            let leftPaddingState = Self.serializeMetadata(leftPadding)
            let lengthsState = Self.serializeMetadata(lengths)
            var result = [
                "\(cache.count)",
                presentSlotIndices.map(String.init).joined(separator: ","),
            ]
            if let leftPaddingState {
                result.append(leftPaddingState)
            } else if lengthsState != nil {
                result.append("")
            }
            if let lengthsState {
                result.append(lengthsState)
            }
            return result
        }
        set {
            assertionFailure(
                "ArraysCache.metaState should not be set directly. Use restoreFromMetaState() instead"
            )
        }
    }

    /// Restore from saved metaState + state arrays. Handles both new (slot-aware) and legacy formats.
    internal func restoreFromMetaState(state: [MLXArray], savedMetaState: [String]) {
        // Detect new format: first element parses as int (slotCount), second element is present slots
        if savedMetaState.count >= SavedValueCounts.arraysMinimumMetaStateCount,
            let slotCount = Int(savedMetaState[SavedMetaStateIndex.arraysSlotCount])
        {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = Self.metadataArray(
                savedMetaState, at: SavedMetaStateIndex.arraysLeftPadding)
            self.lengths = Self.metadataArray(savedMetaState, at: SavedMetaStateIndex.arraysLengths)
        } else {
            // Legacy: best-effort, state is compacted
            self.cache = state.map { $0 as MLXArray? }
        }
    }

    private static func serializeMetadata(_ array: MLXArray?) -> String? {
        array?.asArray(Int.self).map(String.init).joined(separator: ",")
    }

    private static func metadataArray(_ state: [String], at index: Int) -> MLXArray? {
        guard state.indices.contains(index), !state[index].isEmpty else { return nil }
        return MLXArray(state[index].split(separator: ",").compactMap { Int($0) })
    }
}

/// Simple cache for Mamba-style state space models
public class MambaCache: ArraysCache {
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "MambaCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "MambaCache", state: state, metaState: metaState)
    }

    /// The position of the cache: its offset and its batch masks.
    private struct Position {
        var offset: Int
        var leftPadding: MLXArray?
        var lengths: MLXArray?

        /// The position `tokenCount` tokens further on.
        func advanced(by tokenCount: Int) -> Position {
            Position(
                offset: offset + tokenCount,
                leftPadding: leftPadding.map { $0 - tokenCount },
                lengths: lengths.map { $0 - tokenCount })
        }
    }

    /// The snapshot slots of the last verify pass: the states after each of
    /// its tokens, from the position before the pass.
    private struct SpeculativeSlots {
        var snapshots: GatedDeltaDecodeKernel.Snapshots
        /// The position before the verify pass.
        var start: Position
        /// The tokens of the pass the cache holds: all of them after the pass,
        /// the kept tokens after a rollback. A rollback selects among the slots
        /// of these tokens only.
        var keptTokenCount: Int
    }

    /// The transient speculative data of the last verify pass: what a
    /// rollback of that pass restores. It is never part of the state, a copy,
    /// a prompt cache checkpoint, a saved prompt cache or the byte count.
    private var speculative: SpeculativeSlots?

    /// The position of the cache now.
    private var position: Position {
        get { Position(offset: offset, leftPadding: leftPadding, lengths: lengths) }
        set {
            offset = newValue.offset
            leftPadding = newValue.leftPadding
            lengths = newValue.lengths
        }
    }

    /// The slots of a Mamba layer: the convolution state and the recurrent state.
    private static let stateSlotCount = 2

    public init(leftPadding: [Int]? = nil) {
        super.init(size: Self.stateSlotCount, leftPadding: leftPadding)
    }

    /// Keep the snapshot slots of a verify pass, as transient speculative
    /// data: a rollback of `r` tokens of a pass of `T` tokens selects the slot
    /// after token `T - r` (``selectSpeculativeSlot(rollingBack:)``).
    ///
    /// The caller saves BEFORE it moves this cache past the whole input, thus
    /// the slots start at the current position. The slots replace the slots
    /// of an earlier verify pass.
    ///
    /// - Parameter snapshots: The states after each token of the verify pass.
    package func saveSpeculativeSnapshots(_ snapshots: GatedDeltaDecodeKernel.Snapshots) {
        speculative = SpeculativeSlots(
            snapshots: snapshots, start: position, keptTokenCount: snapshots.tokenCount)
    }

    /// The number of snapshot slots of the last verify pass this cache can
    /// select: the tokens of the pass it holds, or zero when it keeps no slots.
    package var speculativeSlotCount: Int {
        speculative?.keptTokenCount ?? 0
    }

    /// Roll the last verify pass back by `tokenCount` tokens: the cache takes
    /// the states after the accepted tokens from their slot.
    ///
    /// The cache keeps the slots of the accepted tokens, as it keeps the slots
    /// of a pass that accepted every token, so that a later rollback of the
    /// same pass rewinds into them: a stream that stops before it returns the
    /// accepted tokens rewinds them again. The slots go with
    /// ``discardSpeculativeCheckpoint()`` or with the slots of the next pass,
    /// and at once when the rollback keeps the bonus token of the pass only.
    ///
    /// Each selected state is a separate array, a copy of its slot rather than
    /// a view of it, so that the `(tokens, …)` snapshot buffer goes with the
    /// slots and not with the states: ``rewindSpeculativePromptCache(_:numTokens:)``
    /// evaluates the copies of every layer together.
    ///
    /// - Parameter tokenCount: The rejected tokens, from 1 to the kept tokens
    ///   of the pass less one.
    /// - Returns: False when the cache keeps no slots, or no slot for that
    ///   rollback; the cache is then left as it was.
    @discardableResult
    package func selectSpeculativeSlot(rollingBack tokenCount: Int) -> Bool {
        guard let slots = speculative, (1 ..< slots.keptTokenCount).contains(tokenCount)
        else { return false }
        let accepted = slots.keptTokenCount - tokenCount
        cache = [
            contiguous(slots.snapshots.convState(afterToken: accepted)),
            contiguous(slots.snapshots.recurrentState(afterToken: accepted)),
        ]
        position = slots.start.advanced(by: accepted)
        // No rollback rejects the head token of a pass, thus a pass that holds only that token
        // has no slot left to roll back into, and gives its snapshot buffer back at once.
        speculative =
            accepted > Self.unrejectableHeadTokenCount
            ? SpeculativeSlots(
                snapshots: slots.snapshots, start: slots.start, keptTokenCount: accepted)
            : nil
        return true
    }

    /// The tokens at the head of a verify pass that no rollback rejects: its bonus token.
    private static let unrejectableHeadTokenCount = 1

    /// Release the transient speculative data: the snapshot slots.
    package func discardSpeculativeCheckpoint() {
        speculative = nil
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        copyContents(to: new)
        return new
    }
}
