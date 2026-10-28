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

    /// The states of the cache after the checkpoint token of a verify pass.
    private struct SpeculativeCheckpoint {
        var state: [MLXArray?]
        var position: Position
    }

    /// The snapshot slots of the last verify pass: the states after each of
    /// its tokens, from the position before the pass.
    private struct SpeculativeSlots {
        var snapshots: GatedDeltaDecodeKernel.Snapshots
        /// The token after which ``restoreSpeculativeCheckpoint()`` stands.
        var checkpointAfter: Int
        /// The position before the verify pass.
        var start: Position
    }

    /// The transient speculative data of the last verify pass: what a
    /// rollback of that pass restores. It is never part of the state, a copy,
    /// a prompt cache checkpoint, a saved prompt cache or the byte count.
    private enum Speculative {
        /// The states after the checkpoint token, from the op path.
        case checkpoint(SpeculativeCheckpoint)
        /// The states after each token, from the gated delta decode kernel.
        case slots(SpeculativeSlots)
    }

    private var speculative: Speculative?

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

    /// Save the recurrent state at the last unconditionally committed token
    /// inside a speculative verification pass.
    ///
    /// The caller saves BEFORE it moves this cache past the whole input, thus
    /// the checkpoint's position is the current position plus `tokenCount`.
    /// The checkpoint replaces the slots of an earlier verify pass.
    package func saveSpeculativeCheckpoint(
        convState: MLXArray,
        recurrentState: MLXArray,
        advancedBy tokenCount: Int
    ) {
        speculative = .checkpoint(
            SpeculativeCheckpoint(
                state: [convState, recurrentState], position: position.advanced(by: tokenCount)))
    }

    /// Keep the snapshot slots of a verify pass of the gated delta decode
    /// kernel, as transient speculative data: a rollback of `r` tokens selects
    /// the slot after token `T - r` (``selectSpeculativeSlot(rollingBack:)``),
    /// and ``restoreSpeculativeCheckpoint()`` selects the slot after token
    /// `checkpointAfter`.
    ///
    /// The caller saves BEFORE it moves this cache past the whole input, as
    /// for ``saveSpeculativeCheckpoint(convState:recurrentState:advancedBy:)``.
    /// The slots replace the checkpoint or the slots of an earlier verify pass.
    ///
    /// - Parameters:
    ///   - snapshots: The states after each token of the verify pass.
    ///   - tokenCount: The token after which the checkpoint stands, from 1
    ///     to the tokens of the pass less one.
    package func saveSpeculativeSnapshots(
        _ snapshots: GatedDeltaDecodeKernel.Snapshots, checkpointAfter tokenCount: Int
    ) {
        precondition(
            (1 ..< snapshots.tokenCount).contains(tokenCount),
            "a verify of \(snapshots.tokenCount) tokens has no checkpoint after token \(tokenCount)"
        )
        speculative = .slots(
            SpeculativeSlots(snapshots: snapshots, checkpointAfter: tokenCount, start: position))
    }

    /// Save what a verify pass of a gated delta net layer keeps for a
    /// rollback: the checkpoint of the op path or the slots of the kernel.
    ///
    /// - Parameters:
    ///   - states: The states the pass keeps.
    ///   - tokenCount: The token after which the checkpoint stands.
    package func saveSpeculativeStates(
        _ states: GatedDeltaSpeculativeStates, checkpointAfter tokenCount: Int
    ) {
        switch states {
        case .checkpoint(let conv, let recurrent):
            saveSpeculativeCheckpoint(
                convState: conv, recurrentState: recurrent, advancedBy: tokenCount)
        case .snapshots(let snapshots):
            saveSpeculativeSnapshots(snapshots, checkpointAfter: tokenCount)
        }
    }

    /// The number of snapshot slots of the last verify pass this cache keeps:
    /// zero when it keeps none.
    package var speculativeSlotCount: Int {
        guard case .slots(let slots) = speculative else { return 0 }
        return slots.snapshots.tokenCount
    }

    package var hasSpeculativeCheckpoint: Bool {
        speculative != nil
    }

    /// Roll the last verify pass back by `tokenCount` tokens: the cache takes
    /// the states after the accepted tokens from their slot, and releases the
    /// slots.
    ///
    /// Each selected state is a separate array, a copy of its slot rather than
    /// a view of it, so that the `(tokens, …)` snapshot buffer goes once the
    /// copy is evaluated: ``rewindSpeculativePromptCache(_:numTokens:)``
    /// evaluates the copies of every layer together.
    ///
    /// - Parameter tokenCount: The rejected tokens, from 1 to the tokens of
    ///   the pass less one.
    /// - Returns: False when the cache keeps no slots, or no slot for that
    ///   rollback; the cache is then left as it was.
    @discardableResult
    package func selectSpeculativeSlot(rollingBack tokenCount: Int) -> Bool {
        guard case .slots(let slots) = speculative,
            (1 ..< slots.snapshots.tokenCount).contains(tokenCount)
        else { return false }
        let accepted = slots.snapshots.tokenCount - tokenCount
        cache = [
            contiguous(slots.snapshots.convState(afterToken: accepted)),
            contiguous(slots.snapshots.recurrentState(afterToken: accepted)),
        ]
        position = slots.start.advanced(by: accepted)
        speculative = nil
        return true
    }

    @discardableResult
    package func restoreSpeculativeCheckpoint() -> Bool {
        switch speculative {
        case .checkpoint(let checkpoint):
            cache = checkpoint.state
            position = checkpoint.position
            speculative = nil
            return true
        case .slots(let slots):
            return selectSpeculativeSlot(
                rollingBack: slots.snapshots.tokenCount - slots.checkpointAfter)
        case nil:
            return false
        }
    }

    /// Release the transient speculative data: the checkpoint or the slots.
    package func discardSpeculativeCheckpoint() {
        speculative = nil
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        copyContents(to: new)
        return new
    }
}
