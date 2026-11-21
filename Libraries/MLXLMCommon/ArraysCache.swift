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
        (cache + [leftPadding, lengths]).compactMap { $0 }
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

    package func copyForCheckpoint() -> any KVCache {
        let new = copy() as! ArraysCache
        new.cache = cache.map { $0.map(materializedCacheState) }
        new.leftPadding = leftPadding.map(materializedCacheState)
        new.lengths = lengths.map(materializedCacheState)
        return new
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
    /// its tokens but the last, which no rollback selects, from the position
    /// before the pass.
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

/// Compact convolution cache with a bounded token history for safe rewinds.
///
/// Hybrid models normally need only the final `stateLength` projected inputs,
/// but retaining a small rolling history makes speculative rejection and short
/// chat-template rewinds possible without keeping the full prompt activation
/// history. Trims larger than the retained window fail atomically and callers
/// can rebuild from the common prefix.
public final class RewindableConvolutionCache: MambaCache {
    public override class var serializedClassName: String? { "RewindableConvolutionCache" }

    /// The configuration and contents of a saved `RewindableConvolutionCache`.
    fileprivate struct SavedLayout {
        let offset: Int
        let stateLength: Int
        let rollbackCapacity: Int
        /// The slot of each saved array, in the order of the arrays.
        let presentSlots: [Int]
    }

    /// Checks saved values and reads the layout they describe.
    ///
    /// - Parameters:
    ///   - state: The saved arrays.
    ///   - metaState: The saved meta state.
    /// - Returns: The layout.
    /// - Throws: ``KVCacheError`` when the values are not a saved `RewindableConvolutionCache`.
    fileprivate static func savedLayout(
        state: [MLXArray], metaState: [String]
    ) throws -> SavedLayout {
        guard metaState.count == 5,
            let offset = Int(metaState[0]),
            let stateLength = Int(metaState[1]),
            let rollbackCapacity = Int(metaState[2]),
            let slotCount = Int(metaState[3]),
            slotCount == 2,
            offset >= 0,
            stateLength > 0,
            rollbackCapacity > 0,
            state.allSatisfy({ $0.ndim == 3 })
        else {
            throw KVCacheError(
                message:
                    "Corrupt prompt cache: invalid RewindableConvolutionCache state or metadata.")
        }
        let presentSlots =
            metaState[4].isEmpty
            ? [] : metaState[4].split(separator: ",").compactMap { Int($0) }
        guard presentSlots.count == state.count,
            presentSlots.allSatisfy({ (0 ..< slotCount).contains($0) })
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid RewindableConvolutionCache slots.")
        }
        return SavedLayout(
            offset: offset, stateLength: stateLength, rollbackCapacity: rollbackCapacity,
            presentSlots: presentSlots)
    }

    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        let layout = try savedLayout(state: state, metaState: metaState)
        let cache = RewindableConvolutionCache(
            stateLength: layout.stateLength, rollbackCapacity: layout.rollbackCapacity)
        cache.write(state: state, layout: layout)
        return cache
    }

    /// Writes saved arrays into their slots and moves to the saved offset.
    fileprivate func write(state: [MLXArray], layout: SavedLayout) {
        for (arrayIndex, slotIndex) in layout.presentSlots.enumerated() {
            self[slotIndex] = state[arrayIndex]
        }
        offset = layout.offset
    }

    public let stateLength: Int
    public let rollbackCapacity: Int

    public init(stateLength: Int, rollbackCapacity: Int = 64) {
        precondition(stateLength > 0)
        precondition(rollbackCapacity > 0)
        self.stateLength = stateLength
        self.rollbackCapacity = rollbackCapacity
        super.init()
    }

    /// Commits one convolution call and retains only the bounded rewind window.
    public func record(
        input: MLXArray,
        initialState: MLXArray,
        currentState: MLXArray
    ) {
        let timeline: MLXArray
        if let history = self[1] {
            timeline = concatenated([history, input], axis: 1)
        } else {
            timeline = concatenated([initialState, input], axis: 1)
        }

        let retainedLength = min(timeline.dim(1), stateLength + rollbackCapacity)
        self[0] = currentState
        self[1] = contiguous(timeline[0..., (timeline.dim(1) - retainedLength)..., 0...])
        // `advance` moves `offset` too in this fork.
        advance(input.dim(1))
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        positions >= 0 && positions <= rollbackCapacity
    }

    public override var maxTrimCount: Int {
        guard let history = self[1] else { return 0 }
        return max(0, min(offset, history.dim(1) - stateLength))
    }

    func canRewind(_ positions: Int) -> Bool {
        guard positions >= 0 else { return false }
        if positions == 0 { return true }
        guard let history = self[1] else { return false }
        return positions <= min(offset, history.dim(1) - stateLength)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, canRewind(n), let history = self[1] else { return 0 }

        let retained = history[0..., ..<(history.dim(1) - n), 0...]
        self[0] = contiguous(retained[0..., (retained.dim(1) - stateLength)..., 0...])
        self[1] = contiguous(retained)
        offset -= n
        return n
    }

    public override func copy() -> any KVCache {
        let copy = RewindableConvolutionCache(
            stateLength: stateLength, rollbackCapacity: rollbackCapacity)
        copyContents(to: copy)
        return copy
    }

    /// Stable prompt-cache metadata. Transient batch lengths are finalized
    /// before serialization and intentionally excluded.
    public override var metaState: [String] {
        get {
            [
                "\(offset)", "\(stateLength)", "\(rollbackCapacity)",
                "\(slotCount)", presentSlotIndices.map(String.init).joined(separator: ","),
            ]
        }
        set {
            assertionFailure(
                """
                RewindableConvolutionCache.metaState cannot be set directly; \
                restore through prompt-cache loading.
                """)
        }
    }
}

/// A model makes its convolution caches as templates for a restore, thus the saved values must
/// have the configuration of the template.
extension RewindableConvolutionCache: PromptCacheRestorable {
    public func validatePromptCacheRestore(state: [MLXArray], metaState: [String]) throws {
        let layout = try Self.savedLayout(state: state, metaState: metaState)
        guard layout.stateLength == stateLength, layout.rollbackCapacity == rollbackCapacity
        else {
            throw KVCacheError(
                message:
                    "RewindableConvolutionCache restore: saved state length \(layout.stateLength) "
                    + "and rollback capacity \(layout.rollbackCapacity) do not match the model's "
                    + "\(stateLength) and \(rollbackCapacity).")
        }
    }

    public func restorePromptCache(state: [MLXArray], metaState: [String]) {
        guard let layout = try? Self.savedLayout(state: state, metaState: metaState) else {
            preconditionFailure("restorePromptCache runs only after validatePromptCacheRestore")
        }
        write(state: state, layout: layout)
    }
}

/// Convolution states that rewind the rejected tokens of a speculative verify pass.
///
/// Each slot holds the last inputs of one causal convolution, as an ``ArraysCache`` does. A
/// verify pass of a few tokens also hands the cache its input window for each slot: the state
/// before the pass, then the input of each token. A rewind of `n` tokens then takes the state
/// after the kept tokens from those windows, without replaying the model.
///
/// The windows are transient. The next forward pass drops them (``advance(_:)``), and they are
/// never part of the state, a copy, a prompt cache checkpoint or a saved prompt cache: the cache
/// saves and restores as an ``ArraysCache``, under its name.
public final class SpeculativeConvolutionCache: ArraysCache {

    /// The windows of the last verify pass and the tokens of the pass the cache holds.
    private struct VerifyWindows {
        /// One window per slot: the state rows before the pass, then one row per token.
        var windows: [MLXArray]
        /// The tokens of the pass.
        var tokenCount: Int
        /// The tokens of the pass the cache holds: all of them after the pass, the kept tokens
        /// after a rewind.
        var keptTokenCount: Int
    }

    /// The most tokens of a verify pass whose windows the cache keeps.
    public let maximumVerifyPositions: Int

    /// The windows of the last verify pass, or `nil` when the cache keeps none.
    private var verify: VerifyWindows?

    /// Creates the cache.
    ///
    /// - Parameters:
    ///   - size: the slots, one for each convolution of the layer.
    ///   - maximumVerifyPositions: the most tokens of a verify pass whose windows the cache
    ///     keeps, thus the deepest rewind.
    public init(size: Int, maximumVerifyPositions: Int) {
        precondition(maximumVerifyPositions > 0)
        self.maximumVerifyPositions = maximumVerifyPositions
        super.init(size: size)
    }

    /// Keeps the input windows of the forward pass that just moved the cache, so that a rewind
    /// can take the state after any of its tokens.
    ///
    /// Call after the pass set each slot and called ``advance(_:)``. A window holds the state
    /// rows before the pass, then one row per token of the pass, along axis 1.
    ///
    /// - Parameter windows: one window for each slot, in slot order.
    public func keepVerifyWindows(_ windows: [MLXArray]) {
        guard let window = windows.first, windows.count == slotCount else {
            verify = nil
            return
        }
        let stateLength = self[0]?.dim(1) ?? 0
        let tokens = window.dim(1) - stateLength
        verify =
            (1 ... maximumVerifyPositions).contains(tokens)
            ? VerifyWindows(windows: windows, tokenCount: tokens, keptTokenCount: tokens) : nil
    }

    /// Drops the windows of the last verify pass.
    public func discardVerifyWindows() {
        verify = nil
    }

    /// Moves the cache past a forward pass and drops the windows of the previous pass.
    ///
    /// - Parameter tokenCount: the tokens the layer fed.
    public override func advance(_ tokenCount: Int) {
        verify = nil
        super.advance(tokenCount)
    }

    /// Whether a verify pass of `positions` tokens keeps the windows of its every token.
    public override func isTrimmable(after positions: Int) -> Bool {
        positions >= 0 && positions <= maximumVerifyPositions
    }

    /// The tokens of the last verify pass the cache holds, or zero without windows.
    public override var maxTrimCount: Int {
        verify?.keptTokenCount ?? 0
    }

    /// Rewinds the newest `n` tokens of the last verify pass: each slot takes the state after the
    /// kept tokens from its window. The windows stay, so that a later rewind of the same pass
    /// rewinds further.
    ///
    /// - Parameter n: the tokens to remove.
    /// - Returns: `n`, or zero, leaving the cache as it was, when the windows do not hold them.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, var verify, n <= verify.keptTokenCount else { return 0 }
        let kept = verify.keptTokenCount - n
        for (slot, window) in verify.windows.enumerated() {
            let stateLength = window.dim(1) - verify.tokenCount
            self[slot] = contiguous(window[0..., kept ..< (kept + stateLength), 0...])
        }
        verify.keptTokenCount = kept
        self.verify = verify
        offset -= n
        return n
    }

    /// A copy of the states, without the windows of the last verify pass.
    public override func copy() -> any KVCache {
        let new = SpeculativeConvolutionCache(
            size: slotCount, maximumVerifyPositions: maximumVerifyPositions)
        copyContents(to: new)
        return new
    }
}
