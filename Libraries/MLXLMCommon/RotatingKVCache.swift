// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "RotatingKVCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "RotatingKVCache", state: state, metaState: metaState)
    }

    package enum CapacityOrigin: String {
        case modelNative
        case requested
    }

    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0

    /// Additional historical rows retained for exact prefix rewinds.
    /// The attention window remains ``maxSize``. Zero preserves the usual memory bound.
    public private(set) var rewindCapacity: Int

    private var storageCapacity: Int { maxCacheSize + rewindCapacity }

    /// In ring layout all rows are live, with `idx...` preceding `keep ..< idx`.
    /// At the end of the buffer the ring is already in temporal order. Otherwise,
    /// temporal layout holds only the first `idx` rows, even when `offset` is larger.
    /// `nil` defers legacy inference until arrays and metadata have both been restored.
    /// Every write or trim resolves it before changing either buffers or counters.
    private var wrappedFlag: Bool? = false

    private var wrapped: Bool {
        wrappedFlag ?? (idx < (keys?.dim(HeadRowsAxis.row) ?? 0) && offset > idx)
    }

    /// Model-native sliding-window caches deliberately keep their architectural
    /// window and do not participate in requested-capacity validation.
    package var capacityOrigin = CapacityOrigin.modelNative

    package var preservedPrefixTokens: Int { keep }

    public override var maxSize: Int? { maxCacheSize }

    public override var evictedTokenCount: Int {
        let retained = wrapped ? (keys?.dim(HeadRowsAxis.row) ?? 0) : idx
        return Swift.max(0, offset - retained)
    }

    /// Number of leading tokens that are never rotated out of the window.
    var keepCount: Int { keep }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256, rewindCapacity: Int = 0) {
        precondition(rewindCapacity >= 0 && maxSize <= Int.max - rewindCapacity)
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        self.rewindCapacity = rewindCapacity
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: HeadRowsAxis.row)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused.
        // `idx` bounds the live rows: after a post-wrap trim the logical `offset`
        // exceeds the rows actually held, so the layout question is `wrapped`,
        // never an `idx`/`offset` comparison.
        if idx == array.dim(HeadRowsAxis.row) {
            return array
        } else if wrapped {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: HeadRowsAxis.row)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    /// The trailing `tail` cache entries in chronological order, without mutating the cache.
    ///
    /// `update(keys:values:)` is the only other way to read a rotating cache's contents, and it
    /// necessarily writes. This is the read-only counterpart: `keys`, `values`, `idx` and
    /// `offset` are all left exactly as they were, so a caller can present the ring's history
    /// alongside K/V it has not committed yet.
    ///
    /// The view slices only the requested rows, preserving the pinned `keep` prefix.
    /// Contiguous views need no copy; views crossing the ring boundary concatenate only
    /// the selected segments. Historical reserve rows are never linearized for attention.
    ///
    /// The pinned `keep` prefix is a floor, not just a splice point: a `tail` below it still
    /// comes back, because those entries are not evictable and a view that dropped them would be
    /// a context this ring can never present. With `keep == 0` -- every sliding-window model --
    /// the floor is zero and the length is exactly `min(tail, count)`.
    ///
    /// - Parameter tail: Requested number of trailing entries. Clamped to what the cache holds;
    ///   a negative value is read as zero.
    /// A configured rewind reserve is excluded from this attention view.
    ///
    /// - Returns: `(keys, values)` shaped `[B, kvHeads, n, headDim]`, or `nil` before the
    ///   first write. The length is clamped to available attention history and floored
    ///   at the retained pinned prefix length.
    package func logicalView(tail: Int) -> (MLXArray, MLXArray)? {
        retainedView(tail: rewindCapacity > 0 ? Swift.min(tail, maxCacheSize) : tail)
    }

    private func retainedView(tail: Int) -> (MLXArray, MLXArray)? {
        guard let keys = self.keys, let values = self.values else { return nil }

        let available = wrapped ? keys.dim(HeadRowsAxis.row) : idx
        let prefix = Swift.min(keep, available)
        let requested = Swift.min(Swift.max(tail, 0), available)
        let bound = Swift.max(requested, prefix)

        if !wrapped || idx == available {
            if prefix == 0 || bound == available {
                return (
                    keys[.ellipsis, (available - bound) ..< available, 0...],
                    values[.ellipsis, (available - bound) ..< available, 0...]
                )
            }
        } else if prefix == 0 && bound <= idx {
            return (
                keys[.ellipsis, (idx - bound) ..< idx, 0...],
                values[.ellipsis, (idx - bound) ..< idx, 0...]
            )
        }

        let recent = Swift.min(bound - prefix, idx - prefix)
        let older = bound - prefix - recent
        let ranges = [0 ..< prefix, (available - older) ..< available, (idx - recent) ..< idx]
            .filter { !$0.isEmpty }
        func view(_ array: MLXArray) -> MLXArray {
            let parts = ranges.map { array[.ellipsis, $0, 0...] }
            return parts.count == 1 ? parts[0] : concatenated(parts, axis: HeadRowsAxis.row)
        }
        return (view(keys), view(values))
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(HeadRowsAxis.row)

            // Prefill temporarily retains storageCapacity + S - 1 rows so each
            // query has its attention window and the rewind reserve survives.
            let trimSize = idx - storageCapacity + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(HeadRowsAxis.row)
        idx = self.keys!.dim(HeadRowsAxis.row)
        wrappedFlag = false

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(HeadRowsAxis.row)
        let kHeadDim = keys.dim(HeadRowsAxis.width)
        let vHeadDim = values.dim(HeadRowsAxis.width)

        // May not have hit the max size yet, so potentially keep growing the cache.
        // Fill is tracked by `idx`, not `offset`: after a post-wrap trim the logical
        // offset exceeds the rows actually held, and growth must resume from the rows.
        let filled = self.keys?.dim(HeadRowsAxis.row) ?? 0
        if self.keys == nil || (!wrapped && idx >= filled && filled < storageCapacity) {
            let newSize = min(step, storageCapacity - filled)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: HeadRowsAxis.row)
                self.values = concatenated([currentValues, newV], axis: HeadRowsAxis.row)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        // Trim if needed
        let trimSize = self.keys!.dim(HeadRowsAxis.row) - storageCapacity
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = storageCapacity
        }

        // Rotate if we've hit the end
        if idx == storageCapacity {
            idx = keep
            wrappedFlag = true
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice: live rows are bounded by `idx` in
        // temporal layout, while a wrapped ring is fully live.
        if !wrapped, idx < self.keys!.dim(HeadRowsAxis.row) {
            return (
                self.keys![.ellipsis, ..<idx, 0...],
                self.values![.ellipsis, ..<idx, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        wrappedFlag = wrapped
        let count = keys.dim(HeadRowsAxis.row)
        if rewindCapacity > 0, count == 0 {
            return logicalView(tail: maxCacheSize) ?? (keys, values)
        }
        let live = wrapped ? (self.keys?.dim(HeadRowsAxis.row) ?? 0) : idx
        let result =
            if count == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        if rewindCapacity > 0 {
            // Historical reserve rows never participate in attention.
            return retainedView(tail: Swift.min(live, maxCacheSize - 1) + count)!
        }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if !wrapped, idx < keys.dim(HeadRowsAxis.row) {
                return [
                    keys[.ellipsis, ..<idx, 0...],
                    values[.ellipsis, ..<idx, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == SavedValueCounts.keysAndValues else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            var metadata = [
                String(keep), String(maxCacheSize), String(step), String(offset), String(idx),
                capacityOrigin.rawValue, String(wrapped),
            ]
            if rewindCapacity > 0 { metadata.append(String(rewindCapacity)) }
            return metadata
        }
        set {
            guard SavedValueCounts.rotatingMetaStates.contains(newValue.count) else {
                fatalError("RotatingKVCache metaState must have 5 to 8 values")
            }
            guard let keepVal = Int(newValue[SavedMetaStateIndex.rotatingKeep]),
                let stepVal = Int(newValue[SavedMetaStateIndex.rotatingStep]),
                let offsetVal = Int(newValue[SavedMetaStateIndex.rotatingOffset]),
                let idxVal = Int(newValue[SavedMetaStateIndex.rotatingIndex])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            let maxSize = newValue[SavedMetaStateIndex.rotatingMaxSize]
            if maxSize == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(maxSize) else {
                fatalError("Failed to convert maxCacheSize '\(maxSize)' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            if newValue.count > SavedMetaStateIndex.rotatingRewindCapacity {
                guard let reserve = Int(newValue[SavedMetaStateIndex.rotatingRewindCapacity]),
                    reserve >= 0,
                    maxSizeVal <= Int.max - reserve
                else { fatalError("Invalid RotatingKVCache rewind capacity") }
                self.rewindCapacity = reserve
            } else {
                self.rewindCapacity = 0
            }
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
            if newValue.count > SavedMetaStateIndex.rotatingCapacityOrigin {
                let originValue = newValue[SavedMetaStateIndex.rotatingCapacityOrigin]
                guard let origin = CapacityOrigin(rawValue: originValue) else {
                    fatalError("Invalid RotatingKVCache capacity origin '\(originValue)'")
                }
                self.capacityOrigin = origin
            } else {
                self.capacityOrigin = .modelNative
            }
            if newValue.count > SavedMetaStateIndex.rotatingWrapped {
                let wrappedText = newValue[SavedMetaStateIndex.rotatingWrapped]
                guard let wrappedValue = Bool(wrappedText) else {
                    fatalError("Invalid RotatingKVCache wrapped flag '\(wrappedText)'")
                }
                self.wrappedFlag = wrappedValue
            } else {
                // Either setter may run first. Defer inference until the restored
                // arrays are available, then freeze the layout before any mutation.
                self.wrappedFlag = nil
            }
        }
    }

    public override var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    public override var maxTrimCount: Int {
        let live = wrapped ? (keys?.dim(HeadRowsAxis.row) ?? 0) : idx
        guard live <= offset else { return 0 }
        return live == offset ? offset : Swift.max(0, live - maxCacheSize)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        // This conservative predicate requires the whole prefix. Counted rewinds
        // can also use the limited history described by `maxTrimCount`.
        let live = wrapped ? (keys?.dim(HeadRowsAxis.row) ?? 0) : idx
        guard keys == nil || live == offset else { return false }
        return positions >= 0 && offset >= 0 && offset < storageCapacity
            && positions < storageCapacity - offset
    }

    /// Rewind the newest `n` positions.
    ///
    /// Before the ring wraps this is exact bookkeeping. After it wraps, the ring is
    /// linearized and the newest rows are cut: the cache stays consistent and the
    /// logical offset rewinds, but rows the rewound writes overwrote at the old edge
    /// of the window cannot come back, so the window is up to `n` rows short until
    /// it refills. An exact rewind requires `n <= maxTrimCount`; a configured
    /// reserve extends that limit beyond rotation.
    /// Once older rows have been evicted, trimming stops at the pinned `keep` prefix.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, let keys, let values else { return 0 }
        wrappedFlag = wrapped
        let live = wrapped ? keys.dim(HeadRowsAxis.row) : idx
        // A gap between history and live rows means eviction has occurred. Preserve
        // the pinned prefix regardless of layout, including after repeated trims.
        // Without a gap, an exact rewind can still remove any of the original rows.
        let minimum = offset > live ? Swift.min(keep, live) : 0
        let trimmed = Swift.min(n, live - minimum)
        guard trimmed > 0 else { return 0 }
        let bound = live - trimmed

        if wrapped || keys.dim(HeadRowsAxis.row) > storageCapacity {
            // Linearize a ring before cutting its newest rows. Also shrink oversized
            // prefill buffers: the next single-token write compacts those buffers to
            // storageCapacity and must not treat a discarded suffix as live history.
            self.keys = temporalOrder(keys)[.ellipsis, ..<bound, 0...]
            self.values = temporalOrder(values)[.ellipsis, ..<bound, 0...]
        }
        idx = bound
        offset -= trimmed
        wrappedFlag = false
        return trimmed
    }

    /// The mask of the next write of `n` rows, over the rows that write presents.
    ///
    /// A write of several rows into a ring that keeps a prefix, under the ring's own window,
    /// gives each query the view a write of one row gives it: the kept prefix and the newest
    /// rows up to the query. Chunked prefill and a speculative verify then attend as plain
    /// decode does, so how a stream batches its tokens does not change them.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            // Multi-token case. The mask must span the rows the write will present:
            // in temporal layout that is the live rows (`idx`), which fall below the
            // logical offset after a post-wrap trim.
            let actualWindowSize = windowSize ?? maxCacheSize
            let liveRows = wrapped ? maxCacheSize : idx
            let cappedOffset = min(maxCacheSize - 1, liveRows)

            // Decide if we need an array mask
            if cappedOffset + n > actualWindowSize || returnArray {
                if keep > 0, actualWindowSize == maxCacheSize {
                    return .array(prefixKeepingCausalMask(n: n, offset: cappedOffset))
                }
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single token case (n == 1)
            guard let windowSize = windowSize else {
                return .none
            }

            if rewindCapacity > 0 {
                let live = wrapped ? storageCapacity : idx
                let count = Swift.min(maxCacheSize - 1, live) + 1
                guard count > windowSize else { return .none }
                return .array(MLXArray(0 ..< count) .>= (count - windowSize))
            }

            // May need a mask when window_size < max_size and cache has wrapped
            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }

                let maskSize = (!wrapped && idx < maxCacheSize) ? idx + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)

                // Roll the mask to account for rotation
                let rolledMask = roll(mask, shift: currentIdx + 1)

                return .array(rolledMask)
            }
            return .none
        }
    }

    /// The causal mask of a write of `n` rows after `offset` presented rows, in which each query
    /// sees the kept prefix and the `maxCacheSize - keep` newest rows up to it.
    ///
    /// A write presents the kept prefix, then the newest rows in temporal order, then its own
    /// rows. A write of one row overwrites the oldest row after the prefix, so its query sees
    /// the prefix and the rows up to it that fill the rest of the window. A plain window over
    /// the presentation would instead drop the prefix first for each query past the window.
    ///
    /// - Parameters:
    ///   - n: The rows of the write.
    ///   - offset: The rows the write presents before its own.
    /// - Returns: A mask that is `true` where a query of the write may attend to a key.
    private func prefixKeepingCausalMask(n: Int, offset: Int) -> MLXArray {
        let newest = createCausalMask(n: n, offset: offset, windowSize: maxCacheSize - keep)
        let causal = createCausalMask(n: n, offset: offset)
        let prefix = MLXArray(Int32(0) ..< Int32(offset + n)) .< Int32(keep)
        return newest | (causal & prefix)
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx), wrapped: \(wrapped)"
    }

    public override func copy() -> any KVCache {
        let new = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to a quantized cache.
    ///
    /// Rotating-cache quantization needs a representation that preserves temporal ordering and
    /// rotation metadata. Until that representation exists, callers can recover by retaining the
    /// full-precision rotating cache.
    ///
    /// - Throws: Always, because rotating-cache quantization is not implemented.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) throws -> QuantizedKVCache {
        throw KVCacheError(
            message:
                "RotatingKVCache quantization is not implemented because its temporal ordering requires dedicated handling."
        )
    }
}
