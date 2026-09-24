// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
///
/// See ``KVCache/ropeOffset``.
public enum RoPEOffset {
    case scalar(Int)
    case batch(MLXArray)
}

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = QuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
/// `KVCache`'s `Evaluatable.innerState()` requirement is structurally
/// identical to `Updatable.innerState()`, so every conforming cache already
/// satisfies `Updatable` -- this conformance (added directly to the
/// protocol's inheritance list, since a protocol extension cannot declare a
/// new inheritance relationship) is what lets a `[any KVCache]` be passed as
/// `compile(inputs:outputs:...)`'s state so `compile()` can observe (and
/// replay) cache mutation across calls. See kanban
/// 01KYD3ZCWTZ414Y79RSAKVQXXZ for the design note.
public protocol KVCache: Evaluatable, Updatable {
    /// get the current offset
    var offset: Int { get }

    /// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
    var ropeOffset: RoPEOffset { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// The number of bytes that the buffers of this cache hold.
    ///
    /// This count includes the full allocated buffers, for example the step
    /// padding of ``KVCacheSimple``. It can thus be larger than the size of
    /// ``state``. An empty cache gives 0.
    ///
    /// The count reads only the shape and the element type of each array
    /// (`MLXArray.nbytes`), thus it evaluates nothing.
    ///
    /// This is a requirement and not only an extension member, so that a call
    /// through `any KVCache` goes to the implementation of the dynamic type.
    /// The default implementation is the sum of `nbytes` over ``innerState()``.
    var residentByteCount: Int { get }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// Predict whether this cache can still be trimmed after appending `positions`.
    ///
    /// - Parameter positions: The nonnegative number of sequence positions that
    ///   would be appended.
    /// - Returns: `true` when a subsequent rewind would remain valid.
    func isTrimmable(after positions: Int) -> Bool

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create an attention mask for this cache
    ///
    /// This method encapsulates cache-specific mask creation logic. Implementations should handle offset capping, window size logic,
    /// and optimization decisions (symbolic vs array masks).
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    /// Create an independent deep copy of this cache.
    func copy() -> any KVCache

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: [Int]?)

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: MLXArray?)

    /// Clear transient cache metadata after generation.
    func finalize()
}

extension KVCache {
    public var ropeOffset: RoPEOffset {
        .scalar(offset)
    }

    public func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    public func prepare(lengths: [Int]?) {}

    public func prepare(lengths: MLXArray?) {}

    public func finalize() {}

    /// The sum of `nbytes` over ``innerState()``.
    ///
    /// This default is correct for a cache whose ``innerState()`` gives every
    /// buffer the cache holds, at its full allocated size.
    public var residentByteCount: Int {
        innerState().totalByteCount
    }
}

extension Array where Element == MLXArray {
    /// The sum of `nbytes` over the arrays. It evaluates nothing.
    var totalByteCount: Int {
        reduce(0) { $0 + $1.nbytes }
    }
}

public func withPreparedCache<Result>(
    _ cache: [any KVCache],
    lengths: [Int]?,
    _ body: () throws -> Result
) rethrows -> Result {
    guard let lengths else {
        return try body()
    }
    for cache in cache {
        cache.prepare(lengths: lengths)
    }
    defer {
        for cache in cache {
            cache.finalize()
        }
    }
    return try body()
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMM(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
public protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// Quantization mode
    var mode: QuantizationMode { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
}

/// Protocol for caches that can update and compute attention from their own storage layout.
///
/// Used by compressed caches such as ``VarianceNormalizedKVCache`` that keep completed
/// tiles in a non-materialized representation and compute scores/values via cache-native
/// kernels (e.g. `quantizedMM` in a rotated domain).
public protocol KVCacheAttentionProtocol: KVCache {
    /// Update the cache with new K/V tensors and compute attention without first returning a
    /// fully materialized cache tensor pair.
    func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    /// RoPE offset for this cache. `open` so subclasses can return a non-scalar
    /// offset (e.g. a batched cache's per-row `.batch(...)`).
    open var ropeOffset: RoPEOffset { .scalar(offset) }

    public func innerState() -> [MLXArray] { [] }

    /// The sum of `nbytes` over ``innerState()``.
    ///
    /// `open` so that a subclass whose ``innerState()`` does not give its full
    /// buffers can override it. A call through `any KVCache` then reaches the
    /// override by dynamic dispatch.
    open var residentByteCount: Int {
        innerState().totalByteCount
    }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    open func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func copy() -> any KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    open func prepare(lengths: [Int]?) {}

    open func prepare(lengths: MLXArray?) {}

    open func finalize() {}

    /// Default implementation for caches without special mask requirements
    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For single token, no mask needed
        if n == 1 {
            return .none
        }

        // For multi-token sequences
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    return mask
}

/// Create an attention mask matching mlx-lm's create_attention_mask helper.
///
/// This returns `.causal` when a symbolic mask is sufficient, avoiding
/// materializing a full mask array.
public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also `MultiHeadAttention.createAdditiveCausalMask(_:dtype:)` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

@available(
    *, deprecated,
    message: "Use createAttentionMask(h:cache:windowSize:returnArray:) with a single cache instead"
)
public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize - 1, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

/// Create an attention mask with explicit window size parameter.
///
/// - Parameters:
///   - h: The input array (used to determine sequence length)
///   - cache: Optional single KV cache
///   - windowSize: Optional sliding window size (if provided, creates windowed attention)
///   - returnArray: Force return of array mask instead of symbolic "causal"
/// - Returns: Attention mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's makeMask if available
    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    // Fallback for no cache
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

/// Standard KV cache implementation based on Python's KVCache
/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to a quantized cache for maximum efficiency.
    ///
    /// Use `updateQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    ///
    /// - Throws: If neither the requested group size nor another supported group size can
    ///   represent both the key and value head dimensions.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) throws -> QuantizedKVCache {
        if let keys = self.keys, let values = self.values {
            // Quantize the current keys and values
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            guard
                let effectiveGroupSize = resolvedKVQuantizationGroupSize(
                    requested: groupSize,
                    keyHeadDim: currentKeys.dim(3),
                    valueHeadDim: currentValues.dim(3)
                )
            else {
                throw KVCacheError(
                    message:
                        "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(currentKeys.dim(3)). Value head dim: \(currentValues.dim(3))."
                )
            }
            let quantizedCache = QuantizedKVCache(groupSize: effectiveGroupSize, bits: bits)
            quantizedCache.offset = self.offset

            let quantizedKeys = quantized(
                currentKeys, groupSize: effectiveGroupSize, bits: bits)
            let quantizedValues = quantized(
                currentValues, groupSize: effectiveGroupSize, bits: bits)

            // Set the quantized state
            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }

            return quantizedCache
        }

        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset
        return quantizedCache
    }

    public override func copy() -> any KVCache {
        let new = KVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
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

    /// In ring layout all rows are live, with `idx...` preceding `keep ..< idx`.
    /// At the end of the buffer the ring is already in temporal order. Otherwise,
    /// temporal layout holds only the first `idx` rows, even when `offset` is larger.
    /// `nil` defers legacy inference until arrays and metadata have both been restored.
    /// Every write or trim resolves it before changing either buffers or counters.
    private var wrappedFlag: Bool? = false

    private var wrapped: Bool {
        wrappedFlag ?? (idx < (keys?.dim(2) ?? 0) && offset > idx)
    }

    /// Model-native sliding-window caches deliberately keep their architectural
    /// window and do not participate in requested-capacity validation.
    package var capacityOrigin = CapacityOrigin.modelNative

    package var preservedPrefixTokens: Int { keep }

    public override var maxSize: Int? { maxCacheSize }

    /// Number of leading tokens that are never rotated out of the window.
    var keepCount: Int { keep }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
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
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused.
        // `idx` bounds the live rows: after a post-wrap trim the logical `offset`
        // exceeds the rows actually held, so the layout question is `wrapped`,
        // never an `idx`/`offset` comparison.
        if idx == array.dim(2) {
            return array
        } else if wrapped {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
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
    /// The result is built by the same two steps the multi-token write path uses --
    /// ``temporalOrder(_:)`` to linearize, then the front-trim that preserves the pinned `keep`
    /// prefix -- so a view of length `n` holds exactly the entries a write that front-trimmed to
    /// `n` rows would have presented. When the ring is already chronological (`idx` at the end of
    /// the buffer, which is where every multi-token write leaves it) both steps degrade to
    /// slices and nothing is copied.
    ///
    /// The pinned `keep` prefix is a floor, not just a splice point: a `tail` below it still
    /// comes back, because those entries are not evictable and a view that dropped them would be
    /// a context this ring can never present. With `keep == 0` -- every sliding-window model --
    /// the floor is zero and the length is exactly `min(tail, count)`.
    ///
    /// - Parameter tail: Requested number of trailing entries. Clamped to what the cache holds;
    ///   a negative value is read as zero.
    /// - Returns: `(keys, values)` shaped `[B, kvHeads, n, headDim]` where
    ///   `n == max(min(tail, count), min(keep, count))`, or `nil` before the first write.
    package func logicalView(tail: Int) -> (MLXArray, MLXArray)? {
        guard let keys = self.keys, let values = self.values else { return nil }

        let orderedKeys = temporalOrder(keys)
        let orderedValues = temporalOrder(values)

        let available = orderedKeys.dim(2)
        // Raising the bound to the pinned prefix is what keeps the front-trim's second slice in
        // range; stated here rather than left to slice clamping, since the length it produces is
        // the documented contract.
        let requested = Swift.min(Swift.max(tail, 0), available)
        let bound = Swift.max(requested, Swift.min(keep, available))
        let trimSize = available - bound
        guard trimSize > 0 else { return (orderedKeys, orderedValues) }

        // `keep == 0` is the sliding-window case (Gemma 3/3n/4, GPT-OSS, Exaone4): no pinned
        // prefix to splice around, so the trailing window is one slice per array.
        if keep == 0 {
            return (
                orderedKeys[.ellipsis, trimSize..., 0...],
                orderedValues[.ellipsis, trimSize..., 0...]
            )
        }
        return (
            trim(trimSize: trimSize, orderedKeys),
            trim(trimSize: trimSize, orderedValues)
        )
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            // Allow temporary cache growth during multi-token processing (e.g., prompt prefill).
            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context
            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)
        wrappedFlag = false

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)

        // May not have hit the max size yet, so potentially keep growing the cache.
        // Fill is tracked by `idx`, not `offset`: after a post-wrap trim the logical
        // offset exceeds the rows actually held, and growth must resume from the rows.
        let filled = self.keys?.dim(2) ?? 0
        if self.keys == nil || (!wrapped && idx >= filled && filled < maxCacheSize) {
            let newSize = min(step, maxCacheSize - filled)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
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
        if !wrapped, idx < self.keys!.dim(2) {
            return (
                self.keys![.ellipsis, ..<idx, 0...],
                self.values![.ellipsis, ..<idx, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        wrappedFlag = wrapped
        let result =
            if keys.dim(2) == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if !wrapped, idx < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<idx, 0...],
                    values[.ellipsis, ..<idx, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
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
            return [
                String(keep), String(maxCacheSize), String(step), String(offset), String(idx),
                capacityOrigin.rawValue, String(wrapped),
            ]
        }
        set {
            guard (5 ... 7).contains(newValue.count) else {
                fatalError("RotatingKVCache metaState must have 5 to 7 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
            if newValue.count >= 6 {
                guard let origin = CapacityOrigin(rawValue: newValue[5]) else {
                    fatalError("Invalid RotatingKVCache capacity origin '\(newValue[5])'")
                }
                self.capacityOrigin = origin
            } else {
                self.capacityOrigin = .modelNative
            }
            if newValue.count == 7 {
                guard let wrappedValue = Bool(newValue[6]) else {
                    fatalError("Invalid RotatingKVCache wrapped flag '\(newValue[6])'")
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

    public override func isTrimmable(after positions: Int) -> Bool {
        // This is the *exact rewind* predicate: past the window a trim is merely
        // consistent (see `trim`), because rows the rewound writes overwrote are
        // gone. Consumers that must undo writes exactly -- the staged-round and
        // prompt-cache-reuse machinery -- key off this and fall back to staging,
        // snapshots, or a rebuild once it turns false.
        offset + positions < maxCacheSize
    }

    /// Rewind the newest `n` positions.
    ///
    /// Before the ring wraps this is exact bookkeeping. After it wraps, the ring is
    /// linearized and the newest rows are cut: the cache stays consistent and the
    /// logical offset rewinds, but rows the rewound writes overwrote at the old edge
    /// of the window cannot come back, so the window is up to `n` rows short until
    /// it refills. Callers that need an exact rewind must gate on
    /// ``isTrimmable(after:)`` instead of calling this unconditionally.
    /// Once older rows have been evicted, trimming stops at the pinned `keep` prefix.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, let keys, let values else { return 0 }
        wrappedFlag = wrapped
        let live = wrapped ? keys.dim(2) : idx
        // A gap between history and live rows means eviction has occurred. Preserve
        // the pinned prefix regardless of layout, including after repeated trims.
        // Without a gap, an exact rewind can still remove any of the original rows.
        let minimum = offset > live ? Swift.min(keep, live) : 0
        let trimmed = Swift.min(n, live - minimum)
        guard trimmed > 0 else { return 0 }
        let bound = live - trimmed

        if wrapped || keys.dim(2) > maxCacheSize {
            // Linearize a ring before cutting its newest rows. Also shrink oversized
            // prefill buffers: the next single-token write compacts those buffers to
            // maxCacheSize and must not treat a discarded suffix as live history.
            self.keys = temporalOrder(keys)[.ellipsis, ..<bound, 0...]
            self.values = temporalOrder(values)[.ellipsis, ..<bound, 0...]
        }
        idx = bound
        offset -= trimmed
        wrappedFlag = false
        return trimmed
    }

    /// Optimized mask creation for rotating cache with offset capping
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
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single token case (n == 1)
            guard let windowSize = windowSize else {
                return .none
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

func resolvedKVQuantizationGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = [32, 64, 128].filter {
        keyHeadDim.isMultiple(of: $0) && valueHeadDim.isMultiple(of: $0)
    }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = abs(lhs - requested)
        let rhsDistance = abs(rhs - requested)
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public private(set) var groupSize: Int
    public private(set) var bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset
        let effectiveGroupSize = resolvedKVQuantizationGroupSize(
            requested: groupSize,
            keyHeadDim: kHeadDim,
            valueHeadDim: vHeadDim
        )
        if let effectiveGroupSize,
            effectiveGroupSize != groupSize,
            self.keys == nil,
            self.values == nil,
            offset == 0
        {
            self.groupSize = effectiveGroupSize
        }
        guard effectiveGroupSize != nil else {
            fatalError(
                "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(kHeadDim). Value head dim: \(vHeadDim)."
            )
        }

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    /// Array of keys and values -- this will have either 6 elements or 4 elements (if biases are nil).
    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 4:
                // nil biases case
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }
            guard
                let offset = Int(newValue[1]),
                let groupSize = Int(newValue[2]),
                let bits = Int(newValue[3])
            else {
                fatalError("Failed to convert QuantizedKVCache metaState values to integers")
            }

            self.offset = offset
            self.groupSize = groupSize
            self.bits = bits
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

/// Chunked KV cache for processing large contexts in chunks
public class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = ChunkedKVCache(chunkSize: chunkSize)
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
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

    public func advance(_ N: Int) {
        if let currentLengths = lengths {
            lengths = currentLengths - N
        }
        if let currentLeftPadding = leftPadding {
            leftPadding = currentLeftPadding - N
        }
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
        if savedMetaState.count >= 2, let slotCount = Int(savedMetaState[0]) {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = Self.metadataArray(savedMetaState, at: 2)
            self.lengths = Self.metadataArray(savedMetaState, at: 3)
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
    private struct SpeculativeCheckpoint {
        var state: [MLXArray?]
        var offset: Int
        var leftPadding: MLXArray?
        var lengths: MLXArray?
    }

    private var speculativeCheckpoint: SpeculativeCheckpoint?

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    /// Moves this cache past `tokenCount` tokens the layer fed through it.
    ///
    /// The conv and recurrent state already hold those tokens. This moves the
    /// batch bookkeeping of ``ArraysCache/advance(_:)`` AND the position
    /// `offset`, thus a prompt cache can compare this cache with the attention
    /// caches of the same model. A recurrent cache whose position stays at zero
    /// never agrees with its token ledger, and the model starts every round
    /// cold.
    ///
    /// - Parameter tokenCount: the number of tokens the layer fed.
    public func advancePosition(by tokenCount: Int) {
        advance(tokenCount)
        offset += tokenCount
    }

    /// Save the recurrent state at the last unconditionally committed token
    /// inside a speculative verification pass.
    ///
    /// The caller saves BEFORE it moves this cache past the whole input, thus
    /// the checkpoint's position is the current position plus `tokenCount`.
    package func saveSpeculativeCheckpoint(
        convState: MLXArray,
        recurrentState: MLXArray,
        advancedBy tokenCount: Int
    ) {
        speculativeCheckpoint = SpeculativeCheckpoint(
            state: [convState, recurrentState],
            offset: offset + tokenCount,
            leftPadding: leftPadding.map { $0 - tokenCount },
            lengths: lengths.map { $0 - tokenCount })
    }

    package var hasSpeculativeCheckpoint: Bool {
        speculativeCheckpoint != nil
    }

    @discardableResult
    package func restoreSpeculativeCheckpoint() -> Bool {
        guard let checkpoint = speculativeCheckpoint else { return false }
        cache = checkpoint.state
        offset = checkpoint.offset
        leftPadding = checkpoint.leftPadding
        lengths = checkpoint.lengths
        speculativeCheckpoint = nil
        return true
    }

    package func discardSpeculativeCheckpoint() {
        speculativeCheckpoint = nil
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        copyContents(to: new)
        return new
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    /// Internal initializer for reconstruction from deserialized children.
    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    /// The sum of ``KVCache/residentByteCount`` over the children, so that each
    /// child counts with its own rule.
    public override var residentByteCount: Int {
        caches.reduce(0) { $0 + $1.residentByteCount }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    /// Recursively apply a transformation to every non-composite child cache.
    ///
    /// `CacheList` children are descended into; any other cache is passed to
    /// `transform` and replaced by the returned value. This is the primitive
    /// used by dynamic cache quantization and other cache-wide rewrites for
    /// models with hybrid attention/recurrent caches (e.g. Falcon-H1).
    public func mapChildren(_ transform: (KVCache) -> KVCache) {
        caches = caches.map { child in
            if let list = child as? CacheList {
                list.mapChildren(transform)
                return list
            }
            return transform(child)
        }
    }

    /// Rewrite every non-composite child while preserving its stable tree path.
    func rewriteLeaves(
        path: [Int],
        using transform: (KVCacheLeaf) -> KVCache
    ) {
        caches = caches.enumerated().map { index, child in
            let childPath = path + [index]
            if let list = child as? CacheList {
                list.rewriteLeaves(path: childPath, using: transform)
                return list
            }
            let leaf = KVCacheLeaf(path: childPath, cache: child)
            return transform(leaf)
        }
    }

    public override func prepare(lengths: [Int]?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func prepare(lengths: MLXArray?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func finalize() {
        caches.forEach { $0.finalize() }
    }

    public override var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        caches.allSatisfy { $0.isTrimmable(after: positions) }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    /// Internal accessor for child caches (used by serialization and policy reporting).
    internal var children: [KVCache] { caches }

    // MARK: - Serialization

    /// metaState format: [childCount, (className, stateCount, metaStateCount, ...metaState)*]
    ///
    /// Like Python's CacheList.meta_state which returns [child_class_names, child_meta_states],
    /// but flattened for Swift's [String] format.
    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className = cacheClassName(cache)
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList.metaState should not be set directly. Use CacheList.fromState() instead")
        }
    }

    /// Reconstruct a CacheList from flattened state + metaState, like Python's from_state()
    internal static func fromState(state: [MLXArray], metaState: [String]) throws -> CacheList {
        let children = try savedChildren(state: state, metaState: metaState).map {
            try restoreCacheFromMetaState(
                className: $0.className, state: $0.state, metaState: $0.metaState)
        }
        return CacheList(caches: children)
    }

    /// Replaces the children after a prompt cache restore. A restore can give a child another
    /// cache class, for example a `QuantizedKVCache` in place of a `KVCacheSimple`.
    ///
    /// - Parameter children: The new children, one for each old child, in the same order.
    internal func replaceChildren(with children: [KVCache]) {
        precondition(
            children.count == caches.count,
            "a CacheList has \(caches.count) children and the restore gave \(children.count)")
        caches = children
    }
}

/// One child of a saved `CacheList`: its class name, its arrays and its meta state.
private struct SavedCacheListChild {
    /// The class name that the save wrote for the child.
    let className: String

    /// The saved arrays of the child.
    let state: [MLXArray]

    /// The saved meta state of the child.
    let metaState: [String]
}

extension CacheList {
    /// The number of meta-state values in front of each child: its class name, its array
    /// count and its meta-state count.
    private static let childHeaderCount = 3

    /// The place of the array count in the header of a child.
    private static let childStateCountOffset = 1

    /// The place of the meta-state count in the header of a child.
    private static let childMetaStateCountOffset = 2

    /// Splits the flat `state` and `metaState` of a saved `CacheList` into its children.
    ///
    /// The format is `[childCount, (className, stateCount, metaStateCount, metaState...)*]`.
    /// Every count is checked against the values that are there, and every value must belong
    /// to a child, thus no slice can start past an end.
    ///
    /// - Parameters:
    ///   - state: The saved arrays of the whole list.
    ///   - metaState: The saved meta state of the whole list.
    /// - Returns: The children, in order.
    /// - Throws: ``KVCacheError`` when a count does not fit the saved values.
    fileprivate static func savedChildren(
        state: [MLXArray], metaState: [String]
    ) throws -> [SavedCacheListChild] {
        guard let childCount = metaState.first.flatMap({ Int($0) }), childCount >= 0 else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }
        var children: [SavedCacheListChild] = []
        var metaIndex = 1
        var stateIndex = 0
        for _ in 0 ..< childCount {
            let child = try savedChild(
                state: state, metaState: metaState, metaIndex: metaIndex, stateIndex: stateIndex)
            children.append(child)
            metaIndex += childHeaderCount + child.metaState.count
            stateIndex += child.state.count
        }
        guard metaIndex == metaState.count, stateIndex == state.count else {
            throw KVCacheError(
                message: "Corrupt prompt cache: CacheList holds values that no child owns.")
        }
        return children
    }

    /// Reads one child of a saved `CacheList`.
    ///
    /// - Parameters:
    ///   - state: The saved arrays of the whole list.
    ///   - metaState: The saved meta state of the whole list.
    ///   - metaIndex: The place of the header of the child in `metaState`.
    ///   - stateIndex: The place of the first array of the child in `state`.
    /// - Returns: The child.
    /// - Throws: ``KVCacheError`` when the header is truncated or a count is out of range.
    private static func savedChild(
        state: [MLXArray], metaState: [String], metaIndex: Int, stateIndex: Int
    ) throws -> SavedCacheListChild {
        let metaStart = metaIndex + childHeaderCount
        guard metaStart <= metaState.count else {
            throw KVCacheError(message: "CacheList metaState truncated")
        }
        guard let stateCount = Int(metaState[metaIndex + childStateCountOffset]),
            let metaCount = Int(metaState[metaIndex + childMetaStateCountOffset]),
            stateCount >= 0, metaCount >= 0,
            stateIndex + stateCount <= state.count, metaStart + metaCount <= metaState.count
        else {
            throw KVCacheError(message: "CacheList: invalid array or metaState count for child")
        }
        return SavedCacheListChild(
            className: metaState[metaIndex],
            state: Array(state[stateIndex ..< stateIndex + stateCount]),
            metaState: Array(metaState[metaStart ..< metaStart + metaCount]))
    }
}

// MARK: - Error Types

/// The error that a prompt cache operation throws: for example a corrupt file, or a file that
/// does not fit the caches it is restored into.
///
/// A cache type outside `MLXLMCommon` throws it from
/// ``PromptCacheRestorable/validatePromptCacheRestore(state:metaState:)``.
public struct KVCacheError: Error, LocalizedError {
    /// The description of the problem.
    public let message: String

    /// Makes an error.
    ///
    /// - Parameter message: The description of the problem.
    public init(message: String) {
        self.message = message
    }

    /// The description of the problem, as `LocalizedError` gives it.
    public var errorDescription: String? { message }
}

// MARK: - Utility Functions

/// Registry for `KVCache` types defined outside `MLXLMCommon` (e.g.
/// model-specific caches in `MLXVLM`/`MLXLLM`) that need `savePromptCache`/
/// `loadPromptCache` support.
///
/// `MLXLMCommon` cannot import downstream modules to switch on their
/// concrete cache types directly, so those types self-register a
/// serialization name and a restore factory -- typically via a one-time
/// `static let` triggered from the type's own `init()`, guaranteeing
/// registration has happened before any instance could be saved. Without
/// registration, an unrecognized cache type falls back to being saved as a
/// plain `KVCache` (``KVCacheSimple``), which silently corrupts round-trips
/// for any type whose `state` doesn't have exactly 2 arrays.
public enum KVCacheSerializationRegistry {
    /// Reconstructs a registered cache type from its saved `state`/`metaState`.
    public typealias RestoreFactory = ([MLXArray], [String]) -> KVCache

    private static let lock = NSLock()
    // Manually synchronized by `lock` above -- the compiler cannot see that,
    // so these are marked `nonisolated(unsafe)` rather than restructured into
    // an actor (which would force `register`/`className`/`restore` async and
    // ripple into every `KVCache` save/restore call site).
    nonisolated(unsafe) private static var classNamesByType: [ObjectIdentifier: String] = [:]
    nonisolated(unsafe) private static var factoriesByClassName: [String: RestoreFactory] = [:]

    /// Registers a custom `KVCache` type under `className` for save/restore support.
    ///
    /// - Parameters:
    ///   - type: the concrete `KVCache` type being registered
    ///   - className: a name unique among all registered and built-in class names
    ///   - restore: reconstructs an instance from previously-saved `state`/`metaState`
    public static func register<T: KVCache>(
        _ type: T.Type, className: String, restore: @escaping RestoreFactory
    ) {
        lock.lock()
        defer { lock.unlock() }
        classNamesByType[ObjectIdentifier(type)] = className
        factoriesByClassName[className] = restore
    }

    fileprivate static func className(for cache: KVCache) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return classNamesByType[ObjectIdentifier(type(of: cache))]
    }

    fileprivate static func restore(
        className: String, state: [MLXArray], metaState: [String]
    ) -> KVCache? {
        lock.lock()
        defer { lock.unlock() }
        return factoriesByClassName[className]?(state, metaState)
    }
}

extension KVCacheSerializationRegistry {
    /// Checks the saved `state` and `metaState` of one built-in cache class before a setter
    /// reads them.
    ///
    /// A cache type outside `MLXLMCommon` that holds a built-in cache (for example a
    /// `RotatingKVCache` window) calls this from
    /// ``PromptCacheRestorable/validatePromptCacheRestore(state:metaState:)``. The built-in
    /// setters stop the process on bad input, thus the check must come first.
    ///
    /// - Parameters:
    ///   - state: The saved arrays of the built-in part.
    ///   - metaState: The saved meta state of the built-in part.
    ///   - cache: The built-in cache that will receive the values. Its class selects the rules.
    /// - Throws: ``KVCacheError`` when the values do not fit the class of `cache`.
    public static func validate(
        state: [MLXArray], metaState: [String], for cache: KVCache
    ) throws {
        guard let className = builtInCacheClassName(cache),
            builtInLeafClassNames.contains(className)
        else {
            throw KVCacheError(
                message: "\(type(of: cache)) is not a built-in cache class that has a check.")
        }
        try validateBuiltInCache(className: className, state: state, metaState: metaState)
    }
}

/// A cache type outside `MLXLMCommon` that ``loadPromptCacheSnapshot(url:into:)`` restores in
/// place.
///
/// Such a type often needs model configuration to build, thus a `(state, metaState)` factory
/// in ``KVCacheSerializationRegistry`` cannot make it. The template restore writes the saved
/// values into an instance that the model made. The setters of a cache usually stop the process
/// on bad input, thus the restore calls
/// ``validatePromptCacheRestore(state:metaState:)`` for every layer before it calls
/// ``restorePromptCache(state:metaState:)`` for any layer.
public protocol PromptCacheRestorable: KVCache {
    /// The class name that `savePromptCache` writes for this type.
    static var promptCacheClassName: String { get }

    /// Checks saved values before any setter reads them.
    ///
    /// - Parameters:
    ///   - state: The saved arrays.
    ///   - metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when ``restorePromptCache(state:metaState:)`` cannot accept
    ///   the values.
    func validatePromptCacheRestore(state: [MLXArray], metaState: [String]) throws

    /// Writes values into this cache. The values passed the check of
    /// ``validatePromptCacheRestore(state:metaState:)``.
    ///
    /// - Parameters:
    ///   - state: The saved arrays.
    ///   - metaState: The saved meta state.
    func restorePromptCache(state: [MLXArray], metaState: [String])
}

/// Map a built-in cache instance to its Python-compatible class name.
///
/// - Parameter cache: The cache to name.
/// - Returns: The class name, or `nil` when `cache` is not a built-in class.
private func builtInCacheClassName(_ cache: KVCache) -> String? {
    switch cache {
    case is ChunkedKVCache: return "ChunkedKVCache"
    case is MambaCache: return "MambaCache"
    case is ArraysCache: return "ArraysCache"
    case is RotatingKVCache: return "RotatingKVCache"
    case is VarianceNormalizedKVCache: return "VarianceNormalizedKVCache"
    case is QuantizedKVCache: return "QuantizedKVCache"
    case is TurboQuantKVCache: return "TurboQuantKVCache"
    case is KVCacheSimple: return "KVCache"
    case is CacheList: return "CacheList"
    default: return nil
    }
}

/// Map a cache instance to its Python-compatible class name for serialization.
private func cacheClassName(_ cache: KVCache) -> String {
    if let className = builtInCacheClassName(cache) {
        return className
    }
    if let restorable = cache as? PromptCacheRestorable {
        return type(of: restorable).promptCacheClassName
    }
    return KVCacheSerializationRegistry.className(for: cache) ?? "KVCache"
}

/// A prompt cache and the model state that belongs with it.
///
/// Keeping the two together is the point: a KV cache restored without its model state positions
/// later tokens as if the cached prefix contained no images, which changes the output silently.
/// Pass a snapshot to a `ChatSession` initializer that accepts a `promptCache` rather than
/// unpacking it into the `cache:` initializer.
///
/// A snapshot does not carry the chat transcript. A `ChatSession` restored from one appends each
/// new message rather than re-rendering the conversation, so a later image-bearing turn builds a
/// different prompt than it would in a session that still holds its history. Positions stay
/// correct either way. On vision encoders that attend across image boundaries (Qwen2-VL today)
/// the new image's features differ too, since it is encoded alone rather than beside the cached
/// one; Qwen2.5-VL, Qwen3-VL, and GLM-OCR isolate each image and are unaffected.
///
/// The cache instances are mutable reference types. Transfer a snapshot to one session or copy the
/// caches before constructing multiple sessions from it.
public struct PromptCacheSnapshot {
    public let cache: [KVCache]
    public let metadata: [String: String]
    public let state: LMOutput.State?

    /// Pair a cache with the model state that belongs with it.
    ///
    /// Use this when the cache was built in process — ``loadPromptCacheSnapshot(url:)`` returns a
    /// snapshot for caches read from disk.
    ///
    /// - Parameters:
    ///   - cache: the KV cache
    ///   - metadata: optional caller metadata to carry alongside it
    ///   - state: the model state captured with the cache, if the model produced any
    public init(cache: [KVCache], metadata: [String: String] = [:], state: LMOutput.State? = nil) {
        self.cache = cache
        self.metadata = metadata
        self.state = state
    }
}

/// Save a pre-computed prompt cache to a file.
///
/// The file also records the `offset` of each top-level cache, because the `metaState` of some
/// classes (for example `MambaCache`) does not hold it. The load functions apply the record and
/// do not give it back in the user metadata.
///
/// This is ``preparePromptCacheSave(cache:metadata:state:)`` and then
/// ``writePromptCache(_:url:)``. A caller that must give the caches to another task before the
/// file write ends calls the two steps itself.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
///   - state: Optional model state associated with the cache
/// - Throws: ``KVCacheError`` when a key of `metadata` uses a reserved prefix
///   (`__mlx_lm_state_` or `__mlx_lm_offset_`), or when `state` has values and `cache` is empty.
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:],
    state: LMOutput.State? = nil
) throws {
    try writePromptCache(
        preparePromptCacheSave(cache: cache, metadata: metadata, state: state), url: url)
}

/// The arrays and the metadata of one prompt cache file, read from the caches and ready to write.
///
/// ``preparePromptCacheSave(cache:metadata:state:)`` makes it, and
/// ``writePromptCache(_:url:)`` writes it. Each array is a new `MLXArray` handle, and the input
/// holds no reference to a cache. Thus a cache can take more tokens after the prepare step, and
/// the file still holds the values that the cache held at that step.
// swiftlint:disable:next no_unchecked_sendable The input is immutable, and each MLXArray in it is a handle that only this input holds: no cache writes it, and the write only reads it.
public struct PromptCacheSaveInput: @unchecked Sendable {
    /// The flat arrays of the file: `"i.j"` for array `j` of cache `i`, and the model-state
    /// tensors under their reserved prefix.
    public let arrays: [String: MLXArray]

    /// The flat metadata of the file: the meta state of each cache, the user metadata with the
    /// offset record and the model-state keys, and the class name of each cache.
    public let metadata: [String: String]

    /// The class name of each top-level cache, in cache order, as the file writes it.
    public let classNames: [String]
}

/// Reads the caches and the model state that a prompt cache file saves, and makes the input of
/// ``writePromptCache(_:url:)``.
///
/// The function reads `state` and `metaState` of each cache at once, thus the caller runs it
/// while it owns the caches. It does not evaluate the arrays and it does not write a file.
///
/// - Parameters:
///   - cache: The model cache state.
///   - metadata: Optional metadata to save along with cache state.
///   - state: Optional model state associated with the cache.
/// - Returns: The input, which holds new array handles and no reference to a cache.
/// - Throws: ``KVCacheError`` when a key of `metadata` uses a reserved prefix
///   (`__mlx_lm_state_` or `__mlx_lm_offset_`), or when `state` has values and `cache` is empty.
public func preparePromptCacheSave(
    cache: [KVCache],
    metadata: [String: String] = [:],
    state: LMOutput.State? = nil
) throws -> PromptCacheSaveInput {
    try PromptCacheOffsetRecord.validateUserMetadata(metadata)
    let stateArrays = try promptCacheStateArrays(state, userMetadata: metadata)
    guard stateArrays.isEmpty || !cache.isEmpty else {
        throw KVCacheError(message: "Model state requires at least one prompt cache")
    }

    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    let cacheClasses = cache.map {
        promptCacheClassName(cacheClassName($0), hasState: !stateArrays.isEmpty)
    }

    // Flatten cache data using tree_flatten compatible structure: "i.j" format
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // Create cache_metadata structure compatible with Python: [cache_info, metadata, cache_classes]
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata). The offset record
    // goes in the same part, under its reserved prefix, which no user key uses.
    let recordedMetadata = metadata.merging(PromptCacheOffsetRecord.entries(for: cache)) {
        user, _ in user
    }
    for (key, value) in recordedMetadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    addPromptCacheState(
        stateArrays, flattenedData: &flattenedData, flattenedMetadata: &flattenedMetadata)

    // A cache can write into the array object that its `state` gave back (a full
    // `RotatingKVCache` does), thus the input takes a new handle of each array.
    return PromptCacheSaveInput(
        arrays: flattenedData.mapValues { $0[.ellipsis] }, metadata: flattenedMetadata,
        classNames: cacheClasses)
}

/// Writes a prompt cache file that ``preparePromptCacheSave(cache:metadata:state:)`` prepared.
///
/// The function reads no cache, thus it can run on another task after the caches have
/// taken more tokens.
///
/// - Parameters:
///   - input: The prepared arrays and metadata.
///   - url: The URL of the `.safetensors` file.
/// - Throws: The error of the safetensors writer, for example when `url` does not end in
///   `.safetensors` or the file cannot be written.
public func writePromptCache(_ input: PromptCacheSaveInput, url: URL) throws {
    try save(arrays: input.arrays, metadata: input.metadata, url: url)
}

/// Load a prompt cache from a file, without model state.
///
/// Prefer ``loadPromptCacheSnapshot(url:)``, which also restores the model state a cache needs to
/// be continued correctly. This tuple form has no slot for that state, so it rejects files that
/// carry any rather than dropping it. Models that position from a carried anchor refuse a warm
/// cache that arrives without one, so a cache saved through this API cannot warm them.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
/// - Throws: If the file contains model state that this tuple return value cannot represent, or
///   for each error of ``loadPromptCacheSnapshot(url:)``
public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let promptCache = try loadPromptCacheSnapshot(url: url)
    guard promptCache.state == nil else {
        throw KVCacheError(
            message:
                "Prompt cache contains model state; use loadPromptCacheSnapshot(url:) to restore it"
        )
    }
    return (promptCache.cache, promptCache.metadata)
}

/// Load a prompt cache and its associated model state from a file.
///
/// - Parameter url: The URL of the `.safetensors` file.
/// - Returns: The snapshot. Each cache has the offset that the file records, when the file has
///   an offset record.
/// - Throws: ``KVCacheError`` when the file is not a prompt cache that this reader can restore,
///   or when a restored offset is not the offset that the file records.
public func loadPromptCacheSnapshot(url: URL) throws -> PromptCacheSnapshot {
    let contents = try PromptCacheFileContents(url: url)
    let caches = try contents.layers.map {
        try restoreCacheFromMetaState(
            className: $0.className, state: $0.state, metaState: $0.metaState)
    }
    try PromptCacheOffsetRecord.apply(contents.offsets, to: caches)
    return PromptCacheSnapshot(
        cache: caches, metadata: contents.userMetadata, state: contents.state)
}

/// Load a prompt cache and its model state from a file into the fresh caches that the model
/// made, for example with `model.newCache(parameters:)`.
///
/// A cache type that needs model configuration to build (for example `DeepSeekV4Cache`) has no
/// `(state, metaState)` factory, thus ``loadPromptCacheSnapshot(url:)`` cannot build it. This
/// function writes the saved values into the caches in `templates` instead:
///
/// - The saved class of each layer must be the class of its template. One conversion is
///   accepted: generation can change a `KVCacheSimple` layer into a `QuantizedKVCache` or a
///   `TurboQuantKVCache`, and a model never makes those classes. For a `KVCacheSimple`
///   template, a saved layer of one of these two classes is built new from the file, and the
///   snapshot holds that new cache in place of the template. The same rule applies to each
///   child of a `CacheList`.
/// - Every layer is checked before any setter runs, because the cache setters stop the process
///   on bad input.
/// - The function evaluates every array that it read from the file before it returns. The
///   safetensors load is lazy, thus without this step the file must stay until the first use.
///   After the return the caller can delete the file.
///
/// - Parameters:
///   - url: The URL of the `.safetensors` file.
///   - templates: The fresh caches of the model, one for each saved layer.
/// - Returns: The snapshot. Its caches are the templates, or the new converted caches.
/// - Throws: ``KVCacheError`` on a layer-count mismatch, a class mismatch, saved values that
///   do not fit a class, or a restored offset that is not the offset that the file records.
///   After a throw the state of the templates is not defined. Discard them.
public func loadPromptCacheSnapshot(
    url: URL, into templates: [KVCache]
) throws -> PromptCacheSnapshot {
    let contents = try PromptCacheFileContents(url: url)
    guard contents.layers.count == templates.count else {
        throw KVCacheError(
            message:
                "The prompt cache holds \(contents.layers.count) layers and the model gave \(templates.count) caches."
        )
    }
    let restoreSteps = try zip(contents.layers, templates).map { layer, template in
        try PromptCacheTemplateRestore.prepare(layer, into: template)
    }
    let caches = restoreSteps.map { $0() }
    try PromptCacheOffsetRecord.apply(contents.offsets, to: caches)
    eval(contents.fileArrays)
    return PromptCacheSnapshot(
        cache: caches, metadata: contents.userMetadata, state: contents.state)
}

/// The saved values of one layer of a prompt cache file.
private struct SavedPromptCacheLayer {
    /// The class name that the save wrote for the layer.
    let className: String

    /// The saved arrays of the layer.
    let state: [MLXArray]

    /// The saved meta state of the layer.
    let metaState: [String]
}

/// The contents of a prompt cache file, before any cache is built.
private struct PromptCacheFileContents {
    /// The number of top-level parts of the metadata: the cache information, the user
    /// metadata and the class names.
    private static let metadataPartCount = 3

    /// The place of the cache information (the meta state of each layer) in the metadata.
    private static let cacheInfoPart = 0

    /// The place of the user metadata in the metadata.
    private static let userMetadataPart = 1

    /// The place of the class names in the metadata.
    private static let classNamesPart = 2

    /// The saved layers, in order.
    let layers: [SavedPromptCacheLayer]

    /// The caller metadata.
    let userMetadata: [String: String]

    /// The recorded offset of each layer, or `nil` when the file has no offset record.
    let offsets: [Int]?

    /// The model state, when the file holds one.
    let state: LMOutput.State?

    /// Every array that the file holds: the cache arrays and the model-state arrays.
    let fileArrays: [MLXArray]

    /// Reads a prompt cache file.
    ///
    /// The arrays are not evaluated: the safetensors load is lazy.
    ///
    /// - Parameter url: The URL of the `.safetensors` file.
    /// - Throws: ``KVCacheError`` when the metadata is not a prompt cache layout.
    init(url: URL) throws {
        var (arrays, metadata) = try loadArraysAndMetadata(url: url)
        fileArrays = Array(arrays.values)

        // Unflatten metadata using tree_unflatten compatible logic.
        // Structure: [cache_info, user_metadata, cache_classes]
        let unflattenedMetadata = unflattenMetadata(metadata)
        guard unflattenedMetadata.count >= Self.metadataPartCount else {
            throw KVCacheError(message: "Invalid cache metadata format")
        }

        let cacheInfo = unflattenedMetadata[Self.cacheInfoPart] as? [[String]] ?? []
        let storedUserMetadata =
            unflattenedMetadata[Self.userMetadataPart] as? [String: String] ?? [:]
        let (loadedState, loadedUserMetadata) = try loadPromptCacheState(
            arrays: &arrays, metadata: storedUserMetadata)
        state = loadedState
        let storedCacheClasses = unflattenedMetadata[Self.classNamesPart] as? [String] ?? []
        let cacheClasses = try loadPromptCacheClasses(
            storedCacheClasses, hasState: loadedState != nil)
        (offsets, userMetadata) = try PromptCacheOffsetRecord.read(
            from: loadedUserMetadata, layerCount: cacheClasses.count)

        guard cacheInfo.count == cacheClasses.count else {
            throw KVCacheError(message: "Mismatch in cache counts")
        }

        // Metadata carries the cache count even when one or more valid caches have no arrays.
        // State tensors were removed from `arrays` above, so only cache arrays remain here.
        let cacheData = try unflattenArrays(arrays, cacheCount: cacheClasses.count)
        layers = cacheData.indices.map {
            SavedPromptCacheLayer(
                className: cacheClasses[$0], state: cacheData[$0], metaState: cacheInfo[$0])
        }
    }
}

/// The steps of ``loadPromptCacheSnapshot(url:into:)`` for one layer.
private enum PromptCacheTemplateRestore {
    /// A restore step that runs after every layer passed its check. It writes the saved values
    /// and gives back the cache that holds them.
    typealias Step = () -> KVCache

    /// The saved classes that can take the place of a `KVCacheSimple` template. Generation
    /// makes them from a `KVCacheSimple` layer, and a model never makes them itself.
    private static let convertedClassNames: Set<String> = ["QuantizedKVCache", "TurboQuantKVCache"]

    /// The meta-state places that hold configuration that a cache sets in `init` and that no
    /// setter writes. The saved values must equal the values of the template.
    private static let fixedConfigurationIndices: [String: [Int]] = [
        "VarianceNormalizedKVCache": [
            VarianceNormalizedSavedLayout.tileSizeIndex, VarianceNormalizedSavedLayout.keyBitsIndex,
            VarianceNormalizedSavedLayout.valueBitsIndex,
            VarianceNormalizedSavedLayout.sinkhornIterationsIndex,
        ],
        "TurboQuantKVCache": [
            SavedMetaStateIndex.turboQuantBits, SavedMetaStateIndex.turboQuantKeyBits,
            SavedMetaStateIndex.turboQuantValueBits, SavedMetaStateIndex.turboQuantSeed,
        ],
        "ArraysCache": [SavedMetaStateIndex.arraysSlotCount],
        "MambaCache": [SavedMetaStateIndex.arraysSlotCount],
    ]

    /// Checks one saved layer against its template, and makes the step that restores it.
    ///
    /// No setter runs here, thus a throw leaves the template as it was.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - template: The fresh cache that the model made for this layer.
    /// - Returns: The step that writes the values.
    /// - Throws: ``KVCacheError`` when the layer does not fit the template.
    static func prepare(_ layer: SavedPromptCacheLayer, into template: KVCache) throws -> Step {
        let templateClassName = cacheClassName(template)
        guard layer.className == templateClassName else {
            return try prepareConverted(layer, into: template, templateClassName: templateClassName)
        }
        if let list = template as? CacheList {
            return try prepareCacheList(layer, into: list)
        }
        if let restorable = template as? PromptCacheRestorable {
            try restorable.validatePromptCacheRestore(
                state: layer.state, metaState: layer.metaState)
            return {
                restorable.restorePromptCache(state: layer.state, metaState: layer.metaState)
                return restorable
            }
        }
        guard builtInCacheClassName(template) != nil else {
            throw KVCacheError(
                message: "\(type(of: template)) cannot receive a restore into a template.")
        }
        try validateBuiltInCache(
            className: layer.className, state: layer.state, metaState: layer.metaState)
        try validateFixedConfiguration(layer, template: template)
        return {
            applySavedValues(state: layer.state, metaState: layer.metaState, to: template)
            return template
        }
    }

    /// Makes the step for a saved layer whose class is not the class of its template.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - template: The fresh cache that the model made for this layer.
    ///   - templateClassName: The class name of `template`.
    /// - Returns: The step, which gives back a new cache of the saved class.
    /// - Throws: ``KVCacheError`` unless the template is a `KVCacheSimple` and the saved class
    ///   is a class that generation converts it to.
    private static func prepareConverted(
        _ layer: SavedPromptCacheLayer, into template: KVCache, templateClassName: String
    ) throws -> Step {
        guard templateClassName == "KVCache", convertedClassNames.contains(layer.className)
        else {
            throw KVCacheError(
                message:
                    "The prompt cache holds a \(layer.className) layer and the model gave a \(templateClassName) cache."
            )
        }
        let converted = try restoreCacheFromMetaState(
            className: layer.className, state: layer.state, metaState: layer.metaState)
        return { converted }
    }

    /// Makes the step for a saved `CacheList` layer. Each child restores into the child of the
    /// template at the same place.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - list: The template list.
    /// - Returns: The step, which gives back `list`.
    /// - Throws: ``KVCacheError`` when the children do not fit the children of `list`.
    private static func prepareCacheList(
        _ layer: SavedPromptCacheLayer, into list: CacheList
    ) throws -> Step {
        let savedChildren = try CacheList.savedChildren(
            state: layer.state, metaState: layer.metaState)
        let templateChildren = list.children
        guard savedChildren.count == templateChildren.count else {
            throw KVCacheError(
                message:
                    "The prompt cache holds a CacheList of \(savedChildren.count) children and the model gave \(templateChildren.count)."
            )
        }
        let childSteps = try zip(savedChildren, templateChildren).map { saved, child in
            try prepare(
                SavedPromptCacheLayer(
                    className: saved.className, state: saved.state, metaState: saved.metaState),
                into: child)
        }
        return {
            list.replaceChildren(with: childSteps.map { $0() })
            return list
        }
    }

    /// Checks that the saved configuration values equal the values of the template.
    ///
    /// - Parameters:
    ///   - layer: The saved layer. Its values passed the check of its class.
    ///   - template: The fresh cache that the model made for this layer.
    /// - Throws: ``KVCacheError`` when a configuration value is different.
    private static func validateFixedConfiguration(
        _ layer: SavedPromptCacheLayer, template: KVCache
    ) throws {
        guard let indices = fixedConfigurationIndices[layer.className] else { return }
        let templateMetaState = template.metaState
        let matches = indices.allSatisfy { index in
            index < layer.metaState.count && index < templateMetaState.count
                && layer.metaState[index] == templateMetaState[index]
        }
        guard matches else {
            throw KVCacheError(
                message:
                    "The saved \(layer.className) configuration is not the configuration of the model cache."
            )
        }
    }
}

/// The record of the offset of each top-level cache in a prompt cache file.
///
/// The `metaState` of some classes does not hold the offset: an `ArraysCache` or a
/// `MambaCache` comes back at offset 0, and a `ChunkedKVCache` after a front trim comes back at
/// the number of rows that it keeps. The save thus writes the offset of each top-level cache in
/// the user-metadata part, under a reserved key prefix. The load removes the record from the
/// user metadata that it gives back, and applies it after the restore. A file without the record
/// loads as before.
private enum PromptCacheOffsetRecord {
    /// The reserved prefix of the metadata keys of the record. User metadata cannot use it.
    static let metadataPrefix = "__mlx_lm_offset_"

    /// The metadata key of the record of one layer.
    ///
    /// - Parameter layer: The index of the top-level cache.
    /// - Returns: The key, without the prefix of the user-metadata part.
    static func key(layer: Int) -> String {
        "\(metadataPrefix)\(layer)"
    }

    /// Refuses user metadata that uses the reserved prefix.
    ///
    /// - Parameter metadata: The caller metadata of a save.
    /// - Throws: ``KVCacheError`` when a key starts with ``metadataPrefix``.
    static func validateUserMetadata(_ metadata: [String: String]) throws {
        guard !metadata.keys.contains(where: { $0.hasPrefix(metadataPrefix) }) else {
            throw KVCacheError(
                message: "User metadata uses the reserved prompt cache offset namespace")
        }
    }

    /// Makes the record of a list of caches.
    ///
    /// - Parameter caches: The top-level caches of a save.
    /// - Returns: One metadata entry for each cache.
    static func entries(for caches: [KVCache]) -> [String: String] {
        Dictionary(
            uniqueKeysWithValues: caches.enumerated().map { layer, cache in
                (key(layer: layer), String(cache.offset))
            })
    }

    /// Reads the record from the user metadata of a file, and removes it.
    ///
    /// - Parameters:
    ///   - metadata: The user metadata of the file, which can hold the record.
    ///   - layerCount: The number of top-level caches in the file.
    /// - Returns: The offset of each layer, or `nil` when the file has no record, and the user
    ///   metadata without the record.
    /// - Throws: ``KVCacheError`` when the record does not have exactly one non-negative integer
    ///   for each layer.
    static func read(
        from metadata: [String: String], layerCount: Int
    ) throws -> (offsets: [Int]?, userMetadata: [String: String]) {
        let record = metadata.filter { $0.key.hasPrefix(metadataPrefix) }
        let userMetadata = metadata.filter { !$0.key.hasPrefix(metadataPrefix) }
        guard !record.isEmpty else { return (nil, userMetadata) }
        guard record.count == layerCount else {
            throw KVCacheError(
                message:
                    "The prompt cache offset record has \(record.count) entries for \(layerCount) caches."
            )
        }
        let offsets = try (0 ..< layerCount).map { layer in
            guard let value = record[key(layer: layer)], let offset = Int(value), offset >= 0
            else {
                throw KVCacheError(
                    message: "The prompt cache offset record of cache \(layer) is not valid.")
            }
            return offset
        }
        return (offsets, userMetadata)
    }

    /// Applies the record to the restored caches.
    ///
    /// `KVCache.offset` has no setter, and some classes compute it from inner caches. Thus the
    /// function sets the offset only of a class whose saved values do not hold it. For each
    /// other class it checks that the restored offset equals the recorded offset.
    ///
    /// - Parameters:
    ///   - offsets: The recorded offsets, or `nil` when the file has no record.
    ///   - caches: The restored top-level caches, in the order of the record.
    /// - Throws: ``KVCacheError`` when a restored offset is not the recorded offset.
    static func apply(_ offsets: [Int]?, to caches: [KVCache]) throws {
        guard let offsets else { return }
        for (layer, (cache, offset)) in zip(caches, offsets).enumerated() {
            if let settable = offsetSettableCache(cache) {
                settable.offset = offset
            } else if cache.offset != offset {
                throw KVCacheError(
                    message:
                        "The restored cache \(layer) has offset \(cache.offset) and the prompt cache records \(offset)."
                )
            }
        }
    }

    /// Gives the cache as a `BaseKVCache` when its saved values do not hold its offset.
    ///
    /// - Parameter cache: A restored cache.
    /// - Returns: The cache for an `ArraysCache` (thus also a `MambaCache`) or a
    ///   `ChunkedKVCache`, and `nil` for each other class.
    private static func offsetSettableCache(_ cache: KVCache) -> BaseKVCache? {
        switch cache {
        case let arrays as ArraysCache: arrays
        case let chunked as ChunkedKVCache: chunked
        default: nil
        }
    }
}

private func promptCacheStateArrays(
    _ state: LMOutput.State?, userMetadata: [String: String]
) throws -> [(key: String, value: MLXArray)] {
    guard !userMetadata.keys.contains(where: { $0.hasPrefix(promptCacheStateMetadataPrefix) })
    else {
        throw KVCacheError(message: "User metadata uses the reserved prompt cache state namespace")
    }
    guard let state else { return [] }
    return try state.serializedArrays().sorted { $0.key < $1.key }
}

private func addPromptCacheState(
    _ stateArrays: [(key: String, value: MLXArray)],
    flattenedData: inout [String: MLXArray], flattenedMetadata: inout [String: String]
) {
    guard !stateArrays.isEmpty else { return }

    flattenedMetadata["1.\(promptCacheStateVersionKey)"] = promptCacheStateFormatVersion
    flattenedMetadata["1.\(promptCacheStateCountKey)"] = String(stateArrays.count)
    for (index, entry) in stateArrays.enumerated() {
        flattenedData[promptCacheStateTensorKey(index)] = entry.value
        flattenedMetadata["1.\(promptCacheStateEntryKey(index))"] = entry.key
    }
}

private func loadPromptCacheState(
    arrays: inout [String: MLXArray], metadata: [String: String]
) throws -> (LMOutput.State?, [String: String]) {
    let stateMetadata = metadata.filter { $0.key.hasPrefix(promptCacheStateMetadataPrefix) }
    let stateTensorKeys = arrays.keys.filter { $0.hasPrefix(promptCacheStateTensorPrefix) }
    guard !stateMetadata.isEmpty || !stateTensorKeys.isEmpty else { return (nil, metadata) }

    guard metadata[promptCacheStateVersionKey] == promptCacheStateFormatVersion else {
        throw KVCacheError(message: "Unsupported prompt cache state format")
    }
    guard let countValue = metadata[promptCacheStateCountKey],
        let count = Int(countValue), count > 0
    else {
        throw KVCacheError(message: "Invalid prompt cache state count")
    }

    var serializedArrays: [String: MLXArray] = [:]
    for index in 0 ..< count {
        let tensorKey = promptCacheStateTensorKey(index)
        guard let key = metadata[promptCacheStateEntryKey(index)],
            let array = arrays.removeValue(forKey: tensorKey)
        else {
            throw KVCacheError(message: "Invalid prompt cache state entry at index \(index)")
        }
        guard serializedArrays.updateValue(array, forKey: key) == nil else {
            throw KVCacheError(message: "Duplicate prompt cache state key: \(key)")
        }
    }

    // The loop proved every expected entry is present; counting rejects the leftovers — a
    // reserved-namespace key this reader does not know about means the file was written by
    // something else, so refuse it rather than silently dropping part of the state.
    guard stateMetadata.count == count + 2 else {
        throw KVCacheError(message: "Unexpected prompt cache state metadata")
    }
    guard stateTensorKeys.count == count else {
        throw KVCacheError(message: "Unexpected prompt cache state tensors")
    }

    let userMetadata = metadata.filter { !$0.key.hasPrefix(promptCacheStateMetadataPrefix) }
    return (LMOutput.State(serializedArrays: serializedArrays), userMetadata)
}

/// Prefixes the stored cache class name when the file carries model state.
///
/// The marker is redundant for this reader — the state metadata already says whether state is
/// present — and exists for older ones. A reader predating model state would ignore the
/// `__mlx_lm_state_` metadata entirely and restore the cache without its continuation anchor,
/// which positions new tokens as if the cached prefix held no images. Poisoning the class name
/// makes that reader fail with "Unknown cache class" instead of continuing at the wrong offsets.
private func promptCacheClassName(_ className: String, hasState: Bool) -> String {
    hasState ? "\(promptCacheStateClassPrefix)\(className)" : className
}

private func loadPromptCacheClasses(_ classNames: [String], hasState: Bool) throws -> [String] {
    if hasState {
        guard !classNames.isEmpty,
            classNames.allSatisfy({ $0.hasPrefix(promptCacheStateClassPrefix) })
        else {
            throw KVCacheError(
                message: "Prompt cache model state is missing its compatibility marker")
        }
        return classNames.map { String($0.dropFirst(promptCacheStateClassPrefix.count)) }
    }

    guard !classNames.contains(where: { $0.hasPrefix(promptCacheStateClassPrefix) }) else {
        throw KVCacheError(message: "Prompt cache compatibility marker has no model state")
    }
    return classNames
}

private func promptCacheStateEntryKey(_ index: Int) -> String {
    "\(promptCacheStateMetadataPrefix)\(index)_key"
}

private func promptCacheStateTensorKey(_ index: Int) -> String {
    "\(promptCacheStateTensorPrefix)\(index)"
}

private let promptCacheStateFormatVersion = "1"
private let promptCacheStateMetadataPrefix = "__mlx_lm_state_"
private let promptCacheStateVersionKey = "__mlx_lm_state_version"
private let promptCacheStateCountKey = "__mlx_lm_state_count"
private let promptCacheStateTensorPrefix = "__mlx_lm_state_tensor_"
// Derived so the format version lives in exactly one place.
private let promptCacheStateClassPrefix =
    "__mlx_lm_state_v\(promptCacheStateFormatVersion)__:"

/// Reconstruct a single cache from its class name, state arrays, and metaState.
///
/// Like Python's `globals()[className].from_state(state, meta_state)`, each cache type
/// encodes enough info in `metaState` to reconstruct itself.
private func restoreCacheFromMetaState(
    className: String,
    state: [MLXArray],
    metaState: [String]
) throws -> KVCache {
    if className == "CacheList" {
        return try CacheList.fromState(state: state, metaState: metaState)
    }
    guard builtInLeafClassNames.contains(className) else {
        if let restored = KVCacheSerializationRegistry.restore(
            className: className, state: state, metaState: metaState)
        {
            return restored
        }
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
    try validateBuiltInCache(className: className, state: state, metaState: metaState)
    let cache = try makeEmptyBuiltInCache(className: className, metaState: metaState)
    applySavedValues(state: state, metaState: metaState, to: cache)
    return cache
}

/// The class names of the built-in caches that hold no child cache. `KVCacheSimple` is the
/// name that older files wrote for `KVCache`.
private let builtInLeafClassNames: Set<String> = [
    "KVCache", "KVCacheSimple", "RotatingKVCache", "QuantizedKVCache",
    "VarianceNormalizedKVCache", "ChunkedKVCache", "MambaCache", "ArraysCache",
    "TurboQuantKVCache",
]

/// The places of the values in the saved meta state of the built-in classes.
private enum SavedMetaStateIndex {
    /// The window size of a `RotatingKVCache`.
    static let rotatingMaxSize = 1
    /// The number of integer values at the start of a `RotatingKVCache` meta state.
    static let rotatingIntegerCount = 5
    /// The capacity origin of a `RotatingKVCache`, when the meta state holds one.
    static let rotatingCapacityOrigin = 5
    /// The wrapped flag of a `RotatingKVCache`, when the meta state holds one.
    static let rotatingWrapped = 6
    /// The group size of a `QuantizedKVCache`.
    static let quantizedGroupSize = 2
    /// The bit width of a `QuantizedKVCache`.
    static let quantizedBits = 3
    /// The chunk size of a `ChunkedKVCache`.
    static let chunkedChunkSize = 0
    /// The start position of a `ChunkedKVCache`.
    static let chunkedStartPosition = 1
    /// The bit width of a `TurboQuantKVCache`.
    static let turboQuantBits = 1
    /// The key bit width of a `TurboQuantKVCache`.
    static let turboQuantKeyBits = 2
    /// The value bit width of a `TurboQuantKVCache`.
    static let turboQuantValueBits = 3
    /// The seed of a `TurboQuantKVCache`.
    static let turboQuantSeed = 4
    /// The slot count of an `ArraysCache`.
    static let arraysSlotCount = 0
    /// The present slots of an `ArraysCache`.
    static let arraysPresentSlots = 1
}

/// The array counts and meta-state counts that the built-in classes save.
private enum SavedValueCounts {
    /// The array count of a cache that holds no value yet.
    private static let noArrays = 0
    /// The array count of the keys and the values.
    private static let keysAndValues = 2
    /// The array count of quantized keys and values without biases: the packed values and
    /// the scales of each.
    private static let quantizedWithoutBiases = 4
    /// The array count of quantized keys and values with biases.
    private static let quantizedWithBiases = 6
    /// The meta-state count of a `QuantizedKVCache`: the step, the offset, the group size and
    /// the bit width.
    private static let quantizedMetaStateCount = 4
    /// The meta-state count of a `ChunkedKVCache`: the chunk size and the start position.
    private static let chunkedMetaStateCount = 2
    /// The smallest meta-state count of an `ArraysCache`: the slot count and the present slots.
    private static let arraysMinimumMetaStateCount = 2
    /// The largest meta-state count of an `ArraysCache`: the left padding and the lengths
    /// follow.
    private static let arraysMaximumMetaStateCount = 4
    /// The rank of a saved key, value or quantized tensor: `[batch, heads, tokens, width]`.
    static let tensorRank = 4

    /// No arrays, or the keys and the values.
    static let keyValueStates: Set<Int> = [noArrays, keysAndValues]
    /// The one empty placeholder of a `KVCacheSimple` meta state.
    static let simpleMetaStates: Set<Int> = [1]
    /// Five integers, then an optional capacity origin and an optional wrapped flag.
    static let rotatingMetaStates: Set<Int> = [
        SavedMetaStateIndex.rotatingIntegerCount, SavedMetaStateIndex.rotatingCapacityOrigin + 1,
        SavedMetaStateIndex.rotatingWrapped + 1,
    ]
    /// No arrays, or the packed values and the scales of keys and values, with or without
    /// biases.
    static let quantizedStates: Set<Int> = [noArrays, quantizedWithoutBiases, quantizedWithBiases]
    /// The step, the offset, the group size and the bit width.
    static let quantizedMetaStates: Set<Int> = [quantizedMetaStateCount]
    /// The chunk size and the start position.
    static let chunkedMetaStates: Set<Int> = [chunkedMetaStateCount]
    /// The slot count, the present slots, then optional left padding and lengths.
    static let arraysMetaStates = arraysMinimumMetaStateCount ... arraysMaximumMetaStateCount
}

/// Checks the saved values of one built-in cache class before any setter reads them.
///
/// - Parameters:
///   - className: The saved class name.
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class, or the class is unknown.
private func validateBuiltInCache(
    className: String, state: [MLXArray], metaState: [String]
) throws {
    switch className {
    case "KVCache", "KVCacheSimple":
        try validateSimpleCache(state: state, metaState: metaState)
    case "RotatingKVCache":
        try validateRotatingCache(state: state, metaState: metaState)
    case "QuantizedKVCache":
        try validatePromptCache(
            className: className, state: state, stateCounts: SavedValueCounts.quantizedStates,
            metadata: metaState, metadataCounts: SavedValueCounts.quantizedMetaStates)
        _ = try promptCacheIntegers(metaState, className: className)
    case "VarianceNormalizedKVCache":
        try validateVarianceNormalizedCache(state: state, metaState: metaState)
    case "ChunkedKVCache":
        try validateChunkedCache(state: state, metaState: metaState)
    case "MambaCache", "ArraysCache":
        try validateArraysCache(className: className, state: state, metaState: metaState)
    case "TurboQuantKVCache":
        try validateTurboQuantCache(state: state, metaState: metaState)
    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

/// Checks the saved values of a `KVCacheSimple`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateSimpleCache(state: [MLXArray], metaState: [String]) throws {
    try validatePromptCache(
        className: "KVCacheSimple", state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.simpleMetaStates)
    guard metaState == [""] else {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: KVCacheSimple metadata must contain its single empty placeholder."
        )
    }
}

/// Checks the saved values of a `RotatingKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateRotatingCache(state: [MLXArray], metaState: [String]) throws {
    let className = "RotatingKVCache"
    try validatePromptCache(
        className: className, state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.rotatingMetaStates)
    _ = try promptCacheIntegers(
        metaState.prefix(SavedMetaStateIndex.rotatingIntegerCount), className: className)
    let originIndex = SavedMetaStateIndex.rotatingCapacityOrigin
    if metaState.count > originIndex,
        RotatingKVCache.CapacityOrigin(rawValue: metaState[originIndex]) == nil
    {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: invalid RotatingKVCache capacity origin '\(metaState[originIndex])'."
        )
    }
    let wrappedIndex = SavedMetaStateIndex.rotatingWrapped
    if metaState.count > wrappedIndex, Bool(metaState[wrappedIndex]) == nil {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: invalid RotatingKVCache wrapped flag '\(metaState[wrappedIndex])'."
        )
    }
}

/// Checks the saved values of a `VarianceNormalizedKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateVarianceNormalizedCache(state: [MLXArray], metaState: [String]) throws {
    let layout = try VarianceNormalizedSavedLayout(metaState: metaState)
    let tailStateCount =
        layout.tailLength > 0 ? VarianceNormalizedKVCache.tailStateCount : 0
    let tileStateCount = state.count - min(state.count, tailStateCount)
    let hasValidTileStateCount =
        if layout.tileCount == 0 {
            tileStateCount == 0
        } else {
            tileStateCount.isMultiple(of: layout.tileCount)
                && [
                    VarianceNormalizedKVCache.compactTileStateCount,
                    VarianceNormalizedKVCache.legacyTileStateCount,
                ].contains(tileStateCount / layout.tileCount)
        }
    guard
        layout.hasConsistentOffset,
        state.count >= tailStateCount,
        hasValidTileStateCount,
        state.allSatisfy({ $0.ndim == SavedValueCounts.tensorRank })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid VarianceNormalizedKVCache state."
        )
    }
}

/// The numbers of a saved `VarianceNormalizedKVCache` meta state.
private struct VarianceNormalizedSavedLayout {
    /// The meta-state count of the legacy layout.
    private static let legacyCount = 7
    /// The meta-state count of the versioned layout.
    private static let versionedCount = 10
    /// The place of the metadata version in the versioned layout.
    private static let versionIndex = 7
    /// The place of the key element type in the versioned layout.
    private static let keyDTypeIndex = 8
    /// The place of the value element type in the versioned layout.
    private static let valueDTypeIndex = 9
    /// The name that the versioned layout writes for "no element type yet".
    private static let noDType = "none"
    /// The place of the tile size.
    fileprivate static let tileSizeIndex = 0
    /// The place of the offset.
    private static let offsetIndex = 1
    /// The place of the key bit width.
    fileprivate static let keyBitsIndex = 2
    /// The place of the value bit width.
    fileprivate static let valueBitsIndex = 3
    /// The place of the Sinkhorn iteration count.
    fileprivate static let sinkhornIterationsIndex = 4
    /// The place of the complete tile count.
    private static let tileCountIndex = 5
    /// The place of the raw tail length.
    private static let tailLengthIndex = 6

    /// The number of tokens in one tile.
    let tileSize: Int
    /// The number of tokens that the cache holds.
    let offset: Int
    /// The key bit width.
    let keyBits: Int
    /// The value bit width.
    let valueBits: Int
    /// The Sinkhorn iteration count.
    let sinkhornIterations: Int
    /// The number of complete tiles.
    let tileCount: Int
    /// The number of tokens in the raw tail.
    let tailLength: Int

    /// Reads the numbers and checks the configuration and the element-type names.
    ///
    /// - Parameter metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when the meta state is not a valid layout.
    init(metaState: [String]) throws {
        let className = "VarianceNormalizedKVCache"
        guard metaState.count == Self.legacyCount || metaState.count == Self.versionedCount
        else {
            throw KVCacheError(
                message:
                    "Corrupt prompt cache: VarianceNormalizedKVCache metadata must contain 7 legacy or 10 versioned values."
            )
        }
        let values = try promptCacheIntegers(
            Array(metaState.prefix(Self.legacyCount)), className: className)
        tileSize = values[Self.tileSizeIndex]
        offset = values[Self.offsetIndex]
        keyBits = values[Self.keyBitsIndex]
        valueBits = values[Self.valueBitsIndex]
        sinkhornIterations = values[Self.sinkhornIterationsIndex]
        tileCount = values[Self.tileCountIndex]
        tailLength = values[Self.tailLengthIndex]
        guard
            (try? VarianceNormalizedKVCacheConfiguration(
                keyBits: keyBits,
                valueBits: valueBits,
                tileSize: tileSize,
                sinkhornIterations: sinkhornIterations)) != nil,
            offset >= 0,
            tileCount >= 0,
            (0 ..< tileSize).contains(tailLength),
            metaState.count == Self.legacyCount || Self.hasValidVersionFields(metaState)
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid VarianceNormalizedKVCache metadata."
            )
        }
    }

    /// True when the offset equals the tokens of the tiles and the tail, with no overflow.
    var hasConsistentOffset: Bool {
        let (tiledLength, tileLengthOverflow) = tileCount.multipliedReportingOverflow(
            by: tileSize)
        let (representedLength, offsetOverflow) = tiledLength.addingReportingOverflow(tailLength)
        return !tileLengthOverflow && !offsetOverflow && offset == representedLength
    }

    /// Checks the version and the element-type names of a versioned layout.
    ///
    /// - Parameter metaState: The saved meta state, which holds the versioned count.
    /// - Returns: True when the version is known and each element-type name is supported.
    private static func hasValidVersionFields(_ metaState: [String]) -> Bool {
        let isSupportedName = { (name: String) in
            name == noDType
                || varianceNormalizedDType(named: name).map(isSupportedVarianceNormalizedDType)
                    == true
        }
        return Int(metaState[versionIndex]) == VarianceNormalizedKVCache.metadataVersion
            && isSupportedName(metaState[keyDTypeIndex])
            && isSupportedName(metaState[valueDTypeIndex])
    }
}

/// Checks the saved values of a `ChunkedKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateChunkedCache(state: [MLXArray], metaState: [String]) throws {
    let className = "ChunkedKVCache"
    try validatePromptCache(
        className: className, state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.chunkedMetaStates)
    _ = try savedChunkSize(metaState)
    _ = try promptCacheInteger(
        metaState[SavedMetaStateIndex.chunkedStartPosition], className: className)
}

/// Reads the chunk size of a saved `ChunkedKVCache`.
///
/// - Parameter metaState: The saved meta state, which holds two values.
/// - Returns: The chunk size, or `nil` for a cache with no chunk size.
/// - Throws: ``KVCacheError`` when the value is not an integer or `None`.
private func savedChunkSize(_ metaState: [String]) throws -> Int? {
    let value = metaState[SavedMetaStateIndex.chunkedChunkSize]
    return value == "None" ? nil : try promptCacheInteger(value, className: "ChunkedKVCache")
}

/// Checks the saved values of an `ArraysCache` or a `MambaCache`.
///
/// The legacy meta state `[""]` holds no slot list. The restore then writes the arrays into
/// the slots in order, thus any array count fits it.
///
/// - Parameters:
///   - className: The saved class name.
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateArraysCache(
    className: String, state: [MLXArray], metaState: [String]
) throws {
    guard metaState != [""] else { return }
    let slotIndex = SavedMetaStateIndex.arraysSlotCount
    let presentIndex = SavedMetaStateIndex.arraysPresentSlots
    guard SavedValueCounts.arraysMetaStates.contains(metaState.count) else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid \(className) state or metadata shape.")
    }
    let slotCount = try promptCacheInteger(metaState[slotIndex], className: className)
    let presentSlots =
        metaState[presentIndex].isEmpty
        ? []
        : try metaState[presentIndex].split(separator: ",").map {
            try promptCacheInteger(String($0), className: className)
        }
    guard slotCount >= 0, presentSlots.count == state.count,
        Set(presentSlots).count == presentSlots.count,
        presentSlots.allSatisfy({ (0 ..< slotCount).contains($0) })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: \(className) slots do not fit its arrays.")
    }
}

/// Checks the saved values of a `TurboQuantKVCache`.
///
/// The array count depends on the key mode that the key bit width selects: 2 raw arrays in
/// every mode, 3 in raw-key mode, 5 in affine-key mode, and 4 or 5 in the standard mode.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateTurboQuantCache(state: [MLXArray], metaState: [String]) throws {
    let configuration = try TurboQuantSavedConfiguration(metaState: metaState)
    let acceptedCounts = configuration.acceptedStateCounts
    // The setter reads the token axis of the first array, and of the second array in
    // raw-key mode.
    let firstRank = 4
    let secondMinimumRank = 3
    guard
        state.isEmpty
            || (acceptedCounts.contains(state.count)
                && state[0].ndim == firstRank && state[1].ndim >= secondMinimumRank)
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid TurboQuantKVCache state or metadata shape.")
    }
}

/// The configuration numbers of a saved `TurboQuantKVCache` meta state.
private struct TurboQuantSavedConfiguration {
    /// The meta-state count: the offset, the bit width, the key and value bit widths, and the
    /// seed.
    private static let metaStateCount = 5

    /// The array count of a cache that holds raw keys and raw values, in every key mode.
    private static let rawStateCount = 2

    /// The key bit width that selects raw-key mode.
    private static let rawKeyBits = 0

    /// The key bit width that selects affine-key mode.
    private static let affineKeyBits = 8

    /// The compressed array count of raw-key mode: the raw keys, the packed values and the
    /// value norms.
    private static let rawKeyCompressedCount = 3

    /// The compressed array count of the standard mode: the packed keys, the key norms, the
    /// packed values and the value norms.
    private static let standardCompressedCount = 4

    /// The compressed array count of affine-key mode, and of the standard mode with the key
    /// calibration scale.
    private static let fiveArrayCompressedCount = 5

    /// The compressed array counts of the modes that a key bit width selects.
    private static let compressedStateCountsByKeyBits: [Int: Set<Int>] = [
        rawKeyBits: [rawKeyCompressedCount], affineKeyBits: [fiveArrayCompressedCount],
    ]

    /// The compressed array counts of the standard mode, with and without the key calibration
    /// scale.
    private static let standardCompressedStateCounts: Set<Int> = [
        standardCompressedCount, fiveArrayCompressedCount,
    ]

    /// The bit width.
    let bits: Int
    /// The key bit width.
    let keyBits: Int
    /// The value bit width.
    let valueBits: Int
    /// The seed of the rotation.
    let seed: UInt64

    /// Reads the configuration numbers.
    ///
    /// - Parameter metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when the meta state is not a valid layout.
    init(metaState: [String]) throws {
        guard metaState.count == Self.metaStateCount,
            Int(metaState[0]) != nil,
            let bits = Int(metaState[SavedMetaStateIndex.turboQuantBits]),
            let keyBits = Int(metaState[SavedMetaStateIndex.turboQuantKeyBits]),
            let valueBits = Int(metaState[SavedMetaStateIndex.turboQuantValueBits]),
            let seed = UInt64(metaState[SavedMetaStateIndex.turboQuantSeed])
        else {
            throw KVCacheError(message: "Invalid TurboQuantKVCache metaState")
        }
        (self.bits, self.keyBits, self.valueBits, self.seed) = (bits, keyBits, valueBits, seed)
    }

    /// The array counts that the state setter accepts for the key mode of this configuration.
    var acceptedStateCounts: Set<Int> {
        let compressed =
            Self.compressedStateCountsByKeyBits[keyBits] ?? Self.standardCompressedStateCounts
        return compressed.union([Self.rawStateCount])
    }
}

/// Makes the empty built-in cache that a saved layer describes. The values passed the check
/// of ``validateBuiltInCache(className:state:metaState:)``.
///
/// - Parameters:
///   - className: The saved class name.
///   - metaState: The saved meta state.
/// - Returns: The empty cache, with the configuration of the saved meta state.
/// - Throws: ``KVCacheError`` when a configuration value cannot be read.
private func makeEmptyBuiltInCache(className: String, metaState: [String]) throws -> KVCache {
    switch className {
    case "RotatingKVCache":
        return RotatingKVCache(
            maxSize: try promptCacheInteger(
                metaState[SavedMetaStateIndex.rotatingMaxSize], className: className))
    case "QuantizedKVCache":
        let values = try promptCacheIntegers(metaState, className: className)
        return QuantizedKVCache(
            groupSize: values[SavedMetaStateIndex.quantizedGroupSize],
            bits: values[SavedMetaStateIndex.quantizedBits])
    case "VarianceNormalizedKVCache":
        let layout = try VarianceNormalizedSavedLayout(metaState: metaState)
        return VarianceNormalizedKVCache(
            tileSize: layout.tileSize, keyBits: layout.keyBits, valueBits: layout.valueBits,
            sinkhornIterations: layout.sinkhornIterations)
    case "ChunkedKVCache":
        return ChunkedKVCache(chunkSize: try savedChunkSize(metaState))
    case "MambaCache":
        return MambaCache()
    case "ArraysCache":
        return ArraysCache(size: 0)
    case "TurboQuantKVCache":
        let configuration = try TurboQuantSavedConfiguration(metaState: metaState)
        return TurboQuantKVCache(
            bits: configuration.bits, keyBits: configuration.keyBits,
            valueBits: configuration.valueBits, seed: configuration.seed)
    default:
        return KVCacheSimple()
    }
}

/// Writes checked saved values into a built-in cache, in the order that its setters need.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
///   - cache: The built-in cache that receives the values.
private func applySavedValues(state: [MLXArray], metaState: [String], to cache: KVCache) {
    if let arrays = cache as? ArraysCache {
        arrays.restoreFromMetaState(state: state, savedMetaState: metaState)
        return
    }
    var target = cache
    if cache is VarianceNormalizedKVCache {
        // The state setter reads the tile and tail counts that the meta-state setter writes.
        target.metaState = metaState
        target.state = state
        return
    }
    if !state.isEmpty {
        target.state = state
    }
    target.metaState = metaState
}

private func validatePromptCache(
    className: String,
    state: [MLXArray],
    stateCounts: Set<Int>,
    metadata: [String],
    metadataCounts: Set<Int>
) throws {
    guard stateCounts.contains(state.count), metadataCounts.contains(metadata.count),
        state.allSatisfy({ $0.ndim == SavedValueCounts.tensorRank })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid \(className) state or metadata shape."
        )
    }
}

private func promptCacheInteger(_ value: String, className: String) throws -> Int {
    guard let value = Int(value) else {
        throw KVCacheError(
            message: "Corrupt prompt cache: \(className) metadata must contain integers."
        )
    }
    return value
}

private func promptCacheIntegers(
    _ metadata: some Collection<String>, className: String
) throws -> [Int] {
    try metadata.map { try promptCacheInteger($0, className: className) }
}

/// Unflatten arrays from tree_flatten format (e.g., "0.1", "1.0") to nested structure
private func unflattenArrays(
    _ flatArrays: [String: MLXArray],
    cacheCount: Int
) throws -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        guard components.count == 2,
            let i = Int(components[0]),
            let j = Int(components[1]),
            (0 ..< cacheCount).contains(i),
            j >= 0
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid array key '\(key)'.")
        }
        arrayMap[i, default: [:]][j] = array
    }

    return try (0 ..< cacheCount).map { cacheIndex in
        guard let arrays = arrayMap[cacheIndex], !arrays.isEmpty else { return [] }
        var result: [MLXArray] = []
        result.reserveCapacity(arrays.count)
        for arrayIndex in 0 ..< arrays.count {
            guard let array = arrays[arrayIndex] else {
                throw KVCacheError(
                    message:
                        "Corrupt prompt cache: cache \(cacheIndex) has non-contiguous array indices."
                )
            }
            result.append(array)
        }
        return result
    }
}

/// Unflatten metadata from tree_flatten format to nested structure
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or a model-defined
///   cache size is invalid.
public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) throws -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return try model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) throws -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return try makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
///
/// - Throws: ``KVCacheConfigurationError`` when `maxKVSize` is invalid.
public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) throws -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return try (0 ..< numLayers).map { _ in
        try makeAttentionKVCache(parameters: parameters)
    }
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}

/// Rewind a one-token speculative tail in a hybrid attention/recurrent cache.
/// Attention entries trim normally; Mamba entries restore the checkpoint the
/// model captured after the round's committed bonus token.
@discardableResult
package func rewindSpeculativePromptCache(
    _ cache: [KVCache], numTokens: Int
) -> Int {
    guard numTokens == 1,
        cache.allSatisfy({ entry in
            if entry.isTrimmable {
                return entry.offset >= numTokens
            }
            return (entry as? MambaCache)?.hasSpeculativeCheckpoint == true
        })
    else { return 0 }

    for entry in cache {
        if entry.isTrimmable {
            guard entry.trim(numTokens) == numTokens else {
                preconditionFailure("Speculative cache validation and rewind diverged")
            }
        } else {
            guard (entry as? MambaCache)?.restoreSpeculativeCheckpoint() == true else {
                preconditionFailure("Missing recurrent speculative checkpoint")
            }
        }
    }
    return numTokens
}

package func discardSpeculativePromptCacheCheckpoints(_ cache: [KVCache]) {
    for case let entry as MambaCache in cache {
        entry.discardSpeculativeCheckpoint()
    }
}

// MARK: - Type Aliases

/// Standard KV cache - alias to KVCacheSimple for compatibility
public typealias StandardKVCache = KVCacheSimple

// MARK: - Quantized Attention Operations

/// The extra score column a learned attention sink adds.
private let attentionSinkColumnCount = 1

/// Softmax attention scores, with an optional learned per-head sink logit.
///
/// This mirrors what mlx's own fast attention does (`mlx/fast.cpp`): the sink
/// logit is prepended as one extra score column, after the mask and before
/// the softmax, and that column is dropped from the weights afterwards. The
/// sink therefore enters the softmax denominator only, and it does not take
/// the attention scale.
///
/// - Parameters:
///   - scores: Attention scores, `[B, nQHeads, L, S]`, or
///     `[B, nKVHeads, nRepeats, L, S]` when GQA split the head axis.
///   - sinks: One logit per query head, or `nil` for a plain softmax.
///   - keyValueHeads: The number of key/value heads.
///   - repeats: The number of query heads per key/value head.
/// - Returns: Attention weights of the shape of `scores`.
private func softmaxWithSinks(
    _ scores: MLXArray, sinks: MLXArray?, keyValueHeads: Int, repeats: Int
) -> MLXArray {
    guard let sinks else { return softmax(scores, axis: -1) }

    let headShape =
        repeats > 1
        ? [1, keyValueHeads, repeats, 1, 1]
        : [1, keyValueHeads * repeats, 1, 1]
    var columnShape = scores.shape
    columnShape[columnShape.count - 1] = attentionSinkColumnCount
    let column = broadcast(
        sinks.asType(scores.dtype).reshaped(headShape), to: columnShape)

    let widened = softmax(concatenated([column, scores], axis: -1), axis: -1)
    return widened[.ellipsis, attentionSinkColumnCount...]
}

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine,
    sinks: MLXArray? = nil
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Apply mask. A masked position takes the most negative finite number of
    // the score dtype -- the fill mlx itself uses on the path that is not
    // quantized -- thus the softmax drives that position to zero. A finite
    // number, and not `-infinity`, keeps a row whose positions are all masked
    // from giving NaN.
    let maskedFill = MLXArray.maskFill(for: scores.dtype)
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, maskedFill)

    case .array(let maskArray):
        scores = applyMask(maskArray, to: scores)

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            scores = applyMask(maskArray, to: scores)
        }

    case .none:
        break
    }

    let attentionWeights = softmaxWithSinks(
        scores, sinks: sinks, keyValueHeads: nKVHeads, repeats: nRepeats)

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output

    // Apply a boolean/additive mask, broadcasting batched masks over the GQA
    // head-group axis: per-sequence masks are `[B, 1, L, S]`, but with
    // `nRepeats > 1` the scores are 5-D `[B, nKVHeads, nRepeats, L, S]`, so a
    // 4-D mask needs an extra axis to line up `B` with the batch dimension.
    func applyMask(_ maskArray: MLXArray, to scores: MLXArray) -> MLXArray {
        var maskArray = maskArray
        if nRepeats > 1 && maskArray.ndim == 4 {
            maskArray = expandedDimensions(maskArray, axis: -3)
        }
        if maskArray.dtype == .bool {
            return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
        } else {
            return scores + maskArray
        }
    }
}

// MARK: - Dynamic Cache Quantization

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Resolve a kvScheme string to (bits, groupSize) for affine quantization.
/// Returns nil for unrecognized schemes (custom schemes handle their own caches).
public func resolveAffineScheme(_ scheme: String?) -> (bits: Int, groupSize: Int)? {
    switch scheme {
    case "affine4": return (4, 64)
    case "affine8": return (8, 64)
    default: return nil
    }
}

/// Converts regular caches to quantized caches when:
/// - kvBits is specified (or kvScheme resolves to a built-in affine scheme)
/// - The cache is not already quantized
/// - The cache offset is greater than quantizedKVStart
///
/// - Parameters:
///   - cache: Array of KV caches to potentially quantize
///   - kvBits: Number of bits for quantization (nil = no quantization)
///   - kvGroupSize: Group size for quantization
///   - quantizedKVStart: Token count threshold to begin quantizing
///   - kvScheme: Scheme selector; overrides kvBits when it names a built-in
///     affine scheme ("affine4", "affine8") or a TurboQuant scheme
///     ("turbo4", "turbo4v2", ...). Unrecognized schemes are left to custom
///     cache implementations and do not quantize here.
public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0,
    kvScheme: String? = nil
) {
    if let kvScheme,
        resolveAffineScheme(kvScheme) == nil,
        resolveTurboScheme(kvScheme) == nil,
        resolveVarianceNormalizedScheme(kvScheme) == nil
    {
        return
    }
    let parameters = GenerateParameters(
        kvBits: kvBits,
        kvGroupSize: kvGroupSize,
        quantizedKVStart: quantizedKVStart,
        kvScheme: kvScheme)
    guard let plan = try? parameters.kvCachePlan() else { return }
    plan.apply(to: &cache)
}

@discardableResult
func maybeAffineQuantizeKVCache(
    cache: inout [KVCache],
    bits: Int,
    groupSize: Int,
    compressionStart: Int
) -> Bool {
    var awaitsCompressionStart = false
    KVCacheTree.rewrite(&cache) { leaf in
        guard case .simple(let simple) = leaf.kind else { return leaf.cache }
        guard simple.offset > compressionStart else {
            awaitsCompressionStart = true
            return simple
        }

        guard let quantized = try? simple.toQuantized(groupSize: groupSize, bits: bits) else {
            return simple
        }
        return quantized
    }
    return !awaitsCompressionStart
}

@discardableResult
func maybeVarianceNormalizeKVCache(
    cache: inout [KVCache],
    keyBits: Int,
    valueBits: Int,
    tileSize: Int,
    sinkhornIterations: Int,
    compressionStart: Int
) -> Bool {
    var awaitsCompressionStart = false
    KVCacheTree.rewrite(&cache) { leaf in
        guard case .simple(let simple) = leaf.kind else { return leaf.cache }
        guard simple.offset > compressionStart else {
            awaitsCompressionStart = true
            return simple
        }

        let state = simple.innerState()
        if state.count >= 2 {
            guard
                supportsVarianceNormalizedKVCache(
                    keyHeadDim: state[0].dim(3),
                    valueHeadDim: state[1].dim(3),
                    tileSize: tileSize)
            else {
                return simple
            }
        }

        return simple.toVarianceNormalized(
            tileSize: tileSize,
            keyBits: keyBits,
            valueBits: valueBits,
            sinkhornIterations: sinkhornIterations)
    }
    return !awaitsCompressionStart
}

// MARK: - Attention Helpers

/// Apply a symbolic or array attention mask to score logits.
func applyAttentionMask(
    scores: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        return MLX.where(causalMask, scores, MLXArray.maskFill(for: scores.dtype))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
        } else {
            return scores + maskArray
        }

    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
            } else {
                return scores + maskArray
            }
        }
        return scores

    case .none:
        return scores
    }
}

func attentionScores(
    queries: MLXArray,
    keys: MLXArray,
    scale: Float
) -> MLXArray {
    let (batchSize, queryHeadCount, queryLength, headDim) = (
        queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
    )
    let kvHeadCount = keys.dim(1)
    let repeats = queryHeadCount / kvHeadCount
    let scaledQueries = queries * scale

    if repeats > 1 {
        let groupedQueries = scaledQueries.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, headDim,
        ])
        let groupedKeys = expandedDimensions(keys, axis: -3)
        return matmul(groupedQueries, groupedKeys.transposed(0, 1, 2, 4, 3))
            .reshaped(batchSize, queryHeadCount, queryLength, keys.dim(2))
    } else {
        return matmul(scaledQueries, keys.transposed(0, 1, 3, 2))
    }
}

func attentionValues(
    weights: MLXArray,
    values: MLXArray,
    queryHeadCount: Int
) -> MLXArray {
    let (batchSize, _, queryLength, keyLength) = (
        weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(3)
    )
    let kvHeadCount = values.dim(1)
    let repeats = queryHeadCount / kvHeadCount

    if repeats > 1 {
        let groupedWeights = weights.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, keyLength,
        ])
        let groupedValues = expandedDimensions(values, axis: -3)
        return matmul(groupedWeights, groupedValues)
            .reshaped(batchSize, queryHeadCount, queryLength, values.dim(3))
    } else {
        return matmul(weights, values)
    }
}
