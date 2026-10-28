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

    /// Number of historical positions this cache no longer retains.
    var evictedTokenCount: Int { get }

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

    /// The arrays the generation loop evaluates with each token.
    ///
    /// ``TokenIterator`` evaluates these arrays at the end of prefill and
    /// after each decode step. The list must hold every lazy array the cache
    /// keeps for a later call, or an array that a later call reads. Decode
    /// runs on another thread than prefill, and MLX cannot evaluate there a
    /// graph that names the stream of the prefill thread.
    ///
    /// This is a requirement and not only an extension member, so that a call
    /// through `any KVCache` goes to the implementation of the dynamic type.
    /// The default implementation is ``state``. A cache whose ``state`` makes
    /// new arrays on each call (slices or placeholders for serialization) can
    /// give the arrays it already holds instead, which adds no operation to
    /// each step.
    var evaluationState: [MLXArray] { get }

    /// The class name written to prompt cache files, or `nil` if this cache cannot be saved.
    ///
    /// Use a unique name, for example with a module prefix. A subclass that does not override
    /// it saves under its parent's name and restores as the parent.
    static var serializedClassName: String? { get }

    /// Rebuild a cache from the `state` and `metaState` it saved.
    static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// Maximum number of newest positions that can be removed with an exact rewind.
    ///
    /// Unlike ``isTrimmable``, this can describe a bounded rewind into retained history.
    var maxTrimCount: Int { get }

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
    public static var serializedClassName: String? { nil }

    public static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        throw KVCacheError(message: "\(Self.self) cannot be restored from a prompt cache file")
    }

    public var ropeOffset: RoPEOffset {
        .scalar(offset)
    }

    public var evictedTokenCount: Int { 0 }

    public func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    public var maxTrimCount: Int { isTrimmable ? offset : 0 }

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

    /// The serialized ``state``, which holds every array a plain cache keeps.
    public var evaluationState: [MLXArray] {
        state
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

/// A cache entry that holds no attention state, such as a record of the tokens processed so far.
///
/// Cache policy skips these entries: capacity and compression settings, offset alignment
/// checks, and ``KVCacheStatus`` layers ignore them. Generation still trims, copies, and saves
/// them with the other entries, so an auxiliary cache must be trimmable to keep pace, and it
/// needs a ``KVCache/serializedClassName`` to be saved to a prompt cache file.
///
/// Staged speculative rounds (MTP) write through an auxiliary entry and trim it back to the
/// accepted positions, so it must stay trimmable after the round's width.
public protocol AuxiliaryKVCache: KVCache {}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0

    public init() {}

    open class var serializedClassName: String? { nil }

    open class func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        throw KVCacheError(message: "\(Self.self) cannot be restored from a prompt cache file")
    }

    public var maxSize: Int? { nil }
    open var evictedTokenCount: Int { 0 }

    /// RoPE offset for this cache. `open` so subclasses can return a non-scalar
    /// offset (e.g. a batched cache's per-row `.batch(...)`).
    open var ropeOffset: RoPEOffset { .scalar(offset) }

    open func innerState() -> [MLXArray] { [] }

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

    open var maxTrimCount: Int { isTrimmable ? offset : 0 }

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
    /// The Python-compatible class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "KVCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "KVCache", state: state, metaState: metaState)
    }

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
        wrappedFlag ?? (idx < (keys?.dim(2) ?? 0) && offset > idx)
    }

    /// Model-native sliding-window caches deliberately keep their architectural
    /// window and do not participate in requested-capacity validation.
    package var capacityOrigin = CapacityOrigin.modelNative

    package var preservedPrefixTokens: Int { keep }

    public override var maxSize: Int? { maxCacheSize }

    public override var evictedTokenCount: Int {
        let retained = wrapped ? (keys?.dim(2) ?? 0) : idx
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

        let available = wrapped ? keys.dim(2) : idx
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
            return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
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
            idx = self.keys!.dim(2)

            // Prefill temporarily retains storageCapacity + S - 1 rows so each
            // query has its attention window and the rewind reserve survives.
            let trimSize = idx - storageCapacity + 1
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
        if self.keys == nil || (!wrapped && idx >= filled && filled < storageCapacity) {
            let newSize = min(step, storageCapacity - filled)

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
        let trimSize = self.keys!.dim(2) - storageCapacity
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
        let count = keys.dim(2)
        if rewindCapacity > 0, count == 0 {
            return logicalView(tail: maxCacheSize) ?? (keys, values)
        }
        let live = wrapped ? (self.keys?.dim(2) ?? 0) : idx
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
            var metadata = [
                String(keep), String(maxCacheSize), String(step), String(offset), String(idx),
                capacityOrigin.rawValue, String(wrapped),
            ]
            if rewindCapacity > 0 { metadata.append(String(rewindCapacity)) }
            return metadata
        }
        set {
            guard (5 ... 8).contains(newValue.count) else {
                fatalError("RotatingKVCache metaState must have 5 to 8 values")
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
            if newValue.count == 8 {
                guard let reserve = Int(newValue[7]), reserve >= 0,
                    maxSizeVal <= Int.max - reserve
                else { fatalError("Invalid RotatingKVCache rewind capacity") }
                self.rewindCapacity = reserve
            } else {
                self.rewindCapacity = 0
            }
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
            if newValue.count >= 7 {
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

    public override var maxTrimCount: Int {
        let live = wrapped ? (keys?.dim(2) ?? 0) : idx
        guard live <= offset else { return 0 }
        return live == offset ? offset : Swift.max(0, live - maxCacheSize)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        // This conservative predicate requires the whole prefix. Counted rewinds
        // can also use the limited history described by `maxTrimCount`.
        let live = wrapped ? (keys?.dim(2) ?? 0) : idx
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
        let live = wrapped ? keys.dim(2) : idx
        // A gap between history and live rows means eviction has occurred. Preserve
        // the pinned prefix regardless of layout, including after repeated trims.
        // Without a gap, an exact rewind can still remove any of the original rows.
        let minimum = offset > live ? Swift.min(keep, live) : 0
        let trimmed = Swift.min(n, live - minimum)
        guard trimmed > 0 else { return 0 }
        let bound = live - trimmed

        if wrapped || keys.dim(2) > storageCapacity {
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
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "QuantizedKVCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "QuantizedKVCache", state: state, metaState: metaState)
    }

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
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "ChunkedKVCache" }

    /// Rebuilds the cache from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        try restoreBuiltInCache(className: "ChunkedKVCache", state: state, metaState: metaState)
    }

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

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
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

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "CacheList" }

    /// Rebuilds the list and its children from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        return try CacheList.fromState(state: state, metaState: metaState)
    }

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

    /// The number of tokens that the list holds: the largest offset of its children.
    ///
    /// Each child moves its own offset when the model feeds tokens through it. The list has no
    /// tokens of its own, thus it reads the offset from its children. The prompt cache compares
    /// the offset of each top-level cache with the length of its token ledger, thus a list whose
    /// offset stays at 0 can never serve a next prompt. An empty list holds no tokens and gives 0.
    ///
    /// The offset cannot be set, because it comes from the children. Set the offset of each child
    /// that does not keep its offset in its state.
    public override var offset: Int {
        get { caches.map(\.offset).max() ?? 0 }
        set {
            preconditionFailure(
                "The offset of a CacheList comes from its children. Set the offset of each child.")
        }
    }

    /// The sum of ``KVCache/residentByteCount`` over the children, so that each
    /// child counts with its own rule.
    public override var residentByteCount: Int {
        caches.reduce(0) { $0 + $1.residentByteCount }
    }

    public override var evictedTokenCount: Int {
        caches.map(\.evictedTokenCount).max() ?? 0
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
            if child is any AuxiliaryKVCache {
                return child
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

    public override var maxTrimCount: Int {
        caches.map(\.maxTrimCount).min() ?? 0
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
                let className =
                    type(of: cache).serializedClassName ?? String(describing: type(of: cache))
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
    static func savedChildren(
        state: [MLXArray], metaState: [String]
    ) throws -> [SavedPromptCacheLayer] {
        guard let childCount = metaState.first.flatMap({ Int($0) }), childCount >= 0 else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }
        var children: [SavedPromptCacheLayer] = []
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
    ) throws -> SavedPromptCacheLayer {
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
        return SavedPromptCacheLayer(
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

/// Whether every layer retains enough history to rewind exactly by `numTokens`.
public func canTrimPromptCache(_ cache: [KVCache], numTokens: Int) -> Bool {
    numTokens >= 0 && cache.allSatisfy { $0.maxTrimCount >= numTokens }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard numTokens > 0, !cache.isEmpty else { return 0 }
    var count = numTokens
    if !canTrimPromptCache(cache, numTokens: count) {
        // Composite offsets are not token positions. Clamp to their leaves only
        // when the caller requested more than the available prefix.
        let offset = KVCacheTree.leaves(in: cache).map { $0.cache.offset }.min() ?? 0
        count = Swift.min(count, offset)
        guard count > 0, canTrimPromptCache(cache, numTokens: count) else { return 0 }
    }
    for entry in cache.dropFirst() { entry.trim(count) }
    return cache.first?.trim(count) ?? 0
}

/// Rewind a one-token speculative tail in a hybrid attention/recurrent cache.
/// Attention entries trim normally; Mamba entries restore the checkpoint the
/// model captured after the round's committed bonus token, or select its
/// snapshot slot. The restored recurrent states are evaluated together, so
/// that the snapshot buffers of the verify pass go without a wait.
@discardableResult
package func rewindSpeculativePromptCache(
    _ cache: [KVCache], numTokens: Int
) -> Int {
    let rewound = rewindSpeculativeEntries(cache, numTokens: numTokens)
    if rewound > 0 {
        asyncEval(cache.compactMap { $0 as? MambaCache }.flatMap(\.state))
    }
    return rewound
}

/// The rewind of ``rewindSpeculativePromptCache(_:numTokens:)``, before the
/// restored states are evaluated.
private func rewindSpeculativeEntries(
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

    // Reshape output for GQA. The head dim follows the values, which MLA-style
    // caches keep narrower than the keys and queries.
    if nRepeats > 1 {
        let vHeadDim = output.dim(-1)
        output = output.reshaped([B, nQHeads, L, vHeadDim])
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
