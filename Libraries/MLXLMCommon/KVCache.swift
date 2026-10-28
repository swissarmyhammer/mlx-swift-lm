// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
///
/// See ``KVCache/ropeOffset``.
public enum RoPEOffset {
    /// One offset that every sequence of the batch shares.
    case scalar(Int)
    /// One offset for each sequence of the batch.
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
    /// No class name, so a cache that does not name one cannot be saved to a prompt cache file.
    public static var serializedClassName: String? { nil }

    /// Throws, because a cache that does not override it cannot be restored from a prompt cache
    /// file.
    ///
    /// - Parameters:
    ///   - state: The arrays that the cache saved.
    ///   - metaState: The meta state that the cache saved.
    /// - Throws: ``KVCacheError`` always.
    public static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        throw KVCacheError(message: "\(Self.self) cannot be restored from a prompt cache file")
    }

    /// The scalar ``offset``, which every sequence of the batch shares.
    public var ropeOffset: RoPEOffset {
        .scalar(offset)
    }

    /// No evicted position, because a cache that does not override it keeps its whole history.
    public var evictedTokenCount: Int { 0 }

    /// ``isTrimmable``, for a cache whose trimmability does not depend on what it appends.
    ///
    /// - Parameter positions: The number of sequence positions that would be appended.
    /// - Returns: ``isTrimmable``.
    public func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    /// The whole ``offset`` when the cache is trimmable, and no position when it is not.
    public var maxTrimCount: Int { isTrimmable ? offset : 0 }

    /// Does nothing, because a cache that keeps no batch metadata has nothing to prepare.
    ///
    /// - Parameter lengths: The length of each sequence of the batch.
    public func prepare(lengths: [Int]?) {}

    /// Does nothing, because a cache that keeps no batch metadata has nothing to prepare.
    ///
    /// - Parameter lengths: The length of each sequence of the batch.
    public func prepare(lengths: MLXArray?) {}

    /// Does nothing, because a cache that keeps no batch metadata has nothing to clear.
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

/// Run `body` with each cache prepared for the sequence lengths of a batch, and finalize the
/// caches when `body` returns or throws.
///
/// - Parameters:
///   - cache: The caches to prepare.
///   - lengths: The length of each sequence of the batch, or `nil` to run `body` without
///     preparing the caches.
///   - body: The work that reads the prepared caches.
/// - Returns: The result of `body`.
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

/// Create a boolean causal mask for `n` new positions after `offset` cached positions.
///
/// - Parameters:
///   - n: The number of new positions.
///   - offset: The number of cached positions before them.
///   - windowSize: The sliding window size, or `nil` for full attention.
///   - lengths: The length of each sequence of the batch, or `nil` when no sequence is shorter
///     than the mask.
/// - Returns: A mask that is `true` where a query position may attend to a key position.
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

/// Create an attention mask mode from the parameters of the first cache of `cache`.
///
/// - Parameters:
///   - h: The input array, whose axis 1 holds the new positions.
///   - cache: The caches of the model, or `nil`.
///   - returnArray: Force an array mask instead of the symbolic causal mask.
/// - Returns: `.none` for one new position, otherwise `.causal` or an array mask.
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

/// Create the mask of a state space layer from the batch metadata of its cache.
///
/// - Parameters:
///   - h: The input array, whose axis 1 holds the new positions.
///   - cache: The cache of the layer, or `nil`.
/// - Returns: The mask, or `nil` when there is no cache.
public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
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

/// Rewind the rejected tail of a speculative verify in a hybrid
/// attention/recurrent cache by `numTokens` tokens. Attention entries trim
/// normally; Mamba entries select the snapshot slot the verify kept after its
/// last accepted token. A verify of `T` tokens keeps `T` slots and rewinds
/// one to `T - 1` tokens: the bonus token at its head is always committed.
/// The restored recurrent states are evaluated together, so that the
/// snapshot buffers of the verify pass go without a wait.
///
/// - Parameters:
///   - cache: The cache of each layer.
///   - numTokens: The rejected tokens.
/// - Returns: `numTokens`, or 0 when an entry cannot rewind that many tokens;
///   the cache is then left as it was.
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
    guard numTokens > 0,
        cache.allSatisfy({ entry in
            if entry.isTrimmable {
                return entry.offset >= numTokens
            }
            return ((entry as? MambaCache)?.speculativeSlotCount ?? 0) > numTokens
        })
    else { return 0 }

    for entry in cache {
        if entry.isTrimmable {
            guard entry.trim(numTokens) == numTokens else {
                preconditionFailure("Speculative cache validation and rewind diverged")
            }
        } else {
            guard (entry as? MambaCache)?.selectSpeculativeSlot(rollingBack: numTokens) == true
            else {
                preconditionFailure("Missing recurrent speculative snapshot slot")
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
