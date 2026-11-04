// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The narrowest group size that KV cache quantization supports.
private let narrowKVQuantizationGroupSize = 32
/// The group size that KV cache quantization uses unless a request names another one.
private let defaultKVQuantizationGroupSize = 64
/// The widest group size that KV cache quantization supports.
private let wideKVQuantizationGroupSize = 128
/// The group sizes that KV cache quantization supports.
private let supportedKVQuantizationGroupSizes = [
    narrowKVQuantizationGroupSize, defaultKVQuantizationGroupSize, wideKVQuantizationGroupSize,
]

/// The row axis of a `(…, rows, width)` array, counted from the end: the tokens of a quantized
/// key or value array, and the queries of an attention score array.
private let rowAxisFromEnd = -2

/// The axis that grouped-query attention inserts for the query heads that share one key/value
/// head, counted from the end: `[batch, kvHeads, repeats, rows, width]`.
private let groupedQueryRepeatAxis = -3

func resolvedKVQuantizationGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = supportedKVQuantizationGroupSizes.filter {
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

    /// The number of positions that the cache grows its buffers by.
    private static let defaultStep = 256

    /// The position of the biases in a `(packed values, scales, biases)` tuple.
    private static let biasesPosition = 2

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = Self.defaultStep
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
                return concatenated([array, newArray], axis: rowAxisFromEnd)
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
        let B = keys.dim(HeadRowsAxis.batch)
        let nKVHeads = keys.dim(HeadRowsAxis.head)
        let numSteps = keys.dim(HeadRowsAxis.row)
        let kHeadDim = keys.dim(HeadRowsAxis.width)
        let vHeadDim = values.dim(HeadRowsAxis.width)
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
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(rowAxisFromEnd) {
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

            if offset < keys.0.dim(HeadRowsAxis.row) {
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
            case SavedValueCounts.quantizedWithoutBiases, SavedValueCounts.quantizedWithBiases:
                // The keys fill the first half of the arrays and the values the second half.
                let half = newValue.count / SavedValueCounts.keysAndValues
                keys = Self.quantizedTuple(newValue[..<half])
                values = Self.quantizedTuple(newValue[half...])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    /// Read a `(packed values, scales, biases)` tuple from the saved arrays of the keys or of the
    /// values. The arrays hold the biases only when there are three of them.
    ///
    /// - Parameter arrays: The saved arrays of the keys or of the values.
    /// - Returns: The tuple, with `nil` biases when the arrays hold none.
    private static func quantizedTuple(
        _ arrays: ArraySlice<MLXArray>
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let parts = Array(arrays)
        let biases = parts.count > biasesPosition ? parts[biasesPosition] : nil
        return (parts[0], parts[1], biases)
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == SavedValueCounts.quantizedMetaStateCount else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }
            guard
                let offset = Int(newValue[SavedMetaStateIndex.quantizedOffset]),
                let groupSize = Int(newValue[SavedMetaStateIndex.quantizedGroupSize]),
                let bits = Int(newValue[SavedMetaStateIndex.quantizedBits])
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

/// Scaled dot-product attention over quantized keys and values.
///
/// - Parameters:
///   - queries: The queries, `[B, nQHeads, L, D]`.
///   - quantizedKeys: The `(packed values, scales, biases)` of the keys.
///   - quantizedValues: The `(packed values, scales, biases)` of the values.
///   - scale: The attention scale that multiplies the queries.
///   - mask: The attention mask.
///   - groupSize: The quantization group size of the keys and values.
///   - bits: The quantization bit width of the keys and values.
///   - mode: The quantization mode of the keys and values.
///   - sinks: One learned sink logit for each query head, or `nil` for a plain softmax.
/// - Returns: The attention output, `[B, nQHeads, L, vHeadDim]`.
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

    let (B, nQHeads, L, D) = (
        queries.dim(HeadRowsAxis.batch), queries.dim(HeadRowsAxis.head),
        queries.dim(HeadRowsAxis.row), queries.dim(HeadRowsAxis.width)
    )
    let nKVHeads = quantizedKeys.0.dim(HeadRowsAxis.head)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: groupedQueryRepeatAxis),
            expandedDimensions(qKeys.1, axis: groupedQueryRepeatAxis),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: groupedQueryRepeatAxis)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: groupedQueryRepeatAxis),
            expandedDimensions(qValues.1, axis: groupedQueryRepeatAxis),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: groupedQueryRepeatAxis)
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
    switch mask {
    case .causal:
        scores = applyAttentionMask(scores: scores, mask: mask)

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
        if nRepeats > 1 && maskArray.ndim == HeadRowsAxis.count {
            maskArray = expandedDimensions(maskArray, axis: groupedQueryRepeatAxis)
        }
        if maskArray.dtype == .bool {
            return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
        } else {
            return scores + maskArray
        }
    }
}

// MARK: - Dynamic Cache Quantization

/// The bit width of the `affine4` KV cache scheme.
private let affine4SchemeBits = 4
/// The bit width of the `affine8` KV cache scheme.
private let affine8SchemeBits = 8

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Resolve a kvScheme string to (bits, groupSize) for affine quantization.
/// Returns nil for unrecognized schemes (custom schemes handle their own caches).
public func resolveAffineScheme(_ scheme: String?) -> (bits: Int, groupSize: Int)? {
    switch scheme {
    case "affine4": return (affine4SchemeBits, defaultKVQuantizationGroupSize)
    case "affine8": return (affine8SchemeBits, defaultKVQuantizationGroupSize)
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
        if let keys = state.first, let values = state.dropFirst().first {
            guard
                supportsVarianceNormalizedKVCache(
                    keyHeadDim: keys.dim(HeadRowsAxis.width),
                    valueHeadDim: values.dim(HeadRowsAxis.width),
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
        let (qL, kL) = (scores.dim(rowAxisFromEnd), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: rowAxisFromEnd))
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
        queries.dim(HeadRowsAxis.batch), queries.dim(HeadRowsAxis.head),
        queries.dim(HeadRowsAxis.row), queries.dim(HeadRowsAxis.width)
    )
    let kvHeadCount = keys.dim(HeadRowsAxis.head)
    let repeats = queryHeadCount / kvHeadCount
    let scaledQueries = queries * scale

    if repeats > 1 {
        let groupedQueries = scaledQueries.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, headDim,
        ])
        let groupedKeys = expandedDimensions(keys, axis: groupedQueryRepeatAxis)
        return matmul(groupedQueries, groupedKeys.swappedAxes(rowAxisFromEnd, -1))
            .reshaped(batchSize, queryHeadCount, queryLength, keys.dim(HeadRowsAxis.row))
    } else {
        return matmul(scaledQueries, keys.swappedAxes(rowAxisFromEnd, -1))
    }
}

func attentionValues(
    weights: MLXArray,
    values: MLXArray,
    queryHeadCount: Int
) -> MLXArray {
    let (batchSize, _, queryLength, keyLength) = (
        weights.dim(HeadRowsAxis.batch), weights.dim(HeadRowsAxis.head),
        weights.dim(HeadRowsAxis.row), weights.dim(HeadRowsAxis.width)
    )
    let kvHeadCount = values.dim(HeadRowsAxis.head)
    let repeats = queryHeadCount / kvHeadCount

    if repeats > 1 {
        let groupedWeights = weights.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, keyLength,
        ])
        let groupedValues = expandedDimensions(values, axis: groupedQueryRepeatAxis)
        return matmul(groupedWeights, groupedValues)
            .reshaped(batchSize, queryHeadCount, queryLength, values.dim(HeadRowsAxis.width))
    } else {
        return matmul(weights, values)
    }
}
