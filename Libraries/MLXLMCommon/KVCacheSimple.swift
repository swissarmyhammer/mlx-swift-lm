// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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

    /// Grow the buffers by whole steps, so that `keys` and `values` fit after the first `filled`
    /// rows. Only those rows of the old buffers stay.
    ///
    /// - Parameters:
    ///   - keys: The keys that the next write appends.
    ///   - values: The values that the next write appends.
    ///   - filled: The number of rows of the old buffers that hold data.
    func growBuffers(toFit keys: MLXArray, values: MLXArray, after filled: Int) {
        let batchSize = keys.dim(HeadRowsAxis.batch)
        let kvHeads = keys.dim(HeadRowsAxis.head)
        let kHeadDim = keys.dim(HeadRowsAxis.width)
        let vHeadDim = values.dim(HeadRowsAxis.width)

        let nSteps = (step + keys.dim(HeadRowsAxis.row) - 1) / step
        let kShape = [batchSize, kvHeads, nSteps * step, kHeadDim]
        let vShape = [batchSize, kvHeads, nSteps * step, vHeadDim]
        let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
        let newV = MLXArray.zeros(vShape, dtype: values.dtype)

        if var currentKeys = self.keys, var currentValues = self.values {
            if filled % step != 0 {
                currentKeys = currentKeys[.ellipsis, ..<filled, 0...]
                currentValues = currentValues[.ellipsis, ..<filled, 0...]
            }
            self.keys = concatenated([currentKeys, newK], axis: HeadRowsAxis.row)
            self.values = concatenated([currentValues, newV], axis: HeadRowsAxis.row)
        } else {
            self.keys = newK
            self.values = newV
        }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys,
                (previous + keys.dim(HeadRowsAxis.row)) > currentKeys.dim(HeadRowsAxis.row)
            {
                true
            } else {
                self.keys == nil
            }
        if reset {
            growBuffers(toFit: keys, values: values, after: previous)
        }

        self.offset += keys.dim(HeadRowsAxis.row)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(HeadRowsAxis.row) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == SavedValueCounts.keysAndValues else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(HeadRowsAxis.row)
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
                    keyHeadDim: currentKeys.dim(HeadRowsAxis.width),
                    valueHeadDim: currentValues.dim(HeadRowsAxis.width)
                )
            else {
                throw KVCacheError(
                    message:
                        "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(currentKeys.dim(HeadRowsAxis.width)). Value head dim: \(currentValues.dim(HeadRowsAxis.width))."
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
            keys.dim(HeadRowsAxis.row) >= chunkSize
        else { return }

        startPosition += keys.dim(HeadRowsAxis.row) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil
            || (prev + keys.dim(HeadRowsAxis.row)) > self.keys!.dim(HeadRowsAxis.row)
        {
            growBuffers(toFit: keys, values: values, after: prev)
        }

        offset += keys.dim(HeadRowsAxis.row)
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
            guard newValue.count == SavedValueCounts.chunkedMetaStateCount else {
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
