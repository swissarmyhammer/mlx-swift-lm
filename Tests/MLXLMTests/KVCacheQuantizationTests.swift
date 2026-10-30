import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

extension KVCacheTests {
    // MARK: - Quantized attention causal masking
    // Regression for the finfo.min mistranslation (masked fill was Float.leastNormalMagnitude ≈ 0).

    @Test
    func testQuantizedAttentionCausalMaskMatchesFullPrecision() throws {
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let (B, nKVHeads, L, D) = (1, 2, 4, 64)
            let scale = 1.0 / Float(D).squareRoot()

            // nRepeats 1 = MHA, 2 = GQA (exercises the 5-D reshape + .causal path that silently corrupts).
            for nRepeats in [1, 2] {
                let nQHeads = nKVHeads * nRepeats
                let q = MLXRandom.normal([B, nQHeads, L, D])
                let k = MLXRandom.normal([B, nKVHeads, L, D])
                let v = MLXRandom.normal([B, nKVHeads, L, D])

                // Reference: full-precision causal attention.
                let reference = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .causal)

                // Path under test: quantized cache + .causal.
                let cache = QuantizedKVCache(groupSize: 64, bits: 8)
                let (qK, qV) = cache.updateQuantized(keys: k, values: v)
                let out = quantizedScaledDotProductAttention(
                    queries: q, quantizedKeys: qK, quantizedValues: qV,
                    scale: scale, mask: .causal,
                    groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

                #expect(out.shape == reference.shape)
                // 8-bit quant error is << 0.1; the bug diverges by O(1).
                let close = allClose(out, reference, rtol: 0.05, atol: 0.1).item(Bool.self)
                #expect(
                    close,
                    "quantized causal attention diverges from full precision (nRepeats=\(nRepeats))"
                )
            }
        }
    }

    @Test("quantizedScaledDotProductAttention preserves the score dtype")
    func preservesScoreDtype() {
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let (B, H, L, D) = (1, 2, 4, 64)
            let scale = 1.0 / Float(D).squareRoot()

            // f32 passes even with a mis-typed fill; f16/bf16 are exactly what a
            // float32 fill silently promotes — so assert the output keeps its dtype.
            for dtype in [DType.float16, .bfloat16, .float32] {
                let q = MLXRandom.normal([B, H, L, D]).asType(dtype)
                let k = MLXRandom.normal([B, H, L, D]).asType(dtype)
                let v = MLXRandom.normal([B, H, L, D]).asType(dtype)

                let cache = QuantizedKVCache(groupSize: 64, bits: 8)
                let (qK, qV) = cache.updateQuantized(keys: k, values: v)
                let out = quantizedScaledDotProductAttention(
                    queries: q, quantizedKeys: qK, quantizedValues: qV,
                    scale: scale, mask: .causal,
                    groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

                #expect(out.dtype == dtype, "output promoted to \(out.dtype) for input \(dtype)")
                #expect(out.asType(.float32).sum().item(Float.self).isFinite)  // no -inf → NaN
            }
        }
    }

    // MARK: - Quantized attention head dims
    // Regression for #652: MLA caches store [kvLatent, kPe] as keys and only kvLatent as
    // values, so the GQA output reshape must follow the value head dim, not the query one.

    @Test("quantizedScaledDotProductAttention supports values narrower than keys")
    func quantizedAttentionSupportsMLAStyleHeadDims() throws {
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let (B, nKVHeads, nRepeats, L) = (1, 1, 4, 3)
            let (kHeadDim, vHeadDim) = (192, 128)
            let scale = 1.0 / Float(kHeadDim).squareRoot()

            let q = MLXRandom.normal([B, nKVHeads * nRepeats, L, kHeadDim])
            let k = MLXRandom.normal([B, nKVHeads, L, kHeadDim])
            let v = MLXRandom.normal([B, nKVHeads, L, vHeadDim])

            let reference = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal)

            let cache = QuantizedKVCache(groupSize: 64, bits: 8)
            let (qK, qV) = cache.updateQuantized(keys: k, values: v)
            let out = quantizedScaledDotProductAttention(
                queries: q, quantizedKeys: qK, quantizedValues: qV,
                scale: scale, mask: .causal,
                groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

            #expect(out.shape == reference.shape)
            #expect(
                allClose(out, reference, rtol: 0.05, atol: 0.1).item(Bool.self),
                "quantized MLA attention diverges from full precision")
        }
    }

    // MARK: - Variance-normalized KV cache

    @Test func testVarianceNormalizedKVCacheStoresCompletedTilesAndTail() throws {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 40, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 40, 32]).asType(.float16)

        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        eval(cachedKeys, cachedValues)

        #expect(cache.offset == 40)
        #expect(cachedKeys.shape == keys.shape)
        #expect(cachedValues.shape == values.shape)
        #expect(
            cache.metaState == ["32", "40", "4", "4", "2", "1", "8", "1", "float16", "float16"])
        #expect(cache.state.count == 10)
        #expect(relativeRMSError(keys, cachedKeys) < 0.5)
        #expect(relativeRMSError(values, cachedValues) < 0.5)
    }

    @Test func testVarianceNormalizedKVCacheInnerStateHoldsTheStoredSlabs() throws {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 8 * 32 + 5, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 8 * 32 + 5, 32]).asType(.float16)
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        eval(cachedKeys, cachedValues)

        let stateBytes = cache.state.reduce(0) { $0 + $1.nbytes }
        #expect(KVCacheStatus(cache: [cache]).memoryBytes == stateBytes)
        #expect(cache.innerState().count < cache.state.count)
    }

    @Test func testVarianceNormalizedKVCacheSerializationRoundTrip() throws {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 68, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 68, 32]).asType(.float16)
        let (originalKeys, originalValues) = cache.update(keys: keys, values: values)
        eval(originalKeys, originalValues)

        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache], metadata: ["kind": "variance-normalized"])
        let (loaded, metadata) = try loadPromptCache(url: url)

        #expect(metadata["kind"] == "variance-normalized")
        let restored = try #require(loaded[0] as? VarianceNormalizedKVCache)
        #expect(restored.metaState == cache.metaState)

        let moreKeys = MLXRandom.normal([1, 1, 1, 32]).asType(.float16)
        let moreValues = MLXRandom.normal([1, 1, 1, 32]).asType(.float16)
        let (restoredKeys, restoredValues) = restored.update(keys: moreKeys, values: moreValues)
        eval(restoredKeys, restoredValues)

        #expect(restored.offset == 69)
        #expect(restoredKeys.shape == [1, 1, 69, 32])
        #expect(restoredValues.shape == [1, 1, 69, 32])
    }

    @Test func testVarianceNormalizedKVCachePreservesDTypeAtExactTileBoundary() throws {
        for dtype in [DType.float32, .bfloat16] {
            let cache = VarianceNormalizedKVCache(
                tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
            let keys = deterministicTensor(shape: [1, 1, 64, 32], salt: 21).asType(dtype)
            let values = deterministicTensor(shape: [1, 1, 64, 32], salt: 22).asType(dtype)
            _ = cache.update(keys: keys, values: values)
            eval(cache.state)

            let url = TemporaryPromptCacheFile.makeURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: [cache])
            let (loaded, _) = try loadPromptCache(url: url)
            let restored = try #require(loaded[0] as? VarianceNormalizedKVCache)
            let copied = try #require(cache.copy() as? VarianceNormalizedKVCache)
            let emptyKeys = MLXArray.zeros([1, 1, 0, 32], dtype: dtype)
            let emptyValues = MLXArray.zeros([1, 1, 0, 32], dtype: dtype)

            for candidate in [restored, copied] {
                let (materializedKeys, materializedValues) = candidate.update(
                    keys: emptyKeys, values: emptyValues)
                eval(materializedKeys, materializedValues)
                #expect(materializedKeys.dtype == dtype)
                #expect(materializedValues.dtype == dtype)
            }
        }
    }

    @Test func testVarianceNormalizedKVCacheTrimOnlyReconstructsTheAffectedTile() throws {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 70, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 70, 32]).asType(.float16)
        let (beforeKeys, beforeValues) = cache.update(keys: keys, values: values)
        eval(beforeKeys, beforeValues)

        #expect(cache.trim(10) == 10)
        let emptyKeys = MLXArray.zeros([1, 1, 0, 32], dtype: .float16)
        let emptyValues = MLXArray.zeros([1, 1, 0, 32], dtype: .float16)
        let (afterKeys, afterValues) = cache.update(keys: emptyKeys, values: emptyValues)
        eval(afterKeys, afterValues)

        #expect(cache.offset == 60)
        #expect(
            cache.metaState == ["32", "60", "4", "4", "2", "1", "28", "1", "float16", "float16"])
        #expect(cache.state.count == 10)
        #expect(relativeRMSError(beforeKeys[.ellipsis, ..<60, 0...], afterKeys) < 1e-3)
        #expect(relativeRMSError(beforeValues[.ellipsis, ..<60, 0...], afterValues) < 1e-3)
    }

    @Test func testVarianceNormalizedKVCacheSupportsAsymmetricKeyValueBits() {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 2, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)

        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        eval(cachedKeys, cachedValues)

        #expect(cache.offset == 32)
        #expect(
            cache.metaState == ["32", "32", "4", "2", "2", "1", "0", "1", "float16", "float16"])
        #expect(cachedKeys.shape == keys.shape)
        #expect(cachedValues.shape == values.shape)
    }

    @Test func testVarianceNormalizedKVCacheSupportsPaperTargetTwoBitKV() {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 2, valueBits: 2, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)

        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        eval(cachedKeys, cachedValues)

        #expect(cache.offset == 32)
        #expect(
            cache.metaState == ["32", "32", "2", "2", "2", "1", "0", "1", "float16", "float16"])
        #expect(cache.state.count == 8)
        #expect(cachedKeys.shape == keys.shape)
        #expect(cachedValues.shape == values.shape)
    }

    @Test func testVarianceNormalizedKVCacheMaterializationPreservesWideInputDTypes() {
        for dtype in [DType.float32, .bfloat16] {
            let cache = VarianceNormalizedKVCache(
                tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
            let magnitude = MLXArray(Float(100_000), dtype: dtype)
            let keys =
                deterministicTensor(shape: [1, 1, 32, 32], salt: 11).asType(dtype) * magnitude
            let values =
                deterministicTensor(shape: [1, 1, 32, 32], salt: 12).asType(dtype) * magnitude

            let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
            eval(cachedKeys, cachedValues)

            #expect(cachedKeys.dtype == dtype)
            #expect(cachedValues.dtype == dtype)
            #expect(isFinite(cachedKeys).all().item(Bool.self))
            #expect(isFinite(cachedValues).all().item(Bool.self))
        }
    }

    @Test func testVarianceNormalizedKVCacheQuantizedTileAttentionMatchesMaterializedAttention() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let queries = deterministicTensor(shape: [1, 1, 4, 32], salt: 31)
        let keys = deterministicTensor(shape: [1, 1, 36, 32], salt: 32)
        let values = deterministicTensor(shape: [1, 1, 36, 32], salt: 33)
        let scale = 1 / sqrt(Float(32))

        let native = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nativeCache,
            scale: scale)
        let (cachedKeys, cachedValues) = materializedCache.update(keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            cache: nil,
            scale: scale)
        eval(native, materialized)

        #expect(relativeRMSError(materialized, native) < 1.5e-3)
        #expect(nativeCache.offset == 36)
        #expect(
            nativeCache.metaState == [
                "32", "36", "4", "4", "2", "1", "4", "1", "float16", "float16",
            ])
    }

    @Test func testVarianceNormalizedKVCacheQuantizedTileAttentionSupportsGQA() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let queries = deterministicTensor(shape: [1, 4, 3, 32], salt: 34)
        let keys = deterministicTensor(shape: [1, 2, 33, 32], salt: 35)
        let values = deterministicTensor(shape: [1, 2, 33, 32], salt: 36)
        let scale = 1 / sqrt(Float(32))

        let native = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nativeCache,
            scale: scale)
        let (cachedKeys, cachedValues) = materializedCache.update(keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            cache: nil,
            scale: scale)
        eval(native, materialized)

        #expect(relativeRMSError(materialized, native) < 1.5e-3)
        #expect(native.shape == [1, 4, 3, 32])
        #expect(
            nativeCache.metaState == [
                "32", "33", "4", "4", "2", "1", "1", "1", "float16", "float16",
            ])
    }

    @Test func testVarianceNormalizedKVCacheStackedTileAttentionMatchesMaterializedAttention() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let queries = deterministicTensor(shape: [1, 1, 2, 32], salt: 1)
        // Thirty-three tiles exercises one immutable slab plus one pending tile.
        let keys = deterministicTensor(shape: [1, 1, 1_056, 32], salt: 2)
        let values = deterministicTensor(shape: [1, 1, 1_056, 32], salt: 3)
        let scale = 1 / sqrt(Float(32))

        let native = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nativeCache,
            scale: scale)
        let (cachedKeys, cachedValues) = materializedCache.update(keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            cache: nil,
            scale: scale)
        eval(native, materialized)

        #expect(relativeRMSError(materialized, native) < 1e-3)
        #expect(
            nativeCache.metaState == [
                "32", "1056", "4", "4", "2", "33", "0", "1", "float16", "float16",
            ])
    }

    @Test func testVarianceNormalizedKVCacheStackedTileAttentionSupportsGQA() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let queries = deterministicTensor(shape: [1, 4, 2, 32], salt: 4)
        let keys = deterministicTensor(shape: [1, 2, 1_056, 32], salt: 5)
        let values = deterministicTensor(shape: [1, 2, 1_056, 32], salt: 6)
        let scale = 1 / sqrt(Float(32))

        let native = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nativeCache,
            scale: scale)
        let (cachedKeys, cachedValues) = materializedCache.update(keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            cache: nil,
            scale: scale)
        eval(native, materialized)

        #expect(relativeRMSError(materialized, native) < 1e-3)
        #expect(native.shape == [1, 4, 2, 32])
        #expect(
            nativeCache.metaState == [
                "32", "1056", "4", "4", "2", "33", "0", "1", "float16", "float16",
            ])
    }

    @Test func testApplyAttentionMaskSuppressesBoolMaskedLogits() {
        let scores = MLXArray([Float(-10), Float(-20), Float(-30)]).reshaped(1, 1, 1, 3)
        let mask = MLXArray([true, false, false]).reshaped(1, 1, 1, 3)

        let weights = softmax(applyAttentionMask(scores: scores, mask: .array(mask)), axis: -1)
        eval(weights)

        let values = weights.asArray(Float.self)
        #expect(values[0] > 0.999)
        #expect(values[1] < 1e-6)
        #expect(values[2] < 1e-6)
    }

    @Test func testVarianceNormalizedKVCacheMaskedAttentionMatchesMaterializedAttention() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let queries = MLXArray.ones([1, 1, 4, 32], dtype: .float16) * -1
        let keys = MLXArray.ones([1, 1, 36, 32], dtype: .float16)
        let values = concatenated(
            [
                MLXArray.zeros([1, 1, 32, 32], dtype: .float16),
                MLXArray.ones([1, 1, 4, 32], dtype: .float16) * 10,
            ], axis: 2)
        let scale = 1 / sqrt(Float(32))
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        let native = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nativeCache,
            scale: scale,
            mask: mask)
        let (cachedKeys, cachedValues) = materializedCache.update(keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            cache: nil,
            scale: scale,
            mask: mask)
        eval(native, materialized)

        #expect(relativeRMSError(materialized, native) < 1e-3)
    }

    @Test func testVarianceNormalizedKVCacheAttentionTracksFP16OverLongDecode() {
        let fp16Cache = KVCacheSimple()
        let varianceNormalizedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let randomState = MLXRandom.RandomState(seed: 7)
        let scale = 1 / sqrt(Float(32))
        var totalError: Float = 0
        var maxError: Float = 0
        let tokenCount = 96

        for _ in 0 ..< tokenCount {
            let queries = MLXRandom.normal([1, 1, 1, 32], key: randomState).asType(.float16)
            let keys = MLXRandom.normal([1, 1, 1, 32], key: randomState).asType(.float16)
            let values = MLXRandom.normal([1, 1, 1, 32], key: randomState).asType(.float16)

            let fp16 = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: fp16Cache,
                scale: scale)
            let compressed = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: varianceNormalizedCache,
                scale: scale)
            eval(fp16, compressed)

            let error = relativeRMSError(fp16, compressed)
            totalError += error
            maxError = max(maxError, error)
        }

        #expect(
            varianceNormalizedCache.metaState == [
                "32", "96", "4", "4", "2", "3", "0", "1", "float16", "float16",
            ])
        #expect(totalError / Float(tokenCount) < 0.15)
        #expect(maxError < 0.45)
    }

    @Test func testVarianceNormalizedKVCacheMemoryAccountingIncludesScaleOverhead() {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 64, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 64, 32]).asType(.float16)

        _ = cache.update(keys: keys, values: values)
        eval(cache.state)

        let state = cache.state
        let compressedBytes = state.reduce(0) { $0 + $1.nbytes }
        let quantizedPayloadBytes = stride(from: 0, to: state.count, by: 8).reduce(0) {
            $0 + state[$1].nbytes + state[$1 + 4].nbytes
        }
        let fp16Bytes = keys.nbytes + values.nbytes

        #expect(
            cache.metaState == ["32", "64", "4", "4", "2", "2", "0", "1", "float16", "float16"])
        #expect(cache.compactStorageByteCount == compressedBytes)
        #expect(compressedBytes > quantizedPayloadBytes)
        #expect(compressedBytes < fp16Bytes)
    }

    @Test func testVarianceNormalizedKVCacheCoalescesTieredSlabs() {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 4, sinkhornIterations: 1)
        let tokenCount = 32 * 32 * 8
        let queries = deterministicTensor(shape: [1, 1, 1, 32], salt: 41)
        let keys = deterministicTensor(shape: [1, 1, tokenCount, 32], salt: 42)
        let values = deterministicTensor(shape: [1, 1, tokenCount, 32], salt: 43)

        let output = cache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(32)))
        eval(output)

        #expect(cache.offset == tokenCount)
        #expect(cache.attentionPartitionCount == 1)
        #expect(cache.compactStorageByteCount == cache.state.reduce(0) { $0 + $1.nbytes })
    }

    @Test func testVarianceNormalizedKVCacheBoundsPartitionsBeforeLargeSlabBoundary() {
        let nativeCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 2, sinkhornIterations: 2)
        let materializedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 2, sinkhornIterations: 2)
        let tokenCount = 31 * 32
        let queries = deterministicTensor(shape: [1, 4, 1, 32], salt: 51)
        let keys = deterministicTensor(shape: [1, 2, tokenCount, 32], salt: 52)
        let values = deterministicTensor(shape: [1, 2, tokenCount, 32], salt: 53)
        let scale = 1 / sqrt(Float(32))

        let native = nativeCache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale)
        let (materializedKeys, materializedValues) = materializedCache.update(
            keys: keys, values: values)
        let materialized = attentionWithCacheUpdate(
            queries: queries,
            keys: materializedKeys,
            values: materializedValues,
            cache: nil,
            scale: scale)
        eval(native, materialized)

        // Four-tile base slabs cap the old 31-dispatch cliff at ten partitions while preserving
        // the quantized attention result. The next tile still coalesces to one 32-tile slab.
        #expect(nativeCache.attentionPartitionCount == 10)
        #expect(relativeRMSError(materialized, native) < 1.5e-3)
    }

    @Test func testVarianceNormalizedKVCacheHeadBatchesMatchIndependentHeads() {
        let keys = deterministicTensor(shape: [1, 8, 128, 32], salt: 54)
        let values = deterministicTensor(shape: [1, 8, 128, 32], salt: 55)
        let batchedCache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 4, valueBits: 2, sinkhornIterations: 2)
        let (batchedKeys, batchedValues) = batchedCache.update(keys: keys, values: values)

        var independentKeys: [MLXArray] = []
        var independentValues: [MLXArray] = []
        for head in 0 ..< 8 {
            let cache = VarianceNormalizedKVCache(
                tileSize: 32, keyBits: 4, valueBits: 2, sinkhornIterations: 2)
            let (headKeys, headValues) = cache.update(
                keys: keys[0..., head ..< head + 1, 0..., 0...],
                values: values[0..., head ..< head + 1, 0..., 0...])
            independentKeys.append(headKeys)
            independentValues.append(headValues)
        }

        let referenceKeys = concatenated(independentKeys, axis: 1)
        let referenceValues = concatenated(independentValues, axis: 1)
        eval(batchedKeys, batchedValues, referenceKeys, referenceValues)

        #expect(relativeRMSError(referenceKeys, batchedKeys) < 1e-6)
        #expect(relativeRMSError(referenceValues, batchedValues) < 1e-6)
    }

    @Test func testVarianceNormalizedKVCacheTwoBitMemoryAccountingIsPaperRelevant() {
        let cache = VarianceNormalizedKVCache(
            tileSize: 32, keyBits: 2, valueBits: 2, sinkhornIterations: 2)
        let keys = MLXRandom.normal([1, 1, 64, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 64, 32]).asType(.float16)

        _ = cache.update(keys: keys, values: values)
        eval(cache.state)

        let compressedBytes = cache.state.reduce(0) { $0 + $1.nbytes }
        let fp16Bytes = keys.nbytes + values.nbytes

        #expect(
            cache.metaState == ["32", "64", "2", "2", "2", "2", "0", "1", "float16", "float16"])
        #expect(compressedBytes < fp16Bytes)
    }

    @Test func testMaybeQuantizeKVCacheCanUseVarianceNormalizedStrategy() throws {
        let simple = KVCacheSimple()
        let keys = MLXRandom.normal([1, 1, 33, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 33, 32]).asType(.float16)
        _ = simple.update(keys: keys, values: values)

        var cache: [KVCache] = [simple]
        let configuration = try KVCacheConfiguration(
            strategy: .varianceNormalized(
                .init(keyBits: 4, valueBits: 4, tileSize: 32, sinkhornIterations: 4)),
            compatibility: .allowPartial)
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        let converted = cache[0] as? VarianceNormalizedKVCache
        #expect(converted != nil)
        #expect(converted?.offset == 33)
        #expect(
            converted?.metaState == [
                "32", "33", "4", "4", "4", "1", "1", "1", "float16", "float16",
            ])
    }

    @Test func testMaybeQuantizeKVCacheDefersVarianceNormalizedStrategyUntilThreshold() throws {
        let simple = KVCacheSimple()
        let keys = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 32, 32]).asType(.float16)
        _ = simple.update(keys: keys, values: values)

        var cache: [KVCache] = [simple]
        let configuration = try KVCacheConfiguration(
            strategy: .varianceNormalized(
                .init(
                    keyBits: 4, valueBits: 4, tileSize: 32, sinkhornIterations: 4,
                    compressionStart: 64)),
            compatibility: .allowPartial)
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        #expect(cache[0] is KVCacheSimple)

        let moreKeys = MLXRandom.normal([1, 1, 33, 32]).asType(.float16)
        let moreValues = MLXRandom.normal([1, 1, 33, 32]).asType(.float16)
        _ = cache[0].update(keys: moreKeys, values: moreValues)
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        #expect(cache[0] is VarianceNormalizedKVCache)
        #expect(cache[0].offset == 65)
    }

    @Test func testMaybeQuantizeKVCacheSkipsUnsupportedVarianceNormalizedDimensions() throws {
        let simple = KVCacheSimple()
        let keys = MLXRandom.normal([1, 1, 33, 24]).asType(.float16)
        let values = MLXRandom.normal([1, 1, 33, 24]).asType(.float16)
        _ = simple.update(keys: keys, values: values)

        var cache: [KVCache] = [simple]
        let configuration = try KVCacheConfiguration(
            strategy: .varianceNormalized(
                .init(keyBits: 4, valueBits: 4, tileSize: 32, sinkhornIterations: 4)),
            compatibility: .allowPartial)
        _ = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)

        #expect(cache[0] is KVCacheSimple)
        #expect(cache[0].offset == 33)
        #expect(
            !supportsVarianceNormalizedKVCache(
                keyHeadDim: 96, valueHeadDim: 96, tileSize: 32))
    }

    @Test func testLegacyVarianceNormalizedSchemeResolvesToTypedConfiguration() throws {
        let parameters = GenerateParameters(quantizedKVStart: 8, kvScheme: "varn4v2t32")
        let resolved = try #require(try parameters.resolvedKVCacheConfiguration())
        #expect(resolved.strategy.identifier == KVCacheStrategyIdentifier.varianceNormalized)
        guard case .varianceNormalized(let varn) = resolved.strategy.storage else {
            Issue.record("Expected a variance-normalized strategy")
            return
        }
        #expect(varn.keyBits == 4)
        #expect(varn.valueBits == 2)
        #expect(varn.tileSize == 32)
        #expect(varn.sinkhornIterations == 8)
        #expect(varn.compressionStart == 8)
    }
}
