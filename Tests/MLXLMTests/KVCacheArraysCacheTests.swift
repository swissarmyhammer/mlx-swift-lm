import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

extension KVCacheTests {
    // MARK: - ArraysCache sparse slot round-trip

    @Test func testArraysCacheSparseSlots() throws {
        let cache = ArraysCache(size: 3)
        let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
        let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
        cache[0] = a
        // slot 1 stays nil
        cache[2] = b

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.slotCount == 3)
        #expect(restored[0] != nil)
        #expect(restored[1] == nil)
        #expect(restored[2] != nil)
        #expect(allClose(restored[0]!, a).item(Bool.self))
        #expect(allClose(restored[2]!, b).item(Bool.self))
    }

    // MARK: - ArraysCache leftPadding round-trip

    @Test func testArraysCacheLeftPadding() throws {
        let cache = ArraysCache(size: 2, leftPadding: [0, 5])
        let a = MLXArray.ones([2, 4], dtype: .float32)
        let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
        cache[0] = a
        cache[1] = b

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.leftPaddingValues == [0, 5])
        assertArraysClose(restored.state, cache.state)
    }

    @Test func testArraysCacheMaskUsesLeftPaddingAfterStateUpdate() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        let mask = try #require(cache.makeMask(N: 4))
        #expect(
            mask.asArray(Bool.self) == [
                false, true, true, true,
                false, false, false, true,
            ])
    }

    @Test func testArraysCacheMaskUsesLengthsWhenLeftPaddingIsAbsent() throws {
        let cache = ArraysCache(size: 2)
        cache.prepare(lengths: [1, 3])

        let mask = try #require(cache.makeMask(N: 4))
        #expect(
            mask.asArray(Bool.self) == [
                true, false, false, false,
                true, true, true, false,
            ])
    }

    @Test func testTextSequenceLengthsComeFromAttentionMask() throws {
        let tokens = MLXArray(0 ..< 8).reshaped(2, 4)
        let mask = MLXArray([1, 1, 0, 0, 1, 1, 1, 0]).reshaped(2, 4)
        let text = LMInput.Text(tokens: tokens, mask: mask)

        #expect(text.sequenceLengths == [2, 3])
    }

    @Test func testTextSequenceLengthsInferUniformBatches() throws {
        let text = LMInput.Text(tokens: MLXArray(0 ..< 8).reshaped(2, 4))

        #expect(text.sequenceLengths == [4, 4])
    }

    @Test func testCacheListForwardsPrepareAndFinalize() throws {
        let arrays = ArraysCache(size: 2)
        let cache = CacheList(arrays, KVCacheSimple())

        cache.prepare(lengths: [2, 4])
        #expect(arrays.lengthsValues == [2, 4])

        cache.finalize()
        #expect(arrays.lengthsValues == nil)
    }

    @Test func testCacheListForwardsLifecycleThroughKVCacheProtocol() throws {
        let lifecycle = LifecycleRecordingCache()
        let cache = CacheList(KVCacheSimple(), lifecycle)

        cache.prepare(lengths: [2, 4])
        #expect(lifecycle.preparedLengths == [2, 4])

        cache.finalize()
        #expect(lifecycle.finalizeCallCount == 1)
    }

    @Test func testWithPreparedCacheScopesSequenceMetadata() throws {
        let cache = ArraysCache(size: 2)

        withPreparedCache([cache], lengths: [2, 4]) {
            #expect(cache.lengthsValues == [2, 4])
        }

        #expect(cache.lengthsValues == nil)
    }

    @Test func testArraysCacheLengthsRoundTrip() throws {
        let cache = ArraysCache(size: 2)
        cache.prepare(lengths: [4, 2])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.currentLengths?.asArray(Int.self) == [4, 2])
        #expect(restored.lengthsValues == [4, 2])
        assertArraysClose(restored.state, cache.state)
    }

    @Test func testArraysCacheAdvanceUpdatesLengthsAndLeftPaddingMasks() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache.prepare(lengths: [4, 2])
        cache.advance(2)

        #expect(cache.leftPaddingValues == [-1, 1])
        #expect(cache.currentLengths?.asArray(Int.self) == [2, 0])

        let mask = try #require(cache.makeMask(N: 3))
        #expect(mask.asArray(Bool.self) == [true, true, true, false, true, true])

        let lengthOnly = ArraysCache(size: 1)
        lengthOnly.prepare(lengths: [2, 0])
        let lengthMask = try #require(lengthOnly.makeMask(N: 3))
        #expect(lengthMask.asArray(Bool.self) == [true, true, false, false, false, false])

        cache.finalize()
        #expect(cache.leftPaddingValues == nil)
        #expect(cache.currentLengths == nil)
    }

    @Test func testArraysCacheFilterAndExtendPreserveBatchMetadata() throws {
        let first = ArraysCache(size: 1, leftPadding: [0, 2])
        first.prepare(lengths: [5, 3])
        first[0] = MLXArray.ones([2, 2], dtype: .float32)

        first.filter(batchIndices: MLXArray([1]))
        #expect(first.leftPaddingValues == [2])
        #expect(first.currentLengths?.asArray(Int.self) == [3])
        #expect(first[0]?.shape == [1, 2])

        let second = ArraysCache(size: 1, leftPadding: [1, 4])
        second.prepare(lengths: [6, 2])
        second[0] = MLXArray.ones([2, 2], dtype: .float32) * 2

        first.extend(other: second)
        #expect(first.leftPaddingValues == [2, 1, 4])
        #expect(first.currentLengths?.asArray(Int.self) == [3, 6, 2])
        #expect(first[0]?.shape == [3, 2])
    }

    @Test func testAttentionMaskUsesSharedCausalCachePath() throws {
        let cache = KVCacheSimple()
        let prefillInput = MLXArray.ones([1, 3, 8], dtype: .float32)

        let prefillMask = createAttentionMask(h: prefillInput, cache: cache)
        if case .causal = prefillMask {
            // Expected for multi-token prefill: Falcon H1 uses the shared symbolic causal mask path.
        } else {
            Issue.record("Expected symbolic causal attention mask for prefill")
        }

        let tokenInput = MLXArray.ones([1, 1, 8], dtype: .float32)
        let tokenMask = createAttentionMask(h: tokenInput, cache: cache)
        if case .none = tokenMask {
            // Expected for one-token decode: no materialized mask is needed.
        } else {
            Issue.record("Expected no attention mask for one-token decode")
        }

        cache.offset = 2
        let forcedMask = createAttentionMask(h: prefillInput, cache: cache, returnArray: true)
        guard case .array(let mask) = forcedMask else {
            Issue.record("Expected forced attention mask array")
            return
        }
        #expect(mask.shape == [3, 5])
        #expect(
            mask.asArray(Bool.self) == [
                true, true, true, false, false,
                true, true, true, true, false,
                true, true, true, true, true,
            ])
    }

    @Test func testSSMMaskUsesSharedMambaMetadataPath() throws {
        let leftPadded = MambaCache(leftPadding: [1, 3])
        let input = MLXArray.ones([2, 4, 8], dtype: .float32)

        let leftPaddingMask = try #require(createSSMMask(h: input, cache: leftPadded))
        #expect(
            leftPaddingMask.asArray(Bool.self) == [
                false, true, true, true,
                false, false, false, true,
            ])

        let lengthMasked = MambaCache()
        lengthMasked.prepare(lengths: [3, 1])
        let lengthsMask = try #require(createSSMMask(h: input, cache: lengthMasked))
        #expect(
            lengthsMask.asArray(Bool.self) == [
                true, true, true, false,
                true, false, false, false,
            ])
    }

    @Test func testCacheListPrepareFinalizePropagatesThroughNestedHybridCaches() throws {
        let mamba = MambaCache(leftPadding: [0, 2])
        let arrays = ArraysCache(size: 1)
        let nested = CacheList(CacheList(mamba), arrays)

        nested.prepare(lengths: [4, 1])

        #expect(mamba.currentLengths?.asArray(Int.self) == [4, 1])
        #expect(arrays.currentLengths?.asArray(Int.self) == [4, 1])

        nested.finalize()

        #expect(mamba.currentLengths == nil)
        #expect(mamba.leftPaddingValues == nil)
        #expect(arrays.currentLengths == nil)
    }

    @Test func testMambaCacheCopyPreservesBatchMaskMetadata() throws {
        let cache = MambaCache(leftPadding: [2, 0])
        cache.prepare(lengths: [5, 3])
        cache[0] = MLXArray.ones([2, 3, 4], dtype: .float32)
        cache[1] = MLXArray.ones([2, 1, 4, 4], dtype: .float32)

        let copied = try #require(cache.copy() as? MambaCache)

        #expect(copied.leftPaddingValues == [2, 0])
        #expect(copied.currentLengths?.asArray(Int.self) == [5, 3])
        #expect(copied[0]?.shape == [2, 3, 4])
        #expect(copied[1]?.shape == [2, 1, 4, 4])
    }

    @Test func testArraysCacheFilterKeepsSequenceMetadata() throws {
        let cache = ArraysCache(size: 2, leftPadding: [1, 3])
        cache.prepare(lengths: [2, 4])
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)

        cache.filter(batchIndices: MLXArray([1]))

        #expect(cache.leftPaddingValues == [3])
        #expect(cache.lengthsValues == [4])
    }

    @Test func testArraysCacheExtendPadsMissingSlotsAndMetadata() throws {
        let first = ArraysCache(size: 2, leftPadding: [1, 3])
        first.prepare(lengths: [2, 4])
        first[0] = MLXArray.ones([2, 4], dtype: .float32)

        let second = ArraysCache(size: 2)
        second[1] = MLXArray.ones([1, 4], dtype: .float32) * 2

        first.extend(other: second)

        #expect(first[0]?.shape == [3, 4])
        #expect(first[1]?.shape == [3, 4])
        #expect(first.leftPaddingValues == [1, 3, 0])
        #expect(first.lengthsValues == [2, 4, 0])
    }

    @Test func testArraysCacheCopyPreservesSparseSlotsAndMetadata() throws {
        let cache = ArraysCache(size: 3, leftPadding: [2])
        cache.prepare(lengths: [5])
        cache[2] = MLXArray.ones([1, 4], dtype: .float32)

        let copied = try #require(cache.copy() as? ArraysCache)

        #expect(copied.slotCount == 3)
        #expect(copied[0] == nil)
        #expect(copied[1] == nil)
        #expect(copied[2] != nil)
        #expect(copied.leftPaddingValues == [2])
        #expect(copied.lengthsValues == [5])
    }

    // MARK: - MambaCache type preservation

    @Test func testMambaCacheRoundTrip() throws {
        let cache = MambaCache()
        let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
        let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
        cache[0] = a
        cache[1] = b

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? MambaCache)
        #expect(restored.slotCount == 2)
        assertArraysClose(restored.state, cache.state)
    }

    // MARK: - CacheList with KV caches

    @Test func testCacheListKVCaches() throws {
        let simple = KVCacheSimple()
        let rotating = RotatingKVCache(maxSize: 32)

        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = simple.update(keys: keys, values: values)
        _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

        let cacheList = CacheList(simple, rotating)

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cacheList], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? CacheList)
        let child0 = try #require(restored[0] as? KVCacheSimple)
        let child1 = try #require(restored[1] as? RotatingKVCache)

        assertArraysClose(child0.state, simple.state, label: "child0")
        assertArraysClose(child1.state, rotating.state, label: "child1")
        #expect(child1.metaState == rotating.metaState)
    }

    // MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

    @Test func testCacheListHybrid() throws {
        let mamba = MambaCache()
        mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
        mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

        let simple = KVCacheSimple()
        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = simple.update(keys: keys, values: values)

        let cacheList = CacheList(mamba, simple)

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cacheList], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? CacheList)
        let restoredMamba = try #require(restored[0] as? MambaCache)
        let restoredSimple = try #require(restored[1] as? KVCacheSimple)

        assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
        assertArraysClose(restoredSimple.state, simple.state, label: "simple")
    }

    // MARK: - Simple cache round-trip with value assertions

    @Test func testSimpleCacheRoundTrip() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
        _ = cache.update(keys: keys, values: values)

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        #expect(loaded.count == 1)
        assertArraysClose(loaded[0].state, cache.state)
    }

    // MARK: - ArraysCache fully populated round-trip

    @Test func testArraysCacheFullyPopulated() throws {
        let cache = ArraysCache(size: 2)
        cache[0] = MLXArray.ones([2, 4], dtype: .float32)
        cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

        let url = TemporaryPromptCacheFile.makeURL()
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? ArraysCache)
        #expect(restored.slotCount == 2)
        assertArraysClose(restored.state, cache.state)
    }
}
