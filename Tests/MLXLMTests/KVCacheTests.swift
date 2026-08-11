import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
]

// MARK: - Helper

private func tempURL() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")
}

/// Assert two arrays of MLXArray are element-wise close
private func assertArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray], label: String = "") {
    #expect(lhs.count == rhs.count, "state count mismatch \(label)")
    for (i, (a, b)) in zip(lhs, rhs).enumerated() {
        #expect(a.shape == b.shape, "shape mismatch at index \(i) \(label)")
        let close = allClose(a, b).item(Bool.self)
        #expect(close, "values not close at index \(i) \(label)")
    }
}

private final class LifecycleRecordingCache: BaseKVCache {
    private(set) var preparedLengths: [Int]?
    private(set) var finalizeCallCount = 0

    override func prepare(lengths: [Int]?) {
        preparedLengths = lengths
    }

    override func finalize() {
        finalizeCallCount += 1
    }

    override func copy() -> any KVCache {
        let new = LifecycleRecordingCache()
        new.preparedLengths = preparedLengths
        new.finalizeCallCount = finalizeCallCount
        return new
    }
}

// MARK: - Original parameterized test (updated with value assertions)

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = tempURL()

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        assertArraysClose(lhs.state, rhs.state)
    }
}

@Test func testQuantizedKVCacheRestoresNonDefaultQuantizationMetadata() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = QuantizedKVCache(groupSize: 64, bits: 4)
    let keys = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
    let values = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
    _ = cache.updateQuantized(keys: keys, values: values)

    #expect(cache.groupSize == 32)
    #expect(cache.bits == 4)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? QuantizedKVCache)
    #expect(restored.groupSize == 32)
    #expect(restored.bits == 4)
    #expect(restored.metaState == cache.metaState)

    let moreKeys = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 1, 1, 32], dtype: .bfloat16)
    _ = restored.updateQuantized(keys: moreKeys, values: moreValues)

    #expect(restored.groupSize == 32)
    #expect(restored.bits == 4)
}

@Test func testQuantizedKVCacheMetaStateRestoresQuantizationMetadataWithoutState() {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = QuantizedKVCache()

    cache.metaState = ["256", "11", "32", "4"]

    #expect(cache.offset == 11)
    #expect(cache.groupSize == 32)
    #expect(cache.bits == 4)
    #expect(cache.metaState == ["256", "11", "32", "4"])
}

@Test func testQuantizedKVCacheCopyPreservesRestoredQuantizationMetadata() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = QuantizedKVCache()
    cache.metaState = ["256", "5", "32", "4"]

    let copied = try #require(cache.copy() as? QuantizedKVCache)

    #expect(copied.offset == 5)
    #expect(copied.groupSize == 32)
    #expect(copied.bits == 4)
    #expect(copied.metaState == cache.metaState)
}

@Test func testEmptyKVCacheSimpleToQuantizedPreservesRequestedQuantizationMetadata() {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = KVCacheSimple()
    cache.offset = 7

    let quantized = cache.toQuantized(groupSize: 128, bits: 4)

    #expect(quantized.offset == 7)
    #expect(quantized.groupSize == 128)
    #expect(quantized.bits == 4)
    #expect(quantized.metaState == ["256", "7", "128", "4"])
}

// MARK: - ArraysCache sparse slot round-trip

@Test func testArraysCacheSparseSlots() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 3)
    let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
    cache[0] = a
    // slot 1 stays nil
    cache[2] = b

    let url = tempURL()
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2, leftPadding: [0, 5])
    let a = MLXArray.ones([2, 4], dtype: .float32)
    let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.leftPaddingValues == [0, 5])
    assertArraysClose(restored.state, cache.state)
}

@Test func testArraysCacheMaskUsesLeftPaddingAfterStateUpdate() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2, leftPadding: [1, 3])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    let mask = try #require(cache.makeMask(N: 4))
    #expect(
        mask.asArray(Bool.self) == [
            false, true, true, true,
            false, false, false, true,
        ])
}

@Test func testArraysCacheAdvanceUpdatesSequenceMetadataOnly() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2, leftPadding: [3, 5])
    cache.offset = 7
    cache.prepare(lengths: [4, 6])

    cache.advance(2)

    #expect(cache.offset == 7)
    #expect(cache.leftPaddingValues == [1, 3])
    #expect(cache.lengthsValues == [2, 4])
}

@Test func testArraysCacheMaskUsesLengthsWhenLeftPaddingIsAbsent() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let tokens = MLXArray(0 ..< 8).reshaped(2, 4)
    let mask = MLXArray([1, 1, 0, 0, 1, 1, 1, 0]).reshaped(2, 4)
    let text = LMInput.Text(tokens: tokens, mask: mask)

    #expect(text.sequenceLengths == [2, 3])
}

@Test func testTextSequenceLengthsInferUniformBatches() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let text = LMInput.Text(tokens: MLXArray(0 ..< 8).reshaped(2, 4))

    #expect(text.sequenceLengths == [4, 4])
}

@Test func testCacheListForwardsPrepareAndFinalize() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let arrays = ArraysCache(size: 2)
    let cache = CacheList(arrays, KVCacheSimple())

    cache.prepare(lengths: [2, 4])
    #expect(arrays.lengthsValues == [2, 4])

    cache.finalize()
    #expect(arrays.lengthsValues == nil)
}

@Test func testCacheListForwardsLifecycleThroughKVCacheProtocol() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let lifecycle = LifecycleRecordingCache()
    let cache = CacheList(KVCacheSimple(), lifecycle)

    cache.prepare(lengths: [2, 4])
    #expect(lifecycle.preparedLengths == [2, 4])

    cache.finalize()
    #expect(lifecycle.finalizeCallCount == 1)
}

@Test func testWithPreparedCacheScopesSequenceMetadata() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2)

    withPreparedCache([cache], lengths: [2, 4]) {
        #expect(cache.lengthsValues == [2, 4])
    }

    #expect(cache.lengthsValues == nil)
}

@Test func testArraysCacheLengthsRoundTrip() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2)
    cache.prepare(lengths: [4, 2])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.currentLengths?.asArray(Int.self) == [4, 2])
    #expect(restored.lengthsValues == [4, 2])
    assertArraysClose(restored.state, cache.state)
}

@Test func testArraysCacheAdvanceUpdatesLengthsAndLeftPaddingMasks() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2, leftPadding: [1, 3])
    cache.prepare(lengths: [2, 4])
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)

    cache.filter(batchIndices: MLXArray([1]))

    #expect(cache.leftPaddingValues == [3])
    #expect(cache.lengthsValues == [4])
}

@Test func testArraysCacheExtendPadsMissingSlotsAndMetadata() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = MambaCache()
    let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? MambaCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

// MARK: - CacheList with KV caches

@Test func testCacheListKVCaches() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let simple = KVCacheSimple()
    let rotating = RotatingKVCache(maxSize: 32)

    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)
    _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

    let cacheList = CacheList(simple, rotating)

    let url = tempURL()
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let mamba = MambaCache()
    mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let simple = KVCacheSimple()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)

    let cacheList = CacheList(mamba, simple)

    let url = tempURL()
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
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = KVCacheSimple()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: values)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)
    #expect(loaded.count == 1)
    assertArraysClose(loaded[0].state, cache.state)
}

// MARK: - ArraysCache fully populated round-trip

@Test func testArraysCacheFullyPopulated() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let cache = ArraysCache(size: 2)
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)
    cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

// MARK: - trim() + continuation semantics
//
// `PromptCache` (MLXFoundationModels) assumes that trimming a KVCacheSimple
// back to an earlier offset and then continuing generation from there
// produces IDENTICAL cache state to having only ever prefilled the
// truncated sequence followed by the same continuation. That assumption is
// what makes `.trimTo` a safe substitute for a full rebuild on a diverged
// prompt. These tests exercise `trim()` itself (not `PromptCache`'s pure
// decision logic, which `PromptCacheTests` already covers) against a real
// `KVCacheSimple`, proving the assumption holds at the tensor level.

/// One single-position (seqLen == 1) key/value pair filled with a distinct
/// scalar so each fed "token" is individually identifiable in the resulting
/// state tensor.
private func singleTokenKV(fill: Float, kvHeads: Int = 2, headDim: Int = 4) -> (
    keys: MLXArray, values: MLXArray
) {
    (
        MLXArray.ones([1, kvHeads, 1, headDim], dtype: .float32) * fill,
        MLXArray.ones([1, kvHeads, 1, headDim], dtype: .float32) * (fill + 1000)
    )
}

@Test func testTrimThenContinueMatchesFreshPrefillOfTruncatedSequence() throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    // Prefill 7 tokens (fills 0...6) into cacheA.
    let cacheA = KVCacheSimple()
    for i in 0 ..< 7 {
        let (k, v) = singleTokenKV(fill: Float(i))
        _ = cacheA.update(keys: k, values: v)
    }
    #expect(cacheA.offset == 7)

    // Trim back by 3 (offset 7 -> 4) -- KVCacheSimple.trim is unbounded, so
    // the full request is honored.
    let trimmed = cacheA.trim(3)
    #expect(trimmed == 3)
    #expect(cacheA.offset == 4)

    // Continue with 2 NEW tokens (fills 100, 101) from the trimmed offset.
    for fill: Float in [100, 101] {
        let (k, v) = singleTokenKV(fill: fill)
        _ = cacheA.update(keys: k, values: v)
    }
    #expect(cacheA.offset == 6)

    // Independent cache, fed ONLY the truncated prefix (fills 0...3) plus
    // the identical continuation (fills 100, 101) -- what `PromptCache`
    // assumes trim+continue on cacheA is equivalent to.
    let cacheB = KVCacheSimple()
    for i in 0 ..< 4 {
        let (k, v) = singleTokenKV(fill: Float(i))
        _ = cacheB.update(keys: k, values: v)
    }
    for fill: Float in [100, 101] {
        let (k, v) = singleTokenKV(fill: fill)
        _ = cacheB.update(keys: k, values: v)
    }
    #expect(cacheB.offset == 6)

    #expect(cacheA.offset == cacheB.offset)
    eval(cacheA.state)
    eval(cacheB.state)
    assertArraysClose(cacheA.state, cacheB.state)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

// MARK: - Attention sinks on the quantized path

/// The number of query heads the attention-sink tests read.
private let sinkQueryHeads = 2

/// The number of key/value heads the attention-sink tests read.
private let sinkKeyValueHeads = 1

/// The number of tokens the attention-sink tests read.
private let sinkTokenCount = 3

/// The head width the attention-sink tests read. A quantized cache needs a
/// head width divisible by one of the supported group sizes.
private let sinkHeadDim = 64

/// The bit width the attention-sink tests quantize with.
private let sinkBits = 8

/// The group size the attention-sink tests quantize with.
private let sinkGroupSize = 64

/// The learned sink logit of each query head. The first head holds most of
/// its weight back; the second head gives some of it away.
private let sinkLogits: [Float] = [2, -1]

/// The largest gap allowed between the quantized path and mlx's own
/// attention. Eight-bit affine quantization of values inside -1 through 1
/// carries about 0.004 of error, thus this limit sits well above it and well
/// below the effect the sink itself has.
private let sinkQuantizationTolerance: Float = 0.02

/// `quantizedScaledDotProductAttention` must give the same answer as mlx's
/// own attention when both read the same sink logits, and a materially
/// different one when the sink is left out.
@Test func quantizedAttentionReadsTheLearnedSink() {
    MLXRandom.seed(0)
    let queryShape = [1, sinkQueryHeads, sinkTokenCount, sinkHeadDim]
    let keyShape = [1, sinkKeyValueHeads, sinkTokenCount, sinkHeadDim]
    let queries = MLXRandom.normal(queryShape)
    let keys = MLXRandom.normal(keyShape)
    let values = MLXRandom.normal(keyShape)
    let scale = 1 / sqrt(Float(sinkHeadDim))
    let sinks = MLXArray(sinkLogits)

    let quantizedKeys = quantized(keys, groupSize: sinkGroupSize, bits: sinkBits)
    let quantizedValues = quantized(values, groupSize: sinkGroupSize, bits: sinkBits)
    let roundTripKeys = dequantized(
        quantizedKeys.wq, scales: quantizedKeys.scales, biases: quantizedKeys.biases,
        groupSize: sinkGroupSize, bits: sinkBits)
    let roundTripValues = dequantized(
        quantizedValues.wq, scales: quantizedValues.scales, biases: quantizedValues.biases,
        groupSize: sinkGroupSize, bits: sinkBits)

    func quantizedAttention(sinks: MLXArray?) -> [Float] {
        quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: .none,
            groupSize: sinkGroupSize,
            bits: sinkBits,
            sinks: sinks
        ).asType(.float32).asArray(Float.self)
    }

    let expected = MLXFast.scaledDotProductAttention(
        queries: queries, keys: roundTripKeys, values: roundTripValues,
        scale: scale, mask: .none, sinks: sinks
    ).asType(.float32).asArray(Float.self)
    let withSink = quantizedAttention(sinks: sinks)
    let withoutSink = quantizedAttention(sinks: nil)

    #expect(withSink.count == expected.count)
    for (index, pair) in zip(withSink, expected).enumerated() {
        #expect(
            abs(pair.0 - pair.1) <= sinkQuantizationTolerance,
            "sink weight [\(index)]: got \(pair.0), expected \(pair.1)")
    }
    let sinkEffect = zip(withSink, withoutSink).map { abs($0 - $1) }.max() ?? 0
    #expect(sinkEffect > sinkQuantizationTolerance, "the sink must change the answer")
}

// MARK: - Masked positions on the quantized path

/// The number of tokens the masked-fill tests read. Two tokens let a causal
/// mask hold the second key back from the first query.
private let maskedFillTokenCount = 2

/// The head width the masked-fill tests read. A quantized cache needs a head
/// width divisible by one of the supported group sizes.
private let maskedFillHeadDim = 64

/// The bit width the masked-fill tests quantize with.
private let maskedFillBits = 8

/// The group size the masked-fill tests quantize with.
private let maskedFillGroupSize = 64

/// The larger element of the key at the open position. It gives that position
/// a small score, thus a fill near zero would win a large part of the weight.
private let maskedFillOpenKeyPeak: Float = 0.5

/// The larger element of the key at the masked position. It gives that
/// position a score far above the open one, thus only the mask holds it back.
private let maskedFillMaskedKeyPeak: Float = 10

/// The two elements of the value at the open position. The first element is
/// zero, thus the first element of the answer carries no weight from here.
private let maskedFillOpenValue: (low: Float, high: Float) = (0, 1)

/// The two elements of the value at the masked position. The first element is
/// one, thus the first element of the answer is the weight this position kept.
private let maskedFillMaskedValue: (low: Float, high: Float) = (1, 2)

/// The largest weight a masked position may keep after the softmax.
private let maskedFillWeightLimit: Float = 1e-6

/// The smallest weight the large key must win when no mask holds it back.
private let maskedFillOpenWeightFloor: Float = 0.99

/// The mask forms that `quantizedScaledDotProductAttention` fills by itself.
enum MaskedFillCase: String, CaseIterable, Sendable {
    case causal
    case boolArray
}

/// Build one key or value row, the first half `low` and the second half `high`.
///
/// Two distinct elements keep the affine quantization exact: `low` and `high`
/// sit on the ends of the eight-bit grid, thus the round trip loses nothing.
private func maskedFillRow(low: Float, high: Float) -> [Float] {
    let half = maskedFillHeadDim / 2
    return [Float](repeating: low, count: half)
        + [Float](repeating: high, count: maskedFillHeadDim - half)
}

/// The queries, keys and values the masked-fill tests read.
///
/// Every query element is one, thus the score of a key is the scaled sum of
/// its elements. The key at the last position is far larger than the key at
/// the first, thus it wins nearly all of the weight unless a mask stops it.
private struct MaskedFillFixture {
    let queries: MLXArray
    let keys: MLXArray
    let values: MLXArray
    let scale: Float

    init() {
        let shape = [1, 1, maskedFillTokenCount, maskedFillHeadDim]
        queries = MLXArray(
            [Float](repeating: 1, count: maskedFillTokenCount * maskedFillHeadDim)
        ).reshaped(shape)
        keys = MLXArray(
            maskedFillRow(low: 0, high: maskedFillOpenKeyPeak)
                + maskedFillRow(low: 0, high: maskedFillMaskedKeyPeak)
        ).reshaped(shape)
        values = MLXArray(
            maskedFillRow(low: maskedFillOpenValue.low, high: maskedFillOpenValue.high)
                + maskedFillRow(low: maskedFillMaskedValue.low, high: maskedFillMaskedValue.high)
        ).reshaped(shape)
        scale = 1 / sqrt(Float(maskedFillHeadDim))
    }

    /// The answer of the quantized path for the given mask.
    func attend(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantized(keys, groupSize: maskedFillGroupSize, bits: maskedFillBits),
            quantizedValues: quantized(
                values, groupSize: maskedFillGroupSize, bits: maskedFillBits),
            scale: scale,
            mask: mask,
            groupSize: maskedFillGroupSize,
            bits: maskedFillBits
        )
    }

    /// The weight the first query gave to the last key.
    ///
    /// The first element of the value at the open position is zero and the
    /// first element of the value at the last position is one, thus the first
    /// element of the answer is that weight and nothing else.
    func weightOfLastKey(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Float {
        attend(mask: mask)[0, 0, 0, 0].item(Float.self)
    }
}

/// The mask mode of the given case. Every case holds the last key back from
/// the first query, thus the three cases must give the same answer.
private func maskedFillMode(for maskedFillCase: MaskedFillCase)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let causalMask = createCausalMask(n: maskedFillTokenCount, offset: 0)
    switch maskedFillCase {
    case .causal: return .causal
    case .boolArray: return .array(causalMask)
    }
}

/// A masked position must lose all of its weight, even when its key is large
/// enough to win every open position.
@Test(arguments: MaskedFillCase.allCases)
func quantizedAttentionGivesAMaskedPositionNoWeight(maskedFillCase: MaskedFillCase) {
    let weight = MaskedFillFixture().weightOfLastKey(mask: maskedFillMode(for: maskedFillCase))
    #expect(
        weight < maskedFillWeightLimit,
        "masked weight for \(maskedFillCase.rawValue): got \(weight)")
}

/// The deprecated list form of the mask must hold a masked position back the
/// same way. This test carries the same deprecation as the mask mode it reads,
/// thus it builds without a warning.
@available(*, deprecated, message: "reads the deprecated .arrays mask mode")
@Test func quantizedAttentionGivesAMaskedPositionNoWeightWithAListMask() {
    let causalMask = createCausalMask(n: maskedFillTokenCount, offset: 0)
    let weight = MaskedFillFixture().weightOfLastKey(mask: .arrays([causalMask]))
    #expect(weight < maskedFillWeightLimit, "masked weight for a list mask: got \(weight)")
}

/// The large key wins nearly all of the weight when no mask holds it back,
/// thus the test above measures the mask and not a small key.
@Test func quantizedAttentionWithoutAMaskFollowsTheLargeKey() {
    let weight = MaskedFillFixture().weightOfLastKey(mask: .none)
    #expect(
        weight > maskedFillOpenWeightFloor,
        "unmasked weight of the large key: got \(weight)")
}

/// A query whose positions are all masked must still give a finite answer.
///
/// The fill is the most negative finite number of the score dtype, thus the
/// softmax of a row that holds only fill values gives even weights. A fill of
/// `-infinity` would give NaN here.
@Test func quantizedAttentionKeepsAFullyMaskedRowFinite() {
    let closedMask = (MLXArray(0 ..< Int32(maskedFillTokenCount)) .< Int32(0))
        .reshaped([1, maskedFillTokenCount])
    let answer = MaskedFillFixture().attend(mask: .array(closedMask))
    let elements = answer.asType(.float32).asArray(Float.self)
    #expect(elements.allSatisfy { $0.isFinite }, "a fully masked row must stay finite")
}
