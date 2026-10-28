import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

/// The window of the rotating cache and the tile of the variance-normalized cache that
/// `cacheCreators` makes.
private let creatorWindowSize = 32
/// The key and value bit width of the variance-normalized cache that `cacheCreators` makes.
private let creatorBits = 4
/// The chunk size of the chunked cache that `cacheCreators` makes.
private let creatorChunkSize = 16
/// The slot count of the arrays cache that `cacheCreators` makes.
private let creatorSlotCount = 2

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: creatorWindowSize) },
    { QuantizedKVCache() },
    {
        VarianceNormalizedKVCache(
            tileSize: creatorWindowSize, keyBits: creatorBits, valueBits: creatorBits)
    },
    { ChunkedKVCache(chunkSize: creatorChunkSize) },
    { ArraysCache(size: creatorSlotCount) },
    { MambaCache() },
]

// MARK: - Helper

/// The constants of the pseudo-random pattern that `deterministicTensor` fills a tensor with.
private enum DeterministicPattern {
    /// The multiplier of the element index.
    static let indexStride = 37
    /// The multiplier of the salt.
    static let saltStride = 17
    /// The modulus that folds the pattern into a small range.
    static let modulus = 101
    /// The value that centres the folded range on zero.
    static let center = 50
    /// The divisor that scales the centred range to small floats.
    static let divisor: Float = 23
}

func deterministicTensor(shape: [Int], salt: Int) -> MLXArray {
    let count = shape.reduce(1, *)
    let values = (0 ..< count).map { index in
        let mixed =
            index * DeterministicPattern.indexStride + salt * DeterministicPattern.saltStride
        return Float(mixed % DeterministicPattern.modulus - DeterministicPattern.center)
            / DeterministicPattern.divisor
    }
    return MLXArray(values).reshaped(shape).asType(.float16)
}

private struct SerializedCacheFixture {
    let className: String
    let arrayCount: Int
    let metadata: [String]
    var arrayShape = [1, 1, 1, 1]
}

private func writePromptCacheFixture(_ fixture: SerializedCacheFixture) throws -> URL {
    let url = TemporaryPromptCacheFile.makeURL()
    let arrays: [String: MLXArray] = Dictionary(
        uniqueKeysWithValues: (0 ..< fixture.arrayCount).map { index in
            (
                "0.\(index)",
                MLXArray.zeros(fixture.arrayShape, dtype: .float32, stream: .cpu)
            )
        })
    var metadata: [String: String] = Dictionary(
        uniqueKeysWithValues: fixture.metadata.enumerated().map {
            ("0.0.\($0.offset)", $0.element)
        })
    metadata["2.0"] = fixture.className
    try save(arrays: arrays, metadata: metadata, url: url, stream: .cpu)
    return url
}

private func expectPromptCacheLoadToFail(_ fixtures: [SerializedCacheFixture]) throws {
    for fixture in fixtures {
        let url = try writePromptCacheFixture(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            _ = try loadPromptCache(url: url)
        }
    }
}

/// Assert two arrays of MLXArray are element-wise close
func assertArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray], label: String = "") {
    #expect(lhs.count == rhs.count, "state count mismatch \(label)")
    for (i, (a, b)) in zip(lhs, rhs).enumerated() {
        #expect(a.shape == b.shape, "shape mismatch at index \(i) \(label)")
        let close = allClose(a, b).item(Bool.self)
        #expect(close, "values not close at index \(i) \(label)")
    }
}

final class LifecycleRecordingCache: BaseKVCache {
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

/// A direct protocol conformer that deliberately relies on the
/// `KVCache.isTrimmable(after:)` extension default.
private final class ProtocolDefaultTrimmabilityCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let copy = ProtocolDefaultTrimmabilityCache()
        copy.offset = offset
        return copy
    }

    func innerState() -> [MLXArray] { [] }
}

@Suite(.serialized) struct KVCacheTests {
    // MARK: - Predictive trimmability

    @Test func testRotatingKVCachePredictsStrictTrimBoundary() {
        let cache = RotatingKVCache(maxSize: 8)
        cache.offset = 3

        #expect(cache.isTrimmable(after: 4))
        #expect(!cache.isTrimmable(after: 5))
    }

    @Test func testRotatingKVCacheIsNotTrimmableAtOrPastWindow() {
        // `isTrimmable` is the *exact rewind* predicate. Past the window a trim is
        // merely consistent (the window loses its oldest rows), so this must stay
        // false to keep the staged-round and prompt-cache-reuse machinery exact.
        let cache = RotatingKVCache(maxSize: 8)

        cache.offset = 8
        #expect(!cache.isTrimmable)
        #expect(!cache.isTrimmable(after: 0))

        cache.offset = 9
        #expect(!cache.isTrimmable)
        #expect(!cache.isTrimmable(after: 0))
    }

    @Test func testRotatingPredictiveTrimOverrideDispatchesThroughKVCache() {
        let rotating = RotatingKVCache(maxSize: 8)
        rotating.offset = 3
        let cache: any KVCache = rotating

        #expect(cache.isTrimmable(after: 4))
        #expect(!cache.isTrimmable(after: 5))
    }

    @Test func testUnboundedCacheRemainsTrimmableAfterPositiveLookAhead() {
        let cache: any KVCache = KVCacheSimple()

        #expect(cache.isTrimmable(after: 1_000))
    }

    @Test func testCacheListDelegatesPredictiveTrimmabilityToChildren() {
        let rotating = RotatingKVCache(maxSize: 8)
        rotating.offset = 3
        let cache: any KVCache = CacheList(KVCacheSimple(), rotating)

        #expect(cache.isTrimmable(after: 4))
        #expect(!cache.isTrimmable(after: 5))
    }

    @Test func testDirectKVCacheConformerUsesPredictiveTrimDefault() {
        let cache: any KVCache = ProtocolDefaultTrimmabilityCache()

        #expect(cache.isTrimmable(after: 1_000))
    }

    // MARK: - Original parameterized test (updated with value assertions)

    @Test(
        .serialized,
        arguments: cacheCreators)
    func testCacheSerialization(creator: (() -> any KVCache)) async throws {
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

        let url = TemporaryPromptCacheFile.makeURL()

        try savePromptCache(url: url, cache: cache, metadata: [:])
        let (loadedCache, _) = try loadPromptCache(url: url)

        #expect(cache.count == loadedCache.count)
        for (lhs, rhs) in zip(cache, loadedCache) {
            #expect(type(of: lhs) == type(of: rhs))
            #expect(lhs.metaState == rhs.metaState)
            assertArraysClose(lhs.state, rhs.state)
        }
    }

    @Test func testPromptCacheStateRoundTrip() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, 2, 4, 8], dtype: .bfloat16)
        let values = MLXArray.zeros([1, 2, 4, 8], dtype: .bfloat16)
        _ = cache.update(keys: keys, values: values)

        let ropeDeltasKey = LMOutput.Key<MLXArray>("test.ropeDeltas")
        let positionsKey = LMOutput.Key<MLXArray>("test.positionIds")
        let ropeDeltas = MLXArray([Int32(3), 5, 7]).reshaped([1, 3])
        let positions = MLXArray([Int32(11), 13, 17, 19]).reshaped([2, 2])
        var state = LMOutput.State()
        state[ropeDeltasKey] = ropeDeltas
        state[positionsKey] = positions

        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(
            url: url, cache: [cache], metadata: ["source": "test"], state: state)

        let snapshot = try loadPromptCacheSnapshot(url: url)
        let restoredRopeDeltas = try #require(snapshot.state?[ropeDeltasKey])
        let restoredPositions = try #require(snapshot.state?[positionsKey])

        #expect(snapshot.metadata == ["source": "test"])
        #expect(restoredRopeDeltas.dtype == ropeDeltas.dtype)
        #expect(restoredRopeDeltas.shape == ropeDeltas.shape)
        #expect(allClose(restoredRopeDeltas, ropeDeltas, rtol: 0, atol: 0).item(Bool.self))
        #expect(restoredPositions.dtype == positions.dtype)
        #expect(restoredPositions.shape == positions.shape)
        #expect(allClose(restoredPositions, positions, rtol: 0, atol: 0).item(Bool.self))

        let (storedArrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        #expect(storedArrays["__mlx_lm_state_tensor_0"] != nil)
        #expect(storedArrays["__mlx_lm_state_tensor_1"] != nil)
        #expect(
            storedArrays.keys.allSatisfy {
                Int($0.split(separator: ".")[0]) != nil
                    || $0.hasPrefix("__mlx_lm_state_tensor_")
            })
        #expect(storedMetadata["1.__mlx_lm_state_0_key"] == "test.positionIds")
        #expect(storedMetadata["1.__mlx_lm_state_1_key"] == "test.ropeDeltas")
        #expect(storedMetadata["2.0"] == "__mlx_lm_state_v1__:KVCache")
        #expect(throws: KVCacheError.self) {
            try loadPromptCache(url: url)
        }
    }

    @Test func testLegacyPromptCacheLoadsWithNilState() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let snapshot = try loadPromptCacheSnapshot(url: url)
        let (legacyCache, legacyMetadata) = try loadPromptCache(url: url)
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)

        #expect(snapshot.cache.count == 1)
        #expect(snapshot.state == nil)
        #expect(legacyCache.count == 1)
        #expect(legacyMetadata.isEmpty)
        #expect(arrays.keys.allSatisfy { Int($0.split(separator: ".")[0]) != nil })
        #expect(!metadata.keys.contains { $0.contains("__mlx_lm_state_") })
        #expect(metadata["2.0"] == "KVCache")
    }

    @Test func testLoadedPromptCacheNoLongerReadsItsFile() throws {
        let cache = KVCacheSimple()
        let keys = MLXArray(0 ..< 32).reshaped([1, 2, 4, 4]).asType(.float32)
        let values = -keys
        _ = cache.update(keys: keys, values: values)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache])

        // Saving truncates the file before it evaluates what it writes.
        let snapshot = try loadPromptCacheSnapshot(url: url)
        try savePromptCache(url: url, cache: snapshot.cache)

        let restored = try loadPromptCacheSnapshot(url: url)
        assertArraysClose(restored.cache[0].state, [keys, values])
    }

    @Test func testEmptyPromptCacheStateUsesTheLegacyFormat() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let legacyURL = TemporaryPromptCacheFile.makeURL()
        let emptyStateURL = TemporaryPromptCacheFile.makeURL()
        defer {
            try? FileManager.default.removeItem(at: legacyURL)
            try? FileManager.default.removeItem(at: emptyStateURL)
        }

        try savePromptCache(url: legacyURL, cache: [cache])
        try savePromptCache(url: emptyStateURL, cache: [cache], state: LMOutput.State())
        let snapshot = try loadPromptCacheSnapshot(url: emptyStateURL)
        let (legacyCache, _) = try loadPromptCache(url: emptyStateURL)
        let (legacyArrays, legacyMetadata) = try loadArraysAndMetadata(url: legacyURL)
        let (emptyStateArrays, emptyStateMetadata) = try loadArraysAndMetadata(url: emptyStateURL)

        #expect(snapshot.state == nil)
        #expect(legacyCache.count == 1)
        #expect(Set(emptyStateArrays.keys) == Set(legacyArrays.keys))
        #expect(emptyStateMetadata == legacyMetadata)
    }

    @Test func testPromptCacheStateRejectsReservedUserMetadata() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            try savePromptCache(
                url: url, cache: [cache],
                metadata: ["__mlx_lm_state_version": "caller-owned"])
        }
    }

    @Test func testPromptCacheStateRejectsUnsupportedValues() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let unsupportedKey = LMOutput.Key<Bool>("test.unsupported")
        var state = LMOutput.State()
        state[unsupportedKey] = true
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: LMOutput.State.SerializationError.self) {
            try savePromptCache(url: url, cache: [cache], state: state)
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func testPromptCacheStateRejectsMalformedEntries() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)

        // eval before overwriting file -- otherwise the lazy read picks
        // up the corruption
        eval(arrays)

        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_count"] = "2"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsUnknownVersions() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_version"] = "999"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsDuplicateKeys() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let firstKey = LMOutput.Key<MLXArray>("test.first")
        let secondKey = LMOutput.Key<MLXArray>("test.second")
        var state = LMOutput.State()
        state[firstKey] = MLXArray([Int32(1)])
        state[secondKey] = MLXArray([Int32(2)])
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = savedMetadata
        malformedMetadata["1.__mlx_lm_state_1_key"] = "test.first"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRejectsUnexpectedTensors() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (storedArrays, metadata) = try loadArraysAndMetadata(url: url)
        eval(storedArrays)

        var malformedArrays = storedArrays
        malformedArrays["__mlx_lm_state_tensor_99"] = MLXArray([Int32(99)])
        try save(arrays: malformedArrays, metadata: metadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheStateRequiresCompatibilityMarker() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let key = LMOutput.Key<MLXArray>("test.state")
        var state = LMOutput.State()
        state[key] = MLXArray([Int32(1)])
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], state: state)
        let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)
        var malformedMetadata = storedMetadata
        malformedMetadata["2.0"] = "KVCache"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testPromptCacheCompatibilityMarkerRequiresState() throws {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 1, 4]),
            values: MLXArray.zeros([1, 1, 1, 4]))
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let (arrays, storedMetadata) = try loadArraysAndMetadata(url: url)
        eval(arrays)

        var malformedMetadata = storedMetadata
        malformedMetadata["2.0"] = "__mlx_lm_state_v1__:KVCache"
        try save(arrays: arrays, metadata: malformedMetadata, url: url)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
    }

    @Test func testQuantizedKVCacheRestoresNonDefaultQuantizationMetadata() throws {
        let cache = QuantizedKVCache(groupSize: 64, bits: 4)
        let keys = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, 4, 32], dtype: .bfloat16)
        _ = cache.updateQuantized(keys: keys, values: values)

        #expect(cache.groupSize == 32)
        #expect(cache.bits == 4)

        let url = TemporaryPromptCacheFile.makeURL()
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
        let cache = QuantizedKVCache()

        cache.metaState = ["256", "11", "32", "4"]

        #expect(cache.offset == 11)
        #expect(cache.groupSize == 32)
        #expect(cache.bits == 4)
        #expect(cache.metaState == ["256", "11", "32", "4"])
    }

    @Test func testQuantizedKVCacheCopyPreservesRestoredQuantizationMetadata() throws {
        let cache = QuantizedKVCache()
        cache.metaState = ["256", "5", "32", "4"]

        let copied = try #require(cache.copy() as? QuantizedKVCache)

        #expect(copied.offset == 5)
        #expect(copied.groupSize == 32)
        #expect(copied.bits == 4)
        #expect(copied.metaState == cache.metaState)
    }

    @Test func testEmptyKVCacheSimpleToQuantizedPreservesRequestedQuantizationMetadata() throws {
        let cache = KVCacheSimple()
        cache.offset = 7

        let quantized = try cache.toQuantized(groupSize: 128, bits: 4)

        #expect(quantized.offset == 7)
        #expect(quantized.groupSize == 128)
        #expect(quantized.bits == 4)
        #expect(quantized.metaState == ["256", "7", "128", "4"])
    }

    @Test func testDirectKVCacheQuantizationFailuresAreRecoverable() throws {
        let incompatible = KVCacheSimple()
        incompatible.state = [
            MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
            MLXArray.zeros([1, 2, 4, 5], dtype: .float32, stream: .cpu),
        ]

        #expect(throws: KVCacheError.self) {
            _ = try incompatible.toQuantized(groupSize: 64, bits: 4)
        }
        #expect(throws: KVCacheError.self) {
            _ = try RotatingKVCache(maxSize: 32).toQuantized()
        }
    }

    @Test func testPromptCacheRestorationRejectsIncompleteState() throws {
        try expectPromptCacheLoadToFail([
            .init(className: "KVCacheSimple", arrayCount: 1, metadata: [""]),
            .init(
                className: "RotatingKVCache", arrayCount: 1,
                metadata: ["0", "32", "256", "0", "0", "modelNative"]),
            .init(
                className: "QuantizedKVCache", arrayCount: 3,
                metadata: ["256", "0", "64", "4"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 7,
                metadata: ["32", "32", "4", "4", "2", "1", "0"]),
            .init(className: "ChunkedKVCache", arrayCount: 1, metadata: ["None", "0"]),
        ])
    }

    @Test func testPromptCacheRestorationRejectsInvalidMetadata() throws {
        try expectPromptCacheLoadToFail([
            .init(className: "KVCacheSimple", arrayCount: 0, metadata: ["unexpected"]),
            .init(
                className: "RotatingKVCache", arrayCount: 0,
                metadata: ["0", "32", "invalid", "0", "0", "modelNative"]),
            .init(
                className: "RotatingKVCache", arrayCount: 0,
                metadata: ["0", "32", "256", "0", "0", "invalid-origin"]),
            .init(
                className: "QuantizedKVCache", arrayCount: 0,
                metadata: ["256", "invalid", "64", "4"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 0,
                metadata: ["24", "0", "4", "4", "2", "0", "0"]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 8,
                metadata: ["32", "31", "4", "4", "2", "1", "0"]),
            .init(className: "ChunkedKVCache", arrayCount: 0, metadata: ["invalid", "0"]),
        ])
    }

    @Test func testPromptCacheRestorationRejectsInvalidStateRank() throws {
        try expectPromptCacheLoadToFail([
            .init(
                className: "KVCacheSimple", arrayCount: 2, metadata: [""],
                arrayShape: [1, 1]),
            .init(
                className: "VarianceNormalizedKVCache", arrayCount: 8,
                metadata: ["32", "32", "4", "4", "2", "1", "0"],
                arrayShape: [1, 1]),
        ])
    }

    @Test func testPromptCacheRestorationAcceptsLegacyRotatingMetadata() throws {
        let fixture = SerializedCacheFixture(
            className: "RotatingKVCache",
            arrayCount: 2,
            metadata: ["0", "32", "256", "1", "1"])
        let url = try writePromptCacheFixture(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        let (restored, _) = try loadPromptCache(url: url)
        let rotating = try #require(restored.first as? RotatingKVCache)

        #expect(rotating.capacityOrigin == .modelNative)
        // Legacy metadata has no wrapped flag; the restored cache derives it (an
        // unwrapped layout here) and re-serializes with the flag appended.
        #expect(rotating.metaState == ["0", "32", "256", "1", "1", "modelNative", "false"])
    }

    @Test func testPromptCacheRoundTripPreservesEmptyCaches() throws {
        let populated = KVCacheSimple()
        populated.state = [
            MLXArray.ones([1, 1, 1, 32], stream: .cpu),
            MLXArray.ones([1, 1, 1, 32], stream: .cpu),
        ]
        let caches: [KVCache] = [
            KVCacheSimple(),
            populated,
            RotatingKVCache(maxSize: 32),
            QuantizedKVCache(),
            ChunkedKVCache(chunkSize: 16),
        ]
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: caches)
        let (restored, _) = try loadPromptCache(url: url)

        #expect(restored.count == caches.count)
        #expect(restored[0] is KVCacheSimple)
        #expect(restored[0].state.isEmpty)
        #expect(restored[1].state.count == 2)
        #expect(restored[2] is RotatingKVCache)
        #expect(restored[2].state.isEmpty)
        #expect(restored[3] is QuantizedKVCache)
        #expect(restored[3].state.isEmpty)
        #expect(restored[4] is ChunkedKVCache)
        #expect(restored[4].state.isEmpty)
    }

    /// Verify that copy() produces an independent cache: same type, same state,
    /// but mutating the copy does not affect the original.
    @Test(
        .serialized,
        arguments: cacheCreators)
    func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
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
        let empty = creator()
        let copied = empty.copy()

        #expect(type(of: empty) == type(of: copied))
        #expect(copied.offset == 0)
        #expect(copied.state.count == empty.state.count)
    }

    /// CacheList.copy() produces independent sub-caches.
    @Test
    func testCacheListCopyIsIndependent() async throws {
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

    // MARK: - ropeOffset overridability

    /// A `BaseKVCache` subclass reporting a per-row RoPE offset, as a batched cache does.
    private final class BatchOffsetProbeCache: BaseKVCache {
        override var ropeOffset: RoPEOffset { .batch(MLXArray([10, 20])) }

        override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            (keys, values)
        }
    }

    /// Models read `ropeOffset` through a `KVCache` reference, so a subclass override has to
    /// survive that dispatch. When `BaseKVCache` did not declare `ropeOffset`, the extension
    /// default was the witness and this resolved to `.scalar(0)`, silently ignoring the subclass.
    @Test func testSubclassRopeOffsetOverrideIsHonoredThroughKVCacheReference() {
        let cache: any KVCache = BatchOffsetProbeCache()

        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record(
                "subclass ropeOffset override ignored — resolved to the .scalar extension default")
            return
        }
        #expect(offsets.asArray(Int32.self) == [10, 20])
    }
}
