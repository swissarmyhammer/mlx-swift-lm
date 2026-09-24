// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Tests for `KVCache.residentByteCount` and `LMOutput.State.residentByteCount`.
///
/// Each test reads the count through `any KVCache`, because the prompt cache
/// store holds `[KVCache]`. A count that uses static dispatch gives the wrong
/// value there.
struct KVCacheByteCountTests {

    // MARK: - Fixture sizes

    /// The number of sequences in each fixture.
    private static let batchSize = 1

    /// The number of key/value heads in each fixture.
    private static let headCount = 2

    /// The head dimension of each fixture. It is a power of two and a group
    /// size that every quantized cache accepts.
    private static let headDim = 64

    /// The number of tokens of a short prefill. It is not a multiple of the
    /// allocation step, thus a step-padded buffer is larger than the state.
    private static let promptTokenCount = 3

    /// The allocation step of `KVCacheSimple` and of the raw buffers of
    /// `TurboQuantKVCache`.
    private static let allocationStep = 256

    /// The window of the `RotatingKVCache` fixture.
    private static let rotatingWindow = 16

    /// The group size of the `QuantizedKVCache` fixture.
    private static let quantizedGroupSize = 64

    /// The bit width of the `QuantizedKVCache` fixture.
    private static let quantizedBits = 8

    /// The number of bits in one packed `uint32` element.
    private static let bitsPerPackedElement = 32

    /// The number of tokens that fills one tile of `VarianceNormalizedKVCache`
    /// and leaves a raw tail.
    private static let tiledTokenCount = 130

    /// The index of the DeepSeek-V4 fixture layer. Its compress ratio is 4,
    /// thus it has both a compressor and an indexer.
    private static let deepSeekIndexerLayer = 2

    /// The number of pooled chunks in each DeepSeek-V4 branch fixture.
    private static let deepSeekChunkCount = 5

    /// The number of raw carry rows in each DeepSeek-V4 branch fixture.
    private static let deepSeekCarryRowCount = 3

    /// The hidden width of the DeepSeek-V4 branch fixtures.
    private static let deepSeekChunkWidth = 8

    /// The shape of the array in the `ArraysCache` fixture slot.
    private static let arraysSlotShape = [1, 3, 5]

    /// The number of slots of the `ArraysCache` fixture.
    private static let arraysSlotCount = 2

    /// The number of buffers that hold one tensor pair: the keys and the values.
    private static let keysAndValues = 2

    /// The number of step-padded buffers of `MiniMaxM3KVCache`: the keys, the
    /// values and the index keys.
    private static let miniMaxBufferCount = 3

    /// The number of children of the `CacheList` fixture.
    private static let cacheListChildCount = 2

    /// The number of chunk branches of a DeepSeek-V4 layer with an indexer.
    private static let deepSeekBranchCount = 2

    /// The number of group tensors of one quantized tensor: the scales and the
    /// biases, which have the same size.
    private static let groupTensorsPerQuantizedTensor = 2

    /// The factor that makes each fixture array a lazy operation.
    private static let lazyFactor: Float = 2

    // MARK: - Fixture builders

    /// Builds a lazy array: nothing evaluates it.
    ///
    /// - Parameters:
    ///   - shape: The shape of the array.
    ///   - dtype: The element type of the array.
    /// - Returns: The array, which is the product of an operation that has not run.
    private static func lazyArray(_ shape: [Int], dtype: DType = .float16) -> MLXArray {
        MLXArray.ones(shape, dtype: dtype) * lazyFactor
    }

    /// The shape of the keys (and of the values) of one short prefill.
    private static var promptShape: [Int] {
        [batchSize, headCount, promptTokenCount, headDim]
    }

    /// The number of bytes of one step-padded key (or value) buffer after one
    /// short prefill.
    ///
    /// - Parameter dtype: The element type of the buffer.
    /// - Returns: The byte count of the buffer.
    private static func paddedBufferBytes(dtype: DType) -> Int {
        batchSize * headCount * allocationStep * headDim * dtype.size
    }

    /// Tells whether an array holds evaluated data.
    ///
    /// - Parameter array: The array to read.
    /// - Returns: True when the data of the array is available.
    private static func isAvailable(_ array: MLXArray) -> Bool {
        var available = false
        _mlx_array_is_available(&available, array.ctx)
        return available
    }

    /// The sum of `nbytes` over some arrays.
    ///
    /// - Parameter arrays: The arrays to count.
    /// - Returns: The byte count.
    private static func byteCount(of arrays: [MLXArray]) -> Int {
        arrays.reduce(0) { $0 + $1.nbytes }
    }

    /// Feeds one short lazy prefill into a cache.
    ///
    /// - Parameter cache: The cache to feed.
    /// - Returns: The same cache, typed as `any KVCache`.
    private static func prefilled(_ cache: any KVCache) -> any KVCache {
        _ = cache.update(keys: lazyArray(promptShape), values: lazyArray(promptShape))
        return cache
    }

    // MARK: - Empty caches

    /// Builds every cache type before its first write.
    ///
    /// - Returns: One empty cache of each type.
    private static func emptyCaches() throws -> [any KVCache] {
        [
            KVCacheSimple(),
            ChunkedKVCache(chunkSize: allocationStep),
            RotatingKVCache(maxSize: rotatingWindow),
            QuantizedKVCache(groupSize: quantizedGroupSize, bits: quantizedBits),
            ArraysCache(size: arraysSlotCount),
            MambaCache(),
            CacheList(KVCacheSimple(), TurboQuantKVCache()),
            VarianceNormalizedKVCache(),
            TurboQuantKVCache(),
            DeepSeekV4Cache(
                configuration: try deepSeekConfiguration(), layer: deepSeekIndexerLayer),
            MiniMaxM3KVCache(),
        ]
    }

    @Test("An empty cache of each type holds 0 bytes")
    func emptyCacheHoldsZeroBytes() throws {
        for cache in try Self.emptyCaches() {
            #expect(cache.residentByteCount == 0, "\(type(of: cache))")
        }
    }

    // MARK: - One case for each type

    @Test("KVCacheSimple counts the full step-padded buffers, not the state slice")
    func simpleCacheCountsPaddedBuffers() {
        let cache = Self.prefilled(KVCacheSimple())

        #expect(
            cache.residentByteCount == Self.keysAndValues * Self.paddedBufferBytes(dtype: .float16))
        #expect(Self.byteCount(of: cache.state) < cache.residentByteCount)
    }

    @Test("ChunkedKVCache counts the full step-padded buffers")
    func chunkedCacheCountsPaddedBuffers() {
        let cache = Self.prefilled(ChunkedKVCache(chunkSize: Self.allocationStep))

        #expect(
            cache.residentByteCount == Self.keysAndValues * Self.paddedBufferBytes(dtype: .float16))
    }

    @Test("RotatingKVCache counts the full ring buffers")
    func rotatingCacheCountsRingBuffers() {
        let shape = [Self.batchSize, Self.headCount, Self.rotatingWindow, Self.headDim]
        let keys = Self.lazyArray(shape)
        let values = Self.lazyArray(shape, dtype: .float32)
        var cache: any KVCache = RotatingKVCache(maxSize: Self.rotatingWindow)
        cache.state = [keys, values]

        let ringBytes = Self.batchSize * Self.headCount * Self.rotatingWindow * Self.headDim
        #expect(
            cache.residentByteCount
                == ringBytes * DType.float16.size + ringBytes * DType.float32.size)
    }

    @Test("QuantizedKVCache counts packed values, scales and biases of keys and values")
    func quantizedCacheCountsPackedScalesAndBiases() {
        let cache = QuantizedKVCache(groupSize: Self.quantizedGroupSize, bits: Self.quantizedBits)
        _ = cache.updateQuantized(
            keys: Self.lazyArray(Self.promptShape, dtype: .float32),
            values: Self.lazyArray(Self.promptShape, dtype: .float32))

        let rows = Self.batchSize * Self.headCount * Self.allocationStep
        let packedBytes =
            rows * (Self.headDim * Self.quantizedBits / Self.bitsPerPackedElement)
            * DType.uint32.size
        let groupBytes = rows * (Self.headDim / Self.quantizedGroupSize) * DType.float32.size
        let perTensor = packedBytes + Self.groupTensorsPerQuantizedTensor * groupBytes
        #expect((cache as any KVCache).residentByteCount == Self.keysAndValues * perTensor)
    }

    @Test("ArraysCache and MambaCache count the present slots only")
    func arraysCacheCountsPresentSlots() {
        let slot = Self.lazyArray(Self.arraysSlotShape, dtype: .float32)
        let slotBytes = Self.arraysSlotShape.reduce(1, *) * DType.float32.size

        let arrays = ArraysCache(size: Self.arraysSlotCount)
        arrays[0] = slot
        #expect((arrays as any KVCache).residentByteCount == slotBytes)

        let mamba = MambaCache()
        mamba[1] = slot
        #expect((mamba as any KVCache).residentByteCount == slotBytes)
    }

    @Test("CacheList counts the sum of its children, with the rule of each child")
    func cacheListCountsEachChild() {
        let simple = Self.prefilled(KVCacheSimple())
        let turbo = Self.prefilled(TurboQuantKVCache())
        let list: any KVCache = CacheList(simple, turbo)

        #expect(list.residentByteCount == simple.residentByteCount + turbo.residentByteCount)
        #expect(
            list.residentByteCount == Self.cacheListChildCount * Self.keysAndValues
                * Self.paddedBufferBytes(dtype: .float16))
    }

    @Test("VarianceNormalizedKVCache counts the raw tail when it holds no tile")
    func varianceNormalizedCacheCountsTail() {
        let cache = Self.prefilled(VarianceNormalizedKVCache())

        #expect(cache.residentByteCount > 0)
        #expect(cache.residentByteCount == Self.byteCount(of: cache.state))
    }

    @Test("VarianceNormalizedKVCache counts the compact tiles and the raw tail")
    func varianceNormalizedCacheCountsTilesAndTail() {
        let cache = VarianceNormalizedKVCache()
        let shape = [Self.batchSize, Self.headCount, Self.tiledTokenCount, Self.headDim]
        _ = cache.update(keys: Self.lazyArray(shape), values: Self.lazyArray(shape))

        #expect(cache.compactStorageByteCount > 0)
        #expect((cache as any KVCache).residentByteCount == Self.byteCount(of: cache.state))
    }

    @Test("TurboQuantKVCache with data counts its raw prefill buffers")
    func turboQuantCacheCountsRawBuffers() {
        let cache = Self.prefilled(TurboQuantKVCache())

        #expect(cache.residentByteCount > 0)
        #expect(
            cache.residentByteCount == Self.keysAndValues * Self.paddedBufferBytes(dtype: .float16))
    }

    /// Decodes the DeepSeek-V4 configuration that the test bundle holds.
    ///
    /// - Returns: The configuration.
    private static func deepSeekConfiguration() throws -> DeepSeekV4Configuration {
        let url = try #require(
            Bundle.module.url(
                forResource: "DeepSeek-V4-Flash-4bit-config", withExtension: "json"))
        return try JSONDecoder().decode(
            DeepSeekV4Configuration.self, from: Data(contentsOf: url))
    }

    /// Builds the three state slots of one DeepSeek-V4 branch.
    ///
    /// - Returns: The chunks, the carry rows and the carry start.
    private static func deepSeekBranchSlots() -> [MLXArray] {
        [
            lazyArray([batchSize, deepSeekChunkCount, deepSeekChunkWidth]),
            lazyArray([batchSize, deepSeekCarryRowCount, deepSeekChunkWidth]),
            MLXArray([Int32(0)]),
        ]
    }

    @Test("DeepSeekV4Cache counts the window and both chunk branches")
    func deepSeekCacheCountsWindowAndBranches() throws {
        let configuration = try Self.deepSeekConfiguration()
        #expect(configuration.hasIndexer(layer: Self.deepSeekIndexerLayer))
        var cache: any KVCache = DeepSeekV4Cache(
            configuration: configuration, layer: Self.deepSeekIndexerLayer)
        let window = [Self.lazyArray(Self.promptShape), Self.lazyArray(Self.promptShape)]
        cache.state = window + Self.deepSeekBranchSlots() + Self.deepSeekBranchSlots()

        let windowBytes = Self.byteCount(of: window)
        let branchRows = Self.deepSeekChunkCount + Self.deepSeekCarryRowCount
        let branchBytes =
            Self.batchSize * branchRows * Self.deepSeekChunkWidth * DType.float16.size
        #expect(cache.residentByteCount == windowBytes + Self.deepSeekBranchCount * branchBytes)
    }

    @Test("MiniMaxM3KVCache counts the padded keys, values and index keys")
    func miniMaxCacheCountsKeysValuesAndIndexKeys() {
        let cache = MiniMaxM3KVCache()
        _ = cache.update(
            keys: Self.lazyArray(Self.promptShape), values: Self.lazyArray(Self.promptShape))
        _ = cache.updateIndexAndFetch(Self.lazyArray(Self.promptShape))

        #expect(
            (cache as any KVCache).residentByteCount
                == Self.miniMaxBufferCount * Self.paddedBufferBytes(dtype: .float16))
    }

    // MARK: - LMOutput.State

    @Test("LMOutput.State counts its array values and skips other values")
    func lmOutputStateCountsArrayValues() {
        let shape = [Self.batchSize, Self.headDim]
        var state = LMOutput.State()
        state[LMOutput.Key<MLXArray>("hidden")] = Self.lazyArray(shape, dtype: .float32)
        state[LMOutput.Key<Int>("position")] = Self.promptTokenCount

        #expect(state.residentByteCount == Self.batchSize * Self.headDim * DType.float32.size)
        #expect(LMOutput.State().residentByteCount == 0)
    }

    // MARK: - No evaluation

    @Test("Reading the count evaluates no array of the cache or of the state")
    func countEvaluatesNothing() {
        let caches: [any KVCache] = [
            Self.prefilled(KVCacheSimple()),
            Self.prefilled(TurboQuantKVCache()),
            CacheList(Self.prefilled(KVCacheSimple())),
        ]
        var state = LMOutput.State()
        let stateArray = Self.lazyArray(Self.promptShape)
        state[LMOutput.Key<MLXArray>("hidden")] = stateArray

        for cache in caches {
            #expect(cache.residentByteCount > 0)
        }
        #expect(state.residentByteCount > 0)

        for array in caches.flatMap({ $0.innerState() }) + [stateArray] {
            #expect(!Self.isAvailable(array))
        }
    }
}
