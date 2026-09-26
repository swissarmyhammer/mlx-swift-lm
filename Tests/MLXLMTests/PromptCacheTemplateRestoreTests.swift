// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Tests for `loadPromptCacheSnapshot(url:into:)`, which restores a saved prompt cache into the
/// fresh caches that a model made.
///
/// Each round trip saves a filled cache, restores it into a fresh template, and then compares
/// the `state`, the `metaState` and the output of one more step with the original cache.
struct PromptCacheTemplateRestoreTests {

    // MARK: - Fixture sizes

    /// The number of sequences in each fixture.
    fileprivate static let batchSize = 1

    /// The number of key/value heads in each fixture.
    fileprivate static let headCount = 2

    /// The head dimension of each fixture. Each quantized cache accepts it as a group size.
    fileprivate static let headDim = 64

    /// The number of tokens of one prefill.
    fileprivate static let promptTokenCount = 5

    /// The number of tokens that `VarianceNormalizedKVCache` receives. It fills one tile and
    /// leaves a raw tail.
    fileprivate static let varianceTokenCount = 40

    /// The tile size of the `VarianceNormalizedKVCache` fixture.
    fileprivate static let varianceTileSize = 32

    /// The key and value bit width of the `VarianceNormalizedKVCache` fixture.
    fileprivate static let varianceBits = 4

    /// The Sinkhorn iteration count of the `VarianceNormalizedKVCache` fixture.
    fileprivate static let varianceSinkhornIterations = 2

    /// The window of the `RotatingKVCache` fixtures.
    fileprivate static let rotatingWindow = 4

    /// The number of single-token steps that leave the ring before its first wrap.
    fileprivate static let rotatingTokensBeforeWrap = 3

    /// The number of single-token steps that wrap the ring.
    fileprivate static let rotatingTokensAfterWrap = 6

    /// The chunk size of the `ChunkedKVCache` fixture.
    fileprivate static let chunkSize = 16

    /// The group size of the `QuantizedKVCache` fixture.
    fileprivate static let quantizedGroupSize = 64

    /// The bit width of the `QuantizedKVCache` fixture.
    fileprivate static let quantizedBits = 8

    /// The bit width of the `TurboQuantKVCache` fixture.
    fileprivate static let turboQuantBits = 4

    /// The number of slots of the `ArraysCache` fixture.
    fileprivate static let arraysSlotCount = 2

    /// The shape of the convolution slot of the recurrent fixtures.
    fileprivate static let convolutionSlotShape = [1, 3, 5]

    /// The shape of the recurrent slot of the recurrent fixtures.
    fileprivate static let recurrentSlotShape = [1, 2, 4, 4]

    /// The index of the DeepSeek-V4 layer with a compressor and an indexer (ratio 4).
    fileprivate static let deepSeekIndexerLayer = 2

    /// The index of the DeepSeek-V4 layer with a compressor and no indexer (ratio 128).
    fileprivate static let deepSeekPlainLayer = 3

    /// The number of pooled chunks in each DeepSeek-V4 branch fixture.
    fileprivate static let deepSeekChunkCount = 5

    /// The number of raw carry rows in each DeepSeek-V4 branch fixture.
    fileprivate static let deepSeekCarryRowCount = 3

    /// The width of the rows of the DeepSeek-V4 branch fixtures.
    fileprivate static let deepSeekRowWidth = 8

    /// The number of values that each tampered meta state gets in addition. It is more than
    /// the optional values of any cache class.
    fileprivate static let extraMetaStateCount = 3

    /// The first seed of the fixture tensors.
    fileprivate static let firstSeed: UInt64 = 7

    /// The number of tokens that the front-trimmed `ChunkedKVCache` fixture receives before
    /// its front trim. It is more than one chunk, thus the trim removes tokens.
    fileprivate static let frontTrimmedTokenCount = 24

    /// The reserved prefix of the metadata keys that record the offset of each cache.
    fileprivate static let offsetRecordPrefix = "__mlx_lm_offset_"

    /// The prefix of the flattened keys of the user-metadata part of a prompt cache file.
    fileprivate static let userMetadataPart = "1."

    /// The place of the header of the first child in the meta state of a saved `CacheList`.
    /// The child count is in front of it.
    fileprivate static let cacheListFirstChildHeader = 1

    /// The number of values in the header of each child of a saved `CacheList`: its class
    /// name, its array count and its meta-state count.
    fileprivate static let cacheListChildHeaderCount = 3

    /// The place of the array count in the header of a child of a saved `CacheList`.
    fileprivate static let cacheListStateCountOffset = 1

    /// The place of the meta-state count in the header of a child of a saved `CacheList`.
    fileprivate static let cacheListMetaStateCountOffset = 2

    /// The place of the second child of ``CacheKind/cacheList``.
    fileprivate static let cacheListSecondChild = 1

    /// The number of arrays that each filled ``CustomSlotCache`` holds. It is not an array
    /// count of `KVCacheSimple`.
    fileprivate static let customSlotCount = 3

    /// The flattened file key of the class name of the first layer, in the class-name part.
    fileprivate static let firstLayerClassNameKey = "2.0"

    /// A class name that no built-in class and no registry entry uses.
    fileprivate static let unknownClassName = "UnknownTestCache"

    /// The prefix of the flattened keys of the class-name part of a prompt cache file.
    fileprivate static let classNamePart = "2."

    /// The prefix of the flattened keys of the meta state of the second layer, in the
    /// cache-information part.
    fileprivate static let secondLayerMetaStatePrefix = "0.1."

    /// The flattened file key of the first array of the first layer.
    fileprivate static let firstLayerFirstArrayKey = "0.0"

    /// The key of the model-state array that the files with a model state hold.
    fileprivate static let positionsStateKey = LMOutput.Key<MLXArray>("test.positions")

    /// The flattened file key of the model-state count, in the user-metadata part.
    fileprivate static let modelStateCountKey = "1.__mlx_lm_state_count"

    /// A flattened file key in the reserved model-state namespace that no reader knows.
    fileprivate static let unknownModelStateKey = "1.__mlx_lm_state_unknown"

    /// The message of the check that finds a cache-information count that is not the
    /// class-name count.
    fileprivate static let cacheCountMismatchMessage = "Mismatch in cache counts"

    /// The message of the check that finds no valid model-state count.
    fileprivate static let invalidModelStateCountMessage = "Invalid prompt cache state count"

    /// The message of the check that finds a reserved model-state key that the reader does
    /// not know.
    fileprivate static let unexpectedModelStateMessage = "Unexpected prompt cache state metadata"

    /// The place of the wrapped flag in the meta state of a saved `RotatingKVCache`.
    fileprivate static let rotatingWrappedIndex = 6

    /// A wrapped flag that is not `true` or `false`.
    fileprivate static let invalidWrappedFlag = "maybe"

    /// The place of the offset in the meta state of a saved `VarianceNormalizedKVCache`.
    fileprivate static let varianceOffsetIndex = 1

    /// The place of the complete tile count in the meta state of a saved
    /// `VarianceNormalizedKVCache`.
    fileprivate static let varianceTileCountIndex = 5

    /// The meta state that a `MambaCache` or an `ArraysCache` saved before it recorded its
    /// slot count. It has no slot numbers, thus the load puts the arrays in the slots in order.
    fileprivate static let legacyArraysMetaState = [""]

    /// The message of the `CacheList` check that finds no valid child count.
    fileprivate static let missingChildCountMessage = "CacheList metaState missing child count"

    /// The message of the `CacheList` check that finds a child header cut short.
    fileprivate static let truncatedHeaderMessage = "CacheList metaState truncated"

    /// The message of the `CacheList` check that finds a child count past the saved values.
    fileprivate static let invalidChildCountMessage =
        "CacheList: invalid array or metaState count for child"

    /// The kinds whose restored offset the load checks against the record: every kind that
    /// ``OffsetCacheKind`` does not cover.
    fileprivate static let checkedOffsetKinds = CacheKind.allCases.filter {
        ![.mamba, .arrays, .chunked].contains($0)
    }

    /// The checked kinds that `loadPromptCacheSnapshot(url:)` can build without a template.
    fileprivate static let checkedBuiltInKinds: [CacheKind] = [
        .simple, .rotatingBeforeWrap, .rotatingAfterWrap, .quantized, .turboQuant,
        .varianceNormalized, .cacheList,
    ]

    /// The flattened file key of the offset record of one layer.
    ///
    /// - Parameter layer: The index of the layer.
    /// - Returns: The key, in the user-metadata part.
    fileprivate static func offsetRecordKey(layer: Int) -> String {
        "\(userMetadataPart)\(offsetRecordPrefix)\(layer)"
    }

    // MARK: - Fixture builders

    /// Makes a reproducible tensor.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - seed: The seed of the random key.
    /// - Returns: The tensor, as `float16`.
    fileprivate static func tensor(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.float16)
    }

    /// Makes the keys or the values of a block of tokens.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens in the block.
    ///   - seed: The seed of the random key.
    /// - Returns: A tensor of shape `(batch, heads, tokens, headDim)`.
    fileprivate static func block(tokenCount: Int, seed: UInt64) -> MLXArray {
        tensor([batchSize, headCount, tokenCount, headDim], seed: seed)
    }

    /// Makes a temporary file URL for one prompt cache file.
    ///
    /// - Returns: The URL, in the temporary directory.
    static func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
    }

    /// Decodes the DeepSeek-V4 configuration that the test bundle holds.
    ///
    /// - Returns: The configuration.
    fileprivate static func deepSeekConfiguration() throws -> DeepSeekV4Configuration {
        let url = try #require(
            Bundle.module.url(
                forResource: "DeepSeek-V4-Flash-4bit-config", withExtension: "json"))
        return try JSONDecoder().decode(
            DeepSeekV4Configuration.self, from: Data(contentsOf: url))
    }

    /// Makes the three state slots of one DeepSeek-V4 branch.
    ///
    /// - Parameter seed: The seed of the random key.
    /// - Returns: The pooled chunks, the carry rows and the carry start.
    fileprivate static func deepSeekBranchSlots(seed: UInt64) -> [MLXArray] {
        [
            tensor([batchSize, deepSeekChunkCount, deepSeekRowWidth], seed: seed),
            tensor([batchSize, deepSeekCarryRowCount, deepSeekRowWidth], seed: seed + 1),
            MLXArray([Int32(0)]),
        ]
    }

    /// Tells whether an array holds evaluated data.
    ///
    /// - Parameter array: The array to read.
    /// - Returns: True when the data of the array is available.
    fileprivate static func isAvailable(_ array: MLXArray) -> Bool {
        var available = false
        _mlx_array_is_available(&available, array.ctx)
        return available
    }

    // MARK: - Comparisons

    /// Records an issue unless two lists of arrays have equal shapes, types and values.
    ///
    /// - Parameters:
    ///   - actual: The arrays to check.
    ///   - expected: The arrays they must equal.
    ///   - label: The name of the comparison, for the issue text.
    fileprivate static func expectEqual(
        _ actual: [MLXArray], _ expected: [MLXArray], _ label: String
    ) {
        #expect(actual.count == expected.count, "\(label): array count")
        for (index, (lhs, rhs)) in zip(actual, expected).enumerated() {
            #expect(lhs.shape == rhs.shape, "\(label): shape of array \(index)")
            #expect(lhs.dtype == rhs.dtype, "\(label): type of array \(index)")
            #expect(
                arrayEqual(lhs, rhs, equalNAN: true).item(Bool.self),
                "\(label): values of array \(index)")
        }
    }

    /// Records an issue unless a restored cache has the `state` and the `metaState` of the
    /// cache that was saved.
    ///
    /// - Parameters:
    ///   - restored: The restored cache.
    ///   - source: The cache that was saved.
    ///   - label: The name of the comparison, for the issue text.
    static func expectSameContents(
        _ restored: any KVCache, _ source: any KVCache, _ label: String
    ) {
        expectEqual(restored.state, source.state, "\(label) state")
        #expect(restored.metaState == source.metaState, "\(label) metaState")
    }

    /// Records an issue unless one more step gives the same output and the same state on the
    /// restored cache and on the cache that was saved.
    ///
    /// - Parameters:
    ///   - kind: The kind of the two caches.
    ///   - restored: The restored cache.
    ///   - source: The cache that was saved.
    fileprivate static func expectSameStep(
        _ kind: CacheKind, _ restored: any KVCache, _ source: any KVCache
    ) throws {
        let restoredOutput = try kind.step(restored)
        let sourceOutput = try kind.step(source)
        expectEqual(restoredOutput, sourceOutput, "\(kind) step output")
        expectSameContents(restored, source, "\(kind) after the step")
    }

    /// Tells whether two caches are the same instance.
    ///
    /// - Parameters:
    ///   - lhs: The first cache.
    ///   - rhs: The second cache.
    /// - Returns: True when both are one object.
    static func isSameInstance(_ lhs: any KVCache, _ rhs: any KVCache) -> Bool {
        (lhs as AnyObject) === (rhs as AnyObject)
    }

    // MARK: - Tampered files

    /// Saves a cache, then writes a second file whose arrays or metadata a closure changed.
    ///
    /// - Parameters:
    ///   - caches: The caches to save.
    ///   - state: The model state to save with the caches, or `nil` for no model state.
    ///   - edit: Changes the arrays and the metadata of the saved file.
    /// - Returns: The URL of the tampered file. The caller removes it.
    fileprivate static func tamperedFile(
        _ caches: [any KVCache],
        state: LMOutput.State? = nil,
        edit: (inout [String: MLXArray], inout [String: String]) -> Void
    ) throws -> URL {
        let savedURL = temporaryURL()
        defer { try? FileManager.default.removeItem(at: savedURL) }
        try savePromptCache(url: savedURL, cache: caches, state: state)
        var (arrays, metadata) = try loadArraysAndMetadata(url: savedURL)
        edit(&arrays, &metadata)
        let tamperedURL = temporaryURL()
        try save(arrays: arrays, metadata: metadata, url: tamperedURL)
        return tamperedURL
    }

    // MARK: - Tampered CacheList files

    /// The flattened file key of one meta-state value of the first layer.
    ///
    /// - Parameter index: The place of the value in the meta state of the layer.
    /// - Returns: The key, in the cache-information part.
    fileprivate static func firstLayerMetaStateKey(index: Int) -> String {
        "0.0.\(index)"
    }

    /// Makes a filled ``CacheKind/cacheList``: a `KVCacheSimple` and a `MambaCache` that hold
    /// data.
    ///
    /// - Returns: The filled list.
    fileprivate static func makeFilledCacheList() throws -> CacheList {
        try #require(try CacheKind.cacheList.makeFilled() as? CacheList)
    }

    /// Makes the template of ``CacheKind/cacheList``: an empty `KVCacheSimple` and an empty
    /// `MambaCache`.
    ///
    /// - Returns: The template list.
    fileprivate static func makeCacheListTemplate() throws -> CacheList {
        try #require(try CacheKind.cacheList.makeTemplate() as? CacheList)
    }

    /// Finds the header of one child in the meta state of a saved `CacheList`.
    ///
    /// - Parameters:
    ///   - list: The list that the file saves.
    ///   - child: The place of the child in the list.
    /// - Returns: The place of the class name of the child in the meta state of the list.
    fileprivate static func cacheListChildHeader(_ list: CacheList, child: Int) -> Int {
        list.children.prefix(child).reduce(cacheListFirstChildHeader) { place, earlier in
            place + cacheListChildHeaderCount + earlier.metaState.count
        }
    }

    /// Restores a file into a template list, and records an issue unless the restore throws
    /// ``KVCacheError`` with the expected message and the template list still holds its own
    /// empty children.
    ///
    /// The message tells which check threw. Another check can also reject a damaged file, thus
    /// the error type alone does not prove that the restore reached the expected check.
    ///
    /// - Parameters:
    ///   - url: The URL of the file.
    ///   - template: The template list.
    ///   - message: The message of the check that must reject the file.
    fileprivate static func expectRejected(
        _ url: URL, into template: CacheList, message: String
    ) {
        let templateChildren = template.children
        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [template])
        }
        #expect(error?.message == message, "the check that rejects the file")
        #expect(template.children.count == templateChildren.count, "the child count")
        for (index, (child, templateChild)) in zip(template.children, templateChildren)
            .enumerated()
        {
            #expect(isSameInstance(child, templateChild), "child \(index) is the template child")
            #expect(child.state.isEmpty, "child \(index) holds no saved array")
        }
    }

    /// The message of the check that finds a saved configuration value that is not the value of
    /// the template.
    ///
    /// - Parameter className: The saved class name of the layer.
    /// - Returns: The message.
    fileprivate static func fixedConfigurationMessage(className: String) -> String {
        "The saved \(className) configuration is not the configuration of the model cache."
    }

    /// The message of the check that finds a template that no template restore can write.
    ///
    /// - Parameter template: The template.
    /// - Returns: The message.
    fileprivate static func unrestorableTemplateMessage(_ template: any KVCache) -> String {
        "\(type(of: template)) cannot receive a restore into a template."
    }

    /// Makes a filled ``CustomSlotCache`` or ``RegisteredSlotCache``.
    ///
    /// - Parameter cache: The empty cache to fill.
    /// - Returns: `cache`, which holds ``customSlotCount`` arrays.
    fileprivate static func filled<Cache: CustomSlotCache>(_ cache: Cache) -> Cache {
        cache.slots = (0 ..< customSlotCount).map {
            block(tokenCount: promptTokenCount, seed: firstSeed + UInt64($0))
        }
        return cache
    }

    /// Saves a filled `KVCacheSimple`, then writes a second file whose class name no class uses.
    ///
    /// - Returns: The URL of the tampered file. The caller removes it.
    fileprivate static func fileWithUnknownClassName() throws -> URL {
        try tamperedFile([try CacheKind.simple.makeFilled()]) { _, metadata in
            metadata[firstLayerClassNameKey] = unknownClassName
        }
    }

    /// Saves caches and reads back the class name that the file writes for the first layer.
    ///
    /// - Parameters:
    ///   - caches: The caches to save.
    ///   - url: The URL of the file.
    /// - Returns: The saved class name of the first layer.
    fileprivate static func savedFirstClassName(_ caches: [any KVCache], url: URL) throws -> String?
    {
        try savePromptCache(url: url, cache: caches)
        let (_, metadata) = try loadArraysAndMetadata(url: url)
        return metadata[firstLayerClassNameKey]
    }

    // MARK: - Damaged file layouts

    /// Loads a file with no template, and records an issue unless the load throws
    /// ``KVCacheError`` with the expected message.
    ///
    /// The message tells which check threw. Another check can also reject a damaged file, thus
    /// the error type alone does not prove that the load reached the expected check.
    ///
    /// - Parameters:
    ///   - url: The URL of the file.
    ///   - message: The message of the check that must reject the file.
    fileprivate static func expectLoadRejected(_ url: URL, message: String) {
        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
        #expect(error?.message == message, "the check that rejects the file")
    }

    /// Saves a filled `KVCacheSimple` with a model state of one array, then writes a second
    /// file whose metadata a closure changed.
    ///
    /// - Parameter edit: Changes the metadata of the saved file.
    /// - Returns: The URL of the tampered file. The caller removes it.
    fileprivate static func tamperedModelStateFile(
        edit: (inout [String: String]) -> Void
    ) throws -> URL {
        var state = LMOutput.State()
        state[positionsStateKey] = tensor([batchSize, headCount], seed: firstSeed)
        return try tamperedFile([try CacheKind.simple.makeFilled()], state: state) {
            _, metadata in
            edit(&metadata)
        }
    }

    /// Saves one filled cache of one kind, then writes a second file whose meta state of the
    /// first layer has other values at some places.
    ///
    /// - Parameters:
    ///   - kind: The kind of the cache.
    ///   - values: The new meta-state values, by their place in the meta state.
    /// - Returns: The URL of the tampered file. The caller removes it.
    fileprivate static func fileWithFirstLayerMetaState(
        _ kind: CacheKind, values: [Int: String]
    ) throws -> URL {
        try tamperedFile([try kind.makeFilled()]) { _, metadata in
            for (index, value) in values {
                metadata[firstLayerMetaStateKey(index: index)] = value
            }
        }
    }

    // MARK: - Round trip for each cache type

    @Test(
        "A saved cache restores into a fresh template with equal state and metaState",
        arguments: CacheKind.allCases)
    func roundTripRestoresIntoTemplate(kind: CacheKind) throws {
        let source = try kind.makeFilled()
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source])

        let template = try kind.makeTemplate()
        let snapshot = try loadPromptCacheSnapshot(url: url, into: [template])

        let restored = try #require(snapshot.cache.first)
        #expect(snapshot.cache.count == 1)
        #expect(Self.isSameInstance(restored, template), "\(kind): restores into the template")
        Self.expectSameContents(restored, source, "\(kind)")
        try Self.expectSameStep(kind, restored, source)
    }

    // MARK: - Converted layers

    @Test(
        "A converted layer restores into a KVCacheSimple template",
        arguments: [CacheKind.quantized, .turboQuant])
    func convertedLayerRestoresIntoSimpleTemplate(kind: CacheKind) throws {
        let source = try kind.makeFilled()
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source])

        let snapshot = try loadPromptCacheSnapshot(url: url, into: [KVCacheSimple()])

        let restored = try #require(snapshot.cache.first)
        #expect(type(of: restored) == type(of: source), "\(kind): keeps the saved class")
        Self.expectSameContents(restored, source, "\(kind)")
        try Self.expectSameStep(kind, restored, source)
    }

    @Test("A converted child of a CacheList restores into the template list")
    func convertedChildRestoresIntoTemplateList() throws {
        let quantized = try CacheKind.quantized.makeFilled()
        let mamba = try CacheKind.mamba.makeFilled()
        let source = CacheList(quantized, mamba)
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source])

        let template = CacheList(KVCacheSimple(), MambaCache())
        let snapshot = try loadPromptCacheSnapshot(url: url, into: [template])

        let restored = try #require(snapshot.cache.first as? CacheList)
        #expect(restored === template)
        #expect(restored[0] is QuantizedKVCache)
        Self.expectSameContents(restored, source, "CacheList")
        try Self.expectSameStep(.quantized, restored[0], quantized)
        try Self.expectSameStep(.mamba, restored[1], mamba)
    }

    // MARK: - Model state and file life

    @Test("The model state and the caller metadata come back equal")
    func modelStateComesBackEqual() throws {
        let source = try CacheKind.simple.makeFilled()
        let key = LMOutput.Key<MLXArray>("test.ropeDeltas")
        let ropeDeltas = Self.tensor([Self.batchSize, Self.headCount], seed: Self.firstSeed)
        var state = LMOutput.State()
        state[key] = ropeDeltas
        let metadata = ["source": "template-restore"]
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source], metadata: metadata, state: state)

        let snapshot = try loadPromptCacheSnapshot(url: url, into: [KVCacheSimple()])

        let restoredDeltas = try #require(snapshot.state?[key])
        Self.expectEqual([restoredDeltas], [ropeDeltas], "model state")
        #expect(snapshot.metadata == metadata)
    }

    @Test("The file can be deleted after the load, and the restored caches still work")
    func fileCanBeDeletedAfterLoad() throws {
        let kinds: [CacheKind] = [.simple, .deepSeekWithIndexer, .miniMaxWithIndex]
        let sources = try kinds.map { try $0.makeFilled() }
        let key = Self.positionsStateKey
        var state = LMOutput.State()
        state[key] = Self.tensor([Self.batchSize, Self.headCount], seed: Self.firstSeed)
        let url = Self.temporaryURL()
        try savePromptCache(url: url, cache: sources, state: state)

        let templates = try kinds.map { try $0.makeTemplate() }
        let snapshot = try loadPromptCacheSnapshot(url: url, into: templates)
        try FileManager.default.removeItem(at: url)

        let restoredDeltas = try #require(snapshot.state?[key])
        #expect(Self.isAvailable(restoredDeltas), "the model state is evaluated")
        for (kind, (restored, source)) in zip(kinds, zip(snapshot.cache, sources)) {
            for array in restored.innerState() {
                #expect(Self.isAvailable(array), "\(kind): the restored arrays are evaluated")
            }
            Self.expectSameContents(restored, source, "\(kind)")
            try Self.expectSameStep(kind, restored, source)
        }
    }

    // MARK: - Errors

    @Test(
        "A class mismatch throws KVCacheError",
        arguments: [
            (CacheKind.rotatingBeforeWrap, CacheKind.simple),
            (.simple, .rotatingBeforeWrap),
            (.quantized, .rotatingBeforeWrap),
            (.mamba, .arrays),
            (.chunked, .simple),
            (.deepSeekWithIndexer, .rotatingBeforeWrap),
            (.miniMaxWithIndex, .simple),
            (.simple, .miniMaxWithIndex),
        ])
    func classMismatchThrows(saved: CacheKind, template: CacheKind) throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [try saved.makeFilled()])
        let templates = [try template.makeTemplate()]

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }
    }

    @Test(
        "A ring does not restore into a ring of another configuration",
        arguments: RingConfigurationMismatch.allCases)
    func ringConfigurationMismatchThrows(mismatch: RingConfigurationMismatch) throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [try CacheKind.rotatingAfterWrap.makeFilled()])
        let template = mismatch.makeTemplate()
        let templateMetaState = template.metaState

        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [template])
        }
        #expect(error?.message == Self.fixedConfigurationMessage(className: "RotatingKVCache"))
        #expect(template.metaState == templateMetaState)
        #expect(template.state.isEmpty)
    }

    @Test(
        "A quantized cache does not restore into a quantized cache of another configuration",
        arguments: QuantizedConfigurationMismatch.allCases)
    func quantizedConfigurationMismatchThrows(mismatch: QuantizedConfigurationMismatch) throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [try CacheKind.quantized.makeFilled()])
        let template = mismatch.makeTemplate()
        let templateMetaState = template.metaState

        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [template])
        }
        #expect(error?.message == Self.fixedConfigurationMessage(className: "QuantizedKVCache"))
        #expect(template.metaState == templateMetaState)
        #expect(template.state.isEmpty)
    }

    @Test("A DeepSeek-V4 layer with indexer data does not restore into a layer with no indexer")
    func deepSeekIndexerMismatchThrows() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [try CacheKind.deepSeekWithIndexer.makeFilled()])
        let templates = [try CacheKind.deepSeekWithoutIndexer.makeTemplate()]

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }
    }

    @Test("A layer-count mismatch throws KVCacheError")
    func layerCountMismatchThrows() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let sources = [try CacheKind.simple.makeFilled(), try CacheKind.simple.makeFilled()]
        try savePromptCache(url: url, cache: sources)

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [KVCacheSimple()])
        }
        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(
                url: url, into: [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()])
        }
    }

    @Test("A bad later layer throws before any setter writes an earlier template")
    func badLaterLayerLeavesEarlierTemplatesEmpty() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let sources = [
            try CacheKind.simple.makeFilled(), try CacheKind.rotatingBeforeWrap.makeFilled(),
        ]
        try savePromptCache(url: url, cache: sources)
        let first = KVCacheSimple()

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [first, KVCacheSimple()])
        }
        #expect(first.state.isEmpty)
        #expect(first.offset == 0)
    }

    @Test("A wrong state-array count throws KVCacheError", arguments: CacheKind.allCases)
    func wrongStateCountThrows(kind: CacheKind) throws {
        let source = try kind.makeFilled()
        let extraKey = "0.\(source.state.count)"
        let url = try Self.tamperedFile([source]) { arrays, _ in
            arrays[extraKey] = Self.block(tokenCount: 1, seed: Self.firstSeed)
        }
        defer { try? FileManager.default.removeItem(at: url) }
        let templates = [try kind.makeTemplate()]

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }
    }

    @Test("A wrong metaState count throws KVCacheError", arguments: CacheKind.allCases)
    func wrongMetaStateCountThrows(kind: CacheKind) throws {
        let source = try kind.makeFilled()
        let savedCount = source.metaState.count
        let url = try Self.tamperedFile([source]) { _, metadata in
            for index in savedCount ..< savedCount + Self.extraMetaStateCount {
                metadata[Self.firstLayerMetaStateKey(index: index)] = "0"
            }
        }
        defer { try? FileManager.default.removeItem(at: url) }
        let templates = [try kind.makeTemplate()]

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }
    }

    // MARK: - CacheList rejections

    @Test(
        "A saved CacheList with no valid child count throws KVCacheError",
        arguments: [nil, "-1", "not-a-count"] as [String?])
    func badCacheListChildCountThrows(childCount: String?) throws {
        let url = try Self.tamperedFile([try Self.makeFilledCacheList()]) { _, metadata in
            metadata[Self.firstLayerMetaStateKey(index: 0)] = childCount
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectRejected(
            url, into: try Self.makeCacheListTemplate(), message: Self.missingChildCountMessage)
    }

    @Test("A saved CacheList cut inside the header of its second child throws KVCacheError")
    func truncatedCacheListChildHeaderThrows() throws {
        let source = try Self.makeFilledCacheList()
        let savedCount = source.metaState.count
        let secondHeader = Self.cacheListChildHeader(source, child: Self.cacheListSecondChild)
        let url = try Self.tamperedFile([source]) { _, metadata in
            for index in secondHeader + 1 ..< savedCount {
                metadata[Self.firstLayerMetaStateKey(index: index)] = nil
            }
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectRejected(
            url, into: try Self.makeCacheListTemplate(), message: Self.truncatedHeaderMessage)
    }

    @Test(
        "A saved CacheList child count past the saved values throws KVCacheError",
        arguments: [cacheListStateCountOffset, cacheListMetaStateCountOffset])
    func overrunCacheListChildCountThrows(countOffset: Int) throws {
        let source = try Self.makeFilledCacheList()
        let savedCount = source.metaState.count
        let countPlace =
            Self.cacheListChildHeader(source, child: Self.cacheListSecondChild) + countOffset
        let url = try Self.tamperedFile([source]) { _, metadata in
            metadata[Self.firstLayerMetaStateKey(index: countPlace)] = String(savedCount)
        }
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(savedCount > source.state.count, "the count is past the saved arrays also")
        Self.expectRejected(
            url, into: try Self.makeCacheListTemplate(), message: Self.invalidChildCountMessage)
    }

    @Test(
        "A saved CacheList does not restore into a template list of another child count",
        arguments: [[CacheKind.simple], [.simple, .mamba, .simple]])
    func cacheListChildCountMismatchThrows(templateKinds: [CacheKind]) throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try Self.makeFilledCacheList()
        try savePromptCache(url: url, cache: [source])
        let template = CacheList(caches: try templateKinds.map { try $0.makeTemplate() })

        Self.expectRejected(
            url, into: template,
            message:
                "The prompt cache holds a CacheList of \(source.children.count) children and the model gave \(templateKinds.count)."
        )
    }

    // MARK: - Damaged file layout rejections

    @Test(
        "A file whose cache-information count is not its class-name count throws KVCacheError",
        arguments: [
            [classNamePart, "\(userMetadataPart)\(offsetRecordPrefix)"],
            [secondLayerMetaStatePrefix],
        ])
    func cacheCountMismatchThrows(removedKeyPrefixes: [String]) throws {
        // Without its class names the file has a layer count of 0, and the offset records
        // of the two layers then fail their own check first. Thus that case removes the
        // offset records also.
        let sources = [try CacheKind.simple.makeFilled(), try CacheKind.simple.makeFilled()]
        let url = try Self.tamperedFile(sources) { _, metadata in
            metadata = metadata.filter { entry in
                !removedKeyPrefixes.contains(where: { entry.key.hasPrefix($0) })
            }
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(url, message: Self.cacheCountMismatchMessage)
    }

    @Test(
        "A file with no valid model-state count throws KVCacheError",
        arguments: [nil, "0", "not-a-count"] as [String?])
    func invalidModelStateCountThrows(count: String?) throws {
        let url = try Self.tamperedModelStateFile { metadata in
            metadata[Self.modelStateCountKey] = count
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(url, message: Self.invalidModelStateCountMessage)
    }

    @Test("A file with a reserved model-state key that the reader does not know throws")
    func unknownModelStateKeyThrows() throws {
        let url = try Self.tamperedModelStateFile { metadata in
            metadata[Self.unknownModelStateKey] = "1"
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(url, message: Self.unexpectedModelStateMessage)
    }

    @Test(
        "A file with an array key that is not a layer and an array index throws KVCacheError",
        arguments: ["x.0", "9.0"])
    func invalidArrayKeyThrows(key: String) throws {
        let url = try Self.tamperedFile([try CacheKind.simple.makeFilled()]) { arrays, _ in
            arrays[key] = arrays.removeValue(forKey: Self.firstLayerFirstArrayKey)
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(url, message: "Corrupt prompt cache: invalid array key '\(key)'.")
    }

    @Test("A file whose first layer has its second array and not its first throws KVCacheError")
    func nonContiguousArrayIndicesThrow() throws {
        let url = try Self.tamperedFile([try CacheKind.simple.makeFilled()]) { arrays, _ in
            arrays[Self.firstLayerFirstArrayKey] = nil
        }
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(
            url, message: "Corrupt prompt cache: cache 0 has non-contiguous array indices.")
    }

    @Test("A saved RotatingKVCache whose wrapped flag is not true or false throws KVCacheError")
    func invalidRotatingWrappedFlagThrows() throws {
        let url = try Self.fileWithFirstLayerMetaState(
            .rotatingAfterWrap, values: [Self.rotatingWrappedIndex: Self.invalidWrappedFlag])
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(
            url,
            message:
                "Corrupt prompt cache: invalid RotatingKVCache wrapped flag '\(Self.invalidWrappedFlag)'."
        )
    }

    @Test("A saved VarianceNormalizedKVCache with tile arrays and a tile count of 0 throws")
    func varianceNormalizedTileArraysWithoutTilesThrow() throws {
        // The offset becomes the tail length, thus the offset agrees with a tile count of 0,
        // and only the tile arrays do not fit.
        let tailLength = Self.varianceTokenCount % Self.varianceTileSize
        let url = try Self.fileWithFirstLayerMetaState(
            .varianceNormalized,
            values: [
                Self.varianceTileCountIndex: "0", Self.varianceOffsetIndex: String(tailLength),
            ])
        defer { try? FileManager.default.removeItem(at: url) }

        Self.expectLoadRejected(
            url, message: "Corrupt prompt cache: invalid VarianceNormalizedKVCache state.")
    }

    @Test(
        "A saved MambaCache with the legacy meta state restores its arrays into the slots in order")
    func legacyArraysMetaStateRestoresSlotsInOrder() throws {
        let source = try #require(try CacheKind.mamba.makeFilled() as? MambaCache)
        let savedCount = source.metaState.count
        let url = try Self.tamperedFile([source]) { _, metadata in
            for index in 0 ..< savedCount {
                metadata[Self.firstLayerMetaStateKey(index: index)] = nil
            }
            for (index, value) in Self.legacyArraysMetaState.enumerated() {
                metadata[Self.firstLayerMetaStateKey(index: index)] = value
            }
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let restored = try #require(
            try loadPromptCacheSnapshot(url: url).cache.first as? MambaCache)
        let slots = (0 ..< restored.slotCount).map { restored[$0] }

        #expect(restored.slotCount == source.slotCount, "the slot count")
        #expect(slots.allSatisfy { $0 != nil }, "each slot holds an array")
        Self.expectEqual(slots.compactMap { $0 }, source.state, "the legacy slots")
    }

    // MARK: - Cache classes outside the library

    @Test("A registered cache class saves under its registered name and loads back")
    func registeredClassRoundTrips() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let source = Self.filled(RegisteredSlotCache())

        let className = try Self.savedFirstClassName([source], url: url)
        let snapshot = try loadPromptCacheSnapshot(url: url)

        #expect(className == RegisteredSlotCache.className)
        #expect(snapshot.cache.count == 1)
        let restored = try #require(snapshot.cache.first as? RegisteredSlotCache)
        Self.expectSameContents(restored, source, "registered cache")
    }

    @Test("An unregistered cache class saves as KVCache, and its load throws KVCacheError")
    func unregisteredClassSavesAsSimpleCache() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let className = try Self.savedFirstClassName([Self.filled(CustomSlotCache())], url: url)
        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }

        #expect(className == "KVCache")
        #expect(
            error?.message == "Corrupt prompt cache: invalid KVCacheSimple state or metadata shape."
        )
    }

    @Test("The registry check of a built-in part throws for a cache that is not a built-in leaf")
    func registryValidateRejectsCacheWithoutCheck() throws {
        let saved = try CacheKind.simple.makeFilled()
        let caches: [any KVCache] = [
            CacheList(KVCacheSimple()), CustomSlotCache(), RegisteredSlotCache(),
        ]

        for cache in caches {
            let error = #expect(throws: KVCacheError.self) {
                try KVCacheSerializationRegistry.validate(
                    state: saved.state, metaState: saved.metaState, for: cache)
            }
            #expect(
                error?.message
                    == "\(type(of: cache)) is not a built-in cache class that has a check."
            )
            #expect(cache.state.isEmpty, "\(type(of: cache)) holds no saved array")
        }
    }

    @Test("A template of a class outside the library that is not PromptCacheRestorable throws")
    func unrestorableTemplateThrows() throws {
        let cases: [(saved: any KVCache, template: CustomSlotCache)] = [
            (try CacheKind.simple.makeFilled(), CustomSlotCache()),
            (Self.filled(RegisteredSlotCache()), RegisteredSlotCache()),
        ]

        for (saved, template) in cases {
            let url = Self.temporaryURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: [saved])

            let error = #expect(throws: KVCacheError.self) {
                try loadPromptCacheSnapshot(url: url, into: [template])
            }
            #expect(error?.message == Self.unrestorableTemplateMessage(template))
            #expect(template.slots.isEmpty, "\(type(of: template)) holds no saved array")
        }
    }

    @Test("A file with an unknown class name throws KVCacheError on the load")
    func unknownClassNameThrowsOnLoad() throws {
        let url = try Self.fileWithUnknownClassName()
        defer { try? FileManager.default.removeItem(at: url) }

        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
        #expect(error?.message == "Unknown cache class: \(Self.unknownClassName)")
    }

    @Test("A file with an unknown class name does not restore into a KVCacheSimple template")
    func unknownClassNameThrowsOnTemplateRestore() throws {
        let url = try Self.fileWithUnknownClassName()
        defer { try? FileManager.default.removeItem(at: url) }
        let template = KVCacheSimple()

        let error = #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [template])
        }
        #expect(
            error?.message
                == "The prompt cache holds a \(Self.unknownClassName) layer and the model gave a KVCache cache."
        )
        #expect(template.state.isEmpty)
        #expect(template.offset == 0)
    }

    // MARK: - Offset record

    @Test(
        "A recurrent or front-trimmed cache comes back with its offset through each load function",
        arguments: OffsetCacheKind.allCases)
    func offsetComesBackThroughEachLoad(kind: OffsetCacheKind) throws {
        let source = try kind.makeFilled()
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source])

        let restoredCaches: [(String, any KVCache)] = [
            (
                "loadPromptCacheSnapshot(url:)",
                try #require(try loadPromptCacheSnapshot(url: url).cache.first)
            ),
            ("loadPromptCache(url:)", try #require(try loadPromptCache(url: url).0.first)),
            (
                "loadPromptCacheSnapshot(url:into:)",
                try #require(
                    try loadPromptCacheSnapshot(url: url, into: [kind.makeTemplate()]).cache.first)
            ),
        ]

        #expect(source.offset == kind.savedOffset, "\(kind): the fixture offset")
        for (load, restored) in restoredCaches {
            #expect(restored.offset == source.offset, "\(kind) through \(load): offset")
            Self.expectSameContents(restored, source, "\(kind) through \(load)")
            let fresh = try kind.makeFilled()
            try Self.expectSameStep(kind.stepKind, restored, fresh)
            #expect(restored.offset == fresh.offset, "\(kind) through \(load): offset after a step")
        }
    }

    @Test(
        "A record that disagrees with the restored offset throws KVCacheError on a template restore",
        arguments: checkedOffsetKinds)
    func disagreeingRecordThrowsOnTemplateRestore(kind: CacheKind) throws {
        let url = try Self.fileWithDisagreeingRecord(kind)
        defer { try? FileManager.default.removeItem(at: url) }
        let templates = [try kind.makeTemplate()]

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }
    }

    @Test(
        "A record that disagrees with the restored offset throws KVCacheError on a load",
        arguments: checkedBuiltInKinds)
    func disagreeingRecordThrowsOnLoad(kind: CacheKind) throws {
        let url = try Self.fileWithDisagreeingRecord(kind)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
        #expect(throws: KVCacheError.self) {
            try loadPromptCache(url: url)
        }
    }

    @Test("The offset record is in the file and not in the user metadata of any load function")
    func offsetRecordStaysOutOfUserMetadata() throws {
        let source = try OffsetCacheKind.mamba.makeFilled()
        let metadata = ["source": "offset-record"]
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [source], metadata: metadata)

        let (_, storedMetadata) = try loadArraysAndMetadata(url: url)
        #expect(storedMetadata[Self.offsetRecordKey(layer: 0)] == String(source.offset))
        #expect(try loadPromptCacheSnapshot(url: url).metadata == metadata)
        #expect(try loadPromptCache(url: url).1 == metadata)
        #expect(try loadPromptCacheSnapshot(url: url, into: [MambaCache()]).metadata == metadata)
    }

    @Test("savePromptCache refuses user metadata that uses the reserved offset prefix")
    func reservedOffsetPrefixThrows() throws {
        let source = try CacheKind.simple.makeFilled()
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            try savePromptCache(
                url: url, cache: [source], metadata: ["\(Self.offsetRecordPrefix)0": "0"])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("A file with no offset record loads as before", arguments: OffsetCacheKind.allCases)
    func fileWithoutRecordLoadsAsBefore(kind: OffsetCacheKind) throws {
        let source = try kind.makeFilled()
        let url = try Self.tamperedFile([source]) { _, metadata in
            metadata = metadata.filter {
                !$0.key.hasPrefix("\(Self.userMetadataPart)\(Self.offsetRecordPrefix)")
            }
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let snapshot = try loadPromptCacheSnapshot(url: url)
        let loaded = try #require(snapshot.cache.first)
        let intoSnapshot = try loadPromptCacheSnapshot(url: url, into: [kind.makeTemplate()])
        let restored = try #require(intoSnapshot.cache.first)

        #expect(snapshot.metadata.isEmpty)
        #expect(intoSnapshot.metadata.isEmpty)
        for (load, cache) in [("load", loaded), ("template restore", restored)] {
            #expect(cache.offset == kind.offsetWithoutRecord, "\(kind) \(load): offset")
            Self.expectSameContents(cache, source, "\(kind) \(load)")
        }
    }

    @Test(
        "A malformed offset record throws KVCacheError",
        arguments: [(0, "not-an-offset"), (0, "-1"), (1, "0")])
    func malformedRecordThrows(layer: Int, value: String) throws {
        let source = try CacheKind.simple.makeFilled()
        let url = try Self.tamperedFile([source]) { _, metadata in
            metadata[Self.offsetRecordKey(layer: layer)] = value
        }
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url)
        }
        #expect(throws: KVCacheError.self) {
            try loadPromptCacheSnapshot(url: url, into: [KVCacheSimple()])
        }
    }

    /// Saves a filled cache of one kind, and writes a second file whose offset record is one
    /// more than the saved offset.
    ///
    /// - Parameter kind: The kind of the cache.
    /// - Returns: The URL of the second file. The caller removes it.
    fileprivate static func fileWithDisagreeingRecord(_ kind: CacheKind) throws -> URL {
        let source = try kind.makeFilled()
        let key = offsetRecordKey(layer: 0)
        return try tamperedFile([source]) { _, metadata in
            metadata[key] = String(source.offset + 1)
        }
    }
}

// MARK: - Cache kinds

/// One kind of cache that a round trip covers: how to fill one, how to make the fresh cache a
/// model makes, and how to take one more step.
enum CacheKind: String, CaseIterable, CustomTestStringConvertible, Sendable {
    case simple
    case rotatingBeforeWrap
    case rotatingAfterWrap
    case quantized
    case chunked
    case mamba
    case arrays
    case turboQuant
    case varianceNormalized
    case cacheList
    case deepSeekWithIndexer
    case deepSeekWithoutIndexer
    case miniMaxWithIndex
    case miniMaxWithoutIndex

    /// The name of the kind in the test report.
    var testDescription: String { rawValue }

    /// The fixture sizes and builders.
    private typealias Fixture = PromptCacheTemplateRestoreTests

    /// Makes the fresh cache that a model makes for this kind.
    ///
    /// - Returns: An empty cache.
    func makeTemplate() throws -> any KVCache {
        switch self {
        case .simple:
            return KVCacheSimple()
        case .rotatingBeforeWrap, .rotatingAfterWrap:
            return RotatingKVCache(maxSize: Fixture.rotatingWindow)
        case .quantized:
            return QuantizedKVCache(
                groupSize: Fixture.quantizedGroupSize, bits: Fixture.quantizedBits)
        case .chunked:
            return ChunkedKVCache(chunkSize: Fixture.chunkSize)
        case .mamba:
            return MambaCache()
        case .arrays:
            return ArraysCache(size: Fixture.arraysSlotCount)
        case .turboQuant:
            return TurboQuantKVCache(bits: Fixture.turboQuantBits)
        case .varianceNormalized:
            return VarianceNormalizedKVCache(
                tileSize: Fixture.varianceTileSize, keyBits: Fixture.varianceBits,
                valueBits: Fixture.varianceBits,
                sinkhornIterations: Fixture.varianceSinkhornIterations)
        case .cacheList:
            return CacheList(KVCacheSimple(), MambaCache())
        case .deepSeekWithIndexer:
            return DeepSeekV4Cache(
                configuration: try Fixture.deepSeekConfiguration(),
                layer: Fixture.deepSeekIndexerLayer)
        case .deepSeekWithoutIndexer:
            return DeepSeekV4Cache(
                configuration: try Fixture.deepSeekConfiguration(),
                layer: Fixture.deepSeekPlainLayer)
        case .miniMaxWithIndex, .miniMaxWithoutIndex:
            return MiniMaxM3KVCache()
        }
    }

    /// Makes a cache of this kind that holds data.
    ///
    /// - Returns: The filled cache.
    func makeFilled() throws -> any KVCache {
        let cache = try makeTemplate()
        let seed = Fixture.firstSeed
        switch self {
        case .simple, .chunked, .turboQuant:
            _ = cache.update(keys: prompt(seed: seed), values: prompt(seed: seed + 1))
        case .rotatingBeforeWrap:
            try fillRing(cache, tokenCount: Fixture.rotatingTokensBeforeWrap)
        case .rotatingAfterWrap:
            try fillRing(cache, tokenCount: Fixture.rotatingTokensAfterWrap)
        case .quantized:
            let quantized = try #require(cache as? QuantizedKVCache)
            _ = quantized.updateQuantized(keys: prompt(seed: seed), values: prompt(seed: seed + 1))
        case .mamba, .arrays:
            try fillSlots(cache)
        case .varianceNormalized:
            let tokenCount = Fixture.varianceTokenCount
            _ = cache.update(
                keys: Fixture.block(tokenCount: tokenCount, seed: seed),
                values: Fixture.block(tokenCount: tokenCount, seed: seed + 1))
        case .cacheList:
            let list = try #require(cache as? CacheList)
            _ = list[0].update(keys: prompt(seed: seed), values: prompt(seed: seed + 1))
            try fillSlots(list[1])
        case .deepSeekWithIndexer, .deepSeekWithoutIndexer:
            try fillDeepSeek(cache)
        case .miniMaxWithIndex, .miniMaxWithoutIndex:
            try fillMiniMax(cache)
        }
        return cache
    }

    /// Takes one more step on a cache of this kind.
    ///
    /// - Parameter cache: The cache to advance.
    /// - Returns: The arrays that the step gives back.
    func step(_ cache: any KVCache) throws -> [MLXArray] {
        let seed = Fixture.firstSeed
        let keys = Fixture.block(tokenCount: 1, seed: seed)
        let values = Fixture.block(tokenCount: 1, seed: seed + 1)
        switch self {
        case .quantized:
            let quantized = try #require(cache as? QuantizedKVCache)
            let (keyParts, valueParts) = quantized.updateQuantized(keys: keys, values: values)
            return [keyParts.0, keyParts.1, valueParts.0, valueParts.1]
                + [keyParts.2, valueParts.2].compactMap { $0 }
        case .mamba, .arrays:
            return try stepSlots(cache)
        case .cacheList:
            let list = try #require(cache as? CacheList)
            let (stepKeys, stepValues) = list[0].update(keys: keys, values: values)
            return [stepKeys, stepValues] + (try stepSlots(list[1]))
        case .miniMaxWithIndex, .miniMaxWithoutIndex:
            let miniMax = try #require(cache as? MiniMaxM3KVCache)
            let (stepKeys, stepValues) = miniMax.update(keys: keys, values: values)
            return [stepKeys, stepValues, miniMax.updateIndexAndFetch(keys)]
        default:
            let (stepKeys, stepValues) = cache.update(keys: keys, values: values)
            return [stepKeys, stepValues]
        }
    }

    /// Makes the keys or the values of one prefill.
    ///
    /// - Parameter seed: The seed of the random key.
    /// - Returns: The block.
    private func prompt(seed: UInt64) -> MLXArray {
        Fixture.block(tokenCount: Fixture.promptTokenCount, seed: seed)
    }

    /// Feeds single tokens into a ring, as a decode loop does.
    ///
    /// - Parameters:
    ///   - cache: The ring.
    ///   - tokenCount: The number of tokens to feed.
    private func fillRing(_ cache: any KVCache, tokenCount: Int) throws {
        let ring = try #require(cache as? RotatingKVCache)
        for index in 0 ..< tokenCount {
            let seed = Fixture.firstSeed + UInt64(index)
            _ = ring.update(
                keys: Fixture.block(tokenCount: 1, seed: seed),
                values: Fixture.block(tokenCount: 1, seed: seed + 1))
        }
    }

    /// Writes the slots of a recurrent cache. An `ArraysCache` gets its last slot only, thus
    /// the saved slot list is not contiguous.
    ///
    /// - Parameter cache: The recurrent cache.
    private func fillSlots(_ cache: any KVCache) throws {
        let slots = try #require(cache as? ArraysCache)
        let seed = Fixture.firstSeed
        if !(slots is MambaCache) {
            slots[1] = Fixture.tensor(Fixture.recurrentSlotShape, seed: seed)
            return
        }
        slots[0] = Fixture.tensor(Fixture.convolutionSlotShape, seed: seed)
        slots[1] = Fixture.tensor(Fixture.recurrentSlotShape, seed: seed + 1)
    }

    /// Takes one recurrent step: each present slot gets a new value made from its old value.
    ///
    /// - Parameter cache: The recurrent cache.
    /// - Returns: The new slot values.
    private func stepSlots(_ cache: any KVCache) throws -> [MLXArray] {
        let slots = try #require(cache as? ArraysCache)
        var output: [MLXArray] = []
        for index in 0 ..< slots.slotCount {
            if let previous = slots[index] {
                let next = previous * previous + previous
                slots[index] = next
                output.append(next)
            }
        }
        return output
    }

    /// Fills the window and each chunk branch of a DeepSeek-V4 cache.
    ///
    /// - Parameter cache: The DeepSeek-V4 cache.
    private func fillDeepSeek(_ cache: any KVCache) throws {
        let deepSeek = try #require(cache as? DeepSeekV4Cache)
        let seed = Fixture.firstSeed
        _ = deepSeek.update(keys: prompt(seed: seed), values: prompt(seed: seed + 1))
        deepSeek.attentionChunks.restore(state: Fixture.deepSeekBranchSlots(seed: seed))
        deepSeek.indexerChunks?.restore(state: Fixture.deepSeekBranchSlots(seed: seed + 1))
    }

    /// Fills the keys and the values of a MiniMax-M3 cache, and its index keys for the kind
    /// that has them.
    ///
    /// - Parameter cache: The MiniMax-M3 cache.
    private func fillMiniMax(_ cache: any KVCache) throws {
        let miniMax = try #require(cache as? MiniMaxM3KVCache)
        let seed = Fixture.firstSeed
        _ = miniMax.update(keys: prompt(seed: seed), values: prompt(seed: seed + 1))
        if self == .miniMaxWithIndex {
            _ = miniMax.updateIndexAndFetch(prompt(seed: seed))
        }
    }
}

// MARK: - Ring configuration mismatches

/// One `RotatingKVCache` template whose configuration is not the configuration of the saved
/// ring of ``CacheKind/rotatingAfterWrap``.
enum RingConfigurationMismatch: String, CaseIterable, CustomTestStringConvertible, Sendable {
    /// The template has a larger window.
    case window

    /// The template keeps leading tokens in the window. The saved ring keeps none.
    case keep

    /// The name of the case in the test report.
    var testDescription: String { rawValue }

    /// The fixture sizes and builders.
    private typealias Fixture = PromptCacheTemplateRestoreTests

    /// The window of the ``window`` template. It is larger than the window of the saved ring
    /// (``PromptCacheTemplateRestoreTests/rotatingWindow``).
    private static let otherWindow = 8

    /// The number of leading tokens that the ``keep`` template keeps.
    private static let otherKeep = 1

    /// Makes the fresh ring that a model of another configuration makes.
    ///
    /// - Returns: An empty ring.
    func makeTemplate() -> RotatingKVCache {
        switch self {
        case .window: RotatingKVCache(maxSize: Self.otherWindow)
        case .keep: RotatingKVCache(maxSize: Fixture.rotatingWindow, keep: Self.otherKeep)
        }
    }
}

// MARK: - Offset cache kinds

/// One kind of cache whose `state` and `metaState` do not hold its offset. Only the offset
/// record of the file brings the offset back.
enum OffsetCacheKind: String, CaseIterable, CustomTestStringConvertible, Sendable {
    case mamba
    case arrays
    case chunkedFrontTrimmed

    /// The name of the kind in the test report.
    var testDescription: String { rawValue }

    /// The fixture sizes and builders.
    private typealias Fixture = PromptCacheTemplateRestoreTests

    /// The kind of ``CacheKind`` that takes the same step as this kind.
    var stepKind: CacheKind {
        switch self {
        case .mamba: .mamba
        case .arrays: .arrays
        case .chunkedFrontTrimmed: .chunked
        }
    }

    /// The offset of a filled cache of this kind.
    var savedOffset: Int {
        switch self {
        case .mamba, .arrays: Fixture.promptTokenCount
        case .chunkedFrontTrimmed: Fixture.frontTrimmedTokenCount
        }
    }

    /// The offset that a load gives a cache of this kind when the file has no offset record.
    var offsetWithoutRecord: Int {
        switch self {
        case .mamba, .arrays: 0
        case .chunkedFrontTrimmed: Fixture.chunkSize
        }
    }

    /// Makes the fresh cache that a model makes for this kind.
    ///
    /// - Returns: An empty cache.
    func makeTemplate() throws -> any KVCache {
        try stepKind.makeTemplate()
    }

    /// Makes a cache of this kind that holds data and has the offset ``savedOffset``.
    ///
    /// - Returns: The filled cache.
    func makeFilled() throws -> any KVCache {
        switch self {
        case .mamba:
            let mamba = try #require(try CacheKind.mamba.makeFilled() as? MambaCache)
            mamba.advance(savedOffset)
            return mamba
        case .arrays:
            let arrays = try #require(try CacheKind.arrays.makeFilled() as? ArraysCache)
            arrays.offset = savedOffset
            return arrays
        case .chunkedFrontTrimmed:
            let chunked = ChunkedKVCache(chunkSize: Fixture.chunkSize)
            let seed = Fixture.firstSeed
            chunked.state = [
                Fixture.block(tokenCount: savedOffset, seed: seed),
                Fixture.block(tokenCount: savedOffset, seed: seed + 1),
            ]
            chunked.maybeTrimFront()
            return chunked
        }
    }
}

// MARK: - Quantized configuration mismatches

/// One `QuantizedKVCache` template whose configuration is not the configuration of the saved
/// cache of ``CacheKind/quantized``.
enum QuantizedConfigurationMismatch: String, CaseIterable, CustomTestStringConvertible, Sendable {
    /// The template has a smaller group size.
    case groupSize

    /// The template has a smaller bit width.
    case bits

    /// The name of the case in the test report.
    var testDescription: String { rawValue }

    /// The fixture sizes and builders.
    private typealias Fixture = PromptCacheTemplateRestoreTests

    /// The group size of the ``groupSize`` template. It is smaller than the group size of the
    /// saved cache (``PromptCacheTemplateRestoreTests/quantizedGroupSize``).
    private static let otherGroupSize = 32

    /// The bit width of the ``bits`` template. It is smaller than the bit width of the saved
    /// cache (``PromptCacheTemplateRestoreTests/quantizedBits``).
    private static let otherBits = 4

    /// Makes the fresh quantized cache that a model of another configuration makes.
    ///
    /// - Returns: An empty quantized cache.
    func makeTemplate() -> QuantizedKVCache {
        switch self {
        case .groupSize:
            QuantizedKVCache(groupSize: Self.otherGroupSize, bits: Fixture.quantizedBits)
        case .bits:
            QuantizedKVCache(groupSize: Fixture.quantizedGroupSize, bits: Self.otherBits)
        }
    }
}

// MARK: - Cache classes outside the library

/// A cache class outside the library that no registry entry names, and that is not
/// `PromptCacheRestorable`. A filled instance holds more arrays than a `KVCacheSimple`.
class CustomSlotCache: BaseKVCache {
    /// The arrays of the cache.
    var slots: [MLXArray] = []

    /// The arrays of the cache, as a save reads them.
    override var state: [MLXArray] {
        get { slots }
        set { slots = newValue }
    }
}

/// A cache class outside the library that `KVCacheSerializationRegistry` names, and that is not
/// `PromptCacheRestorable`.
final class RegisteredSlotCache: CustomSlotCache {
    /// The class name that the registry holds for this class.
    static let className = "RegisteredSlotCache"

    /// Registers the class one time, before the first instance exists.
    private static let registration: Void = KVCacheSerializationRegistry.register(
        RegisteredSlotCache.self, className: className
    ) { state, _ in
        let cache = RegisteredSlotCache()
        cache.slots = state
        return cache
    }

    /// Makes an empty cache, after the registration of the class.
    override init() {
        _ = Self.registration
        super.init()
    }
}
