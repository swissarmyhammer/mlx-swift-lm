// Copyright © 2026 Apple Inc.

import CryptoKit
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Tests for the two-step save of a prompt cache: `preparePromptCacheSave(cache:metadata:state:)`
/// reads the caches, and `writePromptCache(_:url:)` writes the file.
struct PromptCacheSaveInputTests {

    // MARK: - Fixture sizes

    /// The number of key/value heads in each fixture.
    fileprivate static let headCount = 2

    /// The head dimension of each fixture.
    fileprivate static let headDim = 4

    /// The number of tokens of the prefill of each fixture.
    fileprivate static let promptTokenCount = 3

    /// The window of the `RotatingKVCache` fixture. The fixture fills it exactly, thus the
    /// cache gives its own buffers as its state.
    fileprivate static let rotatingWindow = 4

    /// The shape of each slot of the `MambaCache` fixture.
    fileprivate static let recurrentSlotShape = [1, 2, 3]

    /// The distance between the first values of two fixture arrays, thus no two arrays hold the
    /// same values.
    fileprivate static let valueStride: Float = 100

    /// The distance between the first values of two tokens of the `RotatingKVCache` fixture.
    fileprivate static let rotatingTokenStride: Float = 10

    /// The distance between the first value of the keys and the first value of the values of one
    /// token of the `RotatingKVCache` fixture.
    fileprivate static let rotatingValueOffset: Float = 5

    /// The fixture arrays, in the order of their first values.
    fileprivate enum FixtureArray: Int {
        case simpleKeys
        case simpleValues
        case rotating
        case mambaConvolution
        case mambaRecurrent
        case nextToken
        case nextConvolution
        case nextRecurrent
    }

    /// Gives the first value of a fixture array.
    ///
    /// - Parameter array: The fixture array.
    /// - Returns: The first value.
    fileprivate static func start(of array: FixtureArray) -> Float {
        Float(array.rawValue) * valueStride
    }

    /// The content digest (see ``contentDigest(of:)``) of the file that `savePromptCache`
    /// wrote for ``deterministicCaches()`` and ``fixtureMetadata`` BEFORE the save became two
    /// steps. The digest proves that the two-step save writes the same metadata and array bytes
    /// as the save it replaced.
    fileprivate static let savedFileDigest =
        "1ba2a6a131632edec9cce45c5b632ee8599fe9fb893a5448536c4795d672c424"

    /// The caller metadata of each save of this suite.
    fileprivate static let fixtureMetadata = ["purpose": "fixture"]

    // MARK: - Fixture builders

    /// Makes an array of consecutive values that starts at `start`.
    ///
    /// - Parameters:
    ///   - shape: The shape of the array.
    ///   - start: The first value.
    /// - Returns: The array, as `float16`.
    fileprivate static func ramp(_ shape: [Int], start: Float) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { start + Float($0) }).reshaped(shape).asType(.float16)
    }

    /// Makes the keys or the values of a block of tokens.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens of the block.
    ///   - start: The first value.
    /// - Returns: An array of shape `(1, heads, tokens, headDim)`.
    fileprivate static func block(tokenCount: Int, start: Float) -> MLXArray {
        ramp([1, headCount, tokenCount, headDim], start: start)
    }

    /// Makes one cache of each class that the suite writes, each filled with fixed values.
    ///
    /// - Returns: A `KVCacheSimple`, a full `RotatingKVCache` and a `MambaCache`.
    fileprivate static func deterministicCaches() -> [KVCache] {
        let simple = KVCacheSimple()
        _ = simple.update(
            keys: block(tokenCount: promptTokenCount, start: start(of: .simpleKeys)),
            values: block(tokenCount: promptTokenCount, start: start(of: .simpleValues)))

        let rotating = RotatingKVCache(maxSize: rotatingWindow)
        for token in 0 ..< rotatingWindow {
            let tokenStart = start(of: .rotating) + Float(token) * rotatingTokenStride
            _ = rotating.update(
                keys: block(tokenCount: 1, start: tokenStart),
                values: block(tokenCount: 1, start: tokenStart + rotatingValueOffset))
        }

        let mamba = MambaCache()
        mamba[0] = ramp(recurrentSlotShape, start: start(of: .mambaConvolution))
        mamba[1] = ramp(recurrentSlotShape, start: start(of: .mambaRecurrent))
        mamba.offset = promptTokenCount
        return [simple, rotating, mamba]
    }

    /// Writes one more token into each cache, the way a model layer writes it: an attention
    /// cache takes an `update(keys:values:)`, and a recurrent cache takes new slot arrays.
    ///
    /// - Parameter caches: The caches of ``deterministicCaches()``.
    fileprivate static func advance(_ caches: [KVCache]) {
        let token = block(tokenCount: 1, start: start(of: .nextToken))
        for cache in caches {
            if let recurrent = cache as? MambaCache {
                recurrent[0] = ramp(recurrentSlotShape, start: start(of: .nextConvolution))
                recurrent[1] = ramp(recurrentSlotShape, start: start(of: .nextRecurrent))
            } else {
                _ = cache.update(keys: token, values: token)
            }
        }
    }

    /// Makes a temporary file URL for one prompt cache file.
    ///
    /// - Returns: The URL, in the temporary directory.
    fileprivate static func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
    }

    /// Gives the lowercase hexadecimal SHA-256 digest of the content of a safetensors file:
    /// each metadata entry, and the name, the type, the shape and the bytes of each array, in
    /// the order of the names.
    ///
    /// The writer places the arrays in the order of its own hash map, thus two writes of one
    /// input can place them in another order. The content digest reads every byte of every
    /// array and every metadata entry, and does not depend on that order.
    ///
    /// - Parameter url: The file to read.
    /// - Returns: The digest.
    fileprivate static func contentDigest(of url: URL) throws -> String {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        var hasher = SHA256()
        for (key, value) in metadata.sorted(by: { $0.key < $1.key }) {
            hasher.update(data: Data("\(key)=\(value)\n".utf8))
        }
        for (key, array) in arrays.sorted(by: { $0.key < $1.key }) {
            hasher.update(data: Data("\(key):\(array.dtype):\(array.shape)\n".utf8))
            hasher.update(data: array.asData(access: .copy).data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Tests

    @Test("savePromptCache writes the same content as before the save became two steps")
    func saveWritesTheSameContentAsBefore() throws {
        let url = Self.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(
            url: url, cache: Self.deterministicCaches(), metadata: Self.fixtureMetadata)

        let digest = try Self.contentDigest(of: url)
        #expect(digest == Self.savedFileDigest)
    }

    @Test("prepare and then write gives the content of savePromptCache")
    func prepareAndWriteGiveTheContentOfSave() throws {
        let savedURL = Self.temporaryURL()
        let writtenURL = Self.temporaryURL()
        defer {
            try? FileManager.default.removeItem(at: savedURL)
            try? FileManager.default.removeItem(at: writtenURL)
        }

        try savePromptCache(
            url: savedURL, cache: Self.deterministicCaches(), metadata: Self.fixtureMetadata)
        let input = try preparePromptCacheSave(
            cache: Self.deterministicCaches(), metadata: Self.fixtureMetadata)
        try writePromptCache(input, url: writtenURL)

        #expect(try Self.contentDigest(of: writtenURL) == Self.contentDigest(of: savedURL))
    }

    @Test("the input names the class of each cache")
    func theInputNamesTheClassOfEachCache() throws {
        let input = try preparePromptCacheSave(cache: Self.deterministicCaches())

        #expect(input.classNames == ["KVCache", "RotatingKVCache", "MambaCache"])
        #expect(input.metadata["2.1"] == "RotatingKVCache")
    }

    @Test("an update after prepare does not change what write puts in the file")
    func anUpdateAfterPrepareDoesNotChangeTheWrittenFile() throws {
        let beforeURL = Self.temporaryURL()
        let afterURL = Self.temporaryURL()
        defer {
            try? FileManager.default.removeItem(at: beforeURL)
            try? FileManager.default.removeItem(at: afterURL)
        }
        try savePromptCache(url: beforeURL, cache: Self.deterministicCaches())

        let caches = Self.deterministicCaches()
        let input = try preparePromptCacheSave(cache: caches)
        Self.advance(caches)
        try writePromptCache(input, url: afterURL)

        #expect(try Self.contentDigest(of: afterURL) == Self.contentDigest(of: beforeURL))
    }

    @Test("the input holds no array object of a cache")
    func theInputHoldsNoArrayObjectOfACache() throws {
        let caches = Self.deterministicCaches()
        // The list keeps each state array alive, thus no identifier is used again.
        let cacheArrays = caches.flatMap(\.state)

        let input = try preparePromptCacheSave(cache: caches)

        let inputArrays = Set(input.arrays.values.map(ObjectIdentifier.init))
        #expect(inputArrays.isDisjoint(with: cacheArrays.map(ObjectIdentifier.init)))
    }

    @Test("prepare refuses a model state with no cache")
    func prepareRefusesAModelStateWithNoCache() throws {
        var state = LMOutput.State()
        state[LMOutput.Key<MLXArray>("test.positions")] = Self.ramp(
            Self.recurrentSlotShape, start: Self.start(of: .simpleKeys))

        let error = #expect(throws: KVCacheError.self) {
            try preparePromptCacheSave(cache: [], state: state)
        }
        #expect(error?.message == "Model state requires at least one prompt cache")
    }
}
