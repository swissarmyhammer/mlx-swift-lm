// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Tests for ``ExecutorPromptCacheFile``, which writes one executor prompt cache entry to one
/// file and reads it back.
///
/// No weights are needed. Each entry carries small caches with fixed values: one
/// `KVCacheSimple`, one `RotatingKVCache` and one `MambaCache` layer, thus each read restores
/// every kind of layer into the fresh caches of a model.
@Suite("An executor prompt cache entry goes to one file and comes back")
struct ExecutorPromptCacheFileTests {

    // MARK: - Fixture values

    /// The model that the key of each entry names. It holds a `/`, as a real model ID does.
    private static let modelID = "test-org/prompt-cache-file"

    /// The session that the key of each entry names.
    private static let sessionID = "session-a"

    /// The ledger of each entry. It holds a large token ID, thus the encoding keeps every bit
    /// that an Int32 holds.
    private static let tokens = [1, 42, 151_643]

    /// The whole prompt that the last pass of each entry rendered.
    private static let renderTokens = [1, 42, 151_643, 7, 9]

    /// The number of key/value heads of each attention layer.
    private static let headCount = 2

    /// The head dimension of each attention layer.
    private static let headDim = 4

    /// The window of the `RotatingKVCache` layer. It is larger than the ledger, thus the ring
    /// does not wrap.
    private static let rotatingWindow = 8

    /// The shape of each slot of the `MambaCache` layer.
    private static let recurrentSlotShape = [1, 2, 3]

    /// The distance between the first values of two fixture arrays.
    private static let valueStride: Float = 100

    /// The generation that the file-name tests use.
    private static let generation: UInt64 = 7

    /// The number of hexadecimal digits of a SHA-256 digest.
    private static let digestHexLength = 64

    /// The state key of the model state of each entry.
    private static let ropeDeltasKey = LMOutput.Key<MLXArray>("test.ropeDeltas")

    /// The value of the model state of each entry.
    private static let ropeDeltas: [Int32] = [3, -2]

    /// The fixture arrays, in the order of their first values.
    private enum FixtureArray: Int {
        case simpleKeys
        case simpleValues
        case rotatingKeys
        case rotatingValues
        case mambaConvolution
        case mambaRecurrent
        case nextToken
    }

    /// The size to which a truncation test cuts a file.
    enum Truncation: CaseIterable, CustomTestStringConvertible {
        /// The file loses its last byte, which is in the array data.
        case lastByte

        /// The file keeps only its header length and a part of the header.
        case insideHeader

        /// The number of bytes the header cut keeps.
        static let headerCutLength: UInt64 = 16

        /// The size of the cut file.
        ///
        /// - Parameter size: The size of the whole file.
        /// - Returns: The size after the cut.
        func cutSize(of size: UInt64) -> UInt64 {
            switch self {
            case .lastByte: size - 1
            case .insideHeader: Self.headerCutLength
            }
        }

        /// The name of the case in the test report.
        var testDescription: String {
            switch self {
            case .lastByte: "the last byte"
            case .insideHeader: "the header"
            }
        }
    }

    // MARK: - Fixture builders

    /// The key of each entry of this suite.
    private static var key: ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
    }

    /// Makes an array of consecutive values.
    ///
    /// - Parameters:
    ///   - shape: The shape of the array.
    ///   - array: The fixture array. It sets the first value.
    /// - Returns: The array, as `float16`.
    private static func ramp(_ shape: [Int], array: FixtureArray) -> MLXArray {
        let count = shape.reduce(1, *)
        let start = Float(array.rawValue) * valueStride
        return MLXArray((0 ..< count).map { start + Float($0) }).reshaped(shape).asType(.float16)
    }

    /// Makes the keys or the values of a block of tokens.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens of the block.
    ///   - array: The fixture array.
    /// - Returns: An array of shape `(1, heads, tokens, headDim)`.
    private static func block(tokenCount: Int, array: FixtureArray) -> MLXArray {
        ramp([1, headCount, tokenCount, headDim], array: array)
    }

    /// Makes the three layers of an entry, each at the length of ``tokens``.
    ///
    /// - Returns: A `KVCacheSimple`, a `RotatingKVCache` and a `MambaCache`.
    private static func filledCaches() -> [KVCache] {
        let tokenCount = tokens.count
        let simple = KVCacheSimple()
        _ = simple.update(
            keys: block(tokenCount: tokenCount, array: .simpleKeys),
            values: block(tokenCount: tokenCount, array: .simpleValues))
        let rotating = RotatingKVCache(maxSize: rotatingWindow)
        _ = rotating.update(
            keys: block(tokenCount: tokenCount, array: .rotatingKeys),
            values: block(tokenCount: tokenCount, array: .rotatingValues))
        let mamba = MambaCache()
        mamba[0] = ramp(recurrentSlotShape, array: .mambaConvolution)
        mamba[1] = ramp(recurrentSlotShape, array: .mambaRecurrent)
        mamba.offset = tokenCount
        return [simple, rotating, mamba]
    }

    /// Makes the fresh caches that a model gives for the layers of ``filledCaches()``.
    ///
    /// - Returns: One empty cache for each layer.
    private static func templates() -> [KVCache] {
        [KVCacheSimple(), RotatingKVCache(maxSize: rotatingWindow), MambaCache()]
    }

    /// Makes the model state of an entry.
    ///
    /// - Returns: A state with one array.
    private static func modelState() -> LMOutput.State {
        var state = LMOutput.State()
        state[ropeDeltasKey] = MLXArray(ropeDeltas)
        return state
    }

    /// Makes the entry of each round trip.
    ///
    /// - Parameter state: The model state of the entry, or nil.
    /// - Returns: An entry with the three layers and the two ledgers.
    private static func entry(state: LMOutput.State? = nil) -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(
            caches: filledCaches(), tokens: tokens, renderTokens: renderTokens, state: state)
    }

    /// Makes an empty temporary directory for one test.
    ///
    /// - Returns: The URL of the directory.
    private static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ExecutorPromptCacheFileTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Writes `entry` into `directory` under the name of ``key``.
    ///
    /// - Parameters:
    ///   - entry: The entry to write.
    ///   - directory: The directory of the file.
    /// - Returns: The URL of the file.
    private static func write(_ entry: ExecutorPromptCacheEntry, in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: generation))
        try ExecutorPromptCacheFile.write(ExecutorPromptCacheFile.prepare(entry, key: key), to: url)
        return url
    }

    // MARK: - Comparisons

    /// Records an issue unless two lists of arrays have equal shapes, types and values.
    ///
    /// - Parameters:
    ///   - actual: The arrays to check.
    ///   - expected: The arrays they must equal.
    ///   - label: The name of the comparison, for the issue text.
    private static func expectEqual(
        _ actual: [MLXArray], _ expected: [MLXArray], _ label: String
    ) {
        #expect(actual.count == expected.count, "\(label): array count")
        for (index, (lhs, rhs)) in zip(actual, expected).enumerated() {
            #expect(lhs.shape == rhs.shape, "\(label) array \(index): shape")
            #expect(lhs.dtype == rhs.dtype, "\(label) array \(index): type")
            #expect(arrayEqual(lhs, rhs).item(Bool.self), "\(label) array \(index): values")
        }
    }

    /// Records an issue unless the caches of `actual` hold the offsets and the arrays of the
    /// caches of `expected`.
    ///
    /// - Parameters:
    ///   - actual: The caches that a read gave back.
    ///   - expected: The caches that were written.
    private static func expectEqualCaches(_ actual: [KVCache], _ expected: [KVCache]) {
        #expect(actual.map(\.offset) == expected.map(\.offset))
        for (index, (lhs, rhs)) in zip(actual, expected).enumerated() {
            expectEqual(lhs.state, rhs.state, "cache \(index)")
        }
    }

    // MARK: - The round trip

    @Test("prepare, write and read give back the ledgers, the offsets and the arrays")
    func aRoundTripGivesBackTheEntry() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = Self.entry()

        let url = try Self.write(original, in: directory)
        let restored = try ExecutorPromptCacheFile.read(
            from: url, key: Self.key, templates: Self.templates())

        #expect(restored.tokens == Self.tokens)
        #expect(restored.renderTokens == Self.renderTokens)
        #expect(restored.state == nil)
        let layerCount = Self.templates().count
        #expect(
            restored.caches.map(\.offset) == Array(repeating: Self.tokens.count, count: layerCount))
        Self.expectEqualCaches(restored.caches, Self.filledCaches())
    }

    @Test("a round trip gives back the model state")
    func aRoundTripGivesBackTheModelState() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = try Self.write(Self.entry(state: Self.modelState()), in: directory)
        let restored = try ExecutorPromptCacheFile.read(
            from: url, key: Self.key, templates: Self.templates())

        let ropeDeltas = try #require(restored.state?[Self.ropeDeltasKey])
        #expect(ropeDeltas.asArray(Int32.self) == Self.ropeDeltas)
        #expect(restored.tokens == Self.tokens)
        Self.expectEqualCaches(restored.caches, Self.filledCaches())
    }

    @Test("an update after prepare does not change what the file holds")
    func anUpdateAfterPrepareDoesNotChangeTheFile() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: Self.key, generation: Self.generation))
        let entry = Self.entry()

        let input = try ExecutorPromptCacheFile.prepare(entry, key: Self.key)
        let token = Self.block(tokenCount: 1, array: .nextToken)
        _ = entry.caches[0].update(keys: token, values: token)
        _ = entry.caches[1].update(keys: token, values: token)
        try ExecutorPromptCacheFile.write(input, to: url)
        let restored = try ExecutorPromptCacheFile.read(
            from: url, key: Self.key, templates: Self.templates())

        Self.expectEqualCaches(restored.caches, Self.filledCaches())
    }

    @Test("a successful write leaves no partial file")
    func aSuccessfulWriteLeavesNoPartialFile() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = try Self.write(Self.entry(), in: directory)

        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(names == [url.lastPathComponent])
        #expect(!names.contains { $0.hasSuffix(".partial.safetensors") })
    }

    @Test("a second write replaces the file of the first")
    func aSecondWriteReplacesTheFile() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try Self.write(Self.entry(), in: directory)

        let url = try Self.write(Self.entry(state: Self.modelState()), in: directory)
        let restored = try ExecutorPromptCacheFile.read(
            from: url, key: Self.key, templates: Self.templates())

        #expect(restored.state?[Self.ropeDeltasKey] != nil)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).count == 1)
    }

    // MARK: - A file that must not load

    @Test("a read with the key of another session throws")
    func aReadWithAnotherSessionThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try Self.write(Self.entry(), in: directory)
        let otherSession = ExecutorPromptCacheKey(modelID: Self.modelID, sessionID: "session-b")

        #expect(throws: ExecutorPromptCacheFileError.keyMismatch) {
            try ExecutorPromptCacheFile.read(
                from: url, key: otherSession, templates: Self.templates())
        }
    }

    @Test("a read with the key of another model throws")
    func aReadWithAnotherModelThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try Self.write(Self.entry(), in: directory)
        let otherModel = ExecutorPromptCacheKey(
            modelID: "test-org/other", sessionID: Self.sessionID)

        #expect(throws: ExecutorPromptCacheFileError.keyMismatch) {
            try ExecutorPromptCacheFile.read(
                from: url, key: otherModel, templates: Self.templates())
        }
    }

    @Test("a file of another format throws")
    func aFileOfAnotherFormatThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("other-format.safetensors")
        try savePromptCache(
            url: url, cache: Self.filledCaches(),
            metadata: [
                "format": "another-format", "modelID": Self.modelID, "sessionID": Self.sessionID,
            ])

        #expect(throws: ExecutorPromptCacheFileError.formatMismatch("another-format")) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    @Test("a prompt cache file with no format throws")
    func aFileWithNoFormatThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("no-format.safetensors")
        try savePromptCache(url: url, cache: Self.filledCaches())

        #expect(throws: ExecutorPromptCacheFileError.formatMismatch(nil)) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    @Test("a truncated file throws", arguments: Truncation.allCases)
    func aTruncatedFileThrows(_ truncation: Truncation) throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try Self.write(Self.entry(), in: directory)
        let size = try #require(
            try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: truncation.cutSize(of: size))
        try handle.close()

        #expect(throws: (any Error).self) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    @Test("a missing file throws")
    func aMissingFileThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: Self.key, generation: Self.generation))

        #expect(throws: (any Error).self) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    @Test("a file whose caches do not stand at the end of the ledger throws")
    func aFileWithAnotherOffsetThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let longerLedger = Self.tokens + Self.tokens
        let entry = ExecutorPromptCacheEntry(caches: Self.filledCaches(), tokens: longerLedger)

        let url = try Self.write(entry, in: directory)
        let offsets = Array(repeating: Self.tokens.count, count: Self.templates().count)

        #expect(
            throws: ExecutorPromptCacheFileError.offsetMismatch(
                offsets: offsets, ledgerLength: longerLedger.count)
        ) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    // MARK: - The file name

    @Test("the file name is the digest of the key and the generation")
    func theFileNameIsTheDigestOfTheKeyAndTheGeneration() {
        let keyBytes = Data("\(Self.modelID)\u{0}\(Self.sessionID)".utf8)
        let digest = SHA256.hash(data: keyBytes).map { String(format: "%02x", $0) }.joined()

        let name = ExecutorPromptCacheFile.fileName(for: Self.key, generation: Self.generation)

        #expect(digest.count == Self.digestHexLength)
        #expect(name == "\(digest)-\(Self.generation).safetensors")
        #expect(!name.contains("/"))
    }

    @Test("two sessions get two file names")
    func twoSessionsGetTwoFileNames() {
        let otherSession = ExecutorPromptCacheKey(modelID: Self.modelID, sessionID: "session-b")

        #expect(
            ExecutorPromptCacheFile.fileName(for: Self.key, generation: Self.generation)
                != ExecutorPromptCacheFile.fileName(for: otherSession, generation: Self.generation))
    }
}

#endif
