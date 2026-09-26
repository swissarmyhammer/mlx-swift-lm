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

    /// A token ID one above `Int32.max`, which a file cannot hold.
    private static let outOfRangeToken = Int(Int32.max) + 1

    /// The number of bytes of the header length at the start of a safetensors file.
    private static let headerLengthByteCount = MemoryLayout<UInt64>.size

    /// The metadata key of the file format.
    private static let formatKey = "format"

    /// The metadata key of the model of the key.
    private static let modelIDKey = "modelID"

    /// The metadata key of the session of the key.
    private static let sessionIDKey = "sessionID"

    /// The metadata text of an empty ledger, which decodes.
    private static let emptyLedger = ""

    /// The permissions of a folder whose files cannot be removed.
    private static let readOnlyFolderPermissions = 0o555

    /// The permissions of a folder whose files can be removed.
    private static let writableFolderPermissions = 0o755

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

        /// The size that the length check expects of the cut file: the whole file when the cut
        /// is in the array data, or the end of the header when the cut is in the header.
        ///
        /// - Parameters:
        ///   - size: The size of the whole file.
        ///   - headerEnd: The offset of the first array byte of the whole file.
        /// - Returns: The expected byte count of the `.truncated` error.
        func expectedByteCount(size: UInt64, headerEnd: UInt64) -> UInt64 {
            switch self {
            case .lastByte: size
            case .insideHeader: headerEnd
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

    /// The metadata key of a token ledger of the file.
    enum LedgerKey: String, CaseIterable {
        /// The token ledger of the entry.
        case tokens

        /// The whole prompt that the last pass rendered.
        case renderTokens
    }

    /// The damage that a ledger test gives to one ledger of a file.
    enum LedgerDamage: CaseIterable, CustomTestStringConvertible {
        /// The file has no metadata under the key of the ledger.
        case missing

        /// The text of the ledger is not base64.
        case notBase64

        /// The ledger holds 3 bytes, which is not a whole Int32 value.
        case partialToken

        /// The text of a ledger that is not base64.
        static let notBase64Text = "not base64!"

        /// The number of bytes of a partial token: one less than an Int32.
        static let partialTokenByteCount = MemoryLayout<Int32>.size - 1

        /// The metadata text of the damaged ledger.
        ///
        /// - Returns: The text, or nil when the file has no text under the key.
        var text: String? {
            switch self {
            case .missing: nil
            case .notBase64: Self.notBase64Text
            case .partialToken: Data(count: Self.partialTokenByteCount).base64EncodedString()
            }
        }

        /// The name of the case in the test report.
        var testDescription: String {
            switch self {
            case .missing: "a missing ledger"
            case .notBase64: "a ledger that is not base64"
            case .partialToken: "a ledger of 3 bytes"
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
        let url = fileURL(in: directory)
        try ExecutorPromptCacheFile.write(ExecutorPromptCacheFile.prepare(entry, key: key), to: url)
        return url
    }

    /// The URL of the file that ``write(_:in:)`` writes into `directory`.
    ///
    /// - Parameter directory: The directory of the file.
    /// - Returns: The URL of the file.
    private static func fileURL(in directory: URL) -> URL {
        directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: generation))
    }

    /// The partial file that a write to `url` writes first: `<name>.partial.safetensors`.
    ///
    /// - Parameter url: The URL of the real file.
    /// - Returns: The URL of the partial file.
    private static func partialURL(for url: URL) -> URL {
        url.deletingPathExtension()
            .appendingPathExtension(ExecutorPromptCacheFile.partialMarker)
            .appendingPathExtension(ExecutorPromptCacheFile.fileExtension)
    }

    /// The offset of the first array byte of a safetensors file: the header length field and
    /// the header that it names.
    ///
    /// - Parameter url: The URL of the whole file.
    /// - Returns: The end of the header.
    private static func headerEnd(of url: URL) throws -> UInt64 {
        let bytes = try Data(contentsOf: url).prefix(headerLengthByteCount)
        let headerLength = UInt64(
            littleEndian: bytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        return UInt64(headerLengthByteCount) + headerLength
    }

    /// The names of the items in `directory`.
    ///
    /// - Parameter directory: The directory to list.
    /// - Returns: The names, in sorted order.
    private static func names(in directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    // MARK: - Comparisons

    /// Records an issue unless two lists of arrays have equal shapes, types and values.
    ///
    /// - Parameters:
    ///   - actual: The arrays to check.
    ///   - expected: The arrays they must equal.
    ///   - label: The name of the comparison, for the issue text.
    static func expectEqual(
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
    static func expectEqualCaches(_ actual: [KVCache], _ expected: [KVCache]) {
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
        let url = Self.fileURL(in: directory)
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
                Self.formatKey: "another-format", Self.modelIDKey: Self.modelID,
                Self.sessionIDKey: Self.sessionID,
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

    @Test("a truncated file throws .truncated with the cut size", arguments: Truncation.allCases)
    func aTruncatedFileThrowsTruncated(_ truncation: Truncation) throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try Self.write(Self.entry(), in: directory)
        let size = try #require(
            try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64)
        let headerEnd = try Self.headerEnd(of: url)
        let cutSize = truncation.cutSize(of: size)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: cutSize)
        try handle.close()

        #expect(
            throws: ExecutorPromptCacheFileError.truncated(
                byteCount: cutSize,
                expectedByteCount: truncation.expectedByteCount(size: size, headerEnd: headerEnd))
        ) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
    }

    @Test("a missing file throws the no-such-file error of the file system")
    func aMissingFileThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = Self.fileURL(in: directory)

        let error = #expect(throws: CocoaError.self) {
            try ExecutorPromptCacheFile.read(from: url, key: Self.key, templates: Self.templates())
        }
        #expect(error?.code == .fileNoSuchFile)
    }

    @Test(
        "a ledger that does not decode throws .ledgerNotDecodable with its key",
        arguments: LedgerKey.allCases, LedgerDamage.allCases)
    func aLedgerThatDoesNotDecodeThrows(_ key: LedgerKey, _ damage: LedgerDamage) throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = Self.fileURL(in: directory)
        var metadata = [
            Self.formatKey: ExecutorPromptCacheFile.format,
            Self.modelIDKey: Self.modelID,
            Self.sessionIDKey: Self.sessionID,
            LedgerKey.tokens.rawValue: Self.emptyLedger,
            LedgerKey.renderTokens.rawValue: Self.emptyLedger,
        ]
        metadata[key.rawValue] = damage.text
        try savePromptCache(url: url, cache: Self.filledCaches(), metadata: metadata)

        #expect(throws: ExecutorPromptCacheFileError.ledgerNotDecodable(key.rawValue)) {
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

    // MARK: - A write that must fail

    @Test("prepare of a token that does not fit in an Int32 throws .tokenOutOfRange")
    func prepareOfAnOutOfRangeTokenThrows() {
        let entry = ExecutorPromptCacheEntry(
            caches: Self.filledCaches(), tokens: [Self.outOfRangeToken])

        #expect(throws: ExecutorPromptCacheFileError.tokenOutOfRange(Self.outOfRangeToken)) {
            try ExecutorPromptCacheFile.prepare(entry, key: Self.key)
        }
    }

    @Test("a write into a missing folder throws the safetensors error and leaves no partial file")
    func aWriteIntoAMissingFolderThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let missingFolder = directory.appendingPathComponent("missing", isDirectory: true)
        let url = Self.fileURL(in: missingFolder)
        let partialURL = Self.partialURL(for: url)
        let input = try ExecutorPromptCacheFile.prepare(Self.entry(), key: Self.key)

        let error = #expect(throws: MLXError.self) {
            try ExecutorPromptCacheFile.write(input, to: url)
        }

        let message = try #require(error?.errorDescription)
        #expect(
            message.hasPrefix(
                "MLX Error: [save_safetensors] Failed to open file "
                    + "\(partialURL.path(percentEncoded: false)) at "))
        #expect(!FileManager.default.fileExists(atPath: partialURL.path(percentEncoded: false)))
        #expect(try Self.names(in: directory).isEmpty)
    }

    @Test("a rename onto a folder throws EISDIR and removes the partial file")
    func aRenameOntoAFolderThrows() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = Self.fileURL(in: directory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        let input = try ExecutorPromptCacheFile.prepare(Self.entry(), key: Self.key)

        #expect(throws: POSIXError(.EISDIR)) {
            try ExecutorPromptCacheFile.write(input, to: url)
        }

        #expect(try Self.names(in: directory) == [url.lastPathComponent])
    }

    // MARK: - The removal of a file

    @Test("removeFile of a missing file does nothing and reports nothing")
    func removeFileOfAMissingFileDoesNothing() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var reports: [String] = []

        ExecutorPromptCacheFile.removeFile(at: Self.fileURL(in: directory)) { reports.append($0) }

        #expect(reports.isEmpty)
        #expect(try Self.names(in: directory).isEmpty)
    }

    @Test("a removal that fails keeps the file and reports the error, and does not throw")
    func aRemovalThatFailsReportsTheError() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try Self.write(Self.entry(), in: directory)
        try FileManager.default.setAttributes(
            [.posixPermissions: Self.readOnlyFolderPermissions], ofItemAtPath: directory.path)
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: Self.writableFolderPermissions], ofItemAtPath: directory.path)
        }
        var reports: [String] = []

        ExecutorPromptCacheFile.removeFile(at: url) { reports.append($0) }

        #expect(FileManager.default.fileExists(atPath: url.path(percentEncoded: false)))
        #expect(reports.count == 1)
        let report = try #require(reports.first)
        #expect(report.hasPrefix("prompt cache file cannot remove \(url.lastPathComponent): "))
        #expect(
            report.contains(
                "Domain=NSCocoaErrorDomain Code=\(CocoaError.Code.fileWriteNoPermission.rawValue) "))
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
