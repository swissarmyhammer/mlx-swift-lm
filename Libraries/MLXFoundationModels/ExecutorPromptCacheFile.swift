// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import CryptoKit
import Foundation
import MLXLMCommon

/// Why a prompt cache file cannot give back the entry of a key.
enum ExecutorPromptCacheFileError: Error, Equatable {

    /// The file is not an executor prompt cache file of the format that this reader knows. The
    /// value is the format that the file names, or nil when it names none.
    case formatMismatch(String?)

    /// The file holds the cache of another model or of another session.
    case keyMismatch

    /// A token ledger of the file does not decode. The value is the metadata key of the ledger.
    case ledgerNotDecodable(String)

    /// A token ID of an entry does not fit in an Int32, thus the file cannot hold it.
    case tokenOutOfRange(Int)

    /// The restored caches do not stand at the end of the token ledger.
    case offsetMismatch(offsets: [Int], ledgerLength: Int)

    /// The file is shorter than its header says, thus a part of it is missing.
    case truncated(byteCount: UInt64, expectedByteCount: UInt64)
}

/// Writes one ``ExecutorPromptCacheEntry`` to one safetensors file and reads it back.
///
/// The file holds the caches, the model state and the two token ledgers of the entry. The write
/// has two steps: ``prepare(_:key:)`` reads the caches while the caller owns them, and
/// ``write(_:to:)`` writes the file later, on any task. The write goes to a partial file first
/// and then takes the real name, thus a crash never leaves a half file under the real name.
enum ExecutorPromptCacheFile {

    /// The format that each file of this writer names.
    static let format = "executor-prompt-cache-1"

    /// The file extension. The safetensors writer selects the format from it and refuses every
    /// other extension.
    static let fileExtension = "safetensors"

    /// The marker between the name and the extension of a partial file:
    /// `<name>.partial.safetensors`.
    static let partialMarker = "partial"

    /// The metadata keys of the file.
    private enum MetadataKey {
        /// The format of the file.
        static let format = "format"

        /// The model of the key.
        static let modelID = "modelID"

        /// The session of the key.
        static let sessionID = "sessionID"

        /// The token ledger of the entry.
        static let tokens = "tokens"

        /// The whole prompt that the last pass rendered.
        static let renderTokens = "renderTokens"

        /// The bytes that the entry holds in memory.
        static let byteCount = "byteCount"
    }

    /// Reads the caches, the model state and the ledgers of `entry`, and makes the input of
    /// ``write(_:to:)``.
    ///
    /// The caller runs this function while it owns the caches of `entry`. The input holds new
    /// array handles and no reference to a cache, thus the caches can take more tokens before
    /// the write.
    ///
    /// - Parameters:
    ///   - entry: The entry to write.
    ///   - key: The session that the entry belongs to.
    /// - Returns: The input of ``write(_:to:)``.
    /// - Throws: ``ExecutorPromptCacheFileError/tokenOutOfRange(_:)`` for a token ID that does
    ///   not fit in an Int32, or the `KVCacheError` of the prompt cache save.
    static func prepare(
        _ entry: ExecutorPromptCacheEntry, key: ExecutorPromptCacheKey
    ) throws -> PromptCacheSaveInput {
        let metadata = [
            MetadataKey.format: format,
            MetadataKey.modelID: key.modelID,
            MetadataKey.sessionID: key.sessionID,
            MetadataKey.tokens: try TokenLedger.encode(entry.tokens),
            MetadataKey.renderTokens: try TokenLedger.encode(entry.renderTokens),
            MetadataKey.byteCount: String(entry.byteCount),
        ]
        return try preparePromptCacheSave(
            cache: entry.caches, metadata: metadata, state: entry.state)
    }

    /// Writes a prepared entry to `url`.
    ///
    /// The function writes `<name>.partial.safetensors` beside `url` and then renames it to
    /// `url`, which replaces a file of that name. A failed write removes the partial file.
    ///
    /// - Parameters:
    ///   - input: The input that ``prepare(_:key:)`` made.
    ///   - url: The URL of the file.
    /// - Throws: The error of the safetensors writer, or a `POSIXError` when the rename fails.
    static func write(_ input: PromptCacheSaveInput, to url: URL) throws {
        let partialURL = partialURL(for: url)
        do {
            try writePromptCache(input, url: partialURL)
            try rename(partialURL, to: url)
        } catch {
            removeFile(at: partialURL)
            throw error
        }
    }

    /// Reads the entry of `key` from `url` into the fresh caches of the model.
    ///
    /// - Parameters:
    ///   - url: The URL of the file.
    ///   - key: The session that the entry must belong to.
    ///   - templates: The fresh caches of the model, one for each layer. After a throw their
    ///     state is not defined. Discard them.
    /// - Returns: The entry. Its caches are the templates, or the caches that the load made in
    ///   place of a template.
    /// - Throws: ``ExecutorPromptCacheFileError`` when the file is truncated, names another
    ///   format or another key, holds a ledger that does not decode, or holds caches that do not
    ///   stand at the end of the ledger. The error of the file system or of the prompt cache load
    ///   when the file is missing or is not a prompt cache that fits the templates.
    static func read(
        from url: URL, key: ExecutorPromptCacheKey, templates: [KVCache]
    ) throws -> ExecutorPromptCacheEntry {
        try SafetensorsLength.requireWholeFile(at: url)
        let snapshot = try loadPromptCacheSnapshot(url: url, into: templates)
        let metadata = snapshot.metadata
        guard metadata[MetadataKey.format] == format else {
            throw ExecutorPromptCacheFileError.formatMismatch(metadata[MetadataKey.format])
        }
        guard metadata[MetadataKey.modelID] == key.modelID,
            metadata[MetadataKey.sessionID] == key.sessionID
        else {
            throw ExecutorPromptCacheFileError.keyMismatch
        }
        let tokens = try TokenLedger.decode(metadata, key: MetadataKey.tokens)
        let renderTokens = try TokenLedger.decode(metadata, key: MetadataKey.renderTokens)
        let offsets = snapshot.cache.map(\.offset)
        guard offsets.allSatisfy({ $0 == tokens.count }) else {
            throw ExecutorPromptCacheFileError.offsetMismatch(
                offsets: offsets, ledgerLength: tokens.count)
        }
        return ExecutorPromptCacheEntry(
            caches: snapshot.cache, tokens: tokens, renderTokens: renderTokens,
            state: snapshot.state)
    }

    /// The file name of one entry of `key`.
    ///
    /// A model ID holds a `/`, thus no key string goes into a path. The name is the SHA-256
    /// digest of the model ID, a NUL and the session ID, in lowercase hexadecimal, then the
    /// generation.
    ///
    /// - Parameters:
    ///   - key: The session of the entry.
    ///   - generation: The generation of the entry, which makes each write of one key a new
    ///     name.
    /// - Returns: `<digest>-<generation>.safetensors`.
    static func fileName(for key: ExecutorPromptCacheKey, generation: UInt64) -> String {
        let keyBytes = Data("\(key.modelID)\u{0}\(key.sessionID)".utf8)
        let digest = SHA256.hash(data: keyBytes).map { String(format: "%02x", $0) }.joined()
        return "\(digest)-\(generation).\(fileExtension)"
    }

    /// The partial file of `url`: `<name>.partial.safetensors` in the same directory.
    ///
    /// - Parameter url: The URL of the real file.
    /// - Returns: The URL of the partial file.
    private static func partialURL(for url: URL) -> URL {
        url.deletingPathExtension()
            .appendingPathExtension(partialMarker)
            .appendingPathExtension(fileExtension)
    }

    /// Gives `source` the name `destination` in one step, and replaces a file of that name.
    ///
    /// - Parameters:
    ///   - source: The partial file.
    ///   - destination: The real file.
    /// - Throws: A `POSIXError` when the rename fails.
    private static func rename(_ source: URL, to destination: URL) throws {
        let status = Foundation.rename(
            source.path(percentEncoded: false), destination.path(percentEncoded: false))
        guard status == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    /// Removes a prompt cache file, when it is there: the partial file of a failed write, or a
    /// file that no session needs.
    ///
    /// A failed removal leaves a file that nobody reads, and the caller can do nothing about it.
    /// The error of a failed write is the error that the caller must see. Thus a failed removal
    /// goes to the log and is not thrown.
    ///
    /// - Parameter url: The URL of the file.
    static func removeFile(at url: URL) {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            return
        }
        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            ExecutorPromptCacheLog.info(
                "prompt cache file cannot remove \(url.lastPathComponent): \(error)")
        }
    }
}

/// Encodes a token ledger as file metadata: the base64 text of the little-endian Int32 values.
private enum TokenLedger {

    /// The number of bytes of one token.
    private static let tokenByteCount = MemoryLayout<Int32>.size

    /// Encodes `tokens`.
    ///
    /// A token ID is an index into the vocabulary of the model, thus it never exceeds
    /// `Int32.max`. An ID that does exceed it throws, and is not silently changed.
    ///
    /// - Parameter tokens: The ledger.
    /// - Returns: The base64 text.
    /// - Throws: ``ExecutorPromptCacheFileError/tokenOutOfRange(_:)``.
    static func encode(_ tokens: [Int]) throws -> String {
        let values = try tokens.map { token in
            guard let value = Int32(exactly: token) else {
                throw ExecutorPromptCacheFileError.tokenOutOfRange(token)
            }
            return value.littleEndian
        }
        return values.withUnsafeBytes { Data($0) }.base64EncodedString()
    }

    /// Decodes the ledger under `key` of `metadata`.
    ///
    /// - Parameters:
    ///   - metadata: The user metadata of the file.
    ///   - key: The metadata key of the ledger.
    /// - Returns: The ledger.
    /// - Throws: ``ExecutorPromptCacheFileError/ledgerNotDecodable(_:)`` when the key is
    ///   missing, the text is not base64, or the bytes are not whole Int32 values.
    static func decode(_ metadata: [String: String], key: String) throws -> [Int] {
        guard let text = metadata[key], let bytes = Data(base64Encoded: text),
            bytes.count.isMultiple(of: tokenByteCount)
        else {
            throw ExecutorPromptCacheFileError.ledgerNotDecodable(key)
        }
        return bytes.withUnsafeBytes { raw in
            (0 ..< bytes.count / tokenByteCount).map { index in
                let value = raw.loadUnaligned(
                    fromByteOffset: index * tokenByteCount, as: Int32.self)
                return Int(Int32(littleEndian: value))
            }
        }
    }
}

/// Checks the length of a safetensors file against its header.
///
/// The safetensors load is lazy: it reads the header and makes an array for each tensor, and it
/// reads the tensor bytes only when the array is evaluated. A read past the end of the file at
/// that point stops the process. This check reads the header first, thus a truncated file
/// throws before any array is made.
private enum SafetensorsLength {

    /// The number of bytes of the header length at the start of the file.
    private static let headerLengthByteCount = MemoryLayout<UInt64>.size

    /// The header key of the metadata, which is not a tensor.
    private static let metadataKey = "__metadata__"

    /// The header key of the byte range of a tensor, relative to the end of the header.
    private static let dataOffsetsKey = "data_offsets"

    /// Throws unless the file at `url` holds every byte that its header names.
    ///
    /// - Parameter url: The URL of the safetensors file.
    /// - Throws: ``ExecutorPromptCacheFileError/truncated(byteCount:expectedByteCount:)`` for a
    ///   short file, the error of the file system for a missing file, or the error of the JSON
    ///   parser for a header that is not JSON.
    static func requireWholeFile(at url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let byteCount = try handle.seekToEnd()
        try handle.seek(toOffset: 0)

        let headerStart = UInt64(headerLengthByteCount)
        try require(byteCount, atLeast: headerStart)
        let lengthBytes = try handle.read(upToCount: headerLengthByteCount) ?? Data()
        let headerLength = UInt64(
            littleEndian: lengthBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })

        let dataStart = saturatingSum(headerStart, headerLength)
        try require(byteCount, atLeast: dataStart)
        let header = try handle.read(upToCount: Int(headerLength)) ?? Data()
        try require(byteCount, atLeast: saturatingSum(dataStart, dataLength(of: header)))
    }

    /// Throws unless the file holds `expected` bytes.
    ///
    /// - Parameters:
    ///   - byteCount: The size of the file.
    ///   - expected: The size that the file must have at least.
    /// - Throws: ``ExecutorPromptCacheFileError/truncated(byteCount:expectedByteCount:)``.
    private static func require(_ byteCount: UInt64, atLeast expected: UInt64) throws {
        guard byteCount >= expected else {
            throw ExecutorPromptCacheFileError.truncated(
                byteCount: byteCount, expectedByteCount: expected)
        }
    }

    /// The sum of two sizes, or `UInt64.max` when the sum overflows. A header of a damaged file
    /// can name any size, and no file holds `UInt64.max` bytes, thus the check then throws.
    ///
    /// - Parameters:
    ///   - lhs: The first size.
    ///   - rhs: The second size.
    /// - Returns: The sum, or `UInt64.max`.
    private static func saturatingSum(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? .max : sum
    }

    /// The number of data bytes that a header names: the largest end of a tensor range.
    ///
    /// - Parameter header: The JSON header.
    /// - Returns: The end of the last tensor, relative to the end of the header.
    /// - Throws: The error of the JSON parser.
    private static func dataLength(of header: Data) throws -> UInt64 {
        let entries = try JSONSerialization.jsonObject(with: header) as? [String: Any] ?? [:]
        let ends = entries.compactMap { name, entry -> UInt64? in
            guard name != metadataKey, let fields = entry as? [String: Any],
                let offsets = fields[dataOffsetsKey] as? [NSNumber]
            else {
                return nil
            }
            return offsets.last?.uint64Value
        }
        return ends.max() ?? 0
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
