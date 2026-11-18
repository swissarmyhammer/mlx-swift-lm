// Copyright © 2026 Apple Inc.

import Foundation

/// Why the header of a safetensors file cannot be read.
public enum SafetensorsHeaderError: Error, Equatable, Sendable {

    /// The file holds fewer bytes than the length prefix. The value is the number of bytes read.
    case lengthPrefixMissing(byteCount: Int)

    /// The length prefix names a header length that no header has: zero, or more than
    /// ``SafetensorsHeader/maximumHeaderByteCount``.
    case headerLengthOutOfRange(UInt64)

    /// The file ends inside the header.
    case headerTruncated(expectedByteCount: UInt64, byteCount: Int)

    /// The header is JSON, but not a JSON object.
    case headerNotAnObject

    /// The entry of the named tensor names no valid byte range.
    case malformedTensor(String)
}

/// The header of a safetensors file: the byte range of each tensor in the data that follows it.
///
/// A safetensors file starts with a little-endian `UInt64` length prefix, then a JSON header of
/// that length, then the tensor data. Each header entry but `__metadata__` names a tensor and
/// its `data_offsets`, which count from the end of the header.
///
/// Reading the header reads no tensor data. The MLX safetensors load is lazy and reads the bytes
/// of a tensor only when its array is evaluated, thus a caller reads the header to plan the load
/// or to check that the file is whole before any array is made.
public struct SafetensorsHeader: Equatable, Sendable {

    /// One tensor of the file.
    public struct Tensor: Equatable, Sendable {

        /// The name of the tensor.
        public let name: String

        /// The bytes of the tensor, counted from the end of the header.
        public let dataOffsets: Range<UInt64>

        /// Creates a tensor entry.
        ///
        /// - Parameters:
        ///   - name: The name of the tensor.
        ///   - dataOffsets: The bytes of the tensor, counted from the end of the header.
        public init(name: String, dataOffsets: Range<UInt64>) {
            self.name = name
            self.dataOffsets = dataOffsets
        }
    }

    /// The number of bytes of the length prefix at the start of the file.
    public static let lengthPrefixByteCount = MemoryLayout<UInt64>.size

    /// The largest header length ``read(from:)`` accepts. A length prefix past it is not the
    /// prefix of a safetensors file.
    public static let maximumHeaderByteCount: UInt64 = 512 * 1024 * 1024

    /// The header key of the metadata, which is not a tensor.
    private static let metadataKey = "__metadata__"

    /// The header key of the byte range of a tensor.
    private static let dataOffsetsKey = "data_offsets"

    /// The number of values of a byte range: its start and its end.
    private static let dataOffsetsCount = 2

    /// The tensors, in the order of their bytes in the file.
    public let tensors: [Tensor]

    /// The number of data bytes the header names: the largest end of a tensor range, or zero
    /// for a header with no tensor.
    public var dataByteCount: UInt64 {
        tensors.map(\.dataOffsets.upperBound).max() ?? 0
    }

    /// Parses the JSON header of a safetensors file.
    ///
    /// - Parameter json: The header bytes, without the length prefix.
    /// - Throws: The error of the JSON parser for bytes that are not JSON,
    ///   ``SafetensorsHeaderError/headerNotAnObject`` for JSON that is not an object, and
    ///   ``SafetensorsHeaderError/malformedTensor(_:)`` for a tensor entry without a byte range
    ///   or with a range that ends before it starts.
    public init(json: Data) throws {
        guard let entries = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw SafetensorsHeaderError.headerNotAnObject
        }
        tensors = try entries.filter { $0.key != Self.metadataKey }
            .map { name, entry in try Self.tensor(named: name, entry: entry) }
            .sorted { $0.dataOffsets.lowerBound < $1.dataOffsets.lowerBound }
    }

    /// The tensor of one header entry.
    ///
    /// - Parameters:
    ///   - name: The key of the entry.
    ///   - entry: The value of the entry.
    /// - Returns: The tensor.
    /// - Throws: ``SafetensorsHeaderError/malformedTensor(_:)`` when the entry names no valid
    ///   byte range.
    private static func tensor(named name: String, entry: Any) throws -> Tensor {
        guard let fields = entry as? [String: Any],
            let offsets = fields[dataOffsetsKey] as? [NSNumber],
            offsets.count == dataOffsetsCount,
            let start = offsets.first?.uint64Value, let end = offsets.last?.uint64Value,
            start <= end
        else {
            throw SafetensorsHeaderError.malformedTensor(name)
        }
        return Tensor(name: name, dataOffsets: start ..< end)
    }

    /// Reads the length prefix at the current position of `handle`: the length of the JSON
    /// header that follows it.
    ///
    /// - Parameter handle: A handle at the start of a safetensors file.
    /// - Returns: The header length in bytes.
    /// - Throws: ``SafetensorsHeaderError/lengthPrefixMissing(byteCount:)`` when the file ends
    ///   before the prefix does, or the error of the file handle.
    public static func readLength(from handle: FileHandle) throws -> UInt64 {
        let prefix = try handle.read(upToCount: lengthPrefixByteCount) ?? Data()
        guard prefix.count == lengthPrefixByteCount else {
            throw SafetensorsHeaderError.lengthPrefixMissing(byteCount: prefix.count)
        }
        return UInt64(littleEndian: prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
    }

    /// Reads the length prefix and the JSON header at the current position of `handle`.
    ///
    /// - Parameter handle: A handle at the start of a safetensors file.
    /// - Returns: The header.
    /// - Throws: ``SafetensorsHeaderError`` when the prefix or the header is missing, cut or
    ///   malformed, the error of the JSON parser, or the error of the file handle.
    public static func read(from handle: FileHandle) throws -> SafetensorsHeader {
        let length = try readLength(from: handle)
        guard length > 0, length <= maximumHeaderByteCount else {
            throw SafetensorsHeaderError.headerLengthOutOfRange(length)
        }
        let json = try handle.read(upToCount: Int(length)) ?? Data()
        guard json.count == length else {
            throw SafetensorsHeaderError.headerTruncated(
                expectedByteCount: length, byteCount: json.count)
        }
        return try SafetensorsHeader(json: json)
    }
}
