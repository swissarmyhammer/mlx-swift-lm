// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// Tests for ``SafetensorsHeader``, which reads the length prefix and the JSON header of a
/// safetensors file.
///
/// No array is made. Each file is a few bytes written by hand.
@Suite("A safetensors header names the byte range of each tensor")
struct SafetensorsHeaderTests {

    /// A header of two tensors, written out of file order, and the metadata entry.
    private static let twoTensorJSON = Data(
        """
        {"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},\
        "__metadata__":{"format":"test"},\
        "a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
        """.utf8)

    /// The end of the last tensor of ``twoTensorJSON``.
    private static let twoTensorDataByteCount: UInt64 = 16

    /// A header whose tensor names no byte range.
    private static let noOffsetsJSON = Data(#"{"a":{"dtype":"F32","shape":[2]}}"#.utf8)

    /// A header whose tensor ends before it begins.
    private static let reversedOffsetsJSON = Data(
        #"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#.utf8)

    /// A header length that the length prefix of a test file names.
    private static let headerLength: UInt64 = 1234

    /// A file shorter than the length prefix.
    private static let shortFileBytes = Data([1, 2, 3])

    /// Writes `bytes` to a new temporary file, opens it for reading, and runs `body` on the
    /// handle. The file is closed and removed after `body` returns or throws.
    ///
    /// - Parameters:
    ///   - bytes: The bytes of the file.
    ///   - body: Reads the file.
    /// - Returns: What `body` returns.
    private func withFile<Result>(
        holding bytes: Data, _ body: (FileHandle) throws -> Result
    ) throws -> Result {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SafetensorsHeaderTests-\(UUID().uuidString).safetensors")
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try body(handle)
    }

    /// The little-endian length prefix of `length`.
    ///
    /// - Parameter length: The header length.
    /// - Returns: The eight bytes of the prefix.
    private func lengthPrefix(_ length: UInt64) -> Data {
        withUnsafeBytes(of: length.littleEndian) { Data($0) }
    }

    @Test("the tensors come back in file order without the metadata entry")
    func theTensorsComeBackInFileOrder() throws {
        let header = try SafetensorsHeader(json: Self.twoTensorJSON)

        #expect(
            header.tensors == [
                SafetensorsHeader.Tensor(name: "a", dataOffsets: 0 ..< 8),
                SafetensorsHeader.Tensor(name: "b", dataOffsets: 8 ..< 16),
            ])
    }

    @Test("the data byte count is the end of the last tensor")
    func theDataByteCountIsTheEndOfTheLastTensor() throws {
        let header = try SafetensorsHeader(json: Self.twoTensorJSON)

        #expect(header.dataByteCount == Self.twoTensorDataByteCount)
    }

    @Test("a tensor that names no byte range throws .malformedTensor")
    func aTensorWithNoByteRangeThrows() {
        #expect(throws: SafetensorsHeaderError.malformedTensor("a")) {
            try SafetensorsHeader(json: Self.noOffsetsJSON)
        }
    }

    @Test("a tensor that ends before it begins throws .malformedTensor")
    func aTensorThatEndsBeforeItBeginsThrows() {
        #expect(throws: SafetensorsHeaderError.malformedTensor("a")) {
            try SafetensorsHeader(json: Self.reversedOffsetsJSON)
        }
    }

    @Test("the length prefix reads as a little-endian integer")
    func theLengthPrefixReadsLittleEndian() throws {
        let length = try withFile(holding: lengthPrefix(Self.headerLength)) {
            try SafetensorsHeader.readLength(from: $0)
        }

        #expect(length == Self.headerLength)
    }

    @Test("a file shorter than the length prefix throws .lengthPrefixMissing")
    func aFileShorterThanThePrefixThrows() throws {
        #expect(
            throws: SafetensorsHeaderError.lengthPrefixMissing(
                byteCount: Self.shortFileBytes.count)
        ) {
            try withFile(holding: Self.shortFileBytes) {
                try SafetensorsHeader.readLength(from: $0)
            }
        }
    }

    @Test("a read of a whole header gives its tensors")
    func aReadOfAWholeHeaderGivesItsTensors() throws {
        let header = try withFile(
            holding: lengthPrefix(UInt64(Self.twoTensorJSON.count)) + Self.twoTensorJSON
        ) {
            try SafetensorsHeader.read(from: $0)
        }

        #expect(try header == SafetensorsHeader(json: Self.twoTensorJSON))
    }

    @Test("a read of a header shorter than its length prefix says throws .headerTruncated")
    func aReadOfATruncatedHeaderThrows() throws {
        #expect(
            throws: SafetensorsHeaderError.headerTruncated(
                expectedByteCount: Self.headerLength, byteCount: Self.noOffsetsJSON.count)
        ) {
            try withFile(holding: lengthPrefix(Self.headerLength) + Self.noOffsetsJSON) {
                try SafetensorsHeader.read(from: $0)
            }
        }
    }
}
