// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class LoadWeightsTests: XCTestCase {

    func testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeEmptyFile("optiq_vision.safetensors", in: directory)
        try """
        {
          "metadata": { "total_size": 1 },
          "weight_map": {
            "model.norm.weight": "model.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testSafetensorWeightURLsFindsEverySafetensorsFileWhenNoIndexIsPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeEmptyFile("model-00002-of-00002.safetensors", in: directory)
        try writeEmptyFile("config.json", in: directory)
        let subdirectory = directory.appendingPathComponent("extra", isDirectory: true)
        try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
        try writeEmptyFile("mtp.safetensors", in: subdirectory)

        let names = Set(try safetensorWeightURLs(in: directory).map(\.lastPathComponent))

        XCTAssertEqual(
            names,
            [
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                "mtp.safetensors",
            ])
    }

    func testSafetensorWeightURLsGivesAnEmptyListForADirectoryWithNoWeightFiles() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("config.json", in: directory)

        XCTAssertEqual(try safetensorWeightURLs(in: directory), [])
    }

    /// Records what the function does when the directory is absent.
    ///
    /// `FileManager.enumerator(at:includingPropertiesForKeys:)` gives an
    /// enumerator for a directory that does not exist, and that enumerator
    /// gives no item. An absent directory is thus an empty list and not an
    /// error, which is the behaviour every model on the load path sees today.
    func testSafetensorWeightURLsGivesAnEmptyListForADirectoryThatIsAbsent() throws {
        let directory = try makeTemporaryDirectory()
        try FileManager.default.removeItem(at: directory)

        XCTAssertEqual(try safetensorWeightURLs(in: directory), [])
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }
}
