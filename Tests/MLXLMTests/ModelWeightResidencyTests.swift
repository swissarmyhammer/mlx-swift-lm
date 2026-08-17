// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// A model that holds no parameter, for a load that must stop before it makes
/// the first weight buffer.
private final class WeightlessModel: Module, BaseLanguageModel {}

/// The Metal wired limit that the weight load raises before it allocates a
/// weight buffer.
///
/// The tests run one at a time, because each one reads the wired-limit state of
/// the process.
@Suite(.serialized)
struct ModelWeightResidencyTests {

    /// The size of the weight file the ordering test writes. It is large enough
    /// to stand above the request of any other test of this bundle, and small
    /// enough to write inside a test.
    private static let weightFileBytes = 8 * 1_024 * 1_024

    /// The number of bytes the ordering test expects the load to ask for.
    private static let expectedRequestBytes = 10 * 1_024 * 1_024

    // MARK: Sizing

    /// The request covers the weights and a working set above them, because the
    /// wired-memory manager takes the maximum across policy groups and thus a
    /// ticket taken later at generation time adds nothing to this one.
    @Test func theRequestCoversTheWeightsAndAWorkingSetAboveThem() {
        #expect(
            ModelWeightResidency.limitBytes(forWeightBytes: Self.weightFileBytes)
                == Self.expectedRequestBytes)
    }

    /// A weight file that is a symbolic link counts the size of its target.
    ///
    /// A `huggingface_hub` snapshot holds a symbolic link to a blob for each
    /// weight file. Measured on a snapshot of
    /// `mlx-community/Llama-3.2-1B-Instruct-4bit` before the path was resolved:
    /// the request covered 76 bytes of link in place of 745,270,382 bytes of
    /// weights, thus the raise did nothing at all.
    @Test func aSymbolicLinkCountsTheSizeOfItsTarget() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-weight-link-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let blob = directory.appendingPathComponent("blob")
        try Data(count: Self.weightFileBytes).write(to: blob)
        let link = directory.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: blob)

        #expect(MLXLMCommon.weightFileBytes(of: [link]) == Self.weightFileBytes)
    }

    // MARK: The limit never falls

    /// A smaller request never lowers the limit. A limit that falls empties the
    /// Metal residency set, thus every weight buffer costs its full size again
    /// at each decode step.
    @Test func aSmallerRequestNeverLowersTheLimit() async {
        let residency = ModelWeightResidency()
        _ = await residency.raise(toCoverWeightBytes: 1_024 * 1_024 * 1_024)
        let afterTheLargeRequest = await residency.highWaterMarkBytes

        _ = await residency.raise(toCoverWeightBytes: 1_024 * 1_024)
        let afterTheSmallRequest = await residency.highWaterMarkBytes

        #expect(afterTheSmallRequest == afterTheLargeRequest)
    }

    // MARK: The order of the raise and the allocation

    /// The load raises the limit BEFORE it makes the first weight buffer.
    ///
    /// A buffer joins the Metal residency set when it is made, thus a limit
    /// raised after the load changes nothing. The checkpoint of this test names
    /// two weight files and holds only the second, thus the load stops at the
    /// first file and allocates nothing at all. A limit that stands after that
    /// failure can only have been raised before the load.
    @Test func theLoadRaisesTheLimitBeforeItAllocatesAWeight() async throws {
        let directory = try makeCheckpointWhoseFirstWeightFileIsAbsent()
        defer { try? FileManager.default.removeItem(at: directory) }

        await #expect(throws: (any Error).self) {
            try await loadWeights(modelDirectory: directory, model: WeightlessModel())
        }

        let markAfterTheLoad = await ModelWeightResidency.shared.highWaterMarkBytes
        #expect(markAfterTheLoad >= Self.expectedRequestBytes)
    }

    /// Writes a checkpoint whose safetensors index names two weight files and
    /// whose directory holds only the second one.
    ///
    /// - Returns: The directory of the checkpoint. The caller removes it.
    /// - Throws: An error when the directory or a file cannot be written.
    private func makeCheckpointWhoseFirstWeightFileIsAbsent() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-weight-residency-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let index = """
            {"weight_map": {"first": "a-absent.safetensors", "second": "b-present.safetensors"}}
            """
        try Data(index.utf8)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try Data(count: Self.weightFileBytes)
            .write(to: directory.appendingPathComponent("b-present.safetensors"))
        return directory
    }
}
