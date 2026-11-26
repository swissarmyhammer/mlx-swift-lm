// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

/// Every model family that shards has a tiny checkpoint on random weights, which
/// `ShardCheck tiny-checkpoint` writes so `scripts/shard-check.sh --tiny` checks the family with
/// no download.
@Suite("A tiny shard checkpoint loads for each shardable family", .serialized)
struct TinyShardCheckpointTests {

    /// Writes the checkpoint of `family` to a new temporary directory, runs `body` on the
    /// directory, and removes it.
    private static func withCheckpoint<Result>(
        of family: String, _ body: (URL) async throws -> Result
    ) async throws -> Result {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "tiny-shard-checkpoint-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try await TinyShardCheckpoint.write(family: family, to: directory)
        return try await body(directory)
    }

    /// The model of the checkpoint in `directory`, loaded as a worker of a group of one loads it.
    private static func load(_ directory: URL) async throws -> any LanguageModel {
        try await loadShardedModelContext(
            from: directory, using: PseudoWordTokenizerLoader(), group: MLXDistributed.initialize()
        ).model
    }

    @Test(
        "the checkpoint of each shardable family loads as a shardable model",
        arguments: RemoteModelExecutor.shardableModelFamilies)
    func checkpointLoadsAsAShardableModel(family: String) async throws {
        let model = try await Self.withCheckpoint(of: family, Self.load)

        #expect(model is TensorParallelShardable)
    }

    @Test(
        "the vocabulary of each tiny checkpoint holds the tiny prompt",
        arguments: RemoteModelExecutor.shardableModelFamilies)
    func vocabularyHoldsThePrompt(family: String) async throws {
        let model = try await Self.withCheckpoint(of: family, Self.load)
        let tokens = MLXArray(TinyShardCheckpoint.prompt).expandedDimensions(axis: 0)

        let logits = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits

        #expect(logits.dim(-1) > TinyShardCheckpoint.prompt.max() ?? 0)
    }

    @Test("a family that does not shard has no tiny checkpoint")
    func familyThatDoesNotShardHasNoCheckpoint() async throws {
        await #expect(throws: TinyShardCheckpoint.Failure.unknownFamily("llama")) {
            try await Self.withCheckpoint(of: "llama") { _ in }
        }
    }
}
