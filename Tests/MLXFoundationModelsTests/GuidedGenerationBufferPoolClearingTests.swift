// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import MLXLMCommon
import MLXLMTestSupport
import Testing
import os

@testable import MLXGuidedGeneration

/// The token indices at which a generation cleared the MLX pool, so that a test does not read the
/// process-wide pool.
private struct ClearedTokenLog: Sendable {
    private let indices = OSAllocatedUnfairLock(initialState: [Int]())

    /// The index of the token at hand at each clear, in order.
    var clearedTokenIndices: [Int] { indices.withLock { $0 } }

    /// Makes the cadence that logs each clear, with the tokens `sink` holds at that clear.
    ///
    /// - Parameter sink: The sink of the generation, which holds each token before it is emitted.
    /// - Returns: The cadence.
    func makeClearing(reading sink: GuidedGenerationDiagnosticSink) -> GenerationBufferPoolClearing
    {
        GenerationBufferPoolClearing { [indices] in
            let tokenCount = sink.sampledTokenIDs.count + sink.fastForwardTokenIDs.count
            indices.withLock { $0.append(tokenCount - 1) }
        }
    }
}

/// A long guided generation clears the MLX pool every 256 emitted tokens on both decode paths, as
/// the token iterators do, and counts decided, accepted and forced tokens alike.
@Suite(.serialized)
struct GuidedGenerationBufferPoolClearingTests {

    /// The prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The token budget of each generation: past two whole intervals of the cadence.
    private static let tokenBudget = 520

    /// The block size of the speculating generation.
    private static let blockSize = 4

    /// A schema far longer than ``tokenBudget``, whose fixed keys force tokens in runs: a clear
    /// checked only at a decision misses a multiple that falls inside a run.
    private static let longSchema = """
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": { "type": "string", "enum": ["sightseeing", "foodAndDining"] },
                    "title": { "type": "string", "maxLength": 8 }
                },
                "required": ["type", "title"],
                "additionalProperties": false
            },
            "minItems": 64,
            "maxItems": 64
        }
        """

    /// Runs a long generation, and gives the indices of its clears and what it recorded.
    private static func generate(
        blockSize: Int?
    ) throws -> (clearedTokenIndices: [Int], sink: GuidedGenerationDiagnosticSink) {
        let log = ClearedTokenLog()
        let sink = GuidedGenerationDiagnosticSink()
        _ = try GuidedFixtureRun.generate(
            schema: longSchema, prompt: prompt, fixture: try Qwen35MTPFixture(),
            blockSize: blockSize, tokenBudget: tokenBudget, sink: sink,
            bufferPoolClearing: log.makeClearing(reading: sink))
        return (log.clearedTokenIndices, sink)
    }

    @Test
    func plainDecodingClearsEveryIntervalOfEmittedTokens() throws {
        let run = try Self.generate(blockSize: nil)

        #expect(run.clearedTokenIndices == [0, 256, 512])
    }

    @Test
    func speculatingDecodingClearsEveryIntervalOfEmittedTokens() throws {
        let run = try Self.generate(blockSize: Self.blockSize)

        #expect(run.sink.draftedTokenCount > 0)
        #expect(run.sink.speculationFallbacks.isEmpty)
        #expect(run.clearedTokenIndices == [0, 256, 512])
    }
}

#endif
