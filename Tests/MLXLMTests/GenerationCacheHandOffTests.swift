// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing
import os

/// Holds the caches a generation hands over when it ends.
///
/// The generation calls back on its own thread before its task returns, and the test reads after
/// it awaited that task, thus a lock holds the caches.
private final class GenerationCacheRecorder: Sendable {

    /// The caches the generation handed over, or nil.
    private let received = OSAllocatedUnfairLock<SendableBox<[KVCache]>?>(initialState: nil)

    /// The hand-off the generation gives its caches to.
    var handOff: GenerationHandOff {
        GenerationHandOff(generationCache: { caches in self.received.withLock { $0 = caches } })
    }

    /// The caches the generation handed over, or nil when it handed none.
    func takeCaches() -> [KVCache]? {
        received.withLock { $0.take() }?.consume()
    }
}

/// The package generation functions that the FoundationModels executor runs hand over, when the
/// generation ends, the caches it last wrote into. A KV-cache configuration replaces each cache it
/// compresses while the generation runs, thus these are not the caches the caller handed in.
@Suite(.serialized)
struct GenerationCacheHandOffTests {

    /// The tokens of the prompt that each generation of these tests starts from.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each generation of these tests generates.
    private static let generatedTokenCount = 8

    /// A full-attention head width the affine group size divides, thus the KV-cache configuration
    /// compresses the attention caches of the fixture.
    private static let quantizableHeadDimension = AffineKVCacheConfiguration.eightBit.groupSize

    /// Greedy parameters under affine 8-bit KV-cache compression.
    private static let eightBitParameters = GenerateParameters(
        maxTokens: generatedTokenCount,
        kvCache: KVCacheConfiguration(strategy: .affine(.eightBit)), temperature: 0)

    /// The input of each generation.
    private static var input: LMInput {
        LMInput(tokens: MLXArray(prompt.map(Int32.init)))
    }

    /// The speculation of the fixture's own drafter.
    private func speculation(of fixture: Qwen35MTPFixture) -> MTPSpeculation {
        MTPSpeculation(
            drafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize,
            minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
    }

    /// Runs `generateTaskRecordingTokens` over a fixture whose attention caches compress, and
    /// gives the caches the generation handed over.
    ///
    /// - Parameter speculates: whether the generation speculates with the fixture's drafter.
    /// - Returns: the handed-over caches, or nil when the callback was not called.
    /// - Throws: the error of the fixture or of the generation.
    private func recordingRunHandOff(speculates: Bool) async throws -> [KVCache]? {
        let fixture = try Qwen35MTPFixture(headDimension: Self.quantizableHeadDimension)
        let context = fixture.makeModelContext()
        let recorder = GenerationCacheRecorder()
        let (stream, task) = try generateTaskRecordingTokens(
            input: Self.input, cache: try context.model.newCache(parameters: nil),
            parameters: Self.eightBitParameters, context: context,
            speculation: speculates ? speculation(of: fixture) : nil,
            handOff: recorder.handOff)
        for await _ in stream {}
        _ = await task.value
        return recorder.takeCaches()
    }

    @Test
    func aPlainGenerationHandsOverTheCachesItCompressed() async throws {
        let caches = try #require(try await recordingRunHandOff(speculates: false))

        #expect(caches.contains { $0 is QuantizedKVCache })
    }

    @Test
    func anMTPGenerationHandsOverTheCachesItCompressed() async throws {
        let caches = try #require(try await recordingRunHandOff(speculates: true))

        #expect(caches.contains { $0 is QuantizedKVCache })
    }

    @Test
    func aProtocolGenerationHandsOverTheCachesItCompressed() async throws {
        let fixture = try Qwen35MTPFixture(headDimension: Self.quantizableHeadDimension)
        let context = fixture.makeModelContext()
        let recorder = GenerationCacheRecorder()
        let (stream, task) = try generateProtocolTokensTask(
            input: Self.input, cache: try context.model.newCache(parameters: nil),
            parameters: Self.eightBitParameters, context: context, decoder: nil,
            handOff: recorder.handOff)
        for await _ in stream {}
        _ = await task.value
        let caches = try #require(recorder.takeCaches())

        #expect(caches.contains { $0 is QuantizedKVCache })
    }
}
