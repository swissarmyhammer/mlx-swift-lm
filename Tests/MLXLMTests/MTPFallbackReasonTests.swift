// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

/// A generation that was given a drafter but decodes without drafting reports why: the
/// completion information names the kind of the fallback, whether the stream never speculated
/// or stood down before its first round.
@Suite(.serialized)
struct MTPFallbackReasonTests {

    /// The tokens of the prompt of each decode.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode generates.
    private static let generatedTokenCount = 12

    /// The log probabilities each token of the sampled stream reports.
    private static let logProbabilityCount = 2

    /// The tokens of each round of the staging probe.
    private static let probeBlockSize = 2

    @Test("a cache that can neither verify nor stage a round decodes plainly and says why")
    func aCacheThatCannotStageARoundReportsItsFallback() throws {
        let probe = MTPStagingProbe()
        probe.cache.refusesSpeculativeStaging = true
        let iterator = try MTPSpeculation.makeTokenIterator(
            speculation: MTPSpeculation(
                drafter: QwenStyleMTPDrafter(probe: probe), blockSize: Self.probeBlockSize,
                minimumDraftProbability: 0),
            input: LMInput(tokens: MLXArray([Int32(7)])), cache: nil, state: nil,
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            model: MTPStateEmittingTarget(probe: probe), components: .init())

        let fallback = try #require((iterator as? TokenIterator)?.speculationFallback)
        #expect(fallback.reason == .unsupportedCache)
        #expect(fallback.statedReason.contains("MTPStateEmittingTarget"))
        #expect(fallback.statedReason.contains("QwenStyleMTPDrafter"))
        #expect(probe.draftCount == 0)
    }

    @Test("a stream that stands down before its first round reports the kind of its fallback")
    func aStreamThatStandsDownReportsItsFallback() async throws {
        let fixture = try Qwen35MTPFixture()
        var parameters = GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0)
        parameters.logProbabilities = Self.logProbabilityCount
        let (stream, task) = try generateTaskRecordingTokens(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            parameters: parameters, context: fixture.makeModelContext(),
            speculation: MTPSpeculation(
                drafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize,
                minimumDraftProbability: 0))
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        _ = await task.value

        let completion = try #require(info)
        #expect(completion.proposedDraftTokens == 0)
        #expect(completion.speculativeDecodingFallbackReason == .unsupportedSampling)
        #expect(completion.passthroughReason == MTPSpeculativeTokenIterator.logProbabilitiesReason)
    }

    @Test("a stream that speculates reports no fallback")
    func aSpeculatingStreamReportsNoFallback() async throws {
        let fixture = try Qwen35MTPFixture()
        let (stream, task) = try generateTaskRecordingTokens(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            context: fixture.makeModelContext(),
            speculation: MTPSpeculation(
                drafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize,
                minimumDraftProbability: 0))
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        _ = await task.value

        let completion = try #require(info)
        #expect((completion.proposedDraftTokens ?? 0) > 0)
        #expect(completion.speculativeDecodingFallbackReason == nil)
    }
}
