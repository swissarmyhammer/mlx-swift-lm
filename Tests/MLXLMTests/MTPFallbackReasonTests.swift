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

    /// The one-token prompt of the staging probe's decodes.
    private static let probePromptToken: Int32 = 7

    /// The prompt of the staging probe's decodes.
    private static var probeInput: LMInput { LMInput(tokens: MLXArray([probePromptToken])) }

    /// A cache that the staging probe refuses before the prefill gives the plain iterator, which
    /// carries an ``SpeculativeDecodingFallbackReason/unsupportedCache`` fallback that names
    /// the target and the drafter.
    @Test("a cache that can neither verify nor stage a round decodes plainly and says why")
    func aCacheThatCannotStageARoundReportsItsFallback() throws {
        let probe = MTPStagingProbe()
        probe.cache.refusesSpeculativeStaging = true
        let iterator = try MTPSpeculation.makeTokenIterator(
            speculation: MTPSpeculation(
                drafter: QwenStyleMTPDrafter(probe: probe), blockSize: Self.probeBlockSize,
                minimumDraftProbability: 0),
            input: Self.probeInput, cache: nil, state: nil,
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            model: MTPStateEmittingTarget(probe: probe), components: .init())

        let fallback = try #require((iterator as? TokenIterator)?.speculationFallback)
        #expect(fallback.reason == .unsupportedCache)
        #expect(fallback.statedReason.contains("MTPStateEmittingTarget"))
        #expect(fallback.statedReason.contains("QwenStyleMTPDrafter"))
        #expect(probe.draftCount == 0)
    }

    /// A cache that refuses to stage the first round of the stream turns the stream to
    /// passthrough before any draft, which reports an
    /// ``SpeculativeDecodingFallbackReason/unsupportedCache`` fallback.
    @Test("a cache that stops staging rounds before the first round reports an unsupported cache")
    func aCacheThatStopsStagingReportsAnUnsupportedCache() throws {
        let probe = MTPStagingProbe()
        var iterator = try MTPSpeculativeTokenIterator(
            input: Self.probeInput,
            mainModel: MTPStateEmittingTarget(probe: probe),
            drafter: QwenStyleMTPDrafter(probe: probe),
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            blockSize: Self.probeBlockSize)
        probe.cache.refusesSpeculativeStaging = true
        while iterator.next() != nil {}

        #expect(iterator.proposedDraftTokens == 0)
        #expect(iterator.passthroughFallbackReason == .unsupportedCache)
        #expect(iterator.passthroughReason?.contains("cannot stage") == true)
    }

    /// A stream that drafted, then decoded its tail plainly because its cache stopped staging
    /// rounds, did speculate: it reports no fallback, and its passthrough reason alone states
    /// why its tail decoded plainly.
    @Test("a stream that drafted before its tail decoded plainly reports no fallback")
    func aStreamThatDraftedReportsNoFallback() throws {
        let probe = MTPStagingProbe()
        var iterator = try MTPSpeculativeTokenIterator(
            input: Self.probeInput,
            mainModel: MTPStateEmittingTarget(probe: probe),
            drafter: QwenStyleMTPDrafter(probe: probe),
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            blockSize: Self.probeBlockSize)
        while iterator.proposedDraftTokens == 0, iterator.next() != nil {}
        #expect(iterator.proposedDraftTokens > 0)
        probe.cache.refusesSpeculativeStaging = true
        while iterator.next() != nil {}

        #expect(iterator.passthroughReason?.contains("cannot stage") == true)
        #expect(iterator.passthroughFallbackReason == nil)
    }

    /// A sampled stream that asks for log probabilities stands down before its first round and
    /// reports an ``SpeculativeDecodingFallbackReason/unsupportedSampling`` fallback with no
    /// draft.
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
        let completion = try #require(await completionInfo(of: stream, task: task))
        #expect(completion.proposedDraftTokens == 0)
        #expect(completion.speculativeDecodingFallbackReason == .unsupportedSampling)
        #expect(completion.passthroughReason == MTPSpeculativeTokenIterator.logProbabilitiesReason)
    }

    /// A greedy stream that speculates to its end drafts and reports no fallback.
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
        let completion = try #require(await completionInfo(of: stream, task: task))
        #expect((completion.proposedDraftTokens ?? 0) > 0)
        #expect(completion.speculativeDecodingFallbackReason == nil)
    }
}
