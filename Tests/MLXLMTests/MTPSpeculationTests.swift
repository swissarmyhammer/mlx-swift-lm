// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXVLM
import Testing

/// The iterator choice of the package generation functions that the FoundationModels executor
/// runs: MTP speculation with a compatible drafter, plain decode with no drafter or an
/// incompatible one.
@Suite(.serialized)
struct MTPSpeculationTests {

    /// The tokens of the prompt that each decode of these tests starts from.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode of these tests generates.
    private static let generatedTokenCount = 16

    /// The index in the greedy decode of a token that MTP with the fixture returns as the
    /// final token of a round, which no forward pass has fed.
    private static let bonusSlotStopIndex = 3

    /// The index in the greedy decode of a token that MTP with the fixture returns as an
    /// accepted draft, which the verify pass of its round fed.
    private static let draftSlotStopIndex = 4

    /// What one generation of a package generation function left.
    private struct Run {

        /// The tokens the generation recorded, in order.
        let tokens: [Int]

        /// The completion information of the generation.
        let info: GenerateCompletionInfo

        /// What the drafter-state completion callback received, or `nil` when it was not called.
        let drafterContinuation: MTPDrafterContinuation??

        /// The offset of the main cache after the generation.
        let cacheOffset: Int
    }

    /// The speculation of the fixture's own drafter.
    private func speculation(of fixture: Qwen35MTPFixture) -> MTPSpeculation {
        MTPSpeculation(
            drafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize,
            minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
    }

    /// Runs `generateTaskRecordingTokens` over the fixture's target from a fresh cache.
    private func recordingRun(
        _ fixture: Qwen35MTPFixture, speculation: MTPSpeculation?, eosTokenIds: Set<Int> = []
    ) async throws -> Run {
        let context = fixture.makeModelContext(eosTokenIds: eosTokenIds)
        let cache = try context.model.newCache(parameters: nil)
        // The generation calls back on its own thread; the stream hands the value to this
        // task, and its first element is what the callback received.
        let (continuations, callback) = AsyncStream.makeStream(of: MTPDrafterContinuation?.self)
        let (stream, task) = try generateTaskRecordingTokens(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))), cache: cache,
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            context: context, speculation: speculation,
            drafterContinuation: { callback.yield($0) })
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        let tokens = await task.value
        callback.finish()
        var received = continuations.makeAsyncIterator()
        return Run(
            tokens: tokens, info: try #require(info), drafterContinuation: await received.next(),
            cacheOffset: cache[0].offset)
    }

    /// Runs `generateProtocolTokensTask` over the fixture's target from a fresh cache.
    private func protocolRun(
        _ fixture: Qwen35MTPFixture, speculation: MTPSpeculation?
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        let context = fixture.makeModelContext()
        let (stream, _) = try generateProtocolTokensTask(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            context: context, decoder: nil, speculation: speculation)
        return try await Qwen35MTPFixture.collect(stream)
    }

    /// Runs the package `generateTokensTask` over the fixture's target from a fresh cache.
    private func tokensRun(
        _ fixture: Qwen35MTPFixture, speculation: MTPSpeculation?
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        let (stream, _) = try generateTokensTask(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            context: fixture.makeModelContext(), speculation: speculation)
        return try await Qwen35MTPFixture.collect(stream)
    }

    @Test
    func recordingTokensSpeculatesWithACompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await recordingRun(fixture, speculation: nil)
        let speculative = try await recordingRun(fixture, speculation: speculation(of: fixture))

        #expect(plain.tokens.count == Self.generatedTokenCount)
        #expect(speculative.tokens == plain.tokens)
        #expect(plain.info.proposedDraftTokens == nil)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculative.info.passthroughReason == nil)
    }

    @Test
    func recordingTokensDecodesPlainlyWithAnIncompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()
        let visionLanguageDrafter = try Qwen35VLMMTPFixture().drafter
        #expect(!visionLanguageDrafter.isCompatible(with: fixture.target))

        let plain = try await recordingRun(fixture, speculation: nil)
        let run = try await recordingRun(
            fixture,
            speculation: MTPSpeculation(
                drafter: visionLanguageDrafter, blockSize: Qwen35MTPFixture.blockSize,
                minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability
            ))

        #expect(run.tokens == plain.tokens)
        #expect(run.info.proposedDraftTokens == nil)
    }

    @Test
    func protocolTokensSpeculatesWithACompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await protocolRun(fixture, speculation: nil)
        let speculative = try await protocolRun(fixture, speculation: speculation(of: fixture))

        #expect(plain.tokens.count == Self.generatedTokenCount)
        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }

    @Test
    func tokensTaskSpeculatesWithACompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await tokensRun(fixture, speculation: nil)
        let speculative = try await tokensRun(fixture, speculation: speculation(of: fixture))

        #expect(plain.tokens.count == Self.generatedTokenCount)
        #expect(speculative.tokens == plain.tokens)
        #expect(plain.info.proposedDraftTokens == nil)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }

    @Test
    func theCompletionCallbackGetsTheDrafterStateAlignedWithTheCache() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await recordingRun(fixture, speculation: nil)
        let speculative = try await recordingRun(fixture, speculation: speculation(of: fixture))

        let plainContinuation = try #require(plain.drafterContinuation)
        #expect(plainContinuation == nil)
        let received = try #require(speculative.drafterContinuation)
        let continuation = try #require(received)
        #expect(
            continuation.isAligned(
                withMainTokenCount: speculative.cacheOffset, drafterID: fixture.drafter.drafterID))
    }

    @Test(arguments: [bonusSlotStopIndex, draftSlotStopIndex])
    func theStopTokenIsReportedWhetherOrNotAForwardFedIt(stopIndex: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let decode = try await recordingRun(fixture, speculation: nil)
        let stopToken = decode.tokens[stopIndex]
        #expect(decode.tokens.firstIndex(of: stopToken) == stopIndex)

        let plain = try await recordingRun(fixture, speculation: nil, eosTokenIds: [stopToken])
        let speculative = try await recordingRun(
            fixture, speculation: speculation(of: fixture), eosTokenIds: [stopToken])

        #expect(speculative.tokens == plain.tokens)
        #expect(plain.info.generationTokenCount == stopIndex)
        #expect(speculative.info.generationTokenCount == stopIndex)
        #expect(plain.info.stopTokenFedToCache == stopToken)
        #expect(
            speculative.info.stopTokenFedToCache
                == (stopIndex == Self.draftSlotStopIndex ? stopToken : nil))
        #expect(plain.info.discardedStopToken == stopToken)
        #expect(speculative.info.discardedStopToken == stopToken)
    }

    @Test
    func aGenerationThatRunsToItsBudgetDiscardsNoStopToken() async throws {
        let fixture = try Qwen35MTPFixture()

        let speculative = try await recordingRun(fixture, speculation: speculation(of: fixture))

        #expect(speculative.info.discardedStopToken == nil)
    }
}
