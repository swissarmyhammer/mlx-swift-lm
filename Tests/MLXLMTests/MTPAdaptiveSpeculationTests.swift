// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// An MTP stream whose adaptive guard lowers its block size, pauses speculation and resumes it
/// writes the tokens of plain decode and of the stream without a guard, keeps the drafter one
/// entry for each token the main cache holds, and leaves the main cache and the drafter
/// continuation that a prefill of its cached tokens leaves.
@Suite(.serialized)
struct MTPAdaptiveSpeculationTests {

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The tokens each decode generates: enough for several pauses of ``losingPolicy``.
    private static let generatedTokenCount = 40

    /// The tokens each second call generates.
    private static let continuedTokenCount = 8

    /// The tokens each second call adds after the tokens of the first.
    private static let followUp = [7, 21, 40]

    /// The block sizes of the decodes: the smallest, which pauses at once, and the shipped one,
    /// which lowers its block size first.
    private static let blockSizes = [
        MTPAdaptiveController.minimumBlockSize, Qwen35MTPFixture.blockSize,
    ]

    /// The policy of each guarded decode: every judged round loses to plain decode.
    private static let losingPolicy = MTPAdaptivePolicy.losingEveryRound

    /// The prompt tokens a second chat turn can compute: its one-token message, and the first
    /// turn's last token when the turn ended before a forward fed it.
    private static let secondTurnPromptTokenCounts = 1 ... 2

    /// The temperature of the sampled decode.
    private static let temperature: Float = 0.7

    /// The seed of the sampled decode.
    private static let seed: UInt64 = 7

    /// A greedy MTP iterator of the fixture's drafter over `input` with `policy`.
    private func iterator(
        _ fixture: Qwen35MTPFixture, input: [Int], blockSize: Int, policy: MTPAdaptivePolicy,
        maxTokens: Int = generatedTokenCount, cache: [KVCache]? = nil,
        continuation: MTPDrafterContinuation? = nil
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(input.map(Int32.init))), mainModel: fixture.target,
            drafter: fixture.drafter, mainCache: cache, mtpContinuation: continuation,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            blockSize: blockSize, adaptivePolicy: policy)
    }

    @Test(
        "a greedy decode whose guard pauses speculation equals plain decode and the unguarded decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, blockSizes)
    func pausedDecodeEqualsPlainDecode(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        var unguarded = try iterator(
            fixture, input: Self.prompt, blockSize: blockSize, policy: .disabled)
        var guarded = try iterator(
            fixture, input: Self.prompt, blockSize: blockSize, policy: Self.losingPolicy)

        let unguardedTokens = drain(&unguarded)
        let guardedTokens = drain(&guarded)
        let telemetry = try #require(guarded.speculativeDecodingTelemetry)

        #expect(guardedTokens == plain.tokens)
        #expect(unguardedTokens == plain.tokens)
        #expect(guarded.passthroughReason == nil)
        #expect(telemetry.adaptivePauseCount > 1)
        #expect(telemetry.adaptiveProbeCount == 1)
        #expect(telemetry.adaptivePlainTokenCount > 0)
        #expect(telemetry.adaptiveBlockSize == MTPAdaptiveController.minimumBlockSize)
        #expect(telemetry.adaptiveBlockSizeChangeCount == blockSize - 2)
        #expect(telemetry.emittedTokenCount == Self.generatedTokenCount)
        #expect(unguarded.speculativeDecodingTelemetry?.adaptiveBlockSize == nil)
    }

    @Test(
        "the drafter holds one entry for each token the main cache holds after every round",
        arguments: blockSizes)
    func drafterFollowsTheMainCacheAcrossPauses(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var decode = try iterator(
            fixture, input: Self.prompt, blockSize: blockSize, policy: Self.losingPolicy)
        var proposed = decode.proposedCount
        var roundsAfterAPause = 0

        while decode.next() != nil {
            guard decode.proposedCount != proposed else { continue }
            proposed = decode.proposedCount
            let state = try #require(decode.drafterState)
            #expect(state.nextPosition == decode.mainCacheStorage.processedTokenCount)
            #expect(state.proposalAppended == 0)
            #expect(state.isAligned)
            if (decode.speculativeDecodingTelemetry?.adaptivePauseCount ?? 0) > 0 {
                roundsAfterAPause += 1
            }
        }

        #expect(roundsAfterAPause > 0)
        #expect(decode.passthroughReason == nil)
    }

    /// The token counts after which a decode stops on a token a plain step of its guard returned
    /// and fed, with the main cache one token ahead of a round's end.
    private func pausedStops(_ fixture: Qwen35MTPFixture, blockSize: Int) throws -> [Int] {
        var decode = try iterator(
            fixture, input: Self.prompt, blockSize: blockSize, policy: Self.losingPolicy)
        var stops: [Int] = []
        var plainTokens = 0
        var returned = 0
        while decode.next() != nil {
            returned += 1
            let telemetry = decode.speculativeDecodingTelemetry
            if let telemetry, telemetry.adaptivePlainTokenCount > plainTokens,
                decode.hasCachedLastToken
            {
                stops.append(returned)
            }
            plainTokens = telemetry?.adaptivePlainTokenCount ?? plainTokens
        }
        return stops
    }

    @Test(
        "a stop inside a pause leaves the main cache and the drafter continuation a prefill leaves",
        arguments: blockSizes)
    func stopInsideAPauseLeavesThePrefillContinuation(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let script = Self.prompt + plain.tokens
        let stops = try pausedStops(fixture, blockSize: blockSize)
        #expect(!stops.isEmpty)

        for stop in stops {
            let cache = try fixture.target.newCache(parameters: nil)
            var decode = try iterator(
                fixture, input: Self.prompt, blockSize: blockSize, policy: Self.losingPolicy,
                cache: cache)
            let tokens = drain(&decode, limit: stop)
            decode.finalizeGeneration()
            let held = heldTokenCount(of: cache)
            let continuation = try #require(
                decode.mtpDrafterContinuation, "stop \(stop): \(decode.passthroughReason ?? "")")
            let reference = try #require(
                try fixture.prefillContinuation(of: Array(script[..<held]), blockSize: blockSize))

            #expect(tokens == Array(plain.tokens.prefix(stop)), "stop \(stop)")
            #expect(held == Self.prompt.count + stop, "stop \(stop)")
            try fixture.expectCache(
                cache, holdsThePrefillOf: Array(script[..<held]), comment: "stop \(stop)")
            #expect(continuation.targetProcessedTokenCount == held, "stop \(stop)")
            #expect(continuation.state.nextPosition == reference.state.nextPosition, "stop \(stop)")
            let pairs =
                Array(
                    zip(
                        continuation.state.cache.flatMap(\.state),
                        reference.state.cache.flatMap(\.state)))
                + [(continuation.targetBoundaryHidden, reference.targetBoundaryHidden)]
            for (got, expected) in pairs {
                let gap = AttentionKernelTolerance.worstGap(got, expected)
                #expect(
                    gap <= Qwen35MTPFixture.rebuiltCacheTolerance, "stop \(stop): worst gap \(gap)")
            }
        }
    }

    @Test(
        "a second call after a stop inside a pause continues as plain decode and drafts again",
        arguments: blockSizes)
    func secondCallAfterAPausedStopContinues(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let stop = try #require(try pausedStops(fixture, blockSize: blockSize).first)
        let cache = try fixture.target.newCache(parameters: nil)
        var first = try iterator(
            fixture, input: Self.prompt, blockSize: blockSize, policy: Self.losingPolicy,
            cache: cache)
        let firstTokens = drain(&first, limit: stop)
        first.finalizeGeneration()
        let continuation = try #require(first.mtpDrafterContinuation)
        let sequence = Self.prompt + firstTokens + Self.followUp
        let plain = try await fixture.decodeGreedily(
            prompt: sequence, maxTokens: Self.continuedTokenCount, speculative: false)

        var continued = try iterator(
            fixture, input: Self.followUp, blockSize: blockSize, policy: .disabled,
            maxTokens: Self.continuedTokenCount, cache: cache, continuation: continuation)

        #expect(drain(&continued) == plain.tokens)
        #expect(continued.passthroughReason == nil)
        #expect(continued.proposedCount > 0)
    }

    @Test(
        "a chat session whose guard pauses speculation writes plain decode's turns from its cache")
    func chatSessionWithPausesReusesItsCache() async throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let drafter = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration, model: fixture.drafter))
        let parameters = GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0)
        let guarded = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: drafter, blockSize: Qwen35MTPFixture.blockSize,
                adaptivePolicy: Self.losingPolicy),
            generateParameters: parameters)
        let plain = ChatSession(fixture.makeModelContext(), generateParameters: parameters)

        let guardedFirst = try await collectChatTurn(
            guarded.streamDetails(to: fixture.text(for: [1])))
        let plainFirst = try await collectChatTurn(plain.streamDetails(to: fixture.text(for: [1])))
        let guardedSecond = try await collectChatTurn(
            guarded.streamDetails(to: fixture.text(for: [2])))
        let plainSecond = try await collectChatTurn(plain.streamDetails(to: fixture.text(for: [2])))

        #expect(guardedFirst.text == plainFirst.text)
        #expect(guardedSecond.text == plainSecond.text)
        // The new message, and the first turn's last token when no forward fed it.
        #expect(Self.secondTurnPromptTokenCounts.contains(guardedSecond.info.promptTokenCount))
        #expect(guardedSecond.info.cachedPromptTokenCount > 0)
        #expect((guardedSecond.info.proposedDraftTokens ?? 0) > 0)
        #expect(guardedSecond.info.passthroughReason == nil)
        #expect((guardedSecond.info.speculativeDecodingTelemetry?.adaptivePauseCount ?? 0) > 0)
    }

    @Test("a sampled decode whose guard pauses speculation keeps the drafter aligned")
    func sampledDecodeWithPausesKeepsTheDrafterAligned() throws {
        let fixture = try Qwen35MTPFixture()
        var decode = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter,
            parameters: GenerateParameters(
                maxTokens: Self.generatedTokenCount, temperature: Self.temperature,
                seed: Self.seed),
            blockSize: Qwen35MTPFixture.blockSize, adaptivePolicy: Self.losingPolicy)

        let tokens = drain(&decode)
        let state = try #require(decode.drafterState)
        let telemetry = try #require(decode.speculativeDecodingTelemetry)

        #expect(tokens.count == Self.generatedTokenCount)
        #expect(tokens.allSatisfy { (0 ..< Qwen35MTPFixture.vocabularySize).contains($0) })
        #expect(telemetry.adaptivePauseCount > 0)
        #expect(decode.passthroughReason == nil)
        #expect(state.isAligned)
    }
}
