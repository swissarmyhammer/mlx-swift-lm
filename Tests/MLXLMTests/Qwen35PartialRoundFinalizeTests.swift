import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A greedy MTP stream of a hybrid target that stops inside a round that accepted some of its
/// drafts and rejected the others leaves the main cache and the drafter continuation of the
/// tokens it returned, so that a later call or chat turn continues them without a recompute.
@Suite(.serialized)
struct Qwen35PartialRoundFinalizeTests {

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The tokens each first decode generates: the prefill bonus and one cycle of the plan of
    /// the largest block size, `1 + 2 + … + 6` tokens, and the round after it.
    private static let generatedTokenCount = 24

    /// The tokens each second call or chat turn adds after the tokens of the first.
    private static let followUp = [7, 21, 40]

    /// The tokens each second call or chat turn generates.
    private static let continuedTokenCount = 8

    /// The block sizes that can accept some drafts of a round and reject the others: two
    /// drafts, up to the most drafts a hybrid target rewinds.
    static let blockSizes = Array(3 ... GatedDeltaDecodeKernel.mostTokens)

    /// The block sizes whose rounds can accept two drafts and reject one, thus a stop after
    /// the first accepted draft leaves committed lookahead.
    static let lookaheadBlockSizes = Array(4 ... GatedDeltaDecodeKernel.mostTokens)

    /// Where one token of a decode stands in the round that returned it.
    private struct RoundPosition {

        /// Whether a round that accepted some of its drafts and rejected the others returned it.
        let isInPartialRound: Bool

        /// Whether accepted drafts of its round follow it in the main cache: the lookahead a
        /// stop right after it leaves.
        let leavesLookahead: Bool
    }

    /// What a decode that stops after a number of tokens leaves.
    private struct StoppedDecode {

        /// The tokens the decode returned.
        let tokens: [Int]

        /// Whether the main cache held the last token when the decode returned it.
        let hasCachedLastToken: Bool

        /// The main cache after the decode finalized.
        let cache: [KVCache]

        /// The drafter continuation the decode gave back.
        let continuation: MTPDrafterContinuation?

        /// Why the decode dropped its drafter continuation, or `nil`.
        let passthroughReason: String?

        /// The tokens the main cache holds.
        var heldTokenCount: Int { KVCacheStorage(cache, plan: .disabled).processedTokenCount }
    }

    /// A scripted resumable drafter over `script`, whose first round stands at `promptLength`.
    private func drafter(
        _ fixture: Qwen35MTPFixture, script: [Int], promptLength: Int, blockSize: Int
    ) -> ScriptedResumableDrafter {
        ScriptedResumableDrafter(
            plan: ScriptedDraftPlan(
                script: script, promptLength: promptLength, blockSize: blockSize,
                vocabularySize: Qwen35MTPFixture.vocabularySize),
            stateModel: fixture.drafter)
    }

    /// A greedy MTP iterator over `input`.
    private func iterator(
        _ fixture: Qwen35MTPFixture, drafter: ScriptedResumableDrafter, input: [Int],
        cache: [KVCache], continuation: MTPDrafterContinuation? = nil, blockSize: Int,
        maxTokens: Int
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(input.map(Int32.init))),
            mainModel: fixture.target, drafter: drafter, mainCache: cache,
            mtpContinuation: continuation,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            blockSize: blockSize)
    }

    /// The prompt and the plain greedy tokens after it.
    private func script(_ fixture: Qwen35MTPFixture) async throws -> [Int] {
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        return Self.prompt + plain.tokens
    }

    /// Decodes ``generatedTokenCount`` tokens and names where each stands in its round.
    private func roundPositions(
        _ fixture: Qwen35MTPFixture, script: [Int], blockSize: Int
    ) throws -> [RoundPosition] {
        var decode = try iterator(
            fixture,
            drafter: drafter(
                fixture, script: script, promptLength: Self.prompt.count, blockSize: blockSize),
            input: Self.prompt, cache: try fixture.target.newCache(parameters: nil),
            blockSize: blockSize, maxTokens: Self.generatedTokenCount)
        var positions: [RoundPosition] = []
        var proposed = decode.proposedCount
        var accepted = decode.acceptedCount
        var keptDrafts = 0
        var returnedInRound = 0
        var roundIsPartial = false
        while decode.next() != nil {
            if decode.proposedCount > proposed {
                keptDrafts = decode.acceptedCount - accepted
                roundIsPartial = (1 ..< decode.proposedCount - proposed).contains(keptDrafts)
                returnedInRound = 0
                proposed = decode.proposedCount
                accepted = decode.acceptedCount
            }
            returnedInRound += 1
            positions.append(
                RoundPosition(
                    isInPartialRound: roundIsPartial && returnedInRound <= keptDrafts + 1,
                    leavesLookahead: returnedInRound < keptDrafts))
        }
        return positions
    }

    /// The token counts after which a stop stands inside a partially accepted round.
    private func partialRoundStops(
        _ fixture: Qwen35MTPFixture, script: [Int], blockSize: Int
    ) throws -> [Int] {
        try roundPositions(fixture, script: script, blockSize: blockSize).enumerated()
            .filter { $0.element.isInPartialRound }
            .map { $0.offset + 1 }
    }

    /// Decodes after ``prompt``, stops after `stop` tokens, and finalizes the decode.
    private func stoppedDecode(
        _ fixture: Qwen35MTPFixture, script: [Int], blockSize: Int, stop: Int
    ) throws -> StoppedDecode {
        let cache = try fixture.target.newCache(parameters: nil)
        var decode = try iterator(
            fixture,
            drafter: drafter(
                fixture, script: script, promptLength: Self.prompt.count, blockSize: blockSize),
            input: Self.prompt, cache: cache, blockSize: blockSize,
            maxTokens: Self.generatedTokenCount)
        var tokens: [Int] = []
        while tokens.count < stop, let token = decode.next() {
            tokens.append(token)
        }
        let hasCachedLastToken = decode.hasCachedLastToken
        decode.finalizeGeneration()
        return StoppedDecode(
            tokens: tokens, hasCachedLastToken: hasCachedLastToken, cache: cache,
            continuation: decode.mtpDrafterContinuation,
            passthroughReason: decode.passthroughReason)
    }

    @Test(
        "a stop inside a partially accepted round leaves the main cache of the returned tokens",
        arguments: blockSizes)
    func stopLeavesTheMainCacheOfTheReturnedTokens(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let script = try await script(fixture)
        let stops = try partialRoundStops(fixture, script: script, blockSize: blockSize)
        #expect(!stops.isEmpty)

        for stop in stops {
            let decode = try stoppedDecode(
                fixture, script: script, blockSize: blockSize, stop: stop)
            let held = decode.heldTokenCount

            #expect(decode.tokens == Array(script[Self.prompt.count...].prefix(stop)))
            #expect(
                held == Self.prompt.count + stop - (decode.hasCachedLastToken ? 0 : 1),
                "stop \(stop)")
            try fixture.expectCache(
                decode.cache, holdsThePrefillOf: Array(script[..<held]), comment: "stop \(stop)")
        }
    }

    /// The continuation a decode that prefilled `tokens` holds at their end.
    private func promptBoundaryContinuation(
        _ fixture: Qwen35MTPFixture, tokens: [Int], blockSize: Int
    ) throws -> MTPDrafterContinuation? {
        try iterator(
            fixture,
            drafter: drafter(
                fixture, script: tokens, promptLength: tokens.count, blockSize: blockSize),
            input: tokens, cache: try fixture.target.newCache(parameters: nil),
            blockSize: blockSize, maxTokens: Self.generatedTokenCount
        ).promptBoundaryContinuation
    }

    @Test(
        "a stop inside a partially accepted round leaves the drafter continuation of the returned tokens",
        arguments: blockSizes)
    func stopLeavesTheDrafterContinuationOfTheReturnedTokens(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let script = try await script(fixture)
        let stops = try partialRoundStops(fixture, script: script, blockSize: blockSize)
        #expect(!stops.isEmpty)

        for stop in stops {
            let decode = try stoppedDecode(
                fixture, script: script, blockSize: blockSize, stop: stop)
            let held = decode.heldTokenCount
            let continuation = try #require(
                decode.continuation, "stop \(stop): \(decode.passthroughReason ?? "")")
            let reference = try #require(
                try promptBoundaryContinuation(
                    fixture, tokens: Array(script[..<held]), blockSize: blockSize))

            #expect(continuation.targetProcessedTokenCount == held, "stop \(stop)")
            #expect(continuation.state.nextPosition == reference.state.nextPosition, "stop \(stop)")
            #expect(
                continuation.state.cache.map(\.offset) == reference.state.cache.map(\.offset),
                "stop \(stop)")
            let statePairs = Array(
                zip(
                    continuation.state.cache.flatMap(\.state),
                    reference.state.cache.flatMap(\.state)))
            #expect(!statePairs.isEmpty, "stop \(stop)")
            let pairs =
                statePairs + [(continuation.targetBoundaryHidden, reference.targetBoundaryHidden)]
            for (got, expected) in pairs {
                let gap = AttentionKernelTolerance.worstGap(got, expected)
                #expect(
                    gap <= Qwen35MTPFixture.rebuiltCacheTolerance,
                    "stop \(stop): worst gap \(gap)")
            }
        }
    }

    /// The tokens `decode` generates until its token limit.
    private func drain(_ decode: inout MTPSpeculativeTokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = decode.next() {
            tokens.append(token)
        }
        return tokens
    }

    @Test(
        "a second call after a stop inside a partially accepted round equals a cold call and plain decode",
        arguments: blockSizes)
    func secondCallAfterTheStopEqualsColdAndPlainDecode(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let script = try await script(fixture)
        let stops = try partialRoundStops(fixture, script: script, blockSize: blockSize)
        #expect(!stops.isEmpty)

        for stop in stops {
            let first = try stoppedDecode(
                fixture, script: script, blockSize: blockSize, stop: stop)
            let continuation = try #require(first.continuation, "stop \(stop)")
            let sequence = Array(script[..<first.heldTokenCount]) + Self.followUp
            let plain = try await fixture.decodeGreedily(
                prompt: sequence, maxTokens: Self.continuedTokenCount, speculative: false)
            let secondDrafter = drafter(
                fixture, script: sequence + plain.tokens, promptLength: sequence.count,
                blockSize: blockSize)
            var continued = try iterator(
                fixture, drafter: secondDrafter, input: Self.followUp, cache: first.cache,
                continuation: continuation, blockSize: blockSize,
                maxTokens: Self.continuedTokenCount)
            var cold = try iterator(
                fixture, drafter: secondDrafter, input: sequence,
                cache: try fixture.target.newCache(parameters: nil), blockSize: blockSize,
                maxTokens: Self.continuedTokenCount)

            #expect(drain(&continued) == plain.tokens, "stop \(stop)")
            #expect(drain(&cold) == plain.tokens, "stop \(stop)")
            #expect(continued.passthroughReason == nil, "stop \(stop)")
            #expect(continued.acceptedCount > 0, "stop \(stop)")
        }
    }

    /// Reads every event of one chat turn.
    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await event in stream {
            text += event.chunk ?? ""
            info = event.info ?? info
        }
        return (text, try #require(info))
    }

    @Test(
        "a chat turn that stops inside a partially accepted round leaves a cache the next turn extends",
        arguments: lookaheadBlockSizes)
    func nextChatTurnExtendsTheCacheOfAStopInsideAPartialRound(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let script = try await script(fixture)
        let generated = Array(script[Self.prompt.count...])
        let positions = try roundPositions(fixture, script: script, blockSize: blockSize)
        // The stop token is an accepted draft that accepted drafts of its round follow, and the
        // first of its value, so the turn stops there.
        let stopIndex = try #require(
            positions.indices.first { index in
                positions[index].isInPartialRound && positions[index].leavesLookahead
                    && !generated[..<index].contains(generated[index])
            })
        let stopToken = generated[stopIndex]
        let scriptedDrafter = drafter(
            fixture, script: script, promptLength: Self.prompt.count, blockSize: blockSize)
        let session = ChatSession(
            fixture.makeModelContext(eosTokenIds: [stopToken]),
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: MTPDrafterContainer(
                    context: MTPDrafterContext(
                        configuration: fixture.modelConfiguration, model: scriptedDrafter)),
                blockSize: blockSize),
            generateParameters: GenerateParameters(
                maxTokens: Self.generatedTokenCount, temperature: 0))

        let first = try await collect(session.streamDetails(to: fixture.text(for: Self.prompt)))
        // The template of a real chat model renders the end-of-turn token after the reply; the
        // fixture's template renders contents only, so the next message carries it.
        let nextMessage = [stopToken] + Self.followUp
        let sequence = Self.prompt + generated[...stopIndex] + Self.followUp
        let plain = try await fixture.decodeGreedily(
            prompt: sequence, maxTokens: Self.continuedTokenCount, speculative: false)
        scriptedDrafter.plan = ScriptedDraftPlan(
            script: sequence + plain.tokens, promptLength: sequence.count, blockSize: blockSize,
            vocabularySize: Qwen35MTPFixture.vocabularySize)
        session.generateParameters = GenerateParameters(
            maxTokens: Self.continuedTokenCount, temperature: 0)
        let second = try await collect(session.streamDetails(to: fixture.text(for: nextMessage)))

        #expect(first.text == fixture.text(for: Array(generated[..<stopIndex])))
        #expect(first.info.stopTokenFedToCache == stopToken)
        #expect(second.text == fixture.text(for: Array(plain.tokens.prefix { $0 != stopToken })))
        #expect(second.info.cachedPromptTokenCount == Self.prompt.count + stopIndex + 1)
        #expect(second.info.promptTokenCount == Self.followUp.count)
        #expect((second.info.acceptedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == nil)
    }
}
