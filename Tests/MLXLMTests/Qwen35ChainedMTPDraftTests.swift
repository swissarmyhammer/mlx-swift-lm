import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A Qwen MTP round drafts the seed the last commit left and chains the MTP head on its own
/// hidden row for the rest of the round, so a round of `blockSize` tokens verifies
/// `blockSize - 1` drafts. Greedy output stays the plain greedy output, and the drafter keeps one
/// entry for each token the main cache holds.
@Suite(.serialized)
struct Qwen35ChainedMTPDraftTests {

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The tokens each decode generates: enough rounds of the largest block size to accept
    /// some chained drafts and reject others.
    private static let generatedTokenCount = 32

    /// The block sizes that chain at least one draft after the seed: two drafts, up to the most
    /// drafts a hybrid target rewinds.
    static let chainedBlockSizes = Array(3 ... GatedDeltaDecodeKernel.mostTokens)

    /// The block size of the sampled and the drafter-level tests: the seed and two chained drafts.
    private static let sampledBlockSize = 4

    /// The sampled setting of the release benchmark.
    private static let temperature: Float = 0.7

    /// The nucleus of the sampled setting of the release benchmark.
    private static let topP: Float = 0.95

    /// The seed of each sampled stream.
    private static let seed: UInt64 = 7

    /// The confidence floor llama.cpp users report fastest at five drafts.
    static let reportedFloor: Float = 0.75

    /// A confidence floor no drafter's top-1 probability reaches below certainty, so every chain
    /// stops after the seed.
    private static let stoppingFloor: Float = 1

    /// The seed of the random prompt hidden rows of the drafter-level commit test.
    private static let promptHiddenSeed: UInt64 = 1

    /// The seed of the random verify hidden rows of the drafter-level commit test.
    private static let verifyHiddenSeed: UInt64 = 2

    /// The final token of the round the drafter-level tests commit.
    private static let finalToken: Int32 = 11

    /// One setting of a chained round: its block size and its confidence floor.
    struct ChainSetting: CustomTestStringConvertible, Sendable {

        /// The tokens of each round.
        let blockSize: Int

        /// The drafter top-1 probability below which a chain stops.
        let minimumDraftProbability: Float

        var testDescription: String {
            "blockSize \(blockSize), floor \(minimumDraftProbability)"
        }
    }

    /// Every block size from one draft to the most drafts a hybrid target rewinds, without a
    /// floor and with ``reportedFloor``.
    static let settings = Qwen35VerifyRewindTests.blockSizes.flatMap { blockSize in
        [0, reportedFloor].map {
            ChainSetting(blockSize: blockSize, minimumDraftProbability: $0)
        }
    }

    /// A greedy MTP iterator of the fixture's drafter over ``prompt``.
    private func iterator(
        _ fixture: Qwen35MTPFixture, blockSize: Int, minimumDraftProbability: Float = 0,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: generatedTokenCount, temperature: 0)
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, mainCache: cache,
            parameters: parameters, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability)
    }

    /// A drafter state prefilled over the prompt of ``QwenMTPDistributionDraftTests`` with
    /// `floor`.
    private func preparedState(_ fixture: Qwen35MTPFixture, floor: Float) -> MTPDrafterState {
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)
        state.minimumDraftProbability = floor
        return state
    }

    @Test(
        "greedy chained MTP decode equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, settings)
    func chainedDecodeEqualsPlainDecode(
        mode: Qwen35MTPFixture.DrafterMode, setting: ChainSetting
    ) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        var decode = try iterator(
            fixture, blockSize: setting.blockSize,
            minimumDraftProbability: setting.minimumDraftProbability)

        let tokens = drain(&decode)

        #expect(tokens == plain.tokens)
        #expect(decode.passthroughReason == nil)
        #expect(decode.proposedCount > 0)
    }

    @Test(
        "greedy chained MTP decode of the VLM target equals plain decode",
        arguments: chainedBlockSizes)
    func chainedVisionLanguageDecodeEqualsPlainDecode(blockSize: Int) throws {
        let fixture = try Qwen35VLMMTPFixture()
        let input = LMInput(tokens: MLXArray(Self.prompt.map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0)
        var plain = try TokenIterator(input: input, model: fixture.target, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters, blockSize: blockSize)

        var plainTokens: [Int] = []
        while let token = plain.next() {
            plainTokens.append(token)
        }
        let speculativeTokens = drain(&speculative)

        #expect(speculativeTokens == plainTokens)
        #expect(speculative.passthroughReason == nil)
        #expect(
            speculative.proposedCount > speculative.speculativeDecodingTelemetry?.roundCount ?? 0)
    }

    @Test(
        "the drafter holds one entry for each token the main cache holds after every round",
        arguments: settings)
    func drafterFollowsTheMainCacheAfterEveryRound(setting: ChainSetting) throws {
        let fixture = try Qwen35MTPFixture()
        var decode = try iterator(
            fixture, blockSize: setting.blockSize,
            minimumDraftProbability: setting.minimumDraftProbability)
        var proposed = decode.proposedCount
        var accepted = decode.acceptedCount
        var partialRounds = 0
        var stoppedRounds = 0

        while decode.next() != nil {
            if decode.proposedCount != proposed {
                let state = try #require(decode.drafterState)
                let roundDrafts = decode.proposedCount - proposed
                if (1 ..< roundDrafts).contains(decode.acceptedCount - accepted) {
                    partialRounds += 1
                }
                if roundDrafts < setting.blockSize - 1 {
                    stoppedRounds += 1
                }
                #expect(state.nextPosition == decode.mainCacheStorage.processedTokenCount)
                #expect(state.proposalAppended == 0)
                #expect(state.isAligned)
                proposed = decode.proposedCount
                accepted = decode.acceptedCount
            }
        }

        #expect(decode.passthroughReason == nil)
        // Without a floor the copying drafter accepts some chained drafts and rejects others; with
        // one, the floor of the tiny drafter stops chains early.
        let chainsAfterTheSeed = setting.blockSize > MTPDraftBlockOutput.minimumChainedMTPBlockSize
        let observedRounds = setting.minimumDraftProbability > 0 ? stoppedRounds : partialRounds
        #expect(!chainsAfterTheSeed || observedRounds > 0)
    }

    @Test("a floor of certainty drafts the seed alone in every round", arguments: chainedBlockSizes)
    func certaintyFloorDraftsTheSeedAlone(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        var decode = try iterator(
            fixture, blockSize: blockSize, minimumDraftProbability: Self.stoppingFloor)

        let tokens = drain(&decode)
        let telemetry = try #require(decode.speculativeDecodingTelemetry)

        #expect(tokens == plain.tokens)
        #expect(telemetry.draftTokenCount == telemetry.roundCount)
    }

    @Test("a chain stopped by its floor drafts the seed, and its commit keeps the drafter aligned")
    func floorStopsTheChainAfterTheSeed() throws {
        let fixture = try Qwen35MTPFixture()
        var state = preparedState(fixture, floor: Self.stoppingFloor)
        let committedPosition = state.nextPosition
        let verifyHidden = MLXRandom.normal(
            [1, Self.sampledBlockSize, Qwen35MTPFixture.hiddenSize],
            key: MLXRandom.key(Self.verifyHiddenSeed))

        let drafts = fixture.drafter.draftBlock(
            target: fixture.target, lastToken: MLXArray([QwenMTPDistributionDraftTests.firstBonus]),
            lastHidden: MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize]), sharedKV: [:],
            positionDeltas: nil, queryOffset: committedPosition,
            blockSize: Self.sampledBlockSize, state: &state, sampler: ArgMaxSampler())
        fixture.drafter.commitDrafterState(
            target: fixture.target, targetHidden: verifyHidden, draftTokens: drafts,
            acceptedCount: drafts.dim(-1), finalToken: MLXArray([Self.finalToken]),
            positionDeltas: nil, state: &state, sampler: ArgMaxSampler())

        #expect(drafts.shape == [1, 1])
        #expect(state.nextPosition == committedPosition + drafts.dim(-1) + 1)
        #expect(state.proposalAppended == 0)
        #expect(state.isAligned)
        #expect(state.seedToken != nil)
    }

    @Test("a chain with logits stopped by its floor drafts the seed with its logits")
    func floorStopsTheChainWithLogitsAfterTheSeed() throws {
        let fixture = try Qwen35MTPFixture()
        var state = preparedState(fixture, floor: Self.stoppingFloor)

        let output = fixture.drafter.draftBlockWithLogits(
            target: fixture.target, lastToken: MLXArray([QwenMTPDistributionDraftTests.firstBonus]),
            lastHidden: MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize]), sharedKV: [:],
            positionDeltas: nil, queryOffset: state.nextPosition,
            blockSize: Self.sampledBlockSize, state: &state, processor: nil,
            sampler: ArgMaxSampler())

        #expect(output.tokens.shape == [1, 1])
        #expect(output.processedLogits.shape == [1, 1, Qwen35MTPFixture.vocabularySize])
        #expect(state.isAligned)
    }

    @Test("an iterator refuses a floor outside zero to one")
    func iteratorRefusesAFloorOutsideTheUnitInterval() throws {
        let fixture = try Qwen35MTPFixture()

        #expect(throws: SpeculativeDecodingConfigurationError.self) {
            try iterator(
                fixture, blockSize: Self.sampledBlockSize,
                minimumDraftProbability: Self.stoppingFloor + 1)
        }
    }

    /// Reads every token of `decode`.
    private func drain(_ decode: inout MTPSpeculativeTokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = decode.next() {
            tokens.append(token)
        }
        return tokens
    }

    @Test(
        "the first round of a chained block drafts the seed and the chained drafts",
        arguments: chainedBlockSizes)
    func firstRoundDraftsEveryDraftOfTheBlock(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var decode = try iterator(fixture, blockSize: blockSize)

        // The first token is the prefill bonus; the second runs the first round.
        _ = decode.next()
        _ = decode.next()

        #expect(decode.blockSize == blockSize)
        #expect(decode.proposedCount == blockSize - 1)
    }

    @Test(
        "a sampled stream chains drafts after the seed",
        arguments: MTPSampledVerification.allCases)
    func sampledStreamChainsDrafts(verification: MTPSampledVerification) throws {
        let fixture = try Qwen35MTPFixture()
        var parameters = GenerateParameters(
            maxTokens: Self.generatedTokenCount, temperature: Self.temperature, topP: Self.topP,
            seed: Self.seed)
        parameters.mtpSampledVerification = verification
        var decode = try iterator(
            fixture, blockSize: Self.sampledBlockSize, parameters: parameters)

        let tokens = drain(&decode)
        let telemetry = try #require(decode.speculativeDecodingTelemetry)

        #expect(tokens.count == Self.generatedTokenCount)
        #expect(decode.passthroughReason == nil)
        #expect(telemetry.draftTokenCount > telemetry.roundCount)
        #expect(decode.acceptedCount > 0)
    }

    @Test(
        "a chained decode leaves the drafter continuation that a prefill of the held tokens builds",
        arguments: chainedBlockSizes)
    func chainedDecodeLeavesThePrefillContinuation(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        let cache = try fixture.target.newCache(parameters: nil)
        var decode = try iterator(fixture, blockSize: blockSize, cache: cache)
        let tokens = drain(&decode)
        decode.finalizeGeneration()
        let continuation = try #require(
            decode.mtpDrafterContinuation, "\(decode.passthroughReason ?? "")")
        let held = KVCacheStorage(cache, plan: .disabled).processedTokenCount
        let heldTokens = Array((Self.prompt + tokens).prefix(held))
        let reference = try #require(
            try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(heldTokens.map(Int32.init))),
                mainModel: fixture.target, drafter: fixture.drafter,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                blockSize: blockSize
            ).promptBoundaryContinuation)

        // More drafts accepted than rounds: some round accepted a chained draft.
        #expect(decode.acceptedCount > decode.speculativeDecodingTelemetry?.roundCount ?? 0)
        #expect(continuation.targetProcessedTokenCount == held)
        #expect(continuation.state.nextPosition == reference.state.nextPosition)
        let pairs = Array(
            zip(continuation.state.cache.flatMap(\.state), reference.state.cache.flatMap(\.state)))
        #expect(!pairs.isEmpty)
        for (got, expected) in pairs {
            let gap = AttentionKernelTolerance.worstGap(got, expected)
            #expect(gap <= Qwen35MTPFixture.rebuiltCacheTolerance, "worst gap \(gap)")
        }
    }

    @Test(
        "a commit that accepts chained drafts leaves the drafter a prefill of the accepted tokens builds"
    )
    func commitOfChainedDraftsMatchesAPrefill() throws {
        // The independent drafter reads the hidden rows, so an entry paired with the drafter's
        // own hidden row differs from one paired with the verifier's.
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let prompt = QwenMTPDistributionDraftTests.promptTokens
        let bonus = QwenMTPDistributionDraftTests.firstBonus
        let finalToken = Self.finalToken
        // Every draft but the last: the seed and at least one chained draft.
        let acceptedCount = Self.sampledBlockSize - 2
        let promptHidden = MLXRandom.normal(
            [1, prompt.count, Qwen35MTPFixture.hiddenSize],
            key: MLXRandom.key(Self.promptHiddenSeed))
        let verifyHidden = MLXRandom.normal(
            [1, Self.sampledBlockSize, Qwen35MTPFixture.hiddenSize],
            key: MLXRandom.key(Self.verifyHiddenSeed))
        var state = fixture.drafter.makeState(parameters: nil)
        fixture.drafter.prepareDrafterState(
            target: fixture.target, promptTokens: MLXArray(prompt)[.newAxis, 0...],
            targetHidden: promptHidden, firstBonus: MLXArray([bonus]), positionDeltas: nil,
            state: &state, sampler: ArgMaxSampler())

        let drafts = fixture.drafter.draftBlock(
            target: fixture.target, lastToken: MLXArray([bonus]),
            lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:], positionDeltas: nil,
            queryOffset: prompt.count, blockSize: Self.sampledBlockSize, state: &state,
            sampler: ArgMaxSampler())
        fixture.drafter.commitDrafterState(
            target: fixture.target, targetHidden: verifyHidden, draftTokens: drafts,
            acceptedCount: acceptedCount, finalToken: MLXArray([finalToken]), positionDeltas: nil,
            state: &state, sampler: ArgMaxSampler())

        let accepted = drafts[0, ..<acceptedCount].asArray(Int32.self)
        var prefilled = fixture.drafter.makeState(parameters: nil)
        fixture.drafter.prepareDrafterState(
            target: fixture.target,
            promptTokens: MLXArray(prompt + [bonus] + accepted)[.newAxis, 0...],
            targetHidden: concatenated(
                [promptHidden, verifyHidden[0..., ..<(acceptedCount + 1), 0...]], axis: 1),
            firstBonus: MLXArray([finalToken]), positionDeltas: nil, state: &prefilled,
            sampler: ArgMaxSampler())

        #expect(state.nextPosition == prefilled.nextPosition)
        #expect(state.proposalAppended == 0)
        #expect(state.isAligned)
        let pairs = Array(zip(state.cache.flatMap(\.state), prefilled.cache.flatMap(\.state)))
        #expect(!pairs.isEmpty)
        for (got, expected) in pairs {
            let gap = AttentionKernelTolerance.worstGap(got, expected)
            #expect(gap <= Qwen35MTPFixture.rebuiltCacheTolerance, "worst gap \(gap)")
        }
        let seed = try #require(state.seedToken)
        let prefilledSeed = try #require(prefilled.seedToken)
        #expect(seed.asArray(Int32.self) == prefilledSeed.asArray(Int32.self))
    }

    @Test("a chained draft with logits equals the chained token draft under an argmax sampler")
    func chainedDraftWithLogitsMatchesTheTokenDraft() throws {
        let fixture = try Qwen35MTPFixture()
        var tokenState = QwenMTPDistributionDraftTests.preparedState(fixture)
        var distributionState = QwenMTPDistributionDraftTests.preparedState(fixture)
        let lastToken = MLXArray([QwenMTPDistributionDraftTests.firstBonus])
        let lastHidden = MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize])
        let queryOffset = QwenMTPDistributionDraftTests.promptTokens.count

        let tokens = fixture.drafter.draftBlock(
            target: fixture.target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: [:],
            positionDeltas: nil, queryOffset: queryOffset, blockSize: Self.sampledBlockSize,
            state: &tokenState, sampler: ArgMaxSampler())
        let output = fixture.drafter.draftBlockWithLogits(
            target: fixture.target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: [:],
            positionDeltas: nil, queryOffset: queryOffset, blockSize: Self.sampledBlockSize,
            state: &distributionState, processor: nil, sampler: ArgMaxSampler())

        let draftCount = Self.sampledBlockSize - 1
        #expect(tokens.shape == [1, draftCount])
        #expect(output.tokens.asArray(Int.self) == tokens.asArray(Int.self))
        #expect(output.processedLogits.shape == [1, draftCount, Qwen35MTPFixture.vocabularySize])
        #expect(
            argMax(output.processedLogits, axis: -1).asArray(Int.self) == tokens.asArray(Int.self))
        #expect(tokenState.proposalAppended == draftCount - 1)
        #expect(distributionState.proposalAppended == tokenState.proposalAppended)
        #expect(distributionState.nextPosition == tokenState.nextPosition)
        #expect(tokenState.isAligned)
    }
}
