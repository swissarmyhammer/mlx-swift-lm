import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// A commit keeps the top-1 probability of the seed it draws, and a round whose seed falls
/// below the confidence floor drafts the seed alone without a chained MTP step, so it costs a
/// round of one draft. Greedy output stays the plain greedy output.
@Suite(.serialized)
struct Qwen35MTPSeedGateTests {

    /// The tokens of each drafter-level round: the seed, two chained drafts and the bonus.
    private static let blockSize = 4

    /// A confidence floor no drafter's top-1 probability reaches below certainty.
    private static let certaintyFloor: Float = 1

    /// The gap the seed's probability may hold from the probability the test computes.
    private static let probabilityTolerance: Float = 1e-5

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The tokens each decode generates: enough rounds to gate some seeds and chain others.
    private static let generatedTokenCount = 32

    /// The confidence floor of the decode tests: below the seed's top-1 probability in some
    /// rounds of the copying drafter and above it in others. Over the 64 tokens of the fixture's
    /// vocabulary, its seeds give their most likely token about 0.036 to 0.075.
    private static let decodeFloor: Float = 0.045

    /// The drafts of one round of ``blockSize`` tokens from `state`, drawn as tokens or with
    /// their logits.
    private func draftCount(
        _ fixture: Qwen35MTPFixture, state: inout MTPDrafterState, withLogits: Bool
    ) -> Int {
        let lastToken = MLXArray([QwenMTPDistributionDraftTests.firstBonus])
        let lastHidden = MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize])
        let queryOffset = state.nextPosition
        if withLogits {
            return fixture.drafter.draftBlockWithLogits(
                target: fixture.target, lastToken: lastToken, lastHidden: lastHidden,
                sharedKV: [:], positionDeltas: nil, queryOffset: queryOffset,
                blockSize: Self.blockSize, state: &state, processor: nil, sampler: ArgMaxSampler()
            ).tokens.dim(-1)
        }
        return fixture.drafter.draftBlock(
            target: fixture.target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: [:],
            positionDeltas: nil, queryOffset: queryOffset, blockSize: Self.blockSize,
            state: &state, sampler: ArgMaxSampler()
        ).dim(-1)
    }

    /// The top-1 probability of the seed of `state`, read back.
    private func seedProbability(_ state: MTPDrafterState) throws -> Float {
        try #require(state.seedTopProbability).item(Float.self)
    }

    /// The top-1 probability of the logits `lmHead` gives the seed's hidden row of `state`,
    /// computed apart from the drafter.
    private func expectedSeedProbability(
        _ state: MTPDrafterState, lmHead: Linear?
    ) throws -> Float {
        let seedHidden = try #require(state.seedHidden)
        let lmHead = try #require(lmHead)
        let logits = lmHead(seedHidden)[0..., -1, 0...]
        return softmax(logits, axis: -1).max().item(Float.self)
    }

    @Test("a commit keeps the top-1 probability of the seed it draws")
    func commitKeepsTheSeedProbability() throws {
        let fixture = try Qwen35MTPFixture()
        let state = QwenMTPDistributionDraftTests.preparedState(fixture)

        let expected = try expectedSeedProbability(state, lmHead: fixture.target.lmHead)

        #expect(abs(try seedProbability(state) - expected) <= Self.probabilityTolerance)
    }

    @Test("a VLM commit keeps the top-1 probability of the seed it draws")
    func visionLanguageCommitKeepsTheSeedProbability() throws {
        let fixture = try Qwen35VLMMTPFixture()
        var state = fixture.drafter.makeState(parameters: nil)
        let promptTokens = QwenMTPDistributionDraftTests.promptTokens
        fixture.drafter.prepareDrafterState(
            target: fixture.target, promptTokens: MLXArray(promptTokens)[.newAxis, 0...],
            targetHidden: MLXArray.zeros([1, promptTokens.count, Qwen35MTPFixture.hiddenSize]),
            firstBonus: MLXArray([QwenMTPDistributionDraftTests.firstBonus]),
            positionDeltas: nil, state: &state, sampler: ArgMaxSampler())

        let expected = try expectedSeedProbability(
            state, lmHead: fixture.target.languageModel.lmHead)

        #expect(abs(try seedProbability(state) - expected) <= Self.probabilityTolerance)
    }

    @Test(
        "a seed below the floor drafts alone and appends no chained entry",
        arguments: [false, true])
    func seedBelowTheFloorDraftsAlone(withLogits: Bool) throws {
        let fixture = try Qwen35MTPFixture()
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)
        state.minimumDraftProbability = Self.certaintyFloor
        let committedPosition = state.nextPosition

        let drafts = draftCount(fixture, state: &state, withLogits: withLogits)

        #expect(drafts == 1)
        #expect(state.proposalAppended == 0)
        #expect(state.nextPosition == committedPosition)
        #expect(state.isAligned)
    }

    @Test("a seed at the floor chains drafts after it", arguments: [false, true])
    func seedAtTheFloorChains(withLogits: Bool) throws {
        let fixture = try Qwen35MTPFixture()
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)
        state.minimumDraftProbability = try seedProbability(state)

        _ = draftCount(fixture, state: &state, withLogits: withLogits)

        #expect(state.proposalAppended > 0)
    }

    @Test("a seed without its probability chains as a restored continuation does")
    func seedWithoutItsProbabilityChains() throws {
        let fixture = try Qwen35MTPFixture()
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)
        state.seedTopProbability = nil
        state.minimumDraftProbability = Self.certaintyFloor

        let drafts = draftCount(fixture, state: &state, withLogits: false)

        // The chain's first step runs, and its own floor check stops it.
        #expect(drafts == 1)
        #expect(state.proposalAppended == 1)
    }

    @Test("removing the seed's entry drops the seed's probability")
    func removingTheSeedEntryDropsItsProbability() throws {
        let fixture = try Qwen35MTPFixture()
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)

        let removed = state.removeLastEntries(1)

        #expect(removed)
        #expect(state.seedToken == nil)
        #expect(state.seedTopProbability == nil)
    }

    @Test("taking the seed drops the seed's probability")
    func takingTheSeedDropsItsProbability() throws {
        let fixture = try Qwen35MTPFixture()
        var state = QwenMTPDistributionDraftTests.preparedState(fixture)

        let taken = state.takeSeed()

        let seed = try #require(taken)
        #expect(seed.topProbability != nil)
        #expect(state.seedTopProbability == nil)
    }

    @Test(
        "greedy decode with the seed gate equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases,
        Qwen35ChainedMTPDraftTests.chainedBlockSizes)
    func gatedDecodeEqualsPlainDecode(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        var decode = try fixture.greedyMTPIterator(
            over: Self.prompt, blockSize: blockSize, maxTokens: Self.generatedTokenCount,
            minimumDraftProbability: Self.decodeFloor)

        let tokens = drain(&decode)

        #expect(tokens == plain.tokens)
        #expect(decode.passthroughReason == nil)
        #expect(decode.proposedCount > 0)
    }

    @Test(
        "the copying drafter's seed gate both drafts the seed alone and chains",
        arguments: Qwen35ChainedMTPDraftTests.chainedBlockSizes)
    func gatedDecodeHasSeedAloneAndChainedRounds(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var decode = try fixture.greedyMTPIterator(
            over: Self.prompt, blockSize: blockSize, maxTokens: Self.generatedTokenCount,
            minimumDraftProbability: Self.decodeFloor)
        var proposed = decode.proposedCount
        var seedAloneRounds = 0
        var chainedRounds = 0

        while decode.next() != nil {
            let roundDrafts = decode.proposedCount - proposed
            if roundDrafts == 1 { seedAloneRounds += 1 }
            if roundDrafts > 1 { chainedRounds += 1 }
            proposed = decode.proposedCount
        }

        #expect(seedAloneRounds > 0)
        #expect(chainedRounds > 0)
    }
}
