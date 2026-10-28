import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A drafter that proposes the tokens of a script: the plain greedy decode of the target. In
/// each round it spoils the draft at one depth, so that the target rejects every draft from that
/// depth on. The depths of the rounds cycle through each count of accepted drafts, the last of
/// which spoils nothing.
///
/// A round is known by its position, the tokens the target cache holds before it, thus the
/// drafter keeps no state of its own. The plan of positions follows from the depths: a round
/// that accepts `d` drafts moves the position by `d + 1`.
private final class ScriptedRejectionDrafter: Module, MTPDrafterModel {

    /// The prompt and the plain greedy tokens after it.
    private let script: [Int]

    /// The count of accepted drafts of the round at each position of the plan.
    private let acceptedDraftsAtPosition: [Int: Int]

    /// The tokens of the vocabulary of the target.
    private let vocabularySize: Int

    var maximumBlockSize: Int? { nil }
    var requiresSharedTargetKV: Bool { false }
    var requiresGreedySampling: Bool { true }

    /// - Parameters:
    ///   - script: The prompt and the plain greedy tokens after it.
    ///   - promptLength: The tokens of the prompt: the position of the first round.
    ///   - blockSize: The tokens of each round: the drafts and the bonus token.
    ///   - vocabularySize: The tokens of the vocabulary of the target.
    init(script: [Int], promptLength: Int, blockSize: Int, vocabularySize: Int) {
        self.script = script
        self.vocabularySize = vocabularySize
        var plan: [Int: Int] = [:]
        var position = promptLength
        var round = 0
        while position < script.count {
            let accepted = round % blockSize
            plan[position] = accepted
            position += accepted + 1
            round += 1
        }
        self.acceptedDraftsAtPosition = plan
    }

    func validateCompatibility(with target: any LanguageModel) throws {}

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        let draftCount = blockSize - 1
        let spoiledDepth = acceptedDraftsAtPosition[queryOffset] ?? draftCount
        let drafts = (0 ..< draftCount).map { depth in
            let position = queryOffset + 1 + depth
            let token = position < script.count ? script[position] : 0
            return Int32(depth == spoiledDepth ? (token + 1) % vocabularySize : token)
        }
        return MLXArray(drafts, [lastToken.dim(0), draftCount])
    }
}

/// A verify of a hybrid target followed by a rewind of its rejected drafts leaves the main cache
/// of a run of the kept tokens only, for each count of rejected drafts.
@Suite(.serialized)
struct Qwen35VerifyRewindTests {

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_101

    /// The seed of the verify tokens.
    private static let verifySeed: UInt64 = 20_261_102

    /// The prompt tokens the cache holds before the verify. More than
    /// ``GatedDeltaDecodeKernel/mostTokens``, so the prefill takes the general path.
    private static let promptLength = 8

    /// The tokens of the verify: the bonus token and the most drafts the kernel holds.
    private static let verifyTokenCount = GatedDeltaDecodeKernel.mostTokens

    /// The prompt of each greedy decode of the iterator tests.
    private static let decodePrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each greedy decode of the iterator tests generates: two cycles of the plan of
    /// the largest block size.
    private static let generatedTokenCount = 48

    /// The block sizes of the iterator tests: one draft, up to the most drafts a hybrid target
    /// rewinds.
    static let blockSizes = Array(2 ... GatedDeltaDecodeKernel.mostTokens)

    /// The bound of a cache that MTP decode built against a cache that one prefill built: both
    /// add in float32, each in its own order.
    private static let rebuiltCacheTolerance = AttentionKernelTolerance.float32

    /// The cache of `model` after the prompt.
    private func promptCache(_ model: Qwen35TextModel) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        let prompt = seededTokens(
            Self.promptLength, vocabularySize: model.vocabularySize, seed: Self.promptSeed)
        eval(model(LMInput.Text(tokens: prompt[.newAxis]), cache: cache, state: nil).logits)
        return cache
    }

    /// The cache after the prompt and a verify of `tokens` on the compiled or the general path,
    /// as the MTP iterator runs it.
    private func verifyCache(
        _ model: Qwen35TextModel, tokens: MLXArray, compiled: Bool
    ) throws -> [KVCache] {
        let cache = try promptCache(model)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        let output = model.forward(
            LMInput.Text(tokens: tokens[.newAxis]), cache: cache, state: state,
            lastTokenOnly: false, useCompiledDecode: compiled)
        eval(output.logits)
        return cache
    }

    @Test(
        "a verify rewound by each count of rejected drafts holds the cache of the kept tokens",
        arguments: [false, true])
    func rewoundVerifyMatchesARunOfTheKeptTokens(compiled: Bool) throws {
        let model = try Qwen35CompiledVerifyTests.kernelModel()
        let tolerance = try #require(AttentionKernelTolerance.of(Qwen35CompiledVerifyTests.dtype))
        let tokens = seededTokens(
            Self.verifyTokenCount, vocabularySize: model.vocabularySize, seed: Self.verifySeed)

        for rollback in 1 ..< Self.verifyTokenCount {
            let round = "compiled \(compiled), rollback \(rollback)"
            let rewound = try verifyCache(model, tokens: tokens, compiled: compiled)
            #expect(
                rewindSpeculativePromptCache(rewound, numTokens: rollback) == rollback, "\(round)")
            let kept = try verifyCache(
                model, tokens: tokens[..<(Self.verifyTokenCount - rollback)], compiled: compiled)

            #expect(rewound.map(\.offset) == kept.map(\.offset), "\(round)")
            for (layer, (gotCache, expectedCache)) in zip(rewound, kept).enumerated() {
                let gotStates = gotCache.state
                let expectedStates = expectedCache.state
                eval(gotStates, expectedStates)
                try #require(gotStates.count == expectedStates.count, "\(round) layer \(layer)")
                for (got, expected) in zip(gotStates, expectedStates) {
                    if gotCache is MambaCache {
                        let gap = AttentionKernelTolerance.worstGap(got, expected)
                        #expect(gap <= tolerance, "\(round) GDN layer \(layer): worst gap \(gap)")
                    } else {
                        #expect(
                            arrayEqual(got, expected).item(Bool.self),
                            "\(round) attention layer \(layer)")
                    }
                }
            }
        }
    }

    /// What one MTP decode with the scripted drafter gives: its tokens, the counts of rejected
    /// drafts of its rounds, its passthrough reason, and its main cache.
    private struct ScriptedDecode {
        let tokens: [Int]
        let rejectedDraftCounts: Set<Int>
        let passthroughReason: String?
        let cache: [KVCache]
    }

    /// Decodes ``generatedTokenCount`` greedy MTP tokens after ``decodePrompt`` with a drafter
    /// that proposes `script` and spoils one depth of each round.
    private func scriptedDecode(
        _ fixture: Qwen35MTPFixture, script: [Int], blockSize: Int
    ) throws -> ScriptedDecode {
        let cache = try fixture.target.newCache(parameters: nil)
        let drafter = ScriptedRejectionDrafter(
            script: script, promptLength: Self.decodePrompt.count, blockSize: blockSize,
            vocabularySize: Qwen35MTPFixture.vocabularySize)
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.decodePrompt.map(Int32.init))),
            mainModel: fixture.target, drafter: drafter, mainCache: cache,
            parameters: GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0),
            blockSize: blockSize)
        var tokens: [Int] = []
        var rejectedDraftCounts: Set<Int> = []
        var proposedCount = iterator.proposedCount
        var acceptedCount = iterator.acceptedCount
        while let token = iterator.next() {
            tokens.append(token)
            if iterator.proposedCount > proposedCount {
                let proposed = iterator.proposedCount - proposedCount
                rejectedDraftCounts.insert(proposed - (iterator.acceptedCount - acceptedCount))
            }
            proposedCount = iterator.proposedCount
            acceptedCount = iterator.acceptedCount
        }
        iterator.finalizeGeneration()
        return ScriptedDecode(
            tokens: tokens, rejectedDraftCounts: rejectedDraftCounts,
            passthroughReason: iterator.passthroughReason, cache: cache)
    }

    @Test(
        "greedy MTP decode of a hybrid target with each count of rejected drafts equals plain decode",
        arguments: blockSizes)
    func everyRejectedDraftCountEqualsPlainDecode(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: Self.decodePrompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let script = Self.decodePrompt + plain.tokens

        let decode = try scriptedDecode(fixture, script: script, blockSize: blockSize)

        #expect(decode.tokens == plain.tokens)
        #expect(decode.passthroughReason == nil)
        #expect(decode.rejectedDraftCounts == Set(0 ..< blockSize))

        let held = KVCacheStorage(decode.cache, plan: .disabled).processedTokenCount
        let rebuilt = try fixture.target.newCache(parameters: nil)
        eval(
            fixture.target(
                LMInput.Text(tokens: MLXArray(script[..<held].map(Int32.init))[.newAxis]),
                cache: rebuilt, state: nil
            ).logits)
        #expect(decode.cache.map(\.offset) == rebuilt.map(\.offset))
        for (got, expected) in zip(decode.cache.flatMap(\.state), rebuilt.flatMap(\.state)) {
            let gap = AttentionKernelTolerance.worstGap(got, expected)
            #expect(gap <= Self.rebuiltCacheTolerance, "worst gap \(gap)")
        }
    }
}
