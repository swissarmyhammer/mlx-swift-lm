import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

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
    static let decodePrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each greedy decode of the iterator tests generates: two cycles of the plan of
    /// the largest block size.
    private static let generatedTokenCount = 48

    /// The block sizes of the iterator tests: one draft, up to the most drafts a hybrid target
    /// rewinds.
    static let blockSizes = Array(2 ... GatedDeltaDecodeKernel.mostTokens)

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
            plan: ScriptedDraftPlan(
                script: script, promptLength: Self.decodePrompt.count, blockSize: blockSize,
                vocabularySize: Qwen35MTPFixture.vocabularySize))
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
        try fixture.expectCache(decode.cache, holdsThePrefillOf: Array(script[..<held]))
    }
}
