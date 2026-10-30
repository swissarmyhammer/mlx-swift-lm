// Copyright © 2026 Apple Inc.
//
// A Qwen3.8 speculative verify reads its few tokens over a long key cache
// through ``VerifyAttentionKernel``. Both attention paths of a full-attention
// layer give the tokens of a verify to it: the traced step of the compiled
// verify schedule (`attentionCacheStep`) and the general attention body. The
// kernel is deterministic, thus each path must give the kernel output bit for
// bit; the SDPA of MLX adds in another order, so a path that kept it fails
// these tests. A verify with a mask array keeps the SDPA of MLX, bit for bit.
//
// The verify forward of the whole model must still give the logits that plain
// decode gives for the same tokens, one at a time, inside the bfloat16
// tolerance, and the same greedy token in each row up to a near tie.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35VerifyAttentionTests {

    /// The cached keys ahead of the verify: the fewest keys the kernel reads,
    /// and some more.
    private static let cachedKeys = DecodeAttentionKernel.fewestKeys + 76

    /// The tokens of the verify: the bonus token and four drafts, as the MTP
    /// head of Qwen3.8 drafts a block of five.
    private static let verifyTokens = 5

    /// The layer: a head of 256 numbers, the only width the kernel reads.
    private static let fixture = Qwen35AttentionLayerFixture(headDim: 256)

    /// The seed of each random operand of the layer tests.
    private static let seeds = Qwen35AttentionLayerFixture.Seeds(
        cache: 20_261_220, tokens: 20_261_221)

    /// The seed of the token ids of the model test.
    private static let tokenSeed: UInt64 = 20_261_222

    /// The largest gap of plain decode's two best logits at which the verify
    /// may pick the second one: the two paths round a near tie apart.
    private static let nearTieLogitGap: Float = 1e-3

    /// The number of best logits of one row that the near-tie rule reads.
    private static let topCandidateCount = 2

    /// The traced verify step gives the kernel output with a causal mask and
    /// with no mask, and the SDPA of MLX with a mask array, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func theTracedVerifyStepGivesTheKernelOutputOrTheSDPA(
        dtype: DType, mask: AttentionMaskKind
    ) throws {
        try Self.fixture.expectTheTracedStepGivesTheRouteOutput(
            cachedKeys: Self.cachedKeys, tokens: Self.verifyTokens, dtype: dtype, mask: mask,
            picksFrom: VerifyAttentionKernelTests.kernels, seeds: Self.seeds)
    }

    /// The general attention body of a verify gives the kernel output with a
    /// causal mask and with no mask, and the SDPA of MLX with a mask array,
    /// in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func theGeneralAttentionBodyOfAVerifyGivesTheKernelOutputOrTheSDPA(
        dtype: DType, mask: AttentionMaskKind
    ) throws {
        try Self.fixture.expectTheAttentionBodyGivesTheRouteOutput(
            cachedKeys: Self.cachedKeys, tokens: Self.verifyTokens, dtype: dtype, mask: mask,
            picksFrom: VerifyAttentionKernelTests.kernels, seeds: Self.seeds)
    }

    /// The verify forward of a bfloat16 model over a long cache, as the MTP
    /// iterator runs it, gives the logits of plain decode over the same
    /// tokens inside the bfloat16 tolerance, and the greedy token of plain
    /// decode in each row up to a near tie.
    @Test func theVerifyForwardGivesThePlainDecodeLogits() throws {
        let model = Qwen35TextModel(try Self.fixture.configuration())
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        eval(model)
        let tokens = MLXRandom.randInt(
            0 ..< model.vocabularySize, [1, Self.cachedKeys + Self.verifyTokens],
            key: MLXRandom.key(Self.tokenSeed))
        let context = tokens[0..., ..<Self.cachedKeys]
        let draft = tokens[0..., Self.cachedKeys...]

        let verify = Self.verifyLogits(model, draft: draft, cache: try Self.prefilled(model, context))
        let decode = Self.decodeLogits(model, draft: draft, cache: try Self.prefilled(model, context))

        let gap = AttentionKernelTolerance.worstGap(verify, decode)
        #expect(gap <= AttentionKernelTolerance.bfloat16, "worst gap \(gap)")
        Self.expectTheGreedyTokensUpToANearTie(verify: verify, decode: decode)
    }

    /// A cache of `model` that holds `context`.
    private static func prefilled(_ model: Qwen35TextModel, _ context: MLXArray) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        eval(model(context, cache: cache))
        return cache
    }

    /// The logits of one verify forward of `draft`: the drafter state asked
    /// of the target and the recurrent snapshots kept, as the MTP iterator
    /// asks them.
    private static func verifyLogits(
        _ model: Qwen35TextModel, draft: MLXArray, cache: [KVCache]
    ) -> MLXArray {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        return model(LMInput.Text(tokens: draft), cache: cache, state: state).logits
            .asType(.float32)
    }

    /// The logits of plain decode over `draft`, one token at a time.
    private static func decodeLogits(
        _ model: Qwen35TextModel, draft: MLXArray, cache: [KVCache]
    ) -> MLXArray {
        let rows = (0 ..< draft.dim(1)).map { row in
            model(LMInput.Text(tokens: draft[0..., row ..< row + 1]), cache: cache, state: nil)
                .logits
        }
        return concatenated(rows, axis: 1).asType(.float32)
    }

    /// Checks that each row of the verify picks plain decode's greedy token,
    /// or the second one when the two best logits of plain decode stand
    /// within ``nearTieLogitGap``.
    private static func expectTheGreedyTokensUpToANearTie(verify: MLXArray, decode: MLXArray) {
        let verifyTokens = argMax(verify[0], axis: -1).asType(.int32).asArray(Int32.self)
        let decodeTokens = argMax(decode[0], axis: -1).asType(.int32).asArray(Int32.self)
        let best = sorted(decode[0], axis: -1)[0..., (-topCandidateCount)...]
        let gaps = (best[0..., 1] - best[0..., 0]).asArray(Float.self)
        for (row, (verifyToken, decodeToken)) in zip(verifyTokens, decodeTokens).enumerated() {
            #expect(
                verifyToken == decodeToken || gaps[row] < nearTieLogitGap,
                "row \(row): verify \(verifyToken), decode \(decodeToken), gap \(gaps[row])")
        }
    }
}
