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

    /// The tokens of the verify: the bonus token and four drafts, a block of
    /// five.
    private static let verifyTokens = 5

    /// The layer: a head of 256 numbers, the only width the kernel reads.
    private static let fixture = Qwen35AttentionLayerFixture(headDim: 256)

    /// The seed of each random operand of the layer tests.
    private static let seeds = Qwen35AttentionLayerFixture.Seeds(
        cache: 20_261_220, tokens: 20_261_221)

    /// The seed of the token ids of the model test.
    private static let tokenSeed: UInt64 = 20_261_222

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

        let verifyCache = try Self.prefilled(model, context)
        let decodeCache = try Self.prefilled(model, context)
        let verify = VerifyForwardLogits.verify(
            draft, after: nil, forward: Self.forward(model, over: verifyCache))
        let decode = VerifyForwardLogits.decode(
            draft, after: nil, forward: Self.forward(model, over: decodeCache))

        let gap = AttentionKernelTolerance.worstGap(verify, decode)
        #expect(gap <= AttentionKernelTolerance.bfloat16, "worst gap \(gap)")
        AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie(verify: verify, decode: decode)
    }

    /// A cache of `model` that holds `context`.
    private static func prefilled(_ model: Qwen35TextModel, _ context: MLXArray) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        eval(model(context, cache: cache))
        return cache
    }

    /// One forward of `model` over `cache`.
    private static func forward(
        _ model: Qwen35TextModel, over cache: [KVCache]
    ) -> VerifyForwardLogits.Forward {
        { tokens, state in model(LMInput.Text(tokens: tokens), cache: cache, state: state) }
    }
}
