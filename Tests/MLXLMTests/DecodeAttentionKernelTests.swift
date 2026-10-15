// Copyright © 2026 Apple Inc.
//
// The contract of ``DecodeAttentionKernel``.
//
// A decode step of a grouped-query model reads its cached keys through this
// kernel when the cache holds enough keys. The kernel adds the keys in blocks
// with an online softmax, thus its sums run in another order than the SDPA of
// MLX, and the two paths are not bit-exact. The kernel output must stay inside
// a small tolerance of the SDPA of MLX over the same inputs in float32.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct DecodeAttentionKernelTests {

    /// The shapes of the tolerance test, each of one query token.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest keys the kernel reads.
    /// 2. The Qwen3.8 shape with a last block shorter than the others.
    /// 3. The Qwen3.8 shape at the largest block count.
    /// 4. Two sequences of a small model: 2 query heads for each KV head, a
    ///    head of 128 numbers.
    static let attentionCases = [
        AttentionKernelCase.qwen38(queries: 1, keys: 1024, capacity: 1280),
        AttentionKernelCase.qwen38(queries: 1, keys: 3001, capacity: 3072),
        AttentionKernelCase.qwen38(queries: 1, keys: 16_500, capacity: 16_640),
        AttentionKernelCase(
            batch: 2, queryHeads: 4, kvHeads: 2, headDim: 128, queries: 1, keys: 1500,
            capacity: 1536),
    ]

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_015, keys: 20_261_016, values: 20_261_017)

    /// The kernel under test. It reads no mask. The prefill suite and the
    /// Qwen3.5 decode suite also read it.
    static var call: AttentionKernelCall {
        AttentionKernelCall(
            canRun: DecodeAttentionKernel.canRun, attention: DecodeAttentionKernel.attention,
            causal: false)
    }

    /// The kernel that the route picks for each mask kind: the kernel for no
    /// mask. The route gives a causal mask and a mask array to the SDPA of
    /// MLX. The prefill suite and the Qwen3.5 decode suite also read it.
    static var kernels: AttentionKernelTable {
        [call.maskKind: call]
    }

    @Test(arguments: attentionCases, AttentionKernelTolerance.dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) throws {
        try Self.random.operands(attentionCase, dtype: dtype)
            .expectInsideTheTolerance(of: Self.call)
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the strided read of the kernel and stay inside
    /// the tolerance too, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes)
    func keysWithSpacedHeadNumbersStayInsideTheTolerance(dtype: DType) throws {
        try Self.random.operands(Self.attentionCases[0], dtype: dtype)
            .expectSpacedHeadNumbersStayInsideTheTolerance(of: Self.call)
    }

    /// Each dtype that the kernels read has a tolerance, and a dtype that they
    /// do not read has no tolerance. Thus a test in that dtype fails, and does
    /// not use the tolerance of another dtype.
    @Test func onlyTheKernelDtypesHaveATolerance() {
        for dtype in AttentionKernelTolerance.dtypes {
            #expect(AttentionKernelTolerance.of(dtype) != nil, "\(dtype)")
        }
        #expect(AttentionKernelTolerance.of(AttentionKernelTolerance.refusedDtype) == nil)
    }

    /// The positive control of the tolerance test: a kernel that did not read
    /// the keys of its last block would break the tolerance.
    @Test(arguments: attentionCases)
    func theKeysOfTheLastBlockMoveTheOutputPastTheTolerance(attentionCase: AttentionKernelCase) {
        let operands = Self.random.operands(attentionCase, dtype: .float32)
        let blocks = DecodeAttentionKernel.blockCount(keys: attentionCase.keys)
        let lastBlock = DecodeAttentionKernel.keysPerBlock(keys: attentionCase.keys, blocks: blocks)

        let full = operands.reference(causal: false)
        let short = operands.firstKeys(attentionCase.keys - lastBlock).reference(causal: false)

        AttentionKernelTolerance.expectPastEachTolerance(short, full)
    }

    /// The kernel reads one query over enough keys, in each dtype: below
    /// ``DecodeAttentionKernel/fewestKeys`` the SDPA of MLX is faster.
    @Test(arguments: AttentionKernelTolerance.dtypes)
    func theKernelReadsOnlyOneQueryOverEnoughKeys(dtype: DType) {
        let fewest = DecodeAttentionKernel.fewestKeys
        let enough = Self.random.qwen38(queries: 1, keys: fewest, dtype: dtype)
        let few = Self.random.qwen38(queries: 1, keys: fewest - 1, dtype: dtype)
        let twoQueries = enough.withQueries(
            concatenated([enough.queries, enough.queries], axis: AttentionKernelOperands.rowAxis))

        Self.call.expectReads([enough], skips: [few, twoQueries])
    }

    /// The kernel does not read a dtype outside the float dtypes, or keys and
    /// values in another dtype than the queries.
    @Test func theKernelRefusesOtherDtypes() {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)

        Self.call.expectReads([], skips: operands.inRefusedDtypes())
    }

    /// The kernel reads only head shapes it holds in registers: a whole
    /// number of query heads for each KV head, an even number of them, and a
    /// head width that the lanes of a SIMD group share evenly.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let oddGroup = operands.firstQueryHeads(12)
        let unevenGroup = operands.firstQueryHeads(10)
        let narrow = operands.firstHeadNumbers(200)

        Self.call.expectReads([operands], skips: [oddGroup, unevenGroup, narrow])
    }

    /// Key counts at each edge of the block count rule.
    static let blockRuleKeys = [1024, 4096, 8191, 8192, 16_383, 16_384, 32_768, 131_072]

    /// The block count of each key count of ``blockRuleKeys``.
    static let blockRuleBlocks = [32, 32, 32, 64, 64, 128, 128, 128]

    /// The block count grows with the keys, as a power of two, from 32 to
    /// 128 blocks.
    @Test(arguments: zip(blockRuleKeys, blockRuleBlocks))
    func theBlockCountFollowsTheKeys(keys: Int, blocks: Int) {
        #expect(DecodeAttentionKernel.blockCount(keys: keys) == blocks)
    }

    /// A call over too few keys keeps the SDPA of MLX bit for bit, in each
    /// dtype and with each mask kind.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aCallOverTooFewKeysKeepsTheSDPA(dtype: DType, mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: 1, keys: DecodeAttentionKernel.fewestKeys - 1, dtype: dtype)

        operands.expectTheRoute(
            DecodeAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }

    /// A call over enough keys gives the kernel output bit for bit with no
    /// mask, and keeps the SDPA of MLX bit for bit with a causal mask or a
    /// mask array, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aCallOverEnoughKeysGivesTheKernelOutputOrTheSDPA(dtype: DType, mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: 1, keys: DecodeAttentionKernel.fewestKeys, dtype: dtype)

        operands.expectTheRoute(
            DecodeAttentionKernel.scaledDotProductAttention, with: mask, picksFrom: Self.kernels)
    }

    /// A call with keys and values in another dtype than the queries keeps
    /// the SDPA of MLX bit for bit, with each mask kind.
    @Test(arguments: AttentionMaskKind.allCases)
    func aCallOfMixedDtypesKeepsTheSDPA(mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)

        operands.withKeysInAnotherDtype().expectTheRoute(
            DecodeAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }
}
