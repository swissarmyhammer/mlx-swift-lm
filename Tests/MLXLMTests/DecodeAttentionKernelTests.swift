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

    /// The kernel output over the operands.
    private static func kernel(_ operands: AttentionKernelOperands) -> MLXArray {
        DecodeAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale)
    }

    /// Tells if the kernel runs over the operands.
    private static func runs(_ operands: AttentionKernelOperands) -> Bool {
        DecodeAttentionKernel.canRun(
            queries: operands.queries, keys: operands.keys, values: operands.values)
    }

    @Test(arguments: attentionCases, AttentionKernelTolerance.dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) {
        let operands = Self.random.operands(attentionCase, dtype: dtype)

        #expect(Self.runs(operands))
        operands.expectInsideTheTolerance(kernel: Self.kernel(operands), causal: false)
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the strided read of the kernel and stay inside
    /// the tolerance too, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes)
    func keysWithSpacedHeadNumbersStayInsideTheTolerance(dtype: DType) {
        Self.random.operands(Self.attentionCases[0], dtype: dtype)
            .expectSpacedHeadNumbersStayInsideTheTolerance(causal: false, kernel: Self.kernel)
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

        #expect(AttentionKernelTolerance.worstGap(short, full) > AttentionKernelTolerance.bfloat16)
    }

    /// The kernel reads one query over enough keys: below
    /// ``DecodeAttentionKernel/fewestKeys`` the SDPA of MLX is faster.
    @Test func theKernelReadsOnlyOneQueryOverEnoughKeys() {
        let enough = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let few = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys - 1)
        let twoQueries = concatenated(
            [enough.queries, enough.queries], axis: AttentionKernelOperands.rowAxis)

        #expect(Self.runs(enough))
        #expect(!Self.runs(few))
        #expect(
            !DecodeAttentionKernel.canRun(
                queries: twoQueries, keys: enough.keys, values: enough.values))
    }

    /// The kernel reads only head shapes it holds in registers: a whole
    /// number of query heads for each KV head, an even number of them, and a
    /// head width that the lanes of a SIMD group share evenly.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let oddGroup = operands.queries[0..., ..<12]
        let unevenGroup = operands.queries[0..., ..<10]
        let narrowKeys = operands.keys[.ellipsis, ..<200]

        #expect(
            !DecodeAttentionKernel.canRun(
                queries: oddGroup, keys: operands.keys, values: operands.values))
        #expect(
            !DecodeAttentionKernel.canRun(
                queries: unevenGroup, keys: operands.keys, values: operands.values))
        #expect(
            !DecodeAttentionKernel.canRun(
                queries: operands.queries[.ellipsis, ..<200], keys: narrowKeys,
                values: operands.values[.ellipsis, ..<200]))
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

    /// A call the kernel does not read keeps the SDPA of MLX, bit for bit.
    @Test func aCallTheKernelDoesNotReadKeepsTheSDPA() {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys - 1)

        let routed = DecodeAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .none)
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .none)

        #expect(arrayEqual(routed, sdpa).item(Bool.self))
    }

    /// A call the kernel reads gives the kernel output, bit for bit, and a
    /// masked call keeps the SDPA of MLX.
    @Test func aCallTheKernelReadsGivesTheKernelOutput() {
        let operands = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)

        let routed = DecodeAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .none)
        let masked = DecodeAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)

        #expect(arrayEqual(routed, Self.kernel(operands)).item(Bool.self))
        #expect(arrayEqual(masked, sdpa).item(Bool.self))
    }
}
