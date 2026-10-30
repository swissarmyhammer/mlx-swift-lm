// Copyright © 2026 Apple Inc.
//
// The contract of ``VerifyAttentionKernel``.
//
// A speculative verify of a grouped-query model with a head width of 256
// reads its few query tokens over a long key cache through this kernel. The
// kernel adds the keys in blocks and tiles with an online softmax, thus its
// sums run in another order than the SDPA of MLX, and the two paths are not
// bit-exact. The kernel output must stay inside a small tolerance of the SDPA
// of MLX over the same inputs in float32.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct VerifyAttentionKernelTests {

    /// The shapes of the tolerance test.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest queries and the fewest
    ///    keys the kernel reads.
    /// 2. The Qwen3.8 shape of a verify of five tokens, with a last block
    ///    shorter than the others.
    /// 3. The Qwen3.8 shape at the most queries the kernel reads, over many
    ///    blocks.
    /// 4. Two sequences of 3 query heads for each KV head.
    static let attentionCases = [
        AttentionKernelCase.qwen38(
            queries: VerifyAttentionKernel.fewestQueries, keys: DecodeAttentionKernel.fewestKeys,
            capacity: 1280),
        AttentionKernelCase.qwen38(queries: 5, keys: 3001, capacity: 3072),
        AttentionKernelCase.qwen38(
            queries: VerifyAttentionKernel.mostQueries, keys: 16_500, capacity: 16_640),
        AttentionKernelCase(
            batch: 2, queryHeads: 6, kvHeads: 2, headDim: AttentionKernelCase.qwen38HeadDim,
            queries: 3, keys: 1500, capacity: 1536),
    ]

    /// The masks the kernel reads.
    static let causalMasks = [true, false]

    /// The query count of the routing tests: a verify of a block of five
    /// tokens, as the MTP head of Qwen3.8 drafts.
    private static let routedTokens = 5

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_215, keys: 20_261_216, values: 20_261_217)

    /// The check of the kernel with one mask kind.
    private static func canRun(_ kind: AttentionMaskKind) -> AttentionKernelCall.Check {
        {
            VerifyAttentionKernel.canRun(
                queries: $0, keys: $1, values: $2, mask: kind.mode(queries: $0, keys: $1))
        }
    }

    /// The kernel under test with the mask of a causal flag. The Qwen3.5
    /// verify suite also reads it.
    static func call(causal: Bool) -> AttentionKernelCall {
        AttentionKernelCall(
            canRun: canRun(AttentionMaskKind(causal: causal)),
            attention: {
                VerifyAttentionKernel.attention(
                    queries: $0, keys: $1, values: $2, scale: $3, causal: causal)
            },
            causal: causal)
    }

    /// The kernel that the route picks for each mask kind: the kernel with a
    /// causal mask, and the kernel with no mask. The route gives a mask array
    /// to the SDPA of MLX. The Qwen3.5 verify suite also reads it.
    static var kernels: AttentionKernelTable {
        Dictionary(uniqueKeysWithValues: causalMasks.map(call).map { ($0.maskKind, $0) })
    }

    @Test(arguments: attentionCases, AttentionKernelTolerance.dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) throws {
        let operands = Self.random.operands(attentionCase, dtype: dtype)

        for causal in Self.causalMasks {
            try operands.expectInsideTheTolerance(of: Self.call(causal: causal))
        }
    }

    /// The positive control of the tolerance test: a kernel that did not read
    /// the keys of its last block would break the tolerance.
    @Test(arguments: attentionCases)
    func theKeysOfTheLastBlockMoveTheOutputPastTheTolerance(attentionCase: AttentionKernelCase) {
        let operands = Self.random.operands(attentionCase, dtype: .float32)
        let blocks = VerifyAttentionKernel.blockCount(keys: attentionCase.keys)
        let keysPerBlock = DecodeAttentionKernel.keysPerBlock(
            keys: attentionCase.keys, blocks: blocks)
        let lastBlock = attentionCase.keys - (blocks - 1) * keysPerBlock

        let full = operands.reference(causal: true)
        let short = operands.firstKeys(attentionCase.keys - lastBlock).reference(causal: true)

        AttentionKernelTolerance.expectPastEachTolerance(short, full)
    }

    /// The kernel reads only the few query tokens of a verify, in each dtype
    /// and with each mask it reads: one query keeps the route of
    /// ``DecodeAttentionKernel``, two queries keep the SDPA of MLX, which is
    /// faster for them, and more queries than one SIMD-group matrix holds
    /// keep the other routes.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func theKernelReadsOnlyTheQueryCountsOfAVerify(dtype: DType, causal: Bool) {
        let keys = DecodeAttentionKernel.fewestKeys
        let fewest = Self.random.qwen38(
            queries: VerifyAttentionKernel.fewestQueries, keys: keys, dtype: dtype)
        let most = Self.random.qwen38(
            queries: VerifyAttentionKernel.mostQueries, keys: keys, dtype: dtype)
        let one = Self.random.qwen38(queries: 1, keys: keys, dtype: dtype)
        let tooFew = Self.random.qwen38(
            queries: VerifyAttentionKernel.fewestQueries - 1, keys: keys, dtype: dtype)
        let tooMany = Self.random.qwen38(
            queries: VerifyAttentionKernel.mostQueries + 1, keys: keys, dtype: dtype)

        Self.call(causal: causal).expectReads(
            [fewest, most], skips: [one, tooFew, tooMany])
    }

    /// The kernel reads only a long cache, in each dtype and with each mask
    /// it reads: below ``DecodeAttentionKernel/fewestKeys`` the SDPA of MLX
    /// is as fast.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func theKernelReadsOnlyALongCache(dtype: DType, causal: Bool) {
        let fewest = DecodeAttentionKernel.fewestKeys
        let enough = Self.random.qwen38(queries: Self.routedTokens, keys: fewest, dtype: dtype)
        let short = Self.random.qwen38(queries: Self.routedTokens, keys: fewest - 1, dtype: dtype)

        Self.call(causal: causal).expectReads([enough], skips: [short])
    }

    /// The kernel reads only the head shapes it holds, in each dtype and with
    /// each mask it reads: a head width of 256 and a whole number of
    /// threadgroup head groups for each KV head.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func theKernelReadsOnlyTheHeadShapesItHolds(dtype: DType, causal: Bool) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys, dtype: dtype)
        let narrow = operands.firstHeadNumbers(128)
        let twoPerKVHead = operands.firstQueryHeads(8)
        let fourPerKVHead = operands.firstQueryHeads(16)

        Self.call(causal: causal).expectReads(
            [operands], skips: [narrow, twoPerKVHead, fourPerKVHead])
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the narrow loads of the kernel and stay inside
    /// the tolerance too, in each dtype and with each mask.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func keysWithSpacedHeadNumbersStayInsideTheTolerance(dtype: DType, causal: Bool) throws {
        try Self.random.operands(Self.attentionCases[1], dtype: dtype)
            .expectSpacedHeadNumbersStayInsideTheTolerance(of: Self.call(causal: causal))
    }

    /// In each dtype, the kernel reads a causal mask and no mask. It does not
    /// read a mask array.
    @Test(arguments: AttentionKernelTolerance.dtypes)
    func theKernelReadsOnlyTheMasksItHolds(dtype: DType) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys, dtype: dtype)

        for causal in Self.causalMasks {
            Self.call(causal: causal).expectReads([operands], skips: [])
        }
        #expect(!operands.runs(by: Self.canRun(.array)), "\(operands.testDescription)")
    }

    /// The kernel does not read a dtype outside the float dtypes, or keys and
    /// values in another dtype than the queries, with each mask it reads.
    @Test(arguments: causalMasks)
    func theKernelRefusesOtherDtypes(causal: Bool) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys)

        Self.call(causal: causal).expectReads([], skips: operands.inRefusedDtypes())
    }

    /// Key counts at each edge of the block count rule.
    static let blockRuleKeys = [1024, 4095, 4096, 8192, 16_384, 32_767, 32_768, 131_072]

    /// The block count of each key count of ``blockRuleKeys``.
    static let blockRuleBlocks = [4, 4, 8, 16, 32, 32, 64, 64]

    /// The block count grows with the keys, as the power of two at or below
    /// `keys / 512`, from 4 to 64 blocks.
    @Test(arguments: zip(blockRuleKeys, blockRuleBlocks))
    func theBlockCountFollowsTheKeys(keys: Int, blocks: Int) {
        #expect(VerifyAttentionKernel.blockCount(keys: keys) == blocks)
    }

    /// A verify call through the attention route of a model gives the kernel
    /// output bit for bit with a causal mask and with no mask, and keeps the
    /// SDPA of MLX bit for bit with a mask array, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aVerifyCallGivesTheKernelOutputOrTheSDPA(dtype: DType, mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys + 77, dtype: dtype)

        operands.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, with: mask, picksFrom: Self.kernels)
    }

    /// A verify call over a short cache keeps the SDPA of MLX bit for bit,
    /// in each dtype and with each mask kind.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aVerifyCallOverAShortCacheKeepsTheSDPA(dtype: DType, mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys - 1, dtype: dtype)

        operands.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }

    /// A verify call with keys and values in another dtype than the queries
    /// keeps the SDPA of MLX bit for bit, with each mask kind.
    @Test(arguments: AttentionMaskKind.allCases)
    func aCallOfMixedDtypesKeepsTheSDPA(mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: DecodeAttentionKernel.fewestKeys)

        operands.withKeysInAnotherDtype().expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }
}
