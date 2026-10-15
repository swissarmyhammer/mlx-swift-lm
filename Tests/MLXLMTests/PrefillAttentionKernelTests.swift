// Copyright © 2026 Apple Inc.
//
// The contract of ``PrefillAttentionKernel``.
//
// A prefill block of a grouped-query model with a head width of 256 reads its
// cached keys through this kernel. The kernel adds the keys in tiles with an
// online softmax, thus its sums run in another order than the SDPA of MLX, and
// the two paths are not bit-exact. The kernel output must stay inside a small
// tolerance of the SDPA of MLX over the same inputs in float32.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct PrefillAttentionKernelTests {

    /// The shapes of the tolerance test.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest queries the kernel reads,
    ///    with no cached prefix.
    /// 2. The Qwen3.8 shape with a cached prefix, and query and key counts
    ///    that do not fill the last threadgroup and the last key tile.
    /// 3. Two sequences of 3 query heads for each KV head.
    static let attentionCases = [
        AttentionKernelCase.qwen38(queries: 64, keys: 64, capacity: 256),
        AttentionKernelCase.qwen38(queries: 203, keys: 1203, capacity: 1280),
        AttentionKernelCase(
            batch: 2, queryHeads: 6, kvHeads: 2, headDim: AttentionKernelCase.qwen38HeadDim,
            queries: 100, keys: 100, capacity: 100),
    ]

    /// The masks the kernel reads.
    static let causalMasks = [true, false]

    /// The query count of the routing tests: more than the fewest queries
    /// the kernel reads.
    private static let routedTokens = 96

    /// The key count of the routing tests: a cached prefix ahead of the
    /// queries.
    private static let routedKeys = 160

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_115, keys: 20_261_116, values: 20_261_117)

    /// The check of the kernel with one mask kind.
    private static func canRun(_ kind: AttentionMaskKind) -> AttentionKernelCall.Check {
        {
            PrefillAttentionKernel.canRun(
                queries: $0, keys: $1, values: $2, mask: kind.mode(queries: $0, keys: $1))
        }
    }

    /// The kernel under test with the mask of a causal flag. The Qwen3.5
    /// prefill suite also reads it.
    static func call(causal: Bool) -> AttentionKernelCall {
        AttentionKernelCall(
            canRun: canRun(AttentionMaskKind(causal: causal)),
            attention: {
                PrefillAttentionKernel.attention(
                    queries: $0, keys: $1, values: $2, scale: $3, causal: causal)
            },
            causal: causal)
    }

    /// The kernel that the route picks for each mask kind: the kernel with a
    /// causal mask, and the kernel with no mask. The route gives a mask array
    /// to the route of ``DecodeAttentionKernel``. The Qwen3.5 prefill suite
    /// also reads it.
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

    /// The positive controls of the tolerance test: a kernel that ignored the
    /// causal mask, or that did not write the last query token, would break
    /// the tolerance.
    @Test(arguments: attentionCases)
    func aWrongMaskOrAMissingTokenMovesTheOutputPastTheTolerance(
        attentionCase: AttentionKernelCase
    ) {
        let operands = Self.random.operands(attentionCase, dtype: .float32)
        let causal = operands.reference(causal: true)
        let open = operands.reference(causal: false)
        let lastToken = attentionCase.queries - 1
        let written = (MLXArray(0 ..< attentionCase.queries) .< lastToken)
            .reshaped(1, 1, attentionCase.queries, 1)
        let missingToken = causal * written

        AttentionKernelTolerance.expectPastEachTolerance(open, causal)
        AttentionKernelTolerance.expectPastEachTolerance(missingToken, causal)
    }

    /// The kernel reads only a prefill block of enough query tokens, in each
    /// dtype and with each mask it reads: below
    /// ``PrefillAttentionKernel/fewestQueries`` the other routes are faster.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func theKernelReadsOnlyEnoughQueries(dtype: DType, causal: Bool) {
        let fewest = PrefillAttentionKernel.fewestQueries
        let enough = Self.random.qwen38(queries: fewest, keys: fewest, dtype: dtype)
        let few = Self.random.qwen38(queries: fewest - 1, keys: fewest, dtype: dtype)

        Self.call(causal: causal).expectReads([enough], skips: [few])
    }

    /// The kernel reads only the head shapes it holds, in each dtype and with
    /// each mask it reads: a head width of 256 and a whole number of
    /// threadgroup head groups for each KV head.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func theKernelReadsOnlyTheHeadShapesItHolds(dtype: DType, causal: Bool) {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest, dtype: dtype)
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

    /// In each dtype, the kernel reads a causal mask over at least as many
    /// keys as queries, and no mask over each key count. It does not read a
    /// mask array.
    @Test(arguments: AttentionKernelTolerance.dtypes)
    func theKernelReadsOnlyTheMasksItHolds(dtype: DType) {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest, dtype: dtype)
        let fewerKeys = operands.firstKeys(fewest - 1)

        Self.call(causal: true).expectReads([operands], skips: [fewerKeys])
        Self.call(causal: false).expectReads([operands, fewerKeys], skips: [])
        for read in [operands, fewerKeys] {
            #expect(!read.runs(by: Self.canRun(.array)), "\(read.testDescription)")
        }
    }

    /// The kernel does not read a dtype outside the float dtypes, or keys and
    /// values in another dtype than the queries, with each mask it reads.
    @Test(arguments: causalMasks)
    func theKernelRefusesOtherDtypes(causal: Bool) {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest)

        Self.call(causal: causal).expectReads([], skips: operands.inRefusedDtypes())
    }

    /// A prefill call gives the kernel output bit for bit with a causal mask
    /// and with no mask, and keeps the SDPA of MLX bit for bit with a mask
    /// array, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aPrefillCallGivesTheKernelOutputOrTheSDPA(dtype: DType, mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: Self.routedKeys, dtype: dtype)

        operands.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, with: mask, picksFrom: Self.kernels)
    }

    /// A causal prefill call over fewer keys than queries keeps the SDPA of
    /// MLX bit for bit, and the same call with no mask gives the kernel
    /// output, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aCallOverFewerKeysThanQueriesKeepsTheSDPAWhenCausal(
        dtype: DType, mask: AttentionMaskKind
    ) {
        let operands = Self.random.qwen38(
            queries: Self.routedTokens, keys: Self.routedTokens - 1, dtype: dtype)

        operands.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, with: mask,
            picksFrom: Self.kernels.filter { !$0.value.causal })
    }

    /// A call of too few queries keeps the route of ``DecodeAttentionKernel``
    /// bit for bit, in each dtype and with each mask kind: a decode step over
    /// a long cache picks the decode kernel when it has no mask, and a block
    /// below ``PrefillAttentionKernel/fewestQueries`` keeps the SDPA of MLX.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func aCallOfTooFewQueriesKeepsTheDecodeRoute(dtype: DType, mask: AttentionMaskKind) {
        let fewest = PrefillAttentionKernel.fewestQueries
        let decode = Self.random.qwen38(
            queries: 1, keys: DecodeAttentionKernel.fewestKeys, dtype: dtype)
        let fewQueries = Self.random.qwen38(queries: fewest - 1, keys: fewest, dtype: dtype)

        decode.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, with: mask,
            picksFrom: DecodeAttentionKernelTests.kernels)
        fewQueries.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }

    /// A prefill call with keys and values in another dtype than the queries
    /// keeps the SDPA of MLX bit for bit, with each mask kind.
    @Test(arguments: AttentionMaskKind.allCases)
    func aCallOfMixedDtypesKeepsTheSDPA(mask: AttentionMaskKind) {
        let operands = Self.random.qwen38(queries: Self.routedTokens, keys: Self.routedKeys)

        operands.withKeysInAnotherDtype().expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: mask)
    }
}
