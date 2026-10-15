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

    /// The query and key count of the routing tests: more than the fewest
    /// queries the kernel reads.
    private static let routedTokens = 96

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_115, keys: 20_261_116, values: 20_261_117)

    /// The check of the kernel with one mask.
    private static func canRun(
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> AttentionKernelCall.Check {
        { PrefillAttentionKernel.canRun(queries: $0, keys: $1, values: $2, mask: mask) }
    }

    /// The kernel under test with the mask of a causal flag. The Qwen3.5
    /// prefill suite also reads it.
    static func call(causal: Bool) -> AttentionKernelCall {
        AttentionKernelCall(
            canRun: canRun(mask: AttentionKernelOperands.maskMode(causal: causal)),
            attention: {
                PrefillAttentionKernel.attention(
                    queries: $0, keys: $1, values: $2, scale: $3, causal: causal)
            },
            causal: causal)
    }

    /// An array mask over the query and key count of the routing tests.
    private static func arrayMask() -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(MLXArray.zeros([routedTokens, routedTokens], dtype: .bfloat16))
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

    /// The kernel reads only a prefill block of enough query tokens: below
    /// ``PrefillAttentionKernel/fewestQueries`` the other routes are faster.
    @Test func theKernelReadsOnlyEnoughQueries() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let enough = Self.random.qwen38(queries: fewest, keys: fewest)
        let few = Self.random.qwen38(queries: fewest - 1, keys: fewest)

        Self.call(causal: true).expectReads([enough], skips: [few])
    }

    /// The kernel reads only the head shapes it holds: a head width of 256
    /// and a whole number of threadgroup head groups for each KV head.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest)
        let narrow = operands.firstHeadNumbers(128)
        let twoPerKVHead = operands.firstQueryHeads(8)
        let fourPerKVHead = operands.firstQueryHeads(16)

        Self.call(causal: true).expectReads(
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

    /// The kernel reads the three float dtypes, with a causal mask over at
    /// least as many keys as queries, or with no mask.
    @Test func theKernelReadsOnlyTheDtypesAndMasksItHolds() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let typed = AttentionKernelTolerance.dtypes.map {
            Self.random.qwen38(queries: fewest, keys: fewest, dtype: $0)
        }
        let fewerKeys = typed[0].firstKeys(fewest - 1)
        let routed = Self.random.qwen38(queries: Self.routedTokens, keys: Self.routedTokens)

        Self.call(causal: true).expectReads(typed, skips: [fewerKeys])
        Self.call(causal: false).expectReads([fewerKeys], skips: [])
        #expect(!routed.runs(by: Self.canRun(mask: Self.arrayMask())))
    }

    /// A prefill call gives the kernel output, bit for bit.
    @Test func aPrefillCallGivesTheKernelOutput() {
        let operands = Self.random.qwen38(queries: Self.routedTokens, keys: 160)

        operands.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, picks: Self.call(causal: true))
    }

    /// A call the kernel does not read keeps the route of
    /// ``DecodeAttentionKernel``, bit for bit: a decode step over a long
    /// cache, and a prefill call with an array mask.
    @Test func aCallTheKernelDoesNotReadKeepsTheDecodeRoute() {
        let decode = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let prefill = Self.random.qwen38(queries: Self.routedTokens, keys: Self.routedTokens)

        decode.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention,
            picks: DecodeAttentionKernelTests.call)
        prefill.expectTheRoute(
            PrefillAttentionKernel.scaledDotProductAttention, keepsTheSDPAWith: Self.arrayMask())
    }
}
