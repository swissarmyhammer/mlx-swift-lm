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

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_115, keys: 20_261_116, values: 20_261_117)

    /// The kernel output over the operands.
    private static func kernel(_ operands: AttentionKernelOperands, causal: Bool) -> MLXArray {
        PrefillAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, causal: causal)
    }

    /// Tells if the kernel runs the operands with the mask of a causal flag.
    private static func runs(_ operands: AttentionKernelOperands, causal: Bool) -> Bool {
        PrefillAttentionKernel.canRun(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            mask: AttentionKernelOperands.maskMode(causal: causal))
    }

    @Test(arguments: attentionCases, AttentionKernelTolerance.dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) {
        let operands = Self.random.operands(attentionCase, dtype: dtype)

        for causal in Self.causalMasks {
            #expect(Self.runs(operands, causal: causal))
            operands.expectInsideTheTolerance(
                kernel: Self.kernel(operands, causal: causal), causal: causal)
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

        #expect(AttentionKernelTolerance.worstGap(open, causal) > AttentionKernelTolerance.bfloat16)
        #expect(
            AttentionKernelTolerance.worstGap(missingToken, causal)
                > AttentionKernelTolerance.bfloat16)
    }

    /// The kernel reads only a prefill block of enough query tokens: below
    /// ``PrefillAttentionKernel/fewestQueries`` the other routes are faster.
    @Test func theKernelReadsOnlyEnoughQueries() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let enough = Self.random.qwen38(queries: fewest, keys: fewest)
        let few = Self.random.qwen38(queries: fewest - 1, keys: fewest)

        #expect(Self.runs(enough, causal: true))
        #expect(!Self.runs(few, causal: true))
    }

    /// The kernel reads only the head shapes it holds: a head width of 256
    /// and a whole number of threadgroup head groups for each KV head.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest)
        let narrow = AttentionKernelOperands(
            queries: operands.queries[.ellipsis, ..<128], keys: operands.keys[.ellipsis, ..<128],
            values: operands.values[.ellipsis, ..<128], scale: operands.scale)
        let twoPerKVHead = AttentionKernelOperands(
            queries: operands.queries[0..., ..<8], keys: operands.keys, values: operands.values,
            scale: operands.scale)
        let fourPerKVHead = AttentionKernelOperands(
            queries: operands.queries[0..., ..<16], keys: operands.keys, values: operands.values,
            scale: operands.scale)

        #expect(Self.runs(operands, causal: true))
        #expect(!Self.runs(narrow, causal: true))
        #expect(!Self.runs(twoPerKVHead, causal: true))
        #expect(!Self.runs(fourPerKVHead, causal: true))
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the narrow loads of the kernel and stay inside
    /// the tolerance too, in each dtype and with each mask.
    @Test(arguments: AttentionKernelTolerance.dtypes, causalMasks)
    func keysWithSpacedHeadNumbersStayInsideTheTolerance(dtype: DType, causal: Bool) {
        Self.random.operands(Self.attentionCases[1], dtype: dtype)
            .expectSpacedHeadNumbersStayInsideTheTolerance(causal: causal) {
                Self.kernel($0, causal: causal)
            }
    }

    /// The kernel reads the three float dtypes, with a causal mask over at
    /// least as many keys as queries, or with no mask.
    @Test func theKernelReadsOnlyTheDtypesAndMasksItHolds() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.random.qwen38(queries: fewest, keys: fewest)
        let fewerKeys = operands.firstKeys(fewest - 1)
        let arrayMask = MLXArray.zeros([fewest, fewest], dtype: .bfloat16)

        for dtype in AttentionKernelTolerance.dtypes {
            let typed = Self.random.qwen38(queries: fewest, keys: fewest, dtype: dtype)
            #expect(Self.runs(typed, causal: true), "\(dtype)")
        }
        #expect(!Self.runs(fewerKeys, causal: true))
        #expect(Self.runs(fewerKeys, causal: false))
        #expect(
            !PrefillAttentionKernel.canRun(
                queries: operands.queries, keys: operands.keys, values: operands.values,
                mask: .array(arrayMask)))
    }

    /// A prefill call gives the kernel output, bit for bit.
    @Test func aPrefillCallGivesTheKernelOutput() {
        let operands = Self.random.qwen38(queries: 96, keys: 160)

        let routed = PrefillAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)

        #expect(arrayEqual(routed, Self.kernel(operands, causal: true)).item(Bool.self))
    }

    /// A call the kernel does not read keeps the route of
    /// ``DecodeAttentionKernel``, bit for bit: a decode step over a long
    /// cache, and a prefill call with an array mask.
    @Test func aCallTheKernelDoesNotReadKeepsTheDecodeRoute() {
        let decode = Self.random.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let prefill = Self.random.qwen38(queries: 96, keys: 96)
        let arrayMask = MLXArray.zeros([96, 96], dtype: .bfloat16)

        let routedDecode = PrefillAttentionKernel.scaledDotProductAttention(
            queries: decode.queries, keys: decode.keys, values: decode.values,
            scale: decode.scale, mask: .none)
        let decodeKernel = DecodeAttentionKernel.attention(
            queries: decode.queries, keys: decode.keys, values: decode.values, scale: decode.scale)
        let routedMasked = PrefillAttentionKernel.scaledDotProductAttention(
            queries: prefill.queries, keys: prefill.keys, values: prefill.values,
            scale: prefill.scale, mask: .array(arrayMask))
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: prefill.queries, keys: prefill.keys, values: prefill.values,
            scale: prefill.scale, mask: .array(arrayMask))

        #expect(arrayEqual(routedDecode, decodeKernel).item(Bool.self))
        #expect(arrayEqual(routedMasked, sdpa).item(Bool.self))
    }
}
