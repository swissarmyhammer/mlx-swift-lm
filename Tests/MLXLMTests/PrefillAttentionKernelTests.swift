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

    /// One shape of the tolerance test.
    struct AttentionCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of query heads.
        let queryHeads: Int
        /// The number of key and value heads.
        let kvHeads: Int
        /// The number of query tokens.
        let queries: Int
        /// The number of cached keys, the new keys included.
        let keys: Int
        /// The number of rows the cache buffer holds. The keys are a slice of
        /// the first `keys` rows, as a `KVCacheSimple` gives them back.
        let capacity: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(queryHeads) heads over \(kvHeads) KV heads, "
                + "\(queries) queries over \(keys) keys of \(capacity) rows"
        }
    }

    /// The width of one head, the only width the kernel reads.
    private static let headDim = 256

    /// The shapes of the tolerance test.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest queries the kernel reads,
    ///    with no cached prefix.
    /// 2. The Qwen3.8 shape with a cached prefix, and query and key counts
    ///    that do not fill the last threadgroup and the last key tile.
    /// 3. Two sequences of 3 query heads for each KV head.
    static let attentionCases = [
        AttentionCase(batch: 1, queryHeads: 24, kvHeads: 4, queries: 64, keys: 64, capacity: 256),
        AttentionCase(
            batch: 1, queryHeads: 24, kvHeads: 4, queries: 203, keys: 1203, capacity: 1280),
        AttentionCase(batch: 2, queryHeads: 6, kvHeads: 2, queries: 100, keys: 100, capacity: 100),
    ]

    /// The dtypes of the tolerance test: the float32 of the synthetic tests
    /// and the bfloat16 of the published checkpoints.
    static let dtypes: [DType] = [.float32, .bfloat16]

    /// The masks the kernel reads.
    static let causalMasks = [true, false]

    /// The seed of each random operand.
    private enum Seed {
        static let queries: UInt64 = 20_261_115
        static let keys: UInt64 = 20_261_116
        static let values: UInt64 = 20_261_117
    }

    /// The axis a `(batch, heads, rows, width)` tensor holds its heads on.
    private static let headAxis = 1

    /// The axis a `(batch, heads, rows, width)` tensor holds its rows on.
    private static let rowAxis = 2

    /// The axis a `(batch, heads, rows, width)` tensor holds the numbers of
    /// one head on.
    private static let widthAxis = 3

    /// The operands of one attention call.
    struct Operands {
        /// The queries, shape `(batch, queryHeads, queries, headDim)`, a
        /// transposed view of a `(batch, queries, queryHeads, headDim)`
        /// buffer, as the model gives them.
        let queries: MLXArray
        /// The cached keys, a slice of shape `(batch, kvHeads, keys, headDim)`.
        let keys: MLXArray
        /// The cached values, of the shape of ``keys``.
        let values: MLXArray
        /// The factor the scores take before the softmax.
        let scale: Float
    }

    /// Builds repeatable random operands for one case in one dtype. The keys
    /// and the values are slices of a larger buffer, as a cache holds them.
    static func operands(_ attentionCase: AttentionCase, dtype: DType) -> Operands {
        let bufferShape = [attentionCase.batch, attentionCase.kvHeads, attentionCase.capacity, headDim]
        let keyBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(Seed.keys)).asType(dtype)
        let valueBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(Seed.values))
            .asType(dtype)
        let queries = MLXRandom.normal(
            [attentionCase.batch, attentionCase.queries, attentionCase.queryHeads, headDim],
            key: MLXRandom.key(Seed.queries)
        ).asType(dtype).swappedAxes(headAxis, rowAxis)
        eval(keyBuffer, valueBuffer, queries)
        return Operands(
            queries: queries,
            keys: keyBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            values: valueBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            scale: 1 / Float(headDim).squareRoot())
    }

    /// The mask mode of a causal flag.
    static func maskMode(causal: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        causal ? .causal : .none
    }

    /// The SDPA of MLX over the operands, in float32.
    static func reference(_ operands: Operands, causal: Bool) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: operands.queries.asType(.float32),
            keys: operands.keys.asType(.float32),
            values: operands.values.asType(.float32),
            scale: operands.scale, mask: maskMode(causal: causal))
    }

    @Test(arguments: attentionCases, dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(attentionCase: AttentionCase, dtype: DType) {
        let operands = Self.operands(attentionCase, dtype: dtype)

        for causal in Self.causalMasks {
            #expect(
                PrefillAttentionKernel.canRun(
                    queries: operands.queries, keys: operands.keys, values: operands.values,
                    mask: Self.maskMode(causal: causal)))
            let kernel = PrefillAttentionKernel.attention(
                queries: operands.queries, keys: operands.keys, values: operands.values,
                scale: operands.scale, causal: causal)
            let reference = Self.reference(operands, causal: causal)

            #expect(kernel.shape == operands.queries.shape)
            #expect(kernel.dtype == dtype)
            #expect(abs(reference).max().item(Float.self) > 0, "every reference value was zero")
            let gap = AttentionKernelTolerance.worstGap(kernel, reference)
            #expect(gap <= AttentionKernelTolerance.of(dtype), "causal \(causal), worst gap \(gap)")
        }
    }

    /// The positive controls of the tolerance test: a kernel that ignored the
    /// causal mask, or that did not write the last query token, would break
    /// the tolerance.
    @Test(arguments: attentionCases)
    func aWrongMaskOrAMissingTokenMovesTheOutputPastTheTolerance(attentionCase: AttentionCase) {
        let operands = Self.operands(attentionCase, dtype: .float32)
        let causal = Self.reference(operands, causal: true)
        let open = Self.reference(operands, causal: false)
        let lastToken = attentionCase.queries - 1
        let missingToken = causal * (MLXArray(0 ..< attentionCase.queries) .< lastToken)
            .reshaped(1, 1, attentionCase.queries, 1)

        #expect(AttentionKernelTolerance.worstGap(open, causal) > AttentionKernelTolerance.bfloat16)
        #expect(
            AttentionKernelTolerance.worstGap(missingToken, causal)
                > AttentionKernelTolerance.bfloat16)
    }

    /// The Qwen3.8 attention shape of `queries` new tokens over `keys` cached
    /// keys, in bfloat16.
    private static func qwen38Operands(queries: Int, keys: Int, dtype: DType = .bfloat16)
        -> Operands
    {
        operands(
            AttentionCase(
                batch: 1, queryHeads: 24, kvHeads: 4, queries: queries, keys: keys, capacity: keys),
            dtype: dtype)
    }

    /// Tells if the kernel runs a causal call of the operands.
    private static func runsCausal(_ operands: Operands) -> Bool {
        PrefillAttentionKernel.canRun(
            queries: operands.queries, keys: operands.keys, values: operands.values, mask: .causal)
    }

    /// The kernel reads only a prefill block of enough query tokens: below
    /// ``PrefillAttentionKernel/fewestQueries`` the other routes are faster.
    @Test func theKernelReadsOnlyEnoughQueries() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let enough = Self.qwen38Operands(queries: fewest, keys: fewest)
        let few = Self.qwen38Operands(queries: fewest - 1, keys: fewest)

        #expect(Self.runsCausal(enough))
        #expect(!Self.runsCausal(few))
    }

    /// The kernel reads only the head shapes it holds: a head width of 256
    /// and a whole number of threadgroup head groups for each KV head.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let operands = Self.qwen38Operands(
            queries: PrefillAttentionKernel.fewestQueries, keys: PrefillAttentionKernel.fewestQueries)
        let narrow = Operands(
            queries: operands.queries[.ellipsis, ..<128], keys: operands.keys[.ellipsis, ..<128],
            values: operands.values[.ellipsis, ..<128], scale: operands.scale)
        let twoPerKVHead = Operands(
            queries: operands.queries[0..., ..<8], keys: operands.keys, values: operands.values,
            scale: operands.scale)
        let fourPerKVHead = Operands(
            queries: operands.queries[0..., ..<16], keys: operands.keys, values: operands.values,
            scale: operands.scale)

        #expect(Self.runsCausal(operands))
        #expect(!Self.runsCausal(narrow))
        #expect(!Self.runsCausal(twoPerKVHead))
        #expect(!Self.runsCausal(fourPerKVHead))
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the narrow loads of the kernel and stay inside
    /// the tolerance too.
    @Test func keysWithSpacedHeadNumbersStayInsideTheTolerance() {
        let operands = Self.operands(Self.attentionCases[1], dtype: .float32)
        let spaced = Operands(
            queries: operands.queries,
            keys: operands.keys.swappedAxes(Self.rowAxis, Self.widthAxis).contiguous()
                .swappedAxes(Self.rowAxis, Self.widthAxis),
            values: operands.values.swappedAxes(Self.rowAxis, Self.widthAxis).contiguous()
                .swappedAxes(Self.rowAxis, Self.widthAxis),
            scale: operands.scale)
        eval(spaced.keys, spaced.values)

        let kernel = PrefillAttentionKernel.attention(
            queries: spaced.queries, keys: spaced.keys, values: spaced.values,
            scale: spaced.scale, causal: true)
        let reference = Self.reference(operands, causal: true)

        #expect(spaced.keys.asData(access: .noCopy).strides[Self.widthAxis] != 1)
        let gap = AttentionKernelTolerance.worstGap(kernel, reference)
        #expect(gap <= AttentionKernelTolerance.float32, "worst gap \(gap)")
    }

    /// The kernel reads the three float dtypes, with a causal mask over at
    /// least as many keys as queries, or with no mask.
    @Test func theKernelReadsOnlyTheDtypesAndMasksItHolds() {
        let fewest = PrefillAttentionKernel.fewestQueries
        let operands = Self.qwen38Operands(queries: fewest, keys: fewest)
        let half = Self.qwen38Operands(queries: fewest, keys: fewest, dtype: .float16)
        let fewerKeys = Operands(
            queries: operands.queries, keys: operands.keys[.ellipsis, ..<(fewest - 1), 0...],
            values: operands.values[.ellipsis, ..<(fewest - 1), 0...], scale: operands.scale)
        let arrayMask = MLXArray.zeros([fewest, fewest], dtype: .bfloat16)

        #expect(Self.runsCausal(half))
        #expect(!Self.runsCausal(fewerKeys))
        #expect(
            PrefillAttentionKernel.canRun(
                queries: fewerKeys.queries, keys: fewerKeys.keys, values: fewerKeys.values,
                mask: .none))
        #expect(
            !PrefillAttentionKernel.canRun(
                queries: operands.queries, keys: operands.keys, values: operands.values,
                mask: .array(arrayMask)))
    }

    /// A prefill call gives the kernel output, bit for bit.
    @Test func aPrefillCallGivesTheKernelOutput() {
        let operands = Self.qwen38Operands(queries: 96, keys: 160)

        let routed = PrefillAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)
        let kernel = PrefillAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, causal: true)

        #expect(arrayEqual(routed, kernel).item(Bool.self))
    }

    /// A call the kernel does not read keeps the route of
    /// ``DecodeAttentionKernel``, bit for bit: a decode step over a long
    /// cache, and a prefill call with an array mask.
    @Test func aCallTheKernelDoesNotReadKeepsTheDecodeRoute() {
        let decode = Self.qwen38Operands(queries: 1, keys: DecodeAttentionKernel.fewestKeys)
        let prefill = Self.qwen38Operands(queries: 96, keys: 96)
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
