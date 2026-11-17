// Copyright © 2026 Apple Inc.
//
// The contract of ``QuantizedKVAttentionKernel``.
//
// A decode step and a speculative verify of Qwen3.8 over an 8-bit
// `QuantizedKVCache` read the packed keys and values through these kernels,
// which dequantize each number inline. The kernels add the keys in blocks
// with an online softmax, in float32, thus they are not bit-exact with the
// quantized path of MLX (`quantizedScaledDotProductAttention`). Their output
// must stay inside the tolerance of the SDPA of MLX over the dequantized keys
// and values in float32, and inside the tolerance of the quantized path.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct QuantizedKVAttentionKernelTests {

    /// The shapes of the decode tolerance test, each of one query token.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest keys the kernel reads.
    /// 2. The Qwen3.8 shape with a last block shorter than the others, over
    ///    a cache buffer longer than its keys.
    /// 3. The Qwen3.8 shape at the largest block count.
    /// 4. Two sequences of 4 query heads for each KV head.
    static let decodeCases = [
        AttentionKernelCase.qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys),
        AttentionKernelCase.qwen38(queries: 1, keys: 3001),
        AttentionKernelCase.qwen38(queries: 1, keys: 16_500),
        AttentionKernelCase(
            batch: 2, queryHeads: 8, kvHeads: 2, headDim: AttentionKernelCase.qwen38HeadDim,
            queries: 1, keys: 1500, capacity: 1500),
    ]

    /// The shapes of the verify tolerance test.
    ///
    /// 1. The Qwen3.8 shape at the fewest queries and the fewest keys the
    ///    kernel reads.
    /// 2. The Qwen3.8 shape of a verify of four tokens, the MTP verify.
    /// 3. The Qwen3.8 shape of a verify of five tokens, whose rows split a
    ///    query head over two matrices, with a last block shorter than the
    ///    others.
    /// 4. The Qwen3.8 shape at the most queries the kernel reads, over many
    ///    blocks.
    /// 5. Two sequences of 3 query heads for each KV head.
    static let verifyCases = [
        AttentionKernelCase.qwen38(
            queries: VerifyAttentionKernel.fewestQueries, keys: DecodeAttentionKernel.fewestKeys),
        AttentionKernelCase.qwen38(queries: 4, keys: 2100),
        AttentionKernelCase.qwen38(queries: 5, keys: 3001),
        AttentionKernelCase.qwen38(queries: VerifyAttentionKernel.mostQueries, keys: 16_500),
        AttentionKernelCase(
            batch: 2, queryHeads: 6, kvHeads: 2, headDim: AttentionKernelCase.qwen38HeadDim,
            queries: 3, keys: 1500, capacity: 1500),
    ]

    /// The masks the verify kernel reads.
    static let causalMasks = [true, false]

    /// The query count of the MTP verify of Qwen3.8.
    private static let mtpVerifyTokens = QuantizedRouteCall.mtpVerifyTokens

    /// The cached keys of the route tests.
    private static let routedKeys = QuantizedRouteCall.routedKeys

    /// A head width the kernels do not read.
    private static let otherHeadDim = 128

    /// The query tokens of a decode step and of the MTP verify of Qwen3.8.
    static let qwen38StepTokens = [1, QuantizedRouteCall.mtpVerifyTokens]

    /// The largest gap allowed between the kernels and the quantized path,
    /// as a part of the largest value of the quantized path. Each of the two
    /// paths rounds to bfloat16 in its own places, and the quantized path
    /// also rounds its scores and its softmax weights, thus their gap reaches
    /// up to one bfloat16 tolerance from each side.
    private static let quantizedPathTolerance = 2 * AttentionKernelTolerance.bfloat16

    /// The random operands of this suite, from its own seeds.
    private static let random = AttentionKernelRandomOperands(
        queries: 20_261_117, keys: 20_261_118, values: 20_261_119)

    @Test(arguments: decodeCases, AttentionKernelTolerance.dtypes)
    func theDecodeKernelStaysInsideTheToleranceOfTheDequantizedSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) throws {
        let operands = QuantizedAttentionOperands(Self.random, attentionCase, dtype: dtype)

        try operands.expectInsideTheToleranceOfTheDequantizedSDPA(causal: false)
    }

    @Test(arguments: verifyCases, AttentionKernelTolerance.dtypes)
    func theVerifyKernelStaysInsideTheToleranceOfTheDequantizedSDPA(
        attentionCase: AttentionKernelCase, dtype: DType
    ) throws {
        let operands = QuantizedAttentionOperands(Self.random, attentionCase, dtype: dtype)

        for causal in Self.causalMasks {
            try operands.expectInsideTheToleranceOfTheDequantizedSDPA(causal: causal)
        }
    }

    /// The kernels stay inside the tolerance of the quantized path that they
    /// replace, at a decode step and at the MTP verify of Qwen3.8 over a
    /// bfloat16 cache, the shapes of the published checkpoints.
    @Test(arguments: qwen38StepTokens)
    func theKernelsStayInsideTheToleranceOfTheQuantizedPath(tokens: Int) throws {
        let operands = try Self.qwen38Step(tokens: tokens)
        let mask = Self.stepMask(tokens: tokens)

        let gap = AttentionKernelTolerance.worstGap(
            operands.kernelOutput(mask: mask), operands.quantizedPathOutput(mask: mask))

        #expect(gap <= Self.quantizedPathTolerance, "worst gap \(gap)")
    }

    /// The kernels stay at least as close to the SDPA of MLX over the
    /// dequantized keys and values as the quantized path that they replace,
    /// which rounds its scores and its softmax weights to bfloat16.
    @Test(arguments: qwen38StepTokens)
    func theKernelsStayAsCloseToTheDequantizedSDPAAsTheQuantizedPath(tokens: Int) throws {
        let operands = try Self.qwen38Step(tokens: tokens)
        let mask = Self.stepMask(tokens: tokens)
        let reference = operands.dequantizedReference(causal: tokens > 1)

        let kernelGap = AttentionKernelTolerance.worstGap(
            operands.kernelOutput(mask: mask), reference)
        let pathGap = AttentionKernelTolerance.worstGap(
            operands.quantizedPathOutput(mask: mask), reference)

        #expect(kernelGap <= pathGap, "kernel gap \(kernelGap), quantized path gap \(pathGap)")
    }

    /// The operands of a Qwen3.8 step of some tokens over a bfloat16 cache,
    /// which the kernels read.
    ///
    /// - Parameter tokens: The query tokens of the step.
    /// - Returns: The operands.
    /// - Throws: A failed requirement when the kernels do not read them.
    private static func qwen38Step(tokens: Int) throws -> QuantizedAttentionOperands {
        let operands = QuantizedAttentionOperands(
            random, .qwen38(queries: tokens, keys: routedKeys), dtype: .bfloat16)
        try #require(operands.kernelRuns(mask: stepMask(tokens: tokens)))
        return operands
    }

    /// The mask the Qwen3.5 model gives a step: no mask for one token, the
    /// causal mask for a verify.
    private static func stepMask(tokens: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        tokens == 1 ? .none : .causal
    }

    /// The route of a Qwen3.5 attention layer over an 8-bit cache gives the
    /// kernel output bit for bit: a decode step with no mask, and a verify
    /// with a causal mask or no mask.
    @Test(arguments: [
        (1, AttentionMaskKind.unmasked), (mtpVerifyTokens, .causal), (mtpVerifyTokens, .unmasked),
    ])
    func theRouteGivesTheKernelOutput(tokens: Int, mask: AttentionMaskKind) throws {
        let routed = QuantizedRouteCall(
            Self.random, .qwen38(queries: tokens, keys: Self.routedKeys), mask: mask)
        try #require(routed.expected.kernelRuns(mask: routed.mask))

        AttentionKernelTolerance.expectBitForBit(
            routed.routeOutput(), routed.expected.kernelOutput(mask: routed.mask))
    }

    /// The route keeps the quantized path of MLX, bit for bit, for each call
    /// the kernels do not read: another bit width, another group size, a mask
    /// array, a short cache and a prefill block.
    @Test(arguments: QuantizedRouteCall.Variant.allCases)
    func theRouteKeepsTheQuantizedPathForTheCallsTheKernelsDoNotRead(
        variant: QuantizedRouteCall.Variant
    ) throws {
        let routed = QuantizedRouteCall(Self.random, variant: variant)
        #expect(!routed.expected.kernelRuns(mask: routed.mask))

        AttentionKernelTolerance.expectBitForBit(
            routed.routeOutput(), routed.expected.quantizedPathOutput(mask: routed.mask))
    }

    /// The kernels read only a head width of 256 and at most
    /// ``VerifyAttentionKernel/mostQueries`` query tokens.
    @Test func theKernelsReadOnlyTheShapesTheyHold() {
        let narrow = QuantizedAttentionOperands(
            Self.random,
            AttentionKernelCase(
                batch: 1, queryHeads: AttentionKernelCase.qwen38QueryHeads,
                kvHeads: AttentionKernelCase.qwen38KVHeads, headDim: Self.otherHeadDim,
                queries: 1, keys: Self.routedKeys, capacity: Self.routedKeys),
            dtype: .bfloat16)
        let tooMany = QuantizedAttentionOperands(
            Self.random,
            .qwen38(queries: VerifyAttentionKernel.mostQueries + 1, keys: Self.routedKeys),
            dtype: .bfloat16)

        #expect(!narrow.kernelRuns(mask: .none))
        #expect(!tooMany.kernelRuns(mask: .causal))
    }
}

/// The operands of one attention call over a quantized cache, as a
/// `QuantizedKVCache` gives them: the cache takes the keys ahead of the
/// queries, then the keys of the query tokens, and gives slices of its
/// buffers.
struct QuantizedAttentionOperands {

    /// The `(packed numbers, scales, biases)` of the keys or of the values.
    typealias Rows = (MLXArray, MLXArray, MLXArray?)

    /// The queries, shape `(batch, queryHeads, tokens, headDim)`.
    let queries: MLXArray
    /// The quantized keys the cache gives back.
    let keys: Rows
    /// The quantized values the cache gives back.
    let values: Rows
    /// The factor the scores take before the softmax.
    let scale: Float
    /// The group size of the cache.
    let groupSize: Int
    /// The bit width of the cache.
    let bits: Int

    /// Builds the operands of one case in one dtype, from a cache that takes
    /// the cached keys first and the keys of the query tokens second.
    ///
    /// - Parameters:
    ///   - random: The seeds of the random operands.
    ///   - attentionCase: The shape of the operands.
    ///   - dtype: The dtype of the queries, the keys and the values.
    ///   - bits: The bit width of the cache.
    ///   - groupSize: The group size of the cache.
    init(
        _ random: AttentionKernelRandomOperands, _ attentionCase: AttentionKernelCase,
        dtype: DType, bits: Int = QuantizedKVAttentionKernel.bits,
        groupSize: Int = QuantizedKVAttentionKernel.groupSize
    ) {
        let plain = random.operands(attentionCase, dtype: dtype)
        let cache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        let step = Self.filled(cache, plain: plain)
        let (keys, values) = cache.updateQuantized(keys: step.keys, values: step.values)
        self.init(
            queries: plain.queries, keys: keys, values: values, scale: plain.scale,
            groupSize: cache.groupSize, bits: bits)
    }

    /// Builds the operands from their parts.
    init(
        queries: MLXArray, keys: Rows, values: Rows, scale: Float, groupSize: Int, bits: Int
    ) {
        self.queries = queries
        self.keys = keys
        self.values = values
        self.scale = scale
        self.groupSize = groupSize
        self.bits = bits
    }

    /// Gives a cache the keys and values ahead of the query tokens.
    ///
    /// - Parameters:
    ///   - cache: The empty cache.
    ///   - plain: The plain operands, whose last keys belong to the query
    ///     tokens.
    /// - Returns: The keys and values of the query tokens.
    static func filled(
        _ cache: QuantizedKVCache, plain: AttentionKernelOperands
    ) -> (keys: MLXArray, values: MLXArray) {
        let tokens = plain.queries.dim(AttentionKernelOperands.rowAxis)
        let cached = plain.keys.dim(AttentionKernelOperands.rowAxis) - tokens
        _ = cache.updateQuantized(
            keys: plain.keys[.ellipsis, ..<cached, 0...],
            values: plain.values[.ellipsis, ..<cached, 0...])
        return (plain.keys[.ellipsis, cached..., 0...], plain.values[.ellipsis, cached..., 0...])
    }

    /// Tells if the kernels read the operands with a mask.
    func kernelRuns(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        QuantizedKVAttentionKernel.canRun(
            queries: queries, keys: keys, values: values, mask: mask, groupSize: groupSize,
            bits: bits, mode: .affine)
    }

    /// The output of the kernels over the operands with a mask.
    func kernelOutput(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        QuantizedKVAttentionKernel.attention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    /// The output of the quantized path of MLX over the operands with a mask.
    func quantizedPathOutput(mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        quantizedCacheAttention(
            queries: queries, quantizedKeys: keys, quantizedValues: values, scale: scale,
            mask: mask, groupSize: groupSize, bits: bits, mode: .affine)
    }

    /// The rows dequantized in float32.
    private func dequantizedRows(_ rows: Rows) -> MLXArray {
        dequantized(
            rows.0, scales: rows.1, biases: rows.2, groupSize: groupSize, bits: bits,
            mode: .affine, dtype: .float32)
    }

    /// The SDPA of MLX over the queries and the dequantized keys and values,
    /// in float32.
    func dequantizedReference(causal: Bool) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32), keys: dequantizedRows(keys),
            values: dequantizedRows(values), scale: scale, mask: causal ? .causal : .none)
    }

    /// Checks that the kernels run over the operands, that their output has
    /// the shape and the dtype of the queries, and that it stays inside the
    /// tolerance of the SDPA of MLX over the dequantized keys and values.
    ///
    /// - Parameters:
    ///   - causal: The causal flag of the call.
    ///   - sourceLocation: The location that a failure reports.
    /// - Throws: A failed requirement when the kernels do not run.
    func expectInsideTheToleranceOfTheDequantizedSDPA(
        causal: Bool, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = causal ? .causal : .none
        try #require(
            kernelRuns(mask: mask), "must run: queries \(queries.shape)",
            sourceLocation: sourceLocation)
        let tolerance = try #require(
            AttentionKernelTolerance.of(queries.dtype), sourceLocation: sourceLocation)
        let kernel = kernelOutput(mask: mask)

        #expect(kernel.shape == queries.shape, sourceLocation: sourceLocation)
        #expect(kernel.dtype == queries.dtype, sourceLocation: sourceLocation)
        let gap = AttentionKernelTolerance.worstGap(kernel, dequantizedReference(causal: causal))
        #expect(
            gap <= tolerance, "\(queries.dtype), causal \(causal), worst gap \(gap)",
            sourceLocation: sourceLocation)
    }
}

/// One call of the Qwen3.5 attention route over a quantized cache, beside the
/// operands that a twin cache gives for the same call.
struct QuantizedRouteCall {

    /// A call the kernels do not read.
    enum Variant: CaseIterable, CustomTestStringConvertible, Sendable {
        /// A 4-bit cache.
        case otherBits
        /// A cache with groups of 32 numbers.
        case otherGroupSize
        /// A verify with a mask array.
        case maskArray
        /// A decode step over fewer keys than the kernels read.
        case shortCache
        /// A prefill block over the cache.
        case prefillBlock

        /// The text the test report shows for this variant.
        var testDescription: String {
            "\(self)"
        }
    }

    /// The query count of the MTP verify of Qwen3.8: one token and three
    /// drafted tokens.
    static let mtpVerifyTokens = 4

    /// The cached keys of the route tests: more than the fewest keys the
    /// kernels read.
    static let routedKeys = 2100

    /// The bit width of a quantized cache the kernels do not read.
    private static let otherBits = 4

    /// A group size of a quantized cache the kernels do not read.
    private static let otherGroupSize = 32

    /// The query count of a prefill block, which keeps the quantized path.
    private static let prefillTokens = 64

    /// The output of the route reads this cache.
    private let cache: QuantizedKVCache
    /// The keys of the query tokens.
    private let newKeys: MLXArray
    /// The values of the query tokens.
    private let newValues: MLXArray
    /// The mask of the call.
    let mask: MLXFast.ScaledDotProductAttentionMaskMode
    /// The operands that a twin cache gives for the same call.
    let expected: QuantizedAttentionOperands

    /// Builds the call of one case and one mask kind over an 8-bit cache.
    ///
    /// - Parameters:
    ///   - random: The seeds of the random operands.
    ///   - attentionCase: The shape of the call.
    ///   - mask: The mask kind of the call.
    ///   - bits: The bit width of the cache.
    ///   - groupSize: The group size of the cache.
    init(
        _ random: AttentionKernelRandomOperands, _ attentionCase: AttentionKernelCase,
        mask: AttentionMaskKind, bits: Int = QuantizedKVAttentionKernel.bits,
        groupSize: Int = QuantizedKVAttentionKernel.groupSize
    ) {
        let plain = random.operands(attentionCase, dtype: .bfloat16)
        cache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        (newKeys, newValues) = QuantizedAttentionOperands.filled(cache, plain: plain)
        self.mask = plain.mask(mask)
        expected = QuantizedAttentionOperands(
            random, attentionCase, dtype: .bfloat16, bits: bits, groupSize: groupSize)
    }

    /// Builds a call the kernels do not read.
    ///
    /// - Parameters:
    ///   - random: The seeds of the random operands.
    ///   - variant: The call.
    init(_ random: AttentionKernelRandomOperands, variant: Variant) {
        let verify = AttentionKernelCase.qwen38(
            queries: Self.mtpVerifyTokens, keys: Self.routedKeys)
        switch variant {
        case .otherBits:
            self.init(random, verify, mask: .causal, bits: Self.otherBits)
        case .otherGroupSize:
            self.init(random, verify, mask: .causal, groupSize: Self.otherGroupSize)
        case .maskArray:
            self.init(random, verify, mask: .array)
        case .shortCache:
            self.init(
                random, .qwen38(queries: 1, keys: DecodeAttentionKernel.fewestKeys - 1),
                mask: .unmasked)
        case .prefillBlock:
            self.init(
                random, .qwen38(queries: Self.prefillTokens, keys: Self.routedKeys), mask: .causal)
        }
    }

    /// The output of the Qwen3.5 attention route for the call.
    func routeOutput() -> MLXArray {
        PrefillAttentionKernel.attentionWithCacheUpdate(
            queries: expected.queries, keys: newKeys, values: newValues, cache: cache,
            scale: expected.scale, mask: mask)
    }
}
