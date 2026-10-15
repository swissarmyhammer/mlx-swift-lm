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

    /// One shape of the tolerance test.
    struct AttentionCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of query heads.
        let queryHeads: Int
        /// The number of key and value heads.
        let kvHeads: Int
        /// The width of one head.
        let headDim: Int
        /// The number of cached keys the query reads.
        let keys: Int
        /// The number of rows the cache buffer holds. The keys are a slice of
        /// the first `keys` rows, as a `KVCacheSimple` gives them back.
        let capacity: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(queryHeads) heads over \(kvHeads) KV heads of \(headDim), "
                + "\(keys) keys of \(capacity) rows"
        }
    }

    /// The shapes of the tolerance test.
    ///
    /// 1. The Qwen3.8 attention shape at the fewest keys the kernel reads.
    /// 2. The Qwen3.8 shape with a last block shorter than the others.
    /// 3. The Qwen3.8 shape at the largest block count.
    /// 4. Two sequences of a small model: 2 query heads for each KV head, a
    ///    head of 128 numbers.
    static let attentionCases = [
        AttentionCase(
            batch: 1, queryHeads: 24, kvHeads: 4, headDim: 256, keys: 1024, capacity: 1280),
        AttentionCase(
            batch: 1, queryHeads: 24, kvHeads: 4, headDim: 256, keys: 3001, capacity: 3072),
        AttentionCase(
            batch: 1, queryHeads: 24, kvHeads: 4, headDim: 256, keys: 16_500, capacity: 16_640),
        AttentionCase(
            batch: 2, queryHeads: 4, kvHeads: 2, headDim: 128, keys: 1500, capacity: 1536),
    ]

    /// The dtypes of the tolerance test: the float32 of the synthetic tests
    /// and the bfloat16 of the published checkpoints.
    static let dtypes: [DType] = [.float32, .bfloat16]

    /// The largest gap allowed between the kernel and the SDPA of MLX, as a
    /// part of the largest reference value, for float32 inputs. Both paths
    /// add in float32, each in its own order.
    private static let float32Tolerance: Float = 1e-4

    /// The largest gap for bfloat16 inputs. The reference reads the same
    /// bfloat16 inputs in float32. The kernel adds in float32 and rounds its
    /// output to bfloat16, whose step is 2^-8 of the value.
    private static let bfloat16Tolerance: Float = 8e-3

    /// The seed of each random operand.
    private enum Seed {
        static let queries: UInt64 = 20_261_015
        static let keys: UInt64 = 20_261_016
        static let values: UInt64 = 20_261_017
    }

    /// The axis a `(batch, heads, rows, width)` tensor holds its rows on.
    private static let rowAxis = 2

    /// The axis a `(batch, heads, rows, width)` tensor holds the numbers of
    /// one head on.
    private static let widthAxis = 3

    /// The operands of one attention call.
    struct Operands {
        /// The queries, shape `(batch, queryHeads, 1, headDim)`.
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
        let bufferShape = [
            attentionCase.batch, attentionCase.kvHeads, attentionCase.capacity,
            attentionCase.headDim,
        ]
        let keyBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(Seed.keys)).asType(dtype)
        let valueBuffer = MLXRandom.normal(bufferShape, key: MLXRandom.key(Seed.values))
            .asType(dtype)
        eval(keyBuffer, valueBuffer)
        return Operands(
            queries: MLXRandom.normal(
                [attentionCase.batch, attentionCase.queryHeads, 1, attentionCase.headDim],
                key: MLXRandom.key(Seed.queries)
            ).asType(dtype),
            keys: keyBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            values: valueBuffer[.ellipsis, ..<attentionCase.keys, 0...],
            scale: 1 / Float(attentionCase.headDim).squareRoot())
    }

    /// The SDPA of MLX over the first `keys` rows of the operands, in float32.
    static func reference(_ operands: Operands, keys: Int) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: operands.queries.asType(.float32),
            keys: operands.keys[.ellipsis, ..<keys, 0...].asType(.float32),
            values: operands.values[.ellipsis, ..<keys, 0...].asType(.float32),
            scale: operands.scale, mask: .none)
    }

    /// The largest gap between two outputs, as a part of the largest
    /// absolute value of the expected output.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let gap = abs(got.asType(.float32) - expected).max().item(Float.self)
        return gap / abs(expected).max().item(Float.self)
    }

    /// The tolerance of one dtype.
    static func tolerance(_ dtype: DType) -> Float {
        dtype == .float32 ? float32Tolerance : bfloat16Tolerance
    }

    @Test(arguments: attentionCases, dtypes)
    func theKernelStaysInsideTheToleranceOfTheSDPA(attentionCase: AttentionCase, dtype: DType) {
        let operands = Self.operands(attentionCase, dtype: dtype)

        #expect(
            DecodeAttentionKernel.canRun(
                queries: operands.queries, keys: operands.keys, values: operands.values))
        let kernel = DecodeAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale)
        let reference = Self.reference(operands, keys: attentionCase.keys)

        #expect(kernel.shape == operands.queries.shape)
        #expect(kernel.dtype == dtype)
        #expect(abs(reference).max().item(Float.self) > 0, "every reference value was zero")
        let gap = Self.worstGap(kernel, reference)
        #expect(gap <= Self.tolerance(dtype), "worst gap \(gap)")
    }

    /// Keys and values whose head numbers do not stand side by side, from a
    /// transposed buffer, take the strided read of the kernel and stay inside
    /// the tolerance too.
    @Test func keysWithSpacedHeadNumbersStayInsideTheTolerance() {
        let attentionCase = Self.attentionCases[0]
        let operands = Self.operands(attentionCase, dtype: .float32)
        let spacedKeys = operands.keys.swappedAxes(Self.rowAxis, Self.widthAxis).contiguous()
            .swappedAxes(Self.rowAxis, Self.widthAxis)
        let spacedValues = operands.values.swappedAxes(Self.rowAxis, Self.widthAxis).contiguous()
            .swappedAxes(Self.rowAxis, Self.widthAxis)
        eval(spacedKeys, spacedValues)

        let kernel = DecodeAttentionKernel.attention(
            queries: operands.queries, keys: spacedKeys, values: spacedValues,
            scale: operands.scale)
        let reference = Self.reference(operands, keys: attentionCase.keys)

        #expect(spacedKeys.asData(access: .noCopy).strides[Self.widthAxis] != 1)
        let gap = Self.worstGap(kernel, reference)
        #expect(gap <= Self.float32Tolerance, "worst gap \(gap)")
    }

    /// The positive control of the tolerance test: a kernel that did not read
    /// the keys of its last block would break the tolerance.
    @Test(arguments: attentionCases)
    func theKeysOfTheLastBlockMoveTheOutputPastTheTolerance(attentionCase: AttentionCase) {
        let operands = Self.operands(attentionCase, dtype: .float32)
        let blocks = DecodeAttentionKernel.blockCount(keys: attentionCase.keys)
        let lastBlock = DecodeAttentionKernel.keysPerBlock(keys: attentionCase.keys, blocks: blocks)

        let full = Self.reference(operands, keys: attentionCase.keys)
        let short = Self.reference(operands, keys: attentionCase.keys - lastBlock)

        #expect(Self.worstGap(short, full) > Self.bfloat16Tolerance)
    }

    /// The Qwen3.8 attention shape over `keys` cached keys, in bfloat16.
    private static func qwen38Operands(keys: Int) -> Operands {
        operands(
            AttentionCase(
                batch: 1, queryHeads: 24, kvHeads: 4, headDim: 256, keys: keys, capacity: keys),
            dtype: .bfloat16)
    }

    /// The kernel reads one query over enough keys: below
    /// ``DecodeAttentionKernel/fewestKeys`` the SDPA of MLX is faster.
    @Test func theKernelReadsOnlyOneQueryOverEnoughKeys() {
        let enough = Self.qwen38Operands(keys: DecodeAttentionKernel.fewestKeys)
        let few = Self.qwen38Operands(keys: DecodeAttentionKernel.fewestKeys - 1)
        let twoQueries = concatenated([enough.queries, enough.queries], axis: Self.rowAxis)

        #expect(
            DecodeAttentionKernel.canRun(
                queries: enough.queries, keys: enough.keys, values: enough.values))
        #expect(
            !DecodeAttentionKernel.canRun(queries: few.queries, keys: few.keys, values: few.values))
        #expect(
            !DecodeAttentionKernel.canRun(
                queries: twoQueries, keys: enough.keys, values: enough.values))
    }

    /// The kernel reads only head shapes it holds in registers: a whole
    /// number of query heads for each KV head, an even number of them, and a
    /// head width that the lanes of a SIMD group share evenly.
    @Test func theKernelReadsOnlyTheHeadShapesItHolds() {
        let operands = Self.qwen38Operands(keys: DecodeAttentionKernel.fewestKeys)
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
        let operands = Self.qwen38Operands(keys: DecodeAttentionKernel.fewestKeys - 1)

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
        let operands = Self.qwen38Operands(keys: DecodeAttentionKernel.fewestKeys)

        let routed = DecodeAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .none)
        let kernel = DecodeAttentionKernel.attention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale)
        let masked = DecodeAttentionKernel.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: operands.queries, keys: operands.keys, values: operands.values,
            scale: operands.scale, mask: .causal)

        #expect(arrayEqual(routed, kernel).item(Bool.self))
        #expect(arrayEqual(masked, sdpa).item(Bool.self))
    }
}
