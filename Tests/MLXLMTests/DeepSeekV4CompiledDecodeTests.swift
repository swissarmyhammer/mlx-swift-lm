// Copyright © 2026 Apple Inc.
//
// Parity tests for the compiled functions of the DeepSeek-V4 decode step.
//
// Each compiled function must give the SAME BITS as the op path it replaces.
// MLX fuses the elementwise operations of a compiled function into one Metal
// kernel, and the Metal compiler can contract a multiply and an add into one
// `fma` there. A contraction like that moves a result by one unit in the last
// place, and 43 layers carry such a gap into the greedy token. A tolerance
// would hide it, thus every test below compares the bits.
//
// The functions compiled with a shape serve the decode step alone, thus their
// inputs have the shapes of one decode step: one sequence of one token. The
// shapeless functions serve each block, thus one instance of each reads
// blocks of several lengths in turn.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4CompiledDecodeTests {

    // MARK: - The shapes of one decode step

    /// The leading shape of one decode step: one sequence of one token.
    private static let decodeShape = [1, 1]

    /// The token counts of the blocks that one shapeless function reads in
    /// turn: two prefill blocks of different lengths, and one decode step.
    /// One function instance reads all of them, thus a trace of one length
    /// must give the bits of the op path at each other length.
    private static let blockLengths = [5, 3, 1]

    /// The leading shape of one block of `tokens` tokens in one sequence.
    private static func blockShape(_ tokens: Int) -> [Int] {
        [1, tokens]
    }

    /// The number of parallel copies of the residual stream.
    private static let hcMult = 4

    /// The number of one-value-for-each-copy fields ahead of the mixing
    /// matrix in the mixes vector.
    private static let splitFieldCount = 2

    /// The width of the mixes vector, `(2 + hcMult) * hcMult`.
    private static let mixWidth = (splitFieldCount + hcMult) * hcMult

    /// The number of learned scales of a hyper-connection.
    private static let scaleCount = 3

    /// The width of the residual stream. It is not a multiple of a Metal
    /// thread group, thus a fused kernel must handle a partial group.
    private static let hiddenSize = 72

    /// The width of one expert.
    private static let expertWidth = 24

    /// The number of routed experts.
    private static let routedExpertCount = 16

    /// The number of routed experts one token reads.
    private static let expertsPerToken = 6

    /// The number of tokens in the vocabulary of the hash table.
    private static let vocabSize = 10

    /// The SwiGLU limit of the DeepSeek-V4-Flash checkpoint.
    private static let swigluLimit: Float = 10

    /// The factor the routed weights of the DeepSeek-V4-Flash checkpoint take.
    private static let routedScalingFactor: Float = 1.5

    /// The RMS-norm epsilon of the DeepSeek-V4-Flash checkpoint.
    private static let normEps: Float = 1e-6

    /// The mHC epsilon of the DeepSeek-V4-Flash checkpoint.
    private static let hcEps: Float = 1e-6

    /// The range a random activation spans. It reaches past the SwiGLU limit,
    /// thus the clamp takes effect on some values.
    private static let activationRange: Float = 16

    // MARK: - Helpers

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(
        _ shape: [Int], seed: UInt64, range: Float = 1, dtype: DType = .float32
    ) -> MLXArray {
        MLXRandom.uniform(low: -range, high: range, shape, key: MLXRandom.key(seed))
            .asType(dtype)
    }

    /// Checks that two arrays hold the same dtype, the same shape and the
    /// same bits.
    private func expectSameBits(
        _ compiled: MLXArray, _ reference: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(compiled.dtype == reference.dtype, "\(what): dtype", sourceLocation: sourceLocation)
        #expect(compiled.shape == reference.shape, "\(what): shape", sourceLocation: sourceLocation)
        let compiledValues = DeepSeekV4TestArrays.floats(compiled)
        let referenceValues = DeepSeekV4TestArrays.floats(reference)
        let firstGap = zip(compiledValues, referenceValues).enumerated().first {
            $0.element.0 != $0.element.1
        }
        #expect(
            firstGap == nil,
            """
            \(what)[\(firstGap?.offset ?? 0)]: compiled \(firstGap?.element.0 ?? 0), \
            op path \(firstGap?.element.1 ?? 0)
            """,
            sourceLocation: sourceLocation)
    }

    // MARK: - The clamped SwiGLU

    /// The dtypes the activation reads: the float32 of the tests and the
    /// bfloat16 of the checkpoint.
    private static let activationDTypes: [DType] = [.float32, .bfloat16]

    @Test(arguments: activationDTypes)
    func theCompiledClampedSwiGLUGivesTheBitsOfTheOpPathAtEachBlockLength(dtype: DType) {
        let activation = DeepSeekV4CompiledDecode.clampedSwiGLU(limit: Self.swigluLimit)
        for tokens in Self.blockLengths {
            let shape = Self.blockShape(tokens) + [Self.expertsPerToken, 1, Self.expertWidth]
            let gate = Self.randomArray(shape, seed: 101, range: Self.activationRange, dtype: dtype)
            let up = Self.randomArray(shape, seed: 102, range: Self.activationRange, dtype: dtype)

            let reference = DeepSeekV4Math.clampedSwiGLU(
                gate: gate, up: up, limit: Self.swigluLimit)

            expectSameBits(activation(gate, up), reference, "the clamped SwiGLU, \(tokens) tokens")
        }
    }

    @Test
    func theCompiledSwiGLUWithNoLimitGivesTheBitsOfTheOpPathAtEachBlockLength() {
        let activation = DeepSeekV4CompiledDecode.clampedSwiGLU(limit: 0)
        for tokens in Self.blockLengths {
            let shape = Self.blockShape(tokens) + [Self.expertWidth]
            let gate = Self.randomArray(shape, seed: 103, range: Self.activationRange)
            let up = Self.randomArray(shape, seed: 104, range: Self.activationRange)

            let reference = DeepSeekV4Math.clampedSwiGLU(gate: gate, up: up, limit: 0)

            expectSameBits(
                activation(gate, up), reference, "the SwiGLU with no limit, \(tokens) tokens")
        }
    }

    // MARK: - The hyper-connection expand

    @Test
    func theCompiledExpandGivesTheBitsOfTheOpPathAtEachBlockLength() {
        for tokens in Self.blockLengths {
            let block = Self.blockShape(tokens)
            let blockOutput = Self.randomArray(
                block + [Self.hiddenSize], seed: 111, dtype: .bfloat16)
            let residual = Self.randomArray(
                block + [Self.hcMult, Self.hiddenSize], seed: 112, dtype: .bfloat16)
            let post = Self.randomArray(block + [Self.hcMult], seed: 113)
            let comb = softmax(
                Self.randomArray(block + [Self.hcMult, Self.hcMult], seed: 114), axis: -1)

            let compiled = DeepSeekV4CompiledDecode.expand(blockOutput, residual, post, comb)
            let reference = ManifoldStream.expand(
                blockOutput: blockOutput, residual: residual, post: post, comb: comb)

            expectSameBits(compiled, reference, "the expanded residual stream, \(tokens) tokens")
        }
    }

    // MARK: - The split activations

    @Test
    func theCompiledSplitActivationsGiveTheBitsOfTheSplitFieldsAtEachBlockLength() {
        let activations = DeepSeekV4CompiledDecode.splitActivations(hcEps: Self.hcEps)
        let scale = Self.randomArray([Self.scaleCount], seed: 125)
        let base = Self.randomArray([Self.mixWidth], seed: 126)
        for tokens in Self.blockLengths {
            let mixes = Self.randomArray(Self.blockShape(tokens) + [Self.mixWidth], seed: 127)

            let compiled = activations(
                DeepSeekV4Math.hcSplitOperands(
                    mixes: mixes, scale: scale, base: base, hcMult: Self.hcMult))
            let reference = DeepSeekV4Math.hcSplitFields(
                mixes: mixes, scale: scale, base: base, hcMult: Self.hcMult, eps: Self.hcEps)

            expectSameBits(compiled.pre, reference.pre, "the pre weights, \(tokens) tokens")
            expectSameBits(compiled.post, reference.post, "the post gains, \(tokens) tokens")
            expectSameBits(
                compiled.comb, reference.comb, "the row softmax of comb, \(tokens) tokens")
        }
    }

    // MARK: - The collapse fields

    @Test
    func theCompiledCollapseFieldsGiveTheBitsOfTheOpPath() {
        let stream = Self.randomArray(
            Self.decodeShape + [Self.hcMult, Self.hiddenSize], seed: 121, dtype: .bfloat16)
        let projection = Self.randomArray(
            [Self.mixWidth, Self.hcMult * Self.hiddenSize], seed: 122)
        let normWeight = MLXArray.ones([Self.hcMult * Self.hiddenSize])
        let scale = Self.randomArray([Self.scaleCount], seed: 123)
        let base = Self.randomArray([Self.mixWidth], seed: 124)

        let compiled = DeepSeekV4CompiledDecode.collapseFields(
            hcMult: Self.hcMult, normEps: Self.normEps, hcEps: Self.hcEps)(
                stream, projection, normWeight, scale, base)
        let reference = ManifoldStream.collapseFields(
            of: stream, projection: projection, normWeight: normWeight, scale: scale, base: base,
            hcMult: Self.hcMult, normEps: Self.normEps, hcEps: Self.hcEps)

        expectSameBits(compiled.0, reference.stream, "the float32 stream")
        expectSameBits(compiled.1, reference.pre, "the pre weights")
        expectSameBits(compiled.2, reference.post, "the post gains")
        expectSameBits(compiled.3, reference.comb, "the row softmax of the mixing matrix")
    }

    // MARK: - The routing gate

    /// The routing constants of the tests, with and without the
    /// normalization of the selected weights.
    private static let routings = [true, false].map {
        DeepSeekV4ExpertRouting(
            expertsPerToken: expertsPerToken, routedScalingFactor: routedScalingFactor,
            normalizesTopkProbabilities: $0)
    }

    @Test(arguments: routings)
    func theCompiledTopKRoutingGivesTheBitsOfTheOpPath(routing: DeepSeekV4ExpertRouting) {
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: 131, dtype: .bfloat16)
        let weight = Self.randomArray([Self.routedExpertCount, Self.hiddenSize], seed: 132)
        let bias = Self.randomArray([Self.routedExpertCount], seed: 133)

        let compiled = DeepSeekV4CompiledDecode.topKRouting(routing)(x, weight, bias)
        let reference = routing.topK(x, weight: weight, bias: bias)

        #expect(compiled.0.dtype == reference.indices.dtype)
        #expect(
            DeepSeekV4TestArrays.integers(compiled.0)
                == DeepSeekV4TestArrays.integers(reference.indices), "the selected experts")
        expectSameBits(compiled.1, reference.weights, "the routed weights")
    }

    @Test(arguments: routings)
    func theCompiledHashRoutingGivesTheBitsOfTheOpPath(routing: DeepSeekV4ExpertRouting) {
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: 141, dtype: .bfloat16)
        let weight = Self.randomArray([Self.routedExpertCount, Self.hiddenSize], seed: 142)
        let table = MLXRandom.randInt(
            0 ..< Self.routedExpertCount, [Self.vocabSize, Self.expertsPerToken],
            key: MLXRandom.key(143)
        ).asType(.int32)
        let inputIds = MLXArray([Int32(7)], Self.decodeShape)

        let compiled = DeepSeekV4CompiledDecode.hashRouting(routing)(x, weight, table, inputIds)
        let reference = routing.hashed(x, weight: weight, table: table, inputIds: inputIds)

        #expect(compiled.0.dtype == reference.indices.dtype)
        #expect(
            DeepSeekV4TestArrays.integers(compiled.0)
                == DeepSeekV4TestArrays.integers(reference.indices), "the hashed experts")
        expectSameBits(compiled.1, reference.weights, "the routed weights")
    }

    // MARK: - The mixture-of-experts combine

    @Test
    func theCompiledCombineGivesTheBitsOfTheOpPathAtEachBlockLength() {
        for tokens in Self.blockLengths {
            let block = Self.blockShape(tokens)
            let x = Self.randomArray(block + [Self.hiddenSize], seed: 151, dtype: .bfloat16)
            let routed = Self.randomArray(
                block + [Self.expertsPerToken, Self.hiddenSize], seed: 152, dtype: .bfloat16)
            let weights = Self.randomArray(block + [Self.expertsPerToken], seed: 153)
            let shared = Self.randomArray(block + [Self.hiddenSize], seed: 154, dtype: .bfloat16)

            let compiled = DeepSeekV4CompiledDecode.combineWithSharedExpert(
                x, routed, weights, shared)
            let reference = DeepSeekV4MoE.combine(
                x, routed: routed, weights: weights, shared: shared)

            expectSameBits(compiled, reference, "the mixture-of-experts output, \(tokens) tokens")
        }
    }

    @Test
    func theCompiledCombineWithNoSharedExpertGivesTheBitsOfTheOpPathAtEachBlockLength() {
        for tokens in Self.blockLengths {
            let block = Self.blockShape(tokens)
            let x = Self.randomArray(block + [Self.hiddenSize], seed: 161, dtype: .bfloat16)
            let routed = Self.randomArray(
                block + [Self.expertsPerToken, Self.hiddenSize], seed: 162, dtype: .bfloat16)
            let weights = Self.randomArray(block + [Self.expertsPerToken], seed: 163)

            let compiled = DeepSeekV4CompiledDecode.combineRoutedExperts(x, routed, weights)
            let reference = DeepSeekV4MoE.combine(
                x, routed: routed, weights: weights, shared: nil)

            expectSameBits(compiled, reference, "the routed-only output, \(tokens) tokens")
        }
    }

    // MARK: - Which path a block reads

    @Test
    func onlyABlockOfOneTokenReadsTheFunctionsCompiledWithAShape() {
        let oneToken = MLXArray.zeros(Self.decodeShape + [Self.hiddenSize])
        let twoTokens = MLXArray.zeros([1, 2, Self.hiddenSize])

        #expect(DeepSeekV4CompiledDecode.isDecodeStep(oneToken))
        #expect(!DeepSeekV4CompiledDecode.isDecodeStep(twoTokens))
    }
}
