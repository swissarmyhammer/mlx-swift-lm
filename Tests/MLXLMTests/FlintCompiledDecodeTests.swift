// Copyright © 2026 Apple Inc.
//
// Parity tests for the compiled functions of the Flint decode step.
//
// Each compiled function must give the SAME BITS as the op path it replaces.
// MLX fuses the elementwise operations of a compiled function into one Metal
// kernel. A fused kernel that rounds one step differently moves a result by
// one unit in the last place, and 47 MoE layers carry such a gap into the
// greedy token. A tolerance would hide it, thus every test below compares the
// bits.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct FlintCompiledDecodeTests {

    // MARK: - The shapes of one decode step

    /// The leading shape of one decode step: one sequence of one token.
    private static let decodeShape = [1, 1]

    /// The width of the residual stream. It is not a multiple of a Metal
    /// thread group, thus a fused kernel must handle a partial group.
    private static let hiddenSize = 72

    /// The number of routed experts.
    private static let expertCount = 16

    /// The number of routed experts one token reads.
    private static let expertsPerToken = 6

    /// The dtypes a decode step reads: the float32 of the tests and the
    /// bfloat16 of the checkpoint.
    private static let activationDTypes: [DType] = [.float32, .bfloat16]

    /// The two settings of `norm_topk_prob`.
    private static let normalizations = [true, false]

    /// The seeds of the random arrays the tests read, one for each role.
    private enum Seed {
        static let logits: UInt64 = 11
        static let bias: UInt64 = 12
        static let x: UInt64 = 21
        static let routed: UInt64 = 22
        static let weights: UInt64 = 23
        static let shared: UInt64 = 24
        static let residual: UInt64 = 25
        static let expertBias: UInt64 = 26
        static let blockWeights: UInt64 = 27
    }

    // MARK: - Helpers

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(
        _ shape: [Int], seed: UInt64, dtype: DType = .float32
    ) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
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
            $0.element.0.bitPattern != $0.element.1.bitPattern
        }
        #expect(
            firstGap == nil,
            """
            \(what)[\(firstGap?.offset ?? 0)]: compiled \(firstGap?.element.0 ?? 0), \
            op path \(firstGap?.element.1 ?? 0)
            """,
            sourceLocation: sourceLocation)
    }

    /// The routing of a block that reads ``expertsPerToken`` experts.
    private static func routing(normalizesWeights: Bool) -> FlintExpertRouting {
        FlintExpertRouting(topK: expertsPerToken, normalizesWeights: normalizesWeights)
    }

    /// The router weights of one decode step: the sigmoid scores of the
    /// selected experts, thus positive and below one.
    private static func routerWeights(seed: UInt64) -> MLXArray {
        sigmoid(randomArray(decodeShape + [expertsPerToken], seed: seed))
    }

    // MARK: - The router scores

    @Test(arguments: activationDTypes)
    func theCompiledRouterScoresGiveTheBitsOfTheOpPath(dtype: DType) {
        let logits = Self.randomArray(
            Self.decodeShape + [Self.expertCount], seed: Seed.logits, dtype: dtype)
        let bias = Self.randomArray([Self.expertCount], seed: Seed.bias)

        let reference = FlintExpertRouting.scores(logits: logits, bias: bias)
        let (scores, choice) = FlintCompiledDecode.routerScores(logits, bias)

        expectSameBits(scores, reference.scores, "scores")
        expectSameBits(choice, reference.choice, "choice")
    }

    // MARK: - The combine

    @Test(arguments: activationDTypes, normalizations)
    func theCompiledCombineGivesTheBitsOfTheOpPath(dtype: DType, normalizesWeights: Bool) {
        let routing = Self.routing(normalizesWeights: normalizesWeights)
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: Seed.x, dtype: dtype)
        let routed = Self.randomArray(
            Self.decodeShape + [Self.expertsPerToken, Self.hiddenSize], seed: Seed.routed,
            dtype: dtype)
        let weights = Self.routerWeights(seed: Seed.weights)
        let shared = Self.randomArray(
            Self.decodeShape + [Self.hiddenSize], seed: Seed.shared, dtype: dtype)
        let residual = Self.randomArray(
            Self.decodeShape + [Self.hiddenSize], seed: Seed.residual, dtype: dtype)

        let reference =
            residual + routing.combined(x, routed: routed, weights: weights, shared: shared)
        let compiled = FlintCompiledDecode.combine(routing)(x, routed, weights, shared, residual)

        expectSameBits(compiled, reference, "combine")
    }

    @Test(arguments: activationDTypes, normalizations)
    func theCompiledCombineWithoutASharedExpertGivesTheBitsOfTheOpPath(
        dtype: DType, normalizesWeights: Bool
    ) {
        let routing = Self.routing(normalizesWeights: normalizesWeights)
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: Seed.x, dtype: dtype)
        let routed = Self.randomArray(
            Self.decodeShape + [Self.expertsPerToken, Self.hiddenSize], seed: Seed.routed,
            dtype: dtype)
        let weights = Self.routerWeights(seed: Seed.weights)
        let residual = Self.randomArray(
            Self.decodeShape + [Self.hiddenSize], seed: Seed.residual, dtype: dtype)

        let reference =
            residual + routing.combined(x, routed: routed, weights: weights, shared: nil)
        let compiled = FlintCompiledDecode.combineRoutedExperts(routing)(
            x, routed, weights, residual)

        expectSameBits(compiled, reference, "combine without a shared expert")
    }

    // MARK: - The MoE block

    /// A tiny Flint configuration whose MoE block reads ``expertCount``
    /// experts, ``expertsPerToken`` of them for each token, and one shared
    /// expert.
    private static let configurationJSON = """
        {
            "model_type": "flint", "hidden_size": \(hiddenSize), "num_hidden_layers": 1,
            "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "layer_types": ["full_attention"],
            "softmax_type": "learnable", "num_experts": \(expertCount),
            "num_experts_per_tok": \(expertsPerToken), "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 40, "router_score_function": "sigmoid",
            "norm_topk_prob": true, "hidden_act": "silu"
        }
        """

    /// The key of the expert bias among the parameters of a MoE block.
    private static let expertBiasKey = "expert_bias"

    /// A MoE block of ``configurationJSON`` with a nonzero expert bias, whose
    /// floating-point parameters take `dtype`. The expert bias stays in
    /// float32, as ``FlintModel/sanitize(weights:)`` stores it.
    private static func moeBlock(dtype: DType) throws -> FlintSparseMoeBlock {
        let configuration = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configurationJSON.utf8))
        // The initial weights draw from a random state of their own, so that
        // building the block leaves the global random state of the bundle as
        // it was for every other suite.
        let block = withRandomState(MLXRandom.RandomState(seed: Seed.blockWeights)) {
            FlintSparseMoeBlock(configuration)
        }
        let parameters = block.parameters().flattened().map { key, value in
            guard key != expertBiasKey else {
                return (key, randomArray(value.shape, seed: Seed.expertBias))
            }
            return (key, value.dtype == .float32 ? value.asType(dtype) : value)
        }
        try block.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(block)
        return block
    }

    @Test(arguments: activationDTypes)
    func aDecodeStepOfTheMoEBlockGivesTheBitsOfTheOpPath(dtype: DType) throws {
        let block = try Self.moeBlock(dtype: dtype)
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: Seed.x, dtype: dtype)
        let residual = Self.randomArray(
            Self.decodeShape + [Self.hiddenSize], seed: Seed.residual, dtype: dtype)

        let reference = residual + block(x)
        let compiled = block(x, residual: residual)

        expectSameBits(compiled, reference, "MoE block")
    }

    /// The token count of a prefill block in the trace test.
    private static let prefillTokens = 5

    @Test func aPrefillBlockKeepsTheOpPathOfTheMoEBlock() throws {
        let block = try Self.moeBlock(dtype: .float32)
        let x = Self.randomArray([1, Self.prefillTokens, Self.hiddenSize], seed: Seed.x)

        eval(block(x, residual: x))

        #expect(!block.compiledDecodeStep.isCompiled)
    }

    @Test func aDecodeStepRunsThroughTheCompiledTraceOfTheMoEBlock() throws {
        let block = try Self.moeBlock(dtype: .float32)
        let x = Self.randomArray(Self.decodeShape + [Self.hiddenSize], seed: Seed.x)

        eval(block(x, residual: x))

        #expect(block.compiledDecodeStep.isCompiled)
    }
}
