// Copyright © 2026 Apple Inc.
//
// The shared operands of the tests of ``GatedDeltaDecodeKernel`` in the gated
// delta net (GDN) layers: the head shapes, the dtypes, the tolerances, the
// random layers, states and tokens, and the measure of the gap between two
// paths. Each GDN layer meets the same contract, thus each test of a layer
// reads this one fixture.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// A gated delta net layer, as the decode kernel tests read it.
protocol GDNDecodeKernelTestLayer: GatedDeltaDecodeLayer {
    /// The number of conv channels: the q, the k and the value channels.
    var convDim: Int { get }
    /// The number of value heads.
    var numVHeads: Int { get }
    /// The width of one value head.
    var headVDim: Int { get }
    /// The width of one key head.
    var headKDim: Int { get }
    /// The output projection.
    var outProj: Linear { get }
    /// The gated RMSNorm.
    var gatedNorm: Module { get }

    /// The slices of the input projection of `inputs`.
    ///
    /// - Parameters:
    ///   - inputs: The hidden states, shape `(batch, sequence, hiddenSize)`.
    ///   - batch: The number of sequences.
    ///   - sequence: The number of tokens of each sequence.
    /// - Returns: The conv input, the gate, the beta input and the decay input.
    func projectInputs(_ inputs: MLXArray, batch: Int, sequence: Int) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    )

    /// The whole layer over the tokens of `inputs`, from the given states.
    ///
    /// - Parameters:
    ///   - inputs: The hidden states, shape `(batch, sequence, hiddenSize)`.
    ///   - convState: The conv state before the tokens.
    ///   - recurrentState: The recurrent state before the tokens.
    ///   - mask: The token mask, shape `(batch, sequence)`, or nil.
    /// - Returns: The output of the layer and the states after the tokens.
    func runTokens(
        _ inputs: MLXArray, convState: MLXArray, recurrentState: MLXArray, mask: MLXArray?
    ) -> GDNDecodeKernelFixture.LayerStep
}

extension Qwen35GatedDeltaNet: GDNDecodeKernelTestLayer {
    /// The gated RMSNorm.
    var gatedNorm: Module { norm }

    /// The layer body through `forward`.
    func runTokens(
        _ inputs: MLXArray, convState: MLXArray, recurrentState: MLXArray, mask: MLXArray?
    ) -> GDNDecodeKernelFixture.LayerStep {
        let step = forward(inputs, convState: convState, recState: recurrentState, mask: mask)
        return GDNDecodeKernelFixture.LayerStep(
            output: step.output, convState: step.convState, recurrentState: step.recurrentState)
    }
}

extension Qwen3NextGatedDeltaNet: GDNDecodeKernelTestLayer {
    /// The gated RMSNorm.
    var gatedNorm: Module { norm }

    /// The layer body through `forward`.
    func runTokens(
        _ inputs: MLXArray, convState: MLXArray, recurrentState: MLXArray, mask: MLXArray?
    ) -> GDNDecodeKernelFixture.LayerStep {
        let step = forward(inputs, convState: convState, recState: recurrentState, mask: mask)
        return GDNDecodeKernelFixture.LayerStep(
            output: step.output, convState: step.convState, recurrentState: step.recurrentState)
    }
}

extension Qwen35Language.GatedDeltaNet: GDNDecodeKernelTestLayer {
    /// The gated RMSNorm.
    var gatedNorm: Module { norm }

    /// The layer body through `callAsFunction`, which reads and writes the
    /// states in a cache.
    func runTokens(
        _ inputs: MLXArray, convState: MLXArray, recurrentState: MLXArray, mask: MLXArray?
    ) -> GDNDecodeKernelFixture.LayerStep {
        let cache = MambaCache()
        cache[0] = convState
        cache[1] = recurrentState
        let output = self(inputs, mask: mask, cache: cache)
        return GDNDecodeKernelFixture.LayerStep(
            output: output, convState: cache[0]!, recurrentState: cache[1]!)
    }
}

/// The shared operands of the GDN decode kernel tests.
enum GDNDecodeKernelFixture {

    /// One head shape of the tolerance tests.
    struct HeadCase: CustomTestStringConvertible, Sendable {
        /// The number of sequences.
        let batch: Int
        /// The number of key heads.
        let keyHeads: Int
        /// The number of value heads.
        let valueHeads: Int
        /// The width of one key head.
        let keyHeadDim: Int
        /// The width of one value head.
        let valueHeadDim: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(batch) x \(keyHeads) key heads of \(keyHeadDim), "
                + "\(valueHeads) value heads of \(valueHeadDim)"
        }
    }

    /// The shapes of the tolerance tests.
    ///
    /// 1. The Qwen3.8 GDN shape: 16 key heads and 48 value heads of 128.
    /// 2. Two sequences of a small model: one SIMD group for each value row.
    static let headCases = [
        HeadCase(batch: 1, keyHeads: 16, valueHeads: 48, keyHeadDim: 128, valueHeadDim: 128),
        HeadCase(batch: 2, keyHeads: 2, valueHeads: 4, keyHeadDim: 32, valueHeadDim: 32),
    ]

    /// The small head shape, for the tests that do not need the large one.
    static var smallHeadCase: HeadCase { headCases[1] }

    /// The dtypes the kernel reads.
    static let dtypes: [DType] = [.bfloat16, .float16]

    /// One GDN layer kind: a model layer that runs its decode token through
    /// the kernel.
    struct LayerKind: CustomTestStringConvertible, Sendable {
        /// The name the test report shows for this kind.
        let testDescription: String
        /// Makes a layer of this kind for a head shape.
        let make: @Sendable (HeadCase) throws -> any GDNDecodeKernelTestLayer
    }

    /// The GDN layers that run their decode token through the kernel.
    static let layerKinds = [
        LayerKind(testDescription: "MLXLLM Qwen 3.5") { headCase in
            Qwen35GatedDeltaNet(try configuration(Qwen35TextConfiguration.self, headCase))
        },
        LayerKind(testDescription: "Qwen3-Next") { headCase in
            Qwen3NextGatedDeltaNet(try configuration(Qwen3NextConfiguration.self, headCase))
        },
        LayerKind(testDescription: "MLXVLM Qwen 3.5") { headCase in
            Qwen35Language.GatedDeltaNet(
                try configuration(Qwen35Configuration.TextConfiguration.self, headCase))
        },
    ]

    /// The output of a layer over its tokens and the states after them.
    struct LayerStep {
        /// The output of the layer.
        let output: MLXArray
        /// The conv state after the tokens.
        let convState: MLXArray
        /// The recurrent state after the tokens.
        let recurrentState: MLXArray
    }

    /// The hidden size of the test layers. The size of the input projection
    /// does not change the kernel, thus a small size keeps the layer small.
    static let hiddenSize = 64

    /// The conv taps of the Qwen 3.5 and Qwen3-Next checkpoints.
    static let convTaps = 4

    /// The query heads of the full-attention layers of the test
    /// configuration. The tests never run these layers.
    private static let attentionHeads = 2

    /// The head width of the full-attention layers of the test
    /// configuration.
    private static let attentionHeadDim = 32

    /// The vocabulary size of the test configuration.
    private static let vocabularySize = 32

    /// The layer interval of the full-attention layers of the test
    /// configuration.
    private static let fullAttentionInterval = 2

    /// The experts of the sparse MLP blocks of the test configuration. The
    /// tests never run these blocks.
    private static let experts = 4

    /// The experts of each token in the test configuration.
    private static let expertsPerToken = 2

    /// The epsilon of the RMSNorms of the test configuration.
    private static let rmsNormEps: Float = 1e-6

    /// The largest gap allowed between the kernel and the op path, as a part
    /// of the largest reference value, for bfloat16: two steps of 2^-8.
    private static let bfloat16Tolerance: Float = 8e-3

    /// The largest gap for float16: two steps of 2^-10.
    private static let float16Tolerance: Float = 2e-3

    /// The size of the random recurrent state. A state of this size moves
    /// the output well past the tolerance when a path does not read it.
    private static let stateScale: Float = 0.5

    /// The seed of each random operand.
    enum Seed {
        /// The seed of a projected row of token inputs.
        static let projected: UInt64 = 20_261_015
        /// The seed of the conv state.
        static let convState: UInt64 = 20_261_016
        /// The seed of the recurrent state.
        static let recurrentState: UInt64 = 20_261_017
        /// The seed of the gated RMSNorm weight.
        static let normWeight: UInt64 = 20_261_018
        /// The seed of the hidden states of a token.
        static let token: UInt64 = 20_261_019
    }

    /// The configuration of one GDN layer of a head shape. One JSON text
    /// holds the keys of the Qwen 3.5 and the Qwen3-Next configurations, thus
    /// every layer kind reads the same shape.
    ///
    /// - Parameters:
    ///   - type: The configuration type of the layer kind.
    ///   - headCase: The head shape.
    /// - Returns: The configuration.
    /// - Throws: The error of the decoder.
    static func configuration<Configuration: Decodable>(
        _ type: Configuration.Type, _ headCase: HeadCase
    ) throws -> Configuration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 1,
                "intermediate_size": \(hiddenSize),
                "num_attention_heads": \(attentionHeads),
                "num_key_value_heads": 1,
                "head_dim": \(attentionHeadDim),
                "linear_num_value_heads": \(headCase.valueHeads),
                "linear_num_key_heads": \(headCase.keyHeads),
                "linear_key_head_dim": \(headCase.keyHeadDim),
                "linear_value_head_dim": \(headCase.valueHeadDim),
                "linear_conv_kernel_dim": \(convTaps),
                "vocab_size": \(vocabularySize),
                "full_attention_interval": \(fullAttentionInterval),
                "num_experts": \(experts),
                "num_experts_per_tok": \(expertsPerToken),
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": \(hiddenSize),
                "moe_intermediate_size": \(hiddenSize),
                "rms_norm_eps": \(rmsNormEps)
            }
            """
        return try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    /// Sets the parameters of a layer to `dtype`, gives its gated RMSNorm a
    /// random weight, and puts it in inference mode.
    ///
    /// - Parameters:
    ///   - layer: The layer.
    ///   - dtype: The dtype of the parameters.
    /// - Throws: The error of the parameter update.
    static func prepare(_ layer: some GDNDecodeKernelTestLayer, dtype: DType) throws {
        layer.update(parameters: layer.parameters().mapValues { $0.asType(dtype) })
        let normWeight =
            1 + MLXRandom.normal([layer.headVDim], key: MLXRandom.key(Seed.normWeight))
        try layer.gatedNorm.update(
            parameters: ModuleParameters.unflattened(["weight": normWeight.asType(dtype)]),
            verify: .all)
        layer.train(false)
        eval(layer)
    }

    /// A layer of a kind for a head shape, prepared in `dtype`.
    ///
    /// - Parameters:
    ///   - kind: The layer kind.
    ///   - headCase: The head shape.
    ///   - dtype: The dtype of the parameters.
    /// - Returns: The layer.
    /// - Throws: The error of the configuration decoder or of the update.
    static func layer(_ kind: LayerKind, _ headCase: HeadCase, dtype: DType) throws
        -> any GDNDecodeKernelTestLayer
    {
        let layer = try kind.make(headCase)
        try prepare(layer, dtype: dtype)
        return layer
    }

    /// Repeatable random states of a layer before a token.
    ///
    /// - Parameters:
    ///   - layer: The layer.
    ///   - batch: The number of sequences.
    ///   - dtype: The dtype of the conv state.
    /// - Returns: The conv state and the float32 recurrent state.
    static func states(of layer: some GDNDecodeKernelTestLayer, batch: Int, dtype: DType) -> (
        conv: MLXArray, recurrent: MLXArray
    ) {
        let conv = MLXRandom.normal(
            [batch, convTaps - 1, layer.convDim], key: MLXRandom.key(Seed.convState)
        ).asType(dtype)
        let recurrent =
            stateScale
            * MLXRandom.normal(
                [batch, layer.numVHeads, layer.headVDim, layer.headKDim],
                key: MLXRandom.key(Seed.recurrentState))
        eval(conv, recurrent)
        return (conv, recurrent)
    }

    /// The repeatable random hidden states of one decode token.
    ///
    /// - Parameters:
    ///   - batch: The number of sequences.
    ///   - dtype: The dtype of the hidden states.
    /// - Returns: The hidden states, shape `(batch, 1, hiddenSize)`.
    static func token(batch: Int, dtype: DType) -> MLXArray {
        let token = MLXRandom.normal([batch, 1, hiddenSize], key: MLXRandom.key(Seed.token))
            .asType(dtype)
        eval(token)
        return token
    }

    /// The largest gap between two arrays, as a part of the largest absolute
    /// value of the expected array.
    ///
    /// - Parameters:
    ///   - got: The array of the path under test.
    ///   - expected: The array of the reference path.
    /// - Returns: The gap.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let expected = expected.asType(.float32)
        let gap = abs(got.asType(.float32) - expected).max().item(Float.self)
        return gap / abs(expected).max().item(Float.self)
    }

    /// The tolerance of one dtype.
    ///
    /// - Parameter dtype: The dtype of the token inputs.
    /// - Returns: The largest gap allowed.
    static func tolerance(_ dtype: DType) -> Float {
        dtype == .bfloat16 ? bfloat16Tolerance : float16Tolerance
    }
}
