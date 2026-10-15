// Copyright © 2026 Apple Inc.
//
// The contract of ``GatedDeltaDecodeKernel`` in the Qwen 3.5 gated delta net.
//
// On one decode token the layer body runs in one Metal kernel. The kernel
// output must stay inside a small tolerance of the op path
// (`recurrentBody`) over the same inputs, and the conv state after the token
// must be the same bits. A positive control shows that the tolerance
// catches a kernel that does not read its recurrent state or its conv state.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35GDNDecodeKernelTests {

    /// One head shape of the tolerance test.
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

    /// The shapes of the tolerance test.
    ///
    /// 1. The Qwen3.8 GDN shape: 16 key heads and 48 value heads of 128.
    /// 2. Two sequences of a small model: one SIMD group for each value row.
    static let headCases = [
        HeadCase(batch: 1, keyHeads: 16, valueHeads: 48, keyHeadDim: 128, valueHeadDim: 128),
        HeadCase(batch: 2, keyHeads: 2, valueHeads: 4, keyHeadDim: 32, valueHeadDim: 32),
    ]

    /// The dtypes the kernel reads.
    static let dtypes: [DType] = [.bfloat16, .float16]

    /// The hidden size of the test layers. The test never runs the input
    /// projection, thus a small size keeps the layer small.
    private static let hiddenSize = 64

    /// The conv taps of the Qwen 3.5 checkpoints.
    private static let convTaps = 4

    /// The query heads of the full-attention layers of the test
    /// configuration. The test never runs these layers.
    private static let attentionHeads = 2

    /// The head width of the full-attention layers of the test
    /// configuration.
    private static let attentionHeadDim = 32

    /// The vocabulary size of the test configuration.
    private static let vocabularySize = 32

    /// The layer interval of the full-attention layers of the test
    /// configuration.
    private static let fullAttentionInterval = 2

    /// The largest gap allowed between the kernel and the op path, as a part
    /// of the largest reference value, for bfloat16: two steps of 2^-8.
    private static let bfloat16Tolerance: Float = 8e-3

    /// The largest gap for float16: two steps of 2^-10.
    private static let float16Tolerance: Float = 2e-3

    /// A key head width that the 32 lanes of a SIMD group cannot share.
    private static let narrowKeyHeadDim = 16

    /// The size of the random recurrent state. A state of this size moves
    /// the output well past the tolerance when a path does not read it.
    private static let stateScale: Float = 0.5

    /// The seed of each random operand.
    private enum Seed {
        static let projected: UInt64 = 20_261_015
        static let convState: UInt64 = 20_261_016
        static let recurrentState: UInt64 = 20_261_017
        static let normWeight: UInt64 = 20_261_018
    }

    /// The operands of one decode step.
    struct Operands {
        let layer: Qwen35GatedDeltaNet
        let qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        let convState: MLXArray
        let recurrentState: MLXArray
    }

    /// A GDN layer of the head shape of a case, with its parameters in
    /// `dtype` and a random gated RMSNorm weight.
    static func layer(_ headCase: HeadCase, dtype: DType) throws -> Qwen35GatedDeltaNet {
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
                "full_attention_interval": \(fullAttentionInterval)
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let layer = Qwen35GatedDeltaNet(configuration)
        layer.update(parameters: layer.parameters().mapValues { $0.asType(dtype) })
        let normWeight =
            1 + MLXRandom.normal([headCase.valueHeadDim], key: MLXRandom.key(Seed.normWeight))
        try layer.norm.update(
            parameters: ModuleParameters.unflattened(["weight": normWeight.asType(dtype)]),
            verify: .all)
        layer.train(false)
        return layer
    }

    /// Repeatable random operands for one case in one dtype. The token
    /// inputs are slices of one projected row, as the fused input projection
    /// gives them.
    static func operands(_ headCase: HeadCase, dtype: DType) throws -> Operands {
        let layer = try layer(headCase, dtype: dtype)
        let valueDim = layer.valueDim
        let qkvEnd = layer.convDim
        let zEnd = qkvEnd + valueDim
        let bEnd = zEnd + layer.numVHeads
        let projected = MLXRandom.normal(
            [headCase.batch, 1, bEnd + layer.numVHeads], key: MLXRandom.key(Seed.projected)
        ).asType(dtype)
        let convState = MLXRandom.normal(
            [headCase.batch, convTaps - 1, layer.convDim], key: MLXRandom.key(Seed.convState)
        ).asType(dtype)
        let recurrentState =
            stateScale
            * MLXRandom.normal(
                [headCase.batch, layer.numVHeads, layer.headVDim, layer.headKDim],
                key: MLXRandom.key(Seed.recurrentState))
        eval(projected, convState, recurrentState)
        return Operands(
            layer: layer,
            qkv: projected[0..., 0..., ..<qkvEnd],
            z: projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                headCase.batch, 1, layer.numVHeads, layer.headVDim),
            b: projected[0..., 0..., zEnd ..< bEnd],
            a: projected[0..., 0..., bEnd...],
            convState: convState, recurrentState: recurrentState)
    }

    /// The op path over the operands, with a recurrent state and a conv
    /// state of choice.
    static func opPath(
        _ operands: Operands, convState: MLXArray, recurrentState: MLXArray
    ) -> (output: MLXArray, convState: MLXArray, recurrentState: MLXArray) {
        let body = operands.layer.recurrentBody(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
            convState: convState, recState: recurrentState, mask: nil)
        return (body.output, body.convState, body.recurrentState)
    }

    /// The kernel step over the operands, or nil when the kernel refuses
    /// them.
    static func kernelStep(_ operands: Operands) -> GatedDeltaDecodeKernel.Step? {
        operands.layer.decodeKernelStep(
            qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
            convState: operands.convState, recState: operands.recurrentState)
    }

    /// The largest gap between two arrays, as a part of the largest absolute
    /// value of the expected array.
    static func worstGap(_ got: MLXArray, _ expected: MLXArray) -> Float {
        let expected = expected.asType(.float32)
        let gap = abs(got.asType(.float32) - expected).max().item(Float.self)
        return gap / abs(expected).max().item(Float.self)
    }

    /// The tolerance of one dtype.
    static func tolerance(_ dtype: DType) -> Float {
        dtype == .bfloat16 ? bfloat16Tolerance : float16Tolerance
    }

    @Test(arguments: headCases, dtypes)
    func theKernelStaysInsideTheToleranceOfTheOpPath(headCase: HeadCase, dtype: DType) throws {
        let operands = try Self.operands(headCase, dtype: dtype)
        let step = try #require(Self.kernelStep(operands))
        let reference = Self.opPath(
            operands, convState: operands.convState, recurrentState: operands.recurrentState)

        #expect(step.output.shape == reference.output.shape)
        #expect(step.output.dtype == dtype)
        #expect(abs(reference.output).max().item(Float.self) > 0, "every reference value was zero")
        let outputGap = Self.worstGap(step.output, reference.output)
        let stateGap = Self.worstGap(step.recurrentState, reference.recurrentState)
        #expect(outputGap <= Self.tolerance(dtype), "worst output gap \(outputGap)")
        #expect(stateGap <= Self.tolerance(dtype), "worst state gap \(stateGap)")
        #expect(step.convState.dtype == reference.convState.dtype)
        #expect(arrayEqual(step.convState, reference.convState).item(Bool.self))
    }

    /// The positive control of the tolerance test: a path that did not read
    /// the recurrent state, or the conv state, breaks the tolerance.
    @Test(arguments: headCases)
    func aPathThatSkipsAStateMovesTheOutputPastTheTolerance(headCase: HeadCase) throws {
        let dtype = DType.bfloat16
        let operands = try Self.operands(headCase, dtype: dtype)
        let full = Self.opPath(
            operands, convState: operands.convState, recurrentState: operands.recurrentState)
        let noRecurrentState = Self.opPath(
            operands, convState: operands.convState,
            recurrentState: zeros(like: operands.recurrentState))
        let noConvState = Self.opPath(
            operands, convState: zeros(like: operands.convState),
            recurrentState: operands.recurrentState)

        #expect(Self.worstGap(noRecurrentState.output, full.output) > Self.tolerance(dtype))
        #expect(Self.worstGap(noConvState.output, full.output) > Self.tolerance(dtype))
    }

    /// The kernel reads one token in float16 or bfloat16, over a float32
    /// recurrent state, and only key heads that the lanes share evenly.
    @Test func theKernelReadsOnlyTheStepsItHolds() throws {
        let headCase = Self.headCases[1]
        let operands = try Self.operands(headCase, dtype: .bfloat16)
        let layer = operands.layer
        let twoTokens = concatenated([operands.qkv, operands.qkv], axis: 1)
        let float32Layer = try Self.operands(headCase, dtype: .float32)
        let narrowKeys = try Self.operands(
            HeadCase(
                batch: headCase.batch, keyHeads: headCase.keyHeads,
                valueHeads: headCase.valueHeads, keyHeadDim: Self.narrowKeyHeadDim,
                valueHeadDim: headCase.valueHeadDim),
            dtype: .bfloat16)

        #expect(Self.kernelStep(operands) != nil)
        #expect(Self.kernelStep(narrowKeys) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: twoTokens, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState, recState: operands.recurrentState) == nil)
        #expect(Self.kernelStep(float32Layer) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState,
                recState: operands.recurrentState.asType(.bfloat16)) == nil)
        #expect(
            layer.decodeKernelStep(
                qkv: operands.qkv, z: operands.z, b: operands.b, a: operands.a,
                convState: operands.convState, recState: nil) == nil)
    }

    /// A layer in training keeps the op path: the kernel has no gradient.
    @Test func aTrainingLayerKeepsTheOpPath() throws {
        let operands = try Self.operands(Self.headCases[1], dtype: .bfloat16)
        operands.layer.train(true)

        #expect(Self.kernelStep(operands) == nil)
    }

    /// One decode token through `forward` gives the kernel output, through
    /// the output projection.
    @Test func forwardRunsOneTokenThroughTheKernel() throws {
        let headCase = Self.headCases[1]
        let operands = try Self.operands(headCase, dtype: .bfloat16)
        let layer = operands.layer
        let x = MLXRandom.normal(
            [headCase.batch, 1, Self.hiddenSize], key: MLXRandom.key(Seed.projected)
        ).asType(.bfloat16)
        let (qkv, z, b, a) = layer.projectInputs(x, batch: headCase.batch, sequence: 1)

        let forward = layer.forward(
            x, convState: operands.convState, recState: operands.recurrentState, mask: nil)
        let step = try #require(
            layer.decodeKernelStep(
                qkv: qkv, z: z, b: b, a: a, convState: operands.convState,
                recState: operands.recurrentState))

        #expect(arrayEqual(forward.output, layer.outProj(step.output)).item(Bool.self))
        #expect(arrayEqual(forward.recurrentState, step.recurrentState).item(Bool.self))
    }
}
