// Copyright © 2026 Apple Inc.
//
// The contract of ``GatedDeltaDecodeKernel`` in the Qwen 3.5 gated delta net.
//
// On one decode token the layer body runs in one Metal kernel. The kernel
// output must stay inside a small tolerance of the op path
// (`recurrentBody`) over the same inputs, and the conv state after the token
// must be the same bits. A positive control shows that the tolerance
// catches a kernel that does not read its recurrent state or its conv state.
// `GDNDecodeKernelLayerTests` holds the same contract for the whole layer of
// each model kind.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35GDNDecodeKernelTests {

    /// The shared operands of the GDN decode kernel tests.
    typealias Fixture = GDNDecodeKernelFixture

    /// A key head width that the 32 lanes of a SIMD group cannot share.
    private static let narrowKeyHeadDim = 16

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

    /// Repeatable random operands for one case in one dtype. The token
    /// inputs are slices of one projected row, as the fused input projection
    /// gives them.
    static func operands(_ headCase: Fixture.HeadCase, dtype: DType) throws -> Operands {
        let layer = try #require(
            Fixture.layer(Fixture.mlxLLMQwen35, headCase, dtype: dtype) as? Qwen35GatedDeltaNet)
        let valueDim = layer.valueDim
        let qkvEnd = layer.convDim
        let zEnd = qkvEnd + valueDim
        let bEnd = zEnd + layer.numVHeads
        let projected = MLXRandom.normal(
            [headCase.batch, 1, bEnd + layer.numVHeads], key: MLXRandom.key(Fixture.Seed.projected)
        ).asType(dtype)
        let states = Fixture.states(of: layer, batch: headCase.batch, dtype: dtype)
        eval(projected)
        return Operands(
            layer: layer,
            qkv: projected[0..., 0..., ..<qkvEnd],
            z: projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                headCase.batch, 1, layer.numVHeads, layer.headVDim),
            b: projected[0..., 0..., zEnd ..< bEnd],
            a: projected[0..., 0..., bEnd...],
            convState: states.conv, recurrentState: states.recurrent)
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

    @Test(arguments: Fixture.headCases, Fixture.dtypes)
    func theKernelStaysInsideTheToleranceOfTheOpPath(
        headCase: Fixture.HeadCase, dtype: DType
    ) throws {
        let operands = try Self.operands(headCase, dtype: dtype)
        let step = try #require(Self.kernelStep(operands))
        let reference = Self.opPath(
            operands, convState: operands.convState, recurrentState: operands.recurrentState)

        #expect(step.output.shape == reference.output.shape)
        #expect(step.output.dtype == dtype)
        #expect(abs(reference.output).max().item(Float.self) > 0, "every reference value was zero")
        let outputGap = Fixture.worstGap(step.output, reference.output)
        let stateGap = Fixture.worstGap(step.recurrentState, reference.recurrentState)
        #expect(outputGap <= Fixture.tolerance(dtype), "worst output gap \(outputGap)")
        #expect(stateGap <= Fixture.tolerance(dtype), "worst state gap \(stateGap)")
        #expect(step.convState.dtype == reference.convState.dtype)
        #expect(arrayEqual(step.convState, reference.convState).item(Bool.self))
    }

    /// The positive control of the tolerance test: a path that did not read
    /// the recurrent state, or the conv state, breaks the tolerance.
    @Test(arguments: Fixture.headCases)
    func aPathThatSkipsAStateMovesTheOutputPastTheTolerance(
        headCase: Fixture.HeadCase
    ) throws {
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

        #expect(Fixture.worstGap(noRecurrentState.output, full.output) > Fixture.tolerance(dtype))
        #expect(Fixture.worstGap(noConvState.output, full.output) > Fixture.tolerance(dtype))
    }

    /// The kernel reads one token in float16 or bfloat16, over a float32
    /// recurrent state, and only key heads that the lanes share evenly.
    @Test func theKernelReadsOnlyTheStepsItHolds() throws {
        let headCase = Fixture.smallHeadCase
        let operands = try Self.operands(headCase, dtype: .bfloat16)
        let layer = operands.layer
        let twoTokens = concatenated([operands.qkv, operands.qkv], axis: 1)
        let float32Layer = try Self.operands(headCase, dtype: .float32)
        let narrowKeys = try Self.operands(
            Fixture.HeadCase(
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
        let operands = try Self.operands(Fixture.smallHeadCase, dtype: .bfloat16)
        operands.layer.train(true)

        #expect(Self.kernelStep(operands) == nil)
    }
}
