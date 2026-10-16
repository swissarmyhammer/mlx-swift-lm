// Copyright © 2026 Apple Inc.
//
// The contract of ``GatedDeltaDecodeKernel`` in the whole gated delta net
// layer of each model kind: the MLXLLM Qwen 3.5 layer, the Qwen3-Next layer
// and the MLXVLM Qwen 3.5 layer.
//
// On one decode token with no mask the layer runs its body in the kernel. The
// layer output must stay inside a small tolerance of the op path of the same
// layer, and the conv state after the token must be the same bits. The op path
// is the same layer under a mask that keeps the token: the kernel never runs a
// masked step. A positive control shows that the tolerance catches a path that
// does not read its recurrent state or its conv state.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct GDNDecodeKernelLayerTests {

    /// The shared operands of the GDN decode kernel tests.
    typealias Fixture = GDNDecodeKernelFixture

    /// One layer kind with one head shape.
    struct LayerCase: CustomTestStringConvertible, Sendable {
        /// The layer kind.
        let kind: Fixture.LayerKind
        /// The head shape.
        let headCase: Fixture.HeadCase

        /// The text the test report shows for this case.
        var testDescription: String { "\(kind.testDescription), \(headCase.testDescription)" }
    }

    /// Each layer kind with each head shape.
    static let layerCases = Fixture.layerKinds.flatMap { kind in
        Fixture.headCases.map { LayerCase(kind: kind, headCase: $0) }
    }

    /// The operands of one decode token through one layer.
    struct Token {
        /// The layer.
        let layer: any GDNDecodeKernelTestLayer
        /// The hidden states of the token, shape `(batch, 1, hiddenSize)`.
        let inputs: MLXArray
        /// The conv state before the token.
        let convState: MLXArray
        /// The recurrent state before the token.
        let recurrentState: MLXArray
    }

    /// Repeatable random operands of one token for a layer kind and a head
    /// shape in one dtype.
    static func token(_ kind: Fixture.LayerKind, _ headCase: Fixture.HeadCase, dtype: DType)
        throws -> Token
    {
        let layer = try Fixture.layer(kind, headCase, dtype: dtype)
        let states = Fixture.states(of: layer, batch: headCase.batch, dtype: dtype)
        return Token(
            layer: layer, inputs: Fixture.token(batch: headCase.batch, dtype: dtype),
            convState: states.conv, recurrentState: states.recurrent)
    }

    /// The layer over the token with no mask: the path of a decode step.
    static func decodePath(_ token: Token) -> Fixture.LayerStep {
        token.layer.runTokens(
            token.inputs, convState: token.convState, recurrentState: token.recurrentState,
            mask: nil)
    }

    /// The op path of the layer over the token, with a recurrent state and a
    /// conv state of choice: the layer under a mask that keeps the token.
    static func opPath(_ token: Token, convState: MLXArray, recurrentState: MLXArray)
        -> Fixture.LayerStep
    {
        let keepEveryToken = MLXArray.ones([token.inputs.dim(0), 1], dtype: .bool)
        return token.layer.runTokens(
            token.inputs, convState: convState, recurrentState: recurrentState,
            mask: keepEveryToken)
    }

    @Test(arguments: layerCases, Fixture.dtypes)
    func theDecodePathStaysInsideTheToleranceOfTheOpPath(layerCase: LayerCase, dtype: DType)
        throws
    {
        let token = try Self.token(layerCase.kind, layerCase.headCase, dtype: dtype)
        let decode = Self.decodePath(token)
        let reference = Self.opPath(
            token, convState: token.convState, recurrentState: token.recurrentState)

        #expect(decode.output.shape == reference.output.shape)
        #expect(decode.output.dtype == dtype)
        #expect(abs(reference.output).max().item(Float.self) > 0, "every reference value was zero")
        let outputGap = Fixture.worstGap(decode.output, reference.output)
        let stateGap = Fixture.worstGap(decode.recurrentState, reference.recurrentState)
        #expect(outputGap <= Fixture.tolerance(dtype), "worst output gap \(outputGap)")
        #expect(stateGap <= Fixture.tolerance(dtype), "worst state gap \(stateGap)")
        #expect(decode.convState.dtype == reference.convState.dtype)
        #expect(arrayEqual(decode.convState, reference.convState).item(Bool.self))
    }

    /// The positive control of the tolerance test: an op path that did not
    /// read the recurrent state, or the conv state, breaks the tolerance.
    @Test(arguments: layerCases, Fixture.dtypes)
    func anOpPathThatSkipsAStateMovesTheOutputPastTheTolerance(
        layerCase: LayerCase, dtype: DType
    ) throws {
        let token = try Self.token(layerCase.kind, layerCase.headCase, dtype: dtype)
        let full = Self.opPath(
            token, convState: token.convState, recurrentState: token.recurrentState)
        let noRecurrentState = Self.opPath(
            token, convState: token.convState, recurrentState: zeros(like: token.recurrentState))
        let noConvState = Self.opPath(
            token, convState: zeros(like: token.convState), recurrentState: token.recurrentState)

        #expect(Fixture.worstGap(noRecurrentState.output, full.output) > Fixture.tolerance(dtype))
        #expect(Fixture.worstGap(noConvState.output, full.output) > Fixture.tolerance(dtype))
    }

    /// One decode token with no mask runs through the kernel: the graph of
    /// the layer is the graph of the input projections, the kernel and the
    /// output projection, and the layer gives the kernel output and states.
    /// The op path of a small head can give the same bits as the kernel,
    /// thus the graph is what shows that the kernel runs.
    @Test(arguments: Fixture.layerKinds, Fixture.dtypes)
    func oneDecodeTokenRunsThroughTheKernel(kind: Fixture.LayerKind, dtype: DType) throws {
        let headCase = Fixture.smallHeadCase
        let token = try Self.token(kind, headCase, dtype: dtype)
        let layer = token.layer
        let (qkv, z, b, a) = layer.projectInputs(token.inputs, batch: headCase.batch, sequence: 1)
        let step = try #require(
            layer.decodeKernelStep(
                qkv: qkv, z: z, b: b, a: a, convState: token.convState,
                recState: token.recurrentState))
        let kernelOutput = layer.outProj(step.output)

        let decode = Self.decodePath(token)

        #expect(
            DeepSeekV4TestArrays.primitiveNodeCount(
                [decode.output, decode.convState, decode.recurrentState])
                == DeepSeekV4TestArrays.primitiveNodeCount(
                    [kernelOutput, step.convState, step.recurrentState]))
        #expect(arrayEqual(decode.output, kernelOutput).item(Bool.self))
        #expect(arrayEqual(decode.recurrentState, step.recurrentState).item(Bool.self))
        #expect(arrayEqual(decode.convState, step.convState).item(Bool.self))
    }
}
