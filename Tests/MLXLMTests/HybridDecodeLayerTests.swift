import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// The hidden width of ``ArithmeticHybridLayer``.
private let hiddenWidth = 4

/// The tokens of each step of these tests: a verify of the fewest tokens.
private let stepTokens = CompiledVerifyShape.fewestTokens

/// The scale ``ArithmeticHybridLayer`` applies in its gated delta mixer.
private let mixerScale: Float = 2

/// The scale ``ArithmeticHybridLayer`` applies in its feed-forward block.
private let feedForwardScale: Float = 3

/// What ``ArithmeticHybridLayer`` adds to each state it gives back.
private let stateStep: Float = 1

/// The places of the attention head outputs among the outputs of a segment.
private typealias Head = CompiledDecodeSegment.AttentionOutput

/// What ``ArithmeticHybridLayer`` adds to its normed input for the attention head output at
/// `place` (one of the places of ``Head``): one more than the place, so each output differs
/// from the others and from the normed input.
private func headOffset(_ place: Int) -> Float {
    Float(place + 1)
}

/// A hybrid decoder layer whose blocks are plain arithmetic, so each output of a shared body
/// shows which blocks made it. Its norms are real `RMSNorm`s with their initial weights.
private final class ArithmeticHybridLayer: Module, HybridDecodeLayer {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    override init() {
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenWidth)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenWidth)
        super.init()
    }

    func gatedDeltaMixer(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray, checkpointAfter: Int?
    ) -> GatedDeltaLayerStep {
        GatedDeltaLayerStep(
            output: x * mixerScale, convState: convState + stateStep,
            recurrentState: recState + stateStep,
            speculative: checkpointAfter.map { _ in
                .checkpoint(conv: convState, recurrent: recState)
            })
    }

    func projectAttention(_ x: MLXArray) -> (
        queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        (
            x + headOffset(Head.queries), x + headOffset(Head.gate),
            x + headOffset(Head.keys), x + headOffset(Head.values)
        )
    }

    func mergeAttention(_ attention: MLXArray, gate: MLXArray) -> MLXArray {
        attention * gate
    }

    func feedForward(_ x: MLXArray) -> MLXArray {
        x * feedForwardScale
    }
}

/// A hidden-state input of `stepTokens` tokens, distinct for each `offset`.
private func hiddenStates(offset: Float = 0) -> MLXArray {
    (MLXArray(0 ..< (stepTokens * hiddenWidth)).asType(.float32) + offset)
        .reshaped([1, stepTokens, hiddenWidth])
}

/// The residual feed-forward step that closes each body: `h + feedForward(postNorm(h))`.
private func closed(_ h: MLXArray, by layer: ArithmeticHybridLayer) -> MLXArray {
    h + layer.postAttentionLayerNorm(h) * feedForwardScale
}

/// The output the gated delta body of `layer` gives for `x`.
private func linearOutput(_ x: MLXArray, of layer: ArithmeticHybridLayer) -> MLXArray {
    closed(x + layer.inputLayerNorm(x) * mixerScale, by: layer)
}

/// Checks that `got` and `expected` hold the same values.
private func expectEqual(
    _ got: MLXArray, _ expected: MLXArray, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(got.shape == expected.shape, sourceLocation: sourceLocation)
    #expect(allClose(got, expected).item(Bool.self), sourceLocation: sourceLocation)
}

@Suite struct HybridDecodeLayerTests {

    @Test func linearLayerBodyClosesTheMixerWithTheResidualFeedForward() {
        let layer = ArithmeticHybridLayer()
        let x = hiddenStates()
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let step = layer.linearLayerBody(x: x, convState: convState, recState: recState)

        expectEqual(step.output, linearOutput(x, of: layer))
        expectEqual(step.convState, convState + stateStep)
        expectEqual(step.recurrentState, recState + stateStep)
        #expect(step.speculative == nil)
    }

    @Test func linearLayerBodyGivesBackTheStatesAVerifyKeeps() throws {
        let layer = ArithmeticHybridLayer()
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let step = layer.linearLayerBody(
            x: hiddenStates(), convState: convState, recState: recState, checkpointAfter: 1)

        let kept = try #require(step.speculative).traceArrays
        expectEqual(kept[0], recState)
        expectEqual(kept[1], convState)
    }

    @Test func attentionBodiesProjectAndMergeAroundTheResidual() {
        let layer = ArithmeticHybridLayer()
        let x = hiddenStates()
        let attention = hiddenStates(offset: 5)
        let gate = hiddenStates(offset: 7)

        let head = layer.attentionPreBody(x: x)
        let normed = layer.inputLayerNorm(x)
        expectEqual(head.queries, normed + headOffset(Head.queries))
        expectEqual(head.gate, normed + headOffset(Head.gate))
        expectEqual(head.keys, normed + headOffset(Head.keys))
        expectEqual(head.values, normed + headOffset(Head.values))

        expectEqual(
            layer.attentionPostBody(x: x, attention: attention, gate: gate),
            closed(x + attention * gate, by: layer))
    }

    /// A linear layer, a full-attention layer, then a linear layer: two segments, split at
    /// the KV write of layer 1.
    private func schedule() -> (segments: [CompiledDecodeSegment], layers: [ArithmeticHybridLayer])
    {
        let segments = CompiledDecodeSegment.schedule(linearLayers: [true, false, true])
        return (segments, (0 ..< 3).map { _ in ArithmeticHybridLayer() })
    }

    @Test func segmentBodyClosingWithAHeadGivesTheStatesThenTheHead() {
        let (segments, layers) = schedule()
        let x = hiddenStates()
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let outputs = segments[0].body(
            x, arguments: [x, convState, recState], layers: layers,
            finalNorm: RMSNorm(dimensions: hiddenWidth), verify: nil)

        let hidden = linearOutput(x, of: layers[0])
        let normed = layers[1].inputLayerNorm(hidden)
        let head = segments[0].attentionOutputOffset
        #expect(outputs.count == head + Head.count)
        expectEqual(outputs[0], hidden)
        expectEqual(outputs[1], convState + stateStep)
        expectEqual(outputs[2], recState + stateStep)
        expectEqual(outputs[head + Head.queries], normed + headOffset(Head.queries))
        expectEqual(outputs[head + Head.values], normed + headOffset(Head.values))
    }

    @Test func segmentBodyOfADecodeStepEndsWithTheFinalNorm() {
        let (segments, layers) = schedule()
        let x = hiddenStates()
        let attention = hiddenStates(offset: 5)
        let gate = hiddenStates(offset: 7)
        let finalNorm = RMSNorm(dimensions: hiddenWidth)

        let outputs = segments[1].body(
            x, arguments: [x, attention, gate, hiddenStates(offset: 10), hiddenStates(offset: 20)],
            layers: layers, finalNorm: finalNorm, verify: nil)

        let hidden = linearOutput(closed(x + attention * gate, by: layers[1]), of: layers[2])
        #expect(outputs.count == 1 + CompiledDecodeSegment.statesPerLinearLayer)
        expectEqual(outputs[0], finalNorm(hidden))
    }

    @Test func segmentBodyOfAVerifyLeavesTheNormAndGivesBackTheKeptStates() throws {
        let (segments, layers) = schedule()
        let verify = try #require(CompiledVerifyShape(tokens: stepTokens, checkpointAfter: 1))
        let x = hiddenStates()
        let attention = hiddenStates(offset: 5)
        let gate = hiddenStates(offset: 7)
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let outputs = segments[1].body(
            x, arguments: [x, attention, gate, convState, recState], layers: layers,
            finalNorm: RMSNorm(dimensions: hiddenWidth), verify: verify)

        let hidden = linearOutput(closed(x + attention * gate, by: layers[1]), of: layers[2])
        #expect(
            outputs.count
                == 1 + CompiledDecodeSegment.statesPerLinearLayer
                + GatedDeltaSpeculativeStates.traceArrayCount)
        expectEqual(outputs[0], hidden)
        let kept = 1 + CompiledDecodeSegment.statesPerLinearLayer
        expectEqual(outputs[kept], recState)
        expectEqual(outputs[kept + 1], convState)
    }
}
