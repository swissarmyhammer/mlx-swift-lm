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

/// The snapshot slots ``ArithmeticHybridLayer`` keeps for a verify over `x`: the slot after
/// token `t` holds the recurrent state plus `t` state steps, and the conv window holds the
/// conv state, then the mixer input of each token.
///
/// - Parameters:
///   - x: The mixer input, shape `(batch, tokens, hiddenWidth)`.
///   - convState: The conv state before the first token.
///   - recState: The recurrent state before the first token.
/// - Returns: The slots, recurrent shape `(tokens,)` plus the recurrent state shape.
private func arithmeticSnapshots(
    _ x: MLXArray, convState: MLXArray, recState: MLXArray
) -> GatedDeltaDecodeKernel.Snapshots {
    let slots = (1 ... x.dim(1)).map { recState + stateStep * Float($0) }
    return GatedDeltaDecodeKernel.Snapshots(
        recurrent: stacked(slots), convWindow: concatenated([convState, x], axis: 1))
}

/// A hybrid decoder layer whose blocks are plain arithmetic, so each output of a shared body
/// shows which blocks made it. Its norms are real `RMSNorm`s with their initial weights; its
/// mixer modules are only there to be found by the shared accessors.
private final class ArithmeticHybridLayer: Module, HybridDecodeLayer {
    @ModuleInfo(key: "linear_attn") var linearAttn: Identity?
    @ModuleInfo(key: "self_attn") var selfAttn: Identity?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// - Parameters:
    ///   - linearAttn: The gated delta net module of a linear layer, or nil.
    ///   - selfAttn: The attention module of a full-attention layer, or nil.
    init(linearAttn: Identity? = nil, selfAttn: Identity? = nil) {
        _linearAttn.wrappedValue = linearAttn
        _selfAttn.wrappedValue = selfAttn
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenWidth)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenWidth)
        super.init()
    }

    func gatedDeltaMixer(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray, keepsSnapshots: Bool
    ) -> GatedDeltaLayerStep {
        GatedDeltaLayerStep(
            output: x * mixerScale, convState: convState + stateStep,
            recurrentState: recState + stateStep,
            snapshots: keepsSnapshots
                ? arithmeticSnapshots(x, convState: convState, recState: recState) : nil)
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

/// A hybrid model of ``ArithmeticHybridLayer``s, the owner of the compiled segment traces of
/// a test: a linear layer, a full-attention layer, then a linear layer.
private final class ArithmeticHybridModel: Module {
    /// The decoder layers, each with the mixer of its kind.
    let layers = [
        ArithmeticHybridLayer(linearAttn: Identity()), ArithmeticHybridLayer(selfAttn: Identity()),
        ArithmeticHybridLayer(linearAttn: Identity()),
    ]
    /// The final norm.
    let norm = RMSNorm(dimensions: hiddenWidth)
}

/// Runs segment `index` of a decode step of `model` under `schedule`, outside any trace.
private func decodeSegment(
    _ index: Int, of schedule: CompiledDecodeSchedule, _ model: ArithmeticHybridModel,
    _ arguments: [MLXArray]
) -> [MLXArray] {
    schedule.segments[index].body(
        arguments[0], arguments: arguments, layers: model.layers, finalNorm: model.norm,
        verify: nil)
}

/// A hidden-state input of `tokens` tokens, distinct for each `offset`.
private func hiddenStates(offset: Float = 0, tokens: Int = stepTokens) -> MLXArray {
    (MLXArray(0 ..< (tokens * hiddenWidth)).asType(.float32) + offset)
        .reshaped([1, tokens, hiddenWidth])
}

/// True when `mask` is the mask mode `.none`.
private func isNoMask(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
    if case .none = mask { true } else { false }
}

/// True when `mask` is the mask mode `.causal`.
private func isCausal(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
    if case .causal = mask { true } else { false }
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

    @Test func gatedDeltaNetIsTheMixerOfALinearLayer() {
        let mixer = Identity()
        let layer = ArithmeticHybridLayer(linearAttn: mixer)

        #expect(layer.gatedDeltaNet === mixer)
    }

    @Test func fullAttentionIsTheAttentionOfAFullAttentionLayer() {
        let attention = Identity()
        let layer = ArithmeticHybridLayer(selfAttn: attention)

        #expect(layer.fullAttention === attention)
    }

    @Test func linearLayerBodyClosesTheMixerWithTheResidualFeedForward() {
        let layer = ArithmeticHybridLayer()
        let x = hiddenStates()
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let step = layer.linearLayerBody(x: x, convState: convState, recState: recState)

        expectEqual(step.output, linearOutput(x, of: layer))
        expectEqual(step.convState, convState + stateStep)
        expectEqual(step.recurrentState, recState + stateStep)
        #expect(step.snapshots == nil)
    }

    @Test func linearLayerBodyGivesBackTheStatesAVerifyKeeps() throws {
        let layer = ArithmeticHybridLayer()
        let x = hiddenStates()
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let step = layer.linearLayerBody(
            x: x, convState: convState, recState: recState, keepsSnapshots: true)

        let kept = try #require(step.snapshots)
        let expected = arithmeticSnapshots(
            layer.inputLayerNorm(x), convState: convState, recState: recState)
        expectEqual(kept.recurrent, expected.recurrent)
        expectEqual(kept.convWindow, expected.convWindow)
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
        let head = segments[0].attentionOutputOffset(
            outputsPerLinearLayer: CompiledDecodeSegment.statesPerLinearLayer)
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
        let verify = try #require(CompiledVerifyShape(tokens: stepTokens))
        let x = hiddenStates()
        let attention = hiddenStates(offset: 5)
        let gate = hiddenStates(offset: 7)
        let convState = hiddenStates(offset: 10)
        let recState = hiddenStates(offset: 20)

        let outputs = segments[1].body(
            x, arguments: [x, attention, gate, convState, recState], layers: layers,
            finalNorm: RMSNorm(dimensions: hiddenWidth), verify: verify)

        let attended = closed(x + attention * gate, by: layers[1])
        let hidden = linearOutput(attended, of: layers[2])
        #expect(
            outputs.count
                == 1 + CompiledDecodeSegment.statesPerLinearLayer
                + GatedDeltaDecodeKernel.Snapshots.traceArrayCount)
        expectEqual(outputs[0], hidden)
        let kept = 1 + CompiledDecodeSegment.statesPerLinearLayer
        let expected = arithmeticSnapshots(
            layers[2].inputLayerNorm(attended), convState: convState, recState: recState)
        for (got, slots) in zip(outputs[kept...], expected.traceArrays) {
            expectEqual(got, slots)
        }
    }

    // MARK: - Schedule runner

    /// The conv state the GDN cache of layer `layer` holds before a step.
    private func convState(of layer: Int) -> MLXArray {
        hiddenStates(offset: 10 + Float(layer))
    }

    /// The recurrent state the GDN cache of layer `layer` holds before a step.
    private func recState(of layer: Int) -> MLXArray {
        hiddenStates(offset: 20 + Float(layer))
    }

    /// A GDN cache holding ``convState(of:)`` and ``recState(of:)`` of `layer`.
    private func linearCache(of layer: Int) -> MambaCache {
        let cache = MambaCache()
        cache[0] = convState(of: layer)
        cache[1] = recState(of: layer)
        return cache
    }

    /// The caches of ``schedule()``: a GDN cache with its states, `attention`, a GDN cache
    /// with its states.
    private func scheduleCaches(attention: KVCache? = KVCacheSimple()) -> [KVCache?] {
        [linearCache(of: 0), attention, linearCache(of: 2)]
    }

    /// What the attention step of these tests gives back for `step`: the queries plus the
    /// keys, so the result shows which head outputs the runner handed over.
    private func attended(_ step: CompiledDecodeSchedule.AttentionStep) -> MLXArray {
        step.queries + step.keys
    }

    @Test func scheduleOfLayersSplitsAtTheirFullAttentionLayers() {
        let schedule = CompiledDecodeSchedule(layers: ArithmeticHybridModel().layers)

        let expected = CompiledDecodeSegment.schedule(linearLayers: [true, false, true])
        #expect(schedule.segments == expected)
        #expect(schedule.firstLinearLayer == 0)
        #expect(schedule.firstAttentionLayer == 1)
    }

    @Test func scheduleCachesRefuseACacheListOfAnotherLength() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let cache = Array(scheduleCaches().dropLast())

        #expect(schedule.caches(cache, over: hiddenStates()) == nil)
    }

    @Test func scheduleCachesRefuseALinearLayerWithNoStateYet() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let cache: [KVCache?] = [linearCache(of: 0), KVCacheSimple(), MambaCache()]

        #expect(schedule.caches(cache, over: hiddenStates()) == nil)
    }

    @Test func scheduleCachesRefuseAnAttentionCacheWithARouteOfItsOwn() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let cache = scheduleCaches(attention: QuantizedKVCache())

        #expect(schedule.caches(cache, over: hiddenStates()) == nil)
    }

    @Test func scheduleCachesRefuseAMissingAttentionCache() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])

        #expect(schedule.caches(scheduleCaches(attention: nil), over: hiddenStates()) == nil)
    }

    @Test func scheduleCachesRefuseAStepThatNeedsAnSSMMask() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let cache = scheduleCaches()
        cache[0]?.prepare(lengths: [stepTokens])

        #expect(schedule.caches(cache, over: hiddenStates()) == nil)
    }

    @Test func attentionMaskOfOneTokenIsNone() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])

        #expect(isNoMask(schedule.attentionMask(hiddenStates(tokens: 1), scheduleCaches())))
    }

    @Test func attentionMaskOfAVerifyIsCausal() {
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])

        #expect(isCausal(schedule.attentionMask(hiddenStates(), scheduleCaches())))
    }

    /// The compiled segment traces of ``ArithmeticHybridModel`` under `schedule`.
    private func segmentCache(
        _ schedule: CompiledDecodeSchedule
    ) -> CompiledDecodeSegmentCache<ArithmeticHybridModel> {
        CompiledDecodeSegmentCache(
            count: schedule.segments.count,
            state: { model, index in
                schedule.segments[index].layerIndices.map { model.layers[$0] } + [model.norm]
            },
            body: { model, index, arguments in
                decodeSegment(index, of: schedule, model, arguments)
            })
    }

    @Test func decodeRefusesAStepThatNeedsAnAttentionMask() {
        let model = ArithmeticHybridModel()
        let schedule = CompiledDecodeSchedule(layers: model.layers)

        let output = schedule.decode(
            hiddenStates(), cache: scheduleCaches(), segments: segmentCache(schedule),
            of: model, attention: attended)

        #expect(output == nil)
    }

    @Test func decodeRunsTheCompiledSegmentsOfTheSchedule() throws {
        let model = ArithmeticHybridModel()
        let schedule = CompiledDecodeSchedule(layers: model.layers)
        let x = hiddenStates(tokens: 1)
        let expectedCache = scheduleCaches()
        let expected = schedule.run(
            x, caches: try #require(schedule.caches(expectedCache, over: x)), verify: nil,
            segment: { index, arguments in decodeSegment(index, of: schedule, model, arguments) },
            attention: attended)
        let cache = scheduleCaches()

        let output = schedule.decode(
            x, cache: cache, segments: segmentCache(schedule), of: model, attention: attended)

        expectEqual(try #require(output), expected)
        for layer in [0, 2] {
            let linear = try #require(cache[layer] as? MambaCache)
            let expectedLinear = try #require(expectedCache[layer] as? MambaCache)
            expectEqual(try #require(linear[0]), try #require(expectedLinear[0]))
            expectEqual(try #require(linear[1]), try #require(expectedLinear[1]))
        }
    }

    @Test func runOfADecodeStepCommitsTheStatesAndAttendsBetweenSegments() throws {
        let (segments, layers) = schedule()
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let attentionCache = KVCacheSimple()
        let cache = scheduleCaches(attention: attentionCache)
        let x = hiddenStates()
        let caches = try #require(schedule.caches(cache, over: x))
        let finalNorm = RMSNorm(dimensions: hiddenWidth)
        var steps: [CompiledDecodeSchedule.AttentionStep] = []

        let output = schedule.run(
            x, caches: caches, verify: nil,
            segment: { index, arguments in
                segments[index].body(
                    arguments[0], arguments: arguments, layers: layers, finalNorm: finalNorm,
                    verify: nil)
            },
            attention: { step in
                steps.append(step)
                return attended(step)
            })

        let hidden = linearOutput(x, of: layers[0])
        let normed = layers[1].inputLayerNorm(hidden)
        let step = try #require(steps.first)
        #expect(steps.count == 1)
        #expect(step.layer == 1)
        #expect((step.cache as? KVCacheSimple) === attentionCache)
        expectEqual(step.queries, normed + headOffset(Head.queries))
        expectEqual(step.keys, normed + headOffset(Head.keys))
        expectEqual(step.values, normed + headOffset(Head.values))
        let attention = attended(step)
        let gate = normed + headOffset(Head.gate)
        expectEqual(
            output,
            finalNorm(linearOutput(closed(hidden + attention * gate, by: layers[1]), of: layers[2]))
        )
        for layer in [0, 2] {
            let linear = try #require(cache[layer] as? MambaCache)
            expectEqual(try #require(linear[0]), convState(of: layer) + stateStep)
            expectEqual(try #require(linear[1]), recState(of: layer) + stateStep)
            #expect(linear.offset == GatedDeltaDecodeKernel.decodeTokens)
            #expect(linear.speculativeSlotCount == 0)
        }
    }

    @Test func runOfAVerifyKeepsTheSnapshotSlotsAndAdvancesByItsTokens() throws {
        let (segments, layers) = schedule()
        let schedule = CompiledDecodeSchedule(linearLayers: [true, false, true])
        let verify = try #require(CompiledVerifyShape(tokens: stepTokens))
        let cache = scheduleCaches()
        let x = hiddenStates()
        let caches = try #require(schedule.caches(cache, over: x))
        let finalNorm = RMSNorm(dimensions: hiddenWidth)

        let output = schedule.run(
            x, caches: caches, verify: verify,
            segment: { index, arguments in
                segments[index].body(
                    arguments[0], arguments: arguments, layers: layers, finalNorm: finalNorm,
                    verify: verify)
            },
            attention: attended)

        let hidden = linearOutput(x, of: layers[0])
        let normed = layers[1].inputLayerNorm(hidden)
        let attention =
            (normed + headOffset(Head.queries)) + (normed + headOffset(Head.keys))
        let gate = normed + headOffset(Head.gate)
        expectEqual(
            output, linearOutput(closed(hidden + attention * gate, by: layers[1]), of: layers[2]))
        for layer in [0, 2] {
            let linear = try #require(cache[layer] as? MambaCache)
            expectEqual(try #require(linear[0]), convState(of: layer) + stateStep)
            expectEqual(try #require(linear[1]), recState(of: layer) + stateStep)
            #expect(linear.offset == stepTokens)
            #expect(linear.speculativeSlotCount == stepTokens)
        }
    }
}
