import MLX
import MLXNN

/// What the gated delta net layer body gives back on a traced step: the
/// layer output, the states after the last token, and the states a
/// speculative verify keeps for a rollback.
package struct GatedDeltaLayerStep {
    /// The output, shape `(batch, tokens, hiddenSize)`.
    package var output: MLXArray
    /// The conv state after the last token.
    package var convState: MLXArray
    /// The recurrent state after the last token.
    package var recurrentState: MLXArray
    /// The snapshot slots a verify keeps for a rollback; nil for a decode step.
    package var snapshots: GatedDeltaDecodeKernel.Snapshots?

    /// - Parameters:
    ///   - output: The output, shape `(batch, tokens, hiddenSize)`.
    ///   - convState: The conv state after the last token.
    ///   - recurrentState: The recurrent state after the last token.
    ///   - snapshots: The snapshot slots a verify keeps for a rollback, or nil.
    package init(
        output: MLXArray, convState: MLXArray, recurrentState: MLXArray,
        snapshots: GatedDeltaDecodeKernel.Snapshots? = nil
    ) {
        self.output = output
        self.convState = convState
        self.recurrentState = recurrentState
        self.snapshots = snapshots
    }
}

/// A decoder layer of a hybrid model — gated delta net layers between
/// full-attention layers — whose bodies a compiled decode schedule
/// (``CompiledDecodeSegment``) traces.
///
/// Each layer gives its norms, its blocks, and its two optional mixer modules
/// (a linear layer holds the gated delta net, a full-attention layer the
/// attention). The bodies around them — the residual and the feed-forward
/// block that close every layer, the gated delta body with its states passed
/// in and out, and the two halves of a full-attention layer around its KV
/// write — and the accessors that unwrap the mixer a layer kind must hold are
/// shared by this extension.
package protocol HybridDecodeLayer: Module {
    /// The module type of the gated delta net of a linear layer.
    associatedtype LinearMixer: Module
    /// The module type of the attention of a full-attention layer.
    associatedtype AttentionMixer: Module

    /// The gated delta net; non-nil exactly for a linear layer.
    var linearAttn: LinearMixer? { get }
    /// The attention; non-nil exactly for a full-attention layer.
    var selfAttn: AttentionMixer? { get }

    /// The RMSNorm before the mixer, the gated delta net or the attention.
    var inputLayerNorm: RMSNorm { get }
    /// The RMSNorm before the feed-forward block.
    var postAttentionLayerNorm: RMSNorm { get }

    /// The gated delta net of a linear layer, with its states passed in and
    /// out.
    ///
    /// - Parameters:
    ///   - x: The normed layer input, shape `(batch, tokens, hiddenSize)`.
    ///   - convState: The conv state before the first token.
    ///   - recState: The recurrent state before the first token.
    ///   - keepsSnapshots: True for a verify, whose layer keeps the states
    ///     after each token for a rollback; false for a decode step.
    /// - Returns: The mixer output and the states.
    func gatedDeltaMixer(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray, keepsSnapshots: Bool
    ) -> GatedDeltaLayerStep

    /// The projections of a full-attention layer before rope and the KV
    /// write.
    ///
    /// - Parameter x: The normed layer input.
    /// - Returns: The queries, the output gate, the keys and the values.
    func projectAttention(_ x: MLXArray) -> (
        queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray
    )

    /// The gated merge of the attention heads and the output projection of a
    /// full-attention layer.
    ///
    /// - Parameters:
    ///   - attention: The output of the attention over the cache.
    ///   - gate: The output gate from ``projectAttention(_:)``.
    /// - Returns: The mixer output, shape `(batch, tokens, hiddenSize)`.
    func mergeAttention(_ attention: MLXArray, gate: MLXArray) -> MLXArray

    /// The feed-forward block, in the form a trace can hold.
    func feedForward(_ x: MLXArray) -> MLXArray
}

extension HybridDecodeLayer {
    /// The gated delta net of a linear layer; only a linear layer may ask.
    package var gatedDeltaNet: LinearMixer {
        guard let linearAttn else {
            preconditionFailure("only a linear layer has a gated delta net")
        }
        return linearAttn
    }

    /// The attention of a full-attention layer; only a full-attention layer
    /// may ask.
    package var fullAttention: AttentionMixer {
        guard let selfAttn else {
            preconditionFailure("only a full-attention layer has an attention")
        }
        return selfAttn
    }

    /// The gated delta net layer with its states passed in and out, for a
    /// traced step.
    ///
    /// - Parameters:
    ///   - x: The layer input, shape `(batch, tokens, hiddenSize)`.
    ///   - convState: The conv state before the first token.
    ///   - recState: The recurrent state before the first token.
    ///   - keepsSnapshots: True for a verify, whose layer keeps the states
    ///     after each token for a rollback; false for a decode step.
    /// - Returns: The layer output, the states after the last token, and the
    ///   snapshot slots a verify keeps.
    package func linearLayerBody(
        x: MLXArray, convState: MLXArray, recState: MLXArray, keepsSnapshots: Bool = false
    ) -> GatedDeltaLayerStep {
        var step = gatedDeltaMixer(
            inputLayerNorm(x), convState: convState, recState: recState,
            keepsSnapshots: keepsSnapshots)
        step.output = residualFeedForward(x + step.output)
        return step
    }

    /// The head of a full-attention layer: its projections before rope and
    /// the KV write.
    ///
    /// - Parameter x: The layer input.
    /// - Returns: The queries, the output gate, the keys and the values.
    package func attentionPreBody(x: MLXArray) -> (
        queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        projectAttention(inputLayerNorm(x))
    }

    /// The tail of a full-attention layer after the attention over the cache.
    ///
    /// - Parameters:
    ///   - x: The layer input, the residual branch around the attention.
    ///   - attention: The output of the attention over the cache.
    ///   - gate: The output gate from ``attentionPreBody(x:)``.
    /// - Returns: The layer output.
    package func attentionPostBody(x: MLXArray, attention: MLXArray, gate: MLXArray) -> MLXArray {
        residualFeedForward(x + mergeAttention(attention, gate: gate))
    }

    /// The residual feed-forward block that closes each layer.
    private func residualFeedForward(_ h: MLXArray) -> MLXArray {
        h + feedForward(postAttentionLayerNorm(h))
    }
}

/// The places of the hidden states and the two states of a gated delta net
/// layer among the arguments and among the results of the compiled trace that
/// runs ``HybridDecodeLayer/linearLayerBody(x:convState:recState:keepsSnapshots:)``.
package enum LinearTraceSlot {
    /// The layer input, then the layer output.
    package static let hidden = 0
    /// The conv state.
    package static let convState = 1
    /// The recurrent state.
    package static let recurrentState = 2
}

/// The places of the argument and of the results of the compiled trace that
/// runs ``HybridDecodeLayer/attentionPreBody(x:)``, a full-attention layer up
/// to the KV write.
package enum AttentionPreTraceSlot {
    /// The layer input among the arguments.
    package static let hidden = 0
    /// The queries among the results.
    package static let queries = 0
    /// The output gate among the results.
    package static let gate = 1
    /// The keys among the results.
    package static let keys = 2
    /// The values among the results.
    package static let values = 3
}

/// The places of the arguments and of the result of the compiled trace that
/// runs ``HybridDecodeLayer/attentionPostBody(x:attention:gate:)``, a
/// full-attention layer after the KV write.
package enum AttentionPostTraceSlot {
    /// The layer input among the arguments, then the layer output among the
    /// results.
    package static let hidden = 0
    /// The attention over the cache.
    package static let attention = 1
    /// The output gate from the trace before the KV write.
    package static let gate = 2
}

/// A ``HybridDecodeLayer`` whose projections a load fuses, so the compiled
/// traces that run the layer read the fused projections too.
///
/// A fused projection lives beside the layer's checkpoint modules, and it
/// drops each trace that reads it when its fusion turns off. So the compile
/// state of a trace that runs this layer is the layer and its fused
/// projections, which ``layerTraceState(of:)`` gives.
package protocol FusedProjectionDecodeLayer: HybridDecodeLayer {
    /// The fused projections of this layer as compile state of `trace`. Each
    /// one registers `trace` so that it drops it when its fusion turns off.
    ///
    /// - Parameter trace: The trace that runs this layer.
    /// - Returns: The fused projections the layer holds.
    func fusedProjectionTraceState(of trace: any CompiledTraceInvalidating) -> [Module]
}

extension FusedProjectionDecodeLayer {
    /// The compile state of `trace`, which runs this layer: the layer and its
    /// fused projections, which drop `trace` when their fusion turns off.
    ///
    /// - Parameter trace: The trace that runs this layer.
    /// - Returns: The modules whose weights the trace reads for this layer.
    package func layerTraceState(of trace: any CompiledTraceInvalidating) -> [Module] {
        [self] + fusedProjectionTraceState(of: trace)
    }
}
