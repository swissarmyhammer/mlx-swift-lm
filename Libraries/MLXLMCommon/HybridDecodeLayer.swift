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
    /// The states a verify keeps for a rollback; nil for a decode step.
    package var speculative: GatedDeltaSpeculativeStates?

    /// - Parameters:
    ///   - output: The output, shape `(batch, tokens, hiddenSize)`.
    ///   - convState: The conv state after the last token.
    ///   - recurrentState: The recurrent state after the last token.
    ///   - speculative: The states a verify keeps for a rollback, or nil.
    package init(
        output: MLXArray, convState: MLXArray, recurrentState: MLXArray,
        speculative: GatedDeltaSpeculativeStates? = nil
    ) {
        self.output = output
        self.convState = convState
        self.recurrentState = recurrentState
        self.speculative = speculative
    }
}

/// A decoder layer of a hybrid model — gated delta net layers between
/// full-attention layers — whose bodies a compiled decode schedule
/// (``CompiledDecodeSegment``) traces.
///
/// Each layer gives its norms and its blocks. The bodies around them — the
/// residual and the feed-forward block that close every layer, the gated
/// delta body with its states passed in and out, and the two halves of a
/// full-attention layer around its KV write — are shared by this extension.
package protocol HybridDecodeLayer: Module {
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
    ///   - checkpointAfter: For a verify, the token after which the layer
    ///     keeps its states for a rollback; nil for a decode step.
    /// - Returns: The mixer output and the states.
    func gatedDeltaMixer(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray, checkpointAfter: Int?
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
    /// The gated delta net layer with its states passed in and out, for a
    /// traced step.
    ///
    /// - Parameters:
    ///   - x: The layer input, shape `(batch, tokens, hiddenSize)`.
    ///   - convState: The conv state before the first token.
    ///   - recState: The recurrent state before the first token.
    ///   - checkpointAfter: For a verify, the token after which the layer
    ///     keeps its states for a rollback; nil for a decode step.
    /// - Returns: The layer output, the states after the last token, and the
    ///   states a verify keeps.
    package func linearLayerBody(
        x: MLXArray, convState: MLXArray, recState: MLXArray, checkpointAfter: Int? = nil
    ) -> GatedDeltaLayerStep {
        var step = gatedDeltaMixer(
            inputLayerNorm(x), convState: convState, recState: recState,
            checkpointAfter: checkpointAfter)
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
