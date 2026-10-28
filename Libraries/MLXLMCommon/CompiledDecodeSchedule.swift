import MLX
import MLXNN

/// The compiled decode schedule of a hybrid model: its ``CompiledDecodeSegment``s, and the
/// runner that steps the caches of the model through them.
///
/// The runner gathers the caches of a step, runs each segment over `[carry]`, the pending
/// `[attention, gate]` of the previous segment and the states of its gated delta net layers,
/// writes the states each layer gives back into its cache, and runs the attention of each
/// full-attention layer over its cache between two segments.
///
/// Each model passes in what is its own: how it runs one segment (the compiled trace of a
/// decode step, or of a verify shape) and the attention step of a full-attention layer (the
/// rope of the model, the KV write and the attention over the grown cache).
package struct CompiledDecodeSchedule: Sendable {

    /// The segments of a step, in order.
    package let segments: [CompiledDecodeSegment]

    /// The first gated delta net layer, whose cache gives the SSM mask of a step, or nil for
    /// a model with none.
    package let firstLinearLayer: Int?

    /// The first full-attention layer, whose cache gives the attention mask of a step, or nil
    /// for a model with none.
    package let firstAttentionLayer: Int?

    /// The decoder layers the schedule runs, so the cache list of a step holds one cache
    /// for each.
    private let layerCount: Int

    /// The schedule of a model whose decoder layers are `linearLayers`.
    ///
    /// - Parameter linearLayers: For each decoder layer, true for a gated delta net layer and
    ///   false for a full-attention layer.
    package init(linearLayers: [Bool]) {
        self.segments = CompiledDecodeSegment.schedule(linearLayers: linearLayers)
        self.firstLinearLayer = linearLayers.firstIndex(of: true)
        self.firstAttentionLayer = linearLayers.firstIndex(of: false)
        self.layerCount = linearLayers.count
    }

    /// The schedule of a model whose decoder layers are `layers`: a layer that holds a gated
    /// delta net is a linear layer, any other a full-attention layer.
    ///
    /// - Parameter layers: The decoder layers of the model, in order.
    package init<Layer: HybridDecodeLayer>(layers: [Layer]) {
        self.init(linearLayers: layers.map { $0.linearAttn != nil })
    }
}

extension CompiledDecodeSchedule {

    /// The cache of a gated delta net layer, with the conv and recurrent states the layer
    /// reads.
    fileprivate struct LinearCache {
        /// The cache the layer writes its states back into.
        let cache: MambaCache
        /// The conv state, then the recurrent state, before the step.
        let states: [MLXArray]
    }

    /// The full-attention layer a segment closes with, and its cache.
    fileprivate struct AttentionCache {
        /// The decoder layer.
        let layer: Int
        /// The cache of the layer.
        let cache: KVCache
    }

    /// The caches a step reads and writes, segment by segment, from
    /// ``CompiledDecodeSchedule/caches(_:over:)``.
    package struct Caches {
        /// For each segment, the cache of each of its gated delta net layers.
        fileprivate let linear: [[LinearCache]]
        /// For each segment, the attention layer it closes with, or nil for the last one.
        fileprivate let attention: [AttentionCache?]
    }

    /// The attention a full-attention layer runs over its cache, between the segment that
    /// closes with its head and the segment that opens with its tail.
    package struct AttentionStep {
        /// The decoder layer.
        package let layer: Int
        /// The cache of the layer, which the step writes the keys and values into.
        package let cache: KVCache
        /// The queries before rope.
        package let queries: MLXArray
        /// The keys before rope.
        package let keys: MLXArray
        /// The values.
        package let values: MLXArray
    }

    /// The caches of a step over `input`, or nil when the step takes the general path: the
    /// cache list does not hold one cache for each layer, the step needs an SSM mask, a gated
    /// delta net layer has no state yet (a first token, for which the general path builds the
    /// zero states), or a full-attention layer has no cache or a cache with a route of its
    /// own.
    ///
    /// Cache kinds can change mid-generation (`maybeQuantizeKVCache` swaps array entries), so
    /// a model asks again on every step.
    ///
    /// - Parameters:
    ///   - cache: The cache of each decoder layer.
    ///   - input: The first argument of the step, whose axis 1 holds its tokens.
    /// - Returns: The caches of the step, or nil.
    package func caches(_ cache: [KVCache?], over input: MLXArray) -> Caches? {
        guard cache.count == layerCount else { return nil }
        let attentionLayersFit = segments.compactMap(\.attentionPreLayer).allSatisfy {
            cache[$0].map(usesPlainAttentionCacheRoute) ?? false
        }
        let linear = segments.compactMap { Self.linearCaches(of: $0, cache) }
        guard attentionLayersFit, linear.count == segments.count,
            ssmMask(input, cache) == nil
        else { return nil }

        let attention = segments.map { segment in
            segment.attentionPreLayer.flatMap { layer in
                cache[layer].map { AttentionCache(layer: layer, cache: $0) }
            }
        }
        return Caches(linear: linear, attention: attention)
    }

    /// The SSM mask of a step over `input`, from the cache of ``firstLinearLayer``, or nil
    /// when the step needs none.
    ///
    /// - Parameters:
    ///   - input: The first argument of the step, whose axis 1 holds its tokens.
    ///   - cache: The cache of each decoder layer.
    /// - Returns: The mask, or nil.
    private func ssmMask(_ input: MLXArray, _ cache: [KVCache?]) -> MLXArray? {
        firstLinearLayer.flatMap { createSSMMask(h: input, cache: cache[$0] as? MambaCache) }
    }

    /// The attention mask of a step over `input`, the same for each full-attention layer:
    /// the mask the cache of ``firstAttentionLayer`` makes, or `.none` when that layer has no
    /// cache or the model has no such layer.
    ///
    /// - Parameters:
    ///   - input: The first argument of the step, whose axis 1 holds its tokens.
    ///   - cache: The cache of each decoder layer.
    /// - Returns: The mask mode of the step.
    package func attentionMask(
        _ input: MLXArray, _ cache: [KVCache?]
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        firstAttentionLayer.flatMap { cache[$0] }.map { createAttentionMask(h: input, cache: $0) }
            ?? .none
    }

    /// One decode step through the compiled segments in `segments`, or nil when the step
    /// takes the general path: ``caches(_:over:)`` refuses the caches, or the step needs an
    /// attention mask.
    ///
    /// - Parameters:
    ///   - input: The first argument of the first segment: the token ids for a model that
    ///     embeds inside its trace, else the embedded tokens.
    ///   - cache: The cache of each decoder layer.
    ///   - segments: The compiled segment traces of a decode step.
    ///   - owner: The model whose weights the traces read.
    ///   - attention: Runs the attention of a full-attention layer over its cache, and gives
    ///     back its output.
    /// - Returns: The first output of the last segment, or nil.
    package func decode<Owner: Module>(
        _ input: MLXArray, cache: [KVCache?], segments: CompiledDecodeSegmentCache<Owner>,
        of owner: Owner, attention: (AttentionStep) -> MLXArray
    ) -> MLXArray? {
        guard let caches = caches(cache, over: input),
            case .none = attentionMask(input, cache)
        else { return nil }
        return run(
            input, caches: caches, verify: nil,
            segment: { index, arguments in segments(owner, at: index, arguments) },
            attention: attention)
    }

    /// The caches of the gated delta net layers of `segment` with their states, or nil when
    /// one of them has no state yet.
    private static func linearCaches(
        of segment: CompiledDecodeSegment, _ cache: [KVCache?]
    ) -> [LinearCache]? {
        let caches = segment.linearLayers.compactMap { index -> LinearCache? in
            guard let mambaCache = cache[index] as? MambaCache,
                let convState = mambaCache[0], let recurrentState = mambaCache[1]
            else { return nil }
            return LinearCache(cache: mambaCache, states: [convState, recurrentState])
        }
        return caches.count == segment.linearLayers.count ? caches : nil
    }

    /// Runs the segments of a step in order: each gated delta net layer writes its states
    /// into its cache, and each full-attention layer runs `attention` between two segments.
    ///
    /// - Parameters:
    ///   - input: The first argument of the first segment: the token ids for a model that
    ///     embeds inside its trace, else the embedded tokens.
    ///   - caches: The caches of the step, from ``caches(_:over:)``.
    ///   - verify: The shape of a verify, or nil for a decode step.
    ///   - runSegment: Runs segment `index` over its flat argument list.
    ///   - attention: Runs the attention of a full-attention layer over its cache, and gives
    ///     back its output.
    /// - Returns: The first output of the last segment.
    package func run(
        _ input: MLXArray, caches: Caches, verify: CompiledVerifyShape?,
        segment runSegment: (Int, [MLXArray]) -> [MLXArray],
        attention: (AttentionStep) -> MLXArray
    ) -> MLXArray {
        let outputsPerLinearLayer =
            CompiledDecodeSegment.statesPerLinearLayer
            + (verify == nil ? 0 : GatedDeltaDecodeKernel.Snapshots.traceArrayCount)
        var carry = input
        var pendingAttention: [MLXArray] = []

        for (index, segment) in segments.enumerated() {
            let linear = caches.linear[index]
            let outputs = runSegment(index, [carry] + pendingAttention + linear.flatMap(\.states))
            carry = outputs[0]
            for (position, entry) in linear.enumerated() {
                let first = 1 + outputsPerLinearLayer * position
                Self.commit(
                    Array(outputs[first ..< first + outputsPerLinearLayer]), to: entry.cache,
                    verify: verify)
            }

            pendingAttention = []
            if let head = caches.attention[index] {
                let offset = segment.attentionOutputOffset(
                    outputsPerLinearLayer: outputsPerLinearLayer)
                let step = AttentionStep(
                    layer: head.layer, cache: head.cache,
                    queries: outputs[offset + CompiledDecodeSegment.AttentionOutput.queries],
                    keys: outputs[offset + CompiledDecodeSegment.AttentionOutput.keys],
                    values: outputs[offset + CompiledDecodeSegment.AttentionOutput.values])
                pendingAttention = [
                    attention(step), outputs[offset + CompiledDecodeSegment.AttentionOutput.gate],
                ]
            }
        }

        return carry
    }

    /// Writes what a compiled gated delta net layer gave back into its cache: its conv and
    /// recurrent states, then, for a verify, the states it keeps for a rollback, as the
    /// general path of the layer does. The cache advances by the tokens of the step: one for
    /// a decode step, the tokens of the shape for a verify.
    private static func commit(
        _ outputs: [MLXArray], to cache: MambaCache, verify: CompiledVerifyShape?
    ) {
        cache[0] = outputs[0]
        cache[1] = outputs[1]
        if verify != nil {
            cache.saveSpeculativeSnapshots(
                GatedDeltaDecodeKernel.Snapshots(
                    traceArrays: Array(
                        outputs.dropFirst(CompiledDecodeSegment.statesPerLinearLayer))))
        }
        cache.advance(verify?.tokens ?? GatedDeltaDecodeKernel.decodeTokens)
    }
}
