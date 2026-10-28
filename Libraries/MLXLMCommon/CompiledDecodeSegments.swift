import Foundation
import MLX
import MLXNN

/// One static-shaped piece of a hybrid model's single-token decode graph.
///
/// Full-attention cache writes split segments because the cache grows at every
/// token and cannot be mutated inside an MLX compiled function. Linear-layer
/// state is passed explicitly through the compiled function instead.
package struct CompiledDecodeSegment: Sendable, Equatable {
    package var attentionPostLayer: Int?
    package var linearLayers: [Int]
    package var attentionPreLayer: Int?

    package init(
        attentionPostLayer: Int? = nil,
        linearLayers: [Int] = [],
        attentionPreLayer: Int? = nil
    ) {
        self.attentionPostLayer = attentionPostLayer
        self.linearLayers = linearLayers
        self.attentionPreLayer = attentionPreLayer
    }

    /// The state tensors each linear layer takes in and gives back: the conv
    /// state and the recurrent state.
    package static let statesPerLinearLayer = 2

    /// First linear-state input, after the hidden state and an optional
    /// `[attention, gate]` pair.
    package var stateInputOffset: Int { attentionPostLayer == nil ? 1 : 3 }

    /// First `[queries, gate, keys, values]` output after the hidden state and
    /// two updated state tensors per linear layer.
    package var attentionOutputOffset: Int {
        attentionOutputOffset(outputsPerLinearLayer: Self.statesPerLinearLayer)
    }

    /// First `[queries, gate, keys, values]` output after the hidden state and
    /// the outputs of each linear layer.
    ///
    /// - Parameter outputsPerLinearLayer: The tensors each linear layer gives
    ///   back: its two states, and the states a verify keeps for a rollback.
    /// - Returns: The index of the queries among the outputs of the segment.
    package func attentionOutputOffset(outputsPerLinearLayer: Int) -> Int {
        1 + outputsPerLinearLayer * linearLayers.count
    }

    /// Every decoder layer this segment runs, in order. The weights of these
    /// layers are the compile state of the segment's trace.
    package var layerIndices: [Int] {
        (attentionPostLayer.map { [$0] } ?? []) + linearLayers
            + (attentionPreLayer.map { [$0] } ?? [])
    }

    /// Build the maximal segments separated by full-attention cache writes.
    package static func schedule(linearLayers: [Bool]) -> [Self] {
        var segments: [Self] = []
        var current = Self()
        for (index, isLinear) in linearLayers.enumerated() {
            if isLinear {
                current.linearLayers.append(index)
            } else {
                current.attentionPreLayer = index
                segments.append(current)
                current = Self(attentionPostLayer: index)
            }
        }
        segments.append(current)
        return segments
    }
}

extension CompiledDecodeSegment {
    /// The places of the outputs of an attention head among the outputs of a
    /// segment, from ``attentionOutputOffset(outputsPerLinearLayer:)``.
    package enum AttentionOutput {
        /// The queries before rope.
        package static let queries = 0
        /// The output gate.
        package static let gate = 1
        /// The keys before rope.
        package static let keys = 2
        /// The values.
        package static let values = 3
        /// The outputs of an attention head.
        package static let count = 4
    }

    /// The places of the inputs of an attention tail among the arguments of
    /// a segment that opens with one.
    private enum AttentionInput {
        /// The output of the attention over the cache.
        static let attention = 1
        /// The output gate.
        static let gate = 2
    }

    /// Runs this segment of a decode step or of a speculative verify, as the
    /// body of its trace.
    ///
    /// Flat argument/result lists because `compile` takes `[MLXArray]`.
    /// In: `[x]`, then `[attention, gate]` when opening with a full-attention
    /// tail, then `[convState, recState]` per GDN layer. Out: `[x]`, then
    /// `[newConvState, newRecState]` per GDN layer (followed, for a verify,
    /// by the snapshot slots of
    /// ``GatedDeltaDecodeKernel/Snapshots/traceArrays``), then the
    /// ``AttentionOutput`` arrays when closing with a head.
    ///
    /// The segment that closes the schedule, the one with no attention head,
    /// ends a decode step with the final norm. A verify leaves the norm to
    /// its caller, which emits the rows before it.
    ///
    /// - Parameters:
    ///   - input: The hidden states the segment starts from: the first
    ///     argument, or for a model that embeds inside the trace, its
    ///     embedding.
    ///   - arguments: The flat argument list of the segment.
    ///   - layers: Every decoder layer of the model.
    ///   - finalNorm: The final norm of the model.
    ///   - verify: The shape of a verify, or nil for a decode step.
    /// - Returns: The flat result list of the segment.
    package func body<Layer: HybridDecodeLayer>(
        _ input: MLXArray, arguments: [MLXArray], layers: [Layer], finalNorm: RMSNorm,
        verify: CompiledVerifyShape?
    ) -> [MLXArray] {
        var hiddenStates = input
        if let post = attentionPostLayer {
            hiddenStates = layers[post].attentionPostBody(
                x: hiddenStates, attention: arguments[AttentionInput.attention],
                gate: arguments[AttentionInput.gate])
        }

        var states: [MLXArray] = []
        for (position, layerIndex) in linearLayers.enumerated() {
            let slot = stateInputOffset + Self.statesPerLinearLayer * position
            let step = layers[layerIndex].linearLayerBody(
                x: hiddenStates, convState: arguments[slot], recState: arguments[slot + 1],
                keepsSnapshots: verify != nil)
            hiddenStates = step.output
            states +=
                [step.convState, step.recurrentState]
                + Self.snapshotOutputs(step.snapshots, verify: verify)
        }

        if let pre = attentionPreLayer {
            let head = layers[pre].attentionPreBody(x: hiddenStates)
            // The next segment needs the attention layer's input for its residual.
            return [hiddenStates] + states + [head.queries, head.gate, head.keys, head.values]
        }

        if verify == nil {
            hiddenStates = finalNorm(hiddenStates)
        }
        return [hiddenStates] + states
    }

    /// The snapshot slots a GDN layer of a verify segment gives back, as
    /// trace outputs; none for a decode step.
    private static func snapshotOutputs(
        _ snapshots: GatedDeltaDecodeKernel.Snapshots?, verify: CompiledVerifyShape?
    ) -> [MLXArray] {
        guard verify != nil else { return [] }
        guard let snapshots else {
            preconditionFailure("a verify of two tokens or more keeps its snapshot slots")
        }
        return snapshots.traceArrays
    }
}

/// Lazily compiles and retains one `CompiledTrace` per decode segment.
///
/// Models create this with their schedule; compilation stays lazy because
/// weights load after initialization. Each segment declares the modules its
/// body reads, so the trace sees current weights rather than the ones it was
/// first traced with.
package final class CompiledDecodeSegmentCache<Owner: Module>: CompiledTraceInvalidating {

    /// Runs segment `index` over the flat argument list.
    package typealias Body = @Sendable (Owner, Int, [MLXArray]) -> [MLXArray]

    /// Returns the modules whose weights segment `index` reads: the layers in
    /// `layerIndices`, plus anything else the body touches such as the
    /// embedding or the final norm.
    package typealias StateProvider = @Sendable (Owner, Int) -> [Module]

    private let traces: [CompiledTrace<Owner>]

    package init(count: Int, state: @escaping StateProvider, body: @escaping Body) {
        precondition(count > 0, "compiled decode requires at least one segment")
        self.traces = (0 ..< count).map { index in
            CompiledTrace<Owner>(
                state: { state($0, index) },
                body: { owner, arguments in body(owner, index, arguments) })
        }
    }

    package var compiledCount: Int {
        traces.filter(\.isCompiled).count
    }

    package func callAsFunction(
        _ owner: Owner, at index: Int, _ arguments: [MLXArray]
    ) -> [MLXArray] {
        traces[index](owner, arguments)
    }

    /// Drops every segment trace. See `CompiledTrace.invalidate()`.
    package func invalidate() {
        traces.forEach { $0.invalidate() }
    }
}

/// The static shape of a speculative verify step, whose gated delta net
/// layers keep the snapshot slots of its tokens for a rollback: its tokens.
///
/// Each shape traces a decode schedule of its own. The tokens fix the shapes
/// of the trace, and of the snapshot slots each GDN layer gives back.
package struct CompiledVerifyShape: Hashable, Sendable {

    /// The fewest tokens of a verify: the bonus token and one draft.
    package static let fewestTokens = 2

    /// The verify lengths a compiled schedule runs: up to the most tokens
    /// ``GatedDeltaDecodeKernel`` holds in one run.
    package static let tokenRange = fewestTokens ... GatedDeltaDecodeKernel.mostTokens

    /// The tokens of the verify.
    package let tokens: Int

    /// The shape of a verify, or nil when its tokens are outside
    /// ``tokenRange``, which a compiled schedule does not run.
    ///
    /// - Parameter tokens: The tokens of the verify.
    package init?(tokens: Int) {
        guard Self.tokenRange.contains(tokens) else { return nil }
        self.tokens = tokens
    }
}

/// Lazily creates and retains one ``CompiledDecodeSegmentCache`` for each key:
/// one schedule of decode segments for each static shape of a step, such as
/// each ``CompiledVerifyShape`` of a speculative verify.
///
/// It is one ``CompiledTraceInvalidating`` value, so
/// `Module.invalidateCompiledTraces()` reaches the traces of every key. A
/// dictionary of caches stored on a module would hide them from it.
package final class KeyedCompiledDecodeSegmentCaches<Owner: Module, Key: Hashable & Sendable>:
    CompiledTraceInvalidating
{

    /// Runs segment `index` of the schedule of `key` over the flat argument
    /// list.
    package typealias Body = @Sendable (Owner, Key, Int, [MLXArray]) -> [MLXArray]

    private let count: Int
    private let state: CompiledDecodeSegmentCache<Owner>.StateProvider
    private let body: Body

    // The caches of the keys a step has run, made on the first step of each
    // key. The lock keeps two concurrent first steps from making rival caches.
    private let lock = NSLock()
    private var caches: [Key: CompiledDecodeSegmentCache<Owner>] = [:]

    /// - Parameters:
    ///   - count: The segments of each schedule.
    ///   - state: The modules whose weights segment `index` reads, the same
    ///     for each key.
    ///   - body: Runs a segment of the schedule of a key.
    package init(
        count: Int, state: @escaping CompiledDecodeSegmentCache<Owner>.StateProvider,
        body: @escaping Body
    ) {
        self.count = count
        self.state = state
        self.body = body
    }

    /// The compiled segment traces, over every key.
    package var compiledCount: Int {
        lock.withLock { caches.values.map(\.compiledCount).reduce(0, +) }
    }

    /// Runs segment `index` of the schedule of `key`, compiling it on first
    /// use.
    ///
    /// - Parameters:
    ///   - owner: The module whose weights the segment reads.
    ///   - key: The static shape of the step.
    ///   - index: The segment.
    ///   - arguments: The flat argument list of the segment.
    /// - Returns: The flat result list of the segment.
    package func callAsFunction(
        _ owner: Owner, key: Key, at index: Int, _ arguments: [MLXArray]
    ) -> [MLXArray] {
        cache(for: key)(owner, at: index, arguments)
    }

    /// Drops every segment trace of every key. See `CompiledTrace.invalidate()`.
    package func invalidate() {
        lock.withLock {
            for cache in caches.values {
                cache.invalidate()
            }
        }
    }

    /// The cache of `key`, made on its first use.
    private func cache(for key: Key) -> CompiledDecodeSegmentCache<Owner> {
        lock.withLock {
            if let cache = caches[key] {
                return cache
            }
            let body = self.body
            let cache = CompiledDecodeSegmentCache<Owner>(count: count, state: state) {
                owner, index, arguments in
                body(owner, key, index, arguments)
            }
            caches[key] = cache
            return cache
        }
    }
}
