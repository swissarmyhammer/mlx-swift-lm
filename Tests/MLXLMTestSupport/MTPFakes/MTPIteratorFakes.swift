// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLMCommon
import MLXNN

/// A model-state key that `MockMainModel` carries from its input state to its output state,
/// so a test can see which state the iterator handed to each forward call.
public let preservedStateKey = LMOutput.Key<Int>("tests.mtp.preservedState")

/// Stateful drafter mock (the Qwen MTP shape): owns a private cache and
/// proposes tokens only. Exercises the iterator's stateful path, which is
/// greedy-only -- without the proposal distribution nothing can preserve
/// the target's distribution on rejection.
public final class StatefulMockDrafter: Module, StatefulMTPDrafterModel {
    public private(set) var draftBlockCallCount = 0
    public var draftedTokenValue: Int32
    public let requiresGreedySampling: Bool
    public let compatibilityError: MTPDrafterCompatibilityError?
    /// Per-call record of what the iterator handed `draftBlock`: the
    /// sequence-axis span of each sharedKV entry, and the query offset.
    /// Lets tests assert the state the drafter conditions on, not just the
    /// tokens that come out of it.
    public private(set) var receivedSharedKVSpans: [[String: Int]] = []
    public private(set) var receivedQueryOffsets: [Int] = []
    public private(set) var receivedPositionDeltaValues: [Int?] = []
    public private(set) var receivedCacheOffsets: [Int?] = []

    public init(
        draftedTokenValue: Int32 = SyntheticMTPValues.draftedToken,
        requiresGreedySampling: Bool = false,
        compatibilityError: MTPDrafterCompatibilityError? = nil
    ) {
        self.draftedTokenValue = draftedTokenValue
        self.requiresGreedySampling = requiresGreedySampling
        self.compatibilityError = compatibilityError
        super.init()
    }

    public func validateCompatibility(with _: any LanguageModel) throws {
        if let compatibilityError {
            throw compatibilityError
        }
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [CountingKVCache()])
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        receivedSharedKVSpans.append(
            sharedKV.mapValues { $0.0.dim(SyntheticMTPValues.sequenceAxis) })
        receivedQueryOffsets.append(queryOffset)
        receivedPositionDeltaValues.append(positionDeltas?.item(Int.self))
        let mutableCache = state.cache.first as? MutableOffsetKVCache
        receivedCacheOffsets.append(mutableCache?.offset)
        mutableCache?.offset += blockSize - 1
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

/// Records `draftBlock(...)` invocations and returns a fixed token pattern
/// so the iterator's draft/verify/accept flow can be exercised without a
/// real drafter. Exposes the proposal distribution, so the iterator takes
/// the distribution-preserving verify path.
public final class MockDrafter: Module, MTPDistributionDrafterModel {
    public private(set) var draftBlockCallCount = 0
    public private(set) var draftBlockWithLogitsCallCount = 0
    /// Calls through the token-only entry point of the base protocol.
    public private(set) var legacyDraftBlockCallCount = 0
    public var draftedTokenValue: Int32
    public let requiresGreedySampling: Bool
    /// Per-call record of what the iterator handed the drafter: the
    /// sequence-axis span of each sharedKV entry, the query offset, and
    /// the position delta. Lets tests assert the state the drafter
    /// conditions on, not just the tokens that come out of it.
    public private(set) var receivedSharedKVSpans: [[String: Int]] = []
    public private(set) var receivedQueryOffsets: [Int] = []
    public private(set) var receivedPositionDeltaValues: [Int?] = []

    public init(
        draftedTokenValue: Int32 = SyntheticMTPValues.draftedToken,
        requiresGreedySampling: Bool = false
    ) {
        self.draftedTokenValue = draftedTokenValue
        self.requiresGreedySampling = requiresGreedySampling
        super.init()
    }

    public func validateCompatibility(with _: any LanguageModel) throws {}

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        legacyDraftBlockCallCount += 1
        return draftBlockWithLogits(
            target: target, lastToken: lastToken, lastHidden: lastHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas, queryOffset: queryOffset,
            blockSize: blockSize, processor: nil, sampler: sampler
        ).tokens
    }

    public func draftBlockWithLogits(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        processor: (any LogitProcessor)?,
        sampler: any LogitSampler
    ) -> MTPDraftBlockOutput {
        draftBlockCallCount += 1
        draftBlockWithLogitsCallCount += 1
        receivedSharedKVSpans.append(
            sharedKV.mapValues { $0.0.dim(SyntheticMTPValues.sequenceAxis) })
        receivedQueryOffsets.append(queryOffset)
        receivedPositionDeltaValues.append(positionDeltas?.item(Int.self))
        let batch = lastToken.dim(0)
        let vocab = MockMainModel.vocabularySize
        var processor = processor
        var tokens = [MLXArray]()
        var logitsRows = [MLXArray]()
        for _ in 0 ..< (blockSize - 1) {
            var values = [Float](repeating: 0, count: batch * vocab)
            for row in 0 ..< batch {
                values[row * vocab + Int(draftedTokenValue)] = SyntheticMTPValues.selectedLogit
            }
            var logits = MLXArray(values, [batch, vocab])
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            tokens.append(token.expandedDimensions(axis: 1))
            logitsRows.append(logits.expandedDimensions(axis: 1))
        }
        return MTPDraftBlockOutput(
            tokens: concatenated(tokens, axis: 1),
            processedLogits: concatenated(logitsRows, axis: 1)
        )
    }
}

/// Minimal `LanguageModel` mock that emits MTP state on every call when
/// `mtpEmitFlagKey` is true. Returns shaped logits and a trimmable KV cache.
public final class MockMainModel: Module, LanguageModel, KVCacheDimensionProvider {
    /// The number of tokens of the mock vocabulary.
    public static let vocabularySize = 20

    public var kvHeads: [Int] { [1] }
    /// Sequence of token values returned in increasing position order. Length
    /// must cover all positions the iterator will sample across the run.
    public var nextLogitTokens: [Int32]
    public var perPositionIndex: Int = 0
    /// If `true`, omit the MTP state keys from the returned `LMOutput` so the
    /// iterator's passthrough fallback is exercised.
    public var omitDrafterState: Bool = false
    public var emitEmptySharedKVState: Bool = false
    public var maxEmittedSharedKVSpan: Int?
    public var returnPreservedStateWhenOmittingDrafterState: Bool = false
    public var prepareLogitsStateValue: Int?
    public var emittedPositionDelta: Int?

    public private(set) var callCount: Int = 0
    public private(set) var lastIncomingEmitFlag: Bool? = nil
    public private(set) var incomingPreservedStateValues: [Int?] = []
    /// Sequence-axis span of each emitted sharedKV snapshot, in emit order.
    /// Tests assert this against the mock cache offset to pin the mock's
    /// span fidelity (see the emit block below).
    public private(set) var emittedSharedKVSpans: [Int] = []

    public init(nextLogitTokens: [Int32]) {
        self.nextLogitTokens = nextLogitTokens
        super.init()
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    )
        throws -> PrepareResult
    {
        if let prepareLogitsStateValue {
            var state = LMOutput.State()
            state[preservedStateKey] = prepareLogitsStateValue
            if let first = cache.first as? MutableOffsetKVCache {
                first.offset += input.text.cacheSequenceLength
            }
            let logits = makeLogits(positions: input.text.tokens.dim(-1))
            return .logits(LMOutput(logits: logits, state: state))
        }
        // Return `.tokens(...)`; the iterator's `prepare` will follow up with
        // a one-position forward call that primes drafter state.
        return .tokens(input.text)
    }

    /// Returns deterministic one-hot logits at each position so a `softmax/
    /// argmax` sampler picks the planned token sequence.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        return makeLogits(positions: positions)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        lastIncomingEmitFlag = state?[mtpEmitFlagKey]
        incomingPreservedStateValues.append(state?[preservedStateKey])

        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)

        // Update the mock cache to reflect that `positions` tokens were seen.
        if let first = cache?.first {
            if let mutable = first as? MutableOffsetKVCache {
                mutable.offset += positions
            } else {
                let keys = MLXArray.zeros([1, 1, positions, SyntheticMTPValues.rowWidth])
                _ = first.update(keys: keys, values: keys)
            }
        }

        if !omitDrafterState, state?[mtpEmitFlagKey] ?? false {
            // Mirror the real emit hook's spans: sharedKV is captured from
            // what the attention layers consumed, so its sequence axis covers
            // the cache's full logical offset (everything processed so far);
            // lastHidden is the forward's own output, so it covers only the
            // current chunk. Span-accurate sharedKV emission is what lets a
            // test distinguish a trimmed snapshot from a stale one — with a
            // fixed-span mock those are indistinguishable.
            let kvOffset = cache?.first?.offset ?? positions
            let kvSpan = maxEmittedSharedKVSpan.map { Swift.min(kvOffset, $0) } ?? kvOffset
            emittedSharedKVSpans.append(kvSpan)
            var out = LMOutput.State()
            out[preservedStateKey] = state?[preservedStateKey]
            out[mtpLastHiddenStatesKey] = MLXArray.zeros([
                1, positions, SyntheticMTPValues.rowWidth,
            ])
            if emitEmptySharedKVState {
                out[mtpSharedKVStatesKey] = [:]
                out[mtpSharedKVSourceIndicesKey] = [:]
            } else {
                // One cache entry stands in for both layer types here.
                out[mtpSharedKVSourceIndicesKey] = [
                    "full_attention": 0, "sliding_attention": 0,
                ]
                out[mtpSharedKVStatesKey] = [
                    "full_attention": (
                        MLXArray.zeros([1, 1, kvSpan, SyntheticMTPValues.rowWidth]),
                        MLXArray.zeros([1, 1, kvSpan, SyntheticMTPValues.rowWidth])
                    ),
                    "sliding_attention": (
                        MLXArray.zeros([1, 1, kvSpan, SyntheticMTPValues.rowWidth]),
                        MLXArray.zeros([1, 1, kvSpan, SyntheticMTPValues.rowWidth])
                    ),
                ]
            }
            out[mtpSharedKVOffsetsKey] = ["full_attention": kvOffset]
            if let emittedPositionDelta {
                out[mtpPositionDeltasKey] = MLXArray(emittedPositionDelta)
            }
            return LMOutput(logits: logits, state: out)
        }
        if returnPreservedStateWhenOmittingDrafterState {
            var out = LMOutput.State()
            out[preservedStateKey] = state?[preservedStateKey]
            return LMOutput(logits: logits, state: out)
        }
        return LMOutput(logits: logits)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [CountingKVCache()]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        // [B=1, positions, vocab=20]. One-hot at the planned token for each
        // position, taken from `nextLogitTokens[perPositionIndex...]`.
        let vocab = MockMainModel.vocabularySize
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let tokIdx = perPositionIndex + i
            let tok = tokIdx < nextLogitTokens.count ? Int(nextLogitTokens[tokIdx]) : 0
            data[i * vocab + tok] = SyntheticMTPValues.selectedLogit
        }
        perPositionIndex += positions
        return MLXArray(data, [1, positions, vocab])
    }
}

/// A `KVCache` whose offset a mock model moves directly.
public protocol MutableOffsetKVCache: KVCache, AnyObject {
    var offset: Int { get set }
}

/// Minimal `KVCache` that satisfies the protocol's trimmable interface; the
/// mock model adjusts `offset` directly. Inherits the default
/// `ropeOffset = .scalar(offset)` from the `KVCache` protocol extension.
///
/// `wrapAt` is opt-in and defaults to nil (unbounded, always trimmable), which
/// is the behavior every pre-existing test relies on. Setting it mirrors
/// `RotatingKVCache` through the predictive trimmability query, so a mock stream
/// can cross the window the same way a real sliding-window cache does.
public class CountingKVCache: MutableOffsetKVCache {
    public var offset: Int = 0
    public let wrapAt: Int?
    public init(wrapAt: Int? = nil) {
        self.wrapAt = wrapAt
    }
    public var maxSize: Int? { wrapAt }
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }
    public var state: [MLXArray] {
        get { [] }
        set {}
    }
    public var metaState: [String] {
        get { [] }
        set {}
    }
    public var isTrimmable: Bool {
        isTrimmable(after: 0)
    }
    public func isTrimmable(after positions: Int) -> Bool {
        guard let wrapAt else { return true }
        return offset + positions < wrapAt
    }
    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard isTrimmable else { return 0 }
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    public func copy() -> any KVCache {
        let c = CountingKVCache(wrapAt: wrapAt)
        c.offset = offset
        return c
    }
    public func innerState() -> [MLXArray] { [] }
}

/// A ``CountingKVCache`` that never trims.
public final class NonTrimmableCountingKVCache: CountingKVCache {
    public override var isTrimmable: Bool { false }

    @discardableResult
    public override func trim(_ n: Int) -> Int { 0 }

    public override func copy() -> any KVCache {
        let c = NonTrimmableCountingKVCache()
        c.offset = offset
        return c
    }
}

/// A target whose prediction at every position is a pure function of that absolute position, so a
/// speculative run and a greedy run must agree token for token no matter how the drafts land.
/// (The older `MockMainModel` scripts by forward-call index, which only stays in step when every
/// draft is accepted -- too weak to test an accept/reject path across a wrap.)
///
/// It writes real K/V through whatever caches it is handed and emits `sharedKV` from what those
/// caches returned, clamped per layer type exactly as the Gemma 4 emit hook does. So the cache
/// machinery under test is exercised rather than simulated.
public final class PositionScriptedMainModel: Module, LanguageModel, KVCacheDimensionProvider {
    public static let vocabularySize = 32

    /// Which of `prepare`'s two shapes this model returns, and whether it emits drafter state
    /// while doing so. The iterator reconciles the emitted snapshot at a different site in each
    /// case, and the re-prime forward exists only for the middle one.
    public enum PrefillStyle {
        /// `prepare` hands back the prompt; the iterator runs the final position itself.
        case tokens
        /// `prepare` evaluates the prompt and returns logits, carrying no drafter state -- so the
        /// iterator follows up with a re-prime forward to obtain some.
        case logits
        /// `prepare` evaluates the prompt and threads drafter state through, so no re-prime.
        case logitsWithDrafterState
    }

    public var kvHeads: [Int] { Array(repeating: 1, count: newCache(parameters: nil).count) }
    public let script: [Int32]
    public let slidingWindow: Int
    public let wideSlidingWindow: Int?
    public let omitDrafterState: Bool
    /// Emit `mtpSharedKVStatesKey` without the source map that says which entry each tuple came
    /// from -- a snapshot the consumer cannot place against the cache.
    public let omitSharedKVSources: Bool
    public let prefillStyle: PrefillStyle

    public private(set) var emittedSharedKVSpans: [[String: Int]] = []
    public private(set) var verifyChunkLengths: [Int] = []
    public private(set) var forwardCallCount = 0

    public init(
        script: [Int32], slidingWindow: Int, wideSlidingWindow: Int? = nil,
        omitDrafterState: Bool = false, omitSharedKVSources: Bool = false,
        prefillStyle: PrefillStyle = .tokens
    ) {
        self.script = script
        self.slidingWindow = slidingWindow
        self.wideSlidingWindow = wideSlidingWindow
        self.omitDrafterState = omitDrafterState
        self.omitSharedKVSources = omitSharedKVSources
        self.prefillStyle = prefillStyle
        super.init()
    }

    /// Layer 0 is global, layer 1 slides -- Gemma 4's topology in miniature. `wideSlidingWindow`
    /// adds a third layer sliding at a different width, so a timeline between the two widths has
    /// one ring wrapped and the other not.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        var cache: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: slidingWindow, keep: 0)]
        if let wideSlidingWindow {
            cache.append(RotatingKVCache(maxSize: wideSlidingWindow, keep: 0))
        }
        return cache
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        switch prefillStyle {
        case .tokens:
            return .tokens(input.text)
        case .logits:
            return .logits(callAsFunction(input.text, cache: cache, state: nil))
        case .logitsWithDrafterState:
            var emitting = LMOutput.State()
            emitting[mtpEmitFlagKey] = true
            return .logits(callAsFunction(input.text, cache: cache, state: emitting))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(from: cache?.first?.offset ?? 0, positions: inputs.dim(-1))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardCallCount += 1
        let positions = input.tokens.dim(-1)
        // Read the base position before the caches move; every entry is at the same offset.
        let base = cache?.first?.offset ?? 0
        let logits = makeLogits(from: base, positions: positions)

        guard let cache else { return LMOutput(logits: logits) }
        if positions > 1 { verifyChunkLengths.append(positions) }

        let (keys, values) = Self.positionedKV(base ..< (base + positions))
        let presented = cache.map { $0.update(keys: keys, values: values) }

        guard !omitDrafterState, state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }

        // Emitted verbatim: presenting a sliding layer more than its window is what the write
        // path does, and clamping it back is the consumer's job now, not the target's.
        var sharedKV = [String: (MLXArray, MLXArray)]()
        sharedKV["full_attention"] = presented[0]
        sharedKV["sliding_attention"] = presented[1]
        emittedSharedKVSpans.append(
            sharedKV.mapValues { $0.0.dim(SyntheticMTPValues.sequenceAxis) })

        var out = LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, SyntheticMTPValues.rowWidth])
        if !omitSharedKVSources {
            out[mtpSharedKVSourceIndicesKey] = ["full_attention": 0, "sliding_attention": 1]
        }
        out[mtpSharedKVStatesKey] = sharedKV
        return LMOutput(logits: logits, state: out)
    }

    private func makeLogits(from base: Int, positions: Int) -> MLXArray {
        let vocab = Self.vocabularySize
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            data[i * vocab + Int(prediction(at: base + i))] = SyntheticMTPValues.selectedLogit
        }
        return MLXArray(data, [1, positions, vocab])
    }

    /// The token this model predicts will follow absolute position `p`.
    private func prediction(at p: Int) -> Int32 { script[p % script.count] }

    private static func positionedKV(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
        let count = positions.count
        let keys = MLXArray(
            positions.flatMap { Array(repeating: Float($0), count: SyntheticMTPValues.rowWidth) },
            [1, 1, count, SyntheticMTPValues.rowWidth])
        return (keys, keys * -1)
    }

}
