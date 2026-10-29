// Copyright © 2026 Apple Inc.

import Foundation
import MLX

enum MTPInitializationError: Error, LocalizedError {
    // Only emitted by the initial staging probe, before target prefill.
    case unsupportedSpeculativeCache

    var errorDescription: String? {
        switch self {
        case .unsupportedSpeculativeCache:
            "MTP speculative decoding requires a stageable main KV cache."
        }
    }
}

func batchedGreedyMTPVerification(
    logits: MLXArray, draftTokens: MLXArray
) -> BatchedMTPVerification.Result {
    BatchedMTPVerification.matching(
        targetTokens: argMax(logits, axis: -1), draftTokens: draftTokens)
}

/// Generator of tokens using MTP (Multi-Token Prediction) speculative
/// decoding.
///
/// Parallels ``SpeculativeTokenIterator`` but for Gemma 4 - style drafters
/// that share K/V with the target model and produce K - 1 candidate tokens
/// per round in a single ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)`` call (rather
/// than K sequential single-token calls). Every per-round input —
/// `lastToken`, `lastHidden`, `sharedKV`, `positionIds` — is threaded as a
/// method argument, with the target's last hidden state and per-`layer_type`
/// shared K/V extracted from the ``LMOutput/State`` emitted by the target on
/// the previous main-model call.
/// If the drafter needs its own KV cache (Qwen MTP), that cache is owned by
/// this iterator, prefilled over the shifted prompt, and reconciled against
/// accepted target tokens after every verify pass; it is never stored on the
/// shared drafter model.
///
/// The iterator pre-populates each main-model call's incoming `state` with
/// ``mtpEmitFlagKey`` set to `true`, opting the target into populating
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey`` on its returned
/// ``LMOutput/state``. If the target ever returns nil or partial state
/// (for example when a shared-KV drafter can no longer read regular K/V), the
/// iterator transparently switches into a
/// single-token "passthrough" mode for the remainder of generation —
/// a mid-generation capability loss never crashes or corrupts the stream.
///
/// Port of `_speculative_walk` from mlx-vlm/generate.py at SHA `d49d428`,
/// with no-mutation-during-eval idioms (state is threaded through method
/// args; drafter holds no target-derived state — the target and optional
/// per-stream drafter cache are passed as parameters to `draftBlock(...)` so
/// drafter instances are safe to share across iterators).
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    let drafter: any MTPDrafterModel

    var mainState: LMOutput.State?
    let mainCacheStorage: KVCacheStorage
    var mainCache: [KVCache] { mainCacheStorage.cache }
    var kvCachePlan: KVCachePlan { mainCacheStorage.plan }
    var drafterState: MTPDrafterState?
    private var continuationToResume: MTPDrafterContinuation?
    private var finalizedContinuation: MTPDrafterContinuation?

    /// Hidden state of the final token currently represented by the target
    /// cache. This is the architecture-neutral seam a resumable drafter uses
    /// to bridge a later prompt suffix.
    private var targetBoundaryHidden: MLXArray?
    private var lastVerifyHidden: MLXArray?
    private var lastVerifyStart = 0
    private var lastVerifyAccepted = 0

    var processor: LogitProcessor?
    let sampler: LogitSampler

    /// The top candidates whose log probabilities each token reports, or `nil` when the
    /// stream does not ask for them. A stream that asks decodes without speculation.
    let logProbabilityTopK: Int?

    /// Log probabilities of the token sampled last. A stream that asks for them decodes
    /// plainly from its start, thus the token sampled last is the token ``next()`` returns.
    private var deferredLogProbabilities: DeferredTokenLogProbabilities?

    /// Log probabilities for the token returned by the most recent ``next()`` call, or `nil`
    /// when the stream does not ask for them.
    public private(set) var lastLogProbabilities: DeferredTokenLogProbabilities?

    /// Whether the main cache holds the token the most recent ``next()`` call returned.
    ///
    /// An accepted draft is in the cache when it is returned: the verify pass of its round
    /// fed it. The final token of a round, the prefill bonus that no priming forward fed, and
    /// each token of plain decode are returned before any forward pass feeds them.
    public private(set) var hasCachedLastToken = false

    /// Distribution surface for non-greedy speculative sampling. Built-in
    /// temperature/top-p samplers conform; argmax intentionally does not and
    /// keeps the cheaper exact-token greedy path.
    let distributionSampler: (any DistributionLogitSampler)?

    /// Independent RNG used for speculative acceptance and residual/bonus
    /// draws. Draft proposals use `draftSampler`'s own RNG; separating them is
    /// required by the rejection-sampling proof.
    let speculativeRandomState: MLXRandom.RandomState

    /// Independent sampler handed to the drafter, wrapped to mask suppressed
    /// token IDs when the target has any: a draft of a suppressed token would
    /// always be rejected by the verifier, wasting draft slots.
    /// Distribution-capable drafters receive a copy of the full target
    /// processor as well, so their proposal distribution matches the target's
    /// (penalties included).
    ///
    /// A stateful drafter drafts argmax tokens unless the stream verifies by
    /// rejection sampling: an argmax draft reads no random state, so the verifier's
    /// sample-and-match keeps the target distribution exact.
    let draftSampler: LogitSampler

    /// Whether a sampled stream of a stateful distribution drafter verifies its drafts by
    /// rejection sampling (``MTPSampledVerification/rejectionSampling``) rather than by
    /// sample-and-match.
    let verifiesStatefulDraftsByRejection: Bool

    /// Sampler for older greedy-only drafters whose API cannot receive a
    /// processor.
    let legacyDraftSampler: LogitSampler

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    /// Total tokens proposed per round (`blockSize - 1` drafted, plus the
    /// bonus token from the previous verify). Mirrors mlx-vlm's
    /// `draft_block_size` parameter.
    public let blockSize: Int

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Number of pending tokens already represented by `mainCache`. The last
    /// verifier sample is not committed until a later forward pass.
    private var committedPendingTokenCount = 0

    /// Set to `true` when the iterator detects that the target can no
    /// longer emit drafter state (typically due to KV cache quantization
    /// converting `Gemma4SharedKVState.regular` to `.quantized`). Once set,
    /// `next()` runs single-token generation against the main model only —
    /// no further `speculateRound` calls. Sticky: never reverts to `false`.
    private var passthrough = false
    private var passthroughLoggedOnce = false

    /// Verify-position index in the prior round's emitted hidden that
    /// produced the newly-accepted bonus's logit prediction. Set at the end
    /// of each `speculateRound()`. `nil` on the first round means slice the
    /// last position (round 1's `lastHidden` has shape `[B, 1, hidden]`, so
    /// last-position == only-position == the correct slot). Round 2+ slices
    /// at this index, mirroring mlx-lm's `verify.hidden[:, accepted : accepted + 1, :]`.
    /// Mismatch (e.g. unconditional last-position) is silent: drafter still
    /// produces tokens, but they're conditioned on the wrong slot → less
    /// coherent drafts → lower acceptance, especially at higher blockSize.
    private var lastRoundAccepted: Int? = nil

    public var promptPrefillTime: TimeInterval = 0.0
    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }

    public var evictedTokenCount: Int {
        KVCacheTree.leaves(in: mainCache).map(\.cache.evictedTokenCount).max() ?? 0
    }

    public var reasoningTokenCount: Int? {
        (processor as? any GenerationReasoningTokenCounting)?.generationReasoningTokenCount
    }

    public mutating func recordEmittedToken(_ token: Int) {
        processor?.didEmit(token: token)
    }

    /// Target-owned continuation state with transient MTP exchange values
    /// removed. Model-owned anchors such as VLM M-RoPE deltas remain intact.
    public var state: LMOutput.State? {
        persistentTargetState(from: mainState)
    }

    /// A copy of the drafter continuation at the end of the prompt, which a prompt cache
    /// checkpoint there keeps, or `nil`.
    ///
    /// Right after the prefill the main cache stands at the end of the prompt and the drafter
    /// holds one entry more: the entry of the first bonus token, which the main cache does not
    /// hold yet. The copy drops that entry, which leaves the form of a finalized continuation at
    /// the end of the prompt. `nil` once the first token is read, when the stream does not
    /// speculate or its drafter is not resumable, and when the prefill left the main cache past
    /// the prompt.
    package var promptBoundaryContinuation: MTPDrafterContinuation? {
        guard !passthrough, pendingIndex == 0, pendingTokens.count == 1,
            committedPendingTokenCount == 0,
            let resumableDrafter = drafter as? any ResumableMTPDrafterModel,
            let drafterState, let targetBoundaryHidden
        else { return nil }
        let promptEnd = mainCacheStorage.processedTokenCount
        var continuation = MTPDrafterContinuation(
            state: drafterState, targetBoundaryHidden: targetBoundaryHidden,
            targetProcessedTokenCount: promptEnd, drafterID: resumableDrafter.drafterID
        ).copy()
        guard continuation.state.removeLastEntries(1),
            continuation.isAligned(
                withMainTokenCount: promptEnd, drafterID: resumableDrafter.drafterID)
        else { return nil }
        return continuation
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    // Optional instrumentation used by acceptance-rate floor tests.
    // Public read-only so test cases can compute `acceptedCount /
    // proposedCount` after the stream drains.
    public private(set) var acceptedCount: Int = 0
    public private(set) var proposedCount: Int = 0

    // Reason recorded the first time sticky-passthrough engaged, or nil if
    // the iterator stayed speculative for the full stream. Surfaced through
    // ``MTPStatsCollecting`` so `generateLoopTask` can include it on the
    // emitted `.info` event.
    public private(set) var passthroughReason: String?

    /// Prefills the main model with `input` and prepares the drafter for the first round.
    ///
    /// - Parameters:
    ///   - input: The prompt, or the suffix of the prompt that `mainCache` does not hold.
    ///   - mainModel: The target model the drafter speculates for.
    ///   - drafter: The MTP drafter of `mainModel`.
    ///   - mainCache: The cache of `mainModel`, or `nil` for a new cache.
    ///   - mainState: The model state saved with `mainCache`, as ``state`` exported it.
    ///   - mtpContinuation: The drafter continuation the generation that left `mainCache`
    ///     gave back, or `nil` to start the drafter at the cached position.
    ///   - parameters: The generation parameters.
    ///   - blockSize: The tokens of each round: up to `blockSize - 1` drafts and the bonus.
    ///   - minimumDraftProbability: The drafter top-1 probability below which a chaining
    ///     drafter stops a round's chain, from 0 (draft every step) to 1.
    ///   - components: Optional behavioral components, e.g. a custom ``LogitProcessor``.
    /// - Throws: An ``MTPDrafterCompatibilityError`` when `drafter` cannot draft for
    ///   `mainModel`, a ``SpeculativeDecodingConfigurationError`` for a floor outside 0 to 1, or
    ///   the error of the prefill.
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCache: [KVCache]? = nil,
        mainState: LMOutput.State? = nil,
        mtpContinuation: MTPDrafterContinuation? = nil,
        parameters: GenerateParameters,
        blockSize: Int,
        minimumDraftProbability: Float = 0,
        components: GenerationComponents = .init()
    ) throws {
        try drafter.validateCompatibility(with: mainModel)

        let kvCachePlan = try parameters.kvCachePlan()
        try self.init(
            input: input,
            mainModel: mainModel,
            drafter: drafter,
            mainCacheStorage: KVCacheStorage(
                mainCache ?? (try mainModel.newCache(parameters: parameters)), plan: kvCachePlan),
            mainState: mainState,
            mtpContinuation: mtpContinuation,
            parameters: parameters,
            blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability,
            components: components)
    }

    package init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCacheStorage: KVCacheStorage,
        mainState: LMOutput.State? = nil,
        mtpContinuation: MTPDrafterContinuation? = nil,
        parameters: GenerateParameters,
        blockSize: Int,
        minimumDraftProbability: Float = 0,
        components: GenerationComponents = .init()
    ) throws {
        try drafter.validateCompatibility(with: mainModel)

        guard blockSize >= 2 else {
            throw KVCacheError(
                message:
                    "MTPSpeculativeTokenIterator requires blockSize >= 2 (1 bonus + K-1 drafted)"
            )
        }
        try SpeculativeDecodingConfig.validate(minimumDraftProbability: minimumDraftProbability)
        let kvCachePlan = mainCacheStorage.plan
        let mainCacheStorage = try kvCachePlan.validated(mainCacheStorage)
        let mainCache = mainCacheStorage.cache
        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter

        self.mainCacheStorage = mainCacheStorage
        self.mainState = mainState

        let sampler = parameters.sampler()
        // Decided before any prompt work: a stream that cannot speculate gets no
        // drafter state, no emit-flag prefill, and no drafter prefill.
        let standDownReason = Self.standDownReason(
            drafter: drafter, parameters: parameters, sampler: sampler)
        let speculates = standDownReason == nil
        // A continuation of another drafter holds caches this drafter cannot read.
        let ownContinuation = mtpContinuation.flatMap { continuation in
            (drafter as? any ResumableMTPDrafterModel)?.drafterID == continuation.drafterID
                ? continuation : nil
        }
        self.continuationToResume = speculates ? ownContinuation : nil
        self.drafterState =
            speculates
            ? ownContinuation?.state
                ?? (drafter as? any StatefulMTPDrafterModel)?.makeState(parameters: parameters)
            : nil
        // Every later drafter state is this one or a copy of it, so the floor holds for the stream.
        self.drafterState?.minimumDraftProbability = minimumDraftProbability

        self.sampler = sampler
        try components.validate(parameters: parameters)
        self.distributionSampler = sampler as? any DistributionLogitSampler
        self.logProbabilityTopK = parameters.logProbabilities.map { Swift.max($0, 0) }
        self.speculativeRandomState =
            parameters.seed.map {
                MLXRandom.RandomState(seed: $0 &+ 0x9E37_79B9_7F4A_7C15)
            } ?? MLXRandom.RandomState()
        self.processor = components.logitProcessor(parameters: parameters, model: mainModel)

        let verifiesStatefulDraftsByRejection =
            sampler is any DistributionLogitSampler
            && drafter is any StatefulMTPDistributionDrafterModel
            && parameters.mtpSampledVerification == .rejectionSampling
        self.verifiesStatefulDraftsByRejection = verifiesStatefulDraftsByRejection
        var draftParameters = parameters
        draftParameters.seed = parameters.seed.map { $0 &+ 0xD1B5_4A32_D192_ED03 }
        var drafterSampler: LogitSampler =
            drafter is any StatefulMTPDrafterModel && !verifiesStatefulDraftsByRejection
            ? ArgMaxSampler() : draftParameters.sampler()
        // Draft proposals bypass the iterator's processor chain, so suppressed
        // ids have to be masked at the drafter's own sampler.
        if let suppressor = makeSuppressTokensProcessor(model: mainModel) {
            drafterSampler = SuppressTokensSampler(base: drafterSampler, suppressor: suppressor)
        }
        self.draftSampler = drafterSampler
        self.legacyDraftSampler = drafterSampler

        self.maxTokens = parameters.maxTokens
        // A round presents `blockSize` positions at once, and a sliding layer can only show a
        // query the `maxSize` entries before it. Past that the extra drafts still decode
        // correctly -- masks are position-relative and the staged view is clamped -- but the
        // deepest ones attend over less context than the verifier gave them, so acceptance
        // stops meaning what the stats say it means. Clamp rather than trap: the block size is
        // a tuning knob, not a correctness input.
        let drafterBlockSize = Swift.min(blockSize, drafter.maximumBlockSize ?? blockSize)
        let effectiveBlockSize = Swift.min(
            drafterBlockSize, Self.maximumBlockSize(for: mainCache))
        self.blockSize = effectiveBlockSize

        // Probe by opening a round at the width rounds will actually use and discarding it,
        // rather than duplicating the leaf classification as a predicate that could drift from
        // it. Qwen's hybrid cache is the one typed exception: its target advertises a bounded
        // recurrent checkpoint and performs the round in place.
        let usesNativeHybridRewind = Self.verifiesInPlace(
            mainModel: mainModel, mainCache: mainCache, drafts: effectiveBlockSize - 1)
        // A stream that decodes plainly opens no round, so its cache need not stage one.
        if speculates, !usesNativeHybridRewind {
            guard
                let probe = self.mainCacheStorage.beginRound(
                    maximumPositions: effectiveBlockSize)
            else {
                throw MTPInitializationError.unsupportedSpeculativeCache
            }
            self.mainCacheStorage.rollback(probe)
        }
        if let standDownReason {
            switchToPassthrough(reason: standDownReason)
        }

        let prefillStart = Date.timeIntervalSinceReferenceDate
        var mtpPrefill = parameters.prefill
        if speculates, drafter.requiresPromptPrefill || continuationToResume != nil,
            !(drafter is any ResumableMTPDrafterModel)
        {
            // A resumable drafter takes the prompt chunk by chunk through the prefill's chunk
            // handler. Any other drafter primes from one hidden row per prompt token at once, so
            // its target runs the reference single-forward computation.
            mtpPrefill.stepSize = Int.max
            mtpPrefill.chunking = .unchunked
        }
        try prepare(input: input, prefill: mtpPrefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    /// Why a stream cannot speculate at all, or `nil` when it can.
    ///
    /// Every input is known before the prompt is read, so the decision comes before
    /// the prefill: a stream that cannot speculate does no drafter work. A
    /// mid-stream exit would also strand a staged round.
    ///
    /// - Parameters:
    ///   - drafter: The drafter of the stream.
    ///   - parameters: The generation parameters of the stream.
    ///   - sampler: The sampler `parameters` makes.
    /// - Returns: The passthrough reason, or `nil` when the stream can speculate.
    private static func standDownReason(
        drafter: any MTPDrafterModel, parameters: GenerateParameters, sampler: LogitSampler
    ) -> String? {
        if drafter.requiresGreedySampling, parameters.temperature != 0 {
            return greedySamplingReason
        }
        // Speculative sampling can only preserve the target distribution when
        // the drafter reports the distribution its proposals were drawn from, or
        // when its proposals read no random state. A stateful drafter drafts with
        // the argmax samplers this iterator hands it, and its drafts are verified
        // by sample-and-match. Any other token-only drafter under a sampling
        // configuration would silently bias the output, so fail safe to
        // target-only generation.
        if sampler is any DistributionLogitSampler,
            !(drafter is any MTPDistributionDrafterModel),
            !(drafter is any StatefulMTPDrafterModel)
        {
            return samplingWithoutDraftLogitsReason
        }
        if parameters.logProbabilities != nil {
            return logProbabilitiesReason
        }
        return nil
    }

    /// The passthrough reason of a sampled stream whose drafter drafts greedily only.
    static let greedySamplingReason =
        "MTP drafter requires temperature == 0; generating without speculation"

    static let samplingWithoutDraftLogitsReason =
        "MTP drafter does not expose the draft distribution required for sampling; "
        + "generating without speculation"

    /// The passthrough reason of a stream that asks for log probabilities.
    ///
    /// Plain decode reports the log probabilities of each token from the forward
    /// pass that samples it, as ``TokenIterator`` does. A verify pass would report
    /// them from the logits of a multi-token forward, which differ from plain
    /// decode by rounding, so such a stream decodes without speculation.
    static let logProbabilitiesReason =
        "logprobs requested; generating without speculation so each token reports the "
        + "log probabilities of plain decode"

    /// The passthrough reason of a stream whose target did not report a hidden row for each
    /// prompt position its drafter must take.
    static let missingPromptHiddenReason =
        "MTP drafter has no target hidden state for the prompt; continuing without speculation"

    /// The passthrough reason of a stream whose resumable drafter could neither resume nor
    /// cold-start over the prompt positions after the target's cached ones.
    static let coldStartFailedReason =
        "MTP drafter could not start at the cached position; continuing without speculation"

    static let missingSharedKVSourcesReason =
        "main model did not report which cache entries its shared K/V came from; continuing "
        + "without speculation"

    /// Whether a round of `drafts` drafts verifies in place on a hybrid target and rewinds its
    /// rejected drafts there, rather than through a staged round.
    ///
    /// The target must keep a recurrent snapshot slot for each draft (Qwen's Gated DeltaNet
    /// layers), and every attention entry must rewind the drafts after the verify wrote them:
    /// see ``canRewindSpeculativeVerify(_:positions:)``. A rotating ring that has wrapped, from
    /// `maxKVSize`, still rewinds them, and each verify query attends as plain decode does, kept
    /// prefix included (``RotatingKVCache/makeMask(n:windowSize:returnArray:)``); thus such a
    /// stream keeps speculating past its window.
    ///
    /// - Parameters:
    ///   - mainModel: The target model.
    ///   - mainCache: The cache of the target.
    ///   - drafts: The most drafts of the round.
    /// - Returns: `true` when the round verifies in place.
    package static func verifiesInPlace(
        mainModel: any LanguageModel, mainCache: [KVCache], drafts: Int
    ) -> Bool {
        let nativeRewindDepth =
            (mainModel as? any SpeculativeCacheRewindModel)?.maximumNativeTargetCacheRewind ?? 0
        return nativeRewindDepth >= drafts
            && mainCache.contains { $0 is MambaCache }
            && canRewindSpeculativeVerify(mainCache, positions: drafts + 1)
    }

    /// The tightest sliding window among the cache's layers, or `nil` when no layer slides.
    static func narrowestSlidingWindow(in cache: [KVCache]) -> Int? {
        KVCacheTree.leaves(in: cache)
            .compactMap { leaf -> Int? in
                guard case .rotating(let rotating) = leaf.kind else { return nil }
                return rotating.maxSize
            }
            .min()
    }

    /// The widest round the narrowest sliding layer can present coherently: one bonus plus a
    /// window's worth of drafts. Unbounded when no layer slides.
    static func maximumBlockSize(for cache: [KVCache]) -> Int {
        guard let narrowest = narrowestSlidingWindow(in: cache) else { return .max }
        return Swift.max(2, narrowest + 1)
    }

    /// Prefill the main model with the prompt. The drafter's own state starts
    /// empty; its first-round conditioning inputs come from the prefill's
    /// `LMOutput.state`. A stream that already decodes plainly asks the target
    /// for no drafter state and leaves the drafter alone.
    mutating func prepare(input: LMInput, prefill: PrefillParameters = .init()) throws {
        processor?.prompt(input.text.tokens)
        prefill.prefix?(mainCacheStorage.processedTokenCount)
        let inputLength = input.text.cacheSequenceLength
        let cachePositionBeforePrefill = mainCacheStorage.processedTokenCount
        let needsFullPromptHidden =
            !passthrough && (drafter.requiresPromptPrefill || continuationToResume != nil)

        let persistentState = persistentTargetState(from: mainState)
        var prefillState = persistentState ?? LMOutput.State()
        prefillState[mtpEmitFlagKey] = true
        // Stateful drafters and warm continuations need one hidden row per
        // prompt token, so their `prepare` call receives the emit flag. Other
        // drafters are primed by the explicit one-token follow-up below.
        let bonusState: LMOutput.State? = passthrough ? persistentState : prefillState

        // A resumable drafter takes the rows of each prefill chunk as the target evaluates it;
        // the rows of the positions after the last chunk come with the target's final forward.
        let feed =
            needsFullPromptHidden
            ? makePrefillFeed(cachePosition: cachePositionBeforePrefill) : nil
        var incomingPrefillState: LMOutput.State? =
            needsFullPromptHidden ? prefillState : persistentState
        incomingPrefillState?[mtpPrefillChunkHandlerKey] = feed?.chunkHandler
        switch try mainModel.prepare(
            input, cache: mainCache, state: incomingPrefillState, prefill: prefill)
        {
        case .tokens(let tokens):
            let remainingLength = tokens.cacheSequenceLength
            precondition(
                remainingLength <= inputLength,
                "Main model prepare returned more tokens than it received")
            mainCacheStorage.commitProcessedTokens(inputLength - remainingLength)
            y = tokens
            // Final prompt position not yet evaluated -- run one forward to
            // produce the bonus token AND prime drafter state.
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: bonusState)
            mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
            let token = sampleToken(from: result.logits)
            y = .init(tokens: token)
            mainState = result.state
            // Prefill emits too, and a multi-token prefill presents a sliding layer more than its
            // window just as a verify pass does. Nothing was rejected here, so only the head
            // clamp applies -- but a snapshot that cannot be placed against the entries at all is
            // refused here for the same reason it is mid-stream, and before anything reads it.
            if !reconcileSharedKVState(
                &mainState, discarding: 0,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
            {
                switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                mainState = nil
            }
            // Yield the bonus to the iterator's consumer. Without this,
            // the iterator silently starts 1 position ahead of an
            // equivalent autoregressive run, violating speculative
            // decoding's bit-exact-equivalence-to-greedy guarantee.
            pendingTokens.append(token.item(Int.self))

            // the model reported per-chunk progress; the bonus forward above
            // consumed the remainder of the prompt
            let total = input.text.tokens.size
            prefill.progress?(total, total)
        case .logits(let prefillResult):
            mainCacheStorage.commitProcessedTokens(inputLength)
            // Some `prepare` implementations evaluate the final position
            // themselves and return logits directly; their `state` here may
            // or may not carry drafter state depending on whether the model
            // override threads it.
            let token = sampleToken(from: prefillResult.logits)
            y = .init(tokens: token)
            mainState = prefillResult.state
            // The returned state keeps what the prefill was handed, the chunk handler included.
            mainState?[mtpPrefillChunkHandlerKey] = nil
            let placeable = reconcileSharedKVState(
                &mainState, discarding: 0,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
            if !placeable {
                switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                mainState = nil
            }

            // If prefill didn't emit drafter state, do one more forward call
            // with the just-sampled bonus token to prime the state. The cost
            // is one extra token's forward pass; acceptable. Skipped once the
            // stream decodes plainly -- a refused snapshot included: it exists
            // only to obtain drafter state, and such a stream has no use for any.
            let missingRequiredDrafterState =
                mainState?[mtpLastHiddenStatesKey] == nil
                || (drafter.requiresSharedTargetKV
                    && mainState?[mtpSharedKVStatesKey] == nil)
            if !passthrough, missingRequiredDrafterState {
                var primeState = mainState ?? prefillState
                primeState[mtpEmitFlagKey] = true
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: primeState)
                mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
                mainState = primed.state
                if !reconcileSharedKVState(
                    &mainState, discarding: 0,
                    lengths: mainCacheStorage.emittedLength(forLeaf:))
                {
                    switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
                    mainState = nil
                }
                // Resample bonus from this forward's logits so the chain stays
                // coherent at this position (the cache offset moves by 1, so
                // we must re-pick the bonus from the new step's logits).
                let newToken = sampleToken(from: primed.logits)
                y = .init(tokens: newToken)
                // Yield BOTH bonuses to the consumer, in sample order.
                // `token` is the prefill-position-N sample (consumed by the
                // re-prime forward, now committed in cache); `newToken` is
                // the prefill-position-N+1 sample that becomes the input
                // to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
                pendingTokens.append(newToken.item(Int.self))
                committedPendingTokenCount = 1
            } else {
                // Prefill state already carried drafter keys; the single
                // bonus is the input to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
            }
        }

        if let targetHidden = mainState?[mtpLastHiddenStatesKey], targetHidden.dim(1) > 0 {
            targetBoundaryHidden = targetHidden[0..., (-1)..., 0...]
        }

        let continuation = continuationToResume
        continuationToResume = nil
        if !passthrough, let feed, feed.fedTokenCount > 0 {
            primeDrafterState(after: feed, promptTokens: input.text.tokens)
        } else if !passthrough, drafter.requiresPromptPrefill || continuation != nil {
            primeDrafterState(
                promptTokens: input.text.tokens, cachePosition: cachePositionBeforePrefill,
                continuation: continuation)
        } else if !passthrough, cachePositionBeforePrefill > 0,
            drafter is any StatefulMTPDrafterModel
        {
            switchToPassthrough(
                reason:
                    "stateful MTP drafter has no warm continuation; continuing without speculation"
            )
        }

        try kvCachePlan.applyAndValidate(to: mainCacheStorage)
    }

    /// Samples a token from the logits of the last position of `logits`.
    ///
    /// The processor shapes the logits first and learns of the sampled token after. When the
    /// stream asks for log probabilities, they are recorded from the processed logits.
    ///
    /// - Parameter logits: Target logits, shape `[B, positions, vocabulary]`.
    /// - Returns: The sampled token, shape `[B]`.
    private mutating func sampleToken(from logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        deferredLogProbabilities = logProbabilityTopK.map {
            LogProbabilityDistribution(logits: logits).recording(token: token, topK: $0)
        }
        if let deferredLogProbabilities {
            // Settled with the token: the prefill runs on the caller's thread, and the
            // generation worker that reads them cannot evaluate a graph naming its stream.
            eval([token] + deferredLogProbabilities.arrays)
        }
        processor?.didSample(token: token)
        return token
    }

    /// Primes the drafter-owned state over the prompt the target just prefilled.
    ///
    /// A prompt that starts at position zero prefills the drafter from scratch. On a warm target
    /// cache an aligned continuation resumes; when there is none, or the drafter refuses it, a
    /// resumable drafter cold-starts at the cached position instead. Only a drafter that can do
    /// none of these leaves the stream to the target alone.
    ///
    /// - Parameters:
    ///   - promptTokens: The prompt tokens the target prefilled.
    ///   - cachePosition: The target cache position before the prefill.
    ///   - continuation: The drafter continuation carried with the target cache, if any.
    private mutating func primeDrafterState(
        promptTokens: MLXArray, cachePosition: Int, continuation: MTPDrafterContinuation?
    ) {
        let promptLength = promptTokens.dim(-1)
        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var state = drafterState,
            let targetHidden = mainState?[mtpLastHiddenStatesKey],
            targetHidden.dim(1) >= promptLength
        else {
            switchToPassthrough(reason: Self.missingPromptHiddenReason)
            return
        }
        let suffixHidden = targetHidden[0..., ..<promptLength, 0...]

        guard cachePosition > 0 else {
            statefulDrafter.prepareDrafterState(
                target: mainModel, promptTokens: promptTokens, targetHidden: suffixHidden,
                firstBonus: y.tokens, positionDeltas: mainState?[mtpPositionDeltasKey],
                state: &state, sampler: mtpDrafterSeedSampler)
            drafterState = state
            return
        }
        guard let resumableDrafter = drafter as? any ResumableMTPDrafterModel else {
            switchToPassthrough(
                reason:
                    "stateful MTP drafter cannot initialize from a warm target suffix; continuing without speculation"
            )
            return
        }

        resumeOrColdStart(
            resumableDrafter, segmentTokens: promptTokens, segmentHidden: suffixHidden,
            boundary: continuation?.alignedBoundary(
                withMainTokenCount: cachePosition, drafterID: resumableDrafter.drafterID),
            position: cachePosition, state: state)
    }

    /// Primes a resumable drafter over a prompt segment the target just prefilled.
    ///
    /// The segment resumes the drafter from `boundary`. Without a boundary, or when the drafter
    /// refuses the resume, the drafter cold-starts at the segment instead.
    ///
    /// - Parameters:
    ///   - resumableDrafter: The drafter of the stream.
    ///   - segmentTokens: The prompt segment, which ends the prompt.
    ///   - segmentHidden: One target hidden row per segment token.
    ///   - boundary: The target hidden row of the position before the segment, when `state` holds
    ///     the entries up to that position.
    ///   - position: The target cache position of the first segment token.
    ///   - state: The drafter state to extend or replace.
    private mutating func resumeOrColdStart(
        _ resumableDrafter: any ResumableMTPDrafterModel, segmentTokens: MLXArray,
        segmentHidden: MLXArray, boundary: MLXArray?, position: Int, state: MTPDrafterState
    ) {
        let positionDeltas = mainState?[mtpPositionDeltasKey]
        var state = state
        if let boundary,
            resumableDrafter.resumeDrafterState(
                target: mainModel, suffixTokens: segmentTokens, suffixTargetHidden: segmentHidden,
                targetBoundaryHidden: boundary, firstBonus: y.tokens,
                positionDeltas: positionDeltas, state: &state, sampler: mtpDrafterSeedSampler)
        {
            drafterState = state
            return
        }
        guard
            resumableDrafter.coldStartDrafterState(
                target: mainModel, suffixTokens: segmentTokens, suffixTargetHidden: segmentHidden,
                firstBonus: y.tokens, position: position, positionDeltas: positionDeltas,
                state: &state, sampler: mtpDrafterSeedSampler)
        else {
            switchToPassthrough(reason: Self.coldStartFailedReason)
            return
        }
        drafterState = state
    }

    /// A feed that takes each prefill chunk of the prompt into a resumable drafter, or `nil`
    /// when the drafter cannot take the prompt chunk by chunk.
    ///
    /// - Parameter cachePosition: The target cache position before the prefill.
    /// - Returns: The feed, starting from the aligned continuation when the stream carries one.
    private func makePrefillFeed(cachePosition: Int) -> MTPDrafterPrefillFeed? {
        guard let resumableDrafter = drafter as? any ResumableMTPDrafterModel,
            let drafterState
        else { return nil }
        return MTPDrafterPrefillFeed(
            drafter: resumableDrafter, target: mainModel, state: drafterState,
            cachePosition: cachePosition, continuation: continuationToResume)
    }

    /// Primes the drafter after the prefill chunks fed it, over the prompt positions the chunks
    /// did not reach.
    ///
    /// The target's final forward evaluated those positions, so its hidden rows are the rows of
    /// the final prompt segment. The feed resumes the drafter over that segment from the boundary
    /// the chunks left, or cold-starts it there (see ``MTPDrafterPrefillFeed/primedState``).
    ///
    /// - Parameters:
    ///   - feed: The feed the prefill chunks went through.
    ///   - promptTokens: The prompt tokens the target prefilled.
    private mutating func primeDrafterState(
        after feed: MTPDrafterPrefillFeed, promptTokens: MLXArray
    ) {
        let segmentTokens = promptTokens[.ellipsis, feed.fedTokenCount...]
        let segmentLength = segmentTokens.dim(-1)
        guard let targetHidden = mainState?[mtpLastHiddenStatesKey],
            segmentLength > 0, targetHidden.dim(1) >= segmentLength
        else {
            switchToPassthrough(reason: Self.missingPromptHiddenReason)
            return
        }
        guard
            let state = feed.primedState(
                finalSegment: segmentTokens, segmentHidden: targetHidden, firstBonus: y.tokens,
                positionDeltas: mainState?[mtpPositionDeltasKey])
        else {
            switchToPassthrough(reason: Self.coldStartFailedReason)
            return
        }
        drafterState = state
    }

    /// Single round: draft `blockSize - 1` tokens, verify with main, accept
    /// the longest matching prefix, emit the bonus correction.
    mutating func speculateRound() {
        guard !passthrough else { return }
        // A prior round keeps the recurrent snapshot slots of the tokens it accepted until its
        // pending output is drained, so early finalization can rewind its accepted drafts.
        discardSpeculativePromptCacheCheckpoints(mainCache)

        // A speculative round can emit up to `maximumDrafts + 1` tokens: the
        // accepted draft prefix plus the verifier's correction/bonus token.
        // Keep the whole pending buffer within the remaining output budget.
        let maximumDrafts: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }

            let draftBudget = Swift.min(remaining - 1, blockSize - 1)
            guard draftBudget > 0 else {
                if let token = passthroughStep(maintainingDrafterState: true) {
                    pendingTokens.append(token)
                }
                return
            }
            maximumDrafts = draftBudget
        } else {
            maximumDrafts = blockSize - 1
        }

        guard let state = mainState,
            let lastHidden = state[mtpLastHiddenStatesKey]
        else {
            switchToPassthrough(reason: "main model did not emit drafter state")
            return
        }
        let sharedKV = state[mtpSharedKVStatesKey] ?? [:]
        if drafter.requiresSharedTargetKV, sharedKV["full_attention"] == nil {
            switchToPassthrough(reason: "main model did not emit shared target K/V")
            return
        }

        // Attention-only targets verify through a staged round. Qwen's hybrid target instead
        // keeps the snapshot slots of its recurrent entries and rewinds them in place by the
        // rejected drafts; the typed capability and
        // cache topology keep that exception fail-closed.
        let nativeHybridRewind = Self.verifiesInPlace(
            mainModel: mainModel, mainCache: mainCache, drafts: maximumDrafts)
        let round =
            nativeHybridRewind
            ? nil : mainCacheStorage.beginRound(maximumPositions: maximumDrafts + 1)
        guard nativeHybridRewind || round != nil else {
            switchToPassthrough(
                reason: "main KV cache cannot stage a speculative round; continuing without "
                    + "speculation")
            mainState = nil
            return
        }

        // Slice the hidden at the slot that produced the newly-accepted
        // bonus's prediction. Round 1: last (and only) position. Round 2+:
        // index `lastRoundAccepted`, matching mlx-lm's
        // `verify.hidden[:, accepted : accepted + 1, :]` semantic.
        let bonusSlotHidden: MLXArray
        if let idx = lastRoundAccepted {
            bonusSlotHidden = lastHidden[0..., idx ..< (idx + 1), 0...]
        } else {
            bonusSlotHidden = lastHidden[0..., (-1)..., 0...]
        }

        let queryOffset =
            state[mtpSharedKVOffsetsKey]?["full_attention"]
            ?? mainCacheStorage.processedTokenCount

        // Invariant: the snapshot the drafter attends over describes the emitted sequence and
        // nothing else. A global layer's entry spans the whole stream; a sliding layer's spans
        // its window. Both are exact, because the commit that decided what the cache kept also
        // reconciled the snapshot. `dim()` is shape metadata (no eval, no GPU sync).
        assert(
            {
                guard let sources = state[mtpSharedKVSourceIndicesKey] else { return false }
                return sharedKV.allSatisfy { key, entry in
                    guard let leaf = sources[key] else { return false }
                    return entry.0.dim(-2) == mainCacheStorage.emittedLength(forLeaf: leaf)
                }
            }(),
            """
            stale sharedKV: spans \(sharedKV.mapValues { $0.0.dim(-2) }) do not match the \
            emitted lengths implied by a timeline of \(mainCacheStorage.processedTokenCount)
            """
        )

        let bonusToken = y.tokens
        let (draftTokens, draftOutput) = proposeDrafts(
            bonusToken: bonusToken, bonusSlotHidden: bonusSlotHidden, sharedKV: sharedKV,
            positionDeltas: state[mtpPositionDeltasKey], queryOffset: queryOffset,
            maximumDrafts: maximumDrafts)
        // A drafter's confidence floor can stop a chain before `maximumDrafts`.
        let numDraft = draftTokens.dim(-1)
        // draftTokens shape [B, numDraft] -> flatten to [numDraft].
        let flatDraftTokens = draftTokens.flattened()

        // Verify pass: main model evaluates [bonus, draft_1, ..., draft_numDraft]
        // in one forward call, emitting state for next round.
        var verifyState = state
        verifyState[mtpEmitFlagKey] = true
        let verifyTokens = concatenated([bonusToken, flatDraftTokens])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        verifyState[mtpVerifyKeepsSnapshotsKey] = nativeHybridRewind ? true : nil
        let verifyCache = nativeHybridRewind ? mainCache : round!.caches
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: verifyCache, state: verifyState)
        mainState = mainResult.state

        let (accepted, emittedFinalToken) = verifyDrafts(
            targetLogits: mainResult.logits, targetStart: verifyStart,
            draftTokens: flatDraftTokens, draftOutput: draftOutput, numDraft: numDraft)
        // Lookahead bookkeeping for early finalization: the accepted drafts
        // are the pending tokens whose K/V is already committed.
        committedPendingTokenCount = accepted

        proposedCount += numDraft
        acceptedCount += accepted
        lastRoundAccepted = accepted
        if let verifyHidden = mainResult.state?[mtpLastHiddenStatesKey] {
            lastVerifyHidden = verifyHidden
            lastVerifyStart = verifyStart
            lastVerifyAccepted = accepted
            targetBoundaryHidden =
                verifyHidden[0..., (verifyStart + accepted) ..< (verifyStart + accepted + 1), 0...]
        }
        telemetry.recordRound(
            drafted: numDraft,
            accepted: accepted,
            targetVerified: numDraft + 1,
            draftModelCalls: 1
        )

        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            statefulDrafter.commitDrafterState(
                target: mainModel,
                targetHidden: mainResult.state?[mtpLastHiddenStatesKey] ?? lastHidden,
                draftTokens: draftTokens,
                acceptedCount: accepted,
                finalToken: emittedFinalToken,
                positionDeltas: mainResult.state?[mtpPositionDeltasKey],
                state: &currentDrafterState,
                sampler: mtpDrafterSeedSampler)
            drafterState = currentDrafterState
        }

        let rejected = numDraft - accepted
        let snapshotPlaced: Bool
        if nativeHybridRewind {
            if rejected == 0 {
                mainCacheStorage.commitProcessedTokens(verifyInput.cacheSequenceLength)
            } else {
                let rewound = rewindSpeculativePromptCache(
                    mainCache, numTokens: rejected)
                precondition(
                    rewound == rejected,
                    "Target advertised an in-place speculative rewind of \(maximumDrafts) drafts, but rewound \(rewound) of \(rejected) positions"
                )
                // Qwen MTP: attention KV trims the rejected drafts while every
                // GDN cache selects the snapshot slot of its last accepted
                // token, and keeps the slots of the accepted tokens for
                // `finalizeGeneration`. The target's weights are not replayed
                // on rejection.
                mainCacheStorage.commitProcessedTokens(accepted + 1)
            }
            snapshotPlaced = reconcileSharedKVState(
                &mainState, discarding: rejected,
                lengths: mainCacheStorage.emittedLength(forLeaf:))
        } else {
            // Staged layers never held rejected rows; write-through layers clamp back over them.
            let commit = mainCacheStorage.commit(round!, retaining: accepted + 1)
            snapshotPlaced = reconcileSharedKVState(
                &mainState, discarding: rejected,
                lengths: { commit.emittedLengths[$0] })
        }

        guard snapshotPlaced else {
            switchToPassthrough(reason: Self.missingSharedKVSourcesReason)
            mainState = nil
            y = .init(tokens: emittedFinalToken)
            return
        }

        // Dynamic cache quantization may convert regular K/V to quantized K/V,
        // at which point the target's emit-hook cannot provide full_attention
        // shared K/V and the next round transitions to passthrough.
        kvCachePlan.apply(to: mainCacheStorage)

        y = .init(tokens: emittedFinalToken)
    }

    /// Proposes the drafts of one round.
    ///
    /// A stateful drafter extends its own state: with the distribution of each draft when the
    /// stream verifies by rejection sampling, and else with argmax drafts. A stateless
    /// distribution drafter drafts with its distribution, and any other drafter with tokens only.
    ///
    /// - Parameters:
    ///   - bonusToken: The token the round starts from, shape `[B]`.
    ///   - bonusSlotHidden: The target hidden row that predicted `bonusToken`.
    ///   - sharedKV: The target K/V the drafter attends over.
    ///   - positionDeltas: The target position deltas, or `nil`.
    ///   - queryOffset: The absolute position of `bonusToken`.
    ///   - maximumDrafts: The most drafts of the round. A drafter whose confidence floor stops
    ///     its chain drafts fewer.
    /// - Returns: The drafts, shape `[B, n]` with `n` up to `maximumDrafts`, and their
    ///   distribution when the drafter reports it.
    private mutating func proposeDrafts(
        bonusToken: MLXArray, bonusSlotHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        maximumDrafts: Int
    ) -> (tokens: MLXArray, output: MTPDraftBlockOutput?) {
        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            // Stateful drafters own a private cache the iterator must trim in step with
            // acceptance; `commitDrafterState` reconciles it whichever way they drafted.
            defer { drafterState = currentDrafterState }
            if verifiesStatefulDraftsByRejection,
                let distributionDrafter = drafter as? any StatefulMTPDistributionDrafterModel
            {
                let output = distributionDrafter.draftBlockWithLogits(
                    target: mainModel, lastToken: bonusToken, lastHidden: bonusSlotHidden,
                    sharedKV: sharedKV, positionDeltas: positionDeltas, queryOffset: queryOffset,
                    blockSize: maximumDrafts + 1, state: &currentDrafterState,
                    // A copy, discarded with the round: only emitted tokens advance `processor`.
                    processor: processor?.copy(), sampler: draftSampler)
                return (output.tokens, output)
            }
            let tokens = statefulDrafter.draftBlock(
                target: mainModel, lastToken: bonusToken, lastHidden: bonusSlotHidden,
                sharedKV: sharedKV, positionDeltas: positionDeltas, queryOffset: queryOffset,
                blockSize: maximumDrafts + 1, state: &currentDrafterState, sampler: draftSampler)
            return (tokens, nil)
        }
        if let distributionDrafter = drafter as? any MTPDistributionDrafterModel {
            let output = distributionDrafter.draftBlockWithLogits(
                target: mainModel, lastToken: bonusToken, lastHidden: bonusSlotHidden,
                sharedKV: sharedKV, positionDeltas: positionDeltas, queryOffset: queryOffset,
                blockSize: maximumDrafts + 1,
                // A copy, discarded with the round: the drafter's processor starts from the
                // canonical state but must not record the proposals it draws -- only emitted
                // tokens advance `processor`.
                processor: processor?.copy(), sampler: draftSampler)
            return (output.tokens, output)
        }
        // Greedy matching needs tokens only and remains compatible with the original
        // protocol; init already sent sampling runs to passthrough, since without `q`
        // nothing here can preserve the target distribution.
        let tokens = drafter.draftBlock(
            target: mainModel, lastToken: bonusToken, lastHidden: bonusSlotHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas, queryOffset: queryOffset,
            blockSize: maximumDrafts + 1, sampler: legacyDraftSampler)
        return (tokens, nil)
    }

    /// Verifies the drafts of one round against the target logits of its verify pass, and
    /// appends the emitted tokens to the pending buffer.
    ///
    /// Drafts with a reported distribution are verified by rejection sampling. Other drafts are
    /// verified by sample-and-match: in one batched readback when the stream's processor keeps
    /// no state between rows -- argmax rows at temperature 0, sampled rows above it -- and
    /// else row by row.
    ///
    /// - Parameters:
    ///   - targetLogits: The logits of the verify pass, shape `[B, positions, vocabulary]`.
    ///   - targetStart: The position of the bonus token in `targetLogits`.
    ///   - draftTokens: The drafts, shape `[numDraft]`.
    ///   - draftOutput: The drafts with their distribution, or `nil`.
    ///   - numDraft: The drafts of the round.
    /// - Returns: The accepted drafts and the final token of the round, shape `[1]`.
    private mutating func verifyDrafts(
        targetLogits: MLXArray, targetStart: Int, draftTokens: MLXArray,
        draftOutput: MTPDraftBlockOutput?, numDraft: Int
    ) -> (accepted: Int, finalToken: MLXArray) {
        // Other processors can depend on didSample updates between rows.
        let processorIsStateless = processor == nil || processor is SuppressTokensProcessor
        if let distributionSampler, let draftOutput {
            return appendSelection(
                selectDistributionPreservingTokens(
                    targetLogits: targetLogits, targetStart: targetStart,
                    draftTokens: draftTokens, draftProcessedLogits: draftOutput.processedLogits,
                    numDraft: numDraft, sampler: distributionSampler),
                draftTokens: draftTokens)
        }
        guard processorIsStateless,
            sampler is ArgMaxSampler || sampler is any DistributionLogitSampler
        else {
            return appendSelection(
                selectGreedyTokens(
                    targetLogits: targetLogits, targetStart: targetStart,
                    draftTokens: draftTokens, numDraft: numDraft),
                draftTokens: draftTokens)
        }
        var logits = targetLogits[0, targetStart ..< (targetStart + numDraft + 1), 0...]
        logits = processor?.process(logits: logits) ?? logits
        let result =
            sampler is ArgMaxSampler
            ? batchedGreedyMTPVerification(logits: logits, draftTokens: draftTokens)
            : BatchedMTPVerification.sampled(
                logits: logits, draftTokens: draftTokens, sampler: sampler)
        pendingTokens.append(contentsOf: result.tokens)
        return (result.accepted, result.finalToken)
    }

    /// Appends the tokens a row-by-row verifier walk emitted to the pending buffer.
    ///
    /// - Parameters:
    ///   - selection: The accepted drafts and the tokens the walk emitted, the last of them the
    ///     final token of the round.
    ///   - draftTokens: The drafts the walk verified, evaluated with the emitted tokens.
    /// - Returns: The accepted drafts and the final token of the round.
    private mutating func appendSelection(
        _ selection: (accepted: Int, emitted: [MLXArray]), draftTokens: MLXArray
    ) -> (accepted: Int, finalToken: MLXArray) {
        let emittedTokens = concatenated(selection.emitted)
        eval(emittedTokens, draftTokens)
        // `didSample` already ran inside the selection walk, for emitted tokens only.
        pendingTokens.append(contentsOf: emittedTokens.asArray(Int.self))
        return (selection.accepted, emittedTokens[(selection.emitted.count - 1)...])
    }

    /// Greedy verifier walk. The target is evaluated once for the whole
    /// verify block, then the longest exact candidate prefix is committed and
    /// followed by the target correction/bonus token.
    private mutating func selectGreedyTokens(
        targetLogits: MLXArray,
        targetStart: Int,
        draftTokens: MLXArray,
        numDraft: Int
    ) -> (accepted: Int, emitted: [MLXArray]) {
        // Walk position by position and stop at the first mismatch, so the
        // processor is advanced over emitted tokens only. Sampling the whole
        // block up front and discarding the tail would make a reference-type
        // processor record proposals that never reached the consumer.
        eval(draftTokens)
        let draftTokenList = draftTokens.asArray(Int.self)
        var emitted = [MLXArray]()
        emitted.reserveCapacity(numDraft + 1)

        for i in 0 ..< numDraft {
            var logits = targetLogits[0..., targetStart + i, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            eval(token)
            emitted.append(token)
            processor?.didSample(token: token)
            guard token.item(Int.self) == draftTokenList[i] else {
                return (i, emitted)
            }
        }

        // Every draft matched: the final verifier position predicts one extra
        // target-only bonus token.
        var bonusLogits = targetLogits[0..., targetStart + numDraft, 0...]
        bonusLogits = processor?.process(logits: bonusLogits) ?? bonusLogits
        let bonus = sampler.sample(logits: bonusLogits)
        emitted.append(bonus)
        processor?.didSample(token: bonus)
        return (numDraft, emitted)
    }

    /// Distribution-preserving speculative sampling (Leviathan et al.).
    ///
    /// Candidate `x ~ q` is accepted with `min(1, p(x) / q(x))`. On the first
    /// rejection, the correction is sampled from normalized `max(p - q, 0)`;
    /// if all candidates are accepted, the target bonus is sampled from `p`.
    /// This preserves the target distribution while allowing the target to
    /// verify the entire candidate block in one forward pass.
    private mutating func selectDistributionPreservingTokens(
        targetLogits: MLXArray,
        targetStart: Int,
        draftTokens: MLXArray,
        draftProcessedLogits: MLXArray,
        numDraft: Int,
        sampler distributionSampler: any DistributionLogitSampler
    ) -> (accepted: Int, emitted: [MLXArray]) {
        // As in the greedy walk, the processor advances over emitted tokens
        // only: accepted candidates here, then the correction or bonus.
        let draftTokenList = draftTokens.asArray(Int.self)
        var emitted = [MLXArray]()
        emitted.reserveCapacity(numDraft + 1)

        for i in 0 ..< numDraft {
            let candidateID = draftTokenList[i]
            let candidate = draftTokens[i ..< (i + 1)]

            var pLogits = targetLogits[0..., targetStart + i, 0...]
            pLogits = processor?.process(logits: pLogits) ?? pLogits
            let pLogProbabilities = distributionSampler.logProbabilities(logits: pLogits)
            let qLogits = draftProcessedLogits[0..., i, 0...]
            let qLogProbabilities = distributionSampler.logProbabilities(logits: qLogits)

            let decision = sampleSpeculativeDecision(
                candidateID: candidateID,
                targetLogProbabilities: pLogProbabilities,
                draftLogProbabilities: qLogProbabilities)

            if decision.accepted {
                emitted.append(candidate)
                processor?.didSample(token: candidate)
                continue
            }

            emitted.append(decision.correction)
            processor?.didSample(token: decision.correction)
            return (i, emitted)
        }

        // Every draft was accepted. The final verifier position predicts one
        // extra target-only bonus token.
        var bonusLogits = targetLogits[0..., targetStart + numDraft, 0...]
        bonusLogits = processor?.process(logits: bonusLogits) ?? bonusLogits
        let bonusLogProbabilities = distributionSampler.logProbabilities(logits: bonusLogits)
        let bonus = sampleSpeculative(logProbabilities: bonusLogProbabilities)
        emitted.append(bonus)
        processor?.didSample(token: bonus)
        return (numDraft, emitted)
    }

    /// Draw acceptance and the possible residual correction in one GPU
    /// evaluation. Sampling the correction eagerly (and discarding it on
    /// acceptance) preserves the distribution while avoiding a second or
    /// third CPU/GPU synchronization per candidate.
    private func sampleSpeculativeDecision(
        candidateID: Int,
        targetLogProbabilities p: MLXArray,
        draftLogProbabilities q: MLXArray
    ) -> (accepted: Bool, correction: MLXArray) {
        let logAcceptance = minimum(
            p[0, candidateID] - q[0, candidateID],
            MLXArray(Float(0)))
        let correctionLogProbabilities = speculativeResidualLogProbabilities(
            target: p,
            draft: q)
        let draws = withRandomState(speculativeRandomState) {
            (
                MLXRandom.uniform(Float(0) ..< Float(1), [1]),
                categorical(correctionLogProbabilities)
            )
        }
        let accepted = log(draws.0) .< logAcceptance
        eval(accepted, draws.1)
        return (accepted.item(Bool.self), draws.1)
    }

    private func sampleSpeculative(logProbabilities: MLXArray) -> MLXArray {
        withRandomState(speculativeRandomState) {
            categorical(logProbabilities)
        }
    }

    /// Switch to single-token generation for the remainder of the stream.
    /// Sticky — once flipped, `next()` never returns to speculation.
    private mutating func switchToPassthrough(reason: String) {
        if !passthroughLoggedOnce {
            // Log one-time only so a quantization-onset round doesn't spam.
            // The Swift stdlib `print` is intentional here: the iterator is
            // a low-level component without access to a logger.
            print("[MTPSpeculativeTokenIterator] passthrough mode: \(reason)")
            passthroughLoggedOnce = true
        }
        passthroughReason = reason
        passthrough = true
        discardSpeculativePromptCacheCheckpoints(mainCache)
        mainState?[mtpEmitFlagKey] = false
        mainState?[mtpVerifyKeepsSnapshotsKey] = nil
        mainState?[mtpLastHiddenStatesKey] = nil
        mainState?[mtpSharedKVStatesKey] = nil
        mainState?[mtpSharedKVSourceIndicesKey] = nil
        mainState?[mtpSharedKVOffsetsKey] = nil
        mainState?[mtpPositionDeltasKey] = nil
    }

    /// One single-token forward step against the main model.
    ///
    /// Sticky passthrough leaves the drafter alone. A one-token generation
    /// tail can instead advance stateful drafter state as an empty proposal,
    /// preserving an exact continuation at the final target boundary.
    private mutating func passthroughStep(
        maintainingDrafterState: Bool = false
    ) -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        var stepState = mainState
        if maintainingDrafterState {
            stepState = stepState ?? LMOutput.State()
            stepState?[mtpEmitFlagKey] = true
        }
        let result = mainModel(y[text: .newAxis], cache: mainCache, state: stepState)
        mainState = result.state
        mainCacheStorage.commitProcessedTokens(y.cacheSequenceLength)
        let token = sampleToken(from: result.logits)
        eval(token)
        let tokenInt = token.item(Int.self)

        if maintainingDrafterState,
            let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState,
            let targetHidden = result.state?[mtpLastHiddenStatesKey],
            targetHidden.dim(1) > 0
        {
            statefulDrafter.commitDrafterState(
                target: mainModel,
                targetHidden: targetHidden,
                draftTokens: MLXArray.zeros([0], dtype: token.dtype),
                acceptedCount: 0,
                finalToken: token,
                positionDeltas: result.state?[mtpPositionDeltasKey],
                state: &currentDrafterState,
                sampler: mtpDrafterSeedSampler)
            drafterState = currentDrafterState
            targetBoundaryHidden = targetHidden[0..., (-1)..., 0...]
        } else if maintainingDrafterState,
            drafter is any StatefulMTPDrafterModel
        {
            switchToPassthrough(
                reason:
                    "MTP drafter could not follow the final target-only token; continuation discarded"
            )
        }

        y = .init(tokens: token)
        kvCachePlan.apply(to: mainCacheStorage)
        return tokenInt
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first — before the passthrough branch. A
        // stream that stands down in `init` leaves the prepare-time bonus
        // sitting here, already sampled (`y` holds the token the next forward
        // pass feeds). Short-circuiting to `passthroughStep()` would drop it and
        // start the stream one position ahead of an equivalent autoregressive
        // run. Rounds that flip passthrough mid-stream always clear the buffer
        // first, so this reordering is a no-op for them.
        if pendingIndex < pendingTokens.count {
            return nextPendingToken()
        }

        if passthrough {
            return nextPlainToken()
        }

        // Run a new speculation round (may transition to passthrough).
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        committedPendingTokenCount = 0
        autoreleasepool { speculateRound() }

        if pendingTokens.isEmpty {
            // speculateRound chose passthrough -- fall through.
            return passthrough ? nextPlainToken() : nil
        }

        return nextPendingToken()
    }

    /// Returns the next token of the pending buffer.
    ///
    /// The first ``committedPendingTokenCount`` pending tokens are in the main cache: the
    /// accepted drafts a verify pass fed, or a prefill bonus a priming forward fed. The token
    /// after them waits for the next forward pass.
    private mutating func nextPendingToken() -> Int {
        let token = pendingTokens[pendingIndex]
        hasCachedLastToken = pendingIndex < committedPendingTokenCount
        lastLogProbabilities = deferredLogProbabilities
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }

    /// Decodes one token without speculation, or returns `nil` at the token limit.
    ///
    /// The token is sampled from the forward pass of the token before it, thus no forward
    /// pass has fed it yet.
    private mutating func nextPlainToken() -> Int? {
        guard let token = passthroughStep() else { return nil }
        hasCachedLastToken = false
        lastLogProbabilities = deferredLogProbabilities
        telemetry.recordGeneratedToken()
        return token
    }

}

extension MTPSpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        defer { processor?.finalizeGeneration() }
        // A round retains the recurrent snapshot slots of the tokens it accepted, the
        // checkpoint of an early-finalization rewind, whether it rejected drafts or not.
        // Release them even when no committed lookahead remains.
        var discardedTargetTokens = 0
        defer {
            discardSpeculativePromptCacheCheckpoints(mainCache)
            finalizeDrafterContinuation(discardedTargetTokens: discardedTargetTokens)
        }
        let consumed = Swift.min(pendingIndex, committedPendingTokenCount)
        let lookahead = committedPendingTokenCount - consumed
        guard lookahead > 0 else { return }

        let usesNativeHybridRewind = Self.verifiesInPlace(
            mainModel: mainModel, mainCache: mainCache, drafts: blockSize - 1)
        let rewound: Int
        if usesNativeHybridRewind {
            let trimmed = mainCacheStorage.trim(lookahead)
            rewound = trimmed > 0 ? trimmed : mainCacheStorage.rewindSpeculative(lookahead)
        } else {
            // A staged round keeps what a wrapped ring needs to undo committed lookahead exactly.
            rewound = mainCacheStorage.rewindLastRound(lookahead)
        }
        discardedTargetTokens = rewound
        if rewound == lookahead, let lastVerifyHidden {
            let retainedAccepted = lastVerifyAccepted - lookahead
            targetBoundaryHidden =
                lastVerifyHidden[
                    0...,
                    (lastVerifyStart + retainedAccepted)
                        ..< (lastVerifyStart + retainedAccepted + 1),
                    0...
                ]
        } else if rewound != lookahead {
            // The target cache itself remains authoritative, but there is no
            // exact hidden-state witness from which a drafter may resume.
            targetBoundaryHidden = nil
        }
        // The only site that ignores the result, and the only one that can: the stream is over,
        // so nothing reads the snapshot after this. Everywhere else a refusal stops speculation.
        reconcileSharedKVState(
            &mainState, discarding: rewound,
            lengths: mainCacheStorage.emittedLength(forLeaf:))
    }

    private mutating func finalizeDrafterContinuation(discardedTargetTokens: Int) {
        finalizedContinuation = nil
        guard !passthrough else { return }
        guard let resumableDrafter = drafter as? any ResumableMTPDrafterModel else { return }
        guard mainCacheStorage.processedTokenCount > 0 else { return }
        guard var currentDrafterState = drafterState, let targetBoundaryHidden else {
            passthroughReason =
                "MTP continuation boundary could not be normalized; continuation discarded"
            return
        }

        guard
            resumableDrafter.finalizeDrafterState(
                target: mainModel,
                targetBoundaryHidden: targetBoundaryHidden,
                targetProcessedTokenCount: mainCacheStorage.processedTokenCount,
                discardedTargetTokens: discardedTargetTokens,
                positionDeltas: mainState?[mtpPositionDeltasKey],
                state: &currentDrafterState)
        else {
            passthroughReason =
                "MTP drafter state could not be normalized; continuation discarded"
            return
        }

        drafterState = currentDrafterState
        let continuation = MTPDrafterContinuation(
            state: currentDrafterState,
            targetBoundaryHidden: targetBoundaryHidden,
            targetProcessedTokenCount: mainCacheStorage.processedTokenCount,
            drafterID: resumableDrafter.drafterID)

        // The continuation crosses back from the generation task only after
        // all of its lazy MLX values have been settled.
        eval(continuation.heldArrays)
        finalizedContinuation = continuation
    }
}

package protocol MTPDrafterContinuationProviding {
    var mtpDrafterContinuation: MTPDrafterContinuation? { get }
}

extension MTPSpeculativeTokenIterator: MTPDrafterContinuationProviding {
    package var mtpDrafterContinuation: MTPDrafterContinuation? {
        finalizedContinuation
    }
}

/// Acceptance probability for one candidate in exact speculative sampling.
/// Kept as a scalar helper so boundary cases (`p=0`, `q=0`, and very small
/// ratios) can be tested without requiring a Metal device.
func speculativeAcceptanceProbability(
    targetLogProbability p: Float,
    draftLogProbability q: Float
) -> Double {
    guard q.isFinite else { return 0 }
    guard !p.isNaN else { return 0 }
    if p >= q { return 1 }
    return Foundation.exp(Double(p - q))
}

/// Normalized residual distribution `max(p - q, 0)` used after a rejected
/// speculative candidate. When finite-precision arithmetic collapses the
/// residual mass, `p` is the stable limiting fallback.
func speculativeResidualLogProbabilities(
    target p: MLXArray,
    draft q: MLXArray
) -> MLXArray {
    let residual = maximum(exp(p) - exp(q), MLXArray(Float(0)))
    let residualMass = residual.sum()
    let hasResidual = residualMass .> Float(1e-7)
    let safeMass = maximum(residualMass, MLXArray(Float(1e-7)))
    return MLX.where(hasResidual, log(residual / safeMass), p)
}

extension MTPSpeculativeTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
}

extension MTPSpeculativeTokenIterator {
    /// Test-only setter for the canonical `LogitProcessor`. Lets regression
    /// tests install a recording probe AFTER `init` (which calls `prepare`
    /// and would otherwise consume the prepare-time bonus before the probe
    /// is observable). Used by the emit-only invariant regression tests in
    /// `MTPSpeculativeTokenIteratorTests` (CI-scoped) and
    /// `MTPIteratorEndToEndDiagnosticTests` (31B end-to-end).
    @_spi(Testing) public mutating func _setProcessorForTesting(
        _ processor: LogitProcessor?
    ) {
        self.processor = processor
    }

    /// Test-only getter for the canonical `LogitProcessor` so regression
    /// tests can inspect its post-drain state (e.g., a recording probe's
    /// accumulated didSample log).
    @_spi(Testing) public var _processorForTesting: LogitProcessor? {
        processor
    }
}

/// Remove MTP's transient target↔drafter exchange values while retaining
/// model-owned continuation state such as Qwen VLM M-RoPE anchors.
private func persistentTargetState(from state: LMOutput.State?) -> LMOutput.State? {
    state?.withoutMTPExchange
}

/// Bring the emitted MTP shared-K/V snapshot into line with what the cache actually kept.
///
/// The verify pass emits K/V spanning its whole chunk, materialized before acceptance is known,
/// and a sliding layer is presented more than its window on purpose so every query in the chunk
/// still sees a full one. The drafter has no cache of its own and attends over this snapshot
/// bidirectionally, so both excesses have to go: the rejected tail, then the head beyond what the
/// layer can still describe.
///
/// Order matters. Dropping the tail first and clamping the head second leaves a sliding entry
/// with a full window; clamping first would leave it `discarding` rows short, every round.
///
/// - Parameters:
///   - discarding: trailing positions the commit did not keep.
///   - lengths: for the cache entry behind a given key, how much of the emitted sequence it still
///     describes — the whole stream for a global layer, the trailing window for a sliding one.
///
/// No-op when `state` is nil or carries no snapshot (the quantization-onset round emits none).
/// Returns `false` when the snapshot cannot be placed against cache entries at all, which is a
/// signal to stop speculating rather than reconcile against a guess. Cost is metadata-only: every
/// slice is a lazy view consumed by the next `draftBlock` like the rest of the round's inputs.
@discardableResult
func reconcileSharedKVState(
    _ state: inout LMOutput.State?, discarding: Int, lengths: (Int) -> Int
) -> Bool {
    guard let sharedKV = state?[mtpSharedKVStatesKey] else { return true }
    guard let sources = state?[mtpSharedKVSourceIndicesKey],
        sharedKV.keys.allSatisfy({ sources[$0] != nil })
    else { return false }

    state?[mtpSharedKVStatesKey] = sharedKV.reduce(into: [:]) { result, entry in
        let (key, kv) = entry
        var length = kv.0.dim(-2)
        if discarding > 0 {
            length = Swift.max(0, length - discarding)
        }
        let bound = lengths(sources[key]!)
        let start = Swift.max(0, length - bound)
        result[key] = (
            kv.0[.ellipsis, start ..< length, 0...],
            kv.1[.ellipsis, start ..< length, 0...]
        )
    }
    if discarding > 0, let offsets = state?[mtpSharedKVOffsetsKey] {
        state?[mtpSharedKVOffsetsKey] = offsets.mapValues {
            Swift.max(0, $0 - discarding)
        }
    }
    return true
}
