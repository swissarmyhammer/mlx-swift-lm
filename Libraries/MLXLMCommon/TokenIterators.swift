// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Common properties shared by token-generating iterators.
public protocol TokenIteratorProtocol: Sequence, IteratorProtocol where Element == Int {
    /// The maximum number of tokens to generate, or `nil` for no limit.
    var maxTokens: Int? { get }

    /// The number of tokens generated (or accepted, for speculative iterators) so far.
    var tokenCount: Int { get }

    /// Wall-clock time spent priming the cache with the prompt before the first generated token.
    var promptPrefillTime: TimeInterval { get }
    var state: LMOutput.State? { get }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { get }
    var lastLogProbabilities: DeferredTokenLogProbabilities? { get }
    var evictedTokenCount: Int { get }
    var reasoningTokenCount: Int? { get }

    /// Whether the forward pass of the token the most recent `next()` call returned has
    /// already advanced the cache.
    ///
    /// ``TokenIterator`` runs that pass before it returns the token. A speculative iterator
    /// returns the final token of a round before any pass feeds it. The generation loop
    /// reports a suppressed stop token in ``GenerateCompletionInfo/stopTokenFedToCache``
    /// only when this is `true`. The default is `true`.
    var hasCachedLastToken: Bool { get }

    /// Records a token included in the generated completion.
    mutating func recordEmittedToken(_ token: Int)

    /// Discards the most recently produced token (e.g. a terminal stop token whose
    /// forward pass already advanced the cache) without emitting it downstream.
    mutating func discardGeneratedToken()
}

/// Internal lifecycle capability for iterators that retain generation work
/// which must be reconciled after the token loop stops.
protocol GenerationFinalizingTokenIterator: TokenIteratorProtocol {
    mutating func finalizeGeneration()
}

extension TokenIteratorProtocol {
    /// Default: `nil`, for iterators that keep no model state.
    public var state: LMOutput.State? { nil }

    /// Default: `nil`, for iterators that do not speculate.
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { nil }

    /// Default: `nil`, for iterators that report no log probabilities.
    public var lastLogProbabilities: DeferredTokenLogProbabilities? { nil }

    /// Default: zero, for iterators whose caches evict nothing.
    public var evictedTokenCount: Int { 0 }

    /// Default: `nil`, for iterators that count no reasoning tokens.
    public var reasoningTokenCount: Int? { nil }

    /// Default: `true`, for iterators that feed each token before they return it.
    public var hasCachedLastToken: Bool { true }

    /// Default: a no-op, for iterators with no processor to tell.
    public mutating func recordEmittedToken(_ token: Int) {}

    /// Default: a no-op, for iterators with nothing to discard.
    public mutating func discardGeneratedToken() {}

    /// Whether the iterator has generated every token its limit allows.
    var isAtTokenLimit: Bool {
        guard let maxTokens else { return false }
        return tokenCount >= maxTokens
    }
}

/// A token iterator whose logit processor counts reasoning tokens and learns of emitted tokens.
protocol TokenIteratorProcessing: TokenIteratorProtocol {
    /// The logit processor of the stream.
    var processor: LogitProcessor? { get set }
}

extension TokenIteratorProcessing {
    /// The reasoning tokens the processor counted, or `nil` when it counts none.
    public var reasoningTokenCount: Int? {
        (processor as? any GenerationReasoningTokenCounting)?.generationReasoningTokenCount
    }

    /// Tells the processor that the stream emitted `token`.
    public mutating func recordEmittedToken(_ token: Int) {
        processor?.didEmit(token: token)
    }
}

/// Normalized logits shared by reporting and filtered sampling.
struct LogProbabilityDistribution {
    let values: MLXArray

    init(logits: MLXArray) {
        self.values = logSoftmax(upcastingBFloat16(logits))
    }

    func recording(token: MLXArray, topK: Int) -> DeferredTokenLogProbabilities {
        DeferredTokenLogProbabilities(logProbabilities: values, token: token, topK: topK)
    }
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}

/// `count` limited to `0...limit`.
func clampedCount(_ count: Int, to limit: Int) -> Int {
    Swift.min(Swift.max(count, 0), limit)
}

/// Log probabilities for one generated token that are still on the GPU.
///
/// ``TokenIterator`` evaluates them with the decode step that samples the token.
/// Handlers that ignore them skip the GPU-to-host copy. MLX arrays are not
/// `Sendable`, so call ``materialize()`` on the generation worker and send the result.
public struct DeferredTokenLogProbabilities {
    let token: MLXArray
    let chosenLogProbability: MLXArray
    let topTokenIDs: MLXArray?
    let topLogProbabilities: MLXArray?

    init(logProbabilities: MLXArray, token: MLXArray, topK: Int) {
        self.token = token
        // Cast to the read types in the graph: a cast at read time is new GPU
        // work that waits behind the decode step already in flight.
        let tokenIndices = token[0..., .newAxis]
        self.chosenLogProbability = takeAlong(logProbabilities, tokenIndices, axis: -1)
            .asType(.float32)

        let count = clampedCount(topK, to: logProbabilities.dim(-1))
        guard count > 0 else {
            self.topTokenIDs = nil
            self.topLogProbabilities = nil
            return
        }

        let unorderedTokenIDs: MLXArray
        if count == logProbabilities.dim(-1) {
            unorderedTokenIDs = argSort(-logProbabilities, axis: -1)
        } else {
            unorderedTokenIDs =
                argPartition(
                    -logProbabilities, kth: count - 1, axis: -1)[.ellipsis, ..<count]
        }
        let unorderedLogProbabilities = takeAlong(
            logProbabilities, unorderedTokenIDs, axis: -1)
        let order = argSort(-unorderedLogProbabilities, axis: -1)
        self.topTokenIDs = takeAlong(unorderedTokenIDs, order, axis: -1).asType(.int32)
        self.topLogProbabilities = takeAlong(unorderedLogProbabilities, order, axis: -1)
            .asType(.float32)
    }

    var arrays: [MLXArray] {
        [chosenLogProbability, topTokenIDs, topLogProbabilities].compactMap { $0 }
    }

    /// Copies the values from the GPU into a `Sendable` value.
    public func materialize() -> GenerateTokenLogProbabilities {
        let materializedTopLogProbabilities: [GenerateTokenLogProbability]
        if let topTokenIDs, let topLogProbabilityValues = topLogProbabilities {
            materializedTopLogProbabilities = zip(
                topTokenIDs.asArray(Int32.self),
                topLogProbabilityValues.asArray(Float.self)
            )
            .map { token, logProbability in
                GenerateTokenLogProbability(token: Int(token), logProbability: logProbability)
            }
        } else {
            materializedTopLogProbabilities = []
        }

        return GenerateTokenLogProbabilities(
            chosen: .init(
                token: token.item(Int.self),
                logProbability: chosenLogProbability.item(Float.self)),
            topLogProbabilities: materializedTopLogProbabilities)
    }
}

/// Draws a token from `logits` as the processor shapes them, and tells the processor of it.
///
/// - Parameters:
///   - logits: The logits of one position.
///   - processor: The logit processor, or `nil`.
///   - draw: Draws the token from the processed logits.
/// - Returns: The token.
private func drawToken(
    logits: MLXArray, processor: inout LogitProcessor?, draw: (MLXArray) -> MLXArray
) -> MLXArray {
    let logits = processor?.process(logits: logits) ?? logits
    let token = draw(logits)
    processor?.didSample(token: token)
    return token
}

/// Samples a token from `logits` as the processor shapes them, and tells the processor of it.
///
/// - Parameters:
///   - logits: The logits of one position.
///   - processor: The logit processor, or `nil`.
///   - sampler: The sampler.
/// - Returns: The token.
private func sampleToken(
    logits: MLXArray, processor: inout LogitProcessor?, sampler: LogitSampler
) -> MLXArray {
    drawToken(logits: logits, processor: &processor) { sampler.sample(logits: $0) }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: generateParameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: TokenIteratorProcessing {
    let model: any LanguageModel

    /// Per-call model state (e.g. M-RoPE rope deltas), seeded from the
    /// initializer's `state`, replaced by prefill, then threaded through
    /// every decode step. A caller continuing this cache later (as
    /// ``ChatSession`` does across turns) reads it back after
    /// initialization and seeds the next iterator with it.
    public internal(set) var state: LMOutput.State?

    /// The tokens the next forward pass feeds the model.
    var pendingInput: LMInput.Text
    let cacheStorage: KVCacheStorage
    var cache: [KVCache] {
        get { cacheStorage.cache }
        set { cacheStorage.replace(with: newValue) }
    }
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let logProbabilityTopK: Int?
    private var deferredLogProbabilities: DeferredTokenLogProbabilities?

    /// Log probabilities for the token returned by the most recent ``next()`` call.
    public private(set) var lastLogProbabilities: DeferredTokenLogProbabilities?

    /// The number of tokens generated so far.
    public var tokenCount = 0

    /// The cadence that returns the freed buffers of the MLX pool to the system.
    var bufferPoolClearing = GenerationBufferPoolClearing()

    /// The maximum number of tokens to generate, or `nil` for no limit.
    public let maxTokens: Int?

    public var evictedTokenCount: Int {
        KVCacheTree.maxEvictedTokenCount(in: cache)
    }

    var kvCachePlan: KVCachePlan { cacheStorage.plan }

    /// Wall-clock time spent priming the cache with the prompt before the first generated token.
    public var promptPrefillTime: TimeInterval = 0.0

    /// Why the MTP drafter the generation was configured with does not speculate in this
    /// iterator, or `nil` when the generation named no drafter that falls back. The generation
    /// loop reports it on the completion information.
    package var speculationFallback: MTPDrafterFallback?

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:state:parameters:components:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters, components: GenerationComponents = .init()
    ) throws {
        let plan = try parameters.kvCachePlan()
        try self.init(
            input: .init(text: .init(tokens: prompt)), model: model,
            cacheStorage: KVCacheStorage(
                cache ?? (try model.newCache(parameters: parameters)), plan: plan),
            parameters: parameters, components: components)
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:state:processor:sampler:prefill:maxTokens:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - state: optional per-call model state carried over from earlier
    ///     evaluation against `cache` (e.g. by a caller resuming a session)
    ///   - parameters: the generation parameters
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil,
        parameters: GenerateParameters, components: GenerationComponents = .init()
    ) throws {
        let plan = try parameters.kvCachePlan()
        try self.init(
            input: input, model: model,
            cacheStorage: KVCacheStorage(
                cache ?? (try model.newCache(parameters: parameters)), plan: plan),
            state: state, parameters: parameters, components: components)
    }

    package init(
        input: LMInput, model: any LanguageModel,
        cacheStorage: KVCacheStorage,
        state: LMOutput.State? = nil,
        parameters: GenerateParameters,
        components: GenerationComponents = .init()
    ) throws {
        let kvCachePlan = cacheStorage.plan
        let cacheStorage = try kvCachePlan.validated(cacheStorage)

        self.model = model
        self.state = state
        self.pendingInput = input.text
        self.cacheStorage = cacheStorage

        try components.validate(parameters: parameters)
        self.processor = components.logitProcessor(parameters: parameters, model: model)
        self.sampler = parameters.makeSampler()
        self.logProbabilityTopK = parameters.logProbabilities.map { Swift.max($0, 0) }
        self.maxTokens = parameters.maxTokens

        self.promptPrefillTime = try measure {
            try prepare(input: input, prefill: parameters.prefill)
        }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - state: optional per-call model state carried over from earlier
    ///     evaluation against `cache` (e.g. by a caller resuming a session)
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefill: prefill parameters (step size, chunking, progress)
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil,
        processor: LogitProcessor?, sampler: LogitSampler,
        prefill: PrefillParameters = .init(),
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.state = state
        self.pendingInput = input.text
        self.cacheStorage = KVCacheStorage(
            try cache ?? model.newCache(parameters: nil), plan: .disabled)

        self.processor = processor
        self.sampler = sampler
        self.logProbabilityTopK = nil
        self.maxTokens = maxTokens

        self.promptPrefillTime = try measure {
            try prepare(input: input, prefill: prefill)
        }
    }

    @available(
        *, deprecated,
        renamed: "init(input:model:cache:state:processor:sampler:prefill:maxTokens:)",
        message:
            "prefill now defaults to balanced chunking; use prefill.chunking = .remainder for the legacy chunk boundaries"
    )
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int?,
        maxTokens: Int? = nil
    ) throws {
        try self.init(
            input: input, model: model, cache: cache, state: state,
            processor: processor, sampler: sampler,
            prefill: .init(stepSize: prefillStepSize), maxTokens: maxTokens)
    }

    mutating func prepare(input: LMInput, prefill: PrefillParameters = .init()) throws {
        processor?.prompt(input.text.tokens)
        prefill.prefix?(cacheStorage.processedTokenCount)
        let result = try model.prepare(input, cache: cache, state: state, prefill: prefill)
        cacheStorage.commitPrefill(of: input, result: result, role: "LanguageModel.prepare")
        switch result {
        case .tokens(let tokens):
            pendingInput = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: pendingInput)
            pendingInput = .init(tokens: token)
            evaluateTokenAndLogProbabilities(pendingInput.tokens)

            // the model reported per-chunk progress; the remainder it left to us
            // completes the prompt (models returning .logits report their own terminal)
            let total = input.text.tokens.size
            prefill.progress?(total, total)

        case .logits(let output):
            // carry the prefill state into decode, as step(previous:) does for later steps
            self.state = output.state
            pendingInput = .init(tokens: convertToToken(logits: output.logits))
            evaluateTokenAndLogProbabilities(pendingInput.tokens)

            break
        }

        try kvCachePlan.applyAndValidate(to: cacheStorage)
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        let sampler = sampler
        let logProbabilityTopK = logProbabilityTopK
        var logProbabilities: DeferredTokenLogProbabilities?
        let token = drawToken(logits: logits[0..., -1, 0...], processor: &processor) { logits in
            guard let logProbabilityTopK else { return sampler.sample(logits: logits) }
            // Filtered sampling already normalizes logits, so reuse that distribution.
            let distribution = LogProbabilityDistribution(logits: logits)
            let token =
                (sampler as? TopPSampler)?.sample(logProbabilities: distribution.values)
                ?? sampler.sample(logits: logits)
            logProbabilities = distribution.recording(token: token, topK: logProbabilityTopK)
            return token
        }
        deferredLogProbabilities = logProbabilities
        return token
    }

    /// Schedules a token and the cache state: the first token at the end of
    /// prefill, and each next token in ``next()``.
    ///
    /// The cache state goes with the token. Some caches hold
    /// state that the token does not read (the DeepSeek-V4 compressor rows, for
    /// example). That state must not stay lazy: decode runs on the
    /// `GenerationWorker` thread, and MLX cannot evaluate a graph that names the
    /// stream of the prefill thread. ``KVCache/evaluationState`` holds every
    /// lazy array a cache keeps.
    private func evaluateTokenAndLogProbabilities(_ token: MLXArray) {
        let cacheState = cache.flatMap { $0.evaluationState }
        if let deferredLogProbabilities {
            asyncEval([token] + cacheState + deferredLogProbabilities.arrays)
        } else {
            asyncEval([token] + cacheState)
        }
    }

    /// Evaluate the next token and return it, updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = withPreparedCache(cache, lengths: previous.sequenceLengths) {
            model.nextTokenLogits(
                previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        }
        cacheStorage.commitProcessedTokens(previous.cacheSequenceLength)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        kvCachePlan.apply(to: cacheStorage)

        return convertToToken(logits: result.logits)
    }

    /// Returns the previously computed token while asynchronously prefetching
    /// the next one, pipelining the model's forward pass with the caller's
    /// consumption of the prior token. Returns `nil` once `maxTokens` has
    /// been reached.
    mutating public func next() -> Int? {
        guard !isAtTokenLimit else { return nil }

        // Drain autoreleased MLX temporaries every step: a full model forward
        // produces hundreds of autoreleased wrapper objects, and the token
        // loop never returns to a pool boundary on its own; without this,
        // long generations grow host memory without bound.
        return autoreleasepool {
            // save current value -- this will be returned
            let previousInput = pendingInput
            let previousLogProbabilities = deferredLogProbabilities

            // compute the next state and async eval the next token
            let token = step(previous: previousInput)
            pendingInput = .init(tokens: token)
            // Evaluate the cache state together with the token: caches update
            // through functional ops (concatenation, slice assignment), and an
            // unevaluated chain of those updates keeps every prior step's
            // intermediates alive. Python mlx-lm settles cache state the same
            // way in its generation loop. `evaluationState` gives the arrays a
            // cache already holds, thus this adds no operation to the step.
            evaluateTokenAndLogProbabilities(token)

            bufferPoolClearing.clearPoolIfDue(tokenCount: tokenCount)

            tokenCount += 1

            lastLogProbabilities = previousLogProbabilities
            return previousInput.tokens.item(Int.self)
        }
    }
}

extension TokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        processor?.finalizeGeneration()
    }
}

/// Generator of tokens using speculative decoding.
///
/// This is typically used via a call to ``generate(input:cache:state:parameters:context:draftModel:draftCache:numDraftTokens:components:wiredMemoryTicket:tools:)``
/// returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let mainModel: LanguageModel
/// let draftModel: LanguageModel
///
/// let iterator = try SpeculativeTokenIterator(
///     input: input, mainModel: mainModel, draftModel: draftModel,
///     parameters: generateParameters, numDraftTokens: 2)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `speculative_generate_step()` from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
public struct SpeculativeTokenIterator: TokenIteratorProcessing {

    /// The tokens the next forward pass feeds the main (verifier) model.
    var mainPendingInput: LMInput.Text

    /// The tokens the next forward pass feeds the draft model.
    var draftPendingInput: LMInput.Text

    let mainModel: any LanguageModel
    let draftModel: any LanguageModel

    /// Model state carried by the main model, as ``TokenIterator/state``.
    ///
    /// Seeded from `mainState` and updated by prefill and each verification pass. Read it from
    /// the same iterator value you generate with: this is a struct, so a copy would not observe
    /// later mutations.
    ///
    /// > Note: a rejected proposal trims KV rows, so only state a decode step does not rewrite
    /// > survives speculation. The vision models' M-RoPE anchors qualify — they position from the
    /// > cache offset, which the same trim rewinds.
    public internal(set) var state: LMOutput.State?
    let mainCacheStorage: KVCacheStorage
    let draftCacheStorage: KVCacheStorage
    var mainCache: [KVCache] {
        get { mainCacheStorage.cache }
        set { mainCacheStorage.replace(with: newValue) }
    }
    var draftCache: [KVCache] {
        get { draftCacheStorage.cache }
        set { draftCacheStorage.replace(with: newValue) }
    }
    var kvCachePlan: KVCachePlan { mainCacheStorage.plan }

    var processor: LogitProcessor?
    let sampler: LogitSampler

    /// The number of tokens accepted (emitted) so far, sourced from `telemetry`.
    public var tokenCount: Int { telemetry.emittedTokenCount }

    /// The cadence that returns the freed buffers of the MLX pool to the system.
    var bufferPoolClearing = GenerationBufferPoolClearing()

    /// The maximum number of tokens to generate, or `nil` for no limit.
    public let maxTokens: Int?
    let numDraftTokens: Int

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens: [Int] = []
    private var pendingIndex = 0
    /// Pending tokens already represented by each cache. The final token in a
    /// speculative round is sampled from the verifier logits but is not fed
    /// back into either model yet, so it is deliberately excluded.
    private var mainCommittedPendingTokenCount = 0
    private var draftCommittedPendingTokenCount = 0

    /// Whether the main cache held the last token ``next()`` returned when it returned it.
    ///
    /// `true` for an accepted draft, which the verify pass of its round fed. The final token
    /// of a round, the first token that the main prefill logits gave, and each token of a
    /// plain round are returned before any forward pass feeds them.
    public var hasCachedLastToken: Bool {
        pendingIndex > 0 && pendingIndex - 1 < mainCommittedPendingTokenCount
    }

    /// Why speculation stopped and the rest of the stream decodes one token at a time, or `nil`
    /// while rounds are still speculative. Engaged when a cache can neither stage nor trim a
    /// round: generation stays correct, it just stops speculating.
    private(set) var passthroughReason: String?

    /// Wall-clock time spent priming both caches with the prompt before the first generated token.
    public var promptPrefillTime: TimeInterval = 0.0
    private var telemetry = SpeculativeDecodingTelemetry()

    /// Speculative-decoding round/acceptance telemetry, or `nil` until at least one
    /// draft-then-verify round has run.
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }

    public var evictedTokenCount: Int {
        KVCacheTree.maxEvictedTokenCount(in: mainCache)
    }

    /// Un-counts the most recently emitted token in `telemetry` without altering any cache state.
    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    /// Initialize a `SpeculativeTokenIterator` with the given input.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - mainModel: the main (verifier) ``LanguageModel``
    ///   - draftModel: the draft ``LanguageModel`` (must share the same tokenizer)
    ///   - mainCache: optional ``KVCache`` for the main model
    ///   - draftCache: optional ``KVCache`` for the draft model
    ///   - mainState: optional model state that belongs with `mainCache`
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: number of tokens the draft model proposes per round
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        mainState: LMOutput.State? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        components: GenerationComponents = .init()
    ) throws {
        let plan = try parameters.kvCachePlan()
        try self.init(
            input: input, mainModel: mainModel, draftModel: draftModel,
            mainCacheStorage: KVCacheStorage(
                mainCache ?? (try mainModel.newCache(parameters: parameters)), plan: plan),
            draftCacheStorage: KVCacheStorage(
                draftCache ?? (try draftModel.newCache(parameters: parameters)), plan: plan),
            mainState: mainState,
            parameters: parameters, numDraftTokens: numDraftTokens,
            components: components)
    }

    package init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCacheStorage: KVCacheStorage,
        draftCacheStorage: KVCacheStorage,
        mainState: LMOutput.State? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        components: GenerationComponents = .init()
    ) throws {
        let kvCachePlan = mainCacheStorage.plan
        precondition(
            draftCacheStorage.plan == kvCachePlan,
            "Speculative caches must use the same KV-cache plan")
        let mainCacheStorage = try kvCachePlan.validated(mainCacheStorage)
        let draftCacheStorage = try kvCachePlan.validated(draftCacheStorage)
        guard mainCacheStorage.processedTokenCount == draftCacheStorage.processedTokenCount else {
            throw KVCacheError(
                message: "Speculative caches must represent the same processed-token position.")
        }
        guard canTrimPromptCache(draftCacheStorage.cache) else {
            throw KVCacheError(
                message: "Speculative decoding requires a trimmable draft KV cache.")
        }
        // The main cache rewinds rejected drafts by trimming, or through a staged round once a
        // rotating cache has wrapped and stopped being trimmable. Probe with a round at the
        // width rounds will use, as `MTPSpeculativeTokenIterator` does, so this check cannot
        // drift from the leaf classification.
        if !canTrimPromptCache(mainCacheStorage.cache) {
            guard
                let probe = mainCacheStorage.beginRound(
                    maximumPositions: numDraftTokens + 1)
            else {
                throw KVCacheError(
                    message:
                        "Speculative decoding requires a trimmable or stageable main KV cache.")
            }
            mainCacheStorage.rollback(probe)
        }

        self.mainPendingInput = input.text
        self.draftPendingInput = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel

        self.mainCacheStorage = mainCacheStorage
        self.draftCacheStorage = draftCacheStorage
        self.state = mainState

        self.sampler = parameters.makeSampler()
        try components.validate(parameters: parameters)
        self.processor = components.logitProcessor(parameters: parameters, model: mainModel)

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens

        self.promptPrefillTime = try measure {
            try prepare(input: input, prefill: parameters.prefill)
        }
    }

    /// Prefill both main and draft models with the prompt, priming caches for generation
    mutating func prepare(input: LMInput, prefill: PrefillParameters = .init()) throws {
        processor?.prompt(input.text.tokens)
        prefill.prefix?(mainCacheStorage.processedTokenCount)

        let mainResult = try Self.prefill(
            mainModel, with: input, into: mainCacheStorage, state: state, prefill: prefill,
            role: "Main")
        switch mainResult {
        case .tokens:
            // the remaining tokens are consumed by the first verify pass; the
            // prompt is as processed as prefill will make it
            let total = input.text.tokens.size
            prefill.progress?(total, total)
        case .logits(let output):
            state = output.state
        }

        // The draft prefill gets no progress callback: the main model's prefill
        // already reported the prompt, and a second pass would double-count it.
        var draftPrefill = prefill
        draftPrefill.progress = nil
        let draftResult = try Self.prefill(
            draftModel, with: input, into: draftCacheStorage, state: nil, prefill: draftPrefill,
            role: "Draft")

        startDecoding(main: mainResult, draft: draftResult)

        try kvCachePlan.applyAndValidate(to: mainCacheStorage)
        try kvCachePlan.applyAndValidate(to: draftCacheStorage)
    }

    /// Runs the `prepare` of `model` on the prompt, and records in `storage`
    /// the prompt tokens that the model processed.
    ///
    /// - Parameters:
    ///   - model: the model to prefill.
    ///   - input: the prompt.
    ///   - storage: the cache storage of `model`.
    ///   - state: the model state that belongs with the cache of `storage`.
    ///   - prefill: the prefill parameters.
    ///   - role: the name of the model in the precondition message.
    /// - Returns: the result of the `prepare` of `model`.
    private static func prefill(
        _ model: any LanguageModel, with input: LMInput, into storage: KVCacheStorage,
        state: LMOutput.State?, prefill: PrefillParameters, role: String
    ) throws -> PrepareResult {
        let result = try model.prepare(
            input, cache: storage.cache, state: state, prefill: prefill)
        storage.commitPrefill(of: input, result: result, role: "\(role) model prepare")
        return result
    }

    /// Sets the first input of each model from the results of the two prefills.
    ///
    /// When both models leave a part of the prompt, the first round processes
    /// it: the draft proposals and the verify pass then start at the same
    /// position. When one model processes the whole prompt, the two models
    /// cannot start at the same position that way. Thus both caches then get
    /// the whole prompt, the main model gives the first token, and each model
    /// starts from that token.
    ///
    /// - Parameters:
    ///   - mainResult: the result of the `prepare` of the main model.
    ///   - draftResult: the result of the `prepare` of the draft model.
    private mutating func startDecoding(
        main mainResult: PrepareResult, draft draftResult: PrepareResult
    ) {
        switch (mainResult, draftResult) {
        case (.tokens(let mainTokens), .tokens(let draftTokens)):
            mainPendingInput = mainTokens
            draftPendingInput = draftTokens
        case (.tokens(let mainTokens), .logits):
            startDecoding(mainLogits: forwardMainModel(mainTokens).logits)
        case (.logits(let mainOutput), .tokens(let draftTokens)):
            feedDraftModel(draftTokens)
            startDecoding(mainLogits: mainOutput.logits)
        case (.logits(let mainOutput), .logits):
            startDecoding(mainLogits: mainOutput.logits)
        }
    }

    /// Samples the first generated token from the main logits of the last
    /// prompt position, and makes it the first pending token and the first
    /// input of each model.
    ///
    /// No cache holds the token yet: the first round feeds it to both models,
    /// as it feeds the final token of each round.
    ///
    /// - Parameter mainLogits: the main logits of the prompt.
    private mutating func startDecoding(mainLogits: MLXArray) {
        let token = sampleToken(
            logits: mainLogits[0..., -1, 0...], processor: &processor, sampler: sampler)
        pendingTokens.append(token.item(Int.self))
        mainPendingInput = .init(tokens: token)
        draftPendingInput = mainPendingInput
    }

    /// Feeds `input` to the main model in one forward, which joins the main cache's processed
    /// tokens, and keeps the model state of that forward.
    ///
    /// - Parameter input: The tokens to feed.
    /// - Returns: The output of the forward.
    private mutating func forwardMainModel(_ input: LMInput.Text) -> LMOutput {
        let output = mainModel(input[text: .newAxis], cache: mainCache, state: state)
        mainCacheStorage.commitProcessedTokens(input.cacheSequenceLength)
        state = output.state
        return output
    }

    /// Run one round of speculative decoding: draft, verify, accept/reject
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else {
            return
        }
        guard passthroughReason == nil else {
            plainRound()
            return
        }

        // Rewinding must be possible *before* anything is written. A staged round commits only
        // the accepted rows, so it stays exact after a rotating cache wraps and can no longer
        // trim; with no round available, fall back to trim only while every leaf can still take
        // back the width. After a `.tokens`-returning `prepare`, `mainPendingInput` is the whole
        // remaining prompt, so the width comes from it rather than from `numDraft` alone.
        let pendingInputSize = mainPendingInput.cacheSequenceLength
        let roundWidth = pendingInputSize + numDraft
        // A cache-less model has nothing to stage or rewind: an empty round
        // would report zero written positions and refuse any commit.
        let round =
            mainCache.isEmpty
            ? nil : mainCacheStorage.beginRound(maximumPositions: roundWidth)
        if round == nil,
            !mainCache.allSatisfy({ $0.isTrimmable(after: roundWidth) })
        {
            passthroughReason = "main KV cache can neither stage nor trim a speculative round"
            plainRound()
            return
        }

        // Draft generation: autoregressive loop with draft model
        var draftProcessor = processor?.copy()  // Copy to discard later
        var draftTokens: [MLXArray] = []
        var draftState: LMOutput.State?
        for _ in 0 ..< numDraft {
            let draftResult = draftModel(
                draftPendingInput[text: .newAxis], cache: draftCache, state: draftState)
            draftCacheStorage.commitProcessedTokens(draftPendingInput.cacheSequenceLength)
            draftState = draftResult.state
            let draftToken = sampleToken(
                logits: draftResult.logits[0..., -1, 0...], processor: &draftProcessor,
                sampler: sampler)
            asyncEval(draftToken)
            draftTokens.append(draftToken)
            draftPendingInput = .init(tokens: draftToken)
        }

        // Verification: main model processes proposals in one pass. With a
        // staged round the model runs against the round's presented caches;
        // the live entries stay untouched until the commit below.
        let verifyTokens = [mainPendingInput.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: round?.caches ?? mainCache, state: state)
        if round == nil {
            mainCacheStorage.commitProcessedTokens(verifyInput.cacheSequenceLength)
        }
        let mainLogits = mainResult.logits
        state = mainResult.state

        let mainTokens: MLXArray
        var verifyProcessor = processor?.copy()
        if verifyProcessor != nil {
            // Process each position sequentially so that the processor sees tokens sampled at earlier positions
            var sampled: [MLXArray] = []
            for i in 0 ..< (numDraft + 1) {
                let token = sampleToken(
                    logits: mainLogits[0..., verifyStart + i, 0...], processor: &verifyProcessor,
                    sampler: sampler)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            // Batch-sample all verify tokens from main model in one operation
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        // Compare and accept proposed tokens
        eval(mainTokens, draftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = concatenated(draftTokens).asArray(Int.self)
        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else {
                break
            }

            processor?.didSample(token: draftTokens[i])
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        // Always emit the main model's token at position `accepted`
        // (either the correction token or the bonus token if all drafts matched)
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])
        mainCommittedPendingTokenCount = accepted
        // When every draft is accepted the draft cache still trails the main
        // cache by one accepted token; `draftPendingInput` carries that token into the
        // next round. Otherwise both caches contain the accepted prefix.
        draftCommittedPendingTokenCount =
            accepted == numDraft ? Swift.max(accepted - 1, 0) : accepted

        telemetry.recordRound(
            drafted: numDraft,
            accepted: accepted,
            targetVerified: numDraft + 1
        )

        // Rewind caches for rejected tokens. Committing a staged round keeps what
        // `mainPendingInput` carried plus the accepted prefix and the correction token, drops the rejected tail without the
        // live caches ever holding it, and advances the processed-token timeline to match.
        if let round {
            mainCacheStorage.commit(round, retaining: pendingInputSize + accepted)
        } else {
            mainCacheStorage.trim(numDraft - accepted)
        }
        let draftTrimRequest = Swift.max(numDraft - accepted - 1, 0)
        if draftCacheStorage.trim(draftTrimRequest) < draftTrimRequest,
            !draftCache.isEmpty
        {
            // A draft cache that cannot rewind (its own window wrapped) would desynchronize
            // future proposals: stop speculating rather than draft from a corrupted history.
            passthroughReason = "draft KV cache could not rewind rejected drafts"
        }

        // Apply dynamic cache quantization after rewind
        kvCachePlan.apply(to: mainCacheStorage)
        kvCachePlan.apply(to: draftCacheStorage)

        // Set the pending inputs for the next round
        mainPendingInput = .init(tokens: finalToken)
        draftPendingInput = .init(tokens: finalToken)

        // If all draft tokens were accepted, the draft model hasn't processed
        // the last accepted draft token yet. Feed it through to keep caches in sync.
        if accepted == numDraft {
            draftPendingInput = .init(
                tokens: concatenated([
                    draftTokens[numDraft - 1].reshaped([1]),
                    finalToken,
                ])
            )
        }
    }

    /// One non-speculative decode step for the sticky fallback: feed the last
    /// sampled token, sample the next, and commit exactly one position — no
    /// rewind is ever needed, so no rewindability is required.
    private mutating func plainRound() {
        let token = sampleToken(
            logits: forwardMainModel(mainPendingInput).logits[0..., -1, 0...],
            processor: &processor, sampler: sampler)
        eval(token)
        pendingTokens.append(token.item(Int.self))
        mainCommittedPendingTokenCount = 0
        draftCommittedPendingTokenCount = 0
        mainPendingInput = .init(tokens: token)
        kvCachePlan.apply(to: mainCacheStorage)
    }

    /// Returns the next accepted token from the current speculative-decoding
    /// round, draining a buffer of tokens already verified against the main
    /// model; runs a new draft-then-verify round via `speculateRound()` once
    /// the buffer is exhausted. Returns `nil` once `maxTokens` has been
    /// reached or a round produces no accepted tokens.
    mutating public func next() -> Int? {
        guard !isAtTokenLimit else { return nil }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            return emitPendingToken()
        }

        // Run a new speculation round. Pooled: a round runs draft + verify
        // forwards whose autoreleased MLX temporaries otherwise accumulate
        // across the whole generation.
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        mainCommittedPendingTokenCount = 0
        draftCommittedPendingTokenCount = 0
        autoreleasepool { speculateRound() }

        if pendingTokens.isEmpty {
            return nil
        }

        return emitPendingToken()
    }

    /// Hands out the next pending token and counts it, on the cadence that keeps a long decode
    /// from filling the MLX pool.
    ///
    /// - Returns: The token.
    private mutating func emitPendingToken() -> Int {
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        bufferPoolClearing.clearPoolIfDue(tokenCount: tokenCount)
        telemetry.recordGeneratedToken()
        return token
    }

}

extension SpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        defer { processor?.finalizeGeneration() }
        // Trim through the storages so the model-wide processed-token timeline
        // rewinds with the caches; `ChatSession` reconciles its ledger against
        // that timeline, not against per-entry offsets.
        let mainKeptTokenCount = Swift.min(pendingIndex, mainCommittedPendingTokenCount)
        let draftKeptTokenCount = Swift.min(pendingIndex, draftCommittedPendingTokenCount)
        let mainLookahead = mainCommittedPendingTokenCount - mainKeptTokenCount
        if mainLookahead > 0 {
            // The staged-round commit recorded how to take back positions a
            // trim cannot (a wrapped sliding-window ring); `rewindLastRound`
            // uses that record and falls back to a plain trim otherwise.
            mainCacheStorage.rewindLastRound(mainLookahead)
        }
        Self.trimLookahead(
            of: draftCacheStorage, beyond: draftKeptTokenCount,
            committed: draftCommittedPendingTokenCount)
        feedDraft(pendingTokens[draftKeptTokenCount ..< mainKeptTokenCount])
    }

    /// Feeds the draft model the returned tokens that the main cache holds and
    /// the draft cache does not, so that the two caches end the generation
    /// aligned.
    ///
    /// After a round that accepted every draft, the draft cache does not hold
    /// the last accepted draft: the next round feeds it through
    /// `draftPendingInput`. When
    /// the loop stops after it took that draft, there is no next round. Without
    /// this step the draft cache stays one token behind the main cache, and
    /// `ChatSession` then cannot reuse the draft cache on the next turn.
    ///
    /// A draft cache that stopped speculating (``passthroughReason``) no longer
    /// follows the main cache, and a cancelled generation is not recorded, thus
    /// neither gets the feed.
    ///
    /// - Parameter tokens: the returned tokens that only the main cache holds.
    private func feedDraft(_ tokens: ArraySlice<Int>) {
        guard !tokens.isEmpty, passthroughReason == nil, !Task.isCancelled else {
            return
        }
        feedDraftModel(LMInput.Text(tokens: MLXArray(Array(tokens))))
        kvCachePlan.apply(to: draftCacheStorage)
        eval(draftCache)
    }

    /// Runs the draft model on `input`, so that the draft cache holds it.
    ///
    /// - Parameter input: the tokens that the draft cache does not hold yet.
    private func feedDraftModel(_ input: LMInput.Text) {
        _ = draftModel(input[text: .newAxis], cache: draftCache, state: nil)
        draftCacheStorage.commitProcessedTokens(input.cacheSequenceLength)
    }

    /// Removes the tokens of the current round that `storage` holds after the
    /// first `keptTokenCount` tokens.
    ///
    /// - Parameters:
    ///   - storage: the cache storage to trim.
    ///   - keptTokenCount: the count of pending tokens that the cache keeps.
    ///   - committedTokenCount: the count of pending tokens that the cache holds.
    private static func trimLookahead(
        of storage: KVCacheStorage, beyond keptTokenCount: Int, committed committedTokenCount: Int
    ) {
        let lookahead = committedTokenCount - keptTokenCount
        if lookahead > 0 {
            storage.trim(lookahead)
        }
    }
}
