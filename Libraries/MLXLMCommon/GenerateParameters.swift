// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A sampler that can also expose the normalized distribution it samples.
///
/// Speculative sampling needs both the target distribution `p` and the
/// drafter distribution `q` to apply the acceptance probability
/// `min(1, p(token) / q(token))` and, on rejection, sample from the residual
/// distribution `max(p - q, 0)`. Plain ``LogitSampler`` intentionally keeps
/// the smaller API used by ordinary generation; samplers that can support the
/// exact speculative algorithm opt in through this refinement.
public protocol DistributionLogitSampler: LogitSampler {
    /// Normalized log-probabilities after applying the sampler's temperature
    /// and probability filters. The returned tensor has the same shape as
    /// `logits` and sums to one in probability space along the last axis.
    func logProbabilities(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let token = sampler.sample(logits: logits)
/// processor?.didSample(token: token)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor {

    /// Called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// Called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// Called to provide the sampled token
    mutating func didSample(token: MLXArray)

    /// Called when a sampled token is emitted by the generation stream.
    mutating func didEmit(token: Int)

    /// Called after the generation stream has emitted its last token.
    mutating func finalizeGeneration()

    /// Returns an independent copy of this processor.
    ///
    /// Value types (structs) obtain an independent copy via standard value semantics
    /// by default. Reference types (classes) must explicitly implement this method to
    /// produce a distinct instance; classes that do not provide an implementation trap.
    func copy() -> Self
}

extension LogitProcessor {
    /// Default: this value, which is an independent copy for a value type.
    public func copy() -> Self {
        self
    }

    /// Default: a no-op, for processors that keep no record of emitted tokens.
    public mutating func didEmit(token: Int) {}

    /// Default: a no-op, for processors with nothing to settle at the end of a generation.
    public mutating func finalizeGeneration() {}
}

extension LogitProcessor where Self: AnyObject {
    /// Default for a reference type: traps, as a class must write its own copy.
    public func copy() -> Self {
        fatalError(
            """
            \(Self.self) is a reference type conforming to LogitProcessor but does not implement copy(). \
            Reference-type processors must explicitly implement copy() to support isolated state scoping \
            in speculative decoding and verification passes.
            """
        )
    }
}

/// The log probability assigned to one vocabulary token.
public struct GenerateTokenLogProbability: Sendable, Equatable {
    /// The vocabulary token ID.
    public let token: Int

    /// The token's log probability after logit processing and before sampling.
    public let logProbability: Float

    public init(token: Int, logProbability: Float) {
        self.token = token
        self.logProbability = logProbability
    }
}

/// Log-probability details for one generated token.
///
/// These values match Python MLX-LM's `GenerationResponse.logprobs`: they
/// normalize the post-processor logits before temperature or sampler filters
/// are applied.
public struct GenerateTokenLogProbabilities: Sendable, Equatable {
    /// The token selected by the sampler and its log probability.
    public let chosen: GenerateTokenLogProbability

    /// The most likely vocabulary tokens in descending log-probability order.
    ///
    /// This can include ``chosen`` when the selected token is among the requested
    /// top candidates.
    public let topLogProbabilities: [GenerateTokenLogProbability]

    public init(
        chosen: GenerateTokenLogProbability,
        topLogProbabilities: [GenerateTokenLogProbability] = []
    ) {
        self.chosen = chosen
        self.topLogProbabilities = topLogProbabilities
    }
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.
public struct GenerateParameters: Sendable {

    /// How the prompt is prefilled into the cache: step size, chunking strategy,
    /// and progress observation. See ``PrefillParameters``.
    public var prefill: PrefillParameters

    /// See ``PrefillParameters/stepSize``.
    @available(*, deprecated, renamed: "prefill.stepSize")
    public var prefillStepSize: Int? {
        get { prefill.stepSize }
        set { prefill.stepSize = newValue }
    }

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries are overwritten while retaining
    /// up to the first 4 tokens when the requested size permits.
    /// The value must be greater than zero; generation and cache construction throw
    /// ``KVCacheConfigurationError/invalidCapacity(_:)`` for an invalid value.
    ///
    /// When set, full-attention layers that would otherwise use ``KVCacheSimple`` use
    /// ``RotatingKVCache`` instead. Architecture sliding-window layers use the smaller
    /// of their model-defined window and this limit. State-space / Mamba / GDN layers
    /// are not token-windowed and therefore do not consume this budget.
    ///
    /// Inspect the effective policy with ``LanguageModel/cacheStatus(parameters:)``
    /// (also on ``ModelContainer`` / ``ChatSession``). This is a cache policy, not a
    /// generation stop reason — output token limits still surface as ``GenerateStopReason/length``.
    public var maxKVSize: Int?

    /// Typed key-value cache configuration.
    ///
    /// When set, this is the canonical cache configuration. Do not combine it
    /// with `maxKVSize`, `kvBits`, `kvGroupSize`, `quantizedKVStart`, or
    /// `kvScheme`: generation and cache construction throw
    /// ``KVCacheConfigurationError/conflictingLegacyConfiguration`` when one of
    /// them is also set (`kvGroupSize` and `quantizedKVStart` count as set when
    /// they differ from their defaults).
    public var kvCache: KVCacheConfiguration?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// KV cache compression scheme. Overrides kvBits when set.
    ///
    /// Built-in affine: "affine4", "affine8" (equivalent to kvBits 4/8).
    ///
    /// TurboQuant schemes (`turbo<K>v<V>`, keys × values). Start light,
    /// verify quality on your model, then ratchet V:
    /// - "turbo0v4"  FP16 K + 4-bit V, the safest start
    /// - "turbo0v3"/"turbo0v2"  FP16 K, more aggressive V
    /// - "turbo8v4"  8-bit affine K + 4-bit V, conservative asymmetric
    /// - "turbo8v3"  recommended default (near-lossless K, compressed V)
    /// - "turbo8v2"  aggressive V (boundary-layer protection auto-engages)
    /// - "turbo4"/"turbo4v2"/"turbo3"/"turbo2"  turbo-quantized keys:
    ///   maximum compression; K sensitivity varies by model family, so
    ///   validate on your model (asym is the recommended starting point).
    ///
    /// Variance-normalized (KVarN-inspired) schemes for memory-bound long context:
    /// - "varn" / "varn4v2"  4-bit K + 2-bit V, 128-token tiles
    /// - "varn4" / "varn4v4"  4-bit K/V
    /// - "varn2" / "varn2v2"  2-bit K/V
    /// - "varn4v2t32" / "varn4v2t64"  explicit tile size variants
    ///
    /// Unrecognized schemes are rejected when generation starts. Prefer
    /// ``kvCache`` for compile-time-safe configuration.
    public var kvScheme: String?

    /// Sampling temperature
    public var temperature: Float

    /// Top-p sampling
    public var topP: Float

    /// Top-k sampling (0 disables)
    public var topK: Int

    /// Min-p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// Per-token log-probability reporting. `nil` disables reporting; `0` reports
    /// the selected token only, and a positive value also reports that many top tokens.
    /// Use ``LogProbabilityTokenLoopHandler`` to emit these values through a custom generation loop.
    /// Only ``TokenIterator`` reports them; speculative iterators ignore this setting.
    public var logProbabilities: Int?

    /// Optional seed for reproducible sampling. When set, the sampler's RNG
    /// (`TopPSampler` / `CategoricalSampler`) is seeded deterministically, so
    /// the same `(seed, prompt, parameters)` produces the same sampled
    /// tokens. `nil` ⇒ the sampler is seeded from system entropy (the prior
    /// default). Inert at `temperature == 0` (argmax has no RNG).
    public var seed: UInt64?

    /// How an MTP stream with a stateful drafter (Qwen MTP) verifies its drafts when it
    /// samples. Inert at `temperature == 0`, where the drafts and the verifier are argmax.
    public var mtpSampledVerification: MTPSampledVerification = .sampleAndMatch

    /// The context a stateful MTP drafter (Qwen MTP) attends to on each draft: its newest
    /// `maxTokens` tokens, with the first `preservedPrefixTokens` kept, or the whole context when
    /// `nil`.
    ///
    /// The drafter only proposes tokens and the target verifies them, so the window changes how
    /// many drafts the target accepts, never the tokens. Below the window the drafter attends to
    /// the whole context. A rotating main cache (``maxKVSize``, a ``KVCacheConfiguration``
    /// capacity) narrower than the window bounds the drafter to the main window. A drafter
    /// continuation saved with one window resumes only in a stream of the same window; another
    /// stream starts its drafter at the cached position.
    public var mtpDrafterAttentionWindow: KVCacheConfiguration.Capacity? =
        SpeculativeDecodingConfig.defaultMTPDrafterAttentionWindow

    /// Penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// Number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    /// Recovery and validation rules for generated tool calls.
    public var toolCallPolicy: ToolCallPolicy

    /// Creates generation parameters, filling in mlx-lm-matching defaults for
    /// sampling (`temperature`, `topP`/`topK`/`minP`), penalties, and KV-cache
    /// quantization for any argument not supplied.
    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvCache: KVCacheConfiguration? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        kvScheme: String? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        logProbabilities: Int? = nil,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefill: PrefillParameters = .init(),
        seed: UInt64? = nil,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvCache = kvCache
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.kvScheme = kvScheme
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.logProbabilities = logProbabilities
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefill = prefill
        self.seed = seed
        self.toolCallPolicy = toolCallPolicy
    }

    @available(
        *, deprecated,
        renamed:
            "init(maxTokens:maxKVSize:kvBits:kvGroupSize:quantizedKVStart:kvScheme:temperature:topP:topK:minP:repetitionPenalty:repetitionContextSize:presencePenalty:presenceContextSize:frequencyPenalty:frequencyContextSize:prefill:seed:)",
        message:
            "prefill now defaults to balanced chunking; use prefill.chunking = .remainder for the legacy chunk boundaries"
    )
    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        kvScheme: String? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int?,
        seed: UInt64? = nil
    ) {
        self.init(
            maxTokens: maxTokens, maxKVSize: maxKVSize, kvBits: kvBits,
            kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart, kvScheme: kvScheme,
            temperature: temperature, topP: topP, topK: topK, minP: minP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
            presencePenalty: presencePenalty, presenceContextSize: presenceContextSize,
            frequencyPenalty: frequencyPenalty, frequencyContextSize: frequencyContextSize,
            prefill: .init(stepSize: prefillStepSize), seed: seed)
    }

    /// Deprecated alias for ``makeSampler()``.
    @available(*, deprecated, renamed: "makeSampler()")
    public func sampler() -> LogitSampler { makeSampler() }

    /// Deprecated alias for ``makeProcessor()``.
    @available(*, deprecated, renamed: "makeProcessor()")
    public func processor() -> LogitProcessor? { makeProcessor() }

    /// Builds the ``LogitSampler`` the temperature and the probability filters imply.
    public func makeSampler() -> LogitSampler {
        let usesTopP = TopPSampler.isActive(topP: topP)
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(
                temperature: temperature, topP: topP, topK: topK, minP: minP, seed: seed)
        } else {
            return CategoricalSampler(temperature: temperature, seed: seed)
        }
    }

    /// Builds the ``LogitProcessor`` implied by the configured repetition/presence/frequency
    /// penalties, composing them into a ``PenaltyProcessor``, or `nil` if none are enabled.
    public func makeProcessor() -> LogitProcessor? {
        let repetitionContext = Self.penaltyContext(
            penalty: repetitionPenalty, contextSize: repetitionContextSize,
            make: RepetitionContext.init(repetitionPenalty:repetitionContextSize:))
        let presenceContext = Self.penaltyContext(
            penalty: presencePenalty, contextSize: presenceContextSize,
            make: PresencePenaltyContext.init(presencePenalty:presenceContextSize:))
        let frequencyContext = Self.penaltyContext(
            penalty: frequencyPenalty, contextSize: frequencyContextSize,
            make: FrequencyPenaltyContext.init(frequencyPenalty:frequencyContextSize:))

        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil {
            return nil
        }

        return PenaltyProcessor(
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext
        )
    }

    /// Builds a penalty context via `make` when `penalty` is non-zero and `contextSize`
    /// is positive; otherwise returns `nil`. Shared by `makeProcessor()`'s three near-identical
    /// repetition/presence/frequency penalty checks.
    private static func penaltyContext<Context>(
        penalty: Float?, contextSize: Int, make: (Float, Int) -> Context
    ) -> Context? {
        guard let penalty, penalty != 0, contextSize > 0 else { return nil }
        return make(penalty, contextSize)
    }
}

/// `logits` in a type `logSoftmax` takes without losing precision.
func upcastingBFloat16(_ logits: MLXArray) -> MLXArray {
    logits.dtype == .bfloat16 ? logits.asType(.float32) : logits
}

/// `logits` scaled by `temperature`, for the sampling distribution and the log probabilities
/// reported for it alike.
private func scaledByTemperature(_ logits: MLXArray, temperature: MLXArray) -> MLXArray {
    logits * (1 / temperature)
}

/// The RNG of a sampler: seeded with `seed` when it is set, which makes sampling
/// reproducible, and else seeded from system entropy, the prior behavior.
private func makeRandomState(seed: UInt64?) -> MLXRandom.RandomState {
    seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    /// Creates a sampler; stateless (no seed or configuration is needed for argmax).
    public init() {}

    /// Greedily selects the single highest-probability token (no randomness).
    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
public struct TopPSampler: DistributionLogitSampler {
    let temperature: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    /// Creates a top-p/top-k/min-p sampler; `seed` (when provided) makes sampling
    /// reproducible, otherwise the RNG is seeded from system entropy.
    public init(
        temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0,
        seed: UInt64? = nil
    ) {
        self.temperature = MLXArray(temperature)
        if Self.isActive(topP: topP) {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        self.randomState = makeRandomState(seed: seed)
    }

    /// Whether `topP` turns the nucleus filter on: a value in `(0, 1)` keeps a
    /// strict subset of the vocabulary, and any other value keeps all of it.
    ///
    /// ``GenerateParameters/makeSampler()`` and the initializer decide with this
    /// predicate, thus the two agree on when the filter is active.
    static func isActive(topP: Float) -> Bool {
        topP > 0 && topP < 1
    }

    /// Samples a token from `logits` after applying the configured top-p/top-k/min-p
    /// filters and temperature scaling.
    public func sample(logits: MLXArray) -> MLXArray {
        sample(logProbabilities: LogProbabilityDistribution(logits: logits).values)
    }

    func sample(logProbabilities: MLXArray) -> MLXArray {
        withRandomState(randomState) {
            categorical(
                scaledByTemperature(applyFilters(logProbabilities), temperature: temperature))
        }
    }

    /// The normalized log-probabilities that ``sample(logits:)`` draws from, after the
    /// filters and the temperature. Speculative rejection sampling reads p and q here.
    public func logProbabilities(logits: MLXArray) -> MLXArray {
        let logprobs = applyFilters(LogProbabilityDistribution(logits: logits).values)
        return logSoftmax(scaledByTemperature(logprobs, temperature: temperature))
    }

    /// Applies the configured filters in Python mlx-lm order: top_p → min_p → top_k.
    private func applyFilters(_ logProbabilities: MLXArray) -> MLXArray {
        var logprobs = logProbabilities
        if let topP {
            logprobs = applyTopP(logprobs, topP: topP)
        }
        if let minP {
            logprobs = applyMinP(logprobs, minP: minP)
        }
        if let topK {
            logprobs = applyTopK(logprobs, topK: topK)
        }
        return logprobs
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP` (nucleus sampling).
    /// Matches `apply_top_p` from `mlx_lm/sample_utils.py`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    /// Matches `apply_min_p` from `mlx_lm/sample_utils.py`.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    /// Mirrors `apply_top_k` from `mlx_lm/sample_utils.py`.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        // O(V) partition on negated logprobs so top-k land at [0, topK).
        // Indices at [topK, V) are the tokens to mask out.
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: DistributionLogitSampler {
    let temperature: MLXArray
    let randomState: MLXRandom.RandomState

    /// Creates a temperature-scaled categorical sampler; `seed` (when provided) makes
    /// sampling reproducible, otherwise the RNG is seeded from system entropy.
    public init(temperature: Float, seed: UInt64? = nil) {
        self.temperature = MLXArray(temperature)
        self.randomState = makeRandomState(seed: seed)
    }

    /// Samples a token from `logits` by scaling by `temperature` and drawing from the
    /// resulting categorical distribution (no top-p/top-k/min-p filtering).
    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(scaledByTemperature(logits, temperature: temperature))
        }
    }

    /// The normalized log-probabilities that ``sample(logits:)`` draws from.
    public func logProbabilities(logits: MLXArray) -> MLXArray {
        LogProbabilityDistribution(
            logits: scaledByTemperature(upcastingBFloat16(logits), temperature: temperature)
        ).values
    }
}

/// GPU-resident ring buffer of recent token IDs.
///
/// Shared by penalty processors to avoid duplicating ring buffer logic.
/// Uses `MLX.where` mask operations for GPU-only updates (no CPU←GPU sync),
/// preserving `asyncEval()` pipelining in `TokenIterator`.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt. Keeps the last `capacity` tokens.
    ///
    /// Image sentinels (such as FastVLM's -200) are not vocabulary IDs, so they are
    /// skipped. The prompt is read back to front, only as far as it takes to find
    /// `capacity` tokens.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let prompt = prompt.flattened()
        var tokens = [Int32]()
        var end = prompt.size
        var length = capacity
        while tokens.count < capacity, end > 0 {
            let start = max(0, end - length)
            // Converting on the CPU skips a GPU round trip for a handful of tokens.
            let chunk = prompt[start ..< end].asType(.int32, stream: .cpu)
            tokens = chunk.asArray(Int32.self).filter { $0 >= 0 } + tokens
            end = start
            length *= 2
        }
        let kept = tokens.suffix(capacity)
        buffer = MLXArray(Array(kept) + [Int32](repeating: 0, count: capacity - kept.count))
        count = kept.count
        writeIndex = count % capacity
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Conformers keep their recent-token history in a `TokenRing` for penalty
/// computation; bulk-loading the prompt into that ring at the start of
/// generation is identical for all of them, so the default lives here once.
private protocol TokenRingBackedProcessor: LogitProcessor {
    var ring: TokenRing { get set }
}

extension TokenRingBackedProcessor {
    /// Bulk-loads the prompt's recent-token window into `ring`. Shared by every
    /// conformer since this bookkeeping doesn't vary by penalty type.
    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    /// Records `token` in the recent-token ring so future `process(logits:)` calls
    /// penalize it. Shared by every conformer since this bookkeeping doesn't vary
    /// by penalty type.
    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }

    /// `logits` with the entries of the tokens in the ring replaced by `transform` of them.
    ///
    /// - Parameters:
    ///   - logits: The logits, shape `[B, vocabulary]`.
    ///   - transform: Maps the logits of the recent tokens to their penalized logits.
    /// - Returns: The penalized logits, or `logits` while the ring is empty.
    fileprivate func applyToRecentTokens(
        _ logits: MLXArray, transform: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        let broadcastIndices = indices[.newAxis, 0...]
        let selectedLogits = takeAlong(logits, broadcastIndices, axis: -1)
        return putAlong(logits, broadcastIndices, values: transform(selectedLogits), axis: -1)
    }
}

/// Processor that implements a `repetitionPenalty`.
public struct RepetitionContext: TokenRingBackedProcessor {
    fileprivate var ring: TokenRing
    let repetitionPenalty: Float

    /// Creates a repetition-penalty processor tracking the last `repetitionContextSize` tokens.
    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    /// Scales logits of recently-seen tokens toward zero (dividing positive logits,
    /// multiplying negative ones) by `repetitionPenalty`; leaves other logits unchanged.
    public func process(logits: MLXArray) -> MLXArray {
        applyToRecentTokens(logits) { selectedLogits in
            MLX.where(
                selectedLogits .< 0, selectedLogits * repetitionPenalty,
                selectedLogits / repetitionPenalty)
        }
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
///
/// The penalty is applied once per unique token via scatter-write (writing the
/// same value to the same index multiple times is idempotent).
public struct PresencePenaltyContext: TokenRingBackedProcessor {
    fileprivate var ring: TokenRing
    let presencePenalty: Float

    /// Creates a presence-penalty processor tracking the last `presenceContextSize` tokens.
    public init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    /// Subtracts `presencePenalty` from the logits of every token currently in the ring.
    public func process(logits: MLXArray) -> MLXArray {
        applyToRecentTokens(logits) { $0 - presencePenalty }
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
///
/// Frequency counting is performed on GPU via `scatter_add` to build a histogram
/// of token occurrences, avoiding CPU←GPU synchronization.
public struct FrequencyPenaltyContext: TokenRingBackedProcessor {
    fileprivate var ring: TokenRing
    let frequencyPenalty: Float

    /// Creates a frequency-penalty processor tracking the last `frequencyContextSize` tokens.
    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    /// Subtracts `frequencyPenalty * occurrenceCount` from each vocabulary entry's logit,
    /// based on how often that token appears in the recent-token ring.
    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }

        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
public struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    /// Creates a composite processor from the optional sub-processors enabled by
    /// ``GenerateParameters/makeProcessor()``; each `nil` sub-processor is skipped.
    public init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    /// Forwards the prompt to every enabled sub-processor so each can seed its own ring.
    mutating public func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    /// Applies each enabled sub-processor's `process(logits:)` in repetition → presence →
    /// frequency order, threading the result of one into the next.
    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    /// Forwards the sampled token to every enabled sub-processor so each can update its ring.
    mutating public func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

/// Processor that applies multiple ``LogitProcessor`` instances in order.
///
/// ``GenerationComponents/logitProcessor(parameters:)`` uses this to compose the
/// built-in ``PenaltyProcessor`` with a processor from
/// ``GenerationComponents/logitProcessorFactory``. It can also be used directly
/// to combine several custom processors into one.
public struct ChainedLogitProcessor: LogitProcessor {
    var processors: [any LogitProcessor]

    public init(processors: [any LogitProcessor]) {
        self.processors = processors
    }

    public func copy() -> Self {
        Self(processors: processors.map { $0.copy() })
    }

    mutating public func prompt(_ prompt: MLXArray) {
        forEachProcessor { $0.prompt(prompt) }
    }

    public func process(logits: MLXArray) -> MLXArray {
        processors.reduce(logits) { $1.process(logits: $0) }
    }

    mutating public func didSample(token: MLXArray) {
        forEachProcessor { $0.didSample(token: token) }
    }

    mutating public func didEmit(token: Int) {
        forEachProcessor { $0.didEmit(token: token) }
    }

    mutating public func finalizeGeneration() {
        forEachProcessor { $0.finalizeGeneration() }
    }

    /// Runs `body` on each processor in place, in order.
    private mutating func forEachProcessor(_ body: (inout any LogitProcessor) -> Void) {
        for index in processors.indices {
            body(&processors[index])
        }
    }
}

package protocol GenerationReasoningTokenCounting {
    var generationReasoningTokenCount: Int? { get }
}

extension ChainedLogitProcessor: GenerationReasoningTokenCounting {
    package var generationReasoningTokenCount: Int? {
        processors.lazy.compactMap {
            ($0 as? any GenerationReasoningTokenCounting)?.generationReasoningTokenCount
        }
        .first
    }
}
