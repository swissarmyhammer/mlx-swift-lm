// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
// `_version: 2` gates on the FoundationModels *framework* major version, which
// is 1.4.x on the macOS/iOS 26 SDK and 2.0.x on 27. The third-party-model
// surface this adapter uses (`LanguageModel`, `LanguageModelCapabilities`, the
// generic `LanguageModelSession(model:)` init) only exists on the 27 SDK, so
// this excludes the whole adapter from older SDKs where those symbols are
// absent. A plain `canImport(FoundationModels)` is insufficient — the module
// also ships in 26 — and `@available` cannot help, since it gates runtime
// availability, not the compile-time presence of a symbol in the SDK.
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLX
import os.log
import MLXGuidedGeneration

/// Shared `os.log` subsystem for every logger in this adapter.
let mlxFoundationModelsLoggingSubsystem = "com.apple.FoundationModels-MLX"

// MARK: - Constraint Cache Kind

/// Selects which xgrammar constructor a cached template was compiled
/// with. Used by the constraint cache so a JSON-schema source and a
/// structural-tag source can never alias even if their text collides.
enum ConstraintKind {
    case json
    case structuralTag
}

// MARK: - Tokenizer Bias Cache Entry

/// Tokenizer-derived logit biases, cached per model. Both arrays are pure
/// functions of the tokenizer, so they are identical for a model's lifetime.
/// `@unchecked Sendable`: every field is `let` and read-only after construction
/// (the arrays are only *added* to logits in `GuidedGenerationLoop`, never
/// mutated), and the entry is shared across actors via `ModelCache` — the same
/// pattern as `GrammarTokenizer`/`GrammarConstraint` in `XGrammarBridge.swift`.
struct TokenizerBias: @unchecked Sendable {
    let closing: MLXArray
    let whitespace: MLXArray
    let whitespaceTokenIDs: Set<Int>
}

// MARK: - Constraint Setup

/// The grammar-constraint machinery `prepareConstraintSetup` prepares for the
/// tool-calling and guided-generation paths: a model-keyed xgrammar
/// tokenizer, a compiled constraint, and the token-budget/logit-bias values
/// `GuidedGenerationLoop.run` needs to steer generation toward a structural
/// close.
struct ConstraintSetup {
    let xgTokenizer: GrammarTokenizer
    let constraint: GrammarConstraint
    let maxTokens: Int
    let closingBias: MLXArray
    let completionReserve: Int
    let hardReserve: Int
    let whitespaceBias: MLXArray
    let whitespaceTokenIDs: Set<Int>
}

// MARK: - Model Cache Actor

/// Thread-safe model cache using Swift actor isolation.
/// Prevents race conditions when multiple concurrent requests try to load the model.
/// Supports caching multiple models by their identifiers.
private actor ModelCache {
    /// Class wrapper around `Task` so actor-reentrancy supersession guards can
    /// use `===` identity comparison. `Task` is a value type; a wrapper lets us
    /// detect whether `evictAll()` replaced a loading entry mid-flight.
    private final class LoadTask {
        let task: Task<ModelContainer, Error>
        init(_ task: Task<ModelContainer, Error>) { self.task = task }
    }

    private var containers: [String: ModelContainer] = [:]
    private var loadingTasks: [String: LoadTask] = [:]
    /// In-flight loads tagged as a warmup of an already-present model, which
    /// must NOT surface as `.downloading` (there is no user-facing download).
    /// A subset of `loadingTasks`' keys. See `load` and `isDownloading`.
    private var suppressedLoadIDs: Set<String> = []
    private var xgTokenizers: [String: GrammarTokenizer] = [:]
    /// Cached compiled constraint templates keyed by (modelID, schemaJSON).
    /// Clone from template instead of recompiling the grammar each request.
    private var constraintTemplates: [String: GrammarConstraint] = [:]
    /// Cached per-model logit biases (closing + whitespace). Pure functions of
    /// the tokenizer, so computed once per model and reused across requests.
    private var tokenizerBiases: [String: TokenizerBias] = [:]
    /// Most recent load error per model. Cleared on a subsequent successful
    /// load. Surfaced through `MLXLanguageModel.availability` so callers can
    /// distinguish "never tried" from "tried and failed".
    private var lastErrors: [String: any Error] = [:]

    /// Gets the cached model container for the given model ID, loading it if necessary.
    /// Concurrent callers for the same model will share the same loading task, preventing duplicate loads.
    ///
    /// The `loader` closure carries the transport types (downloader, tokenizer
    /// loader). Keeping them out of the cache means the cache itself stays
    /// agnostic of how a container is acquired -- first caller wins; later
    /// callers reuse the cached container regardless of which loader they
    /// brought along.
    func load(
        modelID: String,
        suppressDownloadingState: Bool = false,
        loader: @Sendable @escaping () async throws -> ModelContainer
    ) async throws -> ModelContainer {
        if let cached = containers[modelID] {
            return cached
        }

        if let existingLoadTask = loadingTasks[modelID] {
            // Coalesced onto an in-flight load: the first caller's
            // classification (downloading vs. suppressed) stands — we do not
            // re-tag. This collision is benign because the suppress decision is
            // conditioned on disk-presence: a warmup and a genuine download for
            // a not-yet-present model both classify as downloading, so they
            // agree; when the model IS present, `availability` resolves to
            // `.available` regardless of the in-flight load.
            return try await existingLoadTask.task.value
        }

        let loadTask = LoadTask(
            Task<ModelContainer, Error> {
                try await loader()
            })
        loadingTasks[modelID] = loadTask
        // Tag a warmup-of-an-already-present model out of the `.downloading`
        // signal (computed by the caller as warmup AND modelExistsOnDisk()).
        if suppressDownloadingState {
            suppressedLoadIDs.insert(modelID)
        }

        do {
            let loaded = try await loadTask.task.value
            // Supersession guard: `evict()`/`evictAll()` may have removed this
            // load while it was suspended (actor reentrancy). If we are no longer
            // the registered task, hand the awaiter its container but do NOT
            // re-populate the cache — ARC frees the weights when the awaiter
            // releases it.
            guard loadingTasks[modelID] === loadTask else { return loaded }
            containers[modelID] = loaded
            loadingTasks[modelID] = nil
            suppressedLoadIDs.remove(modelID)
            lastErrors[modelID] = nil
            return loaded
        } catch {
            // Same guard on the failure path: a superseded load must not re-add a
            // stale lastErrors entry for a model nobody holds.
            if loadingTasks[modelID] === loadTask {
                loadingTasks[modelID] = nil
                suppressedLoadIDs.remove(modelID)
                lastErrors[modelID] = error
            }
            throw error
        }
    }

    /// Whether a *genuine download* is in flight for the given model: a load
    /// task is running and it was not tagged as a warmup of an already-present
    /// model. Drives `availability`'s `.downloading` state, so a background
    /// warmup of an already-downloaded model does not spuriously report
    /// `.downloading`. (A warmup that triggers a real fetch is not tagged and
    /// does report here.)
    func isDownloading(modelID: String) -> Bool {
        loadingTasks[modelID] != nil && !suppressedLoadIDs.contains(modelID)
    }

    /// The most recent load error for the given model, if a previous attempt
    /// failed and no successful load has happened since.
    func lastError(modelID: String) -> (any Error)? {
        lastErrors[modelID]
    }

    /// Gets or creates a cached GrammarTokenizer for the given model.
    func makeXgTokenizer(
        modelID: String,
        tokenizer: any Tokenizer
    ) throws -> GrammarTokenizer {
        if let cached = xgTokenizers[modelID] {
            return cached
        }
        let vocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
        let xgTok = try GrammarTokenizer(
            vocab: vocab.vocab,
            vocabType: vocab.vocabType,
            eosTokenId: Int32(tokenizer.eosTokenId ?? 0)
        )
        xgTokenizers[modelID] = xgTok
        return xgTok
    }

    /// Whether an `GrammarTokenizer` is already cached for the given model.
    /// Used by `MLXLanguageModel.hasCachedXgTokenizer` so tests can assert
    /// that `warmUp()` pre-created it (a genuine cache hit) rather than only
    /// that a later guided respond happens to succeed.
    func hasCachedXgTokenizer(modelID: String) -> Bool {
        xgTokenizers[modelID] != nil
    }

    /// Gets or creates the cached tokenizer-derived logit biases for a model.
    func makeTokenizerBias(
        modelID: String,
        tokenizer: any Tokenizer
    ) -> TokenizerBias {
        if let cached = tokenizerBiases[modelID] {
            return cached
        }
        let closing = ClosingTokenBias.compute(
            tokenizer: tokenizer,
            eosTokenId: tokenizer.eosTokenId
        )
        let (whitespace, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
            tokenizer: tokenizer
        )
        let bias = TokenizerBias(
            closing: closing,
            whitespace: whitespace,
            whitespaceTokenIDs: whitespaceTokenIDs
        )
        tokenizerBiases[modelID] = bias
        return bias
    }

    /// Gets a fresh constraint by cloning a cached template, or compiles and caches one first.
    ///
    /// Grammar compilation is expensive (~5-20ms). By caching the compiled template
    /// and cloning it (~0.1ms), repeated requests with the same schema skip recompilation.
    /// When Fork() is unavailable (xgrammar < v0.1.34), the clone attempt fails gracefully
    /// and each request compiles a fresh constraint instead.
    func makeConstraint(
        modelID: String,
        kind: ConstraintKind,
        source: String,
        tokenizer: GrammarTokenizer,
        hostTokenizer: any Tokenizer,
        fastForward: Bool
    ) throws -> GrammarConstraint {
        let cacheKey = "\(modelID):\(kind):\(source)"
        if let template = constraintTemplates[cacheKey] {
            do {
                return try template.clone()
            } catch GrammarError.forkFailed {
                constraintTemplates.removeValue(forKey: cacheKey)
            }
        }
        let constraint: GrammarConstraint
        switch kind {
        case .json:
            constraint = try GrammarConstraint(
                tokenizer: tokenizer,
                jsonSchema: source,
                fastForward: fastForward,
                hostTokenizer: hostTokenizer
            )
        case .structuralTag:
            constraint = try GrammarConstraint(
                tokenizer: tokenizer,
                structuralTag: source,
                fastForward: fastForward,
                hostTokenizer: hostTokenizer
            )
        }
        if let cloned = try? constraint.clone() {
            constraintTemplates[cacheKey] = constraint
            return cloned
        }
        return constraint
    }

    /// Evicts all cached state: model containers, tokenizers, constraint
    /// templates, and per-model tokenizer biases. No GPU-stream synchronization
    /// is required — in-flight callers retain their own `ModelContainer` and
    /// free it via ARC on completion.
    func evictAll() {
        containers.removeAll()
        loadingTasks.removeAll()
        suppressedLoadIDs.removeAll()
        xgTokenizers.removeAll()
        constraintTemplates.removeAll()
        tokenizerBiases.removeAll()
        lastErrors.removeAll()
    }

    /// Evicts a single model's state across every per-model cache: its container,
    /// xgrammar tokenizer, all compiled constraint templates, tokenizer bias,
    /// last load error, the suppressed-download tag, and any in-flight load
    /// registration.
    /// Best-effort cancels an in-flight load (the load path is not
    /// cancellation-aware today, so this is a no-op safety net); the
    /// load-completion guard in `load()` is what prevents a superseded load
    /// from re-populating after removal.
    func remove(modelID: String) {
        // `loadingTasks` holds a `LoadTask` box; cancel the wrapped `Task`.
        loadingTasks[modelID]?.task.cancel()
        loadingTasks.removeValue(forKey: modelID)
        suppressedLoadIDs.remove(modelID)
        containers.removeValue(forKey: modelID)
        xgTokenizers.removeValue(forKey: modelID)
        constraintTemplates = constraintTemplates.filter {
            !$0.key.hasPrefix("\(modelID):")
        }
        tokenizerBiases.removeValue(forKey: modelID)
        lastErrors.removeValue(forKey: modelID)
    }
}

// MARK: - MLXLanguageModel

/// A language model implementation that uses MLX for local inference.
///
/// Conforms to the FoundationModels `LanguageModel` protocol, allowing MLX models
/// to be used with `LanguageModelSession`.
///
/// Example usage:
/// ```swift
/// import MLXFoundationModels
/// import MLXHuggingFace
/// import MLXLMCommon
/// import Hub
/// import Tokenizers
///
/// let model = MLXLanguageModel(
///     configuration: ModelConfiguration(id: "mlx-community/Qwen2.5-3B-Instruct-4bit"),
///     capabilities: [.guidedGeneration, .toolCalling],
///     weightsLocation: { id in HubApi.shared.localRepoLocation(HubApi.Repo(id: id)) },
///     load: { configuration, progressHandler in
///         try await loadModelContainer(
///             from: #hubDownloader(),
///             using: #huggingFaceTokenizerLoader(),
///             configuration: configuration,
///             progressHandler: progressHandler)
///     })
/// let session = LanguageModelSession(model: model, tools: [], instructions: nil)
/// let response = try await session.respond(to: "Hello!")
/// print(response.content)
/// ```
///
/// **Factory registration**: this target deliberately does not depend on
/// `MLXLLM`. Consumers who want LLM inference must import `MLXLLM` (or another
/// factory provider) in their own target so that
/// `MLXLLM.TrampolineModelFactory` is linked into the binary; otherwise
/// `loadModelContainer` fails with `noModelFactoryAvailable`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
public struct MLXLanguageModel: FoundationModels.LanguageModel, Sendable {

    // MARK: - Model Caching (CRITICAL for performance)

    /// Shared model cache - thread-safe via actor isolation.
    /// Without caching, model loading takes 2-30 seconds per request.
    private static let cache = ModelCache()

    /// The configuration identifying and parameterizing the model to load.
    public let configuration: ModelConfiguration

    /// Resolves a model identifier to its on-disk weights directory. Used by
    /// the availability checks (`modelExistsOnDisk()`, `freeDiskSpaceBytes`),
    /// not by the load path. Injected so this module needs no HuggingFace
    /// path-resolution dependency.
    public let weightsLocation: @Sendable (String) -> URL

    /// Loads the model container for a configuration, forwarding download
    /// progress. Injected so this module carries no HuggingFace or
    /// swift-transformers dependency; the HuggingFace wiring lives in callers.
    public typealias ContainerLoader =
        @Sendable (
            _ configuration: ModelConfiguration,
            _ progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> ModelContainer

    private let load: ContainerLoader

    /// Stable identity for the model cache, executor configuration, tokenizer
    /// caches, availability, and progress reporting. Derived from the
    /// configuration so it is the single place identity is defined.
    public var modelID: String { configuration.name }

    /// Loads the model container for this model, returning a cached instance
    /// when one exists. Shares the process-global cache that `respond()`,
    /// `preload()`, and `session.prewarm()` use, so a caller working directly
    /// with the lower-level `ModelContainer` reuses the adapter's cache.
    public func loadContainer() async throws -> ModelContainer {
        try await loadContainer(suppressDownloadingState: false)
    }

    /// Internal variant that keeps an in-flight load of an already-present
    /// model out of the `.downloading` availability signal.
    func loadContainer(suppressDownloadingState: Bool) async throws -> ModelContainer {
        try await Self.cache.load(
            modelID: modelID,
            suppressDownloadingState: suppressDownloadingState,
            loader: makeContainerLoader())
    }

    /// Sets the process-global MLX buffer-reuse pool limit a single time. A
    /// `static let` initializer runs lazily and exactly once (thread-safe), so
    /// repeated model loads don't re-stomp a consumer's own `Memory.cacheLimit`.
    ///
    /// Higher = less allocator thrash at the cost of slightly higher resident GPU
    /// memory. 256MB comfortably holds activations and KV cache for a 3B model
    /// without forcing pool evictions mid-forward-pass.
    private static let configureGPUCacheOnce: Void = {
        MLX.Memory.cacheLimit = 256 * 1024 * 1024
    }()

    private func makeContainerLoader() -> @Sendable () async throws -> ModelContainer {
        let configuration = self.configuration
        let load = self.load
        return {
            // Configure the buffer pool once per process rather than on every
            // load, so a consumer's own `Memory.cacheLimit` survives our loads.
            _ = Self.configureGPUCacheOnce
            let container = try await load(configuration) { progress in
                MLXDownloadProgress.report(progress: progress, modelID: configuration.name)
            }
            MLXDownloadProgress.reportCompleted()
            return container
        }
    }

    /// Gets or creates a cached GrammarTokenizer for the given model.
    static func makeXgTokenizer(
        modelID: String,
        tokenizer: any Tokenizer
    ) async throws -> GrammarTokenizer {
        try await cache.makeXgTokenizer(modelID: modelID, tokenizer: tokenizer)
    }

    /// Gets the cached per-model tokenizer-derived logit biases (closing +
    /// whitespace), computing them on first use.
    static func makeTokenizerBias(
        modelID: String,
        tokenizer: any Tokenizer
    ) async -> TokenizerBias {
        await cache.makeTokenizerBias(modelID: modelID, tokenizer: tokenizer)
    }

    /// Gets a constraint by cloning a cached compiled template (or compiling one first).
    static func makeConstraint(
        modelID: String,
        kind: ConstraintKind,
        source: String,
        tokenizer: GrammarTokenizer,
        hostTokenizer: any Tokenizer,
        fastForward: Bool
    ) async throws -> GrammarConstraint {
        try await cache.makeConstraint(
            modelID: modelID,
            kind: kind,
            source: source,
            tokenizer: tokenizer,
            hostTokenizer: hostTokenizer,
            fastForward: fastForward
        )
    }

    /// Whether the shared cache already holds an `GrammarTokenizer` for the model.
    /// Internal test seam (not public API): lets `PrewarmGrammarTests` confirm
    /// `warmUp()` pre-created the tokenizer.
    static func hasCachedXgTokenizer(modelID: String) async -> Bool {
        await cache.hasCachedXgTokenizer(modelID: modelID)
    }

    /// Evicts every cached model, tokenizer, constraint template, and per-model
    /// tokenizer bias, freeing the GPU memory held by model weights. Subsequent
    /// requests reload from the on-disk cache.
    ///
    /// Safe to call during in-flight `respond()`/`warmUp()` work: each holds its
    /// own strong reference to the `ModelContainer` and synchronizes the GPU on
    /// exit, so dropping the cache's reference cannot free weights out from under
    /// a live kernel — the weights free via ARC once that work returns.
    public static func evictAll() async {
        await cache.evictAll()
    }

    /// Drops this model from the shared cache, freeing the GPU memory held by its
    /// weights. A subsequent `respond()`/`preload()` triggers a fresh load
    /// (reusing the on-disk snapshot if the model was previously downloaded).
    ///
    /// Safe to call during an in-flight `respond()`: that call retains its own
    /// `ModelContainer` and finishes normally; the weights free via ARC once it
    /// returns. Evicting a model whose load is still in flight removes it cleanly
    /// — the in-flight load completes but does not re-populate the cache.
    public func evict() async {
        await Self.cache.remove(modelID: modelID)
    }

    /// Whether the shared cache has a *genuine download* in flight for the
    /// given model — excludes a warmup of an already-present model. Used by
    /// ``availability`` to surface a `.downloading` state.
    static func isDownloadingInCache(modelID: String) async -> Bool {
        await cache.isDownloading(modelID: modelID)
    }

    /// The most recent load error for the given model, if any. Cleared on a
    /// subsequent successful load. Used by ``availability`` to surface a
    /// `.downloadFailed` state after a failed ``preload()``.
    static func lastLoadErrorInCache(modelID: String) async -> (any Error)? {
        await cache.lastError(modelID: modelID)
    }

    // MARK: - LanguageModel Conformance

    /// MLX supports guided generation via xgrammar grammar-constrained
    /// decoding (provided by the MLXGuidedGeneration library), tool
    /// calling via the synthetic-final-answer envelope, and reasoning
    /// (chain-of-thought) routing on the unconstrained generation path.
    ///
    /// Capabilities are declared explicitly by the caller at ``init(configuration:capabilities:configurationResolver:weightsLocation:load:)``
    /// and stored verbatim. The caller includes
    /// `.guidedGeneration`/`.toolCalling`/`.reasoning` as appropriate; the
    /// adapter does not consult ``ReasoningHeuristics`` (which remains a
    /// standalone helper a caller may use to compute their own capability set).
    ///
    /// Declaring `.reasoning` matters for request routing: the framework only
    /// forwards a `reasoningLevel` to executors that declare `.reasoning`, and
    /// auto-rejects one otherwise (on the developer's behalf) before `respond`
    /// runs. The executor in turn emits `.reasoning` events only when this
    /// capability was declared.
    public let capabilities: LanguageModelCapabilities

    /// The configuration resolver that patches a per-call ``ModelConfiguration``
    /// for this instance. Defaults to ``DefaultConfigurationResolver`` when
    /// omitted.
    public let configurationResolver: any ModelConfigurationResolver

    /// Configuration the framework uses to create and cache executors.
    public var executorConfiguration: Executor.Configuration {
        Executor.Configuration(modelID: modelID)
    }

    // MARK: - Initialization

    /// Creates an MLXLanguageModel from a configuration, deferring model
    /// loading until first inference or `preload()`.
    ///
    /// - Parameters:
    ///   - configuration: Identifies and parameterizes the model (e.g.
    ///     `LLMRegistry.gemma3_1B_qat_4bit` or `ModelConfiguration(id:)`).
    ///   - capabilities: The capabilities this model supports
    ///     (`.guidedGeneration`, `.toolCalling`, `.reasoning`, `.vision`).
    ///     Stored verbatim; the adapter never infers or expands the set.
    ///   - configurationResolver: Patches a per-call ``ModelConfiguration``
    ///     (reasoning config, extra stop tokens) for this instance.
    ///   - weightsLocation: Resolves a model identifier to its on-disk weights
    ///     directory, for the availability checks.
    ///   - load: Loads the model container for a configuration.
    public init(
        configuration: ModelConfiguration,
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration],
        configurationResolver: any ModelConfigurationResolver =
            DefaultConfigurationResolver(),
        weightsLocation: @Sendable @escaping (String) -> URL,
        load: @escaping ContainerLoader
    ) {
        self.configuration = configuration
        self.capabilities = LanguageModelCapabilities(capabilities: capabilities)
        self.configurationResolver = configurationResolver
        self.weightsLocation = weightsLocation
        self.load = load
    }

    /// Downloads the model and loads its weights into memory.
    ///
    /// This is a weights-only load: it runs no forward pass, compiles no Metal
    /// shaders, and performs no GPU work, so the first generation request after
    /// `preload()` still pays the one-time Metal shader JIT cost. The call is
    /// awaitable and fully caller-owned — you decide when it runs and handle
    /// any error it throws.
    ///
    /// Call it early, for example when a view appears, to move the
    /// download-and-load portion of cold-start latency off the first
    /// generation request.
    ///
    /// Safe to call multiple times; once the model is loaded, subsequent calls
    /// return immediately from cache.
    public func preload() async throws {
        _ = try await loadContainer()
    }

    /// Loads the model weights and compiles Metal shaders, so the first
    /// `respond()` afterward pays no (or materially reduced) cold-start
    /// shader-JIT cost.
    ///
    /// Metal kernels JIT-compile lazily on the first *synchronous* readback
    /// (`.item()` inside the generate loop) — scheduling work with `asyncEval`
    /// alone does not compile them — so this runs a minimal throwaway forward
    /// pass to force compilation ahead of a real request.
    ///
    /// The forward pass and its single `Stream.gpu.synchronize()` run inside
    /// `container.perform { }`, the same `SerialAccessContainer` lock the
    /// `respond` path holds for its entire generation, so a warmup cannot race
    /// a concurrent `respond` on the process-global `Stream.gpu`. The 1-token
    /// generate ends naturally and is consumed to completion — never cancelled
    /// mid-flight — so a Metal command buffer is never cancelled after commit and
    /// the stream is drained before teardown.
    ///
    /// Internal by design: it touches process-global Metal and is driven
    /// fire-and-forget by ``Executor/prewarm(model:transcript:)``, reached
    /// publicly through `session.prewarm()`. Safe to call multiple times and
    /// concurrently; subsequent calls reuse the cached container.
    func warmUp() async throws {
        // Distinguish a warmup of an already-present model (suppress the
        // spurious `.available → .downloading → .available` flip) from a
        // genuine first fetch (which still reports `.downloading`). Conditioning
        // on disk-presence — not "is a warmup" alone — is what makes the
        // loadingTasks-dedup collision benign (see `ModelCache.load`) and keeps
        // the partial-download guard intact: we suppress the in-flight
        // `.downloading` signal rather than reorder the availability checks
        // (reordering would let a partial download with only `config.json`
        // present falsely report `.available`).
        let alreadyOnDisk = modelExistsOnDisk()
        let container = try await loadContainer(suppressDownloadingState: alreadyOnDisk)

        // Pre-create the model-keyed GrammarTokenizer so a guided / tool-calling
        // consumer skips the expensive vocab-extraction step on first
        // respond(). It's keyed on modelID alone — the same cache entry
        // respond()'s guided path reads — so this is a genuine cache hit.
        //
        // CPU-only (xgrammar is C++; no Stream.gpu, no Metal), so it adds no
        // GPU-teardown-race exposure: the safe half of warmup. It runs *after*
        // loadContainer because it needs the live Tokenizer from the container,
        // and *before* the forward pass below so the GPU-touching work stays a
        // single contiguous, serialized block.
        //
        // We deliberately do NOT pre-build a constraint template here:
        // makeConstraint is keyed on modelID:kind:source, where `source` is the
        // per-request schema/tool grammar that prewarm doesn't possess — a
        // pre-built constraint would land under a key no real respond() reads.
        let tokenizer = await container.tokenizer
        _ = try await Self.makeXgTokenizer(
            modelID: modelID, tokenizer: tokenizer)

        // Force Metal shader JIT with a minimal 1-token generate, run inside
        // `perform` so the forward pass + synchronize serialize against any
        // concurrent `respond`. `maxTokens: 1` makes the stream end on
        // its own; we consume it fully (no early break) so generation runs to
        // completion and leaves no dangling GPU work to race the teardown sync.
        try await container.perform { context in
            // Exactly one synchronize on every exit path (success or throw),
            // per the Metal teardown invariant. `prepare` is CPU-only, so on a
            // pre-forward-pass throw this just synchronizes an idle stream.
            defer { Stream.gpu.synchronize() }
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user("warmup")]))
            let params = GenerateParameters(maxTokens: 1)
            for await _ in try MLXLMCommon.generate(
                input: input, parameters: params, context: context
            ) {
                // Drain to completion.
            }
        }
    }

    // MARK: - Executor

    /// Executes inference requests for the model.
    public struct Executor: LanguageModelExecutor, Sendable {

        /// Default `maxTokens` when the caller doesn't set
        /// `GenerationOptions.maximumResponseTokens`. Applied uniformly
        /// across guided-JSON, tool-calling, and unconstrained generation
        /// paths so all three share a single definition.
        ///
        /// The guided paths *require* a budget to activate the zone-based
        /// closing bias in `GuidedGenerationLoop` -- without it, open-source
        /// models tend to wander in JSON whitespace before reaching
        /// structural close. 4096 is generous for typical tool calls and
        /// structured outputs. Consumers can override via
        /// `GenerationOptions(maximumResponseTokens:)`.
        private static let defaultMaxTokens = 4096

        /// Map FoundationModels' optional `Double` `GenerationOptions.temperature`
        /// to MLXLMCommon's `Float` `GenerateParameters.temperature`, clamping
        /// negatives to 0.
        ///
        /// Negative sampling temperatures land in `CategoricalSampler` and
        /// produce inverted distributions; we clamp at 0 so the worst the
        /// caller can get is greedy. `0` itself is honored unchanged because
        /// MLXLMCommon's `GenerateParameters.sampler()` routes
        /// `temperature == 0` to `ArgMaxSampler` (greedy) -- no division-by-
        /// zero hazard.
        ///
        /// Kept as a standalone function (not inlined at its call site)
        /// since it's unit-tested directly in `MLXLanguageModelTests.swift`.
        ///
        /// - Returns: `nil` when the caller did not request a specific
        ///   temperature, leaving `GenerateParameters`' built-in default in
        ///   place. Otherwise the clamped `Float`.
        static func clampedTemperature(_ value: Double?) -> Float? {
            guard let value else { return nil }
            return Float(max(0, value))
        }

        /// Translate FoundationModels' `GenerationOptions.SamplingMode` into the
        /// backend-local `MLXSamplingMode`, dropping the best-effort `seed`
        /// (MLX's samplers expose no seed-injection hook). No mode set (`nil`)
        /// and any future/unknown `Kind` both map to `nil` -- "use the provider
        /// default" -- so an unrecognized case never traps and never reaches the
        /// resolver. All value policy lives in `resolveSamplingParameters`; this
        /// shim is a pure 1:1 case translation.
        ///
        /// Kept as a standalone function (not inlined at its call site)
        /// since it's unit-tested directly in `SamplingModeShimTests.swift`.
        static func samplingMode(
            from samplingMode: GenerationOptions.SamplingMode?
        ) -> MLXSamplingMode? {
            guard let kind = samplingMode?.kind else { return nil }
            switch kind {
            case .greedy:
                return .greedy
            case .randomTopK(let k, _):
                return .topK(k)
            case .randomProbabilityThreshold(let threshold, _):
                return .nucleus(threshold)
            @unknown default:
                return nil
            }
        }

        /// Build the `GenerateParameters` for a generation pass, threading the
        /// caller's temperature and sampling mode through the shared resolver so
        /// every real-sampler path (unconstrained, reasoning, tool-call
        /// reasoning) honors `samplingMode` identically. `maxTokens` is the
        /// already-resolved budget -- callers keep their own default/budget
        /// arithmetic, so this helper owns only temperature + sampling resolution.
        static func makeParameters(
            maxTokens: Int,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?
        ) -> GenerateParameters {
            var params = GenerateParameters(maxTokens: maxTokens)
            resolveSamplingParameters(
                mode: samplingMode,
                clampedTemperature: clampedTemperature(requestedTemperature)
            ).apply(to: &params)
            return params
        }

        /// Map xgrammar errors to typed `LanguageModelError` cases where the
        /// cause is provably the user's input; pass everything else through
        /// unchanged.
        ///
        /// Only `GrammarError.invalidJSONSchema` is mapped: that case fires when
        /// xgrammar's JSON-Schema validator outright rejects the schema text
        /// we synthesized from `GenerationSchema`, which is a problem the
        /// developer can fix (simplify the schema, drop an unsupported
        /// construct). `LanguageModelError.unsupportedGenerationGuide` is the
        /// framework's idiomatic surface for that.
        ///
        /// `constraintCompilationFailed` is deliberately NOT mapped to
        /// `unsupportedGenerationGuide`: its origin is ambiguous (could be
        /// schema-level, could be an internal shim failure), and claiming
        /// user-fault when the cause is actually our infrastructure
        /// misleads developers who pattern-match on typed errors.
        ///
        /// `tokenizerCreationFailed` and `bitmaskRetrievalFailed` are
        /// internal shim failures with no recovery path on the developer's
        /// side -- surfacing them untyped is honest.
        static func mapGrammarError(_ grammarError: GrammarError) -> Error {
            switch grammarError {
            case .invalidJSONSchema(let message):
                return LanguageModelError.unsupportedGenerationGuide(
                    .init(schemaName: nil, debugDescription: message)
                )
            default:
                return grammarError
            }
        }

        /// Configuration for creating and caching executors.
        public struct Configuration: Hashable, Sendable {
            /// The model identifier this executor uses for loading and metadata.
            public let modelID: String
        }

        /// The model identifier this executor uses for loading and metadata.
        let modelID: String

        /// Creates an executor from a configuration.
        public init(configuration: Configuration) throws {
            self.modelID = configuration.modelID
        }

        /// Logs warmup failures from the fire-and-forget `prewarm` path. A
        /// failed warmup is otherwise invisible (no throw reaches the caller),
        /// so this is the only diagnostic surface for a persistently-failing
        /// prewarm (bad id, network gone, OOM). Note it cannot intercept a
        /// Metal command-buffer assertion abort — that is a process crash, not
        /// a catchable Swift error.
        private static let logger = Logger(
            subsystem: mlxFoundationModelsLoggingSubsystem, category: "Prewarm")

        /// Prewarms the model: loads weights and pre-compiles Metal shaders so
        /// the first `respond()` pays no cold-start shader-JIT cost.
        ///
        /// This is the protocol witness for `LanguageModelExecutor`'s
        /// `prewarm(model:transcript:)`. The signature must match the
        /// requirement *exactly* — concrete `Transcript`, not a generic
        /// `some Collection<Transcript.Entry>` — otherwise it fails to bind as
        /// the witness and the framework's no-op default silently wins instead.
        /// The session hands us the live model instance, so we route through
        /// its downloader/loader pair.
        ///
        /// Fire-and-forget, mirroring Apple's SLM/PCCLM executors and the
        /// framework's own `session.prewarm()`: the method is synchronous and
        /// non-throwing, so it spawns a detached warmup `Task` and returns
        /// immediately. The `Task` is best-effort — a failure is logged, never
        /// surfaced to or crashed on the caller.
        ///
        /// - Parameters:
        ///   - model: The live model instance to warm.
        ///   - transcript: Accepted per protocol; the shader warmup uses a
        ///     fixed dummy prompt and does not depend on it.
        public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
            Task {
                do {
                    try await model.warmUp()
                } catch {
                    Self.logger.error(
                        "MLX prewarm failed for \(model.modelID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
        }

        /// Generates a response for the given request, streaming events into the channel.
        ///
        /// - Parameters:
        ///   - request: The generation request containing transcript, tools, and options
        ///   - model: The model instance for this request
        ///   - channel: The channel to send response events into
        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: MLXLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            var collected = TranscriptConverter.mlxMessages(for: request.transcript)
            // MLX tokenizer crashes on empty chat input; provide a fallback.
            if collected.isEmpty {
                collected = [Chat.Message.user("")]
            }
            let messages = collected

            // Vision capability gate (adapter-side). Labeled image
            // attachments arrive as public `.attachment` segments that
            // the SDK's own vision guard never inspects, so the adapter
            // is the only place that can enforce `.vision` for this path.
            // Throw the same typed error the SDK would, before loading
            // any weights, so a model declared without `.vision` fails
            // fast and identically across the tool / schema / plain paths.
            if !model.capabilities.contains(.vision),
                messages.contains(where: { !$0.images.isEmpty })
            {
                throw LanguageModelError.unsupportedCapability(
                    LanguageModelError.UnsupportedCapability(
                        capability: .vision,
                        debugDescription:
                            "This request includes an image, but .vision was not declared at MLXLanguageModel init. Declare .vision to accept image inputs."
                    ))
            }

            let container = try await model.loadContainer()

            // Encode schema to JSON if present
            let schemaJSON: String?
            if let schema = request.schema {
                schemaJSON = try SchemaConverter.encodeToJSON(schema)
            } else {
                schemaJSON = nil
            }

            let modelID = self.modelID
            let requestedMaxTokens = request.generationOptions.maximumResponseTokens
            // Translate the SDK sampling mode once, here where generationOptions
            // is in scope; thread the bridge-local value down to every
            // real-sampler path so they honor it identically.
            let requestedSamplingMode = Self.samplingMode(
                from: request.generationOptions.samplingMode)
            // Per SKILL.md: response and tool-calls entries each need a fresh
            // UUID — they live in separate transcript entries. We preserve the
            // framework-supplied `request.id` for tracing by stamping it into
            // the response metadata below, rather than reusing it as an entry id.
            let entryID = UUID().uuidString
            let toolCallsEntryID = UUID().uuidString
            let reasoningEntryID = UUID().uuidString
            // Captured before the actor hop so the perform closure doesn't
            // capture `model`. Reasoning is gated strictly on the declared
            // capability; the resolver-patched configuration supplies the
            // reasoning config we route on.
            let declaresReasoning = model.capabilities.contains(.reasoning)
            let configurationResolver = model.configurationResolver

            do {
                // Send metadata first
                await channel.send(
                    .response(
                        entryID: entryID,
                        action: .updateMetadata([
                            "modelID": modelID,
                            "requestID": request.id.uuidString,
                        ])))

                // Generate tokens inside actor isolation. `messages` carries
                // non-Sendable `Chat.Message` instances (UserInput.Image and
                // .Video are not Sendable), so route the array through
                // perform(nonSendable:_:) which boxes it across the actor hop.
                try await container.perform(nonSendable: messages) { context, messages in
                    // Render the prompt through the model's UserInputProcessor.
                    let userInput = UserInput(chat: messages)
                    let input = try await context.processor.prepare(input: userInput)

                    // Resolve the per-instance configuration. Held strictly as
                    // a local; it never lands in context.configuration or
                    // Executor.Configuration, so two instances with the same id
                    // but different resolvers don't cross-contaminate through
                    // the shared caches. Identity is read from
                    // context.configuration (above, at load time) and never
                    // from `resolved`.
                    let configData = try? Data(
                        contentsOf:
                            context.configuration.modelDirectory
                            .appendingPathComponent("config.json"))
                    let modelType =
                        configData.flatMap {
                            try? JSONDecoder.json5().decode(
                                BaseConfiguration.self, from: $0
                            ).modelType
                        } ?? ""
                    let descriptor = ModelDescriptor(
                        modelType: modelType,
                        modelId: modelID,
                        configData: configData,
                        tokenizer: context.tokenizer)
                    let resolved = configurationResolver.resolve(
                        context.configuration, for: descriptor)

                    // Capability gate. When the caller omits `.reasoning`
                    // but the resolved configuration carries a reasoning
                    // config, the model must not be allowed to think:
                    //
                    // - Toggleable strategies (`.templateFlag`) re-render the
                    //   prompt with thinking off (handled below per path).
                    // - Non-suppressible strategies (`.alwaysOn`) raise
                    //   `unsupportedCapability` BEFORE generation, regardless
                    //   of which path (tools / schema / unconstrained) the
                    //   request would otherwise take. The throw is
                    //   path-independent so a tool-calling or schema-guided
                    //   request on a model that always reasons surfaces the
                    //   same typed error the unconstrained path does, never a
                    //   silent leak through the grammar's malformed-output
                    //   fallback.
                    try validateReasoningCapability(
                        declaresReasoning: declaresReasoning, resolved: resolved)

                    // Reasoning is only consumed by the unconstrained path
                    // (no tools, no schema). On the guided/tool paths the
                    // grammar already constrains output, so suppression-prep
                    // would be wasted work.
                    let mayRunReasoningPath =
                        request.enabledToolDefinitions.isEmpty && request.schema == nil

                    // When .reasoning is OMITTED on the unconstrained path,
                    // re-render the prompt with thinking off so the model
                    // doesn't emit `<think>`. Toggleable-only;
                    // .alwaysOn was already rejected above.
                    let suppressedInput: LMInput?
                    if mayRunReasoningPath, !declaresReasoning,
                        let suppressionConfig = resolved.reasoningConfig
                    {
                        suppressedInput = try await Self.preparedInput(
                            messages: messages, config: suppressionConfig,
                            thinkingEnabled: false, processor: context.processor,
                            cannotDisableMessage:
                                "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."
                        )
                    } else {
                        suppressedInput = nil
                    }

                    let reasoningSetup:
                        (input: LMInput, config: ReasoningConfig, primedInside: Bool)?
                    if mayRunReasoningPath, declaresReasoning,
                        let reasoningConfig = resolved.reasoningConfig
                    {
                        let thinkingEnabled = Self.thinkingEnabled(
                            for: request.contextOptions.reasoningLevel)
                        let reasoningInput = try await Self.preparedInput(
                            messages: messages, config: reasoningConfig,
                            thinkingEnabled: thinkingEnabled, processor: context.processor,
                            cannotDisableMessage:
                                "This model always reasons; reasoning cannot be disabled via reasoningLevel."
                        )
                        reasoningSetup = (
                            reasoningInput, reasoningConfig,
                            Self.reasoningPrimedInside(
                                input: reasoningInput, config: reasoningConfig,
                                tokenizer: context.tokenizer)
                        )
                    } else {
                        reasoningSetup = nil
                    }

                    // The prompt actually fed into generation: the suppressed
                    // prompt when we're forcing thinking off, otherwise the
                    // baseline `input` rendered above.
                    let effectiveInput = suppressedInput ?? input

                    if !request.enabledToolDefinitions.isEmpty {
                        let completedNormally = try await runToolCalling(
                            request: request, messages: messages, modelID: modelID,
                            requestedMaxTokens: requestedMaxTokens,
                            requestedSamplingMode: requestedSamplingMode,
                            declaresReasoning: declaresReasoning, resolved: resolved,
                            entryID: entryID, toolCallsEntryID: toolCallsEntryID,
                            reasoningEntryID: reasoningEntryID, context: context,
                            channel: channel)
                        guard completedNormally else { return }
                    } else if let schemaJSON {
                        try await runGuidedGeneration(
                            schemaJSON: schemaJSON, input: input, modelID: modelID,
                            requestedMaxTokens: requestedMaxTokens, entryID: entryID,
                            context: context, channel: channel)
                    } else {
                        try await runTextGeneration(
                            reasoningSetup: reasoningSetup,
                            fallbackInput: effectiveInput,
                            requestedMaxTokens: requestedMaxTokens,
                            requestedTemperature: request.generationOptions.temperature,
                            samplingMode: requestedSamplingMode,
                            responseEntryID: entryID,
                            reasoningEntryID: reasoningEntryID,
                            context: context,
                            channel: channel
                        )
                    }

                    Stream.gpu.synchronize()
                }
            } catch is CancellationError {
                // Synchronize GPU before rethrowing to ensure in-flight operations complete.
                // Without this, process teardown can crash with Metal assertions.
                Stream.gpu.synchronize()
                throw CancellationError()
            } catch {
                // Synchronize GPU before rethrowing to ensure in-flight operations complete
                Stream.gpu.synchronize()
                // Re-map xgrammar errors to typed `LanguageModelError` cases
                // where the cause is provably user input (see `mapGrammarError`).
                // Internal-shim failures pass through unchanged.
                if let grammarError = error as? GrammarError {
                    throw Self.mapGrammarError(grammarError)
                }
                throw error
            }
        }

        /// Enforces the reasoning capability gate before generation: when
        /// the caller omitted `.reasoning` but the resolved configuration
        /// carries a non-suppressible (`.alwaysOn`) reasoning config, throws
        /// `unsupportedCapability` up front -- before any of the three
        /// generation paths run, so a tool-calling or schema-guided request
        /// surfaces the same typed error the unconstrained path would.
        ///
        /// - Parameters:
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        /// - Throws: `LanguageModelError.unsupportedCapability(.reasoning)`
        ///   when the model always reasons and `.reasoning` wasn't declared.
        private func validateReasoningCapability(
            declaresReasoning: Bool, resolved: ModelConfiguration
        ) throws {
            guard !declaresReasoning, let suppressionConfig = resolved.reasoningConfig else {
                return
            }
            do {
                _ = try suppressionConfig.promptStrategy
                    .additionalContext(forThinkingEnabled: false)
            } catch ReasoningError.cannotDisableReasoning {
                throw LanguageModelError.unsupportedCapability(
                    LanguageModelError.UnsupportedCapability(
                        capability: .reasoning,
                        debugDescription:
                            "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."
                    ))
            }
        }

        /// Prepares the shared grammar-constraint machinery for the
        /// tool-calling and guided-generation paths: the model-keyed
        /// xgrammar tokenizer, a compiled constraint for `constraintSource`,
        /// and the token-budget/logit-bias values `GuidedGenerationLoop.run`
        /// needs to steer generation toward a structural close.
        ///
        /// - Parameters:
        ///   - modelID: The model identifier for the tokenizer/constraint/bias caches.
        ///   - context: The loaded model context (tokenizer, configuration).
        ///   - kind: Which constructor `constraintSource` compiles under --
        ///     keeps a JSON-schema source and a structural-tag source from
        ///     ever aliasing in the constraint cache even if their text collides.
        ///   - constraintSource: The grammar/schema text the constraint is
        ///     compiled from (the tool-calling grammar, or the developer's
        ///     JSON schema).
        ///   - reserveEstimateSource: The text `CompletionReserve.estimate`
        ///     sizes the reserve from. Equal to `constraintSource` on the
        ///     guided-generation path; on the tool-calling path this is the
        ///     inner JSON envelope alone (the wrapper tokens around it are
        ///     small and fixed, so estimating from them would add noise
        ///     rather than accuracy).
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        /// - Throws: Whatever `makeXgTokenizer`/`makeConstraint` throw.
        /// - Returns: The tokenizer, compiled constraint, resolved token
        ///   budget, and the closing/whitespace biases and reserve sizes for
        ///   `GuidedGenerationLoop.run`.
        private func prepareConstraintSetup(
            modelID: String,
            context: ModelContext,
            kind: ConstraintKind,
            constraintSource: String,
            reserveEstimateSource: String,
            requestedMaxTokens: Int?
        ) async throws -> ConstraintSetup {
            let xgTokenizer = try await MLXLanguageModel.makeXgTokenizer(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let constraint = try await MLXLanguageModel.makeConstraint(
                modelID: modelID,
                kind: kind,
                source: constraintSource,
                tokenizer: xgTokenizer,
                hostTokenizer: context.tokenizer,
                fastForward: true
            )

            // Always partition into zones -- the grammar has wiggle room
            // (JSON whitespace before the outer `}`, whitespace before
            // `\n</tool_call>`) that open-source models tend to exploit
            // into infinite loops when not pushed toward structural close.
            // Use the caller's budget when set, otherwise the Executor's
            // default.
            let maxTokens = requestedMaxTokens ?? Self.defaultMaxTokens
            let bias = await MLXLanguageModel.makeTokenizerBias(
                modelID: modelID,
                tokenizer: context.tokenizer
            )
            let structuralReserve = CompletionReserve.estimate(
                schemaJSON: reserveEstimateSource,
                tokenizer: context.tokenizer
            )
            // The structural reserve is the bare minimum tokens for the
            // JSON skeleton (empty strings). Use the larger of 3x structural
            // minimum or 25% of maxTokens, so closing bias activates early
            // enough for the model to generate actual content in closing
            // fields.
            let completionReserve = Swift.max(
                structuralReserve * 3, maxTokens / 4)
            // Hard reserve: the point at which we force structural
            // completion by penalizing non-closing tokens. Must be larger
            // than the raw estimate because grammar-forced key names (FF
            // tokens) and model-inserted whitespace cost more tokens than
            // the compact minimal JSON string.
            let hardReserve = structuralReserve * 8

            return ConstraintSetup(
                xgTokenizer: xgTokenizer, constraint: constraint, maxTokens: maxTokens,
                closingBias: bias.closing, completionReserve: completionReserve,
                hardReserve: hardReserve, whitespaceBias: bias.whitespace,
                whitespaceTokenIDs: bias.whitespaceTokenIDs
            )
        }

        /// Metadata key signaling the model's output was cut off before
        /// completing naturally (budget exhausted mid-thought or
        /// mid-structure), so a consumer doesn't mistake a partial answer
        /// for the model's chosen response.
        private static let incompleteOutputMetadataKey = "incompleteOutput"

        /// Sends the incomplete-output metadata signal for `entryID`.
        ///
        /// - Parameters:
        ///   - entryID: The response entry the signal applies to.
        ///   - channel: The generation channel to send the signal on.
        private static func sendIncompleteOutputMetadata(
            entryID: String, channel: LanguageModelExecutorGenerationChannel
        ) async {
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateMetadata([Self.incompleteOutputMetadataKey: true])))
        }

        /// Sends a single text delta for `entryID`.
        ///
        /// - Parameters:
        ///   - text: The delta text to append.
        ///   - entryID: The response entry to stream into.
        ///   - channel: The generation channel to send the delta on.
        private static func sendTextDelta(
            _ text: String, entryID: String, channel: LanguageModelExecutorGenerationChannel
        ) async {
            await channel.send(
                .response(entryID: entryID, action: .appendText(text, tokenCount: 1)))
        }

        /// Sends the authoritative `.updateUsage` event for `entryID`.
        /// `cachedTokenCount` is always 0 -- prompt-cache reuse isn't
        /// implemented, so every request prefills its full prompt.
        ///
        /// - Parameters:
        ///   - entryID: The response entry the usage applies to.
        ///   - promptTokenCount: The prompt's total token count.
        ///   - outputTokenCount: The generated output's total token count.
        ///   - reasoningTokenCount: The subset of the output spent on
        ///     reasoning (0 on paths that don't reason); clamped to
        ///     `outputTokenCount`.
        ///   - channel: The generation channel to send the event on.
        private static func sendUsageUpdate(
            entryID: String,
            promptTokenCount: Int,
            outputTokenCount: Int,
            reasoningTokenCount: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateUsage(
                        input: .init(totalTokenCount: promptTokenCount, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: outputTokenCount,
                            reasoningTokenCount: Swift.min(reasoningTokenCount, outputTokenCount))
                    )))
        }

        /// Runs think-then-call Phase 1: unconstrained reasoning until
        /// `</think>`, whose token IDs prefill the constrained Phase 2.
        /// Sends the incomplete-output signal itself when cut off, since
        /// that path also needs its caller to skip its own tail GPU sync
        /// and return immediately (Phase 1 already synchronized on its way
        /// out).
        ///
        /// - Parameters:
        ///   - cfg: The think-then-call reasoning config (already gated by the caller).
        ///   - toolAwareInput: The tool-aware-template-rendered prompt.
        ///   - maxTokens: The resolved token budget for this request.
        ///   - request: The generation request (temperature/reasoning options).
        ///   - requestedSamplingMode: The caller's sampling mode override, if any.
        ///   - reasoningEntryID: The entry to stream reasoning segments into.
        ///   - entryID: The response entry (for the incomplete-output signal).
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever `runToolCallReasoningPhase` throws.
        /// - Returns: The reasoning token IDs (empty if Phase 1 wasn't
        ///   entered) and whether Phase 1 was cut off before closing.
        private func executeThinkThenCallPhase1(
            cfg: ReasoningConfig,
            toolAwareInput: LMInput,
            maxTokens: Int,
            request: LanguageModelExecutorGenerationRequest,
            requestedSamplingMode: MLXSamplingMode?,
            reasoningEntryID: String,
            entryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> (tokenIDs: [Int], cutOff: Bool) {
            let primedInside = Self.reasoningPrimedInside(
                input: toolAwareInput, config: cfg, tokenizer: context.tokenizer)
            let phase1 = try await runToolCallReasoningPhase(
                input: toolAwareInput, config: cfg,
                primedInside: primedInside, maxTokens: maxTokens,
                requestedTemperature: request.generationOptions.temperature,
                samplingMode: requestedSamplingMode,
                reasoningEntryID: reasoningEntryID,
                responseEntryID: entryID,
                context: context, channel: channel)
            guard phase1.closed else {
                // Cut off mid-thought (budget exhausted before `</think>`).
                // Don't prefill a truncated thought into the grammar —
                // signal and finish. Phase 1 already synchronized the GPU
                // on its way out.
                await Self.sendIncompleteOutputMetadata(entryID: entryID, channel: channel)
                return (phase1.tokenIDs, true)
            }
            return (phase1.tokenIDs, false)
        }

        /// Runs think-then-call Phase 2: constrained tool-grammar generation
        /// continuing from Phase 1's (possibly empty) reasoning tokens.
        ///
        /// - Parameters:
        ///   - phase2Input: The prompt to generate from (tool-aware input,
        ///     plus any Phase 1 reasoning tokens prefilled).
        ///   - phase2MaxTokens: The remaining token budget for this phase.
        ///   - context: The loaded model context.
        ///   - setup: The constraint/bias/reserve setup from `prepareConstraintSetup`.
        /// - Returns: The buffered output text, the generated token count
        ///   (nil if generation threw before completing), and whether
        ///   output was incomplete.
        private func executeToolCallingPhase2(
            phase2Input: LMInput,
            phase2MaxTokens: Int,
            context: ModelContext,
            setup: ConstraintSetup
        ) throws -> (outputBuffer: String, generatedTokenCount: Int?, incomplete: Bool) {
            var outputBuffer = ""
            let result = try runGuidedGenerationLoop(
                input: phase2Input, context: context, setup: setup, maxTokens: phase2MaxTokens
            ) { text in
                outputBuffer += text
                return !Task.isCancelled
            }
            return (outputBuffer, result.generatedTokenCount, result.incomplete)
        }

        /// Runs `GuidedGenerationLoop.run` with the shared error handling
        /// both the tool-calling and guided-generation paths need: catches
        /// `GuidedGenerationError.incompleteOutput` and reports it as a flag
        /// rather than a thrown error, since best-effort output has
        /// typically already been emitted by the time the grammar exhausts
        /// its budget.
        ///
        /// - Parameters:
        ///   - input: The prompt to generate from.
        ///   - context: The loaded model context.
        ///   - setup: The constraint/bias/reserve setup from `prepareConstraintSetup`.
        ///   - maxTokens: The token budget for this call -- not always
        ///     `setup.maxTokens`, since the tool-calling path's Phase 2 may
        ///     run under a reduced budget after Phase 1 reasoning consumed
        ///     part of it.
        ///   - onText: Called with each generated text delta; return `false`
        ///     to stop generation early.
        /// - Returns: The generated token count (nil if generation threw
        ///   before completing) and whether output was incomplete.
        private func runGuidedGenerationLoop(
            input: LMInput,
            context: ModelContext,
            setup: ConstraintSetup,
            maxTokens: Int,
            onText: @escaping (String) -> Bool
        ) throws -> (generatedTokenCount: Int?, incomplete: Bool) {
            var incomplete = false
            var generatedTokenCount: Int?
            do {
                generatedTokenCount = try GuidedGenerationLoop.run(
                    input: input,
                    context: context,
                    constraint: setup.constraint,
                    maxTokens: maxTokens,
                    vocabSize: Int(setup.xgTokenizer.vocabSize),
                    completionReserve: setup.completionReserve,
                    hardReserve: setup.hardReserve,
                    closingBias: setup.closingBias,
                    whitespaceBias: setup.whitespaceBias,
                    whitespaceTokenIDs: setup.whitespaceTokenIDs,
                    emit: onText
                )
            } catch GuidedGenerationError.incompleteOutput {
                // Grammar exhausted maxTokens before reaching a stop state.
                // Deltas already emitted (or buffered) are best-effort output.
                incomplete = true
            }
            return (generatedTokenCount, incomplete)
        }

        /// Tool-calling generation path. Continuation rounds from
        /// `LanguageModelSession`'s auto-loop re-enter this path too: the
        /// transcript's prior tool-call and tool-output entries are replayed
        /// into the prompt by `TranscriptConverter`, so the model sees the
        /// results and chooses another tool call or the final answer
        /// (multi-turn tool calling).
        ///
        /// Forces the model to emit a JSON object matching one of the
        /// declared tools -- including a synthetic "final answer" tool whose
        /// arguments carry the free-text response. After generation, parses
        /// the output to route to either a toolCallDelta (real tool) or
        /// textDelta (final answer) event. Buffers the full output before
        /// emitting; streaming within the final-answer path
        /// (reparse-each-delta) is not yet implemented.
        ///
        /// - Parameters:
        ///   - request: The generation request; supplies the enabled tools,
        ///     developer schema, and generation options.
        ///   - messages: The rendered chat messages for this round.
        ///   - modelID: The model identifier for constraint/tokenizer caches.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - requestedSamplingMode: The caller's sampling mode override, if any.
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        ///   - entryID: The response entry to stream output into.
        ///   - toolCallsEntryID: The entry to stream tool-call events into.
        ///   - reasoningEntryID: The entry to stream think-then-call reasoning into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever the grammar/tokenizer/generation calls throw.
        /// - Returns: `false` only when think-then-call Phase 1 was cut off
        ///   before `</think>` closed -- Phase 1 already synchronized the
        ///   GPU on its way out, so the caller must skip its own tail
        ///   `Stream.gpu.synchronize()` and return immediately. `true` on
        ///   every other exit.
        private func runToolCalling(
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            modelID: String,
            requestedMaxTokens: Int?,
            requestedSamplingMode: MLXSamplingMode?,
            declaresReasoning: Bool,
            resolved: ModelConfiguration,
            entryID: String,
            toolCallsEntryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> Bool {
            let finalAnswerDef = FinalAnswerTool.makeToolDefinition(
                responseSchema: request.schema
            )
            let allTools =
                Array(request.enabledToolDefinitions) + [finalAnswerDef]

            // Re-tokenize using the model's native tool-aware chat
            // template (Qwen/Llama/Phi/Gemma all ship one in their
            // tokenizer_config.json). This is what teaches the model
            // *what* tools exist and how to decide between them; the
            // grammar constraint below only enforces the *shape* of
            // whatever tool call it emits.
            let toolSpecs = try ToolCallingConversions.makeToolSpecs(
                from: allTools)
            let tokenizerMessages = DefaultMessageGenerator().generate(
                messages: messages)

            // Think-then-call is gated to the enable_thinking
            // family (Qwen3/QwQ): their template both renders the tool
            // block AND honors `enable_thinking`. R1-style `.alwaysOn`
            // models are tool-blind (template ignores `tools:`), so
            // they fall through to the single-phase path unchanged;
            // thinking-disabled requests stay single-phase too.
            let thinkThenCallConfig: ReasoningConfig? = {
                guard declaresReasoning,
                    let cfg = resolved.reasoningConfig,
                    case .templateFlag = cfg.promptStrategy,
                    Self.thinkingEnabled(
                        for: request.contextOptions.reasoningLevel) != false
                else { return nil }
                return cfg
            }()
            // Thread `enable_thinking` through the tool-aware template
            // (3-arg form) so the prompt is both tool-aware and
            // thinking-primed; nil on the single-phase path.
            let reasoningContext = try thinkThenCallConfig.flatMap {
                try $0.promptStrategy.additionalContext(
                    forThinkingEnabled: Self.thinkingEnabled(
                        for: request.contextOptions.reasoningLevel))
            }
            let toolAwareTokens = try context.tokenizer.applyChatTemplate(
                messages: tokenizerMessages,
                tools: toolSpecs,
                additionalContext: reasoningContext
            )
            let toolAwareInput = LMInput(tokens: MLXArray(toolAwareTokens))

            let toolCallingGrammar =
                try SchemaConverter.encodeToolCallingGrammar(
                    tools: allTools
                )
            // The inner JSON envelope is still needed separately to
            // seed `CompletionReserve` -- the wrapper tokens
            // (`<tool_call>`, two `\n`s, `</tool_call>`) are small
            // and fixed, so padding the reserve with their
            // tokenized size adds noise rather than accuracy.
            let toolCallingEnvelopeJSON =
                try SchemaConverter.encodeToolCallingEnvelopeJSON(
                    tools: allTools
                )

            let setup = try await prepareConstraintSetup(
                modelID: modelID,
                context: context,
                kind: .structuralTag,
                constraintSource: toolCallingGrammar,
                reserveEstimateSource: toolCallingEnvelopeJSON,
                requestedMaxTokens: requestedMaxTokens
            )

            // PHASE 1 (think-then-call): reason unconstrained until
            // `</think>`, retaining the token IDs to prefill into the
            // constrained phase below. Empty on the single-phase path.
            var reasoningTokenIDs: [Int] = []
            if let cfg = thinkThenCallConfig {
                let phase1 = try await executeThinkThenCallPhase1(
                    cfg: cfg, toolAwareInput: toolAwareInput, maxTokens: setup.maxTokens,
                    request: request, requestedSamplingMode: requestedSamplingMode,
                    reasoningEntryID: reasoningEntryID, entryID: entryID,
                    context: context, channel: channel)
                reasoningTokenIDs = phase1.tokenIDs
                guard !phase1.cutOff else { return false }
            }

            // Phase 2 continues from the model's completed reasoning;
            // carry the raw IDs (no decode/re-encode) so the grammar
            // starts from the exact post-`</think>` state.
            let phase2Input =
                reasoningTokenIDs.isEmpty
                ? toolAwareInput
                : LMInput(
                    tokens: MLXArray(toolAwareTokens + reasoningTokenIDs))
            // Shared budget (match the unconstrained path): the
            // envelope continues under the remaining budget, floored
            // at the completion reserve so it always has room to close
            // the tool call.
            let phase2MaxTokens =
                reasoningTokenIDs.isEmpty
                ? setup.maxTokens
                : Swift.max(
                    setup.maxTokens - reasoningTokenIDs.count, setup.completionReserve)

            let phase2 = try executeToolCallingPhase2(
                phase2Input: phase2Input, phase2MaxTokens: phase2MaxTokens,
                context: context, setup: setup)
            let outputBuffer = phase2.outputBuffer
            let incomplete = phase2.incomplete
            let generatedTokenCount = phase2.generatedTokenCount

            try await emitToolCallingEvent(
                outputBuffer: outputBuffer,
                userResponseSchema: request.schema,
                entryID: entryID,
                toolCallsEntryID: toolCallsEntryID,
                channel: channel
            )

            if let generatedTokenCount {
                // Output total spans both phases (reasoning + envelope).
                let reasoningCount = reasoningTokenIDs.count
                let totalOutput = generatedTokenCount + reasoningCount
                await Self.sendUsageUpdate(
                    entryID: entryID,
                    promptTokenCount: toolAwareInput.text.tokens.size,
                    outputTokenCount: totalOutput,
                    reasoningTokenCount: reasoningCount,
                    channel: channel)
            }

            if incomplete {
                await Self.sendIncompleteOutputMetadata(entryID: entryID, channel: channel)
            }

            return true
        }

        /// Guided-generation path: streams text deltas constrained to the
        /// developer's JSON schema as they arrive.
        ///
        /// - Parameters:
        ///   - schemaJSON: The developer-supplied JSON schema.
        ///   - input: The rendered prompt to generate from.
        ///   - modelID: The model identifier for constraint/tokenizer caches.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - entryID: The response entry to stream output into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever the grammar/generation calls throw.
        private func runGuidedGeneration(
            schemaJSON: String,
            input: LMInput,
            modelID: String,
            requestedMaxTokens: Int?,
            entryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let setup = try await prepareConstraintSetup(
                modelID: modelID,
                context: context,
                kind: .json,
                constraintSource: schemaJSON,
                reserveEstimateSource: schemaJSON,
                requestedMaxTokens: requestedMaxTokens
            )

            // GuidedGenerationLoop.run's emit closure is synchronous (for
            // performance -- it runs inside the tight MLX generation loop).
            // channel.send is async. Bridge via an AsyncStream + concurrent
            // forwarder so text deltas stream to the channel in order.
            let (textStream, textContinuation) = AsyncStream<String>
                .makeStream()
            async let forwarder: Void = {
                for await text in textStream {
                    await Self.sendTextDelta(text, entryID: entryID, channel: channel)
                }
            }()

            let result = try runGuidedGenerationLoop(
                input: input, context: context, setup: setup, maxTokens: setup.maxTokens
            ) { text in
                textContinuation.yield(text)
                return !Task.isCancelled
            }
            let generatedTokenCount = result.generatedTokenCount
            let incomplete = result.incomplete
            textContinuation.finish()
            await forwarder

            if let generatedTokenCount {
                await Self.sendUsageUpdate(
                    entryID: entryID,
                    promptTokenCount: input.text.tokens.size,
                    outputTokenCount: generatedTokenCount,
                    reasoningTokenCount: 0,
                    channel: channel)
            }

            if incomplete {
                await Self.sendIncompleteOutputMetadata(entryID: entryID, channel: channel)
            }
        }

        /// Unconstrained text generation. Used on the no-tools/no-schema
        /// path when the model has no reasoning config to route through.
        private func runUnconstrained(
            input: LMInput,
            requestedMaxTokens: Int?,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?,
            entryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Use a finite default when the framework doesn't specify a
            // token limit; there's no grammar to stop the model naturally.
            let params = Self.makeParameters(
                maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
                requestedTemperature: requestedTemperature,
                samplingMode: samplingMode
            )

            for await generation in try generate(
                input: input,
                parameters: params,
                context: context
            ) {
                try Task.checkCancellation()
                switch generation {
                case .chunk(let text):
                    await Self.sendTextDelta(text, entryID: entryID, channel: channel)
                case .info(let info):
                    // MLX-LM emits one .info event at end-of-generation with
                    // authoritative scalar token counts (`promptTokenCount`
                    // is the prompt; `generationTokenCount` is the
                    // model-generated completion -- see Evaluate.swift's
                    // `GenerateCompletionInfo` definition).
                    await Self.sendUsageUpdate(
                        entryID: entryID,
                        promptTokenCount: info.promptTokenCount,
                        outputTokenCount: info.generationTokenCount,
                        reasoningTokenCount: 0,
                        channel: channel)
                case .toolCall(_):
                    break
                }
            }
        }

        /// Dispatches the no-tools/no-schema path: reasoning routing when a
        /// config resolved, otherwise plain unconstrained text.
        private func runTextGeneration(
            reasoningSetup: (input: LMInput, config: ReasoningConfig, primedInside: Bool)?,
            fallbackInput: LMInput,
            requestedMaxTokens: Int?,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?,
            responseEntryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            if let reasoning = reasoningSetup {
                try await runReasoning(
                    input: reasoning.input,
                    reasoningConfig: reasoning.config,
                    primedInside: reasoning.primedInside,
                    requestedMaxTokens: requestedMaxTokens,
                    requestedTemperature: requestedTemperature,
                    samplingMode: samplingMode,
                    responseEntryID: responseEntryID,
                    reasoningEntryID: reasoningEntryID,
                    context: context,
                    channel: channel)
            } else {
                try await runUnconstrained(
                    input: fallbackInput,
                    requestedMaxTokens: requestedMaxTokens,
                    requestedTemperature: requestedTemperature,
                    samplingMode: samplingMode,
                    entryID: responseEntryID,
                    context: context,
                    channel: channel)
            }
        }

        /// Reasoning-aware unconstrained generation.
        ///
        /// Routes thinking delimited by the model's reasoning markers to
        /// `.reasoning` events and the rest to `.response`, using a raw
        /// `generateTokens` stream + a self-owned `NaiveStreamingDetokenizer`
        /// (bypassing `ToolCallProcessor`) so the scanner sees clean detokenized
        /// text — no second fragmentation source — and the loop sees real token
        /// IDs for an accurate reasoning token count.
        private func runReasoning(
            input: LMInput,
            reasoningConfig: ReasoningConfig,
            primedInside: Bool,
            requestedMaxTokens: Int?,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?,
            responseEntryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let params = Self.makeParameters(
                maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
                requestedTemperature: requestedTemperature,
                samplingMode: samplingMode
            )

            var emitter = ReasoningEventEmitter(
                config: reasoningConfig, primedInside: primedInside)
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            var reasoningTokenCount = 0
            var completionInfo: GenerateCompletionInfo?

            for await generation in try generateTokens(
                input: input, parameters: params, context: context
            ) {
                try Task.checkCancellation()
                switch generation {
                case .token(let token):
                    // One `.token` == one real token, so this is a true token
                    // count (not a chunk count). Attribute it to reasoning while
                    // the scanner is inside a thinking span. This generously
                    // counts the closing-delimiter tokens as reasoning (the
                    // emitter only flips state once `process` consumes the full
                    // `</think>`); it remains a true token count and the clamp
                    // below keeps it ≤ total.
                    if emitter.isInsideReasoning {
                        reasoningTokenCount += 1
                    }
                    detokenizer.append(token: token)
                    if let chunk = detokenizer.next() {
                        await Self.sendSegments(
                            emitter.process(chunk), responseEntryID: responseEntryID,
                            reasoningEntryID: reasoningEntryID, channel: channel)
                    }
                case .info(let info):
                    completionInfo = info
                }
            }

            await Self.sendSegments(
                emitter.finalize(), responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID, channel: channel)

            // If generation ended while still inside a thinking block, the model
            // was cut off mid-thought (e.g. it exhausted the token budget before
            // emitting `</think>`). Signal it so a consumer doesn't mistake an
            // empty or partial answer for the model's chosen response — mirrors
            // the guided path's `incompleteOutput` convention.
            if emitter.isInsideReasoning {
                await Self.sendIncompleteOutputMetadata(
                    entryID: responseEntryID, channel: channel)
            }

            if let info = completionInfo {
                // Single source of truth for usage: one authoritative
                // `.updateUsage` (the framework's aggregator replaces
                // wholesale, so we must not also rely on per-delta
                // auto-summing).
                await Self.sendUsageUpdate(
                    entryID: responseEntryID,
                    promptTokenCount: info.promptTokenCount,
                    outputTokenCount: info.generationTokenCount,
                    reasoningTokenCount: reasoningTokenCount,
                    channel: channel)
            }
        }

        /// Routes each of `segments` to the appropriate channel entry, in order.
        ///
        /// - Parameters:
        ///   - segments: The scanned segments to route, in emission order.
        ///   - responseEntryID: The entry `.response` segments stream into.
        ///   - reasoningEntryID: The entry `.reasoning` segments stream into.
        ///   - channel: The generation channel to send events on.
        private static func sendSegments(
            _ segments: [ReasoningEventEmitter.Segment],
            responseEntryID: String,
            reasoningEntryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            for segment in segments {
                switch segment {
                case .reasoning(let text):
                    await channel.send(
                        .reasoning(
                            entryID: reasoningEntryID,
                            action: .appendText(text, tokenCount: 1)))
                case .response(let text):
                    await channel.send(
                        .response(
                            entryID: responseEntryID,
                            action: .appendText(text, tokenCount: 1)))
                }
            }
        }

        /// Prepares an `LMInput` for the unconstrained reasoning path with
        /// thinking explicitly on, off, or unspecified. Maps the package-
        /// internal `cannotDisableReasoning` to the framework's
        /// `unsupportedCapability` so always-on models surface a typed error
        /// before generation rather than leaking `<think>` into `.response`.
        private static func preparedInput(
            messages: [Chat.Message],
            config: ReasoningConfig,
            thinkingEnabled: Bool?,
            processor: any UserInputProcessor,
            cannotDisableMessage: String
        ) async throws -> LMInput {
            let additionalContext: [String: any Sendable]?
            do {
                additionalContext = try config.promptStrategy
                    .additionalContext(forThinkingEnabled: thinkingEnabled)
            } catch ReasoningError.cannotDisableReasoning {
                throw LanguageModelError.unsupportedCapability(
                    LanguageModelError.UnsupportedCapability(
                        capability: .reasoning,
                        debugDescription: cannotDisableMessage))
            }
            return try await processor.prepare(
                input: UserInput(chat: messages, additionalContext: additionalContext))
        }

        /// Maps a requested reasoning level to a thinking on/off/unspecified
        /// flag. `nil` (no opinion) defers to the strategy's default; any
        /// concrete level means "think" (v1 does not modulate depth); only the
        /// package convention `.custom("no_think")` means "off".
        static func thinkingEnabled(for level: ContextOptions.ReasoningLevel?) -> Bool? {
            guard let level else { return nil }
            switch level {
            case .light, .moderate, .deep:
                return true
            case .custom(let value):
                let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return normalized == "no_think" ? false : true
            @unknown default:
                // A future level we don't recognize → default to thinking on.
                return true
            }
        }

        /// Decodes the rendered prompt's tail and asks whether it ends inside an
        /// open reasoning block (some model families prefill the opening
        /// delimiter).
        private static func reasoningPrimedInside(
            input: LMInput, config: ReasoningConfig, tokenizer: any Tokenizer
        ) -> Bool {
            let tokens = input.text.tokens.asArray(Int.self)
            let renderedTail = tokenizer.decode(tokenIds: Array(tokens.suffix(64)))
            return ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: renderedTail, config: config)
        }

        /// Think-then-call Phase 1: generate reasoning unconstrained until
        /// the model closes its thinking block, routing reasoning text to
        /// `.reasoning` events and retaining the raw token IDs to prefill into the
        /// constrained Phase 2.
        ///
        /// Uses the `Task`-returning `generateTokensTask` so the GPU loop is
        /// cancelled and drained at the phase boundary — without that, Phase 2's
        /// prefill could overlap Phase 1's in-flight forward pass on the shared
        /// `Stream` and trip a Metal command-buffer assertion.
        ///
        /// Returns the accumulated token IDs and whether `</think>` actually
        /// closed. If it did not (budget exhausted mid-thought), the caller must
        /// skip Phase 2 rather than prefill a truncated thought into the grammar.
        private func runToolCallReasoningPhase(
            input: LMInput,
            config: ReasoningConfig,
            primedInside: Bool,
            maxTokens: Int,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?,
            reasoningEntryID: String,
            responseEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> (tokenIDs: [Int], closed: Bool) {
            let params = Self.makeParameters(
                maxTokens: maxTokens,
                requestedTemperature: requestedTemperature,
                samplingMode: samplingMode
            )
            var collector = ReasoningTokenCollector(
                config: config, primedInside: primedInside, tokenizer: context.tokenizer
            )

            let (stream, task) = try generateTokensTask(
                input: input, parameters: params, context: context)
            var closed = false
            do {
                for await generation in stream {
                    try Task.checkCancellation()
                    guard case .token(let token) = generation else { continue }
                    await Self.sendSegments(
                        collector.ingest(token), responseEntryID: responseEntryID,
                        reasoningEntryID: reasoningEntryID, channel: channel)
                    if collector.shouldStopAfterReasoning {
                        closed = true
                        break
                    }
                }
            } catch {
                // Drain the generation task before propagating, but do NOT sync
                // here: respond's outer `catch` is the single GPU-sync point for
                // this exit path. Keep one clean GPU sync per exit path —
                // cascading syncs across nested catches can race the Metal
                // command-buffer state during teardown.
                task.cancel()
                _ = await task.value
                throw error
            }
            // Drain the generation task before Phase 2 reuses the Stream.
            task.cancel()
            _ = await task.value
            Stream.gpu.synchronize()

            await Self.sendSegments(
                collector.finalize(), responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID, channel: channel)
            return (collector.reasoningTokenIDs, closed)
        }

        /// Parses a tool-calling envelope JSON object and emits the
        /// appropriate channel event.
        ///
        /// The output buffer is expected to be a JSON object matching the
        /// shape `{"name": <tool-name>, "arguments": <args>}`. Grammars from
        /// `SchemaConverter.encodeToolCallingGrammar` guarantee either that
        /// shape directly (bare JSON) or that shape wrapped in Qwen's
        /// `<tool_call>\n...\n</tool_call>` special-token delimiters --
        /// `unwrapToolCallMarkers` below strips the wrapper if present. The
        /// best-effort fallback only exists so that unexpected upstream
        /// changes don't silently swallow output.
        ///
        /// - If `name` is the synthetic final-answer tool:
        ///   - With no developer response schema: unwrap `arguments.response`
        ///     into a `.textDelta` event.
        ///   - With a developer response schema: re-serialize `arguments`
        ///     back to JSON text and emit as a single `.textDelta`. The
        ///     session's normal response-parsing path will decode the JSON
        ///     through the developer's `GenerationSchema`.
        /// - If `name` is any real tool: emit a single `.toolCallDelta`
        ///   with the arguments JSON and a freshly minted toolCallID.
        ///
        /// `entryID` and `toolCallsEntryID` must be distinct: SKILL.md requires
        /// `.response` and `.toolCalls` to live in separate transcript entries.
        private func emitToolCallingEvent(
            outputBuffer: String,
            userResponseSchema: GenerationSchema?,
            entryID: String,
            toolCallsEntryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let unwrapped = Self.unwrapToolCallMarkers(outputBuffer)
            let data = Data(unwrapped.utf8)
            guard
                let obj = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                let name = obj[ToolCallEnvelopeKey.name] as? String
            else {
                // Malformed output. The grammar should have prevented this;
                // emit the raw buffer as text so failures surface loudly.
                await Self.sendTextDelta(outputBuffer, entryID: entryID, channel: channel)
                return
            }

            if name == FinalAnswerTool.toolName {
                let text: String
                if userResponseSchema == nil {
                    let args = obj[ToolCallEnvelopeKey.arguments] as? [String: Any]
                    text = (args?["response"] as? String) ?? ""
                } else if let args = obj[ToolCallEnvelopeKey.arguments],
                    let argsData = try? JSONSerialization.data(withJSONObject: args),
                    let argsStr = String(data: argsData, encoding: .utf8)
                {
                    text = argsStr
                } else {
                    text = ""
                }
                await Self.sendTextDelta(text, entryID: entryID, channel: channel)
            } else {
                guard
                    let args = obj[ToolCallEnvelopeKey.arguments],
                    let argsData = try? JSONSerialization.data(withJSONObject: args),
                    let argsStr = String(data: argsData, encoding: .utf8)
                else {
                    return
                }
                await channel.send(
                    .toolCalls(
                        entryID: toolCallsEntryID,
                        action: .toolCall(
                            id: UUID().uuidString,
                            name: name,
                            action: .appendArguments(argsStr, tokenCount: 1)
                        )
                    ))
            }
        }

        /// Strips Qwen-style `<tool_call>\n...\n</tool_call>` wrapper markers
        /// if present, returning the inner JSON text. Untouched if the buffer
        /// doesn't start with a wrapper -- the `bare_call` grammar alternative
        /// is valid output and parses directly.
        ///
        /// The inner newlines around the JSON come from the Qwen training
        /// format; we're tolerant of whitespace on either side of the markers
        /// so that tokenizer decoding quirks (extra spaces, missing newlines)
        /// don't cause the JSON parse to fail.
        private static func unwrapToolCallMarkers(_ buffer: String) -> String {
            let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            let openMarker = "<tool_call>"
            let closeMarker = "</tool_call>"
            guard trimmed.hasPrefix(openMarker) else { return buffer }
            let afterOpen = trimmed.dropFirst(openMarker.count)
            let inner: Substring
            if let closeRange = afterOpen.range(of: closeMarker, options: .backwards) {
                inner = afterOpen[afterOpen.startIndex ..< closeRange.lowerBound]
            } else {
                inner = afterOpen
            }
            return inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
