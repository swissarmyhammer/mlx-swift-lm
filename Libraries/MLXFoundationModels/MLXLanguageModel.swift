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
import MLXGuidedGeneration

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
/// import HuggingFace
/// import Tokenizers
///
/// let model = MLXLanguageModel(
///     configuration: ModelConfiguration(id: "mlx-community/Qwen2.5-3B-Instruct-4bit"),
///     capabilities: [.guidedGeneration, .toolCalling],
///     weightsLocation: { id in
///         // Resolve against the same HubClient cache the loader below downloads
///         // into, so the availability checks see the downloaded weights.
///         let cache = HubCache.default
///         guard let repo = Repo.ID(rawValue: id) else { return cache.cacheDirectory }
///         if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
///             let snapshot = try? cache.snapshotPath(
///                 repo: repo, kind: .model, commitHash: commit)
///         {
///             return snapshot
///         }
///         return cache.repoDirectory(repo: repo, kind: .model)
///     },
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
    /// not by the load path. The checks pass `configuration.name` (the
    /// repository id, or the path-based name of a directory), not
    /// ``modelID``, thus the closure always receives a path-shaped id.
    /// Injected so this module needs no HuggingFace path-resolution
    /// dependency.
    public let weightsLocation: @Sendable (String) -> URL

    /// Loads the model container for a configuration, forwarding download
    /// progress. Injected so this module carries no HuggingFace or
    /// swift-transformers dependency; the HuggingFace wiring lives in callers.
    ///
    /// This closure must honor cancellation while it is suspended. ``evict()``
    /// and ``evictAll()`` cancel the load task tied to this closure. A closure
    /// that ignores cancellation keeps running and keeps its caller waiting.
    public typealias ContainerLoader =
        @Sendable (
            _ configuration: ModelConfiguration,
            _ progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> ModelContainer

    private let load: ContainerLoader

    /// The revision `ModelConfiguration.Identifier.id(_:revision:)` takes when
    /// the caller gives none. A Hub model at this revision keeps the bare
    /// repository id as its ``modelID``, thus existing cache keys do not change.
    private static let defaultRevision = "main"

    /// Stable identity for the model cache, executor configuration, tokenizer
    /// caches, availability state, and the prompt cache. Derived from the
    /// configuration so it is the single place identity is defined.
    ///
    /// A Hub model at a revision other than ``defaultRevision`` gets
    /// `"\(id)@\(revision)"`, thus two revisions of one repository hold two
    /// cache entries and load two containers. A Hub model at the default
    /// revision, and a directory-backed model, keep `configuration.name`.
    ///
    /// On-disk resolution (`weightsLocation`) and download progress use
    /// `configuration.name`, not this value: both want a path-shaped id.
    public var modelID: String {
        Self.cacheIdentity(of: configuration)
    }

    /// The identity of a checkpoint in the model cache: `"\(id)@\(revision)"` for a Hub
    /// checkpoint at a revision other than ``defaultRevision``, else `configuration.name`.
    ///
    /// ``modelID`` names the model with it, and an MTP drafter
    /// (``MLXSpeculativeDecoding``) names its drafter checkpoint with it.
    ///
    /// - Parameter configuration: the checkpoint.
    /// - Returns: the identity.
    static func cacheIdentity(of configuration: ModelConfiguration) -> String {
        switch configuration.id {
        case .id(let id, let revision) where revision != defaultRevision:
            return "\(id)@\(revision)"
        case .id, .directory:
            return configuration.name
        }
    }

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

    /// The speculation of the generations of one response: the MTP drafter of
    /// ``speculativeDecoding``, which the shared model cache loads one time beside
    /// this model, or `nil` when the model has none.
    ///
    /// The drafter model leaves its container here, before the response enters the
    /// target's `ModelContainer.perform`, thus no generation nests a second lock. The
    /// loaded target picks the drafter loader, so a registration that matches the target
    /// loads the drafter family of the target as it loaded.
    ///
    /// The memory policy of ``speculativeDecoding`` weighs the drafter beside the loaded
    /// target first: on its byte estimate before the load, else on the loaded drafter. A
    /// drafter it does not admit gives `nil`, and the call reports
    /// `SpeculativeDecodingFallbackReason.memoryBudgetExceeded`; a drafter the estimate does
    /// not admit never loads.
    ///
    /// - Parameter container: the loaded target, from ``loadContainer()``.
    /// - Returns: the speculation, or `nil`.
    /// - Throws: the error of the drafter load, and `SpeculativeDecodingMemoryError` when the
    ///   memory policy does not admit the drafter and its action is `.fail`.
    func loadMTPSpeculation(for container: ModelContainer) async throws -> MTPSpeculation? {
        guard let speculativeDecoding else { return nil }
        let (loadDrafter, admitsDrafter) = try await container.perform { context in
            (
                speculativeDecoding.drafterLoader(for: context.model),
                try speculativeDecoding.admitsDrafterBeforeLoading(beside: context.model)
            )
        }
        guard admitsDrafter else { return Self.memoryBudgetFallback() }
        let drafter = try await Self.cache.loadDrafter(
            ModelCache.DrafterKey(modelID: modelID, drafterID: speculativeDecoding.drafterID)
        ) {
            try await loadDrafter(speculativeDecoding.drafterConfiguration)
        }
        let speculation = await speculativeDecoding.speculation(with: drafter)
        let admitsLoadedDrafter = try await container.perform(nonSendable: speculation) {
            context, speculation in
            try speculativeDecoding.admitsLoadedDrafter(speculation.drafter, beside: context.model)
        }
        return admitsLoadedDrafter ? speculation : Self.memoryBudgetFallback()
    }

    /// The speculation of a call whose memory policy did not admit the drafter: none. The
    /// call reports `SpeculativeDecodingFallbackReason.memoryBudgetExceeded`.
    ///
    /// - Returns: `nil`.
    private static func memoryBudgetFallback() -> MTPSpeculation? {
        SpeculativeDecodingOutcome.current?.record(fallbackReason: .memoryBudgetExceeded)
        return nil
    }

    private func makeContainerLoader() -> @Sendable () async throws -> ModelContainer {
        let configuration = self.configuration
        let load = self.load
        return {
            // One identifier per load attempt. Eviction cancels a load and
            // forgets it at once, so a replacement can start while this one is
            // still unwinding, and the progress observable needs to tell them
            // apart.
            let loadID = UUID()
            MLXDownloadProgress.reportStarted(modelID: configuration.name, loadID: loadID)
            defer { MLXDownloadProgress.reportEnded(modelID: configuration.name, loadID: loadID) }
            return try await load(configuration) { progress in
                MLXDownloadProgress.report(
                    progress: progress, modelID: configuration.name, loadID: loadID)
            }
        }
    }

    /// Gets or creates a cached GrammarTokenizer for the given model.
    static func makeGrammarTokenizer(
        modelID: String,
        tokenizer: any Tokenizer
    ) async throws -> GrammarTokenizer {
        try await cache.makeGrammarTokenizer(modelID: modelID, tokenizer: tokenizer)
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
    static func hasCachedGrammarTokenizer(modelID: String) async -> Bool {
        await cache.hasCachedGrammarTokenizer(modelID: modelID)
    }

    /// Evicts every cached model, tokenizer, constraint template, and per-model
    /// tokenizer bias, and every prompt cache entry, then clears the MLX buffer
    /// pool so the GPU memory of the weights and caches returns to the system.
    /// Subsequent requests reload from the on-disk cache. Cancels every registered
    /// in-flight load task, the same way ``evict()`` does for one model.
    ///
    /// Safe to call during in-flight `respond()`/`warmUp()` work that already holds
    /// a `ModelContainer`: that work keeps its own strong reference and synchronizes
    /// the GPU on exit, so dropping the cache's reference cannot free weights out
    /// from under a live kernel. Those weights free via ARC once that work returns,
    /// which is after this clear, so they stay in the pool until a later clear.
    ///
    /// A prompt cache entry in a running or queued spill is the same case: the spill
    /// job holds its buffers until its write ends, which is after this clear, so they
    /// too stay in the pool until a later clear. The clear can wait for a running
    /// spill write, because a safetensors save holds the MLX eval lock for its run.
    ///
    /// Work still inside its load phase is the other case. This call cancels every
    /// registered load task, so if the loader honors cancellation, an awaiting
    /// `preload()` or `respond()` fails instead of completing. The loader decides
    /// the error type, so it may not be `CancellationError`. The cache coalesces
    /// concurrent callers for one model onto a single load task, so one such
    /// failure reaches every caller waiting on that load, including callers that
    /// never asked for an eviction.
    public static func evictAll() async {
        await cache.evictAll()
        await ExecutorPromptCacheStore.current.evict(modelID: nil)
        // Last, and outside both actors: the released buffers reach the pool only when the last
        // reference above drops, and this call waits for any eval that holds the MLX eval lock.
        MLX.Memory.clearCache()
    }

    /// Drops this model and its prompt cache entries from the shared caches, then
    /// clears the MLX buffer pool so the GPU memory of its weights and caches returns
    /// to the system. The pool is process-wide, so the clear also returns the freed
    /// buffers of every other model. A subsequent `respond()`/`preload()` triggers a
    /// fresh load (reusing the on-disk snapshot if the model was previously downloaded).
    ///
    /// Safe to call during an in-flight `respond()`: that call retains its own
    /// `ModelContainer` and finishes normally; the weights free via ARC once it
    /// returns, which is after this clear, so they stay in the pool until a later
    /// clear. A prompt cache entry in a running or queued spill is the same case: the
    /// spill job holds its buffers until its write ends, so they too stay in the pool
    /// until a later clear, and the clear can wait for a running spill write, because
    /// a safetensors save holds the MLX eval lock for its run.
    ///
    /// Evicting a model with an in-flight load cancels that load's task.
    /// If the loader honors cancellation, the awaiting `preload()` or `respond()`
    /// call fails instead of completing. The loader decides the error type, so it
    /// may not be `CancellationError`.
    public func evict() async {
        await Self.cache.remove(modelID: modelID)
        await ExecutorPromptCacheStore.current.evict(modelID: modelID)
        // Last, and outside both actors: the released buffers reach the pool only when the last
        // reference above drops, and this call waits for any eval that holds the MLX eval lock.
        MLX.Memory.clearCache()
    }

    // MARK: - Prompt Cache Budgets and Release

    /// Sets the memory budget of the shared prompt cache, in bytes. Entries
    /// past the budget go to the disk spool, least recently used first.
    ///
    /// The budget applies at once: a smaller budget sends entries to the
    /// spool before this call returns. A larger budget moves nothing.
    ///
    /// When the host sets no budget, the cache takes one quarter of the
    /// recommended working set of the device that is not active at its first
    /// use.
    ///
    /// - Parameter memoryBudgetBytes: the most bytes the entries in memory may
    ///   hold. Zero or less keeps no entry in memory.
    public static func configurePromptCache(memoryBudgetBytes: Int) async {
        await ExecutorPromptCacheStore.current.configure(memoryBudgetBytes: memoryBudgetBytes)
    }

    /// Sets the disk budget of the spool, in bytes.
    ///
    /// The budget applies at once: a smaller budget deletes the least recently
    /// used files before this call returns. A larger budget deletes nothing.
    ///
    /// When the host sets no budget, the spool takes one quarter of the free
    /// space of the volume of its folder at its first use.
    ///
    /// - Parameter diskBudgetBytes: the most bytes the spill files may hold.
    ///   Zero or less keeps no file on disk.
    public static func configurePromptCache(diskBudgetBytes: Int) async {
        await ExecutorPromptCacheStore.current.configure(diskBudgetBytes: diskBudgetBytes)
    }

    /// Releases the cache of one session of this model: in memory, while it
    /// spills, and on disk. A no-op for an unknown session.
    ///
    /// The sessions of other models, and the other sessions of this model,
    /// stay. A spill of the session whose write has not ended deletes its
    /// file when the write ends.
    ///
    /// - Parameter sessionID: the identifier that the host binds with
    ///   `MLXLanguageModel.$promptCacheScope.withValue(.session(id))`. When the
    ///   host binds no scope, it is the identifier of the first transcript
    ///   entry of the session.
    public func releasePromptCache(sessionID: String) async {
        await ExecutorPromptCacheStore.current.remove(
            ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID))
    }

    /// The bytes the prompt cache holds now, for a host that sizes its pool.
    /// `spillingBytes` are still resident in memory until their write ends.
    ///
    /// - `memoryBytes`: the entries in memory.
    /// - `spillingBytes`: the entries whose write to disk has not ended.
    /// - `diskBytes`: the spill files on disk.
    public static var promptCacheUsage: (memoryBytes: Int, spillingBytes: Int, diskBytes: Int) {
        get async { await ExecutorPromptCacheStore.current.usage }
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
    /// decoding (provided by the MLXGuidedGeneration library), native
    /// `.allowed` tool routing, guided `.required` developer tool calls,
    /// and reasoning (chain-of-thought) routing.
    ///
    /// Capabilities are declared explicitly by the caller at ``init(configuration:capabilities:configurationResolver:speculativeDecoding:weightsLocation:load:)``
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

    /// The MTP drafter the text, reasoning and tool turns of this model speculate
    /// with, or `nil` for plain decode.
    public let speculativeDecoding: MLXSpeculativeDecoding?

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
    ///   - speculativeDecoding: The MTP drafter the text, reasoning and tool
    ///     turns speculate with, or `nil` for plain decode.
    ///   - weightsLocation: Resolves a model identifier to its on-disk weights
    ///     directory, for the availability checks.
    ///   - load: Loads the model container for a configuration.
    ///
    /// For example, to read weights from a fixed directory:
    ///
    /// ```swift
    /// weightsLocation: { id in
    ///     URL(fileURLWithPath: "/Volumes/SharedCache/models/\(id)")
    /// }
    /// ```
    public init(
        configuration: ModelConfiguration,
        capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration],
        configurationResolver: any ModelConfigurationResolver =
            DefaultConfigurationResolver(),
        speculativeDecoding: MLXSpeculativeDecoding? = nil,
        weightsLocation: @Sendable @escaping (String) -> URL,
        load: @escaping ContainerLoader
    ) {
        self.configuration = configuration
        self.capabilities = LanguageModelCapabilities(capabilities)
        self.configurationResolver = configurationResolver
        self.speculativeDecoding = speculativeDecoding
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
        _ = try await Self.makeGrammarTokenizer(
            modelID: modelID, tokenizer: tokenizer)

        // The MTP drafter loads here too, once, into the same model cache a
        // later `respond` reads.
        let speculation = try await loadMTPSpeculation(for: container)

        // Force Metal shader JIT with a minimal 1-token generate, run inside
        // `perform` so the forward pass + synchronize serialize against any
        // concurrent `respond`. `maxTokens: 1` makes the stream end on
        // its own; we consume it fully (no early break) so generation runs to
        // completion and leaves no dangling GPU work to race the teardown sync.
        try await container.perform(nonSendable: speculation) { context, speculation in
            // Exactly one synchronize on every exit path (success or throw),
            // per the Metal teardown invariant. `prepare` is CPU-only, so on a
            // pre-forward-pass throw this just synchronizes an idle stream.
            defer { Stream.gpu.synchronize() }
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user("warmup")]))
            let parameters = GenerateParameters(maxTokens: 1)
            for await _ in try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context
            ) {
                // Drain to completion.
            }
            if let speculation {
                try await Self.warmUp(speculation, input: input, context: context)
            }
        }
    }

    /// The token the prefill of an MTP stream samples before its first round.
    private static let prefillBonusTokenCount = 1

    /// Runs one short greedy MTP stream, so the first speculating `respond` pays
    /// no shader-JIT cost for the drafter and verify kernels.
    ///
    /// The stream runs the prefill bonus and one full round, and is drained to its
    /// end, as the plain warm-up generate is.
    ///
    /// - Parameters:
    ///   - speculation: the drafter of the model.
    ///   - input: the warm-up prompt.
    ///   - context: the loaded model context.
    /// - Throws: the error of the iterator.
    private static func warmUp(
        _ speculation: MTPSpeculation, input: LMInput, context: ModelContext
    ) async throws {
        let parameters = GenerateParameters(
            maxTokens: prefillBonusTokenCount + speculation.blockSize, temperature: 0)
        let (stream, task) = try generateTokensTask(
            input: input, parameters: parameters, context: context, speculation: speculation)
        for await _ in stream {
            // Drain to completion.
        }
        await task.value
    }

    // MARK: - Executor

    /// Executes inference requests for the model.
    public struct Executor: LanguageModelExecutor, Sendable {

        // MARK: - Test observation hook
        //
        // The macOS 27 FoundationModels SDK made the generation-channel event
        // and action types opaque: a consumer can no longer read back what was
        // streamed. Tests need to read it, and the only place the content is
        // available is here, right before it enters the channel. These emit
        // helpers are the sole send sites for each event kind; each notifies an
        // optional observer with a readable mirror. The observer is nil in
        // shipping builds (only tests attach one via the task-local), so the
        // arguments handed to `channel.send` are identical to before and
        // behavior is unchanged.

        /// Readable, internal-only mirror of the events this executor streams
        /// into the opaque FoundationModels channel.
        enum GenerationEvent: Sendable {
            enum Destination: Sendable { case response, reasoning }
            case appendText(String, entryID: String?, destination: Destination)
            case toolCall(id: String, name: String, arguments: String)
            case updateMetadata(
                [String: any ConvertibleToGeneratedContent & Sendable], entryID: String?)
            case updateUsage(
                input: LanguageModelExecutorGenerationChannel.Usage.Input,
                output: LanguageModelExecutorGenerationChannel.Usage.Output,
                entryID: String?)

            /// What the generation loop reported at the end of one pass. The
            /// framework has no event for the seconds a prefill took, thus a
            /// test that measures prompt-cache reuse reads this mirror alone;
            /// nothing enters the channel for it.
            case completion(GenerateCompletionInfo)
        }

        /// Attached only by tests (via `$generationObserver.withValue`); nil in
        /// shipping. Task-local so it reaches child tasks that also emit (e.g.
        /// the guided-generation text forwarder).
        @TaskLocal static var generationObserver: (@Sendable (GenerationEvent) -> Void)?

        /// Takes what one generation pass reported when it ended: into the MTP
        /// outcome of the call, when the model speculates, and to the test
        /// observer.
        ///
        /// - Parameter info: the completion report of the pass.
        static func observeCompletion(_ info: GenerateCompletionInfo) {
            SpeculativeDecodingOutcome.current?.record(info)
            generationObserver?(.completion(info))
        }

        /// Sends the MTP outcome of the call to its response entry, when the
        /// model speculates. A call of a model with no drafter sends nothing.
        ///
        /// - Parameters:
        ///   - entryID: the response entry of the call.
        ///   - channel: the channel to send the metadata into.
        static func emitSpeculativeDecodingOutcome(
            entryID: String, into channel: LanguageModelExecutorGenerationChannel
        ) async {
            guard let outcome = SpeculativeDecodingOutcome.current else { return }
            await emitMetadata(outcome.metadata, entryID: entryID, into: channel)
        }

        static func emit(
            text: String, entryID: String?, destination: GenerationEvent.Destination,
            into channel: LanguageModelExecutorGenerationChannel
        ) async {
            generationObserver?(.appendText(text, entryID: entryID, destination: destination))
            switch destination {
            case .response:
                await channel.send(
                    .response(entryID: entryID, action: .appendText(text, tokenCount: 1)))
            case .reasoning:
                await channel.send(
                    .reasoning(entryID: entryID, action: .appendText(text, tokenCount: 1)))
            }
        }

        static func emitMetadata(
            _ values: [String: any ConvertibleToGeneratedContent & Sendable], entryID: String?,
            into channel: LanguageModelExecutorGenerationChannel
        ) async {
            generationObserver?(.updateMetadata(values, entryID: entryID))
            // The channel takes the plain `ConvertibleToGeneratedContent`
            // existential. We add `Sendable` to it, because the observer event
            // crosses task boundaries, thus drop that part here.
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateMetadata(
                        values.mapValues { $0 as any ConvertibleToGeneratedContent })))
        }

        /// The input half of the usage of one generation pass.
        ///
        /// Every count a pass has of its own prompt -- the `promptTokenCount` of
        /// a completion info, or the size of the input the pass handed to the
        /// model -- counts the tokens that pass FED. The whole prompt is thus
        /// that count plus the prefix the cache already held, and the held part
        /// is the cached part. A pass that takes a checkpoint of a hybrid model
        /// feeds part of the prompt before its generation starts; the
        /// generation does not count that part, thus it is added back. A pass
        /// that splices after an MTP turn also feeds the last token of that
        /// turn, which no forward pass fed: it ends the previous answer and is
        /// no prompt token, thus it is taken out.
        ///
        /// - Parameters:
        ///   - fedTokenCount: the prompt tokens the generation of the pass fed
        ///     to the model.
        ///   - promptCache: the slot that carries the prefix the pass reused,
        ///     and the tokens the pass fed before its generation.
        /// - Returns: the whole prompt count, and the part of it the cache
        ///   served.
        static func usageInput(
            fedTokenCount: Int,
            promptCache: ExecutorPromptCacheSlot
        ) -> LanguageModelExecutorGenerationChannel.Usage.Input {
            .init(
                totalTokenCount: fedTokenCount + promptCache.prefilledTokenCount
                    + promptCache.reusedTokenCount - promptCache.uncommittedTokenCount,
                cachedTokenCount: promptCache.reusedTokenCount)
        }

        /// Every token a pass generated, the stop token included.
        ///
        /// The generation loop leaves the stop token out of
        /// `generationTokenCount` when it suppresses that token from the
        /// output, and names it in `discardedStopToken` instead, thus the
        /// output count a turn reports adds it back. Plain decode feeds the
        /// stop token to the caches before it returns it; MTP decode can
        /// return it as the last token of a round, which no forward pass fed.
        /// Both report the same count, because both generated the token. The
        /// prompt-cache ledger follows the caches alone
        /// (`ExecutorPromptCachePlan.commitOutcome`), thus the next turn
        /// reuses what the caches hold either way.
        ///
        /// - Parameter info: the completion report of the pass.
        /// - Returns: the generated token count, with the stop token.
        static func generatedTokenCount(of info: GenerateCompletionInfo) -> Int {
            info.generationTokenCount + (info.discardedStopToken == nil ? 0 : 1)
        }

        /// The cache-reuse rules of the model's response protocol.
        ///
        /// A protocol that keeps tokens in the cache which a chat-template
        /// render cannot write again -- DSML, Qwen 3.5 -- contributes a rule
        /// that splices past the turn the model committed. Every other model
        /// contributes none, thus the standard prefix rules decide alone.
        ///
        /// - Parameter context: the loaded model context.
        /// - Returns: the rules, in the order the policy consults them.
        static func promptCacheReuseRules(of context: ModelContext) -> [any PromptCacheReuseRule] {
            context.configuration.toolCallFormat?
                .promptCacheReuseRules(tokenizer: context.tokenizer) ?? []
        }

        /// The chat messages of `transcript` for a model whose reasoning
        /// protocol `resolved` names.
        ///
        /// A protocol that replays reasoning into history renders the
        /// transcript again with each `.reasoning` entry attached to the
        /// assistant entry of its turn. A protocol that replays only a turn
        /// that stopped inside its reasoning
        /// (`ReasoningConfig.replaysUnclosedReasoningIntoHistory`,
        /// DeepSeek-V4) renders it again with the reasoning of each turn whose
        /// response is empty, as `ChatSession` records such a turn, thus the
        /// render of the next turn extends the tokens the cache holds instead
        /// of omitting the turn. Every other protocol keeps `messages`, the
        /// render taken before the model loaded, which drops the reasoning.
        ///
        /// - Parameters:
        ///   - transcript: the transcript of the request.
        ///   - resolved: the configuration resolved for this call.
        ///   - messages: the messages rendered before the model loaded.
        /// - Returns: the messages to render the prompt from.
        /// - Throws: `LanguageModelError.unsupportedTranscriptContent` if an
        ///   attachment is not an image, or the transcript holds a data entry.
        static func historyMessages(
            of transcript: Transcript, resolved: ModelConfiguration,
            otherwise messages: [Chat.Message]
        ) throws -> [Chat.Message] {
            let reasoning = TranscriptConverter.HistoryReasoning(for: resolved.reasoningConfig)
            guard reasoning != .dropped else {
                return messages
            }
            let replayed = try TranscriptConverter.mlxMessages(
                for: transcript, reasoning: reasoning)
            return replayed.isEmpty ? messages : replayed
        }

        static func emitUsage(
            input: LanguageModelExecutorGenerationChannel.Usage.Input,
            output: LanguageModelExecutorGenerationChannel.Usage.Output,
            entryID: String?,
            into channel: LanguageModelExecutorGenerationChannel
        ) async {
            generationObserver?(.updateUsage(input: input, output: output, entryID: entryID))

            // This send was removed for a time, because the FM-27 beta
            // `.swiftinterface` declared the three-parameter
            // `updateUsage(input:output:metadata: = [:])` while the shipping
            // dylib exported only the two-parameter `updateUsage(input:output:)`.
            // A call that took the `metadata:` default bound a symbol that the
            // runtime did not hold, thus dyld aborted the image at load under
            // chained fixups. While the send was absent, the framework received
            // no usage event: a consumer read an input count of 0 on each
            // response, and only the `generationObserver` above saw the true
            // numbers.
            //
            // Measured again on macOS 27.0 build 26A5388g, on 2026-08-15, and
            // the drift is GONE. The interface, the `.tbd` and the runtime
            // dylib each hold the SAME three-parameter symbol, and the runtime
            // holds NO `updateUsage` export without `metadata`:
            //
            //   dyld_info -exports \
            //     /System/Library/Frameworks/FoundationModels.framework/\
            //     Versions/A/FoundationModels | grep updateUsage
            //   -> 3 symbols, each with `8metadata` in its mangled name
            //
            // Thus the send is correct again, and card ^z2996cp holds the
            // measurement. If a later SDK moves the symbol once more, this
            // call is the first thing that fails, and it fails at image load
            // rather than quietly.
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateUsage(input: input, output: output)))
        }

        static func emitToolCall(
            id: String, name: String, arguments: String, entryID: String,
            into channel: LanguageModelExecutorGenerationChannel
        ) async {
            generationObserver?(.toolCall(id: id, name: name, arguments: arguments))
            await channel.send(
                .toolCalls(
                    entryID: entryID,
                    action: .toolCall(
                        id: id, name: name,
                        action: .appendArguments(arguments, tokenCount: 1))))
        }

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
        static let defaultMaxTokens = 4096

        /// Multiplier on the structural completion reserve. It gives the
        /// guided loop room for more than one attempt to reach structural
        /// close before the budget runs out.
        static let completionReserveStructuralMultiplier = 3

        /// Divisor on the token budget. The completion reserve keeps at
        /// least this fraction (one quarter) of the budget.
        static let completionReserveBudgetDivisor = 4

        /// Multiplier on the structural reserve that sets the hard reserve,
        /// the point where the guided loop forces structural close.
        static let hardReserveStructuralMultiplier = 8

        /// Names the session a request belongs to, for the prompt cache that
        /// session carries between its turns.
        ///
        /// The host sets the session first, through
        /// ``MLXLanguageModel/promptCacheScope``: `.session(id)` names session
        /// `id`, and `.uncached` names no session, thus the pass checks out no
        /// cache and checks in no cache.
        ///
        /// When the host sets no scope, the identity comes from the transcript:
        /// the framework hands this executor no session identity, but a
        /// `LanguageModelSession` keeps the entries of its transcript and
        /// appends to them, thus the FIRST entry is the same entry on every turn
        /// of one session, and two sessions never hold one entry identifier.
        ///
        /// - Parameters:
        ///   - request: the request whose session is named.
        ///   - modelID: the model the cache belongs to.
        /// - Returns: the key, or nil when the scope is `.uncached` or, with no
        ///   scope, for an empty transcript. A request that names no session
        ///   carries no cache.
        static func sessionCacheKey(
            for request: LanguageModelExecutorGenerationRequest, modelID: String
        ) -> ExecutorPromptCacheKey? {
            switch MLXLanguageModel.promptCacheScope {
            case .some(.session(let sessionID)):
                return ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
            case .some(.uncached):
                return nil
            case nil:
                guard let firstEntry = request.transcript.first else { return nil }
                return ExecutorPromptCacheKey(modelID: modelID, sessionID: firstEntry.id)
            }
        }

        /// Checks the prompt cache of a session out of the store the task uses.
        ///
        /// A `.spilled` result gives the file of the cache to the caller: the
        /// store forgot the file, thus the caller must delete it.
        ///
        /// - Parameter key: the session of the request, or nil when the request
        ///   names no session.
        /// - Returns: what the store holds for the session, or `.none` when the
        ///   request names no session.
        static func checkOutPromptCache(
            _ key: ExecutorPromptCacheKey?
        ) async -> ExecutorPromptCacheCheckout {
            guard let key else { return .none }
            return await ExecutorPromptCacheStore.current.checkOut(key)
        }

        /// Map FoundationModels' optional `Double` `GenerationOptions.temperature`
        /// to MLXLMCommon's `Float` `GenerateParameters.temperature`, clamping
        /// negatives to 0.
        ///
        /// - Returns: `nil` when the caller did not request a specific
        ///   temperature, leaving `GenerateParameters`' built-in default in
        ///   place. Otherwise the clamped `Float`.
        ///
        /// Negative sampling temperatures land in `CategoricalSampler` and
        /// produce inverted distributions; we clamp at 0 so the worst the
        /// caller can get is greedy. `0` itself is honored unchanged because
        /// MLXLMCommon's `GenerateParameters.makeSampler()` routes
        /// `temperature == 0` to `ArgMaxSampler` (greedy) -- no division-by-
        /// zero hazard.
        static func clampedTemperature(_ value: Double?) -> Float? {
            guard let value else { return nil }
            return Float(max(0, value))
        }

        /// Translate Foundation Models' `GenerationOptions.SamplingMode` into one
        /// backend-local value that preserves both the sampling strategy and optional
        /// `UInt64` seed. No mode set (`nil`) and any future/unknown `Kind` both map to
        /// `nil`, selecting provider-default behavior without trapping or guessing.
        static func samplingConfiguration(
            from samplingMode: GenerationOptions.SamplingMode?
        ) -> MLXSamplingConfiguration? {
            guard let kind = samplingMode?.kind else { return nil }
            switch kind {
            case .greedy:
                return MLXSamplingConfiguration(mode: .greedy, seed: nil)
            case .randomTopK(let k, let seed):
                return MLXSamplingConfiguration(mode: .topK(k), seed: seed)
            case .randomProbabilityThreshold(let threshold, let seed):
                return MLXSamplingConfiguration(mode: .nucleus(threshold), seed: seed)
            @unknown default:
                return nil
            }
        }

        /// Build `GenerateParameters` for a sampler-backed generation pass. The shared
        /// resolver owns temperature and mode precedence; this helper preserves the
        /// optional seed directly on the backend request.
        static func makeParameters(
            maxTokens: Int,
            requestedTemperature: Double?,
            samplingConfiguration: MLXSamplingConfiguration?
        ) -> GenerateParameters {
            var parameters = GenerateParameters(maxTokens: maxTokens)
            resolveSamplingParameters(
                mode: samplingConfiguration?.mode,
                clampedTemperature: clampedTemperature(requestedTemperature)
            ).apply(to: &parameters)
            parameters.seed = samplingConfiguration?.seed
            return parameters
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
        static let logger = MLXLogger(label: "Prewarm")
        static let protocolLogger = MLXLogger(label: "TokenStreamProtocol")

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
        /// surfaced to or crashed on the caller. No handle is kept, because
        /// nothing could use it: the executor is a `Sendable` value with no
        /// teardown, the protocol offers no way to cancel a prewarm, and a
        /// warmup that outlives its caller only leaves the model cache warm.
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
                        "MLX prewarm failed for \(model.modelID): \(error.localizedDescription)"
                    )
                }
            }
        }

        /// The drain coordinator of the `respond` call the current task runs.
        ///
        /// Task-local, like ``generationObserver``, so every generation site
        /// inside one `respond` reaches its own coordinator without a
        /// parameter through every call. Nil outside `respond`, which makes
        /// the tracking helpers a no-op for any other caller of the
        /// generation functions.
        @TaskLocal static var drainCoordinator: GenerationDrainCoordinator?

        /// Generates a response for the given request, streaming events into the channel.
        ///
        /// A caller that cancels the generation must find the GPU quiet when
        /// its `CancellationError` returns — a process exit inside a window
        /// with generation work still in flight aborts on signal 6 or 11.
        /// The framework resumes a cancelled `LanguageModelSession.respond`
        /// WITHOUT awaiting this executor, so ordering inside this task
        /// alone cannot give that guarantee. A cancellation handler
        /// therefore holds the canceller — the `cancel()` call that cancels
        /// this task — until the coordinator drained every active
        /// generation, and the framework resumes the caller only after that
        /// handler returns. `CancelledGenerationDrainTests` holds the
        /// guarantee at the session level and at this executor level.
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
            let coordinator = GenerationDrainCoordinator()
            // Each pass of a call of a model that speculates reports its MTP
            // outcome here, and the response entry reports the whole call.
            let speculationOutcome = model.speculativeDecoding.map { _ in
                SpeculativeDecodingOutcome()
            }
            try await withTaskCancellationHandler {
                try await Self.$drainCoordinator.withValue(coordinator) {
                    try await SpeculativeDecodingOutcome.$current.withValue(speculationOutcome) {
                        try await runRespond(to: request, model: model, streamingInto: channel)
                    }
                }
            } onCancel: {
                coordinator.cancelActiveAndAwaitQuiescence()
            }
        }

        /// Runs one generation turn on the current task.
        ///
        /// Every exit path — the return, the `CancellationError` and every
        /// other throw — first stops the generation task, waits for it, and
        /// synchronizes the GPU stream, thus the caller of this method finds
        /// the GPU quiet. ``respond(to:model:streamingInto:)`` wraps this
        /// method in the cancellation gate that extends the same guarantee
        /// to a canceller the framework resumes early.
        ///
        /// - Parameters:
        ///   - request: The generation request containing transcript, tools, and options
        ///   - model: The model instance for this request
        ///   - channel: The channel to send response events into
        func runRespond(
            to request: LanguageModelExecutorGenerationRequest,
            model: MLXLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            var collected = try TranscriptConverter.mlxMessages(for: request.transcript)
            // MLX tokenizer crashes on empty chat input; provide a fallback.
            if collected.isEmpty {
                collected = [Chat.Message.user("")]
            }
            let messages = collected

            // The SDK's vision check does not cover `.attachment` segments, so the adapter
            // must enforce `.vision` for image attachments itself.
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

            let toolCallingMode = ToolCallingModeResolution.resolve(
                request.generationOptions.toolCallingMode)
            let enabledToolDefinitions = try ToolCallingModeResolution.enabledToolDefinitions(
                for: toolCallingMode,
                from: request.enabledToolDefinitions)

            let container = try await model.loadContainer()
            // The MTP drafter of the model, taken out of its own container here,
            // before the target's `perform` below, thus no lock nests.
            let speculation = try await model.loadMTPSpeculation(for: container)

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
            let requestedSamplingConfiguration = Self.samplingConfiguration(
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
            let labeledAttachments = TranscriptConverter.labeledAttachments(in: request.transcript)

            // The prompt cache this session carries. It is checked out here and
            // checked back in below, thus one response owns it for its whole
            // run and a second response on the same session starts cold rather
            // than writing the same caches.
            let promptCacheKey = Self.sessionCacheKey(for: request, modelID: modelID)
            let checkout = await Self.checkOutPromptCache(promptCacheKey)
            // A cache on disk is now the file of this response alone. The slot
            // reads it before the first plan, inside the container. The file is
            // deleted on every exit path: after the read, after a failed read,
            // and when the response throws or is cancelled before the read.
            defer {
                if let handle = checkout.spilledHandle {
                    ExecutorPromptCacheFile.removeFile(at: handle.url)
                }
            }
            let promptCache = ExecutorPromptCacheSlot(checkout, key: promptCacheKey)

            let outcome: Result<Void, any Error>
            do {
                // Send metadata first
                await Self.emitMetadata(
                    ["modelID": modelID, "requestID": request.id.uuidString],
                    entryID: entryID, into: channel)

                // Generate tokens inside actor isolation. `messages` carries
                // non-Sendable `Chat.Message` instances (UserInput.Image and
                // .Video are not Sendable), and the drafter model is not
                // Sendable either, so route both through
                // perform(nonSendable:_:) which boxes them across the actor hop.
                try await container.perform(nonSendable: (messages, speculation)) {
                    context, values in
                    let (renderedMessages, speculation) = values
                    try AttachmentLabelValidator.default.validate(
                        labeledAttachments, with: context.tokenizer)

                    let resolved = resolveConfiguration(
                        context: context, modelID: modelID,
                        configurationResolver: configurationResolver)

                    // A protocol that replays reasoning into history renders
                    // the transcript again with the reasoning attached, which
                    // the render taken before the model loaded could not do.
                    let messages = try Self.historyMessages(
                        of: request.transcript, resolved: resolved,
                        otherwise: renderedMessages)

                    // Render the prompt through the model's UserInputProcessor.
                    let userInput = UserInput(chat: messages)
                    let input = try await context.processor.prepare(input: userInput)

                    // Reasoning is only consumed by the unconstrained path
                    // (no tools, no schema). On the guided/tool paths the
                    // grammar already constrains output, so suppression-prep
                    // would be wasted work here. Continuation rounds run the
                    // tool path (below) like fresh turns: that path renders its
                    // own thinking state into the tool-aware prompt
                    // (`toolAwareRender`) -- thinking on with the think-then-call
                    // phase when reasoning is declared, forced off otherwise.
                    let mayRunReasoningPath =
                        enabledToolDefinitions.isEmpty
                        && request.schema == nil
                    // The `.alwaysOn` capability gate throws in here, before
                    // any path starts generation.
                    let reasoningPlan = try await makeReasoningPlan(
                        request: request,
                        messages: messages,
                        resolved: resolved,
                        declaresReasoning: declaresReasoning,
                        mayRunReasoningPath: mayRunReasoningPath,
                        baselineInput: input,
                        context: context)

                    // The prompt actually fed into generation: the suppressed
                    // prompt when we're forcing thinking off, otherwise the
                    // baseline `input` rendered above.
                    let effectiveInput = reasoningPlan.suppressedInput ?? input

                    // Tool path, entered on every round while tools are enabled
                    // -- fresh turns and continuations alike. Allowed mode uses
                    // native generation so the model can answer or call a tool;
                    // required mode constrains generation to a real tool call.
                    if !enabledToolDefinitions.isEmpty {
                        try await runToolGeneration(
                            request: request,
                            messages: messages,
                            enabledToolDefinitions: enabledToolDefinitions,
                            toolCallingMode: toolCallingMode,
                            schemaJSON: schemaJSON,
                            baselineInput: input,
                            resolved: resolved,
                            declaresReasoning: declaresReasoning,
                            modelID: modelID,
                            requestedMaxTokens: requestedMaxTokens,
                            samplingConfiguration: requestedSamplingConfiguration,
                            entryID: entryID,
                            toolCallsEntryID: toolCallsEntryID,
                            reasoningEntryID: reasoningEntryID,
                            promptCache: promptCache,
                            speculation: speculation,
                            context: context,
                            channel: channel)
                    } else if let schemaJSON {
                        try await runSchemaGeneration(
                            schemaJSON: schemaJSON,
                            input: input,
                            modelID: modelID,
                            requestedMaxTokens: requestedMaxTokens,
                            entryID: entryID,
                            promptCache: promptCache,
                            speculation: speculation,
                            context: context,
                            channel: channel)
                    } else {
                        try await runTextGeneration(
                            reasoningSetup: reasoningPlan.reasoningSetup,
                            fallbackInput: effectiveInput,
                            transcriptBoundary: reasoningPlan.transcriptBoundary,
                            requestedMaxTokens: requestedMaxTokens,
                            requestedTemperature: request.generationOptions.temperature,
                            samplingConfiguration: requestedSamplingConfiguration,
                            responseEntryID: entryID,
                            reasoningEntryID: reasoningEntryID,
                            promptCache: promptCache,
                            speculation: speculation,
                            context: context,
                            channel: channel
                        )
                    }

                    Stream.gpu.synchronize()
                }
                outcome = .success(())
            } catch is CancellationError {
                // Synchronize GPU before rethrowing to ensure in-flight operations complete.
                // Without this, process teardown can crash with Metal assertions.
                Stream.gpu.synchronize()
                outcome = .failure(CancellationError())
            } catch {
                // Synchronize GPU before rethrowing to ensure in-flight operations complete
                Stream.gpu.synchronize()
                // Re-map xgrammar errors to typed `LanguageModelError` cases
                // where the cause is provably user input (see `mapGrammarError`).
                // Internal-shim failures pass through unchanged.
                if let grammarError = error as? GrammarError {
                    outcome = .failure(Self.mapGrammarError(grammarError))
                } else {
                    outcome = .failure(error)
                }
            }

            // Return the session's prompt cache whatever the outcome, thus a
            // failed turn leaves nothing stranded outside the store. A turn that
            // could not represent itself checked in nothing, and the next turn
            // of that session starts cold.
            if let promptCacheKey {
                await ExecutorPromptCacheStore.current.checkIn(promptCacheKey, promptCache.entry)
            }
            // The `do` block opened the response entry with its first metadata, thus the entry
            // reports the MTP outcome of the call whatever the outcome: a call that was cancelled
            // or threw reports what its passes recorded before they stopped.
            await Self.emitSpeculativeDecodingOutcome(entryID: entryID, into: channel)
            try outcome.get()
        }

        /// Resolves the per-instance configuration for one generation.
        ///
        /// The result is held strictly as a local by the caller; it never
        /// lands in `context.configuration` or `Executor.Configuration`, so
        /// two instances with the same id but different resolvers don't
        /// cross-contaminate through the shared caches. Identity is read
        /// from `context.configuration` (at load time) and never from the
        /// resolved value.
        ///
        /// - Parameters:
        ///   - context: The loaded model context whose configuration is resolved
        ///   - modelID: The model identifier the descriptor carries
        ///   - configurationResolver: The per-instance resolver to apply
        func resolveConfiguration(
            context: ModelContext,
            modelID: String,
            configurationResolver: any ModelConfigurationResolver
        ) -> ModelConfiguration {
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
            return configurationResolver.resolve(
                context.configuration, for: descriptor)
        }

        /// The reasoning inputs one generation turn carries.
        struct ReasoningPlan {
            /// The prompt re-rendered with thinking forced off, or nil when
            /// no suppression applies to this turn.
            let suppressedInput: LMInput?
            /// The reasoning prompt, its config, and whether the template
            /// primed generation inside the think block; nil when this turn
            /// does not run the reasoning path.
            let reasoningSetup: (input: LMInput, config: ReasoningConfig, primedInside: Bool)?
            /// The index of the prompt the unconstrained path feeds where a
            /// hybrid model takes its checkpoint, or nil when the pass takes
            /// none.
            let transcriptBoundary: Int?
        }

        /// Gates the reasoning capability and prepares the reasoning inputs.
        ///
        /// Capability gate: when the caller omits `.reasoning` but the
        /// resolved configuration carries a reasoning config, the model must
        /// not be allowed to think:
        ///
        /// - Toggleable strategies (`.templateFlag`) re-render the prompt
        ///   with thinking off (the returned ``ReasoningPlan/suppressedInput``,
        ///   which only the unconstrained path consumes).
        /// - Non-suppressible strategies (`.alwaysOn`) raise
        ///   `unsupportedCapability` BEFORE generation, regardless of which
        ///   path (tools / schema / unconstrained) the request would
        ///   otherwise take. The throw is path-independent so a tool-calling
        ///   or schema-guided request on a model that always reasons
        ///   surfaces the same typed error the unconstrained path does,
        ///   never a silent leak through the grammar's malformed-output
        ///   fallback.
        ///
        /// - Parameters:
        ///   - request: The request whose context options select the reasoning level
        ///   - messages: The chat messages the prompts re-render from
        ///   - resolved: The resolver-patched configuration carrying the reasoning config
        ///   - declaresReasoning: Whether `.reasoning` was declared at init
        ///   - mayRunReasoningPath: Whether this turn takes the unconstrained path
        ///   - baselineInput: The prompt rendered with no thinking state, which
        ///     the unconstrained path feeds when no other prompt applies
        ///   - context: The loaded model context whose processor renders the prompts
        func makeReasoningPlan(
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            resolved: ModelConfiguration,
            declaresReasoning: Bool,
            mayRunReasoningPath: Bool,
            baselineInput: LMInput,
            context: ModelContext
        ) async throws -> ReasoningPlan {
            if !declaresReasoning, let suppressionConfig = resolved.reasoningConfig {
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

            // When .reasoning is OMITTED on the unconstrained path,
            // re-render the prompt with thinking off so the model
            // doesn't emit `<think>`. Toggleable-only;
            // .alwaysOn was already rejected above.
            let suppressed: (render: ThinkingRender, input: LMInput)?
            if mayRunReasoningPath, !declaresReasoning,
                let suppressionConfig = resolved.reasoningConfig
            {
                suppressed = try await Self.preparedInput(
                    messages: messages, config: suppressionConfig,
                    thinkingEnabled: false, context: context,
                    cannotDisableMessage:
                        "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."
                )
            } else {
                suppressed = nil
            }

            let reasoning: (render: ThinkingRender, input: LMInput)?
            let reasoningSetup: (input: LMInput, config: ReasoningConfig, primedInside: Bool)?
            if mayRunReasoningPath, declaresReasoning,
                let reasoningConfig = resolved.reasoningConfig
            {
                let thinkingEnabled = Self.thinkingEnabled(
                    for: request.contextOptions.reasoningLevel)
                let prepared = try await Self.preparedInput(
                    messages: messages, config: reasoningConfig,
                    thinkingEnabled: thinkingEnabled, context: context,
                    cannotDisableMessage:
                        "This model always reasons; reasoning cannot be disabled via reasoningLevel."
                )
                reasoning = prepared
                reasoningSetup = (
                    prepared.input, reasoningConfig,
                    Self.reasoningPrimedInside(
                        input: prepared.input, config: reasoningConfig,
                        tokenizer: context.tokenizer)
                )
            } else {
                reasoning = nil
                reasoningSetup = nil
            }

            // The text path feeds the reasoning input, else the suppressed
            // input, else the baseline input. Its checkpoint goes where a
            // next render of the same messages plus a new message parts from
            // the input it feeds.
            let fed = reasoning ?? suppressed ?? (ThinkingRender(), baselineInput)
            let transcriptBoundary =
                mayRunReasoningPath
                ? await fed.render.transcriptBoundary(
                    of: fed.input, messages: messages, tools: nil, context: context)
                : nil

            return ReasoningPlan(
                suppressedInput: suppressed?.input, reasoningSetup: reasoningSetup,
                transcriptBoundary: transcriptBoundary)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
