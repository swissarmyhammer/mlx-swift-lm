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
///
/// Deliberately `internal`, not `private`: it is referenced from other files
/// in this module (e.g. `TranscriptConverter`'s logger), so file-private
/// access would not compile.
let mlxFoundationModelsLoggingSubsystem = "com.apple.FoundationModels-MLX"

// MARK: - Constraint Cache Kind

/// Selects which xgrammar constructor a cached template was compiled
/// with. Used by the constraint cache so a JSON-schema source and a
/// structural-tag source can never alias even if their text collides.
private enum ConstraintKind {
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
///
/// Deliberately `internal`, not `private`: this type is `Executor`-only
/// implementation detail, not part of `MLXFoundationModels`' public API
/// surface, so `internal` already achieves the intended encapsulation for
/// real consumers of the library. It cannot be narrowed to `private`/
/// `fileprivate`, though -- `Tests/MLXFoundationModelsTests/ToolEnvelopeReserveZoneTests.swift`
/// constructs `ConstraintSetup` directly (via `@testable import
/// MLXFoundationModels`) to unit-test `reserves(forMaxTokens:)` in
/// isolation from `prepareConstraintSetup`'s full request plumbing --
/// `@testable import` only elevates `internal`/`public` access to a
/// separate module, never `private`/`fileprivate` (which stay scoped to
/// the declaring file no matter the importer), so a `private` modifier
/// here would make that test target fail to compile.
struct ConstraintSetup {
    let xgTokenizer: GrammarTokenizer
    let constraint: GrammarConstraint
    let maxTokens: Int
    let closingBias: MLXArray
    let structuralReserve: Int
    let whitespaceBias: MLXArray
    let whitespaceTokenIDs: Set<Int>

    /// Multiplier applied to `structuralReserve` to size the "free" normal
    /// zone -- the unbiased run of tokens before `GuidedGenerationLoop`'s
    /// soft-zone EOS boost (+200 at stop positions only, see
    /// `applyBiasAndSample`) engages. Reuses the 3x coefficient the old
    /// formula already had (the term its `maxTokens / 4` floor always
    /// overrode) as a numerically-familiar starting point -- NOT because
    /// other tests corroborate it under this new policy:
    /// `GenerableRoundTripTests`, `HardReserveStressTests`, and
    /// `GuidedGenerationTests` each hardcode that same pre-fix
    /// `max(structuralReserve * 3, maxTokens / 4)` formula verbatim in
    /// their own local helpers, calling `GuidedGenerationLoop.run` directly
    /// and bypassing this function entirely, so their passing validates
    /// only that unrelated direct-call path, not this capped-normal-zone
    /// policy.
    ///
    /// Empirically confirmed against gemma-3-270m-it-4bit's Int-schema
    /// repetition-degeneration failure (kanban t3nynaj): once the soft zone
    /// engages, the model's first biased sample flips to `EOS` almost
    /// immediately (observed at both the original bug's token 3072 and,
    /// with a first candidate of 10x, at ~token 20 -- still 20 repeated
    /// digits, enough to overflow `Int`). Since the flip happens on
    /// essentially the *first* biased token regardless of where the zone
    /// starts, the fix is to start it as early as the schema's actual
    /// structural need allows, not to widen the runway -- 3x keeps a small
    /// margin beyond the bare minimum for genuine per-field content while
    /// converging in single digits for trivial schemas like a bare `Int`.
    private static let normalZoneMultiplier = 3

    /// Derives completion/hard reserves for whatever `maxTokens` budget is
    /// actually in play for a given generation call. Reserves must always
    /// be computed against the budget they'll be applied to -- never
    /// cached against a stale (e.g. pre-Phase-1) `maxTokens` -- or the
    /// reserve can consume the entire (or more than the entire) remaining
    /// budget.
    ///
    /// The "normal" (unbiased) zone is sized off `structuralReserve` --
    /// the schema's actual minimal-content need -- rather than an
    /// unconditional fraction of `maxTokens`. A flat `maxTokens / 4` floor
    /// left trivial schemas (e.g. a bare `Int`, whose minimal JSON is a
    /// single digit) with thousands of unbiased tokens to loop in: at the
    /// Executor's `defaultMaxTokens = 4096`, that floor engaged the soft
    /// zone only in the last 1024 tokens, so a small model's own
    /// repetition-degeneration preference could run unchecked for the
    /// first 3072 -- long enough to produce a JSON-valid but
    /// astronomically large integer literal that overflows on decode (see
    /// kanban t3nynaj). Scaling the normal zone to
    /// `structuralReserve * normalZoneMultiplier` instead makes the soft
    /// zone engage within single digits of tokens for trivial schemas,
    /// while a modest cap at half of `maxTokens` keeps genuinely large/nested
    /// schemas from losing their soft-zone runway entirely.
    ///
    /// `hardReserve` (`structuralReserve * 8`) is capped at a QUARTER of
    /// `maxTokens` -- deliberately a tighter ceiling than `normalZoneLength`'s
    /// own half-of-`maxTokens` cap, not the same one -- for the same reason
    /// `normalZoneLength` is capped at all: an inflated or inaccurate
    /// `structuralReserve` -- e.g. `CompletionReserve.estimate` silently
    /// falling back to its flat default when a schema contains a construct
    /// `synthesizeMinimalJSON` doesn't recognize (kanban 7f091xq, the
    /// tool-calling envelope's `{"name": {"const": ...}}` alternatives before
    /// that gap was fixed) -- could otherwise push `hardReserve` past
    /// `maxTokens` itself. When that happens,
    /// `GuidedGenerationLoop.applyBiasAndSample`'s hard-zone check
    /// (`tokenCount >= maxTokens - hardReserve`) is true from token 0,
    /// forcing the *entire* budget through the hard-closing bias instead of
    /// just its trailing reserve -- crushing legitimate content for the
    /// whole generation. (Historically this was even worse: `0`-`9` were
    /// closing-tier then, so an open-ended string field could ramble in
    /// never-suppressed digits for the whole budget; kanban y4s0w2j removed
    /// digits from the tier and added the repetition-cycle breaker, but an
    /// all-hard-zone budget remains wrong on its own.)
    ///
    /// The quarter-vs-half asymmetry matters, not just "some cap": capping
    /// both `hardReserve` and `normalZoneLength` at the SAME `maxTokens / 2`
    /// ceiling let them collide exactly at that ceiling for large
    /// `structuralReserve` values (as this fix's own regression test caught
    /// via adversarial review), making `completionReserve ==
    /// maxTokens - normalZoneLength == hardReserve` and collapsing the soft
    /// zone -- the gentle "closing bias, no EOS penalty" phase strictly
    /// between `hardReserve` and `completionReserve` -- to zero width.
    /// Capping `hardReserve` a full `maxTokens / 4` below `normalZoneLength`'s
    /// ceiling guarantees `completionReserve - hardReserve >= maxTokens / 4`
    /// whenever both caps are the binding constraint (and a strictly wider
    /// gap otherwise, since `completionReserve`'s `Swift.max(_, hardReserve)`
    /// only ever widens the gap), so the soft zone always has real runway
    /// and zone order is never inverted.
    ///
    /// - Parameter maxTokens: The token budget currently in play for this call.
    /// - Returns: The completion reserve and hard reserve derived from `maxTokens`.
    func reserves(forMaxTokens maxTokens: Int) -> (completionReserve: Int, hardReserve: Int) {
        let hardReserve = Swift.min(structuralReserve * 8, maxTokens / 4)
        let normalZoneLength = Swift.min(
            structuralReserve * Self.normalZoneMultiplier, maxTokens / 2)
        let completionReserve = Swift.max(maxTokens - normalZoneLength, hardReserve)
        return (completionReserve, hardReserve)
    }
}

// MARK: - Model Cache Actor

/// Thread-safe model cache using Swift actor isolation.
///
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
    ///
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
    ///
    /// Concurrent callers for the same model will share the same loading task, preventing duplicate loads.
    ///
    /// The `loader` closure carries the transport types (downloader, tokenizer
    /// loader). Keeping them out of the cache means the cache itself stays
    /// agnostic of how a container is acquired -- first caller wins; later
    /// callers reuse the cached container regardless of which loader they
    /// brought along.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to load or return from cache.
    ///   - suppressDownloadingState: Whether an in-flight load of this
    ///     model should be excluded from the `.downloading` availability
    ///     signal (a warmup of an already-present model).
    ///   - loader: Loads the container when it isn't already cached or in flight.
    /// - Returns: The cached or newly loaded `ModelContainer`.
    /// - Throws: Whatever `loader` throws.
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
            return recordLoadSuccess(modelID: modelID, loadTask: loadTask, loaded: loaded)
        } catch {
            recordLoadFailure(modelID: modelID, loadTask: loadTask, error: error)
            throw error
        }
    }

    /// Finalizes a successful load: populates the container cache and clears
    /// the per-model in-flight/error bookkeeping. Guarded against
    /// supersession: `evict()`/`evictAll()` may have removed this load while
    /// it was suspended (actor reentrancy), in which case we hand the
    /// awaiter its container but do NOT re-populate the cache — ARC frees
    /// the weights when the awaiter releases it.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier the load was registered under.
    ///   - loadTask: The `LoadTask` this load was registered as, for the
    ///     supersession identity check.
    ///   - loaded: The container the load produced.
    /// - Returns: `loaded`, unconditionally.
    private func recordLoadSuccess(
        modelID: String, loadTask: LoadTask, loaded: ModelContainer
    ) -> ModelContainer {
        guard loadingTasks[modelID] === loadTask else { return loaded }
        containers[modelID] = loaded
        loadingTasks[modelID] = nil
        suppressedLoadIDs.remove(modelID)
        lastErrors[modelID] = nil
        return loaded
    }

    /// Records a failed load's error, unless a concurrent `evict()`/`evictAll()`
    /// superseded this load while it was suspended — a superseded load must
    /// not re-add a stale `lastErrors` entry for a model nobody holds.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier the load was registered under.
    ///   - loadTask: The `LoadTask` this load was registered as, for the
    ///     supersession identity check.
    ///   - error: The error the load threw.
    private func recordLoadFailure(modelID: String, loadTask: LoadTask, error: Error) {
        guard loadingTasks[modelID] === loadTask else { return }
        loadingTasks[modelID] = nil
        suppressedLoadIDs.remove(modelID)
        lastErrors[modelID] = error
    }

    /// Whether a *genuine download* is in flight for the given model: a load
    /// task is running and it was not tagged as a warmup of an already-present
    /// model.
    ///
    /// Drives `availability`'s `.downloading` state, so a background
    /// warmup of an already-downloaded model does not spuriously report
    /// `.downloading`. (A warmup that triggers a real fetch is not tagged and
    /// does report here.)
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: `true` when a genuine (non-suppressed) download is in flight.
    func isDownloading(modelID: String) -> Bool {
        loadingTasks[modelID] != nil && !suppressedLoadIDs.contains(modelID)
    }

    /// The most recent load error for the given model, if a previous attempt
    /// failed and no successful load has happened since.
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: The most recent load error, if any.
    func lastError(modelID: String) -> (any Error)? {
        lastErrors[modelID]
    }

    /// Gets or creates a cached value for `modelID`, storing anything newly
    /// created back into `cache` so subsequent calls hit the cache. Shared by
    /// `makeXgTokenizer` and `makeTokenizerBias`, which differ only in which
    /// cache they populate and how the value is created.
    ///
    /// - Parameters:
    ///   - modelID: The cache key to look up and populate.
    ///   - cache: The dictionary to check for a hit and store a miss into.
    ///   - create: Produces the value when `modelID` isn't already cached.
    /// - Returns: The cached or newly created value.
    /// - Throws: Whatever `create` throws.
    private func getOrCreateCached<T>(
        modelID: String,
        cache: inout [String: T],
        create: () throws -> T
    ) rethrows -> T {
        if let cached = cache[modelID] {
            return cached
        }
        let value = try create()
        cache[modelID] = value
        return value
    }

    /// Gets or creates a cached GrammarTokenizer for the given model.
    ///
    /// Registers the model's FULL stop-token set with xgrammar — the same
    /// set `GuidedGenerationLoop.buildStopTokenIDs` stops generation on —
    /// not just `tokenizer.eosTokenId`. An unregistered stop token (e.g.
    /// GLM's secondary `eos_token_id` entries `<|user|>`/`<|observation|>`)
    /// is an ordinary vocab entry the grammar can admit as string
    /// *content*, and constrained decode then samples it mid-envelope and
    /// truncates the output (kanban y4s0w2j).
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to cache the tokenizer under.
    ///   - tokenizer: The host tokenizer to derive the vocabulary from.
    ///   - configuration: The model configuration whose `eosTokenIds`/
    ///     `extraEOSTokens` contribute stop tokens.
    /// - Returns: The cached or newly created `GrammarTokenizer`.
    /// - Throws: Whatever `GrammarTokenizer`'s initializer throws.
    func makeXgTokenizer(
        modelID: String,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration
    ) throws -> GrammarTokenizer {
        try getOrCreateCached(modelID: modelID, cache: &xgTokenizers) {
            let vocab = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
            let stopTokenIDs = GuidedGenerationLoop.buildStopTokenIDs(
                tokenizer: tokenizer, configuration: configuration)
            // Preserve the historical `eosTokenId ?? 0` fallback for the
            // (degenerate) no-stop-tokens-at-all case.
            let stops =
                stopTokenIDs.isEmpty
                ? [Int32(tokenizer.eosTokenId ?? 0)]
                : stopTokenIDs.sorted().map(Int32.init)
            return try GrammarTokenizer(
                vocab: vocab.vocab,
                vocabType: vocab.vocabType,
                stopTokenIDs: stops
            )
        }
    }

    /// Whether an `GrammarTokenizer` is already cached for the given model.
    ///
    /// Used by `MLXLanguageModel.hasCachedXgTokenizer` so tests can assert
    /// that `warmUp()` pre-created it (a genuine cache hit) rather than only
    /// that a later guided respond happens to succeed.
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: `true` when a `GrammarTokenizer` is already cached for `modelID`.
    func hasCachedXgTokenizer(modelID: String) -> Bool {
        xgTokenizers[modelID] != nil
    }

    /// Gets or creates the cached tokenizer-derived logit biases for a model.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to cache the biases under.
    ///   - tokenizer: The host tokenizer to derive the biases from.
    /// - Returns: The cached or newly computed `TokenizerBias`.
    func makeTokenizerBias(
        modelID: String,
        tokenizer: any Tokenizer
    ) -> TokenizerBias {
        getOrCreateCached(modelID: modelID, cache: &tokenizerBiases) {
            let closing = ClosingTokenBias.compute(
                tokenizer: tokenizer,
                eosTokenID: tokenizer.eosTokenId
            )
            let (whitespace, whitespaceTokenIDs) = WhitespaceTokenBias.compute(
                tokenizer: tokenizer
            )
            return TokenizerBias(
                closing: closing,
                whitespace: whitespace,
                whitespaceTokenIDs: whitespaceTokenIDs
            )
        }
    }

    /// Gets a fresh constraint by cloning a cached template, or compiles and caches one first.
    ///
    /// Grammar compilation is expensive (~5-20ms). By caching the compiled template
    /// and cloning it (~0.1ms), repeated requests with the same schema skip recompilation.
    /// When Fork() is unavailable (xgrammar < v0.1.34), the clone attempt fails gracefully
    /// and each request compiles a fresh constraint instead.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier, part of the constraint cache key.
    ///   - kind: Which constructor `source` compiles under -- keeps a
    ///     JSON-schema source and a structural-tag source from ever
    ///     aliasing in the cache even if their text collides.
    ///   - source: The grammar/schema text to compile a constraint from.
    ///   - tokenizer: The `GrammarTokenizer` to compile the constraint with.
    ///   - hostTokenizer: The host tokenizer, forwarded to `GrammarConstraint`.
    ///   - fastForward: Whether to enable xgrammar's fast-forward optimization.
    /// - Returns: A fresh (cloned or newly compiled) `GrammarConstraint`.
    /// - Throws: Whatever `GrammarConstraint`'s initializer throws.
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
        // `GrammarConstraint` exposes `jsonSchema:`/`structuralTag:` as two
        // distinct labeled initializers (an external API shape defined in
        // XGrammarBridge.swift) rather than a single initializer taking a
        // common "source" parameter -- Swift has no way to genericize over
        // an argument label, so each case's call is written out in full
        // even though they differ only in that one label. This is API-
        // mandated duplication, not a missed reduction.
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
    ///
    /// Best-effort cancels an in-flight load (the load path is not
    /// cancellation-aware today, so this is a no-op safety net); the
    /// load-completion guard in `load()` is what prevents a superseded load
    /// from re-populating after removal.
    ///
    /// - Parameter modelID: The model identifier to evict.
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
    ///
    /// Without caching, model loading takes 2-30 seconds per request.
    private static let cache = ModelCache()

    /// Shared token-prefix KV-cache -- thread-safe via actor isolation.
    ///
    /// Lets `Executor.respond` reuse a prior round's KV state instead of
    /// re-prefilling the whole transcript every turn. See `PromptCache`.
    private static let promptCache = PromptCache()

    /// The configuration identifying and parameterizing the model to load.
    public let configuration: ModelConfiguration

    /// Resolves a model identifier to its on-disk weights directory.
    ///
    /// Used by the availability checks (`modelExistsOnDisk()`,
    /// `freeDiskSpaceBytes`), not by the load path. Injected so this module
    /// needs no HuggingFace path-resolution dependency.
    public let weightsLocation: @Sendable (String) -> URL

    /// Loads the model container for a configuration, forwarding download progress.
    ///
    /// Injected so this module carries no HuggingFace or swift-transformers
    /// dependency; the HuggingFace wiring lives in callers.
    public typealias ContainerLoader =
        @Sendable (
            _ configuration: ModelConfiguration,
            _ progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> ModelContainer

    private let load: ContainerLoader

    /// Stable identity for the model cache, executor configuration, tokenizer
    /// caches, availability, and progress reporting.
    ///
    /// Derived from the configuration so it is the single place identity is defined.
    public var modelID: String { configuration.name }

    /// Loads the model container for this model, returning a cached instance
    /// when one exists.
    ///
    /// Shares the process-global cache that `respond()`,
    /// `preload()`, and `session.prewarm()` use, so a caller working directly
    /// with the lower-level `ModelContainer` reuses the adapter's cache.
    ///
    /// - Returns: The cached model container, loading it first if necessary.
    /// - Throws: Whatever the underlying model loader throws while
    ///   downloading or initializing the container.
    public func loadContainer() async throws -> ModelContainer {
        try await loadContainer(suppressDownloadingState: false)
    }

    /// Internal variant that keeps an in-flight load of an already-present
    /// model out of the `.downloading` availability signal.
    ///
    /// - Parameter suppressDownloadingState: Whether an in-flight load of
    ///   this model should be excluded from the `.downloading` availability
    ///   signal (a warmup of an already-present model).
    /// - Returns: The cached model container, loading it first if necessary.
    /// - Throws: Whatever the underlying model loader throws while
    ///   downloading or initializing the container.
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
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to cache the tokenizer under.
    ///   - tokenizer: The host tokenizer to derive the vocabulary from.
    /// - Returns: The cached or newly created `GrammarTokenizer`.
    /// - Throws: Whatever `ModelCache.makeXgTokenizer` throws.
    static func makeXgTokenizer(
        modelID: String,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration
    ) async throws -> GrammarTokenizer {
        try await cache.makeXgTokenizer(
            modelID: modelID, tokenizer: tokenizer, configuration: configuration)
    }

    /// Gets the cached per-model tokenizer-derived logit biases (closing +
    /// whitespace), computing them on first use.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to cache the biases under.
    ///   - tokenizer: The host tokenizer to derive the biases from.
    /// - Returns: The cached or newly computed `TokenizerBias`.
    static func makeTokenizerBias(
        modelID: String,
        tokenizer: any Tokenizer
    ) async -> TokenizerBias {
        await cache.makeTokenizerBias(modelID: modelID, tokenizer: tokenizer)
    }

    /// Gets a constraint by cloning a cached compiled template (or compiling one first).
    ///
    /// - Parameters:
    ///   - modelID: The model identifier, part of the constraint cache key.
    ///   - kind: Which constructor `source` compiles under.
    ///   - source: The grammar/schema text to compile a constraint from.
    ///   - tokenizer: The `GrammarTokenizer` to compile the constraint with.
    ///   - hostTokenizer: The host tokenizer, forwarded to `GrammarConstraint`.
    ///   - fastForward: Whether to enable xgrammar's fast-forward optimization.
    /// - Returns: A fresh (cloned or newly compiled) `GrammarConstraint`.
    /// - Throws: Whatever `ModelCache.makeConstraint` throws.
    private static func makeConstraint(
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
    ///
    /// Internal test seam (not public API): lets `PrewarmGrammarTests` confirm
    /// `warmUp()` pre-created the tokenizer.
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: `true` when a `GrammarTokenizer` is already cached for `modelID`.
    static func hasCachedXgTokenizer(modelID: String) async -> Bool {
        await cache.hasCachedXgTokenizer(modelID: modelID)
    }

    /// Resolves this round's prompt-cache participation: see
    /// `PromptCache.resolve`.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to resolve the prompt cache for.
    ///   - newTokens: This round's full, unreduced prompt token sequence.
    ///   - model: The loaded model, boxed for the actor hop.
    ///   - parameters: Generation parameters threaded through to a rebuilt cache, if any.
    /// - Returns: The `[KVCache]` to generate with and the token suffix to actually feed.
    static func resolvePromptCache(
        modelID: String, newTokens: [Int], model: any MLXLMCommon.LanguageModel,
        parameters: GenerateParameters?
    ) async -> (cache: [KVCache], tokensToFeed: [Int]) {
        await promptCache.resolve(
            modelID: modelID, newTokens: newTokens, model: SendableBox(model),
            parameters: parameters
        ).consume()
    }

    /// Whether `model`'s cache-layer composition can participate in
    /// prompt-cache reuse -- via either of TWO mechanisms, either of which
    /// makes this `true`:
    ///
    /// - Ordinary chunk reuse (`PromptCache.isChunkable`'s doc comment):
    ///   `true` for a uniform `KVCacheSimple` stack. `false` for e.g. a
    ///   `maxKVSize`-driven `RotatingKVCache` stack, which is neither
    ///   chunkable nor a hybrid stack -- that combination genuinely never
    ///   supports reuse today.
    /// - Whole-stack hybrid checkpoint reuse (`PromptCache.isHybridMambaAttention`'s
    ///   doc comment): `true` for hybrid Mamba/attention architectures
    ///   (Qwen3.6/Qwen3-Next family: `Qwen35Model`/`Qwen35MoEModel`/
    ///   `Qwen3NextModel` in `Libraries/MLXLLM/Models/Qwen35.swift`/
    ///   `Qwen3Next.swift`), whose `[KVCache]` mixes `KVCacheSimple`
    ///   (full-attention layers) with `MambaCache` (Gated-DeltaNet
    ///   linear-attention layers) -- these DO now support cache reuse, just
    ///   via a checkpoint-and-restore mechanism rather than chunk slicing
    ///   (an earlier revision of this comment claimed hybrid architectures
    ///   could never participate at all; that claim was a scope-limiting
    ///   decision, not an architectural impossibility, and has been
    ///   superseded).
    ///
    /// A session against a model where this is `false` reports
    /// `LanguageModelSession.Usage.input.cachedTokenCount == 0` on every
    /// round, forever. This lets a caller distinguish that expected,
    /// by-design `0` from "cache reuse mis-fired this round" on a model
    /// that otherwise supports it, instead of only ever observing the
    /// same silent `0` either way.
    ///
    /// - Parameters:
    ///   - model: The loaded model to check.
    ///   - parameters: Generation parameters threaded into
    ///     `model.newCache(parameters:)`, matching whatever a real
    ///     generation call would build (e.g. `maxKVSize`-driven rotating
    ///     caches, which are never chunkable AND never hybrid -- see
    ///     `PromptCache.isChunkable`).
    /// - Returns: `true` when `model`'s cache can participate in either
    ///   chunk reuse or hybrid checkpoint reuse; `false` otherwise.
    public static func supportsPromptCacheReuse(
        model: any MLXLMCommon.LanguageModel, parameters: GenerateParameters? = nil
    ) -> Bool {
        let cache = model.newCache(parameters: parameters)
        return PromptCache.isChunkable(cache) || PromptCache.isHybridMambaAttention(cache)
    }

    /// Persists this round's `(tokens, cache)` for the next round's
    /// `resolvePromptCache` call: see `PromptCache.store`.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier to store the prompt cache under.
    ///   - tokens: The full prompt-plus-generated token sequence this round produced.
    ///   - cache: The `[KVCache]` state to persist for the next round.
    private static func storePromptCache(modelID: String, tokens: [Int], cache: [KVCache]) async {
        await promptCache.store(modelID: modelID, tokens: tokens, cache: SendableBox(cache))
    }

    /// Drops one model's remembered prompt cache without touching the
    /// container/tokenizer/constraint caches.
    ///
    /// Used when a round's actual
    /// generated content can't be reconciled with `cache`'s own `offset`
    /// (see `Executor.commitPromptCache`) -- the entry is untrustworthy,
    /// so the next round rebuilds instead of risking a stale reuse.
    ///
    /// - Parameter modelID: The model identifier to drop the prompt cache for.
    static func removePromptCache(modelID: String) async {
        await promptCache.remove(modelID: modelID)
    }

    /// Evicts every cached model, tokenizer, constraint template, and per-model
    /// tokenizer bias, freeing the GPU memory held by model weights.
    ///
    /// Subsequent
    /// requests reload from the on-disk cache.
    ///
    /// Safe to call during in-flight `respond()`/`warmUp()` work: each holds its
    /// own strong reference to the `ModelContainer` and synchronizes the GPU on
    /// exit, so dropping the cache's reference cannot free weights out from under
    /// a live kernel — the weights free via ARC once that work returns.
    public static func evictAll() async {
        await cache.evictAll()
        await promptCache.evictAll()
    }

    /// Reconfigures the shared prompt cache's chunk span for every model,
    /// clamping non-positive requests up to `1` (see `PromptCache.setChunkSize`).
    ///
    /// A chunk's hash-chain key is only valid for the span it was sliced
    /// under, so a genuine change to `size` evicts every model's stored
    /// chunks -- the next `resolvePromptCache` rebuilds from scratch.
    /// Setting the SAME (already-clamped) span again is a no-op: nothing is
    /// evicted.
    ///
    /// Trade-off: a SMALLER chunk size gives finer fork-point granularity --
    /// two conversations sharing a prefix can diverge, and still share
    /// cached state up to that point, at a shorter interval -- at the cost
    /// of more per-chunk bookkeeping and a longer hash-chain walk on every
    /// lookup. A LARGER chunk size shares more coarsely (a shared prefix
    /// must run the full span before it counts as reusable) but keeps
    /// bookkeeping and walk length down. Either way, the tail of a prompt
    /// past the last chunk boundary always re-prefills in full
    /// (`lookupLongestPrefix` only ever matches whole chunk-aligned
    /// windows), so the worst-case EXTRA prefill work a larger span
    /// introduces is bounded by `chunkSize - 1` tokens.
    ///
    /// - Parameter size: The requested chunk span in tokens; clamped to `>= 1`.
    public static func setPromptCacheChunkSize(_ size: Int) async {
        await promptCache.setChunkSize(size)
    }

    /// Reconfigures the shared prompt cache's total byte budget across
    /// every cached model (see `PromptCache.byteBudget`'s doc comment for
    /// why this budget is GLOBAL rather than per-model), clamping
    /// non-positive requests up to `1` (mirroring
    /// `setPromptCacheChunkSize(_:)`'s own `max(1, _)` clamp) -- an
    /// explicit caller value above `1` is trusted as-is, even a very small
    /// one; only the UNCONFIGURED default
    /// (`PromptCache.defaultByteBudget()`) is clamped up to the larger
    /// `PromptCache.minimumByteBudget` floor. Immediately evicts the
    /// globally least-recently-used chunk -- and,
    /// transitively, every chunk chained under it
    /// (`PromptCache.evictChunkAndDescendants(modelID:key:)`) -- repeatedly
    /// until the store is back at or under the new budget: evicting a
    /// chunk always orphans its chain descendants (a future
    /// `PromptCache.lookupLongestPrefix` walk can never reach past a
    /// missing parent), so this actor reclaims a whole orphaned lineage's
    /// bytes immediately rather than leaving dead weight in the store for
    /// a later sweep.
    ///
    /// PEAK-MEMORY MODEL: this budget bounds the STORE only. Every
    /// `resolve()` additionally materializes ONE assembled prefix copy per
    /// in-flight request (`PromptCache.assemble(chunks:layerCount:)`'s
    /// `ownedCopy` tensors), so peak unified memory roughly tracks:
    ///
    /// ```
    /// peak ≈ byteBudget + (in-flight requests × assembled-prefix size)
    /// ```
    ///
    /// ...and chunk residency competes with MLX's own GPU buffer cache
    /// (`MLX.Memory.cacheLimit`, which itself defaults to the FULL memory
    /// limit) and the model weights themselves for the SAME physical
    /// unified-memory pool. The default budget
    /// (`PromptCache.defaultByteBudget()`) derives from MLX's reported GPU
    /// working-set size (`MLX.GPU.maxRecommendedWorkingSetBytes()`,
    /// backed by Metal's `recommendedMaxWorkingSetSize`) but claims only a
    /// quarter of it (`PromptCache.unifiedMemoryBudgetFraction`) --
    /// falling back to a fixed 2 GiB (`PromptCache.fallbackDefaultByteBudget`)
    /// when MLX can't report a working-set size -- so there is explicit
    /// headroom left over for model weights, MLX's own GPU cache, and
    /// assembly copies. Callers running with many concurrent in-flight
    /// requests, very large assembled prefixes, or a large
    /// `Memory.cacheLimit` should pass a smaller explicit budget here.
    ///
    /// - Parameter bytes: The requested total byte budget across every
    ///   model's stored chunks; clamped up to `1`.
    public static func setPromptCacheByteBudget(_ bytes: Int) async {
        await promptCache.setByteBudget(bytes)
    }

    /// Drops this model from the shared cache, freeing the GPU memory held by its
    /// weights.
    ///
    /// A subsequent `respond()`/`preload()` triggers a fresh load
    /// (reusing the on-disk snapshot if the model was previously downloaded).
    ///
    /// Safe to call during an in-flight `respond()`: that call retains its own
    /// `ModelContainer` and finishes normally; the weights free via ARC once it
    /// returns. Evicting a model whose load is still in flight removes it cleanly
    /// — the in-flight load completes but does not re-populate the cache.
    public func evict() async {
        await Self.cache.remove(modelID: modelID)
        await Self.promptCache.remove(modelID: modelID)
    }

    /// Whether the shared cache has a *genuine download* in flight for the
    /// given model — excludes a warmup of an already-present model. Used by
    /// ``availability`` to surface a `.downloading` state.
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: `true` when a genuine (non-suppressed) download is in flight.
    static func isDownloadingInCache(modelID: String) async -> Bool {
        await cache.isDownloading(modelID: modelID)
    }

    /// The most recent load error for the given model, if any. Cleared on a
    /// subsequent successful load. Used by ``availability`` to surface a
    /// `.downloadFailed` state after a failed ``preload()``.
    ///
    /// - Parameter modelID: The model identifier to check.
    /// - Returns: The most recent load error, if any.
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

    /// The configuration resolver that patches a per-call ``ModelConfiguration`` for this instance.
    ///
    /// Defaults to ``DefaultConfigurationResolver`` when omitted.
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
    ///     `LLMRegistry.gemma31bQAT4bit` or `ModelConfiguration(id:)`).
    ///   - capabilities: The capabilities this model supports
    ///     (`.guidedGeneration`, `.toolCalling`, `.reasoning`, `.vision`).
    ///     Stored verbatim; the adapter never infers or expands the set.
    ///   - configurationResolver: Patches a per-call ``ModelConfiguration``
    ///     (reasoning config, extra stop tokens) for this instance.
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
        weightsLocation: @Sendable @escaping (String) -> URL,
        load: @escaping ContainerLoader
    ) {
        self.configuration = configuration
        self.capabilities = LanguageModelCapabilities(capabilities)
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
    ///
    /// - Throws: Whatever the underlying model loader throws while
    ///   downloading or initializing the container.
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
    ///
    /// - Throws: Whatever `loadContainer(suppressDownloadingState:)`,
    ///   `makeXgTokenizer`, or the warmup forward pass throws.
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
            modelID: modelID, tokenizer: tokenizer,
            configuration: container.configuration)

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
                input: UserInput(chat: [.user(content: "warmup")]))
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
        /// - Parameter value: The caller-requested temperature, if any.
        /// - Returns: `nil` when the caller did not request a specific
        ///   temperature, leaving `GenerateParameters`' built-in default in
        ///   place. Otherwise the clamped `Float`.
        static func clampedTemperature(_ value: Double?) -> Float? {
            guard let value else { return nil }
            return Float(max(0, value))
        }

        /// Translate FoundationModels' `GenerationOptions.SamplingMode` into the
        /// backend-local `MLXSamplingMode`, dropping the best-effort `seed`
        /// (MLX's samplers expose no seed-injection hook).
        ///
        /// No mode set (`nil`)
        /// and any future/unknown `Kind` both map to `nil` -- "use the provider
        /// default" -- so an unrecognized case never traps and never reaches the
        /// resolver. All value policy lives in `resolveSamplingParameters`; this
        /// shim is a pure 1:1 case translation.
        ///
        /// Kept as a standalone function (not inlined at its call site)
        /// since it's unit-tested directly in `SamplingModeShimTests.swift`.
        ///
        /// - Parameter samplingMode: The caller-requested sampling mode, if any.
        /// - Returns: The backend-local `MLXSamplingMode`, or `nil` to use the provider default.
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
        ///
        /// - Parameters:
        ///   - maxTokens: The already-resolved token budget for this call.
        ///   - requestedTemperature: The caller-requested temperature, if any.
        ///   - samplingMode: The caller-requested sampling mode, if any.
        /// - Returns: The resolved `GenerateParameters` for this generation pass.
        private static func makeParameters(
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
        ///
        /// - Parameter grammarError: The xgrammar error to map.
        /// - Returns: The mapped `LanguageModelError`, or `grammarError` unchanged.
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

        /// Calls `processor.prepare(input:)`, remapping a failure to
        /// `LanguageModelError.unsupportedTranscriptContent` when `messages`
        /// carries any image content.
        ///
        /// The concrete VLM's own image-specific errors (mismatched image
        /// count, unsupported resolution, decode/resize failure --
        /// `MLXVLM.VLMError` and its per-architecture siblings) live in
        /// MLXVLM, which MLXFoundationModels deliberately does not depend on
        /// (see Package.swift's "runtime trampoline discovery" note), so
        /// this maps by content shape -- an image is present somewhere in
        /// what's being prepared -- rather than by catching a module-private
        /// error type. A failure while `messages` carries no images passes
        /// through unchanged: remapping unconditionally would misrepresent
        /// an unrelated failure (a tokenizer/config issue, say) as a
        /// vision-content problem. Cancellation and already-typed
        /// `LanguageModelError`s also pass through unchanged, so this never
        /// re-wraps a case another layer already mapped correctly.
        ///
        /// `messages` is `LanguageModelExecutorGenerationRequest`'s full,
        /// re-rendered transcript, not just new content since the prior
        /// round (the `LanguageModelExecutor` protocol has no session
        /// identity -- every `respond()` call receives the complete history
        /// again, see `TranscriptConverter`'s doc comment). So this can
        /// still remap a failure that's unrelated to imagery if an *earlier*
        /// turn carried an image and a *later*, text-only turn's `prepare`
        /// call happens to fail for some other reason -- there is no
        /// narrower signal available without the concrete `VLMError` type.
        /// This is accepted as the best available precision given the
        /// architecture boundary above: an image genuinely present anywhere
        /// in the rendered prompt is exactly what the processor has to
        /// handle on every call (a multi-turn VLM conversation re-processes
        /// every referenced image each round), so "an image is somewhere in
        /// what's being prepared" is a real, non-fabricated correlate of
        /// "this failure could be image-related" -- narrowing to only the
        /// newest turn would *miss* genuine image failures instead (e.g. a
        /// `VLMError.singleImageAllowed`-shaped rejection triggered by an
        /// older image still in scope, even though the newest turn added no
        /// new one).
        ///
        /// - Parameters:
        ///   - processor: The model's `UserInputProcessor`.
        ///   - input: The `UserInput` to prepare.
        ///   - messages: The request's full rendered chat messages, checked
        ///     for image content.
        ///   - transcriptEntries: The request's full transcript, to name the
        ///     entries that carried the image content in the mapped error.
        /// - Throws: `LanguageModelError.unsupportedTranscriptContent` when
        ///   `processor` fails while `messages` carries image content; the
        ///   original error otherwise.
        private static func preparedInputMappingImageFailures(
            processor: any UserInputProcessor,
            input: UserInput,
            messages: [Chat.Message],
            transcriptEntries: some Collection<Transcript.Entry>
        ) async throws -> LMInput {
            do {
                return try await processor.prepare(input: input)
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as LanguageModelError {
                throw error
            } catch {
                guard messages.contains(where: { !$0.images.isEmpty }) else { throw error }
                throw LanguageModelError.unsupportedTranscriptContent(
                    LanguageModelError.UnsupportedTranscriptContent(
                        unsupportedContent: TranscriptConverter.entriesWithImages(
                            for: transcriptEntries),
                        debugDescription:
                            "This request includes image content the local vision pipeline could not process: \(error.localizedDescription)"
                    ))
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
        ///
        /// - Parameter configuration: The executor configuration.
        /// - Throws: Never currently -- `throws` is reserved for future
        ///   configuration validation and to match the initializer shape
        ///   `LanguageModelExecutor` conformance expects.
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
        /// the first `respond()` pays no cold-start shader-JIT cost, and
        /// proactively populates the shared prompt-cache chunk store with
        /// `transcript`'s own token prefix (see
        /// `populatePromptCacheChunks(model:transcript:)`).
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
        /// immediately. Both steps below are best-effort — a failure in
        /// either is logged, never surfaced to or crashed on the caller —
        /// and a prompt-cache-population failure never skips (or is skipped
        /// by) the shader warmup, since the two are independently useful
        /// even if one fails.
        ///
        /// - Parameters:
        ///   - model: The live model instance to warm.
        ///   - transcript: The transcript whose token prefix should be
        ///     proactively cached (see
        ///     `populatePromptCacheChunks(model:transcript:)`); the shader
        ///     warmup itself still uses a fixed dummy prompt and does not
        ///     depend on it.
        public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
            Task.detached {
                do {
                    try await model.warmUp()
                } catch {
                    Self.logger.error(
                        "MLX prewarm failed for \(model.modelID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                    )
                }
                do {
                    try await self.populatePromptCacheChunks(model: model, transcript: transcript)
                } catch {
                    Self.logger.error(
                        "MLX prompt-cache prewarm failed for \(model.modelID, privacy: .public): \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
        }

        /// Proactively populates the shared prompt-cache chunk store with
        /// `transcript`'s own token prefix, so a later `respond()` round that
        /// continues this transcript reuses it starting on its FIRST round
        /// (`PromptCacheSlot.cachedTokenCount > 0`) instead of paying full
        /// prefill lazily. This is what makes `prewarm(model:transcript:)`
        /// genuinely valuable for a `FoundationModelsRouter`-style fork
        /// scenario: prewarming the shared PARENT transcript before any fork
        /// session's first `respond()` means every fork's first turn hits
        /// the shared prefix, rather than the first fork alone paying to
        /// build it.
        ///
        /// Mirrors `respond()`'s own resolve→prefill→store cycle (see
        /// `makePromptCacheSlot`/`commitPromptCache`), reusing the exact same
        /// `resolvePromptCache`/`storePromptCache` entry points every real
        /// generation round goes through -- so a transcript whose prefix is
        /// already fully cached (e.g. prewarming the same transcript twice)
        /// does near-zero work: `resolvePromptCache` finds the existing
        /// chunk chain and only the capped remainder (at most `chunkSize -
        /// 1` tokens -- see `PromptCache.lookupLongestPrefix`'s doc)
        /// prefills, and storing the same `(tokens, cache)` pair again is a
        /// dedup no-op (`PromptCache.insert` only refreshes the existing
        /// chunks' `lastUsed`).
        ///
        /// Unlike a real generation round, the "prefill" here generates
        /// ZERO tokens (`GenerateParameters(maxTokens: 0)`): prewarm is
        /// speculative warmup, not answering the transcript, so sampling a
        /// real reply would waste GPU time producing output nobody reads
        /// (and a wrong/unwanted reply can't be un-emitted once it exists).
        /// With no generated tokens there is nothing to reconcile against
        /// `cache`'s `offset` advance the way `commitPromptCache`'s
        /// `generatedTokenIDs`-based reconciliation does (and that
        /// function's own `!generatedTokenIDs.isEmpty` guard would in fact
        /// make it silently drop a zero-token round entirely) -- the
        /// forward pass itself is the ground truth once it completes, so
        /// this stores directly via `MLXLanguageModel.storePromptCache`
        /// instead of routing through `commitPromptCache`.
        ///
        /// Multimodal transcripts are skipped without error, via the exact
        /// same `isTextOnly` gate `makePromptCacheSlot` uses: reducing a
        /// multimodal prompt to a raw token suffix would drop image/video/
        /// audio content the chunk store has nowhere to carry. A transcript
        /// that converts to no chat messages at all (e.g. empty, or only
        /// dropped `.reasoning` entries) is also a no-op -- there is no real
        /// prefix to cache, and forcing a dummy prompt here (as `warmUp()`
        /// does for shader JIT) would only ever match a dummy continuation,
        /// never a real one.
        ///
        /// Not `private`: also a direct test seam, mirroring
        /// `hasCachedXgTokenizer` -- callable and awaitable directly (unlike
        /// the fire-and-forget `prewarm(model:transcript:)`, which only
        /// reaches this indirectly from inside a detached `Task`), so a test
        /// can observe the chunk store deterministically once this method
        /// returns.
        ///
        /// - Parameters:
        ///   - model: The live model instance to warm the transcript's
        ///     prefix cache for.
        ///   - transcript: The transcript whose token prefix should be
        ///     proactively cached.
        /// - Throws: Whatever container loading, message preparation, or the
        ///   prefill forward pass throws.
        func populatePromptCacheChunks(
            model: MLXLanguageModel, transcript: Transcript
        ) async throws {
            // Whether the transcript renders to any messages is
            // format-invariant, so gate emptiness on the cheap
            // format-agnostic render *before* paying for a container load --
            // preserving the original early-return-before-load behavior for an
            // empty transcript.
            guard !TranscriptConverter.mlxMessages(for: transcript).isEmpty else { return }

            let container = try await model.loadContainer()
            // Re-render format-aware (see `respond`): a `.mistral` model's
            // tool-call turns must replay as structured `tool_calls` so its
            // strict-alternation chat template accepts a warmed transcript
            // that already contains a tool round. History-preserving
            // families likewise replay reasoning and render with their
            // history context, so the warmed tokens match what `respond`'s
            // own renders will re-produce.
            let loadedConfiguration = await container.configuration
            let historyContext = loadedConfiguration.reasoningConfig?.historyPreservationContext
            let messages = TranscriptConverter.mlxMessages(
                for: transcript, toolCallFormat: loadedConfiguration.toolCallFormat,
                replayReasoning: loadedConfiguration.reasoningConfig?.historyPreservationKey
                    != nil)

            let modelID = self.modelID
            try await container.perform(nonSendable: messages) { context, messages in
                let input = try await context.processor.prepare(
                    input: UserInput(chat: messages, additionalContext: historyContext))

                // Bail out before ever materializing `tokens` below: unlike
                // `makePromptCacheSlot` (whose callers need a real, uncached
                // response either way, multimodal or not), this method has
                // nothing left to do for a multimodal transcript, so there's
                // no reason to pay for `.asArray(Int.self)` -- a real
                // host-side pull of the token buffer -- just to discard it.
                // `resolvePromptCacheIfTextOnly` below re-checks this same
                // gate, but that re-check is just a cheap boolean read of
                // `input`, not the array materialization this guard exists
                // to skip.
                guard Self.isTextOnly(input) else { return }

                let tokens = input.text.tokens.asArray(Int.self)
                guard !tokens.isEmpty else { return }

                // Shares its resolve→gate logic with `makePromptCacheSlot`
                // via `resolvePromptCacheIfTextOnly` -- see that method's
                // doc for why the two don't also share `PromptCacheSlot`
                // itself: a multimodal `nil` there means "generate for real
                // anyway, just uncached" (`makePromptCacheSlot`'s callers
                // always need a response), whereas here it means "nothing
                // left to do," so this skips outright instead of running a
                // wasted prefill over unused multimodal input.
                guard
                    let resolved = await resolvePromptCacheIfTextOnly(
                        input: input, tokens: tokens, model: context.model, parameters: nil)
                else { return }

                // Drain a zero-token generation to run the prefill forward
                // pass -- advancing `resolved.cache`'s offset to exactly
                // `tokens.count` -- without sampling (or committing to) any
                // reply text; see this method's doc comment for why a real
                // answer here would be wasted, un-readable, un-undoable work.
                for await _ in try generate(
                    input: LMInput(tokens: MLXArray(resolved.tokensToFeed)),
                    cache: resolved.cache,
                    parameters: GenerateParameters(maxTokens: 0),
                    context: context
                ) {
                    // No output tokens are ever produced (maxTokens: 0); the
                    // loop body never runs. Draining the stream to
                    // completion is what waits for the prefill's GPU work
                    // and its internal `Stream().synchronize()` to finish.
                }

                await MLXLanguageModel.storePromptCache(
                    modelID: modelID, tokens: tokens, cache: resolved.cache)
            }
        }

        /// Generates a response for the given request, streaming events into the channel.
        ///
        /// - Parameters:
        ///   - request: The generation request containing transcript, tools, and options
        ///   - model: The model instance for this request
        ///   - channel: The channel to send response events into
        /// - Throws: Whatever the underlying container load, grammar/tokenizer
        ///   setup, or generation path throws.
        public func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: MLXLanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Vision capability gate (adapter-side). Labeled image
            // attachments arrive as public `.attachment` segments that
            // the SDK's own vision guard never inspects, so the adapter
            // is the only place that can enforce `.vision` for this path.
            // Throw the same typed error the SDK would, before loading
            // any weights, so a model declared without `.vision` fails
            // fast and identically across the tool / schema / plain paths.
            // `entriesWithImages` inspects the transcript directly, so this
            // gate stays before the weight load without depending on the
            // rendered `messages` below (which are built only after the
            // container loads, to read the model's tool-call format).
            if !model.capabilities.contains(.vision),
                !TranscriptConverter.entriesWithImages(for: request.transcript).isEmpty
            {
                throw LanguageModelError.unsupportedCapability(
                    LanguageModelError.UnsupportedCapability(
                        capability: .vision,
                        debugDescription:
                            "This request includes an image, but .vision was not declared at MLXLanguageModel init. Declare .vision to accept image inputs."
                    ))
            }

            let container = try await model.loadContainer()

            // Render the transcript into chat messages *after* the container
            // loads so the model's resolved tool-call format is known: a
            // `.mistral` model needs its tool-call turns replayed as
            // structured `tool_calls` (not verbatim assistant text) to satisfy
            // its strict-alternation chat template. The format is read from
            // the loaded configuration -- the documented source of truth for
            // it (see `ModelConfigurationResolver`); non-Mistral families
            // render exactly as before.
            let loadedConfiguration = await container.configuration
            let toolCallFormat = loadedConfiguration.toolCallFormat
            var collected = TranscriptConverter.mlxMessages(
                for: request.transcript, toolCallFormat: toolCallFormat,
                // History-preserving families (Qwen3.6) replay past turns'
                // reasoning as `reasoning_content` so history re-renders
                // stay byte-continuous with what each round actually fed --
                // see `ReasoningConfig.historyPreservationKey`. Read from the
                // loaded configuration like `toolCallFormat` above (message
                // construction precedes per-call resolution); a resolver
                // patch of `reasoningConfig` still governs the render-time
                // `preserve_thinking` context, not this replay gate.
                replayReasoning: loadedConfiguration.reasoningConfig?.historyPreservationKey
                    != nil)
            // MLX tokenizer crashes on empty chat input; provide a fallback.
            if collected.isEmpty {
                collected = [Chat.Message.user(content: "")]
            }
            let messages = collected

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
                    let setup = try await prepareRespondSetup(
                        request: request, messages: messages, modelID: modelID,
                        context: context, declaresReasoning: declaresReasoning,
                        configurationResolver: configurationResolver, schemaJSON: schemaJSON)

                    let completedNormally = try await dispatchGeneration(
                        setup: setup, request: request, messages: messages,
                        modelID: modelID, schemaJSON: schemaJSON,
                        requestedMaxTokens: requestedMaxTokens,
                        requestedSamplingMode: requestedSamplingMode,
                        declaresReasoning: declaresReasoning, entryID: entryID,
                        toolCallsEntryID: toolCallsEntryID,
                        reasoningEntryID: reasoningEntryID, context: context,
                        channel: channel)
                    guard completedNormally else { return }

                    Stream.gpu.synchronize()
                }
            } catch {
                // Synchronize GPU before rethrowing/remapping to ensure
                // in-flight operations complete on every non-cutoff error
                // exit -- without this, process teardown can crash with
                // Metal assertions. Shared by every error path below so the
                // sync happens exactly once regardless of which one fires.
                Stream.gpu.synchronize()
                if error is CancellationError {
                    throw CancellationError()
                }
                // Re-map xgrammar errors to typed `LanguageModelError` cases
                // where the cause is provably user input (see `mapGrammarError`).
                // Internal-shim failures pass through unchanged.
                if let grammarError = error as? GrammarError {
                    throw Self.mapGrammarError(grammarError)
                }
                throw error
            }
        }

        /// Bundles what `respond()`'s dispatch step needs, once the prompt
        /// has been rendered and the per-instance configuration resolved.
        private struct RespondSetup {
            /// The prompt rendered by `UserInputProcessor.prepare` through
            /// the model's *default* (non-tool-aware) chat template, before
            /// any reasoning-suppression rewrite. `nil` when tools are
            /// enabled: the tool-calling branch re-tokenizes independently
            /// via its own `context.processor.prepare(input:)` call (with
            /// `tools:` populated, taking the tool-aware template branch --
            /// see `runToolCalling`) and never reads this field, so
            /// rendering it eagerly is both wasted work and unsafe -- a
            /// continuation round's `messages` can contain a replayed
            /// `.tool`-role message, and the default (no-`tools`) template
            /// of a toolCalling-capable model isn't guaranteed to handle
            /// `role == "tool"`.
            let input: LMInput?
            /// The per-instance configuration resolved from `config.json`.
            let resolved: ModelConfiguration
            /// The prompt actually fed into generation: the suppressed
            /// prompt when forcing thinking off, otherwise the baseline
            /// `input`. `nil` under the same condition as `input`.
            let effectiveInput: LMInput?
            /// Reasoning-path setup, present only when `.reasoning` was
            /// declared, a reasoning config resolved, and no tools/schema
            /// are in play.
            let reasoningSetup: (input: LMInput, reasoningConfig: ReasoningConfig, primedInside: Bool)?
            /// The prompt actually fed into guided generation: `messages`
            /// routed through ``Executor/guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:)``
            /// when the request carries a schema, so `ContextOptions.includeSchemaInPrompt`
            /// is consulted at the seam where the guided-generation prompt is
            /// actually assembled -- not just `input`, which never consults
            /// the schema at all. Equal to `input` today whenever the
            /// resolved family carries no history-preservation context (the
            /// schema seam is currently a no-op -- see its doc comment);
            /// re-rendered separately only when the schema seam or the
            /// history context would actually change the render, so a
            /// future schema-in-prompt renderer added there does not need to
            /// touch this call site to take effect. `nil` under the same
            /// condition as `input` (tools enabled), and also when the
            /// request carries no schema at all. Paired with the
            /// stable-boundary baseline built from the SAME
            /// history-affecting context as the render itself (see
            /// `makeGuidedRender`), so `runGuidedGeneration`'s hybrid
            /// prompt-cache boundary always agrees with the prompt it fed.
            let guidedInput: (input: LMInput, stableBoundary: StableBoundaryRender)?
            /// The model's context window length, read from `config.json`'s
            /// `max_position_embeddings` field. `nil` when the model's
            /// configuration doesn't expose a recognized context-length
            /// field, in which case `dispatchGeneration`/`runToolCalling`
            /// skip context-size validation rather than guessing a default.
            let contextLength: Int?
        }

        /// Unwraps a `RespondSetup` field that's `nil` only when tools are
        /// enabled, on a dispatch path reachable only when they are not.
        /// Shared by every such unwrap in `dispatchGeneration` so the
        /// "this can't actually be nil here" reasoning is stated once.
        ///
        /// - Parameters:
        ///   - value: The optional field to unwrap.
        ///   - fieldName: The `RespondSetup` field's name, for the crash message.
        ///   - contextPath: The dispatch path performing the unwrap, for the crash message.
        /// - Returns: `value`, unwrapped.
        private static func unwrapSetupField<T>(
            _ value: T?, fieldName: String, contextPath: String
        ) -> T {
            guard let value else {
                preconditionFailure(
                    "RespondSetup.\(fieldName) must be present on the \(contextPath) path; prepareRespondSetup only skips rendering it when tools are enabled, and that branch already returned above."
                )
            }
            return value
        }

        /// Throws `LanguageModelError.contextSizeExceeded` when a prompt's
        /// real token count exceeds the model's context window, so a
        /// transcript that has grown too large for the model fails with an
        /// accurate typed error before any generation work (weight compute,
        /// grammar setup) is attempted -- rather than crashing, silently
        /// truncating, or surfacing an unrelated low-level error deeper in
        /// the generation path.
        ///
        /// `contextLength` is `nil` when `config.json` doesn't expose a
        /// recognized context-length field -- naming varies by architecture
        /// (see `BaseConfiguration.contextLength`) and some omit it
        /// entirely. There is nothing trustworthy to compare against in
        /// that case, so the check is skipped rather than guessing a
        /// default that could reject a request the model can actually
        /// serve.
        ///
        /// - Parameters:
        ///   - tokenCount: The prompt's actual token count, as tokenized for this request.
        ///   - contextLength: The model's context window length, if known.
        /// - Throws: `LanguageModelError.contextSizeExceeded` when `tokenCount` exceeds `contextLength`.
        private static func validateContextSize(tokenCount: Int, contextLength: Int?) throws {
            guard let contextLength, tokenCount > contextLength else { return }
            throw LanguageModelError.contextSizeExceeded(
                LanguageModelError.ContextSizeExceeded(
                    contextSize: contextLength,
                    tokenCount: tokenCount,
                    debugDescription:
                        "The transcript's prompt (\(tokenCount) tokens) exceeds this model's context window (\(contextLength) tokens)."
                ))
        }

        /// Bundles the per-instance configuration `prepareRespondSetup` needs
        /// once `config.json` has been read and the resolver has run: the
        /// resolved ``ModelConfiguration`` (reasoning config, extra stop
        /// tokens) and the model's context-window length, if known.
        ///
        /// Split out of `prepareRespondSetup` so the "read config.json,
        /// decode it, resolve it, validate the reasoning capability" chain
        /// reads as one self-contained step, separate from the prompt-
        /// rendering gates in ``PromptVariants`` that consume its result.
        private struct RespondConfigResolution {
            /// The per-instance configuration resolved from `config.json`.
            let resolved: ModelConfiguration
            /// The model's context window length, read from `config.json`'s
            /// `max_position_embeddings` field. `nil` when the model's
            /// configuration doesn't expose a recognized context-length
            /// field, in which case `dispatchGeneration`/`runToolCalling`
            /// skip context-size validation rather than guessing a default.
            let contextLength: Int?
        }

        /// Reads `config.json`, resolves the per-instance ``ModelConfiguration``,
        /// and validates the reasoning capability gate against it -- everything
        /// `prepareRespondSetup` needs before it can compute the reasoning-
        /// dependent prompt-rendering gates in ``PromptVariants``.
        ///
        /// - Parameters:
        ///   - modelID: The model identifier used to build the `ModelDescriptor`.
        ///   - context: The loaded model context (tokenizer, configuration).
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - configurationResolver: Patches the per-call `ModelConfiguration`
        ///     read from `config.json`.
        /// - Returns: The resolved configuration and context length.
        /// - Throws: `LanguageModelError.unsupportedCapability` when the
        ///   resolved configuration always reasons but `.reasoning` was not
        ///   declared (see `validateReasoningCapability`).
        private func resolveRespondConfiguration(
            modelID: String,
            context: ModelContext,
            declaresReasoning: Bool,
            configurationResolver: any ModelConfigurationResolver
        ) throws -> RespondConfigResolution {
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
            let baseConfiguration = configData.flatMap {
                try? JSONDecoder.json5().decode(BaseConfiguration.self, from: $0)
            }
            let modelType = baseConfiguration?.modelType ?? ""
            // Only recognized when `config.json` exposes
            // `max_position_embeddings` (the dominant convention across the
            // architectures this repository ports); `nil` otherwise, which
            // `dispatchGeneration`/`runToolCalling` treat as "unknown --
            // skip validation" rather than guessing a default.
            let contextLength = baseConfiguration?.contextLength
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
            //   prompt with thinking off (handled by `preparePromptVariants`).
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

            return RespondConfigResolution(resolved: resolved, contextLength: contextLength)
        }

        /// Bundles the reasoning-suppression, reasoning-priming, and guided-
        /// input prompt-rendering gates `prepareRespondSetup` must track
        /// together once the per-instance configuration is resolved -- see
        /// `preparePromptVariants` for how each field is computed.
        /// Deliberately does not include the eager (tooling-gated) render:
        /// that gate runs before configuration resolution (see
        /// `prepareRespondSetup`) and doesn't depend on `resolved`, so it
        /// stays a plain local there.
        private struct PromptVariants {
            /// The reasoning-suppressed render, when thinking must be forced
            /// off; `nil` when suppression doesn't apply.
            let suppressed: LMInput?
            /// The reasoning-primed render plus its config and primed-state,
            /// when declared reasoning applies; `nil` otherwise.
            let reasoningSetup: (input: LMInput, reasoningConfig: ReasoningConfig, primedInside: Bool)?
            /// The guided-generation render honoring `includeSchemaInPrompt`,
            /// paired with the stable-boundary baseline built from the SAME
            /// history-affecting context (see `makeGuidedRender`); `nil`
            /// when tools are enabled or the request carries no schema.
            let guided: (input: LMInput, stableBoundary: StableBoundaryRender)?
        }

        /// Whether the unconstrained path should render a reasoning-
        /// SUPPRESSED prompt this round: reasoning is declared OFF, the
        /// round is on the unconstrained path (no tools/schema --
        /// reasoning is only consumed there), and the resolved
        /// configuration actually has a reasoning config to suppress
        /// against.
        ///
        /// - Parameters:
        ///   - mayRunReasoningPath: Whether this round is on the unconstrained path.
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        /// - Returns: `true` when a suppressed prompt should be rendered.
        private static func shouldSuppressReasoning(
            mayRunReasoningPath: Bool, declaresReasoning: Bool, resolved: ModelConfiguration
        ) -> Bool {
            mayRunReasoningPath && !declaresReasoning && resolved.reasoningConfig != nil
        }

        /// Whether the unconstrained path should render a reasoning-PRIMED
        /// prompt this round and prepare `reasoningSetup`: reasoning is
        /// declared ON, the round is on the unconstrained path, and the
        /// resolved configuration actually has a reasoning config to prime.
        ///
        /// - Parameters:
        ///   - mayRunReasoningPath: Whether this round is on the unconstrained path.
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        /// - Returns: `true` when `reasoningSetup` should be prepared.
        private static func shouldSetupReasoning(
            mayRunReasoningPath: Bool, declaresReasoning: Bool, resolved: ModelConfiguration
        ) -> Bool {
            mayRunReasoningPath && declaresReasoning && resolved.reasoningConfig != nil
        }

        /// Whether a guided-generation-specific prompt render is needed
        /// this round: tools are disabled (so the eager render is
        /// meaningful), a developer-supplied schema is present, and the
        /// eager render itself succeeded.
        ///
        /// - Parameters:
        ///   - needsEagerInput: Whether tools are disabled.
        ///   - schemaJSON: The developer-supplied JSON schema, if any.
        ///   - input: The eager render, if any.
        /// - Returns: `true` when `guidedInput` should be built.
        private static func shouldBuildGuidedInput(
            needsEagerInput: Bool, schemaJSON: String?, input: LMInput?
        ) -> Bool {
            needsEagerInput && schemaJSON != nil && input != nil
        }

        /// Computes the reasoning-suppression, reasoning-priming, and guided-
        /// input prompt-rendering gates, once the per-instance configuration
        /// has been resolved and the eager render is in hand. Each gate is
        /// independently conditioned (tool state, schema presence, declared
        /// vs. resolved reasoning), but all three read from the same
        /// `resolved`/`input`/`messages`, so they are computed together here
        /// rather than re-derived inline across `prepareRespondSetup`.
        ///
        /// - Parameters:
        ///   - request: The generation request; supplies tool/schema gating and reasoning level.
        ///   - messages: The rendered chat messages for this round.
        ///   - context: The loaded model context (tokenizer, processor).
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The per-instance configuration already resolved from `config.json`.
        ///   - needsEagerInput: Whether tools are disabled, so the eager
        ///     render (and, in turn, the guided-input render) is meaningful.
        ///   - input: The eager render `prepareRespondSetup` already computed,
        ///     or `nil` when tools are enabled.
        ///   - schemaJSON: The developer-supplied JSON schema, if any, already
        ///     encoded. Used only to build the guided render's schema-in-prompt
        ///     rendering; the grammar constraint itself is compiled from this
        ///     same value independently in `runGuidedGeneration`.
        /// - Returns: The `PromptVariants` bundling every gate's render.
        /// - Throws: Whatever `UserInputProcessor.prepare` or reasoning-prompt
        ///   preparation throw.
        private func preparePromptVariants(
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            context: ModelContext,
            declaresReasoning: Bool,
            resolved: ModelConfiguration,
            needsEagerInput: Bool,
            input: LMInput?,
            schemaJSON: String?
        ) async throws -> PromptVariants {
            // Reasoning is only consumed by the unconstrained path
            // (no tools, no schema). On the guided/tool paths the
            // grammar already constrains output, so suppression-prep
            // would be wasted work.
            let mayRunReasoningPath =
                request.enabledToolDefinitions.isEmpty && request.schema == nil

            // When .reasoning is OMITTED on the unconstrained path,
            // re-render the prompt with thinking off so the model
            // doesn't emit `<think>`. Toggleable-only;
            // .alwaysOn was already rejected by `resolveRespondConfiguration`.
            let suppressedInput: LMInput?
            if Self.shouldSuppressReasoning(
                mayRunReasoningPath: mayRunReasoningPath, declaresReasoning: declaresReasoning,
                resolved: resolved),
                let suppressionConfig = resolved.reasoningConfig
            {
                suppressedInput = try await Self.preparedInput(
                    messages: messages, reasoningConfig: suppressionConfig,
                    thinkingEnabled: false, processor: context.processor,
                    transcriptEntries: request.transcript,
                    cannotDisableMessage: Self.alwaysReasoningDebugDescription
                )
            } else {
                suppressedInput = nil
            }

            let reasoningSetup:
                (input: LMInput, reasoningConfig: ReasoningConfig, primedInside: Bool)?
            if Self.shouldSetupReasoning(
                mayRunReasoningPath: mayRunReasoningPath, declaresReasoning: declaresReasoning,
                resolved: resolved),
                let reasoningConfig = resolved.reasoningConfig
            {
                let thinkingEnabled = Self.thinkingEnabled(
                    for: request.contextOptions.reasoningLevel)
                let reasoningInput = try await Self.preparedInput(
                    messages: messages, reasoningConfig: reasoningConfig,
                    thinkingEnabled: thinkingEnabled, processor: context.processor,
                    transcriptEntries: request.transcript,
                    cannotDisableMessage:
                        "This model always reasons; reasoning cannot be disabled via reasoningLevel."
                )
                reasoningSetup = (
                    reasoningInput, reasoningConfig,
                    Self.reasoningPrimedInside(
                        input: reasoningInput, reasoningConfig: reasoningConfig,
                        tokenizer: context.tokenizer)
                )
            } else {
                reasoningSetup = nil
            }

            // The prompt actually fed to guided generation. Honors
            // `ContextOptions.includeSchemaInPrompt` at the seam where the
            // guided-generation prompt is assembled -- see
            // `guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:)` --
            // and threads the resolved family's history-preservation context
            // through `makeGuidedRender`, so the live render and its
            // stable-boundary baseline agree (exactly like the unconstrained
            // path). Only re-renders when the schema seam actually appended
            // something OR a history context is in play: the eager `input`
            // was rendered context-free, so it is only reusable when the
            // guided render is context-free and message-identical --
            // reusing it then avoids a second, wasted
            // `UserInputProcessor.prepare` call for byte-identical messages.
            let guided: (input: LMInput, stableBoundary: StableBoundaryRender)?
            if Self.shouldBuildGuidedInput(
                needsEagerInput: needsEagerInput, schemaJSON: schemaJSON, input: input),
                let schemaJSON, let input
            {
                let guidedMessages = Self.guidedGenerationMessages(
                    from: messages, schemaJSON: schemaJSON,
                    includeSchemaInPrompt: request.contextOptions.includeSchemaInPrompt)
                let historyContext = resolved.reasoningConfig?.historyPreservationContext
                let render = Self.makeGuidedRender(
                    messages: guidedMessages, historyContext: historyContext)
                if guidedMessages.count == messages.count, historyContext == nil {
                    guided = (input, render.stableBoundary)
                } else {
                    guided = (
                        try await Self.preparedInputMappingImageFailures(
                            processor: context.processor, input: render.userInput,
                            messages: guidedMessages, transcriptEntries: request.transcript),
                        render.stableBoundary
                    )
                }
            } else {
                guided = nil
            }

            return PromptVariants(
                suppressed: suppressedInput, reasoningSetup: reasoningSetup, guided: guided)
        }

        /// Renders the prompt, resolves the per-instance configuration, and
        /// prepares whatever the reasoning path needs -- everything
        /// `respond()`'s `container.perform` closure must do before it can
        /// dispatch to one of the three generation paths.
        ///
        /// Delegates to ``resolveRespondConfiguration(modelID:context:declaresReasoning:configurationResolver:)``
        /// for the config-resolution chain and
        /// ``preparePromptVariants(request:messages:context:declaresReasoning:resolved:needsEagerInput:input:schemaJSON:)``
        /// for the reasoning/guided-input rendering gates, so this function
        /// itself only renders the eager (tooling-gated) prompt, calls both
        /// helpers in the order their throws must observe, and combines the
        /// results into a `RespondSetup`.
        ///
        /// - Parameters:
        ///   - request: The generation request; supplies tool/schema gating and reasoning level.
        ///   - messages: The rendered chat messages for this round.
        ///   - modelID: The model identifier used to build the `ModelDescriptor`.
        ///   - context: The loaded model context (tokenizer, configuration, processor).
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - configurationResolver: Patches the per-call `ModelConfiguration` read from `config.json`.
        ///   - schemaJSON: The developer-supplied JSON schema, if any, already
        ///     encoded. Used only to build `guidedInput`'s schema-in-prompt
        ///     rendering; the grammar constraint itself is compiled from this
        ///     same value independently in `runGuidedGeneration`.
        /// - Returns: A `RespondSetup` bundling the resolved configuration and effective input.
        /// - Throws: Whatever `UserInputProcessor.prepare`, capability validation, or
        ///   reasoning-prompt preparation throw.
        private func prepareRespondSetup(
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            modelID: String,
            context: ModelContext,
            declaresReasoning: Bool,
            configurationResolver: any ModelConfigurationResolver,
            schemaJSON: String?
        ) async throws -> RespondSetup {
            // Only render the prompt through the model's *default*
            // (no-`tools`) UserInputProcessor call when the tool-calling
            // branch won't run: `runToolCalling` re-tokenizes independently
            // via its own `context.processor.prepare(input:)` call (with
            // `tools:` populated, taking the tool-aware template branch --
            // see `runToolCalling`) and never reads
            // `input`/`effectiveInput`/`reasoningSetup`, so this render
            // would be both wasted work and unsafe on that branch -- a
            // continuation round's `messages` can carry a replayed
            // `.tool`-role message (see `TranscriptConverter.mlxMessages`),
            // and nothing guarantees a toolCalling-capable model's default
            // (no-`tools`) template handles `role == "tool"`.
            let needsEagerInput = request.enabledToolDefinitions.isEmpty
            let userInput = UserInput(chat: messages)
            let input: LMInput? =
                if needsEagerInput {
                    try await Self.preparedInputMappingImageFailures(
                        processor: context.processor, input: userInput, messages: messages,
                        transcriptEntries: request.transcript)
                } else {
                    nil
                }

            // Config resolution (and its reasoning-capability validation)
            // must run *after* the eager render above and *before* the
            // reasoning/guided-input gates below -- both a genuine ordering
            // dependency (the gates below read `resolved`) and a preserved
            // throw order (a request that would fail both the eager render
            // and the capability check surfaces the same error it always
            // has: the eager render's).
            let configResolution = try resolveRespondConfiguration(
                modelID: modelID, context: context, declaresReasoning: declaresReasoning,
                configurationResolver: configurationResolver)

            let variants = try await preparePromptVariants(
                request: request, messages: messages, context: context,
                declaresReasoning: declaresReasoning, resolved: configResolution.resolved,
                needsEagerInput: needsEagerInput, input: input, schemaJSON: schemaJSON)

            // The prompt actually fed into generation: the suppressed
            // prompt when we're forcing thinking off, otherwise the
            // baseline `input` rendered above. Both operands are `nil`
            // together when tools are enabled, so this stays `nil` in
            // that case too -- exactly the branch that never reads it.
            let effectiveInput = variants.suppressed ?? input

            return RespondSetup(
                input: input, resolved: configResolution.resolved, effectiveInput: effectiveInput,
                reasoningSetup: variants.reasoningSetup, guidedInput: variants.guided,
                contextLength: configResolution.contextLength)
        }

        /// Dispatches to the tool-calling, guided-generation, or plain-text
        /// path based on the request's tools/schema -- the three-way branch
        /// `respond()`'s `container.perform` closure previously ran inline.
        ///
        /// Each branch validates the resolved prompt's token count against
        /// `setup.contextLength` before doing any generation work (weight
        /// compute, grammar setup) -- see `validateContextSize`. The
        /// tool-calling branch re-tokenizes independently, so it performs
        /// its own check inside `runToolCalling` once that prompt exists.
        ///
        /// - Parameters:
        ///   - setup: The prepared prompt/configuration bundle from `prepareRespondSetup`.
        ///   - request: The generation request; supplies the enabled tools and schema.
        ///   - messages: The rendered chat messages for this round.
        ///   - modelID: The model identifier for constraint/tokenizer caches.
        ///   - schemaJSON: The developer-supplied JSON schema, if any, already encoded.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - requestedSamplingMode: The caller's sampling mode override, if any.
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - entryID: The response entry to stream output into.
        ///   - toolCallsEntryID: The entry to stream tool-call events into.
        ///   - reasoningEntryID: The entry to stream think-then-call/reasoning output into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Returns: `false` only when `runToolCalling`'s think-then-call Phase 1
        ///   was cut off before `</think>` closed -- the caller must skip its own
        ///   tail `Stream.gpu.synchronize()` and return immediately. `true` on
        ///   every other exit.
        /// - Throws: Whatever the underlying generation path throws.
        private func dispatchGeneration(
            setup: RespondSetup,
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            modelID: String,
            schemaJSON: String?,
            requestedMaxTokens: Int?,
            requestedSamplingMode: MLXSamplingMode?,
            declaresReasoning: Bool,
            entryID: String,
            toolCallsEntryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> Bool {
            if !request.enabledToolDefinitions.isEmpty {
                return try await runToolCalling(
                    request: request, messages: messages, modelID: modelID,
                    requestedMaxTokens: requestedMaxTokens,
                    requestedSamplingMode: requestedSamplingMode,
                    declaresReasoning: declaresReasoning, resolved: setup.resolved,
                    contextLength: setup.contextLength,
                    entryID: entryID, toolCallsEntryID: toolCallsEntryID,
                    reasoningEntryID: reasoningEntryID, context: context,
                    channel: channel)
            } else if let schemaJSON {
                // `prepareRespondSetup` only omits `guidedInput` when tools
                // are enabled, which this `else if` already excludes (the
                // tool-calling branch above returns first) -- `guidedInput`
                // is guaranteed present here. Validate whichever prompt will
                // actually reach generation -- `guidedInput`, which may carry
                // the adapter's own schema-in-prompt rendering and the
                // family's history-preservation context on top of `input` --
                // mirroring how the reasoning path below validates its own
                // primed prompt rather than always the baseline.
                let guided = Self.unwrapSetupField(
                    setup.guidedInput, fieldName: "guidedInput", contextPath: "guided-generation")
                try Self.validateContextSize(
                    tokenCount: guided.input.text.tokens.size, contextLength: setup.contextLength)
                try await runGuidedGeneration(
                    schemaJSON: schemaJSON, input: guided.input,
                    stableBoundary: guided.stableBoundary, modelID: modelID,
                    requestedMaxTokens: requestedMaxTokens, entryID: entryID,
                    context: context, channel: channel)
                return true
            } else {
                // Same guarantee as above: `effectiveInput` is only `nil`
                // when tools are enabled, and this is the no-tools/no-schema
                // path.
                let fallbackInput = Self.unwrapSetupField(
                    setup.effectiveInput, fieldName: "effectiveInput", contextPath: "text-generation")
                // `runTextGeneration` feeds `reasoningSetup.input` instead of
                // `fallbackInput` whenever reasoning is active (see its
                // body) -- validate whichever prompt will actually reach
                // generation, not always the baseline `fallbackInput`.
                let promptToValidate = setup.reasoningSetup?.input ?? fallbackInput
                try Self.validateContextSize(
                    tokenCount: promptToValidate.text.tokens.size, contextLength: setup.contextLength)
                try await runTextGeneration(
                    reasoningSetup: setup.reasoningSetup,
                    fallbackInput: fallbackInput,
                    messages: messages,
                    // The suppressed render (the only fed render that can
                    // carry it -- see `prepareRespondSetup`'s
                    // `effectiveInput`) merged the resolved family's history
                    // context; a config-free model resolves nil here.
                    historyContext: setup.resolved.reasoningConfig?.historyPreservationContext,
                    requestedMaxTokens: requestedMaxTokens,
                    requestedTemperature: request.generationOptions.temperature,
                    samplingMode: requestedSamplingMode,
                    responseEntryID: entryID,
                    reasoningEntryID: reasoningEntryID,
                    context: context,
                    channel: channel
                )
                return true
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
            _ = try Self.additionalContextOrThrowCapabilityError(
                promptStrategy: suppressionConfig.promptStrategy,
                thinkingEnabled: false,
                debugDescription: Self.alwaysReasoningDebugDescription)
        }

        /// Shared debug description for the two "this model cannot stop
        /// reasoning" error sites (`validateReasoningCapability`'s pre-flight
        /// gate and `preparedInput`'s unconstrained-path suppression
        /// attempt) that occur when `.reasoning` was not declared at init.
        private static let alwaysReasoningDebugDescription =
            "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output."

        /// Resolves `promptStrategy`'s chat-template `additionalContext` for
        /// `thinkingEnabled`, remapping `ReasoningError.cannotDisableReasoning`
        /// to the framework's typed `unsupportedCapability` error -- the
        /// catch-block pattern shared by every call site that asks a
        /// reasoning config to suppress thinking and must surface a
        /// developer-facing error when the strategy can't comply.
        ///
        /// - Parameters:
        ///   - promptStrategy: The reasoning config's prompt strategy to resolve.
        ///   - thinkingEnabled: `true` / `false` to force thinking on / off,
        ///     `nil` for no preference.
        ///   - debugDescription: The message to surface if `promptStrategy`
        ///     cannot honor `thinkingEnabled`.
        /// - Returns: The `additionalContext` to merge into the rendered prompt.
        /// - Throws: `LanguageModelError.unsupportedCapability(.reasoning)`
        ///   when `promptStrategy` cannot honor `thinkingEnabled`.
        private static func additionalContextOrThrowCapabilityError(
            promptStrategy: ReasoningPromptStrategy,
            thinkingEnabled: Bool?,
            debugDescription: String
        ) throws -> [String: any Sendable]? {
            do {
                return try promptStrategy.additionalContext(forThinkingEnabled: thinkingEnabled)
            } catch ReasoningError.cannotDisableReasoning {
                throw LanguageModelError.unsupportedCapability(
                    LanguageModelError.UnsupportedCapability(
                        capability: .reasoning,
                        debugDescription: debugDescription))
            }
        }

        /// Merges two optional chat-template `additionalContext` fragments
        /// -- the prompt strategy's thinking flag (`enable_thinking`) and the
        /// family's history-preservation flag (`preserve_thinking`) -- into
        /// the single dictionary `applyChatTemplate` takes. `nil` + `nil`
        /// stays `nil`, so renders without any context remain byte-identical
        /// to before. The fragments' keys are disjoint by construction
        /// (strategy kwarg vs. `ReasoningConfig.historyPreservationKey`);
        /// `rhs` wins on a collision.
        ///
        /// - Parameters:
        ///   - lhs: The first context fragment, if any.
        ///   - rhs: The second context fragment, if any (wins on collision).
        /// - Returns: The merged context, or `nil` when both are `nil`.
        static func mergedAdditionalContext(
            _ lhs: [String: any Sendable]?,
            _ rhs: [String: any Sendable]?
        ) -> [String: any Sendable]? {
            switch (lhs, rhs) {
            case (nil, nil): return nil
            case (let some?, nil): return some
            case (nil, let some?): return some
            case (let a?, let b?): return a.merging(b) { _, new in new }
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
                tokenizer: context.tokenizer,
                configuration: context.configuration
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
            // The structural reserve is the bare minimum tokens for the
            // JSON skeleton (empty strings). `ConstraintSetup.reserves`
            // derives the completion reserve (3x structural minimum, or
            // 25% of whatever maxTokens is in play) and the hard reserve
            // (8x structural minimum -- larger than the raw estimate
            // because grammar-forced key names (FF tokens) and
            // model-inserted whitespace cost more tokens than the compact
            // minimal JSON string) from this on demand, so they always
            // stay in sync with the budget of the specific generation call
            // they're applied to.
            let structuralReserve = CompletionReserve.estimate(
                schemaJSON: reserveEstimateSource,
                tokenizer: context.tokenizer
            )

            return ConstraintSetup(
                xgTokenizer: xgTokenizer, constraint: constraint, maxTokens: maxTokens,
                closingBias: bias.closing, structuralReserve: structuralReserve,
                whitespaceBias: bias.whitespace,
                whitespaceTokenIDs: bias.whitespaceTokenIDs
            )
        }

        // MARK: - Channel & Error Surface Conformance
        //
        // Audited against the macOS 27 SDK's
        // `FoundationModels.swiftmodule/*.swiftinterface` (ground truth --
        // not WWDC docs/videos, which can lag or gloss over the exact case
        // list). Every `LanguageModelExecutorGenerationChannel` channel-action
        // case and every `LanguageModelError` case is accounted for below as
        // Emitted or Deliberately-N/A-because-X; no case is silently
        // unimplemented. If a future audit finds a case that's genuinely
        // needed (not legitimately N/A), that becomes its own kanban task
        // rather than a silent gap here.
        //
        // ## Response channel (`Response.Action`)
        // - `.appendText` -- EMITTED. `sendDelta(isReasoning: false)` on every
        //   unconstrained/guided/tool-calling response chunk.
        // - `.updateMetadata` -- EMITTED. The `modelID`/`requestID` stamp sent
        //   at the top of `respond()`, and `sendIncompleteOutputMetadata`
        //   when generation is cut off by the token budget.
        // - `.updateUsage` -- EMITTED. `sendUsageUpdate`, once per round (see
        //   its doc comment: usage is single-source-of-truth, never summed
        //   from per-delta events).
        // - `.replaceTextSegment` -- N/A. MLX decoding is append-only and
        //   text-only: this adapter never revises text it has already
        //   streamed.
        // - `.updateCustomSegment` -- N/A. Custom transcript segments are a
        //   developer-defined extension point; this adapter has no custom
        //   segment type to produce.
        // - `.addAttachmentSegment` / `.removeAttachmentSegment` -- N/A. This
        //   adapter's generation output is text-only -- no attachment
        //   content is ever produced, so none is ever added or removed.
        //
        // ## Tool-calls channel (`ToolCalls.Action` and nested `ToolCall.Action`)
        // - `.toolCall(_:.appendArguments)` -- EMITTED. `handleRealTool` sends
        //   exactly one `.appendArguments` per detected real tool call -- the
        //   grammar-constrained envelope is decoded in full before this
        //   fires, so there is no partial-arguments stream to append
        //   further to.
        // - `.toolCall(_:.updateMetadata)` -- N/A. No per-tool-call metadata
        //   is produced today.
        // - `.removeToolCall` -- N/A. Tool calls are emitted atomically as
        //   one complete envelope; there is never a partially-emitted call
        //   to retract.
        // - Entry-level `.updateMetadata` / `.updateUsage` on `ToolCalls` --
        //   N/A. Usage is reported exactly once, on the `.response` entry
        //   (see `sendUsageUpdate`'s single-source-of-truth design);
        //   duplicating it onto the `.toolCalls` entry would double-count
        //   for any consumer that sums usage across entries.
        //
        // ## Reasoning channel (`Reasoning.Action`)
        // - `.appendText` -- EMITTED. `sendDelta(isReasoning: true)` for
        //   every `<think>...</think>`-routed chunk (see
        //   `ReasoningEventEmitter`).
        // - `.replaceTextSegment` -- N/A. Same append-only rationale as the
        //   response channel.
        // - `.updateSignature` -- N/A. A reasoning signature is an Apple
        //   on-device/PCC-only concept -- cryptographically attesting the
        //   reasoning came from Apple's own model. Open-weights MLX models
        //   have no signing key and no equivalent artifact to attach.
        // - `.updateMetadata` / `.updateUsage` -- N/A. Same
        //   single-source-of-truth rationale as `ToolCalls` above: usage and
        //   incomplete-output metadata for a round live on the `.response`
        //   entry only.
        //
        // ## Thrown errors (`LanguageModelError`)
        // - `.contextSizeExceeded` -- THROWN. `validateContextSize`, when a
        //   prompt (or, for think-then-call, Phase 2's reasoning-extended
        //   prompt) exceeds the model's context window.
        // - `.unsupportedCapability` -- THROWN. Images without `.vision`
        //   declared, and forced/declared reasoning the resolved prompt
        //   strategy can't honor (see `validateReasoningCapability` /
        //   `additionalContextOrThrowCapabilityError`).
        // - `.unsupportedGenerationGuide` -- THROWN. `mapGrammarError` maps
        //   xgrammar's `invalidJSONSchema` (a developer-authored
        //   `GenerationSchema` xgrammar outright rejects) to this case.
        // - `.unsupportedTranscriptContent` -- THROWN.
        //   `preparedInputMappingImageFailures` maps a `UserInputProcessor`
        //   failure to this case when the transcript carries image content
        //   the concrete model can't process.
        // - `.rateLimited` -- N/A. Local, on-device inference has no
        //   request-rate gate to violate; this case models a network/service
        //   quota this adapter has no equivalent of.
        // - `.guardrailViolation` / `.refusal` -- N/A. Both model Apple's
        //   built-in safety classifier rejecting (or explaining a rejection
        //   of) content. This adapter runs open-weights models with no
        //   equivalent built-in content-safety system to report through --
        //   any model-side refusal comes back as ordinary generated text,
        //   not a typed error.
        // - `.unsupportedLanguageOrLocale` -- N/A. This adapter does not gate
        //   on locale; it decodes whatever token stream the model produces
        //   regardless of language.
        // - `.timeout` -- N/A. Models a network/service round-trip timing
        //   out (Apple's Private Cloud Compute path); local MLX generation
        //   has no network round trip. It is bounded instead by
        //   `GenerationOptions.maximumResponseTokens` (`defaultMaxTokens`),
        //   and external cancellation surfaces as `CancellationError`, not
        //   this case.
        //
        // ## Well-formedness invariants (see `ExecutorEventWellFormednessTests.swift`)
        // - Every `.appendText`/`.appendArguments` event this adapter emits
        //   carries `textDeltaTokenCount` (`1`, never `0`) -- the SDK's
        //   `TextFragment`/`ArgumentsFragment.tokenCount` is a non-optional
        //   `Int`, so it must always be an honest, nonzero count.
        // - `entryID`/`reasoningEntryID`/`toolCallsEntryID` are minted
        //   exactly once per `respond()` round (its three `UUID().uuidString`
        //   locals) and threaded unchanged through every nested call for
        //   that round -- every fragment belonging to one round's
        //   response/reasoning/tool-calls entry carries the same id.
        // - `sendUsageUpdate` is the single authoritative `.updateUsage` per
        //   round (never per-delta), and clamps `cachedTokenCount`/
        //   `reasoningTokenCount` to their respective totals via
        //   `clampedUsageCounts` so a round's reported usage can never
        //   overstate itself.
        // - `LanguageModelExecutorGenerationChannel.Event` (and every nested
        //   `Action` type) is opaque once constructed -- the swiftinterface
        //   exposes only static factory functions, no public accessors to
        //   read a value back out. No test -- not even a full end-to-end
        //   `respond()` run -- can drain a channel and confirm the entryID
        //   *value* an emitted `Event` actually carries. The
        //   well-formedness tests instead assert on these values at the
        //   point this adapter computes them, using the very helpers below
        //   (widened from `private` to package-internal access for that
        //   purpose), against a real channel to prove every send actually
        //   completes; the entryID-stability guarantee above remains a
        //   structural property of `respond()`'s source, not something
        //   re-verified at runtime.

        /// Metadata key signaling the model's output was cut off before
        /// completing naturally (budget exhausted mid-thought or
        /// mid-structure), so a consumer doesn't mistake a partial answer
        /// for the model's chosen response.
        private static let incompleteOutputMetadataKey = "incompleteOutput"

        /// The token count reported alongside a single text/argument delta
        /// event (`.appendText`/`.appendArguments`). Every delta this
        /// adapter emits carries exactly one already-decoded chunk, so this
        /// is always `1`, never a real per-chunk token tally -- named so the
        /// several call sites that report it don't repeat the bare literal.
        ///
        /// Package-internal (not `private`) so
        /// `ExecutorEventWellFormednessTests.swift` can assert on it
        /// directly -- see the "Well-formedness invariants" note above.
        static let textDeltaTokenCount = 1

        /// Sends the incomplete-output metadata signal for `entryID`.
        ///
        /// Package-internal (not `private`) so
        /// `ExecutorEventWellFormednessTests.swift` can call it directly --
        /// see the "Well-formedness invariants" note above.
        ///
        /// - Parameters:
        ///   - entryID: The response entry the signal applies to.
        ///   - channel: The generation channel to send the signal on.
        static func sendIncompleteOutputMetadata(
            entryID: String, channel: LanguageModelExecutorGenerationChannel
        ) async {
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateMetadata([Self.incompleteOutputMetadataKey: true])))
        }

        /// Sends a single text or reasoning delta for `entryID`.
        ///
        /// Package-internal (not `private`) so
        /// `ExecutorEventWellFormednessTests.swift` can call it directly --
        /// see the "Well-formedness invariants" note above.
        ///
        /// - Parameters:
        ///   - text: The delta text to append.
        ///   - entryID: The response/reasoning entry to stream into.
        ///   - channel: The generation channel to send the delta on.
        ///   - isReasoning: `true` to send on the `.reasoning` channel entry,
        ///     `false` to send on the `.response` channel entry.
        static func sendDelta(
            _ text: String, entryID: String, channel: LanguageModelExecutorGenerationChannel,
            isReasoning: Bool
        ) async {
            if isReasoning {
                await channel.send(
                    .reasoning(
                        entryID: entryID, action: .appendText(text, tokenCount: textDeltaTokenCount)))
            } else {
                await channel.send(
                    .response(
                        entryID: entryID, action: .appendText(text, tokenCount: textDeltaTokenCount)))
            }
        }

        /// Clamps a round's `cachedTokenCount`/`reasoningTokenCount` to their
        /// respective totals before they're embedded in a `.updateUsage`
        /// event, so a mismatched upstream count (e.g. the tool-calling
        /// think-then-call path's cache count, computed against Phase 2's
        /// longer prompt) can never overstate itself in the reported usage.
        ///
        /// Pulled out of `sendUsageUpdate` as a pure function -- with no
        /// `LanguageModelExecutorGenerationChannel` dependency -- so this
        /// adapter's usage-well-formedness guarantee is directly
        /// unit-testable; see the "Well-formedness invariants" note above
        /// `sendIncompleteOutputMetadata`.
        ///
        /// - Parameters:
        ///   - promptTokenCount: The prompt's total token count.
        ///   - cachedTokenCount: The candidate cached-token count, clamped to
        ///     `promptTokenCount`.
        ///   - outputTokenCount: The generated output's total token count.
        ///   - reasoningTokenCount: The candidate reasoning-token count,
        ///     clamped to `outputTokenCount`.
        /// - Returns: The clamped `(cachedTokenCount, reasoningTokenCount)` pair.
        static func clampedUsageCounts(
            promptTokenCount: Int,
            cachedTokenCount: Int,
            outputTokenCount: Int,
            reasoningTokenCount: Int
        ) -> (cachedTokenCount: Int, reasoningTokenCount: Int) {
            (
                Swift.min(cachedTokenCount, promptTokenCount),
                Swift.min(reasoningTokenCount, outputTokenCount)
            )
        }

        /// Sends the authoritative `.updateUsage` event for `entryID`.
        ///
        /// Package-internal (not `private`) so
        /// `ExecutorEventWellFormednessTests.swift` can call it directly --
        /// see the "Well-formedness invariants" note above.
        ///
        /// - Parameters:
        ///   - entryID: The response entry the usage applies to.
        ///   - promptTokenCount: The prompt's total token count (the full
        ///     transcript, not shrunk by any `PromptCache` reuse -- see
        ///     `PromptCacheSlot.cachedTokenCount`).
        ///   - cachedTokenCount: How many of `promptTokenCount` were served
        ///     from a prior round's `PromptCache` slot rather than re-fed
        ///     this round; 0 when this round didn't reuse a cache. Clamped
        ///     to `promptTokenCount` via `clampedUsageCounts`: the
        ///     tool-calling think-then-call path computes this against
        ///     Phase 2's prompt (the base tool-aware input plus Phase 1's
        ///     reasoning tokens), which can be longer than `promptTokenCount`
        ///     itself (the base input alone, to avoid double-counting
        ///     reasoning tokens as both prompt and output) -- without the
        ///     clamp that mismatch could report more cached tokens than the
        ///     prompt is claimed to contain.
        ///   - outputTokenCount: The generated output's total token count.
        ///   - reasoningTokenCount: The subset of the output spent on
        ///     reasoning (0 on paths that don't reason); clamped to
        ///     `outputTokenCount` via `clampedUsageCounts`.
        ///   - channel: The generation channel to send the event on.
        static func sendUsageUpdate(
            entryID: String,
            promptTokenCount: Int,
            cachedTokenCount: Int,
            outputTokenCount: Int,
            reasoningTokenCount: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            let clamped = Self.clampedUsageCounts(
                promptTokenCount: promptTokenCount, cachedTokenCount: cachedTokenCount,
                outputTokenCount: outputTokenCount, reasoningTokenCount: reasoningTokenCount)
            await channel.send(
                .response(
                    entryID: entryID,
                    action: .updateUsage(
                        input: .init(
                            totalTokenCount: promptTokenCount,
                            cachedTokenCount: clamped.cachedTokenCount),
                        output: .init(
                            totalTokenCount: outputTokenCount,
                            reasoningTokenCount: clamped.reasoningTokenCount)
                    )))
        }

        // MARK: - Prompt Cache

        /// Whether `input` carries only text -- no image/video/audio
        /// payload. The prompt-cache optimization reduces `input` to a raw
        /// token suffix, which would silently drop any attached media
        /// (there is nowhere for it to travel to on a cache-reuse round),
        /// so multimodal rounds are kept off the cache path entirely and
        /// always get the original, unreduced `input` with `cache: nil` --
        /// identical to pre-cache behavior.
        ///
        /// - Parameter input: The prompt to check.
        /// - Returns: `true` when `input` carries no image/video/audio payload.
        private static func isTextOnly(_ input: LMInput) -> Bool {
            input.image == nil && input.video == nil && input.audio == nil
        }

        /// What a generation call needs to participate in prompt-cache
        /// reuse: the `[KVCache]` to generate with (`nil` when this round
        /// doesn't qualify -- see `isTextOnly`), the `LMInput` to actually
        /// feed (the original input, unless a stored prefix let it shrink
        /// to a token suffix), and the full prompt token sequence (for
        /// `commitPromptCache` to key the next round's entry on).
        private struct PromptCacheSlot {
            let cache: [KVCache]?
            let feedInput: LMInput
            let promptTokens: [Int]

            /// How many of `promptTokens` this round's hybrid SPLIT PREFILL
            /// re-fed itself, outside `feedInput` (the span between the
            /// resolved match and the transcript-stable boundary -- see
            /// `makePromptCacheSlot`). Those tokens were forward-passed
            /// fresh this round, so they must not be reported as served
            /// from cache. Zero on every non-split round.
            var splitPrefillTokenCount = 0

            /// How many of `promptTokens` were served from a prior
            /// round's `PromptCache` slot rather than re-fed this round --
            /// `promptTokens.count` minus every token actually fed this
            /// round (the `feedInput` suffix plus any split-prefill span).
            /// Zero when this round didn't participate in prompt-cache
            /// reuse (multimodal input, first turn, or a full rebuild,
            /// all of which feed the entire prompt).
            var cachedTokenCount: Int {
                promptTokens.count - feedInput.text.tokens.size - splitPrefillTokenCount
            }
        }

        /// What `makePromptCacheSlot` needs to re-render this round's
        /// conversation as PAST turns and compute its transcript-stable
        /// boundary (see `transcriptStableLength`): the same messages and
        /// tool specs the round's prompt was rendered from, plus the
        /// history-affecting `additionalContext` the live render carried
        /// (`historyContext`) -- but never the reasoning-suppression/priming
        /// flags, which belong to the generation region a past-turns render
        /// omits. `nil` at `makePromptCacheSlot` means the caller can't
        /// supply them, so the hybrid split prefill is skipped -- behavior
        /// unchanged.
        ///
        /// `internal` (not `private`) solely as a test seam:
        /// `PromptCacheHybridArchitectureTests` inspects the
        /// `makeGuidedRender` pair's boundary fields via `@testable import`,
        /// which never elevates `private`/`fileprivate` access.
        struct StableBoundaryRender {
            /// The chat messages this round's prompt was rendered from.
            let messages: [Chat.Message]
            /// The tool specs rendered into the prompt (the tool-calling
            /// path); `nil` on the tool-free paths.
            let tools: [[String: any Sendable]]?
            /// The history-affecting `additionalContext` the round's LIVE
            /// render carried (`ReasoningConfig.historyPreservationContext`,
            /// e.g. Qwen3.6's `preserve_thinking: true`), or `nil` when the
            /// live render had none. The past-turns baseline must render
            /// with the SAME history context -- a mismatched baseline
            /// diverges from the live prompt at the first history turn,
            /// collapsing the stable prefix. Generation-region-only flags
            /// (`enable_thinking`) are deliberately NOT part of this: they
            /// don't affect a render without a generation prompt.
            let historyContext: [String: any Sendable]?
        }

        /// Builds the guided-generation path's live-render input and its
        /// stable-boundary baseline from ONE history-affecting context, so
        /// the two renders cannot disagree.
        ///
        /// The guided path threads the resolved family's
        /// `ReasoningConfig.historyPreservationContext` (e.g. Qwen3.6's
        /// `preserve_thinking: true`) through this single seam: the returned
        /// `userInput` carries it as the live render's `additionalContext`,
        /// and the returned `stableBoundary` carries the same value for the
        /// past-turns baseline `transcriptStableLength` renders -- exactly
        /// the pairing the unconstrained path maintains (see
        /// `runTextGeneration`). A `nil` context keeps both sides
        /// context-free, byte-identical to the pre-history-preservation
        /// renders. Reasoning-suppression/priming flags never appear here:
        /// the guided path carries no thinking toggle, and the boundary
        /// baseline must exclude generation-region flags by design (see
        /// `StableBoundaryRender.historyContext`).
        ///
        /// - Parameters:
        ///   - messages: The guided-generation messages for this round
        ///     (already routed through `guidedGenerationMessages`).
        ///   - historyContext: The resolved family's history-preservation
        ///     context, or `nil` when it has none.
        /// - Returns: The live-render `UserInput` and the matching
        ///   stable-boundary baseline.
        static func makeGuidedRender(
            messages: [Chat.Message],
            historyContext: [String: any Sendable]?
        ) -> (userInput: UserInput, stableBoundary: StableBoundaryRender) {
            (
                UserInput(chat: messages, additionalContext: historyContext),
                StableBoundaryRender(
                    messages: messages, tools: nil, historyContext: historyContext)
            )
        }

        /// The `isTextOnly` gate plus the `PromptCache.resolve` call it
        /// guards -- the one piece of resolve→gate logic shared verbatim by
        /// `makePromptCacheSlot` and `populatePromptCacheChunks`. Returns
        /// `nil` for multimodal `input` (see `isTextOnly`) without ever
        /// touching `model`, leaving each caller free to decide what "not
        /// cacheable" means for it: `makePromptCacheSlot` still needs to
        /// generate a real, uncached response either way, while
        /// `populatePromptCacheChunks` has nothing left to do and skips
        /// outright.
        ///
        /// - Parameters:
        ///   - input: The full, unreduced prompt for this round.
        ///   - tokens: `input.text.tokens`, already extracted by the caller
        ///     (both callers need it regardless of this gate's outcome, so
        ///     it isn't re-derived here).
        ///   - model: The model to resolve the cache against.
        ///   - parameters: Generation parameters, threaded through to a
        ///     rebuilt cache (see `PromptCache.resolve`); `nil` for the
        ///     grammar-constrained paths, matching
        ///     `GuidedGenerationLoop.run`'s own `model.newCache(parameters:
        ///     nil)`.
        private func resolvePromptCacheIfTextOnly(
            input: LMInput, tokens: [Int], model: any MLXLMCommon.LanguageModel,
            parameters: GenerateParameters?
        ) async -> (cache: [KVCache], tokensToFeed: [Int])? {
            guard Self.isTextOnly(input) else { return nil }
            return await MLXLanguageModel.resolvePromptCache(
                modelID: modelID, newTokens: tokens, model: model, parameters: parameters)
        }

        /// Resolves a generation call's prompt-cache participation against
        /// `input`. See `PromptCacheSlot`.
        ///
        /// For a hybrid Mamba/attention cache with a computable
        /// transcript-stable boundary (see `transcriptStableLength`), SPLITS
        /// the prefill: forward-passes the span between the resolved match
        /// and the stable boundary itself, snapshots a hybrid checkpoint AT
        /// the boundary -- before any generation-region tokens touch the
        /// cache -- and hands only the remainder to generation. This is what
        /// makes hybrid checkpoints matchable across rounds at all for
        /// template families that inject generation-priming tokens after the
        /// final assistant header (Qwen3.6's thinking suppression): a
        /// post-round checkpoint bakes those tokens into the middle of its
        /// sequence, and Mamba state cannot be trimmed backward past them,
        /// so it can never match a later round's re-render (kanban er33v06).
        /// The post-round store in `commitPromptCache` is still made too --
        /// it remains correct, and useful for templates without
        /// generation-region injection -- and both simply compete in
        /// `PromptCache.resolveHybridCheckpoint`'s longest-prefix scan.
        ///
        /// - Parameters:
        ///   - input: The full, unreduced prompt for this round.
        ///   - context: The loaded model context.
        ///   - parameters: Generation parameters, threaded through to a
        ///     rebuilt cache (see `PromptCache.resolve`); `nil` for the
        ///     grammar-constrained paths, matching
        ///     `GuidedGenerationLoop.run`'s own `model.newCache(parameters:
        ///     nil)`.
        ///   - stableBoundary: The messages/tools to compute the
        ///     transcript-stable boundary from, or `nil` to skip the hybrid
        ///     split prefill entirely.
        /// - Throws: `CancellationError` if the task is cancelled during the
        ///   split prefill's forward pass (before any checkpoint is stored,
        ///   so nothing partial ever persists).
        private func makePromptCacheSlot(
            input: LMInput, context: ModelContext, parameters: GenerateParameters?,
            stableBoundary: StableBoundaryRender?
        ) async throws -> PromptCacheSlot {
            let promptTokens = input.text.tokens.asArray(Int.self)
            guard
                let resolved = await resolvePromptCacheIfTextOnly(
                    input: input, tokens: promptTokens, model: context.model,
                    parameters: parameters)
            else {
                return PromptCacheSlot(cache: nil, feedInput: input, promptTokens: promptTokens)
            }
            if let stableBoundary, PromptCache.isHybridMambaAttention(resolved.cache),
                let slot = try await makeSplitPrefillSlot(
                    promptTokens: promptTokens, resolved: resolved,
                    stableBoundary: stableBoundary, context: context)
            {
                return slot
            }
            return PromptCacheSlot(
                cache: resolved.cache,
                feedInput: LMInput(tokens: MLXArray(resolved.tokensToFeed)),
                promptTokens: promptTokens)
        }

        /// The hybrid split-prefill arm of `makePromptCacheSlot` (see that
        /// method's doc comment for the mechanism and why it exists):
        /// computes the transcript-stable boundary, forward-passes
        /// `promptTokens[matchedLength..<stableLength]` through the resolved
        /// cache (`prefillPromptCache`), snapshots a hybrid checkpoint keyed
        /// on exactly `promptTokens[0..<stableLength]` (the cache's offset
        /// equals that length by construction, so `PromptCache.store`'s
        /// hybrid `offset == tokens.count` verification passes), and returns
        /// a slot feeding only `promptTokens[stableLength...]` to generation.
        ///
        /// Converts `stableBoundary.messages` via `DefaultMessageGenerator`:
        /// the model's OWN `MessageGenerator` lives behind
        /// `UserInputProcessor` and isn't reachable from this module, and
        /// the hybrid families this path serves (Qwen3.6/Qwen3-Next) use the
        /// default generator anyway. A mismatched generator merely shortens
        /// the computed common prefix (less reuse, split skipped) -- it can
        /// never mis-position the checkpoint, because the boundary is a
        /// common-prefix length against the REAL `promptTokens`, and the
        /// checkpoint stores exactly the tokens it forward-passed.
        ///
        /// - Parameters:
        ///   - promptTokens: This round's full rendered prompt tokens.
        ///   - resolved: The `PromptCache.resolve` outcome for this round.
        ///   - stableBoundary: The messages/tools to render the past-turns
        ///     baseline from.
        ///   - context: The loaded model context.
        /// - Returns: The split slot, or `nil` when no boundary is
        ///   computable or the split wouldn't help (the resolved match
        ///   already reaches the boundary, or the boundary covers the whole
        ///   prompt, leaving nothing to feed generation) -- the cache is
        ///   untouched in every `nil` case, so the caller's ordinary slot
        ///   stays valid.
        /// - Throws: `CancellationError` if the task is cancelled during the
        ///   prefill forward pass.
        private func makeSplitPrefillSlot(
            promptTokens: [Int],
            resolved: (cache: [KVCache], tokensToFeed: [Int]),
            stableBoundary: StableBoundaryRender,
            context: ModelContext
        ) async throws -> PromptCacheSlot? {
            let matchedLength = promptTokens.count - resolved.tokensToFeed.count
            guard
                let stableLength = Self.transcriptStableLength(
                    promptTokens: promptTokens,
                    messages: DefaultMessageGenerator().generate(
                        messages: stableBoundary.messages),
                    tools: stableBoundary.tools,
                    additionalContext: stableBoundary.historyContext,
                    tokenizer: context.tokenizer),
                matchedLength < stableLength, stableLength < promptTokens.count
            else { return nil }

            try Self.prefillPromptCache(
                tokens: Array(promptTokens[matchedLength ..< stableLength]),
                model: context.model, cache: resolved.cache)
            await MLXLanguageModel.storePromptCache(
                modelID: modelID, tokens: Array(promptTokens[0 ..< stableLength]),
                cache: resolved.cache)
            return PromptCacheSlot(
                cache: resolved.cache,
                feedInput: LMInput(tokens: MLXArray(Array(promptTokens[stableLength...]))),
                promptTokens: promptTokens,
                splitPrefillTokenCount: stableLength - matchedLength)
        }

        /// Length of the longest shared leading run of `lhs` and `rhs`.
        ///
        /// - Parameters:
        ///   - lhs: One token sequence.
        ///   - rhs: The other token sequence.
        /// - Returns: How many leading elements the two sequences share.
        static func commonPrefixLength(_ lhs: [Int], _ rhs: [Int]) -> Int {
            var length = 0
            while length < lhs.count, length < rhs.count, lhs[length] == rhs[length] {
                length += 1
            }
            return length
        }

        /// Computes the transcript-stable boundary of `promptTokens`: the
        /// length of its longest prefix that later rounds re-render
        /// identically once this round's turns are history.
        ///
        /// Renders the SAME conversation as past turns -- `addGenerationPrompt:
        /// false`, with only the history-affecting `additionalContext` the
        /// live render carried (never its reasoning-suppression/priming
        /// flags, which only affect the generation region) -- because that is
        /// exactly the form a future round re-renders these turns in. The
        /// common prefix of this round's live prompt against that baseline is
        /// therefore the prefix that is guaranteed to reappear verbatim at
        /// the start of every later round's prompt: everything past it
        /// (assistant header, thinking-suppression injections, other
        /// generation-priming tokens) belongs to the template's generation
        /// region and is re-rendered differently -- or not at all -- once the
        /// turn is history (see kanban er33v06's token-level trace for
        /// Qwen3.6's `[248068, 198]` injection).
        ///
        /// - Parameters:
        ///   - promptTokens: This round's full rendered prompt tokens.
        ///   - messages: The raw message dictionaries the prompt was rendered from.
        ///   - tools: The tool specs rendered into the prompt, if any.
        ///   - additionalContext: The history-affecting context the live
        ///     render carried (see `StableBoundaryRender.historyContext`),
        ///     or `nil` when it had none.
        ///   - tokenizer: The tokenizer to render the past-turns baseline with.
        /// - Returns: The stable prefix length, or `nil` when the tokenizer
        ///   cannot control the generation prompt (the optional
        ///   `applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:)`
        ///   capability returned `nil`) or the render throws -- callers skip
        ///   the boundary-dependent behavior, leaving this round unchanged.
        static func transcriptStableLength(
            promptTokens: [Int],
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            tokenizer: any Tokenizer
        ) -> Int? {
            // `try?` flattens the render's own optional (the capability's
            // `nil` opt-out) and a thrown error into one `nil` -- both mean
            // "no stable boundary computable" here.
            guard
                let baseTokens = try? tokenizer.applyChatTemplate(
                    messages: messages, tools: tools, additionalContext: additionalContext,
                    addGenerationPrompt: false)
            else { return nil }
            return commonPrefixLength(promptTokens, baseTokens)
        }

        /// Forward-passes `tokens` through `model` into `cache`, in chunked
        /// prefill steps -- mirroring `LLMModel.prepare`'s own prefill
        /// conventions (512-token windows, `asyncEval` per chunk so the CPU
        /// builds chunk N+1's graph while the GPU evaluates chunk N, one
        /// `eval` to flush at the end).
        ///
        /// Used by `makePromptCacheSlot`'s hybrid split prefill to advance a
        /// resolved cache exactly to the transcript-stable boundary before
        /// snapshotting a checkpoint there -- and an internal test seam, so
        /// unit tests can drive the same forward-pass path against tiny
        /// synthetic hybrid models without a full `ModelContext`.
        ///
        /// - Parameters:
        ///   - tokens: The token span to prefill.
        ///   - model: The model to forward-pass through.
        ///   - cache: The live cache to advance; every layer's state advances
        ///     by exactly `tokens.count` positions once this returns.
        /// - Throws: `CancellationError` if the task is cancelled between
        ///   chunks (mirroring `LLMModel.prepare`'s cooperative-cancellation
        ///   check; see ml-explore/mlx-swift-examples#230).
        static func prefillPromptCache(
            tokens: [Int], model: any MLXLMCommon.LanguageModel, cache: [KVCache]
        ) throws {
            let prefillStepSize = 512
            var y = LMInput.Text(tokens: MLXArray(tokens))
            var state: LMOutput.State?
            try withPreparedCache(cache, lengths: y.sequenceLengths) {
                while y.tokens.size > 0 {
                    try Task.checkCancellation()
                    let step = min(prefillStepSize, y.tokens.size)
                    let output = model(
                        y[.newAxis, ..<step], cache: cache.isEmpty ? nil : cache, state: state)
                    state = output.state
                    asyncEval(cache)
                    y = y[step...]
                }
                eval(cache)
            }
        }

        /// Trims `cache` back into sync with the observed generated-token
        /// count after `PromptCache.reconcileCacheAdvance` reported
        /// `.trimCacheByOne`: the cache's real offset is one token ahead of
        /// what was observed (see `reconcileCacheAdvance`'s doc), so trim it
        /// back to match rather than storing a token count that doesn't
        /// correspond to the cache's actual state.
        ///
        /// - Parameters:
        ///   - cache: The slot's KV cache to trim.
        ///   - slot: The slot this round generated with (supplies the
        ///     prompt-token baseline the offsets are relative to).
        ///   - cacheAdvance: `cache`'s observed offset advance beyond
        ///     `slot.promptTokens`.
        ///   - generatedTokenCount: The observed generated-token count to
        ///     trim back to.
        /// - Returns: `true` when the cache was trimmable and the trim
        ///   verified, so the entry is safe to store; `false` when it must
        ///   be dropped instead.
        private static func trimCacheIfValid(
            _ cache: [KVCache], slot: PromptCacheSlot, cacheAdvance: Int, generatedTokenCount: Int
        ) -> Bool {
            canTrimPromptCache(cache)
                && PromptCache.trimAndVerify(
                    cache, from: slot.promptTokens.count + cacheAdvance,
                    to: slot.promptTokens.count + generatedTokenCount)
        }

        /// Reconciles this round's actual generated token IDs against
        /// `cache`'s own authoritative `offset` advance beyond
        /// `slot.promptTokens`, then persists (or invalidates)
        /// `PromptCache`'s entry for this model.
        ///
        /// `cache.offset` is the only ground truth for how many tokens the
        /// cache actually holds -- trusting a caller-supplied token count
        /// that doesn't match it risks seeding a future round with token
        /// values that don't correspond to real cache state (wrong
        /// positions, wrong attention). A mismatch here means
        /// `generatedTokenIDs` can't be trusted to reflect `cache`
        /// exactly, so the entry is dropped rather than stored.
        ///
        /// - Parameters:
        ///   - slot: The slot this round generated with. A no-op when
        ///     `slot.cache` is `nil` (this round didn't participate in
        ///     prompt-cache reuse -- see `isTextOnly`).
        ///   - generatedTokenIDs: This round's generated token IDs, in
        ///     order -- the REAL IDs the model produced (from a raw
        ///     `.token(Int)`/`onTokenCommitted` stream), never a
        ///     re-encoding of decoded text.
        private static func commitPromptCache(
            modelID: String, slot: PromptCacheSlot, generatedTokenIDs: [Int],
            stopTokenFedToCache: Int? = nil
        ) async {
            guard let cache = slot.cache else { return }
            guard !generatedTokenIDs.isEmpty else { return }
            // `PromptCache.cacheAdvanceOffset` picks the right ground-truth
            // offset for whichever cache shape `cache` actually is -- for a
            // genuine hybrid Mamba/attention stack (Qwen3.6/Qwen3-Next
            // family) this is NOT `cache.first?.offset`: layer 0 is a
            // `MambaCache` by default, and neither
            // `Libraries/MLXLLM/Models/Qwen35.swift` nor `Qwen3Next.swift`
            // ever assigns to that layer's `offset` (unlike e.g.
            // `FalconH1.swift`'s hybrid attention/recurrent layer, which
            // DOES advance its cache's `offset` -- a per-model-family fact,
            // not a blanket guarantee of `MambaCache`/`ArraysCache` itself).
            // See `cacheAdvanceOffset`'s doc comment for the full reasoning.
            guard let referenceOffset = PromptCache.cacheAdvanceOffset(cache) else {
                await MLXLanguageModel.removePromptCache(modelID: modelID)
                return
            }
            let cacheAdvance = referenceOffset - slot.promptTokens.count
            let reconciliation = PromptCache.reconcileCacheAdvance(
                observedTokenCount: generatedTokenIDs.count, cacheAdvance: cacheAdvance)
            // The token sequence to key the stored entry on. Attention stacks
            // trim the cache's one-token EOS overhang and store the observed
            // sequence; hybrid Mamba/attention stacks can't trim, so when the
            // extra fed token's identity is known they store the EXTENDED
            // sequence (which matches the cache's real offset exactly) instead
            // of dropping the round. See `PromptCache.planCacheStore`.
            let tokensToStore: [Int]
            switch PromptCache.planCacheStore(
                reconciliation: reconciliation,
                cacheIsTrimmable: canTrimPromptCache(cache),
                fedStopToken: stopTokenFedToCache)
            {
            case .store:
                tokensToStore = slot.promptTokens + generatedTokenIDs
            case .trimThenStore:
                // `canTrimPromptCache` being true doesn't guarantee the trim
                // lands where expected (e.g. a `RotatingKVCache` shortfall), so
                // `trimCacheIfValid` re-verifies and we drop on failure.
                guard
                    trimCacheIfValid(
                        cache, slot: slot, cacheAdvance: cacheAdvance,
                        generatedTokenCount: generatedTokenIDs.count)
                else {
                    await MLXLanguageModel.removePromptCache(modelID: modelID)
                    return
                }
                tokensToStore = slot.promptTokens + generatedTokenIDs
            case .storeExtended(let fedStopToken):
                tokensToStore = slot.promptTokens + generatedTokenIDs + [fedStopToken]
            case .drop:
                await MLXLanguageModel.removePromptCache(modelID: modelID)
                return
            }
            await MLXLanguageModel.storePromptCache(
                modelID: modelID, tokens: tokensToStore, cache: cache)
        }

        /// Variant of `commitPromptCache` for `runUnconstrained`'s
        /// `generate(input:cache:parameters:context:...)`, the one
        /// remaining generation path that only yields decoded text, not
        /// raw token IDs -- `generate()`'s `Generation` stream
        /// (`.chunk`/`.info`/`.toolCall`) never exposes the token IDs
        /// `TextToolTokenLoopHandler` observes internally, and switching
        /// to a raw-token stream there would drop `effectiveStopStrings`
        /// support (a real behavior regression), so this path reconstructs
        /// the generated token IDs by re-encoding `emittedText` instead.
        /// Every other generation path (`runReasoning`,
        /// `runToolCallReasoningPhase`, `runGuidedGenerationLoop`) threads
        /// real token IDs through directly and never reaches this overload.
        ///
        /// Trusts the reconstruction only when its count matches `cache`'s
        /// real `offset` advance -- the same ground-truth check as the
        /// exact-IDs overload, applied to a best-effort reconstruction
        /// instead of a directly-observed token stream. Also accepts a
        /// count exactly one *more* than the cache's offset advance,
        /// dropping the reconstruction's trailing token before storing:
        /// `TokenIterator`'s next()-ahead prefetch design discards the
        /// terminal EOS/stop token that ends a round without ever handing
        /// it to `generate()`'s stream, even though that token's forward
        /// pass already advanced the cache (see
        /// `PromptCache.reconcileGeneratedTokens`'s doc) -- so `cache`'s
        /// real offset legitimately lands one token short of
        /// `emittedText`'s full re-encoding on that (common, successful)
        /// natural-stop exit path. A wrong trailing token value still
        /// gets caught by the next round's real re-tokenization comparison
        /// in `PromptCache.decide` and safely falls back to trim/rebuild
        /// rather than reusing it.
        ///
        /// Also accepts a count exactly one *fewer* than the cache's
        /// offset advance: the mirror-image case where the model's actual
        /// final generated token IS the EOS/stop token, which advances
        /// `cache`'s offset but decodes to no text at all, so re-encoding
        /// `emittedText` recovers one fewer token than the cache's real
        /// advance. `reconcileGeneratedTokens` trusts that shorter
        /// reconstruction as-is (no fabricated token ID for the missing
        /// EOS position), and this composes with the `generatedTokenIDs`
        /// overload above: its own `PromptCache.reconcileCacheAdvance`
        /// call recognizes the resulting one-token gap as
        /// `.trimCacheByOne` and trims the cache back into sync via
        /// `trimAndVerify` before storing -- no additional handling is
        /// needed here.
        ///
        /// - Parameters:
        ///   - slot: The slot this round generated with.
        ///   - emittedText: The full decoded text this round generated.
        ///   - tokenizer: Used to re-encode `emittedText` for the fidelity check.
        private static func commitPromptCache(
            modelID: String, slot: PromptCacheSlot, emittedText: String, tokenizer: any Tokenizer,
            stopTokenFedToCache: Int? = nil
        ) async {
            guard let cache = slot.cache else { return }
            // See the `generatedTokenIDs` overload's identical guard for
            // why this must use `PromptCache.cacheAdvanceOffset` rather than
            // `cache.first?.offset` directly.
            guard let finalOffset = PromptCache.cacheAdvanceOffset(cache) else {
                await MLXLanguageModel.removePromptCache(modelID: modelID)
                return
            }
            let actualGeneratedCount = finalOffset - slot.promptTokens.count
            guard actualGeneratedCount > 0 else { return }
            let reencoded = tokenizer.encode(text: emittedText, addSpecialTokens: false)
            guard
                let trustedTokens = PromptCache.reconcileGeneratedTokens(
                    reencoded: reencoded, actualGeneratedCount: actualGeneratedCount)
            else {
                await MLXLanguageModel.removePromptCache(modelID: modelID)
                return
            }
            await commitPromptCache(
                modelID: modelID, slot: slot, generatedTokenIDs: trustedTokens,
                stopTokenFedToCache: stopTokenFedToCache)
        }

        /// Runs think-then-call Phase 1: unconstrained reasoning until
        /// `</think>`, whose token IDs prefill the constrained Phase 2.
        /// Sends the incomplete-output signal itself when cut off, since
        /// that path also needs its caller to skip its own tail GPU sync
        /// and return immediately (Phase 1 already synchronized on its way
        /// out).
        ///
        /// - Parameters:
        ///   - reasoningConfig: The think-then-call reasoning config (already gated by the caller).
        ///   - toolAwareInput: The tool-aware-template-rendered prompt.
        ///   - primedInside: Whether the rendered prompt already ends inside
        ///     an open reasoning span (computed once by the caller, which
        ///     also gates on it via `thinkThenCallPhase1Engages`).
        ///   - maxTokens: The resolved token budget for this request.
        ///   - request: The generation request (temperature/reasoning options).
        ///   - requestedSamplingMode: The caller's sampling mode override, if any.
        ///   - stableBoundary: The messages/tools for the hybrid
        ///     stable-boundary computation (see `makePromptCacheSlot`).
        ///   - reasoningEntryID: The entry to stream reasoning segments into.
        ///   - entryID: The response entry (for the incomplete-output signal).
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever `runToolCallReasoningPhase` throws.
        /// - Returns: The reasoning token IDs (empty if Phase 1 wasn't
        ///   entered) and whether Phase 1 was cut off before closing.
        private func executeThinkThenCallPhase1(
            reasoningConfig: ReasoningConfig,
            toolAwareInput: LMInput,
            primedInside: Bool,
            maxTokens: Int,
            request: LanguageModelExecutorGenerationRequest,
            requestedSamplingMode: MLXSamplingMode?,
            stableBoundary: StableBoundaryRender?,
            reasoningEntryID: String,
            entryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> (tokenIDs: [Int], cutOff: Bool) {
            let phase1 = try await runToolCallReasoningPhase(
                input: toolAwareInput, reasoningConfig: reasoningConfig,
                primedInside: primedInside, maxTokens: maxTokens,
                requestedTemperature: request.generationOptions.temperature,
                samplingMode: requestedSamplingMode,
                stableBoundary: stableBoundary,
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
        ///   - stableBoundary: The messages/tools for the hybrid
        ///     stable-boundary computation (see `makePromptCacheSlot`).
        /// - Returns: The buffered output text, the generated token count
        ///   (nil if generation threw before completing), how many prompt
        ///   tokens were served from a `PromptCache` slot this round, and
        ///   whether output was incomplete.
        /// - Throws: Whatever `runGuidedGenerationLoop` throws (see its doc
        ///   for how `GuidedGenerationError.incompleteOutput` is handled
        ///   instead of propagating).
        private func executeToolCallingPhase2(
            phase2Input: LMInput,
            phase2MaxTokens: Int,
            context: ModelContext,
            setup: ConstraintSetup,
            stableBoundary: StableBoundaryRender?
        ) async throws -> (
            outputBuffer: String, generatedTokenCount: Int, cachedTokenCount: Int,
            incomplete: Bool
        ) {
            var outputBuffer = ""
            let result = try await runGuidedGenerationLoop(
                input: phase2Input, context: context, setup: setup, maxTokens: phase2MaxTokens,
                stableBoundary: stableBoundary
            ) { text in
                outputBuffer += text
                return !Task.isCancelled
            }
            return (outputBuffer, result.generatedTokenCount, result.cachedTokenCount, result.incomplete)
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
        ///   - stableBoundary: The messages/tools for the hybrid
        ///     stable-boundary computation (see `makePromptCacheSlot`).
        ///   - onText: Called with each generated text delta; return `false`
        ///     to stop generation early.
        /// - Returns: The generated token count (nil if generation threw
        ///   before completing), how many prompt tokens were served from a
        ///   `PromptCache` slot this round, and whether output was
        ///   incomplete.
        /// - Throws: Whatever `GuidedGenerationLoop.run` throws, except
        ///   `GuidedGenerationError.incompleteOutput`, which is caught here
        ///   and reported via the returned `incomplete` flag instead of
        ///   propagating.
        /// Resolves the `PromptCacheSlot` to commit after a successful
        /// `GuidedGenerationLoop.run` call: adopts `resultCache` -- a
        /// possibly quantization-updated replacement (see
        /// `GuidedGenerationLoop.RunResult.cache`'s doc) -- only when `slot`
        /// itself started with a real cache.
        ///
        /// `run` ALWAYS returns a concrete, non-nil `[KVCache]` (building one
        /// internally via `model.newCache` when passed `nil`), so
        /// unconditionally adopting `resultCache` would turn a multimodal
        /// round's intentional `nil` (see `isTextOnly`/`makePromptCacheSlot`)
        /// into a real cache here -- silently defeating the guard that keeps
        /// image/video/audio-conditioned KV state out of `PromptCache` (it
        /// would get stored keyed only by `slot.promptTokens`, i.e. the TEXT
        /// tokens, with no record that the underlying KV state was also
        /// conditioned on non-text input a later text-only round can't
        /// reproduce).
        ///
        /// - Parameters:
        ///   - slot: The slot this round generated with.
        ///   - resultCache: The cache `GuidedGenerationLoop.run`'s
        ///     `RunResult` returned.
        /// - Returns: `slot` unchanged when `slot.cache` was `nil`;
        ///   otherwise `slot` with its cache replaced by `resultCache`.
        private static func slotAdoptingResultCache(
            _ slot: PromptCacheSlot, resultCache: [KVCache]
        ) -> PromptCacheSlot {
            guard slot.cache != nil else { return slot }
            return PromptCacheSlot(
                cache: resultCache, feedInput: slot.feedInput, promptTokens: slot.promptTokens,
                splitPrefillTokenCount: slot.splitPrefillTokenCount)
        }

        private func runGuidedGenerationLoop(
            input: LMInput,
            context: ModelContext,
            setup: ConstraintSetup,
            maxTokens: Int,
            stableBoundary: StableBoundaryRender?,
            onText: @escaping (String) -> Bool
        ) async throws -> (generatedTokenCount: Int, cachedTokenCount: Int, incomplete: Bool) {
            var incomplete = false
            let generatedTokenCount: Int
            // Derive reserves from this call's own `maxTokens`, not
            // `setup.maxTokens` -- Phase 2 of think-then-call runs under a
            // reduced budget, and a reserve sized to the original,
            // pre-Phase-1 budget could consume the entire (or more than
            // the entire) remaining budget.
            let (completionReserve, hardReserve) = setup.reserves(forMaxTokens: maxTokens)

            // `parameters: nil` matches GuidedGenerationLoop.run's own
            // `model.newCache(parameters: nil)` -- a rebuilt cache here has
            // the same "flavor" it would have gotten unassisted.
            let slot = try await makePromptCacheSlot(
                input: input, context: context, parameters: nil, stableBoundary: stableBoundary)
            // The slot to commit: `slot` itself unless `run` returns
            // (successfully) a replacement cache -- see
            // `slotAdoptingResultCache`'s doc. Stays at `slot` if `run`
            // throws before returning a result; `kvBits` is never set on
            // this path today so that's a no-op in practice (see
            // `RunResult`'s throwing-exit doc).
            var finalSlot = slot
            // Real committed token IDs, reported as `run` feeds them
            // through the model -- never a re-encoding of `onText`'s
            // decoded text (see `PromptCache.reconcileCacheAdvance`).
            var generatedTokenIDs: [Int] = []
            do {
                let result = try GuidedGenerationLoop.run(
                    input: slot.feedInput,
                    context: context,
                    constraint: setup.constraint,
                    maxTokens: maxTokens,
                    vocabSize: Int(setup.xgTokenizer.vocabSize),
                    completionReserve: completionReserve,
                    hardReserve: hardReserve,
                    closingBias: setup.closingBias,
                    whitespaceBias: setup.whitespaceBias,
                    whitespaceTokenIDs: setup.whitespaceTokenIDs,
                    cache: slot.cache,
                    onTokenCommitted: { generatedTokenIDs.append($0) }
                ) { text in
                    onText(text)
                }
                generatedTokenCount = result.tokenCount
                finalSlot = Self.slotAdoptingResultCache(slot, resultCache: result.cache)
            } catch GuidedGenerationError.incompleteOutput {
                // Grammar exhausted maxTokens before reaching a stop state.
                // Deltas already emitted (or buffered) are best-effort
                // output; `generatedTokenIDs` still holds every token
                // `onTokenCommitted` reported before the throw, so a
                // partial cache commit below is still trustworthy. Report
                // that same count for usage too -- `GuidedGenerationLoop.run`
                // doesn't return a `RunResult` on this throw path, but the
                // tally accumulated via `onTokenCommitted` is exactly what
                // `RunResult.tokenCount` would have been, matching how
                // `runReasoning` already reports usage on incomplete output.
                generatedTokenCount = generatedTokenIDs.count
                incomplete = true
            }
            await Self.commitPromptCache(
                modelID: modelID, slot: finalSlot, generatedTokenIDs: generatedTokenIDs)
            return (generatedTokenCount, slot.cachedTokenCount, incomplete)
        }

        /// Whether think-then-call should even be considered for this round
        /// -- the multi-condition pre-render check `makeThinkThenCallConfig`
        /// guards its lookup on, named so the AND of these three
        /// independent conditions (declared reasoning capability, a
        /// resolved reasoning config to run against, and reasoning not
        /// explicitly turned off for this request) reads as one semantic
        /// precondition rather than an anonymous boolean chain. This
        /// predicate says nothing about WHICH strategy family qualifies
        /// once it's `true` -- that dispatch (`.templateFlag`/`.alwaysOn`
        /// vs. `.none`) happens entirely in `makeThinkThenCallConfig`'s own
        /// `switch`, below.
        ///
        /// - Parameters:
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        ///   - reasoningLevel: The caller's requested reasoning level, if any.
        /// - Returns: `true` when a reasoning config exists to gate
        ///   think-then-call's strategy dispatch on.
        private static func shouldRunThinkThenCall(
            declaresReasoning: Bool,
            resolved: ModelConfiguration,
            reasoningLevel: ContextOptions.ReasoningLevel?
        ) -> Bool {
            declaresReasoning && resolved.reasoningConfig != nil
                && thinkingEnabled(for: reasoningLevel) != false
        }

        /// Derives the think-then-call reasoning config for the tool-calling
        /// path — the static, pre-render gate. Two strategy families qualify:
        ///
        /// - `.templateFlag`/`.templateStringFlag` (Qwen3/QwQ, MiniMax-M3):
        ///   their template both renders the tool block AND honors the
        ///   thinking kwarg (boolean `enable_thinking` or M3's string-valued
        ///   `thinking_mode`).
        /// - `.alwaysOn` (MiniMax-M2, R1-style): whether Phase 1 actually
        ///   runs is decided post-render by `thinkThenCallPhase1Engages`,
        ///   which requires the rendered prompt to end inside an open
        ///   reasoning span (MiniMax-M2's template pre-opens `<think>\n`).
        ///   A non-primed alwaysOn prompt (e.g. a tool-blind R1-style
        ///   template that ignores `tools:` and doesn't prefill `<think>`)
        ///   falls through to the single-phase path unchanged.
        ///
        /// Thinking-disabled requests stay single-phase for both.
        ///
        /// - Parameters:
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        ///   - reasoningLevel: The caller's requested reasoning level, if any.
        /// - Returns: The reasoning config to run think-then-call with, or
        ///   `nil` when the model/request doesn't qualify.
        static func makeThinkThenCallConfig(
            declaresReasoning: Bool,
            resolved: ModelConfiguration,
            reasoningLevel: ContextOptions.ReasoningLevel?
        ) -> ReasoningConfig? {
            guard shouldRunThinkThenCall(
                declaresReasoning: declaresReasoning, resolved: resolved,
                reasoningLevel: reasoningLevel),
                let reasoningConfig = resolved.reasoningConfig
            else { return nil }
            switch reasoningConfig.promptStrategy {
            case .templateFlag, .templateStringFlag, .alwaysOn:
                return reasoningConfig
            case .none:
                // No prompt-level thinking protocol to run Phase 1 against.
                return nil
            }
        }

        /// Whether think-then-call Phase 1 (unconstrained reasoning until the
        /// closing delimiter) should run for this round's rendered prompt —
        /// the post-render gate, complementing `makeThinkThenCallConfig`.
        ///
        /// - `.templateFlag`/`.templateStringFlag`: always engages — the
        ///   kwarg primed the template, and the model emits its own `<think>`
        ///   opener (Qwen3-style) or the template pre-opens it when forced on
        ///   (MiniMax-M3's `thinking_mode: "enabled"`), so prompt priming is
        ///   irrelevant to the decision either way.
        /// - `.alwaysOn`: engages only when the rendered prompt ends inside
        ///   an open reasoning span (MiniMax-M2's template pre-opens
        ///   `<think>\n`, so its constrained tool round would otherwise mix
        ///   free-form thinking into the envelope buffer). A non-primed
        ///   prompt stays single-phase: its model may never emit `</think>`,
        ///   and Phase 1 would burn the whole budget waiting for it.
        /// - `.none`: never reaches here (`makeThinkThenCallConfig` filters
        ///   it), but answers `false` for completeness.
        ///
        /// - Parameters:
        ///   - reasoningConfig: The config from `makeThinkThenCallConfig`.
        ///   - primedInside: Whether the rendered prompt ends inside an open
        ///     reasoning span (see `reasoningPrimedInside`).
        /// - Returns: `true` when Phase 1 should run before the constrained
        ///   Phase 2.
        static func thinkThenCallPhase1Engages(
            reasoningConfig: ReasoningConfig,
            primedInside: Bool
        ) -> Bool {
            switch reasoningConfig.promptStrategy {
            case .templateFlag, .templateStringFlag:
                return true
            case .alwaysOn:
                return primedInside
            case .none:
                return false
            }
        }

        /// Resolves Phase 2's token budget for think-then-call tool-calling.
        ///
        /// - Parameters:
        ///   - reasoningTokenIDs: Phase 1's accumulated reasoning token IDs
        ///     (empty on the single-phase path).
        ///   - setup: The constraint/bias/reserve setup from `prepareConstraintSetup`.
        /// - Returns: `setup.maxTokens` unchanged when Phase 1 contributed no
        ///   reasoning tokens; otherwise the remaining budget after Phase 1,
        ///   floored at the completion reserve so the envelope always has
        ///   room to close the tool call (matches the unconstrained path's
        ///   shared-budget behavior).
        private static func resolvePhase2MaxTokens(
            reasoningTokenIDs: [Int], setup: ConstraintSetup
        ) -> Int {
            guard !reasoningTokenIDs.isEmpty else { return setup.maxTokens }
            return Swift.max(
                setup.maxTokens - reasoningTokenIDs.count,
                setup.reserves(forMaxTokens: setup.maxTokens).completionReserve)
        }

        /// What `runToolCalling` prepares before any generation work runs:
        /// the tool-aware prompt, the constraint setup framing the
        /// tool-call envelope, the think-then-call gate, and the
        /// stable-boundary baseline shared by both phases. See
        /// `prepareToolCallingContext`.
        private struct ToolCallingContext {
            /// The think-then-call reasoning config, when the resolved
            /// family qualifies; `nil` keeps the round single-phase.
            let thinkThenCallConfig: ReasoningConfig?
            /// The tool-aware-template-rendered prompt.
            let toolAwareInput: LMInput
            /// The constraint/bias/reserve setup for the tool-call envelope.
            let setup: ConstraintSetup
            /// The stable-boundary baseline both phases render against.
            let stableBoundary: StableBoundaryRender
        }

        /// Prepares everything `runToolCalling`'s generation phases need:
        /// builds the tool list (with the synthetic final-answer tool),
        /// renders the tool-aware prompt through the model's own
        /// `UserInputProcessor` (validated against the context window),
        /// compiles the tool-calling grammar into a `ConstraintSetup`, and
        /// assembles the stable-boundary baseline from the same
        /// messages/tools/history context the prompt was rendered from.
        ///
        /// - Parameters:
        ///   - request: The generation request; supplies the enabled tools,
        ///     developer schema, and reasoning options.
        ///   - messages: The rendered chat messages for this round.
        ///   - modelID: The model identifier for constraint/tokenizer caches.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - declaresReasoning: Whether `.reasoning` was declared at init.
        ///   - resolved: The resolved model configuration.
        ///   - contextLength: The model's context window length, if known.
        ///   - context: The loaded model context.
        /// - Returns: The prepared `ToolCallingContext`.
        /// - Throws: `LanguageModelError.contextSizeExceeded` when the
        ///   tool-aware prompt exceeds `contextLength`, or whatever the
        ///   grammar/tokenizer/render calls throw.
        private func prepareToolCallingContext(
            request: LanguageModelExecutorGenerationRequest,
            messages: [Chat.Message],
            modelID: String,
            requestedMaxTokens: Int?,
            declaresReasoning: Bool,
            resolved: ModelConfiguration,
            contextLength: Int?,
            context: ModelContext
        ) async throws -> ToolCallingContext {
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

            // Think-then-call qualifies two strategy families: the
            // `enable_thinking` templates (Qwen3/QwQ, which render the
            // tool block AND honor the kwarg) and `.alwaysOn` templates
            // (MiniMax-M2), the latter confirmed post-render by the
            // primed-inside gate at the Phase 1 call site — a
            // non-primed alwaysOn prompt (tool-blind R1-style) stays
            // single-phase unchanged. Thinking-disabled requests stay
            // single-phase too.
            let thinkThenCallConfig = Self.makeThinkThenCallConfig(
                declaresReasoning: declaresReasoning,
                resolved: resolved,
                reasoningLevel: request.contextOptions.reasoningLevel)
            // Thread `enable_thinking` through the tool-aware template
            // (3-arg form) so the prompt is both tool-aware and
            // thinking-primed; nil on the single-phase path. History
            // preservation (`preserve_thinking`) is independent of the
            // think-then-call gate: it comes from the resolved family
            // config, so a history-preserving model's tool rounds render
            // past turns consistently in both phases.
            let historyContext = resolved.reasoningConfig?.historyPreservationContext
            let strategyContext = try thinkThenCallConfig.flatMap {
                try $0.promptStrategy.additionalContext(
                    forThinkingEnabled: Self.thinkingEnabled(
                        for: request.contextOptions.reasoningLevel))
            }
            let reasoningContext = Self.mergedAdditionalContext(strategyContext, historyContext)
            // Route through the model's own `UserInputProcessor` -- the
            // same one the eager (non-tool) path uses in
            // `prepareRespondSetup` -- instead of hand-rolling
            // `DefaultMessageGenerator()` + a raw `applyChatTemplate`
            // call. A hardcoded `DefaultMessageGenerator()` silently
            // diverges from what the model actually needs on two axes:
            // some text models render a system-role-free template via
            // `NoSystemMessageGenerator` (see `LlamaModel.messageGenerator`),
            // and VLMs render image/video placeholders via their own
            // generator (see `Qwen2VLMessageGenerator.generate(message:)`)
            // *and* need their pixel data processed into
            // `LMInput.image`/`.video`/`.audio` -- work `applyChatTemplate`
            // alone never does. Passing `tools:`/`additionalContext:`
            // through `UserInput` still drives the processor down the
            // exact tool-aware `applyChatTemplate` branch this used to
            // call directly.
            let toolAwareUserInput = UserInput(
                chat: messages, tools: toolSpecs, additionalContext: reasoningContext)
            let toolAwareInput = try await Self.preparedInputMappingImageFailures(
                processor: context.processor, input: toolAwareUserInput, messages: messages,
                transcriptEntries: request.transcript)
            try Self.validateContextSize(
                tokenCount: toolAwareInput.text.tokens.size, contextLength: contextLength)

            // Frame the constrained envelope in the Qwen tool-call
            // wrapper (with a bare-JSON escape hatch). The inferred
            // `ToolCallFormat` is passed through, but every format
            // currently resolves to the Qwen framing: the envelope
            // content is Qwen-shaped JSON, and constraining a family
            // to its native wrapper around foreign JSON content sent
            // GLM-4.7-Flash off-distribution (leaks + runaways; see
            // `SchemaConverter.ToolCallStructuralTag`).
            let toolCallingGrammar =
                try SchemaConverter.encodeToolCallingGrammar(
                    tools: allTools,
                    format: context.configuration.toolCallFormat
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

            // The tool path's stable-boundary render carries the SAME
            // messages, tool specs, and history context the tool-aware
            // prompt was rendered from -- future rounds re-render past turns
            // with the tool block (and any preserved thinking) still
            // present, so both belong in the baseline.
            return ToolCallingContext(
                thinkThenCallConfig: thinkThenCallConfig,
                toolAwareInput: toolAwareInput,
                setup: setup,
                stableBoundary: StableBoundaryRender(
                    messages: messages, tools: toolSpecs, historyContext: historyContext))
        }

        /// The two-phase result `executeThinkThenCallPhases` reports when
        /// Phase 1 wasn't cut off: Phase 2's buffered envelope output and
        /// counts, plus Phase 1's reasoning-token count.
        private struct ThinkThenCallOutcome {
            /// The buffered Phase 2 output (a JSON tool-call envelope).
            let outputBuffer: String
            /// Phase 2's generated token count.
            let generatedTokenCount: Int
            /// How many prompt tokens were served from a `PromptCache`
            /// slot this round.
            let cachedTokenCount: Int
            /// Whether Phase 2's output was incomplete.
            let incomplete: Bool
            /// Phase 1's reasoning token count (zero on the single-phase path).
            let reasoningTokenCount: Int
        }

        /// Runs `runToolCalling`'s generation phases: think-then-call
        /// Phase 1 when the prepared context qualifies (unconstrained
        /// reasoning until `</think>`, retaining the token IDs), then
        /// constrained Phase 2 continuing from that reasoning under the
        /// remaining budget.
        ///
        /// - Parameters:
        ///   - toolCalling: The prepared prompt/constraint/boundary bundle
        ///     from `prepareToolCallingContext`.
        ///   - request: The generation request (temperature/reasoning options).
        ///   - requestedSamplingMode: The caller's sampling mode override, if any.
        ///   - contextLength: The model's context window length, if known;
        ///     Phase 2's reasoning-extended prompt is re-validated against it.
        ///   - entryID: The response entry (for the incomplete-output signal).
        ///   - reasoningEntryID: The entry to stream Phase 1 reasoning into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Returns: The two-phase outcome, or `nil` when Phase 1 was cut
        ///   off before `</think>` closed -- Phase 1 already signalled
        ///   incomplete output and synchronized the GPU on its way out, so
        ///   the caller must return immediately without its own tail sync.
        /// - Throws: `LanguageModelError.contextSizeExceeded` when Phase
        ///   1's reasoning pushes the prompt past `contextLength`, or
        ///   whatever the phase executors throw.
        private func executeThinkThenCallPhases(
            toolCalling: ToolCallingContext,
            request: LanguageModelExecutorGenerationRequest,
            requestedSamplingMode: MLXSamplingMode?,
            contextLength: Int?,
            entryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> ThinkThenCallOutcome? {
            // PHASE 1 (think-then-call): reason unconstrained until
            // `</think>`, retaining the token IDs to prefill into the
            // constrained phase below. Empty on the single-phase path.
            // `.alwaysOn` configs additionally require the rendered prompt
            // to end inside an open reasoning span (see
            // `thinkThenCallPhase1Engages`) — MiniMax-M2's template
            // pre-opens `<think>\n`; a non-primed alwaysOn prompt stays
            // single-phase.
            var reasoningTokenIDs: [Int] = []
            if let reasoningConfig = toolCalling.thinkThenCallConfig {
                let primedInside = Self.reasoningPrimedInside(
                    input: toolCalling.toolAwareInput, reasoningConfig: reasoningConfig,
                    tokenizer: context.tokenizer)
                if Self.thinkThenCallPhase1Engages(
                    reasoningConfig: reasoningConfig, primedInside: primedInside)
                {
                    let phase1 = try await executeThinkThenCallPhase1(
                        reasoningConfig: reasoningConfig,
                        toolAwareInput: toolCalling.toolAwareInput,
                        primedInside: primedInside,
                        maxTokens: toolCalling.setup.maxTokens,
                        request: request, requestedSamplingMode: requestedSamplingMode,
                        stableBoundary: toolCalling.stableBoundary,
                        reasoningEntryID: reasoningEntryID, entryID: entryID,
                        context: context, channel: channel)
                    reasoningTokenIDs = phase1.tokenIDs
                    guard !phase1.cutOff else { return nil }
                }
            }

            // Phase 2 continues from the model's completed reasoning;
            // carry the raw IDs (no decode/re-encode) so the grammar
            // starts from the exact post-`</think>` state. Also carry
            // forward `toolAwareInput`'s `.image`/`.video`/`.audio` --
            // rebuilding from a bare token array here would silently
            // drop any media the prompt carried, exactly the bug this
            // path used to have unconditionally.
            let phase2Input =
                reasoningTokenIDs.isEmpty
                ? toolCalling.toolAwareInput
                : LMInput(
                    text: .init(
                        tokens: MLXArray(
                            toolCalling.toolAwareInput.text.tokens.asArray(Int.self)
                                + reasoningTokenIDs)),
                    image: toolCalling.toolAwareInput.image,
                    video: toolCalling.toolAwareInput.video,
                    audio: toolCalling.toolAwareInput.audio)
            // Phase 1's reasoning tokens can push the prompt over the
            // context window even when the original `toolAwareInput`
            // (validated at render time) was within it -- re-validate the
            // actual Phase 2 input, not just its no-reasoning-tokens
            // starting point.
            try Self.validateContextSize(
                tokenCount: phase2Input.text.tokens.size, contextLength: contextLength)
            // Shared budget (match the unconstrained path): the
            // envelope continues under the remaining budget, floored
            // at the completion reserve so it always has room to close
            // the tool call.
            let phase2MaxTokens = Self.resolvePhase2MaxTokens(
                reasoningTokenIDs: reasoningTokenIDs, setup: toolCalling.setup)

            let phase2 = try await executeToolCallingPhase2(
                phase2Input: phase2Input, phase2MaxTokens: phase2MaxTokens,
                context: context, setup: toolCalling.setup,
                stableBoundary: toolCalling.stableBoundary)
            return ThinkThenCallOutcome(
                outputBuffer: phase2.outputBuffer,
                generatedTokenCount: phase2.generatedTokenCount,
                cachedTokenCount: phase2.cachedTokenCount,
                incomplete: phase2.incomplete,
                reasoningTokenCount: reasoningTokenIDs.count)
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
        ///   - contextLength: The model's context window length, if known;
        ///     validated against the re-tokenized tool-aware prompt before
        ///     any grammar/generation work runs.
        ///   - entryID: The response entry to stream output into.
        ///   - toolCallsEntryID: The entry to stream tool-call events into.
        ///   - reasoningEntryID: The entry to stream think-then-call reasoning into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: `LanguageModelError.contextSizeExceeded` when the
        ///   tool-aware prompt exceeds `contextLength`, or whatever the
        ///   grammar/tokenizer/generation calls throw.
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
            contextLength: Int?,
            entryID: String,
            toolCallsEntryID: String,
            reasoningEntryID: String,
            context: ModelContext,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> Bool {
            let toolCalling = try await prepareToolCallingContext(
                request: request, messages: messages, modelID: modelID,
                requestedMaxTokens: requestedMaxTokens,
                declaresReasoning: declaresReasoning, resolved: resolved,
                contextLength: contextLength, context: context)

            guard
                let outcome = try await executeThinkThenCallPhases(
                    toolCalling: toolCalling, request: request,
                    requestedSamplingMode: requestedSamplingMode,
                    contextLength: contextLength, entryID: entryID,
                    reasoningEntryID: reasoningEntryID, context: context,
                    channel: channel)
            else {
                // Phase 1 was cut off before `</think>` closed; it already
                // signalled incomplete output and synchronized the GPU on
                // its way out -- the caller must skip its own tail sync.
                return false
            }

            try await emitToolCallingEvent(
                outputBuffer: outcome.outputBuffer,
                userResponseSchema: request.schema,
                entryID: entryID,
                toolCallsEntryID: toolCallsEntryID,
                channel: channel
            )

            // Output total spans both phases (reasoning + envelope).
            await Self.sendUsageUpdate(
                entryID: entryID,
                promptTokenCount: toolCalling.toolAwareInput.text.tokens.size,
                cachedTokenCount: outcome.cachedTokenCount,
                outputTokenCount: outcome.generatedTokenCount + outcome.reasoningTokenCount,
                reasoningTokenCount: outcome.reasoningTokenCount,
                channel: channel)

            if outcome.incomplete {
                await Self.sendIncompleteOutputMetadata(entryID: entryID, channel: channel)
            }

            return true
        }

        /// Whether the guided-generation prompt-assembly seam
        /// (``guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:)``)
        /// is allowed to add the adapter's own schema rendering to the
        /// prompt, per `ContextOptions.includeSchemaInPrompt`. This adapter
        /// has no such rendering to add today (see that function's doc
        /// comment), so the two branches currently behave identically --
        /// but the gate itself is real and correctly derived, ready for
        /// whichever branch a future schema-in-prompt renderer lands in.
        ///
        /// **SDK ground truth** (macOS 27 SDK's `FoundationModels.swiftinterface`,
        /// which this project's own conventions treat as authoritative over
        /// any other source): `ContextOptions.includeSchemaInPrompt` is a
        /// plain `Bool?` defaulting to `nil` in `ContextOptions`'s own
        /// memberwise initializer -- but every schema-taking
        /// `LanguageModelSession.respond`/`streamResponse` overload
        /// (`respond(to:schema:...)`, `respond(generating:...)`, etc.)
        /// instead defaults its own `contextOptions:` parameter to
        /// `ContextOptions(includeSchemaInPrompt: true)`. That asymmetry is
        /// the field's real contract: when a developer goes through the
        /// framework's own schema-based convenience API without overriding
        /// `contextOptions`, the framework's default is `true` -- it already
        /// assumes/arranges for the schema to be present in the prompt, so a
        /// third-party executor must not add a second copy. `nil` therefore
        /// only reaches an executor when a caller has deliberately stepped
        /// off that convenience default (e.g. constructing `ContextOptions`
        /// or a `Transcript.Prompt` directly) -- an explicit, lower-level
        /// path this adapter cannot assume already embedded the schema.
        /// Given that, `nil` is treated identically to `false`: "not
        /// confirmed already in the prompt" -- the conservative reading that
        /// never silently drops the schema from a local (often smaller,
        /// weaker) open-weight model's prompt just because the caller didn't
        /// say either way. `true` is the only value that suppresses the
        /// adapter's own rendering.
        ///
        /// - Parameter includeSchemaInPrompt: `request.contextOptions.includeSchemaInPrompt`.
        /// - Returns: `false` only when `includeSchemaInPrompt == true`; `true` for `false`/`nil`.
        static func shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: Bool?) -> Bool {
            includeSchemaInPrompt != true
        }

        /// The guided-generation prompt-assembly seam: builds the messages
        /// actually rendered into the guided-generation prompt, gated by
        /// `ContextOptions.includeSchemaInPrompt` (see
        /// ``shouldInjectSchemaIntoPrompt(includeSchemaInPrompt:)``).
        ///
        /// **This function has no observable effect today: every branch of
        /// its guard/if-else returns `messages` unchanged.** That is
        /// deliberate scaffolding, not an oversight left uncleaned. Fixing
        /// the schema-duplication concern that motivated this seam turned up
        /// no existing double-injection bug -- schema text was never
        /// rendered into the prompt to begin with (see below) -- so there is
        /// nothing to suppress here today, and no logic was removed to reach
        /// this state.
        ///
        /// This adapter has no schema-in-prompt rendering of its own to add
        /// today: `schemaJSON` only ever feeds xgrammar's constrained-
        /// decoding constraint (`runGuidedGeneration` via
        /// `prepareConstraintSetup`), never prompt text -- the rendered
        /// prompt is 100% whatever `TranscriptConverter.mlxMessages` produced
        /// from `request.transcript`. Both branches below therefore return
        /// `messages` unchanged, preserving that behavior exactly regardless
        /// of `includeSchemaInPrompt`'s value. This function still exists,
        /// and is still wired into `prepareRespondSetup`, as the single seam
        /// any *future* adapter-side schema-in-prompt rendering must route
        /// through: `shouldInjectSchemaIntoPrompt` gates the branch that
        /// would add one, so a `true` value (the app has already put the
        /// schema in the prompt) can never end up with two renderings once
        /// such a feature exists -- it is deliberately guarded from day one
        /// rather than left for a future change to get wrong. When that
        /// feature lands, its schema-in-prompt rendering replaces this
        /// function's `return messages` in the branch gated by
        /// `shouldInjectSchemaIntoPrompt` returning `true` -- the only branch
        /// that should ever inject schema text -- with no restructuring of
        /// this function or its callers required.
        ///
        /// Independent of prompt text either way: the grammar/schema-based
        /// sampling constraint xgrammar compiles from `schemaJSON` in
        /// `runGuidedGeneration` always applies, regardless of
        /// `includeSchemaInPrompt` -- constrained decoding is not something
        /// this flag can or should disable.
        ///
        /// - Parameters:
        ///   - messages: The transcript-rendered messages for this round,
        ///     before any adapter-side schema rendering.
        ///   - schemaJSON: The developer-supplied JSON schema, already
        ///     encoded. Unused today (see above); threaded through so a
        ///     future schema-in-prompt renderer has it available at this
        ///     exact seam without a signature change.
        ///   - includeSchemaInPrompt: `request.contextOptions.includeSchemaInPrompt`.
        /// - Returns: `messages`, unchanged in both branches today -- see
        ///   above for why that is deliberate rather than dead code.
        static func guidedGenerationMessages(
            from messages: [Chat.Message],
            schemaJSON: String,
            includeSchemaInPrompt: Bool?
        ) -> [Chat.Message] {
            guard Self.shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: includeSchemaInPrompt)
            else {
                return messages
            }
            return messages
        }

        /// Guided-generation path: streams text deltas constrained to the
        /// developer's JSON schema as they arrive.
        ///
        /// - Parameters:
        ///   - schemaJSON: The developer-supplied JSON schema.
        ///   - input: The rendered prompt to generate from.
        ///   - stableBoundary: The hybrid stable-boundary baseline built
        ///     alongside `input` from the SAME history-affecting context
        ///     (see `makeGuidedRender` and `makePromptCacheSlot`).
        ///   - modelID: The model identifier for constraint/tokenizer caches.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - entryID: The response entry to stream output into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever the grammar/generation calls throw.
        private func runGuidedGeneration(
            schemaJSON: String,
            input: LMInput,
            stableBoundary: StableBoundaryRender,
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
                    await Self.sendDelta(text, entryID: entryID, channel: channel, isReasoning: false)
                }
            }()

            // `stableBoundary` was built alongside `input` from the SAME
            // history-affecting context (see `makeGuidedRender`), so the
            // baseline render always agrees with the fed prompt -- see
            // `StableBoundaryRender.historyContext`.
            let result = try await runGuidedGenerationLoop(
                input: input, context: context, setup: setup, maxTokens: setup.maxTokens,
                stableBoundary: stableBoundary
            ) { text in
                textContinuation.yield(text)
                return !Task.isCancelled
            }
            let generatedTokenCount = result.generatedTokenCount
            let incomplete = result.incomplete
            textContinuation.finish()
            await forwarder

            await Self.sendUsageUpdate(
                entryID: entryID,
                promptTokenCount: input.text.tokens.size,
                cachedTokenCount: result.cachedTokenCount,
                outputTokenCount: generatedTokenCount,
                reasoningTokenCount: 0,
                channel: channel)

            if incomplete {
                await Self.sendIncompleteOutputMetadata(entryID: entryID, channel: channel)
            }
        }

        /// Processes one `Generation` event from the unconstrained `generate`
        /// stream: accumulates emitted text and forwards a text delta, or
        /// reports the authoritative end-of-generation usage update.
        /// Mirrors `processReasoningToken`'s per-token extraction, but for
        /// `runUnconstrained`'s decoded-chunk stream.
        ///
        /// - Parameters:
        ///   - generation: The generation event to process.
        ///   - emittedText: The full decoded text emitted so far (mutated:
        ///     appended to on `.chunk`).
        ///   - entryID: The response entry to stream output into.
        ///   - slot: This round's prompt-cache slot, for usage reporting.
        ///   - channel: The generation channel to send events on.
        private static func handleGenerationEvent(
            _ generation: Generation,
            emittedText: inout String,
            stopTokenFedToCache: inout Int?,
            entryID: String,
            slot: PromptCacheSlot,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch generation {
            case .chunk(let text):
                emittedText += text
                await Self.sendDelta(text, entryID: entryID, channel: channel, isReasoning: false)
            case .info(let info):
                // Carry the terminal stop token that was fed through the model
                // (advancing the cache) but not decoded into `emittedText`, so
                // `commitPromptCache` can reconcile the one-token offset gap on
                // a hybrid cache instead of dropping the round. See
                // `GenerateCompletionInfo.stopTokenFedToCache`.
                stopTokenFedToCache = info.stopTokenFedToCache
                // MLX-LM emits one .info event at end-of-generation with
                // an authoritative scalar output count
                // (`generationTokenCount` -- see Evaluate.swift's
                // `GenerateCompletionInfo` definition). `info.promptTokenCount`
                // itself only reflects the suffix actually fed
                // (`slot.feedInput`) when this round reused a cached
                // prefix, so report the FULL transcript length from
                // `slot.promptTokens` instead, alongside how much of it
                // was served from cache -- this is what makes the
                // cache's effect observable end-to-end without
                // shrinking the reported prompt size.
                await Self.sendUsageUpdate(
                    entryID: entryID,
                    promptTokenCount: slot.promptTokens.count,
                    cachedTokenCount: slot.cachedTokenCount,
                    outputTokenCount: info.generationTokenCount,
                    reasoningTokenCount: 0,
                    channel: channel)
            case .toolCall:
                break
            }
        }

        /// Unconstrained text generation. Used on the no-tools/no-schema
        /// path when the model has no reasoning config to route through --
        /// or when reasoning is suppressed (the suppressed render is fed as
        /// `input`, and any history context it carried rides along in
        /// `historyContext`).
        ///
        /// - Parameters:
        ///   - input: The rendered prompt to generate from.
        ///   - messages: The chat messages `input` was rendered from, for
        ///     the hybrid stable-boundary computation (see `makePromptCacheSlot`).
        ///   - historyContext: The history-affecting `additionalContext` the
        ///     fed render carried, for the stable-boundary baseline (see
        ///     `StableBoundaryRender.historyContext`); `nil` when it had none.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - requestedTemperature: The caller's temperature override, if any.
        ///   - samplingMode: The caller's sampling mode override, if any.
        ///   - entryID: The response entry to stream output into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: `CancellationError` if the task is cancelled mid-loop.
        private func runUnconstrained(
            input: LMInput,
            messages: [Chat.Message],
            historyContext: [String: any Sendable]?,
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

            let slot = try await makePromptCacheSlot(
                input: input, context: context, parameters: params,
                stableBoundary: StableBoundaryRender(
                    messages: messages, tools: nil, historyContext: historyContext))
            var emittedText = ""
            var stopTokenFedToCache: Int?

            for await generation in try generate(
                input: slot.feedInput,
                cache: slot.cache,
                parameters: params,
                context: context
            ) {
                try Task.checkCancellation()
                await Self.handleGenerationEvent(
                    generation, emittedText: &emittedText,
                    stopTokenFedToCache: &stopTokenFedToCache, entryID: entryID, slot: slot,
                    channel: channel)
            }

            await Self.commitPromptCache(
                modelID: modelID, slot: slot, emittedText: emittedText,
                tokenizer: context.tokenizer, stopTokenFedToCache: stopTokenFedToCache)
        }

        /// Dispatches the no-tools/no-schema path: reasoning routing when a
        /// config resolved, otherwise plain unconstrained text.
        ///
        /// - Parameters:
        ///   - reasoningSetup: The reasoning input/config/primed-state bundle
        ///     from `prepareRespondSetup`, or `nil` to run unconstrained.
        ///   - fallbackInput: The prompt to generate from when `reasoningSetup` is `nil`.
        ///   - messages: The chat messages this round's prompt was rendered
        ///     from, for the hybrid stable-boundary computation (see
        ///     `makePromptCacheSlot`).
        ///   - historyContext: The history-affecting `additionalContext` the
        ///     round's fed render carried, for the unconstrained arm's
        ///     stable-boundary baseline (the reasoning arm derives its own
        ///     from `reasoningSetup.reasoningConfig`); `nil` when none.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - requestedTemperature: The caller's temperature override, if any.
        ///   - samplingMode: The caller's sampling mode override, if any.
        ///   - responseEntryID: The response entry to stream output into.
        ///   - reasoningEntryID: The entry to stream reasoning segments into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: Whatever `runReasoning`/`runUnconstrained` throws.
        private func runTextGeneration(
            reasoningSetup: (input: LMInput, reasoningConfig: ReasoningConfig, primedInside: Bool)?,
            fallbackInput: LMInput,
            messages: [Chat.Message],
            historyContext: [String: any Sendable]?,
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
                    messages: messages,
                    reasoningConfig: reasoning.reasoningConfig,
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
                    messages: messages,
                    historyContext: historyContext,
                    requestedMaxTokens: requestedMaxTokens,
                    requestedTemperature: requestedTemperature,
                    samplingMode: samplingMode,
                    entryID: responseEntryID,
                    context: context,
                    channel: channel)
            }
        }

        /// Processes one generated token during unconstrained reasoning:
        /// attributes it to the reasoning count while inside a thinking span,
        /// feeds it through the detokenizer, and runs any resulting text
        /// chunk through the reasoning-segment scanner.
        ///
        /// - Parameters:
        ///   - token: The freshly generated token ID.
        ///   - emitter: The reasoning-segment scanner (mutated: state advances
        ///     as `</think>` is detected).
        ///   - detokenizer: The streaming detokenizer (mutated: accumulates
        ///     the token, possibly yielding a decoded chunk).
        ///   - reasoningTokenCount: Running count of tokens attributed to
        ///     reasoning (mutated).
        /// - Returns: The segments produced by this token, or `nil` if the
        ///   detokenizer didn't yield a chunk yet (nothing to send this call).
        private static func processReasoningToken(
            _ token: Int,
            emitter: inout ReasoningEventEmitter,
            detokenizer: inout NaiveStreamingDetokenizer,
            reasoningTokenCount: inout Int
        ) -> [ReasoningEventEmitter.Segment]? {
            // One `.token` == one real token, so this is a true token count
            // (not a chunk count). Attribute it to reasoning while the
            // scanner is inside a thinking span. This generously counts the
            // closing-delimiter tokens as reasoning (the emitter only flips
            // state once `process` consumes the full `</think>`); it remains
            // a true token count and the clamp downstream keeps it ≤ total.
            if emitter.isInsideReasoning {
                reasoningTokenCount += 1
            }
            detokenizer.append(token: token)
            guard let chunk = detokenizer.next() else { return nil }
            return emitter.process(chunk)
        }

        /// Handles one `.token` event during unconstrained reasoning
        /// generation: records the token, runs it through the reasoning-
        /// segment scanner via `processReasoningToken`, and forwards any
        /// segments it produced to the channel. Extracted so `runReasoning`'s
        /// generation loop can dispatch the `.token` case with a single call.
        ///
        /// - Parameters:
        ///   - token: The freshly generated token ID.
        ///   - generatedTokenIDs: The running transcript of generated tokens
        ///     (mutated: `token` is appended).
        ///   - emitter: The reasoning-segment scanner (mutated: state advances
        ///     as `</think>` is detected).
        ///   - detokenizer: The streaming detokenizer (mutated: accumulates
        ///     the token, possibly yielding a decoded chunk).
        ///   - reasoningTokenCount: Running count of tokens attributed to
        ///     reasoning (mutated).
        ///   - responseEntryID: The response entry to stream `.response` segments into.
        ///   - reasoningEntryID: The entry to stream `.reasoning` segments into.
        ///   - channel: The generation channel to send events on.
        private static func handleGeneratedToken(
            _ token: Int,
            generatedTokenIDs: inout [Int],
            emitter: inout ReasoningEventEmitter,
            detokenizer: inout NaiveStreamingDetokenizer,
            reasoningTokenCount: inout Int,
            responseEntryID: String,
            reasoningEntryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            generatedTokenIDs.append(token)
            guard
                let segments = processReasoningToken(
                    token, emitter: &emitter, detokenizer: &detokenizer,
                    reasoningTokenCount: &reasoningTokenCount)
            else { return }
            await sendSegments(
                segments, responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID, channel: channel)
        }

        /// Reasoning-aware unconstrained generation.
        ///
        /// Routes thinking delimited by the model's reasoning markers to
        /// `.reasoning` events and the rest to `.response`, using a raw
        /// `generateTokens` stream + a self-owned `NaiveStreamingDetokenizer`
        /// (bypassing `ToolCallProcessor`) so the scanner sees clean detokenized
        /// text — no second fragmentation source — and the loop sees real token
        /// IDs for an accurate reasoning token count.
        ///
        /// - Parameters:
        ///   - input: The rendered, reasoning-primed prompt to generate from.
        ///   - messages: The chat messages `input` was rendered from, for
        ///     the hybrid stable-boundary computation (see `makePromptCacheSlot`).
        ///   - reasoningConfig: The resolved reasoning config for this model.
        ///   - primedInside: Whether the prompt already ends inside an open reasoning span.
        ///   - requestedMaxTokens: The caller's token budget override, if any.
        ///   - requestedTemperature: The caller's temperature override, if any.
        ///   - samplingMode: The caller's sampling mode override, if any.
        ///   - responseEntryID: The response entry to stream `.response` segments into.
        ///   - reasoningEntryID: The entry to stream `.reasoning` segments into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: `CancellationError` if the task is cancelled mid-loop.
        private func runReasoning(
            input: LMInput,
            messages: [Chat.Message],
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
            var generatedTokenIDs: [Int] = []

            // The reasoning-primed render carried the family's history
            // context (merged in `preparedInput`); the stable-boundary
            // baseline must render with the same one.
            let slot = try await makePromptCacheSlot(
                input: input, context: context, parameters: params,
                stableBoundary: StableBoundaryRender(
                    messages: messages, tools: nil,
                    historyContext: reasoningConfig.historyPreservationContext))

            for await generation in try generateTokens(
                input: slot.feedInput, cache: slot.cache, parameters: params, context: context
            ) {
                try Task.checkCancellation()
                switch generation {
                case .token(let token):
                    await Self.handleGeneratedToken(
                        token, generatedTokenIDs: &generatedTokenIDs, emitter: &emitter,
                        detokenizer: &detokenizer, reasoningTokenCount: &reasoningTokenCount,
                        responseEntryID: responseEntryID, reasoningEntryID: reasoningEntryID,
                        channel: channel)
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
                // auto-summing). `info.promptTokenCount` only reflects the
                // suffix actually fed when this round reused a cached
                // prefix; report the full transcript length from
                // `slot.promptTokens` plus how much was served from cache.
                await Self.sendUsageUpdate(
                    entryID: responseEntryID,
                    promptTokenCount: slot.promptTokens.count,
                    cachedTokenCount: slot.cachedTokenCount,
                    outputTokenCount: info.generationTokenCount,
                    reasoningTokenCount: reasoningTokenCount,
                    channel: channel)
            }

            await Self.commitPromptCache(
                modelID: modelID, slot: slot, generatedTokenIDs: generatedTokenIDs,
                stopTokenFedToCache: completionInfo?.stopTokenFedToCache)
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
                    await sendDelta(text, entryID: reasoningEntryID, channel: channel, isReasoning: true)
                case .response(let text):
                    await sendDelta(text, entryID: responseEntryID, channel: channel, isReasoning: false)
                }
            }
        }

        /// Prepares an `LMInput` for the unconstrained reasoning path with
        /// thinking explicitly on, off, or unspecified. Maps the package-
        /// internal `cannotDisableReasoning` to the framework's
        /// `unsupportedCapability` so always-on models surface a typed error
        /// before generation rather than leaking `<think>` into `.response`.
        ///
        /// - Parameters:
        ///   - messages: The rendered chat messages for this round.
        ///   - reasoningConfig: The resolved reasoning config to prime thinking with.
        ///   - thinkingEnabled: `true` / `false` to force thinking on / off,
        ///     `nil` for no preference.
        ///   - processor: The model's `UserInputProcessor`.
        ///   - transcriptEntries: The request's full transcript, for image-failure mapping.
        ///   - cannotDisableMessage: The message to surface if `reasoningConfig`
        ///     cannot honor `thinkingEnabled`.
        /// - Returns: The prepared `LMInput`.
        /// - Throws: `LanguageModelError.unsupportedCapability(.reasoning)` when
        ///   `reasoningConfig` cannot honor `thinkingEnabled`; otherwise whatever
        ///   `preparedInputMappingImageFailures` throws.
        private static func preparedInput(
            messages: [Chat.Message],
            reasoningConfig: ReasoningConfig,
            thinkingEnabled: Bool?,
            processor: any UserInputProcessor,
            transcriptEntries: some Collection<Transcript.Entry>,
            cannotDisableMessage: String
        ) async throws -> LMInput {
            let strategyContext = try additionalContextOrThrowCapabilityError(
                promptStrategy: reasoningConfig.promptStrategy,
                thinkingEnabled: thinkingEnabled,
                debugDescription: cannotDisableMessage)
            // History-preserving families (Qwen3.6) additionally render past
            // assistant turns WITH their replayed reasoning, so the same
            // fed-token prefix reappears verbatim in later rounds' history
            // re-renders (see `ReasoningConfig.historyPreservationKey`).
            let additionalContext = mergedAdditionalContext(
                strategyContext, reasoningConfig.historyPreservationContext)
            return try await preparedInputMappingImageFailures(
                processor: processor,
                input: UserInput(chat: messages, additionalContext: additionalContext),
                messages: messages, transcriptEntries: transcriptEntries)
        }

        /// Maps a requested reasoning level to a thinking on/off/unspecified
        /// flag. `nil` (no opinion) defers to the strategy's default; any
        /// concrete level means "think" (v1 does not modulate depth); only the
        /// package convention `.custom("no_think")` means "off".
        ///
        /// - Parameter level: The caller's requested reasoning level, if any.
        /// - Returns: `true` / `false` to force thinking on / off, `nil` for no preference.
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
        ///
        /// - Parameters:
        ///   - input: The rendered prompt to inspect.
        ///   - reasoningConfig: The reasoning config whose delimiters define a reasoning span.
        ///   - tokenizer: Used to decode the prompt's tail.
        /// - Returns: `true` when the prompt's tail ends inside an open reasoning span.
        private static func reasoningPrimedInside(
            input: LMInput, reasoningConfig: ReasoningConfig, tokenizer: any Tokenizer
        ) -> Bool {
            let tokens = input.text.tokens.asArray(Int.self)
            let renderedTail = tokenizer.decode(tokenIds: Array(tokens.suffix(64)))
            return ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: renderedTail, config: reasoningConfig)
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
        ///
        /// - Parameters:
        ///   - input: The tool-aware-template-rendered prompt.
        ///   - reasoningConfig: The think-then-call reasoning config (already gated by the caller).
        ///   - primedInside: Whether the prompt already ends inside an open reasoning span.
        ///   - maxTokens: The resolved token budget for this request.
        ///   - requestedTemperature: The caller's temperature override, if any.
        ///   - samplingMode: The caller's sampling mode override, if any.
        ///   - stableBoundary: The messages/tools for the hybrid
        ///     stable-boundary computation (see `makePromptCacheSlot`).
        ///   - reasoningEntryID: The entry to stream reasoning segments into.
        ///   - responseEntryID: The response entry to stream `.response` segments into.
        ///   - context: The loaded model context.
        ///   - channel: The generation channel to send events on.
        /// - Throws: `CancellationError` if the task is cancelled mid-loop; whatever
        ///   `generateTokensTask` throws otherwise.
        /// - Returns: The reasoning token IDs, and whether `</think>` closed
        ///   before the stream ended.
        private func runToolCallReasoningPhase(
            input: LMInput,
            reasoningConfig: ReasoningConfig,
            primedInside: Bool,
            maxTokens: Int,
            requestedTemperature: Double?,
            samplingMode: MLXSamplingMode?,
            stableBoundary: StableBoundaryRender?,
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
                config: reasoningConfig, primedInside: primedInside, tokenizer: context.tokenizer
            )

            let slot = try await makePromptCacheSlot(
                input: input, context: context, parameters: params, stableBoundary: stableBoundary)

            let (stream, task) = try generateTokensTask(
                input: slot.feedInput, cache: slot.cache, parameters: params, context: context)
            let closed: Bool
            var stopTokenFedToCache: Int?
            do {
                closed = try await collectReasoningTokens(
                    stream: stream, collector: &collector,
                    stopTokenFedToCache: &stopTokenFedToCache,
                    responseEntryID: responseEntryID, reasoningEntryID: reasoningEntryID,
                    channel: channel)
            } catch {
                // Drain the generation task before propagating, but do NOT sync
                // here: respond's outer `catch` is the single GPU-sync point for
                // this exit path. Keep one clean GPU sync per exit path —
                // cascading syncs across nested catches can race the Metal
                // command-buffer state during teardown.
                await Self.drainReasoningTask(task)
                throw error
            }
            // Drain the generation task before Phase 2 reuses the Stream.
            await Self.drainReasoningTask(task)
            Stream.gpu.synchronize()

            await Self.sendSegments(
                collector.finalize(), responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID, channel: channel)

            await Self.commitPromptCache(
                modelID: modelID, slot: slot, generatedTokenIDs: collector.reasoningTokenIDs,
                stopTokenFedToCache: stopTokenFedToCache)

            return (collector.reasoningTokenIDs, closed)
        }

        /// What `collectReasoningTokens`' consume loop should do after
        /// `processReasoningStreamEvent` handles one stream event.
        private enum ReasoningCollectionStep {
            /// Bookkeeping-only event; nothing to route, keep consuming.
            case proceed
            /// Route `segments` to the channel; when `stopsAfter` is `true`
            /// the collector's reasoning span closed on this token and the
            /// loop must stop consuming after routing them.
            case emit(segments: [ReasoningEventEmitter.Segment], stopsAfter: Bool)
        }

        /// Processes one stream event for `collectReasoningTokens`: records
        /// completion-info bookkeeping, or feeds a generated token through
        /// `collector` -- returning what the consume loop should do next,
        /// so the loop body stays a flat dispatch with no nested
        /// conditionals.
        ///
        /// - Parameters:
        ///   - generation: The stream event to process.
        ///   - collector: The reasoning-token collector to feed tokens into.
        ///   - stopTokenFedToCache: The terminal stop token fed into the
        ///     cache but not emitted (mutated on `.info`), so
        ///     `commitPromptCache` can key a hybrid checkpoint on the full
        ///     true sequence rather than dropping it. See
        ///     `GenerateCompletionInfo.stopTokenFedToCache`.
        /// - Returns: The step the consume loop should take.
        private static func processReasoningStreamEvent(
            _ generation: TokenGeneration,
            collector: inout ReasoningTokenCollector,
            stopTokenFedToCache: inout Int?
        ) -> ReasoningCollectionStep {
            switch generation {
            case .info(let info):
                stopTokenFedToCache = info.stopTokenFedToCache
                return .proceed
            case .token(let token):
                return .emit(
                    segments: collector.ingest(token),
                    stopsAfter: collector.shouldStopAfterReasoning)
            }
        }

        /// Consumes `stream`, feeding each generated token into `collector`
        /// and routing its emitted segments to the appropriate channel
        /// entry, until the stream ends or the collector's reasoning span
        /// closes.
        ///
        /// - Parameters:
        ///   - stream: The raw token stream from `generateTokensTask`.
        ///   - collector: The reasoning-token collector to feed tokens into.
        ///   - stopTokenFedToCache: The terminal stop token recorded from
        ///     the stream's completion info, if any (mutated).
        ///   - responseEntryID: The entry `.response` segments stream into.
        ///   - reasoningEntryID: The entry `.reasoning` segments stream into.
        ///   - channel: The generation channel to send events on.
        /// - Throws: `CancellationError` if the task is cancelled mid-loop.
        /// - Returns: Whether the collector's reasoning span closed
        ///   (`</think>` reached) before the stream ended.
        private func collectReasoningTokens(
            stream: AsyncStream<TokenGeneration>,
            collector: inout ReasoningTokenCollector,
            stopTokenFedToCache: inout Int?,
            responseEntryID: String,
            reasoningEntryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws -> Bool {
            for await generation in stream {
                try Task.checkCancellation()
                switch Self.processReasoningStreamEvent(
                    generation, collector: &collector,
                    stopTokenFedToCache: &stopTokenFedToCache)
                {
                case .proceed:
                    continue
                case .emit(let segments, let stopsAfter):
                    await Self.sendSegments(
                        segments, responseEntryID: responseEntryID,
                        reasoningEntryID: reasoningEntryID, channel: channel)
                    if stopsAfter {
                        return true
                    }
                }
            }
            return false
        }

        /// Cancels and drains `task` before its `Stream` is reused by Phase 2
        /// or before an error propagates -- without this, Phase 2's prefill
        /// could overlap Phase 1's in-flight forward pass on the shared
        /// `Stream` and trip a Metal command-buffer assertion.
        ///
        /// - Parameter task: The generation task returned by `generateTokensTask`.
        private static func drainReasoningTask(_ task: Task<Void, Never>) async {
            task.cancel()
            _ = await task.value
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
        ///
        /// - Parameters:
        ///   - outputBuffer: The full buffered generation output (a JSON tool-call envelope).
        ///   - userResponseSchema: The developer's response schema, if any.
        ///   - entryID: The response entry to stream a final-answer/fallback text delta into.
        ///   - toolCallsEntryID: The entry to stream a real tool-call event into.
        ///   - channel: The generation channel to send the event on.
        /// - Throws: Never currently -- `throws` matches the call chain's
        ///   shape from `runToolCalling` down; no step here actually throws today.
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
                // surface the failure content loudly as text, but with the
                // wrapper markers stripped -- `<tool_call>` marker text
                // must never leak into a final reply (kanban y4s0w2j).
                Logger(
                    subsystem: mlxFoundationModelsLoggingSubsystem, category: "ToolCalling"
                ).error(
                    "Tool-call envelope failed to parse; falling back to text (\(outputBuffer.count) chars). buffer=\(outputBuffer)"
                )
                await Self.sendDelta(
                    Self.malformedToolCallFallbackText(outputBuffer),
                    entryID: entryID, channel: channel, isReasoning: false)
                return
            }

            if name == FinalAnswerTool.toolName {
                await Self.handleFinalAnswerTool(
                    obj: obj, userResponseSchema: userResponseSchema,
                    entryID: entryID, channel: channel)
            } else {
                await Self.handleRealTool(
                    obj: obj, name: name, toolCallsEntryID: toolCallsEntryID, channel: channel)
            }
        }

        /// Handles the synthetic final-answer tool: unwraps its arguments
        /// into a single `.textDelta` event.
        ///
        /// - With no developer response schema: unwraps `arguments.response`
        ///   directly.
        /// - With a developer response schema: re-serializes `arguments`
        ///   back to JSON text; the session's normal response-parsing path
        ///   decodes it through the developer's `GenerationSchema`.
        ///
        /// - Parameters:
        ///   - obj: The parsed tool-call envelope (`{"name":..., "arguments":...}`).
        ///   - userResponseSchema: The developer's response schema, if any.
        ///   - entryID: The response entry to stream the final answer into.
        ///   - channel: The generation channel to send the event on.
        private static func handleFinalAnswerTool(
            obj: [String: Any],
            userResponseSchema: GenerationSchema?,
            entryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            let text: String
            if userResponseSchema == nil {
                let args = obj[ToolCallEnvelopeKey.arguments] as? [String: Any]
                text = (args?["response"] as? String) ?? ""
            } else if let args = obj[ToolCallEnvelopeKey.arguments],
                let argsStr = extractToolArgumentsAsJSON(args)
            {
                text = argsStr
            } else {
                text = ""
            }
            await sendDelta(text, entryID: entryID, channel: channel, isReasoning: false)
        }

        /// Handles a real (developer-declared) tool call: re-serializes its
        /// arguments to JSON and emits a single `.toolCallDelta` with a
        /// freshly minted tool-call id.
        ///
        /// - Parameters:
        ///   - obj: The parsed tool-call envelope (`{"name":..., "arguments":...}`).
        ///   - name: The tool's name.
        ///   - toolCallsEntryID: The entry to stream the tool-call event into.
        ///   - channel: The generation channel to send the event on.
        private static func handleRealTool(
            obj: [String: Any],
            name: String,
            toolCallsEntryID: String,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            guard
                let args = obj[ToolCallEnvelopeKey.arguments],
                let argsStr = extractToolArgumentsAsJSON(args)
            else {
                return
            }
            await channel.send(
                .toolCalls(
                    entryID: toolCallsEntryID,
                    action: .toolCall(
                        id: UUID().uuidString,
                        name: name,
                        action: .appendArguments(argsStr, tokenCount: textDeltaTokenCount)
                    )
                ))
        }

        /// Serializes a tool call's `arguments` value back to JSON text.
        ///
        /// Shared by `handleFinalAnswerTool` (developer-schema branch) and
        /// `handleRealTool`, which both re-serialize the parsed envelope's
        /// `arguments` value for their respective events.
        ///
        /// - Parameter arguments: The `arguments` value from a parsed
        ///   tool-call envelope (`obj[ToolCallEnvelopeKey.arguments]`); any
        ///   JSON-serializable value.
        /// - Returns: The JSON-encoded string, or `nil` if serialization or
        ///   UTF-8 decoding fails.
        private static func extractToolArgumentsAsJSON(_ arguments: Any) -> String? {
            guard let data = try? JSONSerialization.data(withJSONObject: arguments),
                let json = String(data: data, encoding: .utf8)
            else { return nil }
            return json
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
        ///
        /// - Parameter buffer: The full buffered generation output.
        /// - Returns: The inner JSON text, or `buffer` unchanged if it isn't wrapped.
        static func unwrapToolCallMarkers(_ buffer: String) -> String {
            let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            // Derived from the SAME structural tag the constrain side
            // frames the envelope with (`SchemaConverter
            // .encodeToolCallingGrammar` via `ToolCallStructuralTag
            // .forFormat`, which resolves every format to `.qwen`), so the
            // parse and constrain sides can never silently disagree about
            // the wrapper again (kanban y4s0w2j acceptance criterion).
            let tag = SchemaConverter.ToolCallStructuralTag.qwen
            let openMarker = tag.begin.trimmingCharacters(in: .whitespacesAndNewlines)
            let closeMarker = tag.end.trimmingCharacters(in: .whitespacesAndNewlines)
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

        /// The reply text to stream when a tool-calling round's buffered
        /// output fails to parse as an envelope.
        ///
        /// Strips the structural-tag wrapper markers before surfacing the
        /// buffer: the failure content should still surface loudly, but
        /// `<tool_call>` marker text must never leak into a final reply.
        /// On real hardware this is exactly what a budget-truncated
        /// runaway round produced — the raw wrapped buffer streamed as
        /// the reply, `"<tool_call>{\"name\": \"runCode\", …"` verbatim
        /// (kanban y4s0w2j).
        ///
        /// - Parameter outputBuffer: The full buffered generation output
        ///   that failed envelope parsing.
        /// - Returns: The buffer with any wrapper markers stripped.
        static func malformedToolCallFallbackText(_ outputBuffer: String) -> String {
            unwrapToolCallMarkers(outputBuffer)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
