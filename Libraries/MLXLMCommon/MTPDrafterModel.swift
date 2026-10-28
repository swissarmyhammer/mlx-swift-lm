// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A target model cannot provide the architecture-specific state required by
/// an MTP drafter.
public enum MTPDrafterCompatibilityError: Error, Sendable, Equatable, LocalizedError {
    case incompatibleTarget(drafter: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .incompatibleTarget(let drafter, let expected, let actual):
            "\(drafter) requires \(expected); received \(actual)."
        }
    }
}

/// Protocol for Multi-Token Prediction (MTP) speculative drafter models.
///
/// Mirrors `EmbeddingModel`'s relationship to `BaseLanguageModel`: this
/// protocol refines `BaseLanguageModel` with drafter-specific surface, so
/// implementations inherit weight loading and `sanitize` hooks while defining
/// their own forward signature.
///
/// MTP drafters do **not** conform to `LanguageModel` — their I/O contract is
/// different: a drafter consumes the target's last hidden state and per
/// layer-type shared K/V, produces a block of K-1 candidate tokens in a
/// single call, and does not store transient round-state on the model object.
/// The `MTPSpeculativeTokenIterator` extracts the shared K/V from the
/// target's `LMOutput.state` and threads it to the drafter as a method
/// argument.
///
/// Conforming types are expected to be stateless with respect to the target
/// model: every per-round input — including the target itself — flows through
/// `draftBlock(...)` as a method parameter. This makes drafter instances safe
/// to share across iterators without per-iterator mutable state on the model
/// instance. Drafters that need their own per-stream state additionally
/// conform to ``StatefulMTPDrafterModel``.
public protocol MTPDrafterModel: BaseLanguageModel {
    /// Whether the drafter can consume hidden states and model-owned values
    /// from this target architecture.
    func isCompatible(with target: any LanguageModel) -> Bool

    /// Largest total verification block the drafter can produce efficiently.
    /// `nil` means the caller may choose any block size.
    var maximumBlockSize: Int? { get }

    /// Whether this drafter consumes target-emitted shared K/V. Qwen owns a
    /// private full-attention cache and therefore only needs target hidden
    /// states; Gemma-style assistants consume the shared target K/V directly.
    var requiresSharedTargetKV: Bool { get }

    /// Whether the drafter must be prefilled over the shifted prompt before
    /// its first proposal. Qwen's private decoder cache requires this.
    var requiresPromptPrefill: Bool { get }

    /// Whether this drafter is currently supported only for greedy decoding.
    /// This keeps stochastic generation on the ordinary target path until a
    /// probability-ratio acceptance sampler is available.
    var requiresGreedySampling: Bool { get }

    /// Validate that `target` provides the architecture-specific state and
    /// model structure required by this drafter.
    ///
    /// Callers must invoke this before prefill or drafting. Implementations
    /// should keep this check side-effect free and throw
    /// ``MTPDrafterCompatibilityError`` for unsupported pairings.
    func validateCompatibility(with target: any LanguageModel) throws

    /// K-step drafting from a constant position.
    ///
    /// Returns the proposed tokens as a `[B, blockSize - 1]` MLXArray. The
    /// drafter holds no transient round-state on the model instance — every
    /// per-round input is threaded as a method argument.
    ///
    /// - Parameters:
    ///   - target: The main language model this drafter is speculating for.
    ///     Used to look up target-derived constants (input embeddings, embed
    ///     scale, etc.) inline per round; conformers must not retain or
    ///     mutate references derived from `target`.
    ///   - lastToken: Bonus token from the target's last verify pass, shape `[B]`.
    ///   - lastHidden: Target's last hidden state, shape `[B, 1, backbone_hidden_size]`.
    ///   - sharedKV: Dict keyed by `layer_type` (`"full_attention"` /
    ///     `"sliding_attention"`) mapping to `(keys, values)` `MLXArray`s for
    ///     the last layer of that layer-type in the target.
    ///   - positionDeltas: Optional target-emitted position delta state for
    ///     multimodal RoPE models whose continuation positions are not
    ///     derivable from cache length alone.
    ///   - queryOffset: Constant absolute position for the round (the
    ///     position the bonus token sits at in the target's KV cache).
    ///     Passed as a Swift `Int` rather than an `MLXArray` to avoid the
    ///     `.item()` round-trip that would otherwise stall the GPU per
    ///     speculation round.
    ///   - blockSize: Total tokens in the round (the drafter returns
    ///     `blockSize - 1`; the bonus token is implicit).
    ///   - sampler: `LogitSampler` to apply to each step's logits.
    /// - Returns: `[B, blockSize - 1]` token array.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray
}

extension MTPDrafterModel {
    public func isCompatible(with target: any LanguageModel) -> Bool {
        (try? validateCompatibility(with: target)) != nil
    }

    public var maximumBlockSize: Int? { nil }
    public var requiresSharedTargetKV: Bool { true }
    public var requiresPromptPrefill: Bool { false }
    public var requiresGreedySampling: Bool { false }
}

/// Target-side capability for rewinding an in-place speculative verify pass.
///
/// Rollback belongs to the target/cache implementation, not to the drafter.
/// The depth is explicit so a future multi-token drafter cannot accidentally
/// select an in-place path whose recurrent checkpoint is too shallow.
public protocol SpeculativeCacheRewindModel {
    var maximumNativeTargetCacheRewind: Int { get }
}

/// Per-stream state for MTP drafters that need their own transient storage.
///
/// The state is intentionally separate from ``LMOutput/State``: `LMOutput`
/// belongs to the target model's step output, while this value is owned by the
/// MTP iterator and passed only to the drafter.
public struct MTPDrafterState {
    public var cache: [KVCache]

    /// Absolute position of the first entry in `cache`.
    ///
    /// Zero for a drafter prefilled from the start of the prompt. A cold start at a position
    /// the target cache already holds begins its entries there, because the target hidden rows
    /// of the earlier positions are no longer available.
    public var cacheStartPosition: Int

    /// Absolute next position in a drafter-owned autoregressive cache.
    public var nextPosition: Int

    /// A proposal already computed while committing the previous verified
    /// round. Qwen uses this to avoid advancing its cache twice.
    public var seedToken: MLXArray?
    public var seedHidden: MLXArray?

    /// Number of tentative cache entries appended by the current proposal.
    public var proposalAppended: Int

    public init(
        cache: [KVCache],
        cacheStartPosition: Int = 0,
        nextPosition: Int = 0,
        seedToken: MLXArray? = nil,
        seedHidden: MLXArray? = nil,
        proposalAppended: Int = 0
    ) {
        self.cache = cache
        self.cacheStartPosition = cacheStartPosition
        self.nextPosition = nextPosition
        self.seedToken = seedToken
        self.seedHidden = seedHidden
        self.proposalAppended = proposalAppended
    }

    /// Entries the cache holds: committed entries, then the tentative entries of the current
    /// proposal.
    public var entryCount: Int { nextPosition - cacheStartPosition }

    /// Whether every cache layer holds exactly ``entryCount`` entries. A state without cache
    /// layers tracks positions only, and is always aligned.
    public var isAligned: Bool { cache.allSatisfy { $0.offset == entryCount } }

    /// Removes the last `count` entries: tentative proposal entries first, then committed ones.
    ///
    /// The precomputed seed proposal belongs to the removed tail, so it is dropped too.
    ///
    /// - Parameter count: The entries to remove.
    /// - Returns: `false`, leaving the state unchanged, when the state is not aligned, holds
    ///   fewer than `count` entries, or a cache layer cannot trim them.
    public mutating func removeLastEntries(_ count: Int) -> Bool {
        guard count > 0 else { return count == 0 && isAligned }
        guard isAligned, count <= entryCount,
            cache.isEmpty || trimPromptCache(cache, numTokens: count) == count
        else { return false }

        nextPosition -= count
        proposalAppended = Swift.max(0, proposalAppended - count)
        seedToken = nil
        seedHidden = nil
        return isAligned
    }

    /// Removes every entry and starts the empty cache at `position`.
    ///
    /// - Parameter position: The absolute position of the first entry the cache will hold.
    /// - Returns: `false`, leaving the state unchanged, when the entries cannot be removed.
    mutating func restart(at position: Int) -> Bool {
        guard position >= 0, removeLastEntries(entryCount) else { return false }
        cacheStartPosition = position
        nextPosition = position
        proposalAppended = 0
        return true
    }
}

/// Optional capability for MTP drafters that maintain per-stream state.
///
/// This keeps the base ``MTPDrafterModel`` surface minimal for stateless
/// drafters such as Gemma 4 assistants while giving Qwen-style MTP predictors
/// an iterator-owned state that can carry and trim their private KV cache. The
/// state must not be stored on the model instance because drafter models can be
/// shared across concurrent generation streams.
public protocol StatefulMTPDrafterModel: MTPDrafterModel {
    /// Create per-stream state owned and trimmed by the iterator.
    func makeState(parameters: GenerateParameters?) -> MTPDrafterState

    /// Prefill drafter-owned state from the target's prompt hidden states.
    func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )

    /// Stateful variant of ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)``.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray

    /// Reconcile tentative proposal writes with the sequence accepted by the
    /// target, then seed the next proposal if the architecture supports it.
    func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )
}

/// Optional capability for stateful MTP drafters whose private state can be
/// normalized at a generation boundary and resumed from a later prompt suffix.
///
/// `ChatSession` deliberately does not know how a drafter's cache is shifted,
/// which seed values are speculative, or how target hidden states must be
/// paired with suffix tokens. The MTP iterator owns that lifecycle and calls
/// this interface at the model seam. A conforming drafter also cold-starts at a
/// position a warm target cache already holds, so a warm cache without an aligned
/// continuation still speculates. Stateful drafters that do not conform are
/// still usable for cold generations; a warm target cache continues without
/// speculation rather than combining it with a fresh drafter state.
///
/// The default implementations describe a shifted drafter (Qwen NextN), which
/// pairs each input token with the target hidden state of the preceding
/// position. Such a drafter implements only ``appendDrafterState(target:tokens:targetHidden:positionDeltas:state:sampler:)``.
public protocol ResumableMTPDrafterModel: StatefulMTPDrafterModel {
    /// The identity a continuation of this drafter records, so that only a drafter of the same
    /// identity resumes it. Drafters that hold the same identity build the same cache layers and
    /// read them the same way.
    ///
    /// The default is the type name. A drafter type whose instances can differ in configuration
    /// gives ``MTPDrafterID/init(drafterType:configuration:)``.
    var drafterID: MTPDrafterID { get }

    /// Convert `state` to a canonical boundary after the target cache has
    /// discarded any verified-but-unemitted lookahead.
    ///
    /// Return `false` when the private state cannot be proven to represent the
    /// supplied target position. The caller then keeps the target cache and
    /// drops only the MTP continuation.
    func finalizeDrafterState(
        target: any LanguageModel,
        targetBoundaryHidden: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens: Int,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool

    /// Extend a canonical boundary over a prompt suffix and prime the next
    /// speculative round.
    ///
    /// `targetBoundaryHidden` is the hidden state of the final token already
    /// represented by the warm target cache. `suffixTargetHidden` contains one
    /// row per newly evaluated suffix token, and `firstBonus` is the token
    /// sampled after that suffix.
    func resumeDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        targetBoundaryHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool

    /// Append one committed entry for each token of `tokens` at `state.nextPosition`, and
    /// seed the next proposal from the last appended entry.
    ///
    /// Entry `i` pairs `tokens[i]` with the target hidden row `targetHidden[i]`, the row of
    /// the position before the token. Together with
    /// ``MTPDrafterState/removeLastEntries(_:)`` this edits the drafter state the same way a
    /// cold prefill over the resulting tokens would build it.
    ///
    /// - Parameters:
    ///   - target: The main language model the drafter speculates for.
    ///   - tokens: The tokens to append, shape `[B, n]`.
    ///   - targetHidden: At least `n` target hidden rows, shape `[B, ≥n, hidden]`.
    ///   - positionDeltas: Optional target position deltas for multimodal RoPE.
    ///   - state: The drafter state. It must be aligned and hold no tentative entries.
    ///   - sampler: The sampler of the seed proposal.
    /// - Returns: `false`, leaving the state unchanged, when the inputs or the state do not
    ///   allow the append.
    func appendDrafterState(
        target: any LanguageModel,
        tokens: MLXArray,
        targetHidden: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool

    /// Start an empty drafter state at `position`, a position the warm target cache already
    /// holds, and prime it over the prompt suffix the target just prefilled.
    ///
    /// The entry for the first suffix token needs the target hidden row of the position before
    /// it, which no cache keeps, so the entries begin with the second suffix token. Speculation
    /// stays exact without the earlier entries: only the acceptance rate depends on them.
    ///
    /// - Parameters:
    ///   - target: The main language model the drafter speculates for.
    ///   - suffixTokens: The prompt suffix the target prefilled at `position`, shape `[B, n]`.
    ///   - suffixTargetHidden: One target hidden row per suffix token.
    ///   - firstBonus: The token the target sampled after the suffix.
    ///   - position: The target cache position of the first suffix token.
    ///   - positionDeltas: Optional target position deltas for multimodal RoPE.
    ///   - state: The drafter state to replace. Any entries it holds are removed.
    ///   - sampler: The sampler of the seed proposal.
    /// - Returns: `false` when the drafter cannot start at `position`.
    func coldStartDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        firstBonus: MLXArray,
        position: Int,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool
}

extension ResumableMTPDrafterModel {
    /// The fully qualified type name of the drafter.
    public var drafterID: MTPDrafterID {
        MTPDrafterID(String(reflecting: Self.self))
    }

    /// Resumes a shifted drafter: the boundary hidden row pairs with the first suffix token, and
    /// each suffix hidden row pairs with the token after it, ending with `firstBonus`.
    public func resumeDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        targetBoundaryHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool {
        let suffix = normalizedMTPTokenBatch(suffixTokens)
        let suffixLength = suffix.dim(-1)
        guard suffixLength > 0, suffixTargetHidden.dim(1) >= suffixLength,
            targetBoundaryHidden.dim(1) == 1
        else { return false }

        return appendDrafterState(
            target: target,
            tokens: concatenated([suffix, normalizedMTPColumn(firstBonus)], axis: 1),
            targetHidden: concatenated(
                [targetBoundaryHidden, suffixTargetHidden[0..., ..<suffixLength, 0...]], axis: 1),
            positionDeltas: positionDeltas, state: &state, sampler: sampler)
    }

    /// Cold-starts a shifted drafter: each suffix hidden row pairs with the token after it,
    /// ending with `firstBonus`, so the entries begin at `position`.
    public func coldStartDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        firstBonus: MLXArray,
        position: Int,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool {
        let suffix = normalizedMTPTokenBatch(suffixTokens)
        let suffixLength = suffix.dim(-1)
        guard suffixLength > 0, suffixTargetHidden.dim(1) >= suffixLength,
            state.restart(at: position)
        else { return false }

        return appendDrafterState(
            target: target,
            tokens: concatenated([suffix[0..., 1...], normalizedMTPColumn(firstBonus)], axis: 1),
            targetHidden: suffixTargetHidden[0..., ..<suffixLength, 0...],
            positionDeltas: positionDeltas, state: &state, sampler: sampler)
    }

    /// Prefills a shifted drafter over the whole prompt: a cold start at position zero.
    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        _ = coldStartDrafterState(
            target: target, suffixTokens: promptTokens, suffixTargetHidden: targetHidden,
            firstBonus: firstBonus, position: 0, positionDeltas: positionDeltas, state: &state,
            sampler: sampler)
    }

    /// Commits a verified round to a shifted drafter: removes the rejected tentative entries,
    /// then appends the accepted drafts that have no entry yet and `finalToken`, each paired
    /// with the verifier hidden row of the position before it.
    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let keptTentative = Swift.min(acceptedCount, state.proposalAppended)
        _ = state.removeLastEntries(state.proposalAppended - keptTentative)
        state.proposalAppended = 0

        let finalColumn = normalizedMTPColumn(finalToken)
        let tokens =
            acceptedCount > keptTentative
            ? concatenated(
                [draftTokens[0..., keptTentative ..< acceptedCount], finalColumn], axis: 1)
            : finalColumn
        _ = appendDrafterState(
            target: target, tokens: tokens,
            targetHidden: targetHidden[0..., keptTentative ..< (acceptedCount + 1), 0...],
            positionDeltas: positionDeltas, state: &state, sampler: sampler)
    }

    /// Normalizes a shifted drafter at a resumable target boundary.
    ///
    /// A shifted drafter pairs each input token with the target hidden state from the
    /// preceding position. At a generation boundary its private cache must therefore end one
    /// position before the target cache so the saved boundary hidden state can bridge the next
    /// prompt suffix. Tentative entries of a proposal the target never verified are removed
    /// first.
    public func finalizeDrafterState(
        target _: any LanguageModel,
        targetBoundaryHidden _: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens: Int,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool {
        let committedPosition = state.nextPosition - state.proposalAppended
        guard targetProcessedTokenCount > 0,
            committedPosition == targetProcessedTokenCount + discardedTargetTokens,
            state.removeLastEntries(state.proposalAppended + discardedTargetTokens + 1)
        else { return false }

        return state.nextPosition == targetProcessedTokenCount - 1
    }
}

extension StatefulMTPDrafterModel {
    public func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {}

    public func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        let rejected = draftTokens.dim(-1) - acceptedCount
        if rejected > 0 {
            trimPromptCache(state.cache, numTokens: rejected)
        }
    }
}

/// Tokens proposed by an MTP drafter together with the processed logits that
/// defined each proposal distribution.
///
/// `processedLogits` has shape `[B, blockSize - 1, vocabularySize]`. Each row
/// is captured after the drafter-side ``LogitProcessor`` and before the
/// ``LogitSampler``. Keeping this information is what lets the verifier apply
/// distribution-preserving speculative sampling instead of accepting only
/// independently sampled exact-token matches.
public struct MTPDraftBlockOutput {
    public let tokens: MLXArray
    public let processedLogits: MLXArray

    public init(tokens: MLXArray, processedLogits: MLXArray) {
        self.tokens = tokens
        self.processedLogits = processedLogits
    }
}

/// Optional refinement for MTP drafters that support exact speculative
/// sampling.
///
/// The base ``MTPDrafterModel`` API is retained for source compatibility and
/// for greedy-only custom drafters. Non-greedy MTP generation requires this
/// refinement: without the drafter distribution `q`, no verifier can preserve
/// the target distribution `p` on rejection.
public protocol MTPDistributionDrafterModel: MTPDrafterModel {
    func draftBlockWithLogits(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        processor: (any LogitProcessor)?,
        sampler: any LogitSampler
    ) -> MTPDraftBlockOutput
}

/// Lightweight context for an MTP drafter — simpler than `ModelContext`
/// because drafters have no tokenizer, no user input processor, no chat
/// template.
///
/// Not `Sendable`; cross-domain access goes through ``MTPDrafterContainer``.
public struct MTPDrafterContext {
    public var configuration: ModelConfiguration
    public var model: any MTPDrafterModel

    public init(configuration: ModelConfiguration, model: any MTPDrafterModel) {
        self.configuration = configuration
        self.model = model
    }
}

/// Sendable container for an ``MTPDrafterContext``.
///
/// Mirrors the ``ModelContainer`` pattern: a `final class : Sendable` that
/// wraps the non-Sendable context in a `SerialAccessContainer` and exposes
/// async `perform(_:)` closures for serialized access.
public final class MTPDrafterContainer: Sendable {
    private let context: SerialAccessContainer<MTPDrafterContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public init(context: consuming MTPDrafterContext) {
        self.context = .init(context)
    }

    /// Perform an action on the ``MTPDrafterContext``. Callers _must_ eval
    /// any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    public func perform<R: Sendable>(
        _ action: @Sendable (MTPDrafterContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }
}

// MARK: - Cross-model state keys
//
// Public ``LMOutput/Key`` declarations for MTP speculative decoding. The
// target model (e.g. ``Gemma4`` in MLXVLM) writes these into its
// ``LMOutput/state`` when the iterator opts in via ``mtpEmitFlagKey``; the
// ``MTPSpeculativeTokenIterator`` reads them and threads them to the drafter
// as method arguments. Public scope is required because writer and reader
// live in different modules.

/// Target writes the hidden representation required by its MTP head here.
/// This is architecture-specific, but must match the representation used to
/// train the paired MTP head. Qwen and Gemma assistants both publish their
/// post-final-norm representation. The iterator threads it to the drafter
/// unchanged as `lastHidden`.
public let mtpLastHiddenStatesKey =
    LMOutput.Key<MLXArray>("mtp.lastHiddenStates")

/// Target writes one `(keys, values)` tuple per `layer_type`
/// (`"full_attention"`, `"sliding_attention"`) here, drawn from the last
/// layer of each type. ``MTPSpeculativeTokenIterator`` reads it and threads
/// it to the drafter as `sharedKV`.
public let mtpSharedKVStatesKey =
    LMOutput.Key<[String: (MLXArray, MLXArray)]>("mtp.sharedKVStates")

/// Target writes the absolute cache offset for each emitted shared-K/V
/// layer-type here. The K/V tensors themselves may have a capped sequence
/// axis when backed by a rotating cache, so their shape is not always the
/// absolute RoPE/query position.
public let mtpSharedKVOffsetsKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVOffsets")

/// Target writes optional position delta state here for MTP drafters that
/// need to reproduce target-specific RoPE continuation positions.
public let mtpPositionDeltasKey =
    LMOutput.Key<MLXArray>("mtp.positionDeltas")

/// The MTP iterator sets this key on the ``LMOutput/State`` it passes into
/// the main model on each call to opt the target into emitting
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey``. An absent key
/// reads as `false` (no emit), so non-MTP callers are unaffected.
public let mtpEmitFlagKey = LMOutput.Key<Bool>("mtp.emitDrafterState")

/// Receives the target hidden rows of one prefill chunk, as a target's
/// ``LanguageModel/prepare(_:cache:state:prefill:)`` evaluates it.
///
/// - Parameters:
///   - tokens: The tokens of the chunk, shape `[B, n]`.
///   - hiddenStates: One target hidden row per token of the chunk, shape `[B, n, hidden]`, in the
///     representation of ``mtpLastHiddenStatesKey``.
///   - positionDeltas: The position deltas the drafter applies to the chunk, as
///     ``mtpPositionDeltasKey`` carries them, or `nil` for a target without them.
public typealias MTPPrefillChunkHandler = (
    _ tokens: MLXArray, _ hiddenStates: MLXArray, _ positionDeltas: MLXArray?
) -> Void

/// The MTP iterator sets this key, beside ``mtpEmitFlagKey``, on the ``LMOutput/State`` it passes
/// to the target's ``LanguageModel/prepare(_:cache:state:prefill:)``. A target that prefills the
/// prompt in chunks calls the handler after the forward of each chunk but the last one, with
/// that chunk's hidden rows, so a drafter fills its own cache chunk by chunk and no chunk's rows
/// outlive the next chunk. The ``mtpLastHiddenStatesKey`` of the state `prepare` returns, or of
/// the iterator's forward over the tokens `prepare` returns, then holds the rows of the remaining
/// positions only.
public let mtpPrefillChunkHandlerKey = LMOutput.Key<MTPPrefillChunkHandler>(
    "mtp.prefillChunkHandler")

extension LMOutput.State {
    /// The ``mtpPrefillChunkHandlerKey`` handler a target's chunked prefill calls, or `nil` when
    /// the state carries none or does not ask the target for MTP drafter state at all.
    package var mtpPrefillChunkHandler: MTPPrefillChunkHandler? {
        self[mtpEmitFlagKey] == true ? self[mtpPrefillChunkHandlerKey] : nil
    }

    /// Hands one prefill chunk to the ``mtpPrefillChunkHandler``: the chunk's tokens, the
    /// ``mtpLastHiddenStatesKey`` rows its forward emitted, and its position deltas. Does nothing
    /// when the state carries no handler or the forward emitted no hidden rows.
    ///
    /// A target's chunked `prepare` calls this on the state it was handed, after each chunk's
    /// forward.
    ///
    /// - Parameters:
    ///   - tokens: The tokens of the chunk, shape `[B, n]`.
    ///   - output: The output of the chunk's forward.
    ///   - positionDeltas: The position deltas of the chunk, for a target that computes them
    ///     before the forward; `nil` takes the ``mtpPositionDeltasKey`` deltas of `output`.
    package func feedMTPPrefillChunk(
        tokens: MLXArray, output: LMOutput, positionDeltas: MLXArray? = nil
    ) {
        guard let handler = mtpPrefillChunkHandler,
            let hiddenStates = output.state?[mtpLastHiddenStatesKey]
        else { return }
        handler(tokens, hiddenStates, positionDeltas ?? output.state?[mtpPositionDeltasKey])
    }
}

/// Requests a recurrent-cache checkpoint after this many verification input
/// tokens. Hybrid Qwen models use `1` for MTP-1 so a rejected draft restores
/// state after the always-committed bonus token without replaying the model.
public let mtpCacheCheckpointIndexKey =
    LMOutput.Key<Int>("mtp.cacheCheckpointIndex")

/// Which cache entry each ``mtpSharedKVStatesKey`` tuple was read from, keyed the same way.
///
/// A speculative round keeps only part of what its verify pass wrote, so the emitted snapshot has
/// to be reconciled against the cache afterwards. That reconciliation is exact only if the
/// consumer knows the entry behind each tuple: a sliding layer's snapshot is bounded by its ring
/// and a global layer's is not, and at the crossing the two are indistinguishable by length.
///
/// Public alongside its siblings, because it is part of the same contract rather than an
/// implementation detail of one writer: a target that sets ``mtpSharedKVStatesKey`` is expected to
/// set this too, and a snapshot that arrives without it is refused rather than reconciled against
/// a guessed bound. A target outside this package cannot satisfy that contract without it.
public let mtpSharedKVSourceIndicesKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVSourceIndices")

// MARK: - Iterator stats surface

/// Introspection surface for token iterators that perform MTP speculative
/// decoding.
///
/// The iterator value lives inside `generateLoopTask` and never escapes the
/// stream, so its per-stream draft proposal and acceptance counters are not
/// reachable through the high-level `generate(...)` API. Conforming the
/// iterator to this protocol lets `generateLoopTask` downcast and thread the
/// counters through the emitted `.info` event's
/// ``GenerateCompletionInfo/proposedDraftTokens``,
/// ``GenerateCompletionInfo/acceptedDraftTokens``, and
/// ``GenerateCompletionInfo/passthroughReason`` fields. Non-MTP iterators do
/// not conform; the downcast returns nil and the fields default to nil for
/// non-MTP streams.
public protocol MTPStatsCollecting {
    /// Total tokens proposed across all speculation rounds in the stream.
    var proposedDraftTokens: Int { get }

    /// Total tokens accepted by the target across all speculation rounds.
    var acceptedDraftTokens: Int { get }

    /// Nil when speculation completed with a resumable boundary. Non-nil if
    /// sticky passthrough engaged or a stateful continuation had to be
    /// discarded, with the first failure reason captured by the iterator.
    var passthroughReason: String? { get }
}
