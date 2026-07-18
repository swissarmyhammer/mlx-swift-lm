// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

/// Common shape shared by ``PromptCache/StoredChunk`` and
/// ``PromptCache/HybridCheckpoint``: both carry a recency stamp for LRU
/// eviction and a real byte footprint for budget accounting. Lets
/// `PromptCache`'s generic `insertOrUpdateEntry(_:key:modelID:into:)`/
/// `updateBest(_:scanning:victimCase:)` helpers operate identically over
/// either store, instead of duplicating the same dedup/recency/byte-
/// accounting pattern across `insert`/`insertTail`/`insertHybridCheckpoint`,
/// or the same scanning loop across `globalLRUVictim()`'s two stores.
private protocol CacheStoreEntry {
    var lastUsed: Int { get set }
    var byteSize: Int { get }
}

extension PromptCache.StoredChunk: CacheStoreEntry {}
extension PromptCache.HybridCheckpoint: CacheStoreEntry {}

/// Per-model, chunked shared KV-cache store for `MLXLanguageModel.Executor`.
///
/// FoundationModels' `LanguageModelExecutor` protocol has no session
/// identity: `Executor.respond(to:model:streamingInto:)` receives the
/// *full* transcript on every round with no indication of which prior
/// round (if any) it continues, or whether this is even the same session
/// as last time. `PromptCache` keys reuse on the token prefix itself
/// instead -- but not as a handful of whole-conversation slots checked out
/// and returned. Each model's KV state is sliced into fixed-size,
/// content-addressed chunks (see `PromptCacheChunks.swift`'s
/// `sliceChunks`/`StoredChunk`) that form a hash chain over chunk-aligned
/// token windows (`chunkKey`), stored once in a per-model shared pool
/// (``chunkStore``) and deduplicated across every conversation that shares
/// a prefix -- SGLang RadixAttention-style, not llama.cpp server's
/// per-conversation slot pool this actor used before this cutover.
///
/// `resolve` walks the stored chunk chain against the freshly-tokenized
/// prompt (``lookupLongestPrefix(modelID:newTokens:chunkSize:)``),
/// assembles any matched chunks into a fresh, PRIVATE `[KVCache]`
/// (``assemble(chunks:layerCount:)``), and feeds only the suffix past the
/// matched prefix. Because chunks are read-only and every `resolve` gets
/// its own assembled copy, there is no checkout, no steal, and no
/// double-checkout hazard by construction -- unlike the old slot pool's
/// resolve/store check-out-and-return dance, nothing is ever removed from
/// the store on a read. `store` slices a completed round's cache into
/// chunks (``PromptCache/sliceChunks(tokens:cache:chunkSize:)``) and checks
/// them into the shared pool (``insert(modelID:chunks:)``), where dedup
/// means two conversations sharing a prefix retain that prefix's KV state
/// exactly once.
///
/// Assembly always produces a CONTIGUOUS copy of the matched chunks' K/V
/// tensors (see ``assemble(chunks:layerCount:)``'s `ownedCopy(of:)` use):
/// MLX's scaled-dot-product-attention kernel requires contiguous K/V
/// tensors, so a naive "reference the chunk store's own tensors" assembly
/// would either force a copy at attention time anyway or crash on a
/// non-contiguous input -- assembly pays that copy cost once, up front,
/// rather than repeatedly inside the hot generation loop.
///
/// CAPACITY: the chunk store is bounded by a GLOBAL byte budget shared
/// across every cached model (``byteBudget``), not a per-model budget --
/// see that property's doc comment for why a single shared pool is the
/// right scope for a process serving multiple models. Every
/// ``insert(modelID:chunks:)`` that pushes the store's accounted total
/// (``totalStoredBytes``) over budget runs ``evictToBudget()``, which
/// evicts the globally least-recently-used chunk -- and, transitively,
/// every chunk chained under it (``evictChunkAndDescendants(modelID:key:)``)
/// -- until the store is back under budget. Transitive eviction is
/// deliberate: evicting a chunk orphans its descendants (a future
/// ``lookupLongestPrefix(modelID:newTokens:chunkSize:)`` walk can never
/// reach past a missing parent), and this actor reclaims that orphaned
/// lineage's bytes immediately rather than leaving dead weight in the
/// store for a later sweep. See
/// `MLXLanguageModel.setPromptCacheByteBudget(_:)` for the public API and
/// the full peak-memory model the budget participates in (kanban
/// `2sdt6dj`).
///
/// Mirrors `ModelCache`'s per-model-actor-cache pattern in
/// `MLXLanguageModel.swift` otherwise: a private actor, instantiated once
/// as a process-global `static let`, exposed to `Executor` through thin
/// passthrough statics on `MLXLanguageModel`.
actor PromptCache {

    /// Default token span each stored chunk covers (see `sliceChunks`/
    /// `lookupLongestPrefix`) -- the initial value of this actor's
    /// `chunkSize` property, until `setChunkSize(_:)` changes it. 64
    /// balances fork-point granularity (how finely two conversations
    /// sharing a prefix can diverge and still share chunks) against
    /// per-chunk bookkeeping overhead and hash-chain walk length.
    static let defaultChunkSize = 64

    /// This actor's currently configured chunk span, threaded into every
    /// `resolve()`/`store()` call (see `lookupLongestPrefix`/`sliceChunks`'s
    /// own `chunkSize` parameters). Changed only through `setChunkSize(_:)`,
    /// which enforces the `>= 1` invariant and evicts the store on a
    /// genuine change -- see that method's doc comment.
    private var chunkSize = PromptCache.defaultChunkSize

    /// Per-model chunk store: each model's chunks keyed by their own
    /// ``ChunkKey`` (see `PromptCacheChunks.swift`'s hash chain) -- the
    /// sole backing store for `resolve()`/`store()` since this cutover
    /// (see this actor's own doc comment). See ``insert(modelID:chunks:)``
    /// for dedup semantics.
    private var chunkStore: [String: [ChunkKey: StoredChunk]] = [:]

    /// Per-model index of the currently ACTIVE tail chunk at each hash-chain
    /// attachment point: `activeTailKey[modelID][parentKey]` is the
    /// ``ChunkKey`` of the one tail `StoredChunk` (see
    /// `PromptCacheChunks.sliceTailChunk` -- a variable-length, sub-`chunkSize`
    /// chunk covering a conversation's trailing partial span) currently
    /// checked into ``chunkStore`` at that chain position, if any.
    ///
    /// Tails are stored in ``chunkStore`` alongside full chunks (both are
    /// `StoredChunk`s, keyed by their own `ChunkKey`), so byte accounting and
    /// LRU/orphan eviction (``totalStoredBytes``, ``evictChunkAndDescendants(modelID:key:)``,
    /// ``globalLRUVictim()``) already see and reclaim them without any change.
    /// This index exists only so ``insertTail(modelID:tail:)`` can find and
    /// replace the PREVIOUS tail at a given attachment point in O(1): unlike a
    /// full chunk's content (which never changes once written), a tail's
    /// content changes on nearly every turn -- a growing conversation appends
    /// new tokens each round, producing a different tail (and therefore a
    /// different `chunkKey`) even though it's still chained from the SAME
    /// `parentKey`. Retaining every historical tail at a position would only
    /// ever serve the stalest one usefully for at most one turn before
    /// becoming garbage, so this actor keeps just ONE active tail per
    /// attachment point.
    private var activeTailKey: [String: [ChunkKey: ChunkKey]] = [:]

    /// Per-model store of whole-stack hybrid Mamba/attention checkpoints
    /// (see `PromptCacheChunks.swift`'s `HybridCheckpoint`), keyed by
    /// `chunkKey(parentKey: rootChunkKey, tokens: checkpoint.tokens)` -- a
    /// content hash over the FULL token prefix a checkpoint covers, not a
    /// hash-chain position like ``chunkStore``'s keys. Unlike the ordinary
    /// chunk store, entries here have no parent/child chain: each
    /// checkpoint is a flat, independent whole-stack snapshot, matched by
    /// ``resolveHybridCheckpoint(modelID:newTokens:)``'s longest-prefix scan
    /// rather than a chained walk.
    ///
    /// Populated by ``insertHybridCheckpoint(modelID:checkpoint:)`` (called
    /// from ``store(modelID:tokens:cache:)`` when
    /// `PromptCache.isHybridMambaAttention(_:)` is `true`), and shares
    /// ``totalStoredBytes``/``byteBudget`` accounting and LRU eviction with
    /// ``chunkStore`` (see ``globalLRUVictim()``/``evictToBudget()``).
    private var hybridCheckpoints: [String: [ChunkKey: HybridCheckpoint]] = [:]

    /// Total bytes currently accounted across EVERY model's chunk store
    /// (``StoredChunk/byteSize``, summed) -- a single, process-global count
    /// rather than one per model. See ``byteBudget``'s doc comment for why
    /// this actor treats the byte budget (and therefore this total) as
    /// shared across every model it caches, not scoped per model.
    ///
    /// Maintained incrementally (incremented on a genuinely NEW key in
    /// ``insert(modelID:chunks:)``, decremented in
    /// ``evictChunkAndDescendants(modelID:key:)``/``remove(modelID:)``/
    /// ``evictAll()``) rather than recomputed by scanning `chunkStore`, so
    /// checking it against ``byteBudget`` after every insert is O(1).
    private var totalStoredBytes = 0

    /// The total byte budget every cached model's stored chunks share,
    /// enforced by ``evictToBudget()`` after every
    /// ``insert(modelID:chunks:)``.
    ///
    /// GLOBAL across models (not per-model) by design: every model this
    /// actor caches competes for the SAME physical unified-memory pool (see
    /// `MLXLanguageModel.setPromptCacheByteBudget(_:)`'s peak-memory model),
    /// and this is a single-process cache that may hold multiple models at
    /// once (``chunkStore``'s doc comment). A per-model split would either
    /// have to be decided up front -- starving whichever model isn't
    /// currently busy while wasting headroom on one nobody has touched in a
    /// while -- or reduce back to a global figure anyway to reason about
    /// real memory pressure. A single shared budget instead lets whichever
    /// model is actively serving requests claim more of the pool, with
    /// `lastUsed`-driven LRU naturally reclaiming from cold models first.
    ///
    /// Changed only through ``setByteBudget(_:)`` (reached publicly via
    /// `MLXLanguageModel.setPromptCacheByteBudget(_:)`), which re-runs
    /// ``evictToBudget()`` immediately in case the new value is smaller
    /// than what's currently stored. ``setByteBudget(_:)`` clamps only up
    /// to `1` (mirroring ``setChunkSize(_:)``'s own `max(1, size)`
    /// pattern) -- a caller EXPLICITLY requesting a small budget (a
    /// memory-constrained embedded scenario, or a test asserting eviction
    /// behavior under tight pressure) is trusted to mean it. The larger
    /// ``minimumByteBudget`` floor applies only to the UNCONFIGURED
    /// default (``defaultByteBudget()``), not to an explicit caller value.
    private var byteBudget = PromptCache.defaultByteBudget()

    /// Floor applied only to the DERIVED/fallback default
    /// (``defaultByteBudget()``), never to an explicit caller-supplied
    /// value passed to ``setByteBudget(_:)`` (which clamps only up to `1`
    /// -- see ``byteBudget``'s doc comment for why those two floors
    /// differ).
    ///
    /// A single stored chunk's real byte footprint is a function of the
    /// model's layer count, per-layer head/embedding dimensions, and this
    /// actor's own `chunkSize` -- none of which this actor can inspect in
    /// the abstract (it caches chunks for whatever models happen to load),
    /// so this is a conservative, hand-picked floor rather than a computed
    /// "exactly one chunk" size: 16 MiB comfortably holds many chunks'
    /// worth of KV state for typical small-to-mid model configurations, so
    /// an UNCONFIGURED budget clamped to this floor is never reasonably a
    /// single-chunk-or-less budget in practice -- avoiding a pathological
    /// default (e.g. a broken working-set query returning near-zero) that
    /// would otherwise evict every chunk on every insert.
    static let minimumByteBudget = 16 * 1_024 * 1_024

    /// Fixed fallback default byte budget when MLX cannot report a device
    /// working-set size (see ``defaultByteBudget()``) -- 2 GiB, a size that
    /// comfortably holds many chunks' worth of KV state across typical
    /// small-to-mid model configurations without assuming any particular
    /// device's actual capacity.
    static let fallbackDefaultByteBudget = 2 * 1_024 * 1_024 * 1_024

    /// Fraction of the device's reported GPU working-set size (see
    /// ``defaultByteBudget()``) this store claims by default.
    ///
    /// Deliberately well under `1.0`: the SAME unified-memory pool this
    /// fraction is drawn from also has to hold the model's own weights
    /// (typically the largest fixed consumer), MLX's own GPU buffer cache
    /// (`Memory.cacheLimit`, which itself defaults to the FULL memory
    /// limit -- see `MLX.Memory.cacheLimit`'s doc comment), and one
    /// assembled prefix copy PER IN-FLIGHT request
    /// (``assemble(chunks:layerCount:)``). Reserving only a quarter of the
    /// working set for the persistent chunk store leaves the remaining
    /// three quarters as headroom for those other consumers -- see
    /// `MLXLanguageModel.setPromptCacheByteBudget(_:)`'s doc comment for
    /// the full peak-memory model this trades against.
    static let unifiedMemoryBudgetFraction = 0.25

    /// Derives this actor's default byte budget from MLX's reported GPU
    /// working-set size (``MLX/GPU/maxRecommendedWorkingSetBytes()``,
    /// backed by Metal's `recommendedMaxWorkingSetSize`) when available,
    /// scaled by ``unifiedMemoryBudgetFraction`` to leave headroom for
    /// model weights, MLX's own GPU buffer cache, and assembled-prefix
    /// copies (see that constant's doc comment). Falls back to
    /// ``fallbackDefaultByteBudget`` when MLX reports no usable working-set
    /// size (e.g. no GPU device). Either way, the result is clamped up to
    /// ``minimumByteBudget``.
    ///
    /// - Returns: The derived (or fallback) default byte budget.
    static func defaultByteBudget() -> Int {
        guard let workingSet = MLX.GPU.maxRecommendedWorkingSetBytes(), workingSet > 0 else {
            return max(minimumByteBudget, fallbackDefaultByteBudget)
        }
        let derived = Int(Double(workingSet) * unifiedMemoryBudgetFraction)
        return max(minimumByteBudget, derived)
    }

    /// Monotonic source for `StoredChunk.lastUsed`. Actor isolation
    /// serializes every read/increment, so this is a strict per-call
    /// counter with no concurrent-access races.
    private var recencyCounter = 0

    private func nextRecency() -> Int {
        recencyCounter += 1
        return recencyCounter
    }

    /// Trims `cache` from `currentOffset` back to `targetOffset`,
    /// verifying every layer's cache actually reached `targetOffset`
    /// afterward.
    ///
    /// `trim(_:)`'s return value is not guaranteed to satisfy the full
    /// requested amount for every conformer -- e.g. `ChunkedKVCache` is
    /// bounded by `offset - startPosition` (see `KVCache.swift`) -- so a
    /// caller that blindly trusted the request would silently feed
    /// subsequent tokens against a cache whose real offset doesn't match
    /// what it assumes, corrupting attention with no crash. Checking
    /// every layer (not just `cache.first`) catches a mixed-cache-type
    /// array where layers could in principle trim by different amounts,
    /// even though in practice all layers of one `[KVCache]` always
    /// advance in lockstep during normal generation.
    ///
    /// - Parameters:
    ///   - cache: The cache array to trim, in place.
    ///   - currentOffset: The offset every layer is assumed to be at
    ///     before trimming (the token count the caller believes is cached).
    ///   - targetOffset: The offset every layer must reach for the trim
    ///     to be trusted.
    /// - Returns: `true` when every layer now reports `offset ==
    ///   targetOffset`; `false` otherwise (including an empty `cache`).
    @discardableResult
    nonisolated static func trimAndVerify(
        _ cache: [KVCache], from currentOffset: Int, to targetOffset: Int
    ) -> Bool {
        guard !cache.isEmpty else { return false }
        trimPromptCache(cache, numTokens: currentOffset - targetOffset)
        return cache.allSatisfy { $0.offset == targetOffset }
    }

    /// The hash-chain attachment point for a tail chunked off the END of
    /// `chunks` -- the last chunk's own `chunkKey`, or ``rootChunkKey`` when
    /// `chunks` is empty (no full chunks at all). Shared by ``resolve``
    /// (the matched-prefix chain endpoint) and ``store`` (the freshly
    /// sliced full-chunk chain endpoint), which otherwise each recompute
    /// this identical expression from their own `[StoredChunk]`.
    ///
    /// - Parameter chunks: The full-chunk chain, in chain order.
    /// - Returns: The `ChunkKey` a tail chained off `chunks`' end should use
    ///   as its `parentKey`.
    private func parentKeyOf(_ chunks: [StoredChunk]) -> ChunkKey {
        chunks.last?.chunkKey ?? Self.rootChunkKey
    }

    /// Resolves the `[KVCache]` to generate with and the tokens to
    /// actually feed it this round: walks `modelID`'s stored chunk chain
    /// against `newTokens` (``lookupLongestPrefix(modelID:newTokens:chunkSize:)``),
    /// assembles any matched chunks into a fresh, PRIVATE cache
    /// (``assemble(chunks:layerCount:)``), and feeds only the suffix past
    /// the matched prefix. No chunks match (including the first call for a
    /// model) ⇒ a freshly built cache, feeding every token.
    ///
    /// Nothing is checked out or removed here: chunks are read-only and
    /// every call gets its own assembled copy, so this cannot race a
    /// concurrent call for the same model -- see this actor's own doc
    /// comment on the chunk store's checkout-free design.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this cache is scoped to.
    ///   - newTokens: The freshly-tokenized full prompt for this round.
    ///   - model: The loaded model, for building a fresh cache when no
    ///     chunk has any overlap.
    ///   - parameters: Generation parameters, threaded into
    ///     `model.newCache(parameters:)` so a rebuilt cache matches what
    ///     the generation call would have created unassisted (e.g.
    ///     `maxKVSize`-driven rotating caches).
    /// - Returns: The `[KVCache]` to pass as generation's `cache:`
    ///   argument, and the token subsequence to actually feed (a suffix of
    ///   `newTokens`, or all of `newTokens` when nothing matched).
    ///
    /// After the full-chunk walk, also looks for a stored TAIL chunk chained
    /// from the matched prefix's endpoint (``lookupTailMatch(modelID:newTokens:consumedTokenCount:parentKey:remainingBudget:)``)
    /// and, if found, extends the match by its longest-common-prefix overlap
    /// with `newTokens`' remaining suffix -- this is what makes a
    /// conversation prefix SHORTER than one full chunk (previously cached as
    /// NOTHING at all) reusable, and what lets a growing conversation reuse
    /// everything but its newest tokens even between full-chunk boundaries.
    /// The combined full-chunk + tail match still never consumes the ENTIRE
    /// prompt: `lookupLongestPrefix`'s own cap already leaves >=1 token of
    /// budget, and the tail match is additionally capped by whatever budget
    /// remains after the full chunks (see `lookupTailMatch`'s
    /// `remainingBudget` parameter).
    func resolve(
        modelID: String, newTokens: [Int], model: SendableBox<any LanguageModel>,
        parameters: GenerateParameters?
    ) -> SendableBox<(cache: [KVCache], tokensToFeed: [Int])> {
        let model = model.consume()
        let freshCache = model.newCache(parameters: parameters)

        // Hybrid Mamba/attention stacks never satisfy `verifiedSimpleLayers`
        // (see `PromptCache.isChunkable`'s doc comment), so the chunk-store
        // walk below would always miss for them anyway -- check the
        // whole-stack checkpoint mechanism first and never fall through to
        // chunk logic for a genuinely hybrid stack.
        if PromptCache.isHybridMambaAttention(freshCache) {
            guard
                let checkpoint = resolveHybridCheckpoint(modelID: modelID, newTokens: newTokens)
                    .consume()
            else {
                return SendableBox((freshCache, newTokens))
            }
            let restored = PromptCache.restoreHybridCheckpoint(checkpoint)
            let suffix = Array(newTokens.suffix(newTokens.count - checkpoint.tokens.count))
            return SendableBox((restored, suffix))
        }

        let fullChunks = lookupLongestPrefix(
            modelID: modelID, newTokens: newTokens, chunkSize: chunkSize
        ).consume()
        let consumedByFullChunks = fullChunks.count * chunkSize
        let tailParentKey = parentKeyOf(fullChunks)
        let remainingBudget = (newTokens.count - 1) - consumedByFullChunks
        let tailMatch = lookupTailMatch(
            modelID: modelID, newTokens: newTokens, consumedTokenCount: consumedByFullChunks,
            parentKey: tailParentKey, remainingBudget: remainingBudget)

        guard let layerCount = (fullChunks.first ?? tailMatch?.chunk)?.layers.count else {
            return SendableBox((freshCache, newTokens))
        }

        let assembleChunks = tailMatch.map { fullChunks + [$0.chunk] } ?? fullChunks
        let matchedTokenCount = consumedByFullChunks + (tailMatch?.matchedLength ?? 0)
        let assembled = Self.assemble(
            chunks: assembleChunks, layerCount: layerCount,
            lastChunkMatchedLength: tailMatch?.matchedLength)
        return SendableBox(
            (assembled, Array(newTokens.suffix(newTokens.count - matchedTokenCount))))
    }

    /// Pure reconciliation between a re-encoded text's token count and a
    /// KV cache's authoritative `offset` advance -- no I/O, no actor
    /// state; unit-tested directly by `PromptCacheReconciliationTests`.
    ///
    /// Used by `Executor.commitPromptCache(...emittedText:...)` for the
    /// one remaining generation path that only yields decoded text, not
    /// raw token IDs (`generate(input:cache:parameters:context:...)`, used
    /// by `runUnconstrained`): that path reconstructs the generated token
    /// IDs by re-encoding the emitted text, and this function decides
    /// whether that reconstruction can be trusted for storage. Every
    /// other generation path threads real token IDs through directly (see
    /// `reconcileCacheAdvance`), so this is deliberately narrow in scope.
    ///
    /// Accepts an exact count match, a count exactly one *more* than
    /// `actualGeneratedCount`, or a count exactly one *fewer*.
    ///
    /// One more: `TokenIterator`'s next()-ahead prefetch design (see
    /// `MLXLMCommon/Evaluate.swift`) discards the terminal EOS/stop token
    /// without ever passing it to a stream consumer once that token has
    /// already been fed through the model, so the emitted text's full
    /// re-encoding legitimately lands one token ahead of the cache's real
    /// advance on that (common, successful) natural-stop exit path -- the
    /// trailing token is dropped before storing.
    ///
    /// One fewer: the model's actual final generated token can itself BE
    /// the EOS/stop token. That token advances `cache.offset` (contributing
    /// to `actualGeneratedCount`) but decodes to no text at all (special
    /// tokens produce empty content from the streaming detokenizer), so
    /// re-encoding `emittedText` recovers one FEWER token than the cache's
    /// real advance. Rather than fabricating a guessed token ID for the
    /// missing EOS position, this trusts the shorter `reencoded` array
    /// as-is: the caller (`Executor.commitPromptCache(...generatedTokenIDs:)`)
    /// independently calls `reconcileCacheAdvance`, which recognizes the
    /// resulting one-token gap (`cacheAdvance - observedTokenCount == 1`)
    /// as `.trimCacheByOne` and trims the cache back into sync via
    /// `trimAndVerify` -- so this function doesn't need to trim anything
    /// itself, it only needs to stop rejecting the reconstruction.
    ///
    /// This trusts the SHORTFALL is the trailing EOS position by count
    /// alone, not by verifying content -- a false positive (some other,
    /// non-EOS cause of a one-token-short re-encoding) would store a
    /// `trustedTokens` array that doesn't actually match the real token
    /// IDs the trimmed cache's tensors were produced from. That is caught
    /// downstream rather than left as silent corruption:
    /// `lookupLongestPrefix`'s and `lookupTailMatch`'s collision-safety
    /// checks (`chunk.tokens == window`, longest-common-prefix) always
    /// re-verify stored token content against a fresh, independent
    /// re-tokenization of the full transcript before ever reusing a
    /// chunk, so a wrongly-guessed entry simply stops matching at that
    /// boundary next round (a cache-reuse miss, i.e. falls back to
    /// rebuilding from there) rather than serving mismatched KV state --
    /// the same protection the existing `actualGeneratedCount + 1` case
    /// above already relies on.
    ///
    /// Any other mismatch is untrustworthy.
    ///
    /// - Parameters:
    ///   - reencoded: The re-encoded token IDs for the round's emitted text.
    ///   - actualGeneratedCount: `cache.offset` advance beyond the prompt
    ///     tokens fed this round -- the ground truth for how many tokens
    ///     the cache actually holds.
    /// - Returns: The token IDs to trust for storage, or `nil` when the
    ///   reconstruction can't be reconciled with `actualGeneratedCount`.
    nonisolated static func reconcileGeneratedTokens(
        reencoded: [Int], actualGeneratedCount: Int
    ) -> [Int]? {
        switch reencoded.count {
        case actualGeneratedCount:
            return reencoded
        case actualGeneratedCount + 1:
            return Array(reencoded.dropLast())
        case actualGeneratedCount - 1:
            return reencoded
        default:
            return nil
        }
    }

    /// How to reconcile a round's REAL generated token IDs (observed
    /// directly from a raw token stream -- no re-encoding involved)
    /// against the cache's own authoritative `offset` advance.
    enum CacheAdvanceReconciliation: Equatable {
        /// The observed token count matches the cache's advance exactly;
        /// store as-is.
        case matches
        /// The cache's real advance is exactly one MORE than the observed
        /// tokens -- trim the cache back by one token before storing, to
        /// bring its offset back in sync with the (trustworthy) observed
        /// tokens, rather than fabricating a value for the missing one.
        case trimCacheByOne
        /// Any other mismatch is untrustworthy; drop the entry.
        case untrustworthy
    }

    /// Pure reconciliation between a round's real (not re-encoded)
    /// generated token IDs and the cache's authoritative `offset` advance
    /// -- no I/O, no actor state; unit-tested directly.
    ///
    /// Two token-generation constructs in this adapter structurally
    /// discard their stream's terminal token without ever handing it to a
    /// consumer, even though that token's forward pass already advanced
    /// the cache: `TokenIterator`'s next()-ahead prefetch design silently
    /// drops the EOS/stop token that ends a round (`generateLoopTask`'s
    /// `iterator.discardGeneratedToken()` branch in
    /// `MLXLMCommon/Evaluate.swift`), used by `runReasoning` and
    /// `runToolCallReasoningPhase`'s raw token streams. In both known
    /// cases the gap is always exactly one token, always in the direction
    /// of the cache having *more* than what was observed -- never the
    /// reverse -- so any other delta is untrustworthy.
    ///
    /// - Parameters:
    ///   - observedTokenCount: How many real token IDs were observed
    ///     (e.g. `generatedTokenIDs.count`).
    ///   - cacheAdvance: The cache's real `offset` advance beyond the
    ///     prompt tokens fed this round.
    /// - Returns: How the caller should proceed.
    nonisolated static func reconcileCacheAdvance(
        observedTokenCount: Int, cacheAdvance: Int
    ) -> CacheAdvanceReconciliation {
        switch cacheAdvance - observedTokenCount {
        case 0: return .matches
        case 1: return .trimCacheByOne
        default: return .untrustworthy
        }
    }

    /// Slices this round's `(tokens, cache)` into fixed-size chunks
    /// (``PromptCache/sliceChunks(tokens:cache:chunkSize:)``) and checks
    /// them into `modelID`'s shared chunk store (``insert(modelID:chunks:)``),
    /// then does the same for the trailing partial TAIL span, if any
    /// (``PromptCache/sliceTailChunk(tokens:cache:chunkSize:parentKey:)``,
    /// checked in via ``insertTail(modelID:tail:)``) -- this is what makes a
    /// conversation prefix SHORTER than one full chunk reusable at all,
    /// rather than silently dropped.
    ///
    /// For a genuine hybrid Mamba/attention stack
    /// (`PromptCache.isHybridMambaAttention(_:)`), skips the chunk-slicing
    /// path entirely (it can never succeed against a `MambaCache` layer --
    /// see `PromptCache.isChunkable`'s doc comment) and instead snapshots a
    /// whole-stack ``PromptCache/HybridCheckpoint`` (`PromptCache.snapshotHybridCheckpoint(tokens:cache:)`)
    /// and checks it into ``hybridCheckpoints`` via
    /// ``insertHybridCheckpoint(modelID:checkpoint:)``.
    ///
    /// Silently drops the round when `cache`'s layers are neither a
    /// chunkable NOR a hybrid shape (see `sliceChunks`'s degradation for
    /// `RotatingKVCache`/`ChunkedKVCache`/an offset mismatch) -- the same
    /// degradation the old slot pool had for a non-trimmable cache: this
    /// round simply contributes nothing reusable, and the next round for
    /// this model rebuilds from scratch.
    ///
    /// Called once a generation call completes and its actual token count
    /// has been verified against `cache`'s own `offset` (see
    /// `Executor.commitPromptCache`).
    func store(modelID: String, tokens: [Int], cache: SendableBox<[KVCache]>) {
        let cacheValue = cache.consume()

        if PromptCache.isHybridMambaAttention(cacheValue) {
            if let checkpoint = PromptCache.snapshotHybridCheckpoint(
                tokens: tokens, cache: cacheValue)
            {
                insertHybridCheckpoint(modelID: modelID, checkpoint: checkpoint)
            }
            return
        }

        guard
            let chunks = Self.sliceChunks(tokens: tokens, cache: cacheValue, chunkSize: chunkSize)
        else { return }
        insert(modelID: modelID, chunks: chunks)

        let tailParentKey = parentKeyOf(chunks)
        if let tail = Self.sliceTailChunk(
            tokens: tokens, cache: cacheValue, chunkSize: chunkSize, parentKey: tailParentKey)
        {
            insertTail(modelID: modelID, tail: tail)
        }
    }

    /// Drops every model's remembered chunk store AND hybrid checkpoint
    /// store. Mirrors `ModelCache.evictAll`.
    func evictAll() {
        evictChunkStoreOnly()
        totalStoredBytes -= hybridCheckpoints.values.reduce(0) {
            $0 + $1.values.reduce(0) { $0 + $1.byteSize }
        }
        hybridCheckpoints.removeAll()
    }

    /// Drops every model's remembered CHUNK store -- ``chunkStore`` and
    /// ``activeTailKey`` -- reclaiming their bytes, while leaving
    /// ``hybridCheckpoints`` untouched. Extracted from ``evictAll()`` so
    /// ``setChunkSize(_:)`` (a genuine chunk-size change) can evict only the
    /// chunk-size-DEPENDENT store: hybrid checkpoints are keyed by the FULL
    /// token prefix they cover, not by `chunkSize`-aligned windows, so a
    /// `chunkSize` change never invalidates them -- see `setChunkSize(_:)`'s
    /// doc comment for the full invariant.
    private func evictChunkStoreOnly() {
        let chunkBytes = chunkStore.values.reduce(0) { total, entries in
            total + entries.values.reduce(0) { $0 + $1.byteSize }
        }
        totalStoredBytes -= chunkBytes
        chunkStore.removeAll()
        activeTailKey.removeAll()
    }

    /// Drops one model's remembered chunk store AND hybrid checkpoint
    /// store. Mirrors `ModelCache.remove`.
    func remove(modelID: String) {
        if removeAndReclaimBytes(modelID: modelID, from: &chunkStore) != nil {
            activeTailKey.removeValue(forKey: modelID)
        }
        removeAndReclaimBytes(modelID: modelID, from: &hybridCheckpoints)
    }

    /// Removes `modelID`'s entries from `store` (if any), reclaiming their
    /// combined ``StoredChunk/byteSize``/``HybridCheckpoint/byteSize`` from
    /// ``totalStoredBytes``. Shared by ``remove(modelID:)``'s two otherwise
    /// nearly-identical blocks (one per store), which previously each
    /// inlined the same remove-then-reduce-bytes pattern.
    ///
    /// - Parameters:
    ///   - modelID: The model whose entries to remove.
    ///   - store: The store to remove from (``chunkStore`` or ``hybridCheckpoints``).
    /// - Returns: The removed entries, or `nil` if nothing was stored for `modelID`.
    @discardableResult
    private func removeAndReclaimBytes<Entry: CacheStoreEntry>(
        modelID: String, from store: inout [String: [ChunkKey: Entry]]
    ) -> [ChunkKey: Entry]? {
        guard let removed = store.removeValue(forKey: modelID) else { return nil }
        totalStoredBytes -= removed.values.reduce(0) { $0 + $1.byteSize }
        return removed
    }

    /// Reconfigures this actor's total byte budget across EVERY model's
    /// stored chunks (see ``byteBudget``'s doc comment for the GLOBAL-scope
    /// rationale), clamping non-positive requests up to `1` -- the same
    /// `max(1, _)` pattern ``setChunkSize(_:)`` uses. Unlike
    /// ``defaultByteBudget()``'s ``minimumByteBudget`` floor, an explicit
    /// caller value is trusted as-is above `1`: a caller asking for a tiny
    /// budget means it (a memory-constrained scenario, or a test asserting
    /// eviction behavior under tight pressure), and clamping it up to a
    /// multi-megabyte floor would silently defeat that intent.
    ///
    /// Immediately runs ``evictToBudget()`` in case the new value is
    /// smaller than what's currently stored -- a caller lowering the
    /// budget expects eviction to happen right away, not lazily deferred
    /// to the next `insert(modelID:chunks:)`.
    ///
    /// - Parameter bytes: The requested total byte budget; clamped up to `1`.
    func setByteBudget(_ bytes: Int) {
        byteBudget = max(1, bytes)
        evictToBudget()
    }

    /// Reconfigures this actor's chunk span for every future `resolve()`/
    /// `store()` call, clamping non-positive requests up to `1`.
    ///
    /// Every stored chunk's ``ChunkKey`` chains over fixed-width,
    /// `chunkSize`-token windows (see `sliceChunks`/`lookupLongestPrefix`'s
    /// own doc comments): a chunk keyed under one span is not a valid
    /// chunk-aligned window under a DIFFERENT span, so a genuine change
    /// would otherwise let `lookupLongestPrefix` walk the OLD chain against
    /// NEW-span-aligned windows -- comparing windows that don't correspond
    /// to any real chunk boundary, either missing every match (best case)
    /// or, if a stale key ever collided with a new window's key by
    /// coincidence, serving the wrong KV state entirely (worst case, the
    /// same collision hazard `lookupLongestPrefix`'s own token compare
    /// guards against). Rather than risk either, a genuine change
    /// unconditionally calls ``evictChunkStoreOnly()``, discarding every
    /// model's stored CHUNKS so the next `resolve()`/`store()` rebuilds
    /// from scratch under the new span.
    ///
    /// Deliberately does NOT evict ``hybridCheckpoints``: unlike
    /// ``chunkStore``, hybrid checkpoints are keyed by the FULL token
    /// prefix a round covers (`chunkKey(parentKey: rootChunkKey, tokens:)`),
    /// not by `chunkSize`-aligned windows -- see
    /// ``PromptCache/HybridCheckpoint``'s doc comment. A `chunkSize` change
    /// doesn't change what a stored checkpoint's `tokens` prefix means, so
    /// evicting them here would needlessly discard reusable state that
    /// remains perfectly valid under the new span.
    ///
    /// Setting the SAME (already-clamped) value is a deliberate no-op past
    /// the clamp: comparing `clamped` against the CURRENT `chunkSize`
    /// before evicting means a caller that redundantly re-asserts the
    /// existing span (e.g. re-reading a config value on every request)
    /// does not pay a needless full-store eviction.
    ///
    /// - Parameter size: The requested chunk span in tokens; clamped to
    ///   `>= 1` (a chunk spanning zero or negative tokens is meaningless
    ///   and would divide-by-zero in `lookupLongestPrefix`'s chunk-count
    ///   math).
    func setChunkSize(_ size: Int) {
        let clamped = max(1, size)
        guard clamped != chunkSize else { return }
        chunkSize = clamped
        evictChunkStoreOnly()
    }

    // MARK: - Chunk store

    /// Checks `entry` into `store[modelID]` at `key`, deduplicating: an
    /// existing entry at `key` only has its `lastUsed` refreshed -- its
    /// captured content is NEVER replaced, since an existing entry's
    /// content is already a byte-for-byte-equivalent owned copy for the
    /// same key (dedup existing is the entire point: two conversations/
    /// rounds sharing that key must retain its state exactly once). A
    /// genuinely new key is stored fresh, stamped with the current
    /// recency, and its bytes newly accounted into ``totalStoredBytes``.
    ///
    /// Shared by ``insert(modelID:chunks:)``, ``insertTail(modelID:tail:)``,
    /// and ``insertHybridCheckpoint(modelID:checkpoint:)`` -- all three
    /// previously duplicated exactly this dedup/recency/byte-accounting
    /// pattern, differing only in which store and entry type they operate
    /// over.
    ///
    /// - Parameters:
    ///   - entry: The freshly captured entry to check in.
    ///   - key: The key to check `entry` in under.
    ///   - modelID: The model identifier this store is scoped to.
    ///   - store: The store to check `entry` into (``chunkStore`` or
    ///     ``hybridCheckpoints``).
    private func insertOrUpdateEntry<Entry: CacheStoreEntry>(
        _ entry: Entry, key: ChunkKey, modelID: String,
        into store: inout [String: [ChunkKey: Entry]]
    ) {
        var models = store[modelID] ?? [:]
        if var existing = models[key] {
            // Dedup hit: only `lastUsed` changes, content untouched --
            // `totalStoredBytes` is unaffected, since this key's bytes are
            // already accounted.
            existing.lastUsed = nextRecency()
            models[key] = existing
        } else {
            // Genuinely new key: store it fresh, stamped with the current
            // recency, and account its bytes.
            var fresh = entry
            fresh.lastUsed = nextRecency()
            models[key] = fresh
            totalStoredBytes += fresh.byteSize
        }
        store[modelID] = models
    }

    /// Checks freshly sliced chunks (see `PromptCache.sliceChunks`) into
    /// `modelID`'s chunk store, deduplicating by ``ChunkKey`` (see
    /// ``insertOrUpdateEntry(_:key:modelID:into:)``).
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this chunk store is scoped to.
    ///   - chunks: Freshly sliced chunks to check in, in chain order (as
    ///     `sliceChunks` returns them, though this method doesn't require
    ///     that order itself).
    func insert(modelID: String, chunks: [StoredChunk]) {
        for chunk in chunks {
            insertOrUpdateEntry(chunk, key: chunk.chunkKey, modelID: modelID, into: &chunkStore)
        }
        evictToBudget()
    }

    /// Checks a freshly sliced TAIL chunk (see
    /// `PromptCache.sliceTailChunk(tokens:cache:chunkSize:parentKey:)`) into
    /// `modelID`'s chunk store at its hash-chain attachment point
    /// (`tail.parentKey`), replacing whatever tail previously occupied that
    /// SAME position (see ``activeTailKey``'s doc comment for why only one
    /// tail is ever retained per attachment point).
    ///
    /// The dedup/recency/byte-accounting for the check-in itself is
    /// ``insertOrUpdateEntry(_:key:modelID:into:)`` (a tail whose key
    /// EXACTLY matches the position's current occupant is a true dedup hit
    /// there, mirroring ``insert(modelID:chunks:)``'s own full-chunk
    /// semantics); this method's own job is the STALE-occupant bookkeeping
    /// `insertOrUpdateEntry` doesn't know about: a tail with a DIFFERENT
    /// key (or the same key but already evicted by budget pressure, in
    /// which case removal is a harmless no-op) must have whatever it
    /// replaces reclaimed FIRST, since only one tail is ever retained per
    /// attachment point.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this chunk store is scoped to.
    ///   - tail: The freshly sliced tail chunk to check in.
    func insertTail(modelID: String, tail: StoredChunk) {
        var tailIndex = activeTailKey[modelID] ?? [:]

        if let existingKey = tailIndex[tail.parentKey], existingKey != tail.chunkKey,
            var models = chunkStore[modelID], let evicted = models.removeValue(forKey: existingKey)
        {
            // A stale tail (a different key than `tail.chunkKey` -- or the
            // same key but already evicted by budget pressure, in which
            // case this is a harmless no-op) previously occupied this
            // attachment point; only one tail is ever retained per position.
            totalStoredBytes -= evicted.byteSize
            chunkStore[modelID] = models
        }

        tailIndex[tail.parentKey] = tail.chunkKey
        activeTailKey[modelID] = tailIndex
        insertOrUpdateEntry(tail, key: tail.chunkKey, modelID: modelID, into: &chunkStore)
        evictToBudget()
    }

    /// Walks `modelID`'s chunk hash chain from the root over chunk-aligned
    /// windows of `newTokens`, returning the longest prefix of matching,
    /// verified chunks.
    ///
    /// CRITICAL: the walk is capped at `newTokens.count - 1` tokens, so at
    /// least one token always remains unmatched for the caller to feed --
    /// generation always needs >=1 fresh token to produce new logits, even
    /// when the entire prompt is otherwise covered by stored chunks (this is
    /// the chunk-store equivalent of the old (now-removed) slot pool's
    /// single-token-regeneration/n_past-- trick, but simpler: rather than
    /// trimming one token back off a full match, the walk itself never
    /// consumes the prompt's last token).
    ///
    /// COLLISION SAFETY: a key match only counts if the stored chunk's
    /// `tokens` are element-equal to the chunk-aligned window being matched
    /// (a cheap array compare, SGLang-style) -- a 64-bit `Hasher` collision
    /// would otherwise silently serve the wrong KV state, the worst possible
    /// cache failure. Any mismatch is treated as a miss and stops the walk
    /// immediately, exactly like a genuinely absent key.
    ///
    /// Every matched chunk's `lastUsed` is touched (bumped to the current
    /// recency) before being returned, for future LRU eviction bookkeeping.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this chunk store is scoped to.
    ///   - newTokens: The freshly-tokenized full prompt for this round.
    ///   - chunkSize: How many tokens each stored chunk covers (must match
    ///     the `chunkSize` chunks were originally sliced with).
    /// - Returns: The longest prefix of chunks whose chained keys and token
    ///   content both match `newTokens`, oldest-in-chain first, boxed for the
    ///   crossing back out of actor isolation (see `SendableBox`'s doc --
    ///   `StoredChunk`'s `MLXArray` tensors aren't `Sendable`, the same reason
    ///   `resolve`'s `[KVCache]` result is boxed); empty when even the first
    ///   chunk-aligned window misses or fails the collision check.
    func lookupLongestPrefix(
        modelID: String, newTokens: [Int], chunkSize: Int
    ) -> SendableBox<[StoredChunk]> {
        // Computed before the guard (rather than dividing unconditionally) so an
        // invalid chunkSize can't crash on integer division-by-zero, letting both
        // the invalid-chunkSize and empty-result cases share one early-exit guard.
        let maxChunkCount = chunkSize > 0 ? (newTokens.count - 1) / chunkSize : 0
        guard maxChunkCount > 0, var models = chunkStore[modelID] else { return SendableBox([]) }

        var parentKey = PromptCache.rootChunkKey
        var matched: [StoredChunk] = []
        matched.reserveCapacity(maxChunkCount)

        for index in 0 ..< maxChunkCount {
            let start = index * chunkSize
            let end = start + chunkSize
            let window = Array(newTokens[start ..< end])
            let key = PromptCache.chunkKey(parentKey: parentKey, tokens: window)

            guard var chunk = models[key], chunk.tokens == window else { break }

            chunk.lastUsed = nextRecency()
            models[key] = chunk
            matched.append(chunk)
            parentKey = key
        }

        chunkStore[modelID] = models
        return SendableBox(matched)
    }

    /// Finds the currently active TAIL chunk chained from `parentKey` (see
    /// ``activeTailKey``) for `modelID`, and computes how many of its
    /// leading tokens the suffix of `newTokens` starting at
    /// `consumedTokenCount` (the token count already matched by the full
    /// chunk walk) actually agrees with -- the longest common prefix
    /// between the tail's stored tokens and that suffix.
    ///
    /// The match is capped by `remainingBudget` (typically `newTokens.count
    /// - 1 - consumedTokenCount`, preserving `lookupLongestPrefix`'s own
    /// "at least one token always remains to feed" invariant across the
    /// FULL-chunk-walk-plus-tail-match combined) so the tail alone can never
    /// push the total matched count to cover the entire prompt.
    ///
    /// COLLISION SAFETY is inherent here, not a separate check: the matched
    /// length is derived from actual element-by-element token equality
    /// against the tail's own stored `tokens`, never from trusting a
    /// computed hash key alone -- mirroring `lookupLongestPrefix`'s own
    /// token-array compare.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this chunk store is scoped to.
    ///   - newTokens: The freshly-tokenized full prompt for this round.
    ///   - consumedTokenCount: How many of `newTokens`' leading tokens the
    ///     full-chunk walk already matched -- the offset the tail's
    ///     candidate suffix starts at.
    ///   - parentKey: The full-chunk walk's endpoint (the last matched
    ///     chunk's `chunkKey`, or `PromptCache.rootChunkKey` when no full
    ///     chunk matched) -- the tail's hash-chain attachment point.
    ///   - remainingBudget: The maximum number of additional tokens the tail
    ///     match may cover, on top of `consumedTokenCount`.
    /// - Returns: The matched tail chunk and how many of its leading tokens
    ///   matched (between `1` and `min(tail.tokens.count, remainingBudget)`),
    ///   or `nil` when no tail is chained at `parentKey`, `remainingBudget`
    ///   is `<= 0`, or the very first token already diverges.
    private func lookupTailMatch(
        modelID: String, newTokens: [Int], consumedTokenCount: Int, parentKey: ChunkKey,
        remainingBudget: Int
    ) -> (chunk: StoredChunk, matchedLength: Int)? {
        guard remainingBudget > 0,
            var models = chunkStore[modelID],
            let tailKey = activeTailKey[modelID]?[parentKey],
            var tail = models[tailKey]
        else { return nil }

        let cap = min(tail.tokens.count, remainingBudget)
        var matchedLength = 0
        while matchedLength < cap,
            tail.tokens[matchedLength] == newTokens[consumedTokenCount + matchedLength]
        {
            matchedLength += 1
        }
        guard matchedLength > 0 else { return nil }

        tail.lastUsed = nextRecency()
        models[tailKey] = tail
        chunkStore[modelID] = models
        return (chunk: tail, matchedLength: matchedLength)
    }

    // MARK: - Hybrid checkpoint store

    /// Checks a freshly captured whole-stack hybrid checkpoint (see
    /// `PromptCache.snapshotHybridCheckpoint(tokens:cache:)`) into
    /// `modelID`'s ``hybridCheckpoints`` store, deduplicating by content-hash
    /// key via ``insertOrUpdateEntry(_:key:modelID:into:)`` -- exactly the
    /// same dedup/recency/byte-accounting semantics
    /// ``insert(modelID:chunks:)`` uses for ordinary chunks.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this checkpoint store is scoped to.
    ///   - checkpoint: The freshly captured checkpoint to check in.
    func insertHybridCheckpoint(modelID: String, checkpoint: HybridCheckpoint) {
        let key = Self.chunkKey(parentKey: Self.rootChunkKey, tokens: checkpoint.tokens)
        insertOrUpdateEntry(checkpoint, key: key, modelID: modelID, into: &hybridCheckpoints)
        evictToBudget()
    }

    /// Finds `modelID`'s LONGEST stored hybrid checkpoint whose full token
    /// prefix exactly matches a leading prefix of `newTokens`, leaving at
    /// least one token of `newTokens` unmatched (mirroring
    /// `lookupLongestPrefix`'s own "at least one token always remains to
    /// feed" invariant -- generation always needs >=1 fresh token to
    /// produce new logits).
    ///
    /// A flat linear scan, not a hash-chain walk: hybrid checkpoints have
    /// no parent/child chain (see ``hybridCheckpoints``'s doc comment), so
    /// every stored checkpoint for `modelID` is a candidate, and the winner
    /// is whichever candidate's `tokens` is both a genuine prefix of
    /// `newTokens` (COLLISION SAFETY: verified by actual element-by-element
    /// array equality, never by trusting the hash key alone -- mirroring
    /// `lookupLongestPrefix`'s own `chunk.tokens == window` check) and the
    /// LONGEST such prefix among every candidate.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this checkpoint store is scoped to.
    ///   - newTokens: The freshly-tokenized full prompt for this round.
    /// - Returns: The longest matching stored checkpoint, with its
    ///   `lastUsed` bumped for LRU bookkeeping, boxed for the crossing back
    ///   out of actor isolation (see `SendableBox`'s doc -- `HybridCheckpoint`
    ///   carries non-`Sendable` `MLXArray` tensors, the same reason
    ///   `lookupLongestPrefix`'s `[StoredChunk]` result is boxed); `nil`
    ///   (inside the box) when nothing stored for `modelID` is a valid
    ///   prefix of `newTokens`.
    func resolveHybridCheckpoint(
        modelID: String, newTokens: [Int]
    ) -> SendableBox<HybridCheckpoint?> {
        guard var checkpoints = hybridCheckpoints[modelID], !checkpoints.isEmpty else {
            return SendableBox(nil)
        }

        var winnerKey: ChunkKey?
        var winner: HybridCheckpoint?
        for (key, checkpoint) in checkpoints {
            let candidateLength = checkpoint.tokens.count
            let newLength = newTokens.count
            let currentBestLength = winner?.tokens.count ?? -1
            guard candidateLength < newLength,
                candidateLength > currentBestLength,
                Array(newTokens.prefix(candidateLength)) == checkpoint.tokens
            else { continue }
            winner = checkpoint
            winnerKey = key
        }

        guard var matched = winner, let key = winnerKey else { return SendableBox(nil) }
        matched.lastUsed = nextRecency()
        checkpoints[key] = matched
        hybridCheckpoints[modelID] = checkpoints
        return SendableBox(matched)
    }

    /// Number of distinct hybrid checkpoints currently stored for `modelID`
    /// -- exposes checkpoint-store bookkeeping for tests, mirroring
    /// ``chunkCount(modelID:)`` for the ordinary chunk store.
    func hybridCheckpointCount(modelID: String) -> Int {
        hybridCheckpoints[modelID]?.count ?? 0
    }

    /// Number of distinct chunks currently stored for `modelID` -- exposes
    /// dedup/eviction bookkeeping for tests without leaking the actor's
    /// private storage shape.
    func chunkCount(modelID: String) -> Int {
        chunkStore[modelID]?.count ?? 0
    }

    /// Total bytes currently accounted across every model's chunk store --
    /// exposes byte-budget bookkeeping for tests without leaking the
    /// actor's private storage shape (mirrors ``chunkCount(modelID:)``).
    func totalStoredByteCount() -> Int {
        totalStoredBytes
    }

    /// This actor's currently configured byte budget -- test seam mirroring
    /// ``chunkCount(modelID:)``/``totalStoredByteCount()``.
    func currentByteBudget() -> Int {
        byteBudget
    }

    // MARK: - Byte-budget LRU eviction

    /// Which store a globally-least-recently-used eviction candidate
    /// (``globalLRUVictim()``) belongs to -- the ordinary chunk store's
    /// hash-chained ``StoredChunk``s, or the flat, unchained
    /// ``hybridCheckpoints`` store.
    private enum LRUVictim {
        case chunk(modelID: String, key: ChunkKey)
        case hybridCheckpoint(modelID: String, key: ChunkKey)
    }

    /// The globally least-recently-used entry across EVERY model's chunk
    /// store AND hybrid checkpoint store -- the eviction candidate
    /// ``evictToBudget()`` removes first -- or `nil` when both stores are
    /// empty for every model.
    ///
    /// Scans every model's `lastUsed` directly rather than maintaining a
    /// separate ordered LRU index: eviction only runs when an insert pushes
    /// ``totalStoredBytes`` over ``byteBudget``, and even a long-running
    /// process's combined store stays small relative to a linear scan's cost
    /// (bounded by how many distinct token-prefix chunks/checkpoints are
    /// actually resident, not by request volume).
    ///
    /// - Returns: The globally least-recently-used entry, tagged by which
    ///   store it belongs to, or `nil` when both stores are empty.
    private func globalLRUVictim() -> LRUVictim? {
        var best: (victim: LRUVictim, lastUsed: Int)?
        updateBest(&best, scanning: chunkStore) { .chunk(modelID: $0, key: $1) }
        updateBest(&best, scanning: hybridCheckpoints) { .hybridCheckpoint(modelID: $0, key: $1) }
        return best?.victim
    }

    /// Scans `store` for its least-recently-used entry and folds it into
    /// `best` if it beats the current candidate (or if `best` is still
    /// `nil`) -- the identical nested-loop shape ``globalLRUVictim()``
    /// previously repeated once per store (``chunkStore``,
    /// ``hybridCheckpoints``), differing only in which ``LRUVictim`` case
    /// wraps the winning `(modelID, key)` pair.
    ///
    /// - Parameters:
    ///   - best: The best candidate found so far, updated in place.
    ///   - store: The store to scan.
    ///   - victimCase: Wraps a winning `(modelID, key)` pair into the
    ///     ``LRUVictim`` case matching `store`.
    private func updateBest<Entry: CacheStoreEntry>(
        _ best: inout (victim: LRUVictim, lastUsed: Int)?,
        scanning store: [String: [ChunkKey: Entry]],
        victimCase: (String, ChunkKey) -> LRUVictim
    ) {
        for (modelID, entries) in store {
            for (key, entry) in entries {
                if best == nil || entry.lastUsed < best!.lastUsed {
                    best = (victimCase(modelID, key), entry.lastUsed)
                }
            }
        }
    }

    /// Removes `key` from `modelID`'s store, reclaiming its bytes, then
    /// recursively evicts every chunk chained under it (any chunk whose
    /// `parentKey == key`) so a lineage's WHOLE byte footprint is reclaimed
    /// in one pass -- not just the directly-evicted chunk's own bytes.
    ///
    /// ORPHAN-RECLAMATION STRATEGY: evicting a chunk always orphans its
    /// chain descendants -- ``lookupLongestPrefix(modelID:newTokens:chunkSize:)``
    /// already stops its walk at the first missing parent, so a descendant
    /// whose ancestor is gone is unreachable from any future lookup, even
    /// though it's still sitting in `chunkStore` taking up bytes. This
    /// actor reclaims that whole orphaned lineage TRANSITIVELY, right here,
    /// rather than leaving orphans in the store for a later lazy sweep: a
    /// lazy sweep would need its own pass to find unreachable chunks (no
    /// cheaper than this recursive walk) while leaving ``totalStoredBytes``
    /// overcounting reachable-only memory in the meantime -- which would
    /// let ``evictToBudget()`` under-evict, stopping as soon as the RAW
    /// total dropped under budget even though part of that total is
    /// orphaned dead weight that can never be reached again.
    ///
    /// - Parameters:
    ///   - modelID: The model whose store to evict from.
    ///   - key: The chunk key to remove, along with everything chained under it.
    private func evictChunkAndDescendants(modelID: String, key: ChunkKey) {
        guard var models = chunkStore[modelID], let evicted = models.removeValue(forKey: key)
        else { return }
        totalStoredBytes -= evicted.byteSize
        if activeTailKey[modelID]?[evicted.parentKey] == key {
            // The evicted entry was itself the active tail at its
            // attachment point -- clear the index so a future `insertTail`
            // doesn't dedup-compare against, or redundantly try to re-evict,
            // a key no longer in the store.
            activeTailKey[modelID]?[evicted.parentKey] = nil
        }
        let childKeys = models.filter { $0.value.parentKey == key }.map(\.key)
        chunkStore[modelID] = models
        for childKey in childKeys {
            evictChunkAndDescendants(modelID: modelID, key: childKey)
        }
    }

    /// Removes `key` from `modelID`'s hybrid checkpoint store, reclaiming
    /// its bytes. Unlike ``evictChunkAndDescendants(modelID:key:)``, a
    /// hybrid checkpoint has no chain descendants to reclaim transitively
    /// (see ``hybridCheckpoints``'s doc comment) -- each entry is an
    /// independent whole-stack snapshot, so removing one never orphans
    /// another.
    ///
    /// - Parameters:
    ///   - modelID: The model whose checkpoint store to evict from.
    ///   - key: The checkpoint key to remove.
    private func evictHybridCheckpoint(modelID: String, key: ChunkKey) {
        guard var checkpoints = hybridCheckpoints[modelID],
            let evicted = checkpoints.removeValue(forKey: key)
        else { return }
        totalStoredBytes -= evicted.byteSize
        hybridCheckpoints[modelID] = checkpoints
    }

    /// Evicts the globally least-recently-used entry across BOTH the chunk
    /// store and the hybrid checkpoint store -- transitively, its whole
    /// lineage when it's a chunk (``evictChunkAndDescendants(modelID:key:)``),
    /// or just itself when it's a hybrid checkpoint
    /// (``evictHybridCheckpoint(modelID:key:)``) -- repeatedly until
    /// ``totalStoredBytes`` is at or under ``byteBudget``, or both stores
    /// are empty for every model.
    ///
    /// Called after every ``insert(modelID:chunks:)``/``insertTail(modelID:tail:)``/
    /// ``insertHybridCheckpoint(modelID:checkpoint:)`` and from
    /// ``setByteBudget(_:)`` whenever the budget shrinks below what's
    /// currently stored.
    private func evictToBudget() {
        while totalStoredBytes > byteBudget, let victim = globalLRUVictim() {
            switch victim {
            case .chunk(let modelID, let key):
                evictChunkAndDescendants(modelID: modelID, key: key)
            case .hybridCheckpoint(let modelID, let key):
                evictHybridCheckpoint(modelID: modelID, key: key)
            }
        }
    }

    /// Builds a fresh, PRIVATE `[KVCache]` stack from `chunks` (the result of
    /// ``lookupLongestPrefix(modelID:newTokens:chunkSize:)``), one
    /// `KVCacheSimple` per layer: for each layer, concatenates every chunk's
    /// key/value slices in chain order and installs the result via a fresh
    /// `KVCacheSimple`'s `state` setter (which derives `offset` from
    /// `keys.dim(2)`, so the assembled cache's offset lands exactly on the
    /// matched token count -- see `KVCacheSimple.state`'s setter in
    /// `Libraries/MLXLMCommon/KVCache.swift`).
    ///
    /// The result is PRIVATE: `chunks`' tensors are immutable, owned copies
    /// (see `StoredChunk`/`ownedCopy(of:)` in `PromptCacheChunks.swift`), and
    /// every call to `assemble` produces its own independent tensors (see
    /// below) -- so there is no checkout, no steal, and no double-checkout
    /// hazard by construction, unlike the old slot pool's `resolve`/`store`
    /// check-out-and-return dance.
    ///
    /// `concatenated(_:axis:)` alone is NOT sufficient to guarantee
    /// independence: `mlx::core::concatenate` special-cases a SINGLE input
    /// array by returning that exact array unchanged (`mlx/ops.cpp`:
    /// `if (arrays.size() == 1) { return arrays[0]; }`), never allocating a
    /// fresh buffer or invoking the `Concatenate` primitive at all in that
    /// case -- so when a layer has exactly one matched chunk, a bare
    /// `concatenated([chunk.keys], axis: 2)` would hand back the chunk
    /// store's own tensor BY REFERENCE. (Two or more inputs always DO get a
    /// fresh buffer: `Concatenate::eval_cpu`/`eval_gpu` unconditionally
    /// `malloc`s a new output buffer and copies every input into it -- see
    /// `mlx/backend/{cpu,metal}/...`.) Routing every layer's result through
    /// ``ownedCopy(of:)`` (shared with `sliceChunks`) closes that gap
    /// unconditionally, regardless of how many chunks matched.
    ///
    /// Called directly by `resolve()` to build the `[KVCache]` a matched
    /// chunk prefix seeds generation with.
    ///
    /// - Parameters:
    ///   - chunks: The longest matching chunk prefix (from
    ///     `lookupLongestPrefix`), in chain order (oldest first). Empty
    ///     yields an empty `[KVCache]` -- callers should feed everything
    ///     against a freshly built cache in that case (the existing
    ///     `model.newCache(parameters:)` fallback), since `assemble` has no
    ///     `model`/`parameters` to build one itself.
    ///   - layerCount: How many `KVCacheSimple` layers to build -- every
    ///     `StoredChunk` in `chunks` must have exactly this many `layers`, or
    ///     assembly is refused (see below); mirrors `sliceChunks`'s own
    ///     degrade-to-`nil` convention for a shape it can't safely reason
    ///     about, rather than trapping on an out-of-range `layers` index.
    ///   - lastChunkMatchedLength: When the LAST chunk in `chunks` is a
    ///     partially-matched TAIL (see `PromptCacheChunks.sliceTailChunk`),
    ///     how many of its leading tokens actually matched (from
    ///     `lookupTailMatch`'s longest-common-prefix walk) -- that chunk's
    ///     tensors are sliced down to this length (axis 2, the sequence
    ///     axis) BEFORE concatenation. `nil` when every chunk in `chunks` is
    ///     a full, wholly-matched chunk (the common case).
    /// - Returns: One freshly assembled `KVCacheSimple` per layer, offset at
    ///   the matched token count; empty when `chunks` is empty, any
    ///   chunk's `layers.count` doesn't match `layerCount`, or
    ///   `lastChunkMatchedLength` is out of the last chunk's token range.
    nonisolated static func assemble(
        chunks: [StoredChunk], layerCount: Int, lastChunkMatchedLength: Int? = nil
    ) -> [KVCache] {
        guard !chunks.isEmpty, layerCount > 0,
            chunks.allSatisfy({ $0.layers.count == layerCount })
        else { return [] }
        let lastIndex = chunks.count - 1
        if let lastChunkMatchedLength {
            guard lastChunkMatchedLength > 0, lastChunkMatchedLength <= chunks[lastIndex].tokens.count
            else { return [] }
        }

        var result: [KVCache] = []
        result.reserveCapacity(layerCount)
        for layerIndex in 0 ..< layerCount {
            let keyParts = chunks.enumerated().map { index, chunk in
                slicedToMatchedLength(
                    chunk.layers[layerIndex].keys,
                    matchedLength: index == lastIndex ? lastChunkMatchedLength : nil)
            }
            let valueParts = chunks.enumerated().map { index, chunk in
                slicedToMatchedLength(
                    chunk.layers[layerIndex].values,
                    matchedLength: index == lastIndex ? lastChunkMatchedLength : nil)
            }
            let keys = ownedCopy(of: concatenated(keyParts, axis: 2))
            let values = ownedCopy(of: concatenated(valueParts, axis: 2))

            let layerCache = KVCacheSimple()
            layerCache.state = [keys, values]
            result.append(layerCache)
        }
        return result
    }

    /// Slices `array` down to its first `matchedLength` sequence positions
    /// (axis 2) when a partial TAIL match requires it, leaving `array`
    /// unchanged otherwise.
    ///
    /// The slice itself aliases `array`'s underlying buffer (see
    /// ``ownedCopy(of:)``'s doc comment on `Slice`/`copy_shared_buffer`) --
    /// harmless here because `assemble` always routes the final concatenated
    /// result back through ``ownedCopy(of:)`` before returning, exactly as
    /// it already does for the no-slice path.
    ///
    /// - Parameters:
    ///   - array: The chunk layer's key or value tensor.
    ///   - matchedLength: How many leading sequence positions actually
    ///     matched, or `nil` when this isn't the partially-matched last chunk.
    /// - Returns: `array` sliced to `0..<matchedLength`, or `array` itself
    ///   when `matchedLength` is `nil`.
    private static func slicedToMatchedLength(_ array: MLXArray, matchedLength: Int?) -> MLXArray {
        guard let matchedLength else { return array }
        return array[.ellipsis, 0 ..< matchedLength, 0...]
    }
}

#endif
#endif
