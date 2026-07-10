// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

/// Per-model, multi-slot KV-cache reuse for `MLXLanguageModel.Executor`.
///
/// FoundationModels' `LanguageModelExecutor` protocol has no session
/// identity: `Executor.respond(to:model:streamingInto:)` receives the
/// *full* transcript on every round with no indication of which prior
/// round (if any) it continues, or whether this is even the same session
/// as last time. `PromptCache` keys reuse on the token prefix itself
/// instead: it remembers, per model, up to a handful of recent
/// `(token sequence, [KVCache])` slots, and on each round compares the
/// freshly-tokenized prompt against every remembered slot's tokens --
/// picking the one with the longest common prefix (LCP) -- to decide
/// whether to reuse that slot's cache unchanged, trim it back to a common
/// prefix, or (if no slot has any useful overlap) rebuild from scratch.
///
/// Modeled on llama.cpp server's slot pool (`--parallel N`, LCP-based slot
/// selection with an LRU fallback) scaled down for a single-process client
/// library rather than a many-concurrent-client server; mirrors
/// `ModelCache`'s per-model-actor-cache pattern in `MLXLanguageModel.swift`
/// otherwise: a private actor, instantiated once as a process-global
/// `static let`, exposed to `Executor` through thin passthrough statics on
/// `MLXLanguageModel`.
actor PromptCache {

    /// Default for ``maxSlotsPerModel``: how many remembered slots are
    /// kept per model unless the host app configures otherwise (see
    /// `MLXLanguageModel.setPromptCacheSlotLimit(_:)`). Bounds per-model
    /// memory growth under multi-session usage (multiple
    /// concurrent/alternating conversations against the same model) while
    /// still giving several recent conversations a chance at KV-cache
    /// reuse. Matches this being a single-process client library serving
    /// a handful of concurrent sessions, not llama.cpp server's
    /// many-concurrent-client `--parallel N` (commonly much larger).
    /// There is no session-count signal to derive this from:
    /// FoundationModels never tells the executor how many
    /// `LanguageModelSession`s exist, so capacity must be declared, not
    /// inferred.
    static let defaultMaxSlotsPerModel = 4

    /// Maximum number of remembered slots kept per model; consulted by
    /// `store`'s eviction. Actor-isolated mutable state, adjusted via
    /// ``setMaxSlotsPerModel(_:)``.
    private var maxSlotsPerModel = PromptCache.defaultMaxSlotsPerModel

    /// Sets how many conversations can retain KV state per model.
    ///
    /// Clamped to at least 1 (a zero/negative limit would make `store` a
    /// no-op and silently disable caching -- an explicit `evictAll()` is
    /// the way to opt out). Takes effect on subsequent `store` calls; an
    /// already-over-limit slot list shrinks the next time that model
    /// stores a slot.
    func setMaxSlotsPerModel(_ limit: Int) {
        maxSlotsPerModel = max(1, limit)
    }

    /// One remembered `(token sequence, KV cache)` slot, plus a
    /// monotonically increasing recency stamp.
    ///
    /// `lastUsed` is a plain counter, not a wall-clock timestamp: recency
    /// here only needs *relative* ordering (which slot is newer), and the
    /// actor's serialized isolation already guarantees the counter
    /// strictly increases across calls -- a counter sidesteps any
    /// wall-clock testability concern entirely (fake/frozen clocks,
    /// `ContinuousClock` resolution in test contexts, etc.) without
    /// inventing a new injectable-clock pattern this codebase doesn't
    /// otherwise have.
    private struct Slot {
        var tokens: [Int]
        var cache: [KVCache]
        var lastUsed: Int
    }

    /// Remembered slots per model, newest-recency-stamp last is NOT
    /// guaranteed (slots are only appended, never re-sorted on read) --
    /// use `lastUsed` for recency, not array position.
    private var entries: [String: [Slot]] = [:]

    /// Per-model chunk store: each model's chunks keyed by their own
    /// ``ChunkKey`` (see `PromptCacheChunks.swift`'s hash chain), ADDITIVE to
    /// (not a replacement of) ``entries``'s slot pool -- `resolve()`/`store()`
    /// still serve `MLXLanguageModel.swift`'s `Executor` in production today;
    /// nothing yet wires this store's ``insert(modelID:chunks:)``/
    /// ``lookupLongestPrefix(modelID:newTokens:chunkSize:)`` into cache
    /// reconstruction (a later "Assembly" task does that). See
    /// ``insert(modelID:chunks:)`` for dedup semantics.
    private var chunkStore: [String: [ChunkKey: StoredChunk]] = [:]

    /// Monotonic source for `Slot.lastUsed`/`StoredChunk.lastUsed`. Actor
    /// isolation serializes every read/increment, so this is a strict
    /// per-call counter with no concurrent-access races.
    private var recencyCounter = 0

    private func nextRecency() -> Int {
        recencyCounter += 1
        return recencyCounter
    }

    /// How to seed a generation call's `[KVCache]` and which tokens to
    /// actually feed it, given what (if anything) is remembered for a
    /// model and the freshly-tokenized prompt for this round.
    enum Decision: Equatable {
        /// The remembered tokens are a prefix of the new tokens: reuse the
        /// cache unchanged and feed only the trailing `count` new tokens.
        case reuseSuffix(count: Int)
        /// The sequences diverge after `commonPrefixLength` tokens, and
        /// the cache supports trimming back to that point; feed
        /// `thenSuffix` tokens (the new tokens beyond the common prefix)
        /// after trimming.
        case trimTo(commonPrefixLength: Int, thenSuffix: Int)
        /// No usable overlap (or the cache can't be trimmed to one):
        /// rebuild a fresh cache and feed every new token.
        case rebuild
    }

    /// Pure prefix-match/trim/rebuild decision. No I/O, no actor state --
    /// unit-tested directly by `PromptCacheTests`, independent of any
    /// cache/actor machinery. Operates on a single `(cachedTokens, cache)`
    /// candidate; `selectSlot` is the layer above this that picks *which*
    /// of a model's several remembered slots to run this against.
    ///
    /// - Parameters:
    ///   - cachedTokens: The token sequence the KV cache currently
    ///     reflects; empty when nothing has been cached for this model yet.
    ///   - newTokens: The freshly-tokenized prompt for this round.
    ///   - isTrimmable: Whether every cache in the `[KVCache]` array
    ///     supports `trim(_:)` (see `canTrimPromptCache`).
    /// - Returns: How to seed generation for this round.
    nonisolated static func decide(
        cachedTokens: [Int], newTokens: [Int], isTrimmable: Bool
    ) -> Decision {
        guard !cachedTokens.isEmpty else { return .rebuild }

        if newTokens.count >= cachedTokens.count,
            Array(newTokens.prefix(cachedTokens.count)) == cachedTokens
        {
            return .reuseSuffix(count: newTokens.count - cachedTokens.count)
        }

        let commonPrefixLength = zip(cachedTokens, newTokens)
            .prefix { $0 == $1 }
            .count
        guard isTrimmable else { return .rebuild }
        return .trimTo(
            commonPrefixLength: commonPrefixLength,
            thenSuffix: newTokens.count - commonPrefixLength)
    }

    /// Picks the best of a model's remembered slots to check out for
    /// `newTokens`, by longest-common-prefix (LCP) length against each
    /// candidate's tokens -- ties go to the most recently used
    /// (higher `lastUsed`).
    ///
    /// Returns `nil` when every candidate's LCP is zero: no candidate has
    /// any usable overlap with `newTokens`, so checking one out would be
    /// strictly worse than a plain rebuild (it would immediately hit
    /// `decide`'s `commonPrefixLength == 0` trim-to-nothing case, which is
    /// no cheaper than rebuilding and needlessly evicts a slot that might
    /// still be useful for a *different* incoming prompt).
    ///
    /// - Parameters:
    ///   - candidates: Each remembered slot's tokens and recency stamp,
    ///     in the same order as the caller's slot storage (so the
    ///     returned index can be used to remove the chosen slot).
    ///   - newTokens: The freshly-tokenized prompt for this round.
    /// - Returns: The index into `candidates` of the best slot, or `nil`
    ///   if none has any overlap.
    nonisolated static func selectSlot(
        candidates: [(tokens: [Int], lastUsed: Int)], newTokens: [Int]
    ) -> Int? {
        var bestIndex: Int?
        var bestLCP = 0
        var bestLastUsed = Int.min
        for (index, candidate) in candidates.enumerated() {
            let lcp = zip(candidate.tokens, newTokens).prefix { $0 == $1 }.count
            guard lcp > 0 else { continue }
            guard lcp > bestLCP || (lcp == bestLCP && candidate.lastUsed > bestLastUsed) else {
                continue
            }
            bestIndex = index
            bestLCP = lcp
            bestLastUsed = candidate.lastUsed
        }
        return bestIndex
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

    /// llama.cpp server's "n_past--" trick: when the incoming prompt is
    /// fully covered by a slot's cache up to `commonPrefixLength` -- a
    /// full-prefix reuse with nothing new to feed (`decide`'s
    /// `.reuseSuffix(count: 0)`), or a trim that leaves no new suffix
    /// (`.trimTo(_, thenSuffix: 0)`) -- generation still needs at least
    /// one token fed through the model to produce fresh logits. Rather
    /// than rebuild the whole cache from scratch, trim back one token
    /// short of `commonPrefixLength` and feed exactly that one token
    /// again: a single-token forward pass instead of a full re-prefill.
    ///
    /// Falls back to a rebuild when the cache isn't trimmable, when there
    /// is nothing to trim back to (`commonPrefixLength == 0`), or when
    /// the trim doesn't verify (see `trimAndVerify`).
    ///
    /// - Parameters:
    ///   - cache: The candidate slot's cache.
    ///   - cachedTokenCount: The candidate slot's token count (the
    ///     cache's assumed current offset).
    ///   - commonPrefixLength: How much of the candidate's tokens match
    ///     `newTokens` (see `decide`).
    ///   - newTokens: The freshly-tokenized prompt for this round.
    ///   - model: The loaded model, for building a fresh cache on rebuild.
    ///   - parameters: Generation parameters for a rebuilt cache.
    /// - Returns: The `[KVCache]` to generate with and the one token to feed.
    nonisolated private static func regenerateLastToken(
        cache: [KVCache], cachedTokenCount: Int, commonPrefixLength: Int,
        newTokens: [Int], model: any LanguageModel, parameters: GenerateParameters?
    ) -> (cache: [KVCache], tokensToFeed: [Int]) {
        guard commonPrefixLength > 0, canTrimPromptCache(cache),
            trimAndVerify(cache, from: cachedTokenCount, to: commonPrefixLength - 1)
        else {
            return (model.newCache(parameters: parameters), newTokens)
        }
        return (cache, [newTokens[commonPrefixLength - 1]])
    }

    /// Applies `decide`'s outcome against one checked-out slot: the
    /// shared control flow behind `resolve`, factored out so the
    /// trim-verification (see `trimAndVerify`) and n_past-- (see
    /// `regenerateLastToken`) logic each have one call site.
    nonisolated private static func applyDecision(
        _ decision: Decision, slot: Slot, newTokens: [Int],
        model: any LanguageModel, parameters: GenerateParameters?
    ) -> (cache: [KVCache], tokensToFeed: [Int]) {
        switch decision {
        case .reuseSuffix(let count) where count > 0:
            return (slot.cache, Array(newTokens.suffix(count)))

        case .reuseSuffix:
            // count == 0: full match, nothing new to feed.
            return regenerateLastToken(
                cache: slot.cache, cachedTokenCount: slot.tokens.count,
                commonPrefixLength: slot.tokens.count, newTokens: newTokens,
                model: model, parameters: parameters)

        case .trimTo(let commonPrefixLength, let thenSuffix) where thenSuffix > 0:
            guard
                trimAndVerify(
                    slot.cache, from: slot.tokens.count, to: commonPrefixLength)
            else {
                return (model.newCache(parameters: parameters), newTokens)
            }
            return (slot.cache, Array(newTokens.suffix(thenSuffix)))

        case .trimTo(let commonPrefixLength, _):
            // thenSuffix == 0: the new prompt is fully covered by the
            // slot's cache (a shrunk/edited-earlier prompt), nothing new
            // to feed.
            return regenerateLastToken(
                cache: slot.cache, cachedTokenCount: slot.tokens.count,
                commonPrefixLength: commonPrefixLength, newTokens: newTokens,
                model: model, parameters: parameters)

        case .rebuild:
            return (model.newCache(parameters: parameters), newTokens)
        }
    }

    /// Resolves the `[KVCache]` to generate with and the tokens to
    /// actually feed it this round, selecting the best-matching remembered
    /// slot for `modelID` (see `selectSlot`) and applying `decide`'s
    /// outcome against it (see `applyDecision`).
    ///
    /// Removes (rather than merely reads) the *selected* slot from
    /// `modelID`'s slot list: `KVCache` instances are plain classes with
    /// no internal synchronization, so handing the same instances to two
    /// concurrent generation calls for the same model would race.
    /// Removing here means a second concurrent call either selects a
    /// *different* slot (if one has useful overlap for its own prompt) or
    /// finds nothing and safely rebuilds; `store` checks a (possibly
    /// different, newer) slot back in once generation completes.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this cache is scoped to.
    ///   - newTokens: The freshly-tokenized full prompt for this round.
    ///   - model: The loaded model, for building a fresh cache on rebuild.
    ///   - parameters: Generation parameters, threaded into
    ///     `model.newCache(parameters:)` so a rebuilt cache matches what
    ///     the generation call would have created unassisted (e.g.
    ///     `maxKVSize`-driven rotating caches).
    /// - Returns: The `[KVCache]` to pass as generation's `cache:`
    ///   argument, and the token subsequence to actually feed (a suffix of
    ///   `newTokens`, or all of `newTokens` on rebuild).
    func resolve(
        modelID: String, newTokens: [Int], model: SendableBox<any LanguageModel>,
        parameters: GenerateParameters?
    ) -> SendableBox<(cache: [KVCache], tokensToFeed: [Int])> {
        let model = model.consume()
        var slots = entries[modelID] ?? []
        let candidates = slots.map { (tokens: $0.tokens, lastUsed: $0.lastUsed) }
        guard let bestIndex = Self.selectSlot(candidates: candidates, newTokens: newTokens) else {
            return SendableBox((model.newCache(parameters: parameters), newTokens))
        }

        let slot = slots.remove(at: bestIndex)
        entries[modelID] = slots.isEmpty ? nil : slots

        let decision = Self.decide(
            cachedTokens: slot.tokens, newTokens: newTokens,
            isTrimmable: canTrimPromptCache(slot.cache))
        let result = Self.applyDecision(
            decision, slot: slot, newTokens: newTokens, model: model, parameters: parameters)
        return SendableBox(result)
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
    /// Accepts an exact count match, or a count exactly one *more* than
    /// `actualGeneratedCount` -- dropping the reconstruction's trailing
    /// token in that case. `TokenIterator`'s next()-ahead prefetch design
    /// (see `MLXLMCommon/Evaluate.swift`) discards the terminal EOS/stop
    /// token without ever passing it to a stream consumer once that token
    /// has already been fed through the model, so the emitted text's full
    /// re-encoding legitimately lands one token ahead of the cache's real
    /// advance on that (common, successful) natural-stop exit path. Any
    /// other mismatch (including a re-encoding that is too *short*) is
    /// untrustworthy.
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
    ///     (e.g. `generatedTokenIds.count`).
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

    /// Remembers `tokens`/`cache` as a new slot for this model, evicting
    /// the least-recently-used slot(s) if the model's slot count would
    /// exceed `maxSlotsPerModel`.
    ///
    /// Called once a generation call completes and its actual token count
    /// has been verified against `cache`'s own `offset` (see
    /// `Executor.commitPromptCache`).
    func store(modelID: String, tokens: [Int], cache: SendableBox<[KVCache]>) {
        var slots = entries[modelID] ?? []
        slots.append(Slot(tokens: tokens, cache: cache.consume(), lastUsed: nextRecency()))
        if slots.count > maxSlotsPerModel {
            slots.sort { $0.lastUsed < $1.lastUsed }
            slots.removeFirst(slots.count - maxSlotsPerModel)
        }
        entries[modelID] = slots
    }

    /// Drops every model's remembered slots and chunk store. Mirrors
    /// `ModelCache.evictAll`.
    func evictAll() {
        entries.removeAll()
        chunkStore.removeAll()
    }

    /// Drops one model's remembered slots and chunk store. Mirrors
    /// `ModelCache.remove`.
    func remove(modelID: String) {
        entries.removeValue(forKey: modelID)
        chunkStore.removeValue(forKey: modelID)
    }

    // MARK: - Chunk store

    /// Checks freshly sliced chunks (see `PromptCache.sliceChunks`) into
    /// `modelID`'s chunk store, deduplicating by ``ChunkKey``.
    ///
    /// A key already present is dedup'd: only its `lastUsed` is refreshed --
    /// its tensors are NEVER replaced, since an existing entry's tensors are
    /// already a byte-for-byte-equivalent owned copy for the same
    /// `(parentKey, tokens)` chain position (dedup existing is the entire
    /// point: two conversations sharing a prefix must retain that prefix's KV
    /// state exactly once). A new key is stored fresh, stamped with the
    /// current recency.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier this chunk store is scoped to.
    ///   - chunks: Freshly sliced chunks to check in, in chain order (as
    ///     `sliceChunks` returns them, though this method doesn't require
    ///     that order itself).
    func insert(modelID: String, chunks: [StoredChunk]) {
        var models = chunkStore[modelID] ?? [:]
        for chunk in chunks {
            // `entry` binds to the EXISTING stored chunk when its key is
            // already present (dedup: only `lastUsed` changes, tensors
            // untouched), or to the new `chunk` parameter when absent.
            var entry = models[chunk.chunkKey] ?? chunk
            entry.lastUsed = nextRecency()
            models[chunk.chunkKey] = entry
        }
        chunkStore[modelID] = models
    }

    /// Walks `modelID`'s chunk hash chain from the root over chunk-aligned
    /// windows of `newTokens`, returning the longest prefix of matching,
    /// verified chunks.
    ///
    /// CRITICAL: the walk is capped at `newTokens.count - 1` tokens, so at
    /// least one token always remains unmatched for the caller to feed --
    /// generation always needs >=1 fresh token to produce new logits, even
    /// when the entire prompt is otherwise covered by stored chunks (this is
    /// the chunk-store equivalent of the slot pool's `regenerateLastToken`/
    /// n_past-- trick, but simpler: rather than trimming one token back off a
    /// full match, the walk itself never consumes the prompt's last token).
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

    /// Number of distinct chunks currently stored for `modelID` -- exposes
    /// dedup/eviction bookkeeping for tests without leaking the actor's
    /// private storage shape.
    func chunkCount(modelID: String) -> Int {
        chunkStore[modelID]?.count ?? 0
    }
}

#endif
#endif
