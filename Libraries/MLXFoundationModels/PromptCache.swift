// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

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
/// KNOWN WINDOW: as of this cutover, the chunk store has NO capacity bound
/// -- it grows without eviction until a later task (kanban `2sdt6dj`) adds
/// byte-budget LRU eviction. Acceptable on this branch because chunking
/// itself already bounds memory better than the old slot pool did (a
/// shared prefix is stored once, not once per conversation), but a
/// long-running process serving many distinct prompts will still grow
/// ``chunkStore`` unboundedly until that follow-up lands.
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
    func resolve(
        modelID: String, newTokens: [Int], model: SendableBox<any LanguageModel>,
        parameters: GenerateParameters?
    ) -> SendableBox<(cache: [KVCache], tokensToFeed: [Int])> {
        let model = model.consume()
        let chunks = lookupLongestPrefix(
            modelID: modelID, newTokens: newTokens, chunkSize: chunkSize
        ).consume()
        guard let layerCount = chunks.first?.layers.count else {
            return SendableBox((model.newCache(parameters: parameters), newTokens))
        }
        let matchedTokenCount = chunks.count * chunkSize
        let assembled = Self.assemble(chunks: chunks, layerCount: layerCount)
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
    /// them into `modelID`'s shared chunk store (``insert(modelID:chunks:)``).
    ///
    /// Silently drops the round when `cache`'s layers aren't a chunkable
    /// shape (see `sliceChunks`'s degradation for `RotatingKVCache`/
    /// `ChunkedKVCache`/an offset mismatch) -- the same degradation the
    /// old slot pool had for a non-trimmable cache: this round simply
    /// contributes nothing reusable, and the next round for this model
    /// rebuilds from scratch.
    ///
    /// Called once a generation call completes and its actual token count
    /// has been verified against `cache`'s own `offset` (see
    /// `Executor.commitPromptCache`).
    func store(modelID: String, tokens: [Int], cache: SendableBox<[KVCache]>) {
        guard
            let chunks = Self.sliceChunks(
                tokens: tokens, cache: cache.consume(), chunkSize: chunkSize)
        else { return }
        insert(modelID: modelID, chunks: chunks)
    }

    /// Drops every model's remembered chunk store. Mirrors
    /// `ModelCache.evictAll`.
    func evictAll() {
        chunkStore.removeAll()
    }

    /// Drops one model's remembered chunk store. Mirrors
    /// `ModelCache.remove`.
    func remove(modelID: String) {
        chunkStore.removeValue(forKey: modelID)
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
    /// unconditionally calls ``evictAll()``, discarding every model's
    /// stored chunks so the next `resolve()`/`store()` rebuilds from
    /// scratch under the new span.
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
        evictAll()
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

    /// Number of distinct chunks currently stored for `modelID` -- exposes
    /// dedup/eviction bookkeeping for tests without leaking the actor's
    /// private storage shape.
    func chunkCount(modelID: String) -> Int {
        chunkStore[modelID]?.count ?? 0
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
    /// - Returns: One freshly assembled `KVCacheSimple` per layer, offset at
    ///   the matched token count; empty when `chunks` is empty or any
    ///   chunk's `layers.count` doesn't match `layerCount`.
    nonisolated static func assemble(chunks: [StoredChunk], layerCount: Int) -> [KVCache] {
        guard !chunks.isEmpty, layerCount > 0,
            chunks.allSatisfy({ $0.layers.count == layerCount })
        else { return [] }

        var result: [KVCache] = []
        result.reserveCapacity(layerCount)
        for layerIndex in 0 ..< layerCount {
            let keys = ownedCopy(of: concatenated(chunks.map { $0.layers[layerIndex].keys }, axis: 2))
            let values = ownedCopy(
                of: concatenated(chunks.map { $0.layers[layerIndex].values }, axis: 2))

            let layerCache = KVCacheSimple()
            layerCache.state = [keys, values]
            result.append(layerCache)
        }
        return result
    }
}

#endif
#endif
