// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

/// Per-model token-prefix KV-cache reuse for `MLXLanguageModel.Executor`.
///
/// FoundationModels' `LanguageModelExecutor` protocol has no session
/// identity: `Executor.respond(to:model:streamingInto:)` receives the
/// *full* transcript on every round with no indication of which prior
/// round (if any) it continues, or whether this is even the same session
/// as last time. `PromptCache` keys reuse on the token prefix itself
/// instead: it remembers the most-recently-generated-from token sequence
/// and its resulting `[KVCache]` for each model, and on the next round
/// compares the freshly-tokenized prompt against that remembered sequence
/// to decide whether to reuse the cache unchanged, trim it back to a
/// common prefix, or rebuild from scratch.
///
/// Mirrors `ModelCache`'s per-model-actor-cache pattern in
/// `MLXLanguageModel.swift`: a private actor, instantiated once as a
/// process-global `static let`, exposed to `Executor` through thin
/// passthrough statics on `MLXLanguageModel`.
actor PromptCache {

    /// One remembered `(token sequence, KV cache)` pair per model.
    private var entries: [String: (tokens: [Int], cache: [KVCache])] = [:]

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
    /// cache/actor machinery.
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

    /// Resolves the `[KVCache]` to generate with and the tokens to
    /// actually feed it this round, applying `decide`'s outcome against
    /// whatever is remembered for `modelID`. Always returns a concrete,
    /// non-empty-suffix result: a `.reuseSuffix`/`.trimTo` outcome with a
    /// zero-length suffix (an unchanged or shrunk-only prompt) can't
    /// actually continue generation -- producing the next token requires
    /// feeding the model at least one token -- so that case falls back to
    /// a rebuild too.
    ///
    /// Removes (rather than merely reads) `modelID`'s entry: `KVCache`
    /// instances are plain classes with no internal synchronization, so
    /// handing the same instances to two concurrent generation calls for
    /// the same model would race. Removing here means a second concurrent
    /// call for the same model finds no entry and safely rebuilds instead
    /// of aliasing the first call's in-flight cache; `store` checks a
    /// (possibly different, newer) entry back in once generation
    /// completes.
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
        guard let entry = entries.removeValue(forKey: modelID) else {
            return SendableBox((model.newCache(parameters: parameters), newTokens))
        }

        switch Self.decide(
            cachedTokens: entry.tokens, newTokens: newTokens,
            isTrimmable: canTrimPromptCache(entry.cache))
        {
        case .reuseSuffix(let count) where count > 0:
            return SendableBox((entry.cache, Array(newTokens.suffix(count))))
        case .trimTo(let commonPrefixLength, let thenSuffix) where thenSuffix > 0:
            trimPromptCache(entry.cache, numTokens: entry.tokens.count - commonPrefixLength)
            return SendableBox((entry.cache, Array(newTokens.suffix(thenSuffix))))
        default:
            return SendableBox((model.newCache(parameters: parameters), newTokens))
        }
    }

    /// Pure reconciliation between a re-encoded text's token count and a
    /// KV cache's authoritative `offset` advance -- no I/O, no actor
    /// state; unit-tested directly by `PromptCacheReconciliationTests`.
    ///
    /// Used by `Executor.commitPromptCache(...emittedText:...)` for
    /// generation paths that only yield decoded text, not raw token IDs
    /// (`generate(input:cache:parameters:context:...)`,
    /// `GuidedGenerationLoop.run`): those paths reconstruct the generated
    /// token IDs by re-encoding the emitted text, and this function decides
    /// whether that reconstruction can be trusted for storage.
    ///
    /// Accepts an exact count match, or a count exactly one *more* than
    /// `actualGeneratedCount` -- dropping the reconstruction's trailing
    /// token in that case. `GuidedGenerationLoop.run`'s natural
    /// (grammar-accepted) termination commits, detokenizes, and counts its
    /// terminal token *before* checking `commitResult.isTerminated` and
    /// breaking, so that last token is never fed back through the model:
    /// the cache's real offset legitimately lands one token short of the
    /// emitted text's full re-encoding on that common, successful exit
    /// path. Any other mismatch (including a re-encoding that is too
    /// *short*) is untrustworthy.
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

    /// Remembers `tokens`/`cache` as the state to reuse from on this
    /// model's next round. Called once a generation call completes and
    /// its actual token count has been verified against `cache`'s own
    /// `offset` (see `Executor.commitPromptCache`).
    func store(modelID: String, tokens: [Int], cache: SendableBox<[KVCache]>) {
        entries[modelID] = (tokens, cache.consume())
    }

    /// Drops every model's remembered cache. Mirrors `ModelCache.evictAll`.
    func evictAll() {
        entries.removeAll()
    }

    /// Drops one model's remembered cache. Mirrors `ModelCache.remove`.
    func remove(modelID: String) {
        entries.removeValue(forKey: modelID)
    }
}

#endif
#endif
