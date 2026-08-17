// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

// MARK: - Identity

/// Names the one session whose turns share a prompt cache.
struct ExecutorPromptCacheKey: Hashable, Sendable {

    /// The model the cache belongs to, thus two models never share one cache.
    let modelID: String

    /// The session the cache belongs to.
    let sessionID: String
}

// MARK: - What a session carries

/// The live key/value cache of one session, and the tokens it holds.
///
/// This is a reference type because the caches it carries are reference types
/// that generation writes into. ``ExecutorPromptCacheStore`` hands one entry to
/// at most one turn at a time -- a check-out REMOVES the entry from the store --
/// thus no two tasks write the same caches, and the `@unchecked Sendable`
/// conformance rests on that.
final class ExecutorPromptCacheEntry: @unchecked Sendable {

    /// One cache for each layer of the model.
    let caches: [KVCache]

    /// The exact prompt tokens `caches` represents, in order.
    let tokens: [Int]

    /// Creates an entry for caches that hold `tokens`.
    init(caches: [KVCache], tokens: [Int]) {
        self.caches = caches
        self.tokens = tokens
    }
}

// MARK: - The store

/// Holds the prompt cache of each live session between the turns of that
/// session.
///
/// A turn CHECKS OUT its entry, which removes the entry from the store, and
/// checks an entry back in when the turn ends. Two turns of one session thus
/// never write the same caches at the same time: a second turn that starts
/// while the first still runs finds nothing and starts cold.
actor ExecutorPromptCacheStore {

    /// The store every executor shares.
    static let shared = ExecutorPromptCacheStore()

    /// How many sessions hold a cache at the same time.
    ///
    /// Each entry holds the whole key/value state of one conversation, which is
    /// large. The framework never tells this executor that a session ended, thus
    /// this bound is the only thing that releases the memory of a session nobody
    /// uses again. The least recently used entry leaves first.
    static let maximumRetainedSessions = 4

    private var entries: [ExecutorPromptCacheKey: ExecutorPromptCacheEntry] = [:]

    /// The checked-in keys, least recently used first.
    private var usageOrder: [ExecutorPromptCacheKey] = []

    /// How many sessions hold a cache. Read by tests.
    var retainedSessionCount: Int { entries.count }

    /// Takes the entry of `key` out of the store.
    ///
    /// - Returns: the entry, or nil when the store holds none for `key`.
    func checkOut(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheEntry? {
        usageOrder.removeAll { $0 == key }
        return entries.removeValue(forKey: key)
    }

    /// Puts `entry` back under `key`, or drops the key when `entry` is nil.
    func checkIn(_ key: ExecutorPromptCacheKey, _ entry: ExecutorPromptCacheEntry?) {
        usageOrder.removeAll { $0 == key }
        guard let entry else {
            entries.removeValue(forKey: key)
            return
        }
        entries[key] = entry
        usageOrder.append(key)
        while usageOrder.count > Self.maximumRetainedSessions {
            entries.removeValue(forKey: usageOrder.removeFirst())
        }
    }

    /// Releases the cache of every session of `modelID`, or of every session
    /// when `modelID` is nil.
    func evict(modelID: String?) {
        guard let modelID else {
            entries.removeAll()
            usageOrder.removeAll()
            return
        }
        for key in entries.keys.filter({ $0.modelID == modelID }) {
            entries.removeValue(forKey: key)
        }
        usageOrder.removeAll { $0.modelID == modelID }
    }
}

// MARK: - Planning one generation pass

/// What one generation pass does with the prompt cache of its session.
struct ExecutorPromptCachePlan {

    /// The caches to hand to generation.
    let caches: [KVCache]

    /// The input to feed, narrowed to the tokens `caches` does not already hold.
    let input: LMInput

    /// The leading prompt tokens `caches` already holds, thus the tokens this
    /// pass does not feed to the model.
    let reusedTokenCount: Int

    /// The whole rendered prompt of this pass.
    let promptTokens: [Int]

    /// Plans what `entry` may serve for `input`, building fresh caches when it
    /// may serve nothing.
    ///
    /// - Parameters:
    ///   - entry: the cache the session carries, or nil for a cold session.
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    /// - Returns: the plan, or nil when a carried cache cannot serve this input
    ///   at all. Media, an explicit attention mask and a batched token rank each
    ///   place content in the model's input that a token ledger cannot describe;
    ///   the caller then generates with no carried cache.
    static func make(
        reusing entry: ExecutorPromptCacheEntry?,
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> ExecutorPromptCachePlan? {
        guard input.image == nil, input.video == nil, input.audio == nil,
            input.text.mask == nil, input.text.tokens.ndim == 1
        else {
            return nil
        }

        let promptTokens = input.text.tokens.asArray(Int.self)
        guard !promptTokens.isEmpty else { return nil }

        if let entry,
            let reusedTokenCount = reusablePromptPrefix(
                promptTokens: promptTokens, cachedTokens: entry.tokens, caches: entry.caches),
            reusedTokenCount < promptTokens.count
        {
            return ExecutorPromptCachePlan(
                caches: entry.caches,
                input: reusedTokenCount == 0
                    ? input
                    : LMInput(tokens: MLXArray(Array(promptTokens[reusedTokenCount...]))),
                reusedTokenCount: reusedTokenCount,
                promptTokens: promptTokens)
        }

        return ExecutorPromptCachePlan(
            caches: try model.newCache(parameters: parameters),
            input: input,
            reusedTokenCount: 0,
            promptTokens: promptTokens)
    }

    /// The cache this finished pass leaves for the next turn of its session.
    ///
    /// Generation feeds the tokens the model sampled into the caches, thus the
    /// caches hold the render of this pass AND those generated tokens. The
    /// ledger names both, which is the ledger ``ChatSession`` keeps. A rewind
    /// back to the render alone is not available to every model -- a rotating
    /// cache past its sliding window drops the keys a rewind needs -- and this
    /// ledger asks for none.
    ///
    /// The next turn reconciles the two. When its render extends this ledger,
    /// `ExtendCachedPrefixRule` feeds the tail alone. When the render breaks the
    /// prefix at the seam between the two, `RewindToCommonPrefixRule` takes over
    /// and answers what the caches allow.
    ///
    /// - Parameter generatedTokens: every token this pass generated, in order.
    ///   `TokenIterator` feeds a token before it answers it, thus each token of
    ///   this list stands in the caches.
    /// - Returns: the entry to check in, or nil when the caches did not land on
    ///   a position this ledger can name and the session must start cold.
    func committed(generatedTokens: [Int]) -> ExecutorPromptCacheEntry? {
        guard let position = caches.first?.offset,
            caches.allSatisfy({ $0.offset == position }),
            position >= promptTokens.count,
            position - promptTokens.count <= generatedTokens.count
        else {
            return nil
        }
        let committedGeneratedTokenCount = position - promptTokens.count
        return ExecutorPromptCacheEntry(
            caches: caches,
            tokens: promptTokens + generatedTokens.prefix(committedGeneratedTokenCount))
    }
}

// MARK: - Carrying the cache through one response

/// Carries the prompt cache of one session through one response.
///
/// `MLXLanguageModel.Executor.respond(to:model:streamingInto:)` checks the
/// session's cache out before it generates and checks this slot's entry back in
/// when it finishes, whatever the outcome.
///
/// A response runs one generation pass at a time, and every pass runs inside the
/// model container, which serializes the passes of one model. This box is thus
/// written by one task at a time, and the `@unchecked Sendable` conformance
/// rests on that.
final class ExecutorPromptCacheSlot: @unchecked Sendable {

    /// The cache the session carries into its next turn, or nil when the next
    /// turn must start cold.
    private(set) var entry: ExecutorPromptCacheEntry?

    /// The prompt tokens the last planned pass did not feed to the model.
    ///
    /// Zero until a pass carries a cache. A pass that builds its own cache --
    /// the guided loop does -- leaves the value at zero, which is the measured
    /// truth for that pass.
    private(set) var reusedTokenCount = 0

    /// Creates a slot holding the entry a session checked out.
    init(_ entry: ExecutorPromptCacheEntry?) {
        self.entry = entry
    }

    /// Plans the pass that is about to run, and records what that pass reuses.
    ///
    /// The slot gives up its entry: the pass owns the caches until
    /// ``commit(_:generatedTokens:)`` takes them back.
    ///
    /// - Parameters:
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    /// - Returns: the plan, or nil when the pass carries no cache.
    func plan(
        input: LMInput, model: any LanguageModel, parameters: GenerateParameters
    ) throws -> ExecutorPromptCachePlan? {
        let plan = try ExecutorPromptCachePlan.make(
            reusing: entry, input: input, model: model, parameters: parameters)
        entry = nil
        reusedTokenCount = plan?.reusedTokenCount ?? 0
        return plan
    }

    /// Records that the pass about to run carries no cache from an earlier turn.
    ///
    /// The guided loop owns its key/value cache and accepts none from a caller,
    /// thus a guided pass reuses nothing and must report nothing, even when an
    /// earlier pass of the same response reused a prefix.
    func carriesNoCache() {
        reusedTokenCount = 0
    }

    /// Records the cache a finished pass leaves for the next turn.
    ///
    /// - Parameters:
    ///   - plan: the plan the pass ran, or nil when the pass carried no plan.
    ///   - generatedTokens: the tokens the pass generated, in order.
    func commit(_ plan: ExecutorPromptCachePlan?, generatedTokens: [Int]) {
        entry = plan?.committed(generatedTokens: generatedTokens)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
