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
/// that generation writes into, and the model state beside them is not
/// `Sendable`. ``ExecutorPromptCacheStore`` hands one entry to at most one turn
/// at a time -- a check-out REMOVES the entry from the store -- thus no two
/// tasks write the same caches or read the same state, and the
/// `@unchecked Sendable` conformance rests on that.
final class ExecutorPromptCacheEntry: @unchecked Sendable {

    /// One cache for each layer of the model.
    let caches: [KVCache]

    /// The exact prompt tokens `caches` represents, in order.
    let tokens: [Int]

    /// The whole prompt the last pass rendered, which is not the same as
    /// `tokens`: the ledger also holds the tokens the model generated after
    /// that render.
    ///
    /// A protocol rule compares this with the next render to prove that the
    /// template rewrote no cached region before it splices past the turn the
    /// model committed. Empty means no render is on record, thus no rule may
    /// splice.
    let renderTokens: [Int]

    /// The model state the last pass left with `caches`, or nil for a model
    /// that carries none.
    ///
    /// A Qwen VL model keys its M-RoPE anchor here and refuses a warm cache
    /// without it, thus the next turn seeds its iterator with this state. The
    /// state is tied to the prefill that made it, thus the caches never rewind
    /// under it.
    let state: LMOutput.State?

    /// Creates an entry for caches that hold `tokens`.
    ///
    /// - Parameters:
    ///   - caches: one cache for each layer of the model.
    ///   - tokens: the exact tokens `caches` represents.
    ///   - renderTokens: the whole prompt the last pass rendered, or empty
    ///     when no render is on record.
    ///   - state: the model state the last pass left, or nil.
    init(
        caches: [KVCache], tokens: [Int], renderTokens: [Int] = [],
        state: LMOutput.State? = nil
    ) {
        self.caches = caches
        self.tokens = tokens
        self.renderTokens = renderTokens
        self.state = state
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

    /// Reads the entry of `key` and leaves it in the store. Read by tests that
    /// compare a session's ledger with its next render.
    ///
    /// - Returns: the entry, or nil when the store holds none for `key`.
    func peek(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheEntry? {
        entries[key]
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

    /// The tokens `caches` represents once this pass has fed `input`, before
    /// generation. This is `promptTokens` on the standard path. A protocol rule
    /// that splices past a committed turn names the tokens the model wrote
    /// instead, plus the tail of the render it fed.
    let representedTokens: [Int]

    /// The model state to seed the iterator with: the state the entry carried
    /// when this pass reuses its caches, and nil when the pass builds fresh
    /// caches.
    let state: LMOutput.State?

    /// Plans what `entry` may serve for `input`, building fresh caches when it
    /// may serve nothing.
    ///
    /// - Parameters:
    ///   - entry: the cache the session carries, or nil for a cold session.
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    /// - Returns: the plan, or nil when a carried cache cannot serve this input
    ///   at all. Media, a mask that hides a token and a batch of more than one
    ///   row each place content in the model's input that a token ledger cannot
    ///   describe; the caller then generates with no carried cache.
    static func make(
        reusing entry: ExecutorPromptCacheEntry?,
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = []
    ) throws -> ExecutorPromptCachePlan? {
        guard let promptTokens = ledgerTokens(of: input), !promptTokens.isEmpty else {
            return nil
        }

        if let entry,
            let reuse = reconcilePromptCache(
                promptTokens: promptTokens, cachedTokens: entry.tokens,
                previousRenderTokens: entry.renderTokens, caches: entry.caches,
                protocolRules: protocolRules, carriesModelState: entry.state != nil),
            reuse.suffixStart < promptTokens.count
        {
            return ExecutorPromptCachePlan(
                caches: entry.caches,
                input: reuse.suffixStart == 0
                    ? input
                    : narrowed(input, to: Array(promptTokens[reuse.suffixStart...])),
                reusedTokenCount: reuse.suffixStart,
                promptTokens: promptTokens,
                representedTokens: reuse.representedTokens,
                state: entry.state)
        }

        return ExecutorPromptCachePlan(
            caches: try model.newCache(parameters: parameters),
            input: input,
            reusedTokenCount: 0,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: nil)
    }

    /// The rank of a token array a processor batched: one row for each
    /// sequence, and the tokens of that sequence along the other axis.
    private static let batchedTokenRank = 2

    /// The prompt tokens of `input`, or nil when `input` carries content a
    /// token ledger cannot describe.
    ///
    /// A VLM processor batches a text-only prompt to one row and masks every
    /// token present. That input carries nothing a ledger cannot describe, thus
    /// it passes. Media, a batch of more than one row and a mask that hides a
    /// token each carry more than the tokens say, thus they do not.
    private static func ledgerTokens(of input: LMInput) -> [Int]? {
        guard input.image == nil, input.video == nil, input.audio == nil,
            holdsOneSequence(input.text.tokens),
            marksEveryTokenPresent(input.text.mask, of: input.text.tokens)
        else {
            return nil
        }
        return input.text.tokens.asArray(Int.self)
    }

    /// Whether `tokens` holds one sequence: a plain vector, or a batch of one
    /// row.
    private static func holdsOneSequence(_ tokens: MLXArray) -> Bool {
        tokens.ndim == 1 || (tokens.ndim == batchedTokenRank && tokens.dim(0) == 1)
    }

    /// Whether `mask` marks every token of `tokens` present. No mask hides
    /// nothing; a mask of another shape, or one that holds a zero, hides
    /// something the ledger cannot name.
    private static func marksEveryTokenPresent(_ mask: MLXArray?, of tokens: MLXArray) -> Bool {
        guard let mask else { return true }
        return mask.shape == tokens.shape && mask.asArray(Int32.self).allSatisfy { $0 != 0 }
    }

    /// `input` narrowed to `tokens`, in the rank and the mask presence of
    /// `input`, thus the model sees the shape its processor makes.
    private static func narrowed(_ input: LMInput, to tokens: [Int]) -> LMInput {
        var narrowedTokens = MLXArray(tokens)
        if input.text.tokens.ndim == batchedTokenRank {
            narrowedTokens = narrowedTokens.expandedDimensions(axis: 0)
        }
        let narrowedMask = input.text.mask.map { ones(like: narrowedTokens).asType($0.dtype) }
        return LMInput(text: .init(tokens: narrowedTokens, mask: narrowedMask))
    }

    /// The cache this finished pass leaves for the next turn of its session.
    ///
    /// Generation feeds the tokens the model sampled into the caches, thus the
    /// caches hold what this pass represented before generation AND those
    /// generated tokens. The ledger names both, which is the ledger
    /// ``ChatSession`` keeps. A rewind back to the render alone is not available
    /// to every model -- a rotating cache past its sliding window drops the keys
    /// a rewind needs, and a recurrent cache never rewinds -- and this ledger
    /// asks for none.
    ///
    /// The next turn reconciles the two. When its render extends this ledger,
    /// `ExtendCachedPrefixRule` feeds the tail alone. When the render rewrites
    /// the turn the model wrote, a protocol rule splices past the commit that
    /// closed that turn, with the render recorded here as its proof. Otherwise
    /// `RewindToCommonPrefixRule` takes over and answers what the caches allow.
    ///
    /// - Parameters:
    ///   - generatedTokens: every token this pass generated, in order.
    ///     `TokenIterator` feeds a token before it answers it, thus each token
    ///     of this list stands in the caches.
    ///   - state: the model state the prefill of this pass left, which the
    ///     next turn seeds its iterator with, or nil for a model that carries
    ///     none.
    /// - Returns: the entry to check in, or nil when the caches did not land on
    ///   a position this ledger can name and the session must start cold.
    func committed(
        generatedTokens: [Int], state: LMOutput.State? = nil
    ) -> ExecutorPromptCacheEntry? {
        guard let position = caches.first?.offset,
            caches.allSatisfy({ $0.offset == position }),
            position >= representedTokens.count,
            position - representedTokens.count <= generatedTokens.count
        else {
            return nil
        }
        let committedGeneratedTokenCount = position - representedTokens.count
        return ExecutorPromptCacheEntry(
            caches: caches,
            tokens: representedTokens + generatedTokens.prefix(committedGeneratedTokenCount),
            renderTokens: promptTokens,
            state: state)
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
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    /// - Returns: the plan, or nil when the pass carries no cache.
    func plan(
        input: LMInput, model: any LanguageModel, parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = []
    ) throws -> ExecutorPromptCachePlan? {
        let plan = try ExecutorPromptCachePlan.make(
            reusing: entry, input: input, model: model, parameters: parameters,
            protocolRules: protocolRules)
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
    ///   - state: the model state the prefill of the pass left, or nil.
    func commit(
        _ plan: ExecutorPromptCachePlan?, generatedTokens: [Int], state: LMOutput.State? = nil
    ) {
        entry = plan?.committed(generatedTokens: generatedTokens, state: state)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
