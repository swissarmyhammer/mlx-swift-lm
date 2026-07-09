---
position_column: todo
position_ordinal: '8780'
title: Wire cross-turn KV cache in MLXLanguageModel.Executor
---
MLXLanguageModel.Executor currently passes nil for cache: to every generate() call, allocating a fresh [KVCache] per turn and re-processing the entire transcript from token 0 every time. The underlying KVCache.offset mechanism is fully functional -- delta tokens placed at offset N get correct RoPE embeddings and attend to all prior-turn context via the causal mask. The fix is to persist the cache across turns within a session and pass only the delta tokens (new since last turn) to each generate call.

New file Libraries/MLXFoundationModels/ExecutorCacheStore.swift:
actor ExecutorCacheStore {
    struct Entry {
        var cache: [any KVCache]
        var processedTokenCount: Int
    }
    private var sessions: [String: Entry] = [:]

    func entry(forSessionKey key: String, model: any LanguageModel) -> Entry {
        if let existing = sessions[key] { return existing }
        let fresh = Entry(cache: model.newCache(parameters: nil), processedTokenCount: 0)
        sessions[key] = fresh
        return fresh
    }

    func update(key: String, cache: [any KVCache], processedTokenCount: Int) {
        sessions[key] = Entry(cache: cache, processedTokenCount: processedTokenCount)
    }

    func evict(key: String) { sessions.removeValue(forKey: key) }
}

Add a private static let cacheStore = ExecutorCacheStore() to MLXLanguageModel (or to ModelCache) so it is process-global and keyed by session.

Session key: request.transcript.entries.first?.id ?? UUID().uuidString -- the first transcript entry's ID is stable for the entire life of a LanguageModelSession, making it a reliable per-session key without requiring a session-level ID on the request struct.

Executor.respond() before dispatching to generation helpers:
1. Tokenize full transcript -> allTokens
2. Look up (or allocate) the session's cache entry via sessionKey = request.transcript.entries.first?.id ?? UUID().uuidString; entry = await Self.cacheStore.entry(forSessionKey: sessionKey, model: context.model)
3. alreadyCached = min(entry.processedTokenCount, allTokens.count); deltaTokens = Array(allTokens[alreadyCached...])
4. Build deltaInput from deltaTokens only
5. Pass entry.cache + deltaInput to helpers

runUnconstrained, runReasoning, runGuidedGeneration: add cache: [any KVCache] parameter, forward to generate(input:cache:parameters:context:).

After generation completes in each helper: write back await Self.cacheStore.update(key: sessionKey, cache: cache, processedTokenCount: allTokens.count).

Fix cachedTokenCount: 0 -- currently hardcoded at 4 call sites inside Executor.respond(). Replace with alreadyCached at each site.

Cache eviction: hook MLXLanguageModel.evict() to also call await Self.cacheStore.evict(key:) for any sessions associated with that model ID.

Acceptance Criteria:
- ExecutorCacheStore actor exists in the fork
- Executor.respond() derives a session key from the first transcript entry ID
- Delta-token slicing: only tokens beyond processedTokenCount are passed to generate()
- cachedTokenCount in .updateUsage events reflects actual cached token count (not hardcoded 0)
- Fork's own unit tests pass: swift test in the fork repo -- write a test asserting cachedTokenCount > 0 on second turn (TDD: watch it fail, implement, watch it pass)
- Commit and push to origin/mlx-foundationmodels

Downstream: once this lands and is pushed, the consuming repo FoundationModelsRouter needs its Package.resolved bumped to the new commit, and has kanban tasks 070qw7z ("Prove multi-turn conversation state and KV cache usage in router sessions") depending on this work -- that repo's own board should NOT attempt this fork-side implementation itself, only consume the resulting pin bump once available.