---
assignees:
- claude-code
depends_on:
- 01KX3MJKYYNXQ8CK197WX4W28J
position_column: todo
position_ordinal: 8a80
title: 'Chunk store: per-model insert with dedup, longest-prefix lookup, LRU touch'
---
## What
Inside the PromptCache actor (Libraries/MLXFoundationModels/PromptCache.swift): replace the slot dictionary with a per-model chunk store `[String /*modelID*/: [ChunkKey: StoredChunk]]` plus the monotonic recency counter (keep `nextRecency()`).
- `insert(modelID:chunks:)`: walk the chunk list; a key already present is deduplicated (refresh its `lastUsed`, do NOT replace tensors — dedup is the point); new keys are stored.
- `lookupLongestPrefix(modelID:newTokens:chunkSize:) -> [StoredChunk]`: walk the hash chain from the root over chunk-aligned windows of `newTokens`, stopping at the first missing key. CRITICAL: cap the walk at `newTokens.count - 1` tokens so at least one token always remains to feed (this replaces the old regenerateLastToken / n_past-- machinery — generation always needs ≥1 token for fresh logits). Touch every matched chunk's `lastUsed`.
- COLLISION SAFETY: a 64-bit hash collision would serve wrong KV state silently — the worst possible cache failure. On every key match, verify `chunk.tokens == the chunk-aligned window` (cheap array compare, SGLang-style); mismatch = treat as miss and stop the walk.
- Keep `evictAll()` / `remove(modelID:)` semantics over the new storage.

## Acceptance Criteria
- [ ] Storing two conversations sharing a 3-chunk prefix results in the shared chunks existing ONCE (dedup observable via store size / byte accounting)
- [ ] Lookup returns the longest chain prefix; diverging tokens stop the walk at the last shared chunk boundary
- [ ] Lookup for newTokens exactly equal to a fully-stored sequence returns at most floor((count-1)/chunkSize) chunks — never covers the whole prompt
- [ ] A key match whose stored tokens differ from the window (forced collision in test) is treated as a miss — wrong-token KV is never returned
- [ ] evictAll/remove drop chunks for the right scope (per-model isolation preserved)

## Tests
- [ ] Extend Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift (or new PromptCacheChunkStoreTests.swift): dedup, chain walk, divergence, cap-at-count-1, forced-collision-is-miss (inject a StoredChunk under a colliding key with different tokens), evictAll/remove scoping, multi-model isolation
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.