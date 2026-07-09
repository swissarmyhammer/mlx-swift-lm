---
assignees:
- claude-code
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
position_column: todo
position_ordinal: '8e80'
title: Byte-budget LRU eviction; remove the superseded slot-limit API
---
## What
In Libraries/MLXFoundationModels/PromptCache.swift: track total stored bytes (StoredChunk.byteSize — REAL owned-copy footprint per the slicing task, not view metadata). On insert exceeding the budget, evict least-recently-used chunks until under budget. Evicting a chunk orphans its chain descendants (lookup already stops at the missing parent, so orphans are unreachable) — orphaned chunks' bytes MUST still be reclaimed: either evict descendants transitively with their parent, or treat unreachable chunks as immediately evictable; implementer picks one and documents it. Default budget: derive from the GPU/unified-memory limit if mlx-swift exposes one (check `MLX.GPU.memoryLimit` / recommendedMaxWorkingSetSize), else a documented fixed default (e.g. 2 GiB); clamp ≥ one chunk.

PEAK-MEMORY MODEL (document in the API doc): the budget bounds the STORE only — every resolve additionally materializes one assembled prefix copy per in-flight request, so peak unified memory ≈ budget + (in-flight requests × assembled prefix size), and chunk residency competes with MLX's own GPU buffer cache (GPU cacheLimit). The default budget derivation must leave explicit headroom for assembly copies.

Public API in Libraries/MLXFoundationModels/MLXLanguageModel.swift: `public static func setPromptCacheByteBudget(_ bytes: Int) async` beside evictAll(). REMOVE the superseded slot-limit surface if it still exists at this point (`setPromptCacheSlotLimit` — coordinate with the cutover task; exactly one task deletes it).

## Acceptance Criteria
- [ ] Inserting past the budget evicts the least-recently-used chunks first; recently-touched (resolved) chunks survive
- [ ] Total accounted bytes never exceeds the budget after insert returns
- [ ] Orphaning a lineage (evict its head under budget pressure) reclaims the whole lineage's bytes — accounted bytes converge to reachable chunks only
- [ ] `git grep -i 'slotLimit\|SlotsPerModel' Libraries/ Tests/` returns nothing
- [ ] The peak-memory model and headroom rationale appear in setPromptCacheByteBudget's doc comment

## Tests
- [ ] New/extended tests in Tests/MLXFoundationModelsTests: budget-driven eviction order (touch one lineage via resolve, insert until evict, assert the untouched lineage died first), byte-accounting invariant, orphaned-lineage byte reclamation (assert freed bytes, not just unreachability)
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.