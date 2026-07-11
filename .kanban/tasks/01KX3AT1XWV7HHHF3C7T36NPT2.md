---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3c8rbssczz0a9xknekd787
  text: 'Done and green. PromptCacheSlotPoolTests.swift "PromptCache slot-pool LRU eviction" suite: overflow evicts exactly the LRU slot (survivors verified by live-instance identity), resolve+store round-trip refreshes recency and protects from eviction (the untouched oldest is evicted instead), exactly-at-cap evicts nothing. References PromptCache.maxSlotsPerModel, no hard-coded 4. Verified in the 47-test PromptCache run; contributes to 100% coverage. Ready for /review.'
  timestamp: 2026-07-09T12:05:02.969661+00:00
- actor: claude-code
  id: 01kx3ydpm14q4mpc95qq91brem
  text: 'Coverage delivered in Tests/MLXFoundationModelsTests/PromptCacheSlotPoolTests.swift''s PromptCacheLRUEvictionTests suite: overflow evicts LRU, recency-refresh via resolve+store round-trip protects a slot, at-cap nothing evicted. Committed across c4e37a4 (initial), 938e83b/2f80dfc/61b58c3/dd32e04 (doc-format and naming-convention review fixes) — later extended further by 73b0668 for the configurable-slot-limit work (64jm412). Full mandated suite green throughout, review clean.'
  timestamp: 2026-07-09T17:22:19.393707+00:00
position_column: done
position_ordinal: '8980'
title: Add behavioral tests for store() LRU eviction (maxSlotsPerModel) and recency updates
---
Libraries/MLXFoundationModels/PromptCache.swift store() lines 420-423 execute under the churn test but no test asserts WHICH slots survive.

Write unit tests in Tests/MLXFoundationModelsTests asserting LRU semantics through observable resolve() behavior (slot storage is private):

1. Store maxSlotsPerModel+1 (=5) slots with disjoint token prefixes; resolve with the FIRST (oldest) prefix → rebuild (fresh instance: the LRU slot was evicted); resolve with each of the 4 newest prefixes → reuse (same instance identity as stored).
2. Recency refresh: store 4 slots, touch the oldest via a resolve+store round-trip, store a 5th → the touched slot must SURVIVE and the now-oldest untouched slot must be evicted.
3. Cap boundary: storing exactly maxSlotsPerModel slots evicts nothing (all 4 reusable).

Use KVCacheSimple with offset==tokens.count so reuse paths verify, ObjectIdentifier to prove instance identity. Reference PromptCache.maxSlotsPerModel rather than hard-coding 4.