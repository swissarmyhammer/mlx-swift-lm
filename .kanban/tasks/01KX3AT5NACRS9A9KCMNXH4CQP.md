---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3c8sn2w7py55ff9e08yqc4
  text: 'Done and green. PromptCacheSlotPoolTests.swift "PromptCache eviction API and per-model isolation" suite: evictAll drops every model''s slots (both rebuild after), remove() is strictly per-model (bystander still reuses its exact instance), and identical token sequences never cross model IDs (model B rebuilds, model A''s slot intact). Covers previously-untested PromptCache.evictAll (428-430) and remove (433-435). Verified in the 47-test run; 100% coverage. Ready for /review.'
  timestamp: 2026-07-09T12:05:04.290536+00:00
position_column: doing
position_ordinal: '8380'
title: Add tests for PromptCache.evictAll/remove and per-model isolation
---
Libraries/MLXFoundationModels/PromptCache.swift — uncovered lines 428-430 (evictAll) and 433-435 (remove(modelID:)). Also no test proves slots are isolated per modelID.

Write unit tests in Tests/MLXFoundationModelsTests driving a fresh PromptCache() actor instance:

1. evictAll: store slots for two model IDs, evictAll(), resolve for both → rebuild (fresh instances).
2. remove is per-model: store slots for model A and model B, remove(modelID: A) → A rebuilds, B still reuses its stored instance.
3. Multi-model isolation: store a slot for model A; resolve for model B with the SAME tokens → rebuild (A's cache instance must never be returned for B), and A's slot is still intact afterward.