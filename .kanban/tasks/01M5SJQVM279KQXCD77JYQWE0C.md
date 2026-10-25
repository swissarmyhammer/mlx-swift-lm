---
assignees:
- claude-code
depends_on:
- 01M5SJQ62SPB7ZA9CC9H91TTB2
- 01M5SJPFM01WFRYWMQXVGMA4MQ
position_column: todo
position_ordinal: '9080'
title: 'MTP: FoundationModels — keep the drafter state in ExecutorPromptCache (reuse, checkpoint, spill, restore, evict)'
---
## What
`ExecutorPromptCacheStore` (Libraries/MLXFoundationModels/ExecutorPromptCache.swift ~272-312) keeps one entry for each (model ID, session ID). An entry holds `caches`, the `tokens` ledger, `renderTokens`, `state`, `checkpoint`, and `byteCount` (~33-107). A turn checks the entry out (~644-656) and checks it back in on every exit path (MLXLanguageModel.swift ~1196-1197). Without this task, every FoundationModels turn starts the drafter cold, and the acceptance on long multi-turn sessions is low.

Use `MTPDrafterCacheState` for every step:
- [ ] **Entry and prefix reuse:** add `drafterState: MTPDrafterCacheState?` to the entry and add its bytes to `byteCount`. In `ExecutorPromptCachePlan.make` (~1044-1061), keep the drafter state only when the decision extends the cached tokens and `isAligned(withMainTokenCount:drafterID:)` is true (the drafter identity must match the current drafter, because the key is only (model ID, session ID)); else drop it (cold start). Do not pass it as model state.
- [ ] **Checkpoints:** `prefillingToCheckpoint` (~1156-1178) and `prefillDidEnd` (~1186-1188, ~1877-1880) fill the main cache before the iterator runs. Run them with the per-chunk hidden-row hook and feed the drafter, and store a `copy()` of the drafter state in the checkpoint. A checkpoint restore (~1116-1133) restores the drafter state of that checkpoint.
- [ ] **Commit:** `commitOutcome` (~1281-1303) stores the final drafter state from the completion callback. Check alignment with the committed ledger (also when the stop token is in the ledger).
- [ ] **Spill and restore:** `ExecutorPromptCacheFile.prepare` / `read` (ExecutorPromptCacheFile.swift ~86-99, ~150-157) write and read the drafter arrays and metadata under the reserved prefix. A file without drafter keys still loads (cold start). The checkpoint is not spilled today; keep that rule.
- [ ] **Evict:** LRU eviction (~762-768) and `evict(modelID:)` / `evict(sessionID:)` (~710-735) drop the drafter state with the entry; the budget includes the drafter bytes.

## Acceptance Criteria
- [ ] On the shared fixture, a two-turn `LanguageModelSession` with instructions and `.greedy`: turn 2 reuses the main prefix and the drafter state (the drafter is fed only over the new suffix), and the text equals the plain session.
- [ ] Edit-and-regenerate through a checkpoint restore keeps the drafter state of the checkpoint, and the text equals plain decode.
- [ ] Spill an entry to disk, restore it, and continue: the drafter is reused and the text equals plain decode. A spill file of the old format loads and continues with a cold drafter. An entry made with drafter A and used with drafter B uses a cold start.
- [ ] A memory budget that holds the entry without the drafter bytes evicts it (the bytes are counted).

## Tests
- [ ] Tests/MLXFoundationModelsTests/ExecutorPromptCacheMTPTests.swift: reuse, checkpoint restore, spill round trip, old-format spill, drafter change, eviction budget.
- [ ] IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPPromptCacheTests.swift: real-model two-turn session (skips when the MTP head is not local).
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #foundation-models #kv-cache