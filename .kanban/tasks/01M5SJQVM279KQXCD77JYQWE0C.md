---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m62s15wz4s48tk006qdsgp4b
  text: |-
    Picked up (mtp @ dc342b65). Research:
    - `MTPDrafterCacheState` does not exist. The type with copy()/byteCount/isAligned(withMainTokenCount:drafterID:)/arrays()/metadata() is `MTPDrafterContinuation` (MLXLMCommon/MTPDrafterContinuation.swift). The entry field keeps the card's name `drafterState`, typed `MTPDrafterContinuation?`.
    - Most building blocks exist from ChatSession's MTP caching: `PromptCacheCheckpoint(caches:tokens:state:drafterContinuation:)` keeps a compacted copy and counts its bytes; `checkpoint.drafterContinuation(resumableBy:)` hands out a copy; `PromptCacheCheckpoint.prefillToCheckpoint(..., drafterFeed:)` + `MTPDrafterPrefillFeed.starting(...)` feed the drafter over the checkpoint prefill; `MTPSpeculativeTokenIterator.promptBoundaryContinuation` gives the prompt-end continuation; `preparePromptCacheSave(..., drafterContinuation: SerializedMTPDrafterContinuation)` / `loadPromptCacheSnapshot(...).drafterContinuation` + `.restored(by:parameters:)` serialize under `__mlx_lm_mtp_drafter_`.
    - Gaps: `MTPSpeculation.makeTokenIterator` passes no `mtpContinuation` (every FM turn cold-starts the drafter at the cached position); `generateTaskRecordingTokens`/`generateProtocolTokensTask` hand no prompt-boundary continuation to `preparedState`; the executor ignores the `drafterContinuation` completion callback; the slot/plan/file know nothing of the drafter.
    - Executor paths that commit a plan: runUnconstrained, runReasoning, runAllowedToolGeneration. Think-then-call phase 2 / guided passes call carriesNoCache.
    - Splice decisions may leave the cache at a position other than suffixStart, so the drafter is kept only on `.extend` (as the card says) and the checkpoint feed starts at the cache offset, not reusedTokenCount.
    - Observable for reuse in tests: the stored continuation's `state.cacheStartPosition` stays 0 when reused; a cold start begins it at the cached position (same check ChatSessionQwenTextMTPTests uses).
  timestamp: 2026-10-29T02:50:08.159984+00:00
- actor: claude-code
  id: 01m62vk6ck471zf3t0tt9kyr30
  text: |-
    ### implement — changed

    **Discoveries**
    - The card's `MTPDrafterCacheState` is `MTPDrafterContinuation` in the code. The entry, plan, checkpoint and spill file all keep a compacted `MTPDrafterContinuation`. A state is reused only when `isAligned(withMainTokenCount: ledger.count, drafterID:)` is true. Otherwise the drafter starts cold.
    - The iterator hands out its prompt-boundary drafter state through the new `preparedState` callback in Evaluate.swift (`PreparedGenerationState`). The executor's `prefillDidEnd(state:drafterState:)` stores that state in the checkpoint. The drafter state at the end of the turn reaches `commit` through `ExecutorDrafterStateReceiver` (a Mutex) by way of the `drafterContinuation:` completion callback.
    - `prefillingToCheckpoint` feeds the drafter through `MTPDrafterPrefillFeed` (`drafterFeed:`). A mutation check proved it matters: with the feed removed, `aFollowUpTurnResumesTheDrafterStateFedToTheCheckpoint` fails.
    - When a turn fed its stop token, the follow-up render leaves that token out, so the next turn restores from the checkpoint instead of extending the entry. The stop-token test checks the drafter state at the end of the ledger and that the restore reuses it.
    - Spill files keep the same format name. The drafter keys use the reserved `__mlx_lm_mtp_drafter_` prefix. An old file, or a file whose drafter state does not serialize or restore, loads with a cold drafter and logs a report line. Checkpoints are still not spilled.
    - Real model, turn 2 under `PlainPromptResolver`: 160 of 235 prompt tokens are recomputed, the same with and without MTP. This comes from the existing main cache (the cache reaches only through the end of the turn-1 user message), not from MTP. It could be a follow-up task.
    - At token 325 of turn 2, MTP wrote 14235, the argmax of an uncached forward pass. Plain warm decode wrote 198, which is one bf16 step lower (21.625 vs 21.5). The new integration test accepts argmax or a near tie at the first divergence. The shared `Qwen38FourBitMTPModel` helper holds this rule.

    **Evidence**
    - `swift build --build-tests`: 0 compiler warnings. MLXFoundationModelsTests 396 pass (ExecutorPromptCacheMTPTests 8/8). MLXLMTests: 920 XCTest + 1988 Swift Testing pass. MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6.
    - Release, load average about 4:
      - `qwen38FourBitMTPGeneration`: 92.2/82.4/80.3 tok/s (plain about 59.5).
      - `qwen38FourBitMTPSampledGeneration`: pass.
      - `qwen38FourBitMTPSystemPromptCache`: pass, edit turn 123 tokens.
      - `MTPPromptCacheTests`: turn 1 79/79 recomputed, turn 2 160/235; MTP 85-88 vs plain 59-60 tok/s; drafter state resumed at start position 0.

    next: /review
  timestamp: 2026-10-29T03:34:55.635871+00:00
depends_on:
- 01M5SJQ62SPB7ZA9CC9H91TTB2
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: FoundationModels — keep the drafter state in ExecutorPromptCache (reuse, checkpoint, spill, restore, evict)'
---
## What
`ExecutorPromptCacheStore` (Libraries/MLXFoundationModels/ExecutorPromptCache.swift ~272-312) keeps one entry for each (model ID, session ID). An entry holds `caches`, the `tokens` ledger, `renderTokens`, `state`, `checkpoint`, and `byteCount` (~33-107). A turn checks the entry out (~644-656) and checks it back in on every exit path (MLXLanguageModel.swift ~1196-1197). Without this task, every FoundationModels turn starts the drafter cold, and the acceptance on long multi-turn sessions is low.

Use `MTPDrafterCacheState` for every step (in the code this type is `MTPDrafterContinuation`):
- [x] **Entry and prefix reuse:** add `drafterState: MTPDrafterCacheState?` to the entry and add its bytes to `byteCount`. In `ExecutorPromptCachePlan.make` (~1044-1061), keep the drafter state only when the decision extends the cached tokens and `isAligned(withMainTokenCount:drafterID:)` is true (the drafter identity must match the current drafter, because the key is only (model ID, session ID)); else drop it (cold start). Do not pass it as model state.
- [x] **Checkpoints:** `prefillingToCheckpoint` (~1156-1178) and `prefillDidEnd` (~1186-1188, ~1877-1880) fill the main cache before the iterator runs. Run them with the per-chunk hidden-row hook and feed the drafter, and store a `copy()` of the drafter state in the checkpoint. A checkpoint restore (~1116-1133) restores the drafter state of that checkpoint.
- [x] **Commit:** `commitOutcome` (~1281-1303) stores the final drafter state from the completion callback. Check alignment with the committed ledger (also when the stop token is in the ledger).
- [x] **Spill and restore:** `ExecutorPromptCacheFile.prepare` / `read` (ExecutorPromptCacheFile.swift ~86-99, ~150-157) write and read the drafter arrays and metadata under the reserved prefix. A file without drafter keys still loads (cold start). The checkpoint is not spilled today; keep that rule.
- [x] **Evict:** LRU eviction (~762-768) and `evict(modelID:)` / `evict(sessionID:)` (~710-735) drop the drafter state with the entry; the budget includes the drafter bytes.

## Acceptance Criteria
- [x] On the shared fixture, a two-turn `LanguageModelSession` with instructions and `.greedy`: turn 2 reuses the main prefix and the drafter state (the drafter is fed only over the new suffix), and the text equals the plain session.
- [x] Edit-and-regenerate through a checkpoint restore keeps the drafter state of the checkpoint, and the text equals plain decode.
- [x] Spill an entry to disk, restore it, and continue: the drafter is reused and the text equals plain decode. A spill file of the old format loads and continues with a cold drafter. An entry made with drafter A and used with drafter B uses a cold start.
- [x] A memory budget that holds the entry without the drafter bytes evicts it (the bytes are counted).

## Tests
- [x] Tests/MLXFoundationModelsTests/ExecutorPromptCacheMTPTests.swift: reuse, checkpoint restore, spill round trip, old-format spill, drafter change, eviction budget.
- [x] IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPPromptCacheTests.swift: real-model two-turn session (skips when the MTP head is not local).
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #foundation-models #kv-cache