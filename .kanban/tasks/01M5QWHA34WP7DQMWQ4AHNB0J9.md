---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m602m2n40kep9vz08j03sz47
  text: 'Hand-off from ^achvxy1 (implemented, in review): checkpoint creation now stores the drafter continuation. `PromptCacheCheckpoint.drafterContinuation` holds a compacted copy (`MTPDrafterContinuation.compacted()`), and only when it is aligned at `tokens.count`. `byteCount` counts it. All three checkpoints carry one: leading and last-message (fed through `MTPDrafterPrefillFeed` in `prefillToCheckpoint`) and prompt end (`MTPSpeculativeTokenIterator.promptBoundaryContinuation`). That first bullet of this card is therefore done. What is left here is the restore and the policy. On restore, hand out `checkpoint.drafterContinuation?.copy()`, never the stored value itself, because forks share checkpoint values and the iterator mutates drafter caches in place. Check `isAligned(withMainTokenCount: restored position, drafterID:)` and fall back to the cold start. Today `.rebuild` + restore sets `mtpDrafterContinuation = nil`, so the turn cold-starts. The remaining `.unavailableContinuation` producer for a resumable drafter is `.trimToCommonPrefixMainOnly` (a trimmable-target rewind), which this card''s policy change removes.'
  timestamp: 2026-10-28T01:40:01.316334+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8380'
title: 'MTP: drafter continuation through hybrid checkpoint restore and rewind in ChatSession (fork is done by PR 581)'
---
## What
PR 581 (commit 843854ec) deep-copies the drafter continuation in `RealizedCache.copy()`, so fork works. But a stateful MTP drafter has the capability `appendOnly` (Libraries/MLXLMCommon/PromptCacheReusePolicy.swift), so every rewind gives `.trimToCommonPrefixMainOnly`: the drafter is dropped, the turn runs target-only, and MTP stays off for later turns on that cache. On Qwen3.8 a rewind happens through a checkpoint restore on each multi-turn chat whose history is rendered again, so this matters.

- [ ] Checkpoint creation (ChatSession.swift ~1562-1642): store a copy of the `MTPDrafterContinuation` at the checkpoint position in `PromptCacheCheckpoint` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift ~33-65) and count its bytes.
- [ ] Checkpoint restore (~1508-1528) restores that continuation together with the main caches.
- [ ] Policy: when a checkpoint gives an aligned continuation, keep speculation on; else use the cold start. Do not use `trimToCommonPrefixMainOnly` for the MTP strategy.

## Acceptance Criteria
- [ ] Edit-and-regenerate (a turn that goes back to an earlier message end) with MTP gives the same greedy text as plain decode and runs MTP (no target-only turn).
- [ ] After a turn that ran target-only for any reason, the next turn runs MTP again (cold start or restored continuation).

## Tests
- [ ] Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift: checkpoint restore with MTP, recovery after a target-only turn.
- [ ] Tests/MLXLMTests/PromptCacheReusePolicyTests.swift: the MTP strategy with an aligned checkpoint continuation keeps rewind with speculation.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session