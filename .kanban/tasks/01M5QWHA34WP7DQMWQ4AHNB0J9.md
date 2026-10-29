---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m602m2n40kep9vz08j03sz47
  text: 'Hand-off from ^achvxy1 (implemented, in review): checkpoint creation now stores the drafter continuation. `PromptCacheCheckpoint.drafterContinuation` holds a compacted copy (`MTPDrafterContinuation.compacted()`), and only when it is aligned at `tokens.count`. `byteCount` counts it. All three checkpoints carry one: leading and last-message (fed through `MTPDrafterPrefillFeed` in `prefillToCheckpoint`) and prompt end (`MTPSpeculativeTokenIterator.promptBoundaryContinuation`). That first bullet of this card is therefore done. What is left here is the restore and the policy. On restore, hand out `checkpoint.drafterContinuation?.copy()`, never the stored value itself, because forks share checkpoint values and the iterator mutates drafter caches in place. Check `isAligned(withMainTokenCount: restored position, drafterID:)` and fall back to the cold start. Today `.rebuild` + restore sets `mtpDrafterContinuation = nil`, so the turn cold-starts. The remaining `.unavailableContinuation` producer for a resumable drafter is `.trimToCommonPrefixMainOnly` (a trimmable-target rewind), which this card''s policy change removes.'
  timestamp: 2026-10-28T01:40:01.316334+00:00
- actor: claude-code
  id: 01m604xx95797rre0qsb71mz35
  text: |-
    Implementation landed (TDD; RED run showed 5 new failures for the intended reasons, GREEN run 895 XCTest + 1892 Swift Testing, 0 failures; MLXFoundationModelsTests 376).
    - Policy: `SpeculativeCacheReuseCapability.appendOnly` renamed `.restartsOnRewind` (only a resumable stateful MTP drafter has it). `RewindToCommonPrefixRule` now gives `.trimToCommonPrefix` for it (speculation stays on); `.trimToCommonPrefixMainOnly` remains only for `.mainOnlyFallback` (a stateful drafter that cannot resume).
    - `PromptCacheCheckpoint.drafterContinuation(resumableBy:)` hands out `copy()` of the stored continuation when `isAligned(withMainTokenCount: tokens.count, drafterID:)`, else nil (cold start). Never the stored value: checkpoints are shared by forks and a generation writes into drafter caches in place.
    - ChatSession: `.rebuild` + checkpoint restore sets `mtpDrafterContinuation` from that helper (ID of the turn's resumable drafter); no restore → nil. `.trimToCommonPrefix` drops the carried continuation (it stands at the pre-rewind position; the drafter cannot trim without the old hidden rows) → cold start at the common prefix.
    - Discovery: before this change the Qwen-fixture edit-and-regenerate already ran MTP and matched plain text (cold start after restore); only the restored-vs-cold-start assertion failed. The target-only-recovery test on the checkpoint session passed before the change too (kept as acceptance coverage). The trimmable-target edit (fake target, fork(history:)) was target-only with `.unavailableContinuation` before; now speculates.
    - Note: a single Swift Testing test cannot be run alone here (xctest -XCTest filters XCTest only), so each RED/GREEN step is one full MLXLMTests run (~3.5 min).
  timestamp: 2026-10-28T02:20:20.645411+00:00
- actor: claude-code
  id: 01m605cjgekr47rz7z1dwpx1n5
  text: |-
    Release results (M5, qwen38FourBit + MTP drafter). `qwen38FourBitMTPSystemPromptCache()` passed with no known issues. MTP vs plain, per turn (cached/prefilled tok, prefill ms, speculation, decode tok/s, acceptance, text/token match):
    - follow-ups 0: 0/2212, 1841 vs 1826 ms, MTP, 89.3 vs 56.6, 94.9%, exact
    - follow-ups 1: 2403/38 (plain 2404/37), 108 vs 125 ms, MTP, 87.7 vs 56.3, 94.3%, exact
    - follow-ups 2: 2544/29, 107 vs 122 ms, MTP, 89.5 vs 56.2, 96.2%, near-tie at token 1 (gap 0.125)
    - thinking 0: 0/2191, 1804 vs 1806 ms, MTP, 87.4 vs 56.4, 91.4%, exact
    - thinking 1: 2302/25 (plain 2303/24), 94 vs 110 ms, MTP, 87.5 vs 56.3, 92.0%, near-tie at token 52 (gap 0.125)
    - thinking 2: 2423/25 (plain 2425/24), 94 vs 110 ms, MTP, 85.2 vs 56.2, 87.5%, near-tie at token 0 (gap 0.375)
    - edit 0: 0/2197, 1819 vs 1807 ms, MTP, 87.2 vs 56.4, 91.4%, exact
    - edit 1: 2264/26 (plain 2265/25), 105 vs 121 ms, MTP, 89.8 vs 56.3, 97.1%, exact
    - edit 2 (edit, checkpoint restore): 2169/123, 171.0 vs 184.3 ms, MTP (no fallback), 88.8 vs 56.3, 97.1%, exact
    Restore vs cold start (same build, restore line temporarily forced to nil, then reverted): edit turn prefill 171.0 ms (restore) vs 170.7 ms (cold start), acceptance 97.1% vs 94.3%, 88.8 vs 88.6 tok/s; every other turn identical. The drafter takes the same 123 suffix tokens either way, so prefill time does not change; the restored continuation gives the drafter the earlier context and raises acceptance on this 68-token reply (small sample).
    `qwen38FourBitMTPGeneration()` twice: math/code/prose MTP 91.0/81.5/79.4 and 91.0/79.1/78.2 tok/s (baseline ~89.8/81/78), acceptance 93.9/73.6/69.0%, all match plain.
    Open point for the reviewer: a stateful MTP drafter that is NOT resumable (only a test fake today; every shipping Qwen drafter is resumable) still gets `.trimToCommonPrefixMainOnly` on a rewind, because it cannot start at a warm position; the alternative is a full rebuild.
  timestamp: 2026-10-28T02:28:21.134910+00:00
- actor: claude-code
  id: 01m605cq6c7s5dsaa2pp9sr4cd
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift, Libraries/MLXLMCommon/PromptCacheReusePolicy.swift, Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift, Tests/MLXLMTests/PromptCacheReusePolicyTests.swift. MLXLMTests 895 XCTest + 1892 Swift Testing, 0 failures; MLXFoundationModelsTests 376, 0 failures; swift-format lint clean; no new warnings. Release qwen38FourBitMTPSystemPromptCache passed (edit turn 2169 cached / 123 prefilled in 171 ms, MTP, acceptance 97.1%, exact match); qwen38FourBitMTPGeneration 91.0/81.5/79.4 and 91.0/79.1/78.2 tok/s.
    - next: /review (task left in doing; not committed)
  timestamp: 2026-10-28T02:28:25.932896+00:00
- actor: claude-code
  id: 01m605j3b35bn5gemfc7cxy68x
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (b03059a4) — 0 findings, 0 confirmed, 1 refuted; 5 source files reviewed, 4 .kanban files skipped (no validator). Caller checks verified: (1) restore hands out only PromptCacheCheckpoint.drafterContinuation(resumableBy:), which returns copy() after isAligned(withMainTokenCount: tokens.count, drafterID:), and the restored main cache stands at checkpoint.tokens.count; a non-resumable drafter gets nil. (2) .trimToCommonPrefix sets mtpDrafterContinuation = nil; a failed trim downgrades to .rebuild, which goes through the restore or the fresh-cache path. (3) .restartsOnRewind is assigned only when usesMTP && stateful && resumable; a stateful drafter that cannot resume still gets .mainOnlyFallback; the ordinary draft-model and non-speculative capabilities did not change; canContinueWithoutRebuild has the same truth table (CommittedTurnSplice, HarmonyToolRestartRule, OnyxToolRestartRule are not affected). (4) Forks share checkpoints but each restore gets its own copy(); the edit-and-regenerate test checks that the source checkpoints stay aligned after the fork resumes.
    - next: none
  timestamp: 2026-10-28T02:31:22.211519+00:00
- actor: claude-code
  id: 01m605jq7f1es9q5a5kty8j1wd
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 5 files (restore a copy of the checkpoint's drafter continuation; .restartsOnRewind keeps speculation through rewinds)
    - test: green — implementer's run on the final code: MLXLMTests 895 XCTest 0 failures + 1892 Swift Testing; FoundationModels 376; Release SystemPromptCache passed with no known issues (all 9 turns MTP), MTPGeneration 91.0/81.5/79.4 and 91.0/79.1/78.2
    - commit: b03059a4
    - review: clean — 0 findings; task done
  timestamp: 2026-10-28T02:31:42.575642+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffec80
title: 'MTP: drafter continuation through hybrid checkpoint restore and rewind in ChatSession (fork is done by PR 581)'
---
## What
PR 581 (commit 843854ec) deep-copies the drafter continuation in `RealizedCache.copy()`, so fork works. But a stateful MTP drafter has the capability `appendOnly` (Libraries/MLXLMCommon/PromptCacheReusePolicy.swift), so every rewind gives `.trimToCommonPrefixMainOnly`: the drafter is dropped, the turn runs target-only, and MTP stays off for later turns on that cache. On Qwen3.8 a rewind happens through a checkpoint restore on each multi-turn chat whose history is rendered again, so this matters.

- [x] Checkpoint creation (ChatSession.swift ~1562-1642): store a copy of the `MTPDrafterContinuation` at the checkpoint position in `PromptCacheCheckpoint` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift ~33-65) and count its bytes.
- [x] Checkpoint restore (~1508-1528) restores that continuation together with the main caches.
- [x] Policy: when a checkpoint gives an aligned continuation, keep speculation on; else use the cold start. Do not use `trimToCommonPrefixMainOnly` for the MTP strategy.

## Acceptance Criteria
- [x] Edit-and-regenerate (a turn that goes back to an earlier message end) with MTP gives the same greedy text as plain decode and runs MTP (no target-only turn).
- [x] After a turn that ran target-only for any reason, the next turn runs MTP again (cold start or restored continuation).

## Tests
- [x] Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift: checkpoint restore with MTP, recovery after a target-only turn.
- [x] Tests/MLXLMTests/PromptCacheReusePolicyTests.swift: the MTP strategy with an aligned checkpoint continuation keeps rewind with speculation.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session