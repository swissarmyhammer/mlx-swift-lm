---
assignees:
- claude-code
position_column: done
position_ordinal: 8b80
title: 'MTP: a windowed drafter ring copies O(window) rows several times per round (multi-row commit, wrapped-ring rewind, regrow)'
---
Found while measuring ^hkt3a09 (drafter attention window, `GenerateParameters.mtpDrafterAttentionWindow`). With a RotatingKVCache drafter of window W, each MTP round re-linearizes the ring: a multi-row commit (appendDrafterState of the accepted tokens) goes through `updateConcat` (temporalOrder + trim/concatenate of W + reserve rows), a rewind of rejected drafts on a wrapped ring linearizes it again (`trim` → temporalOrder), the next single-row write regrows the shortened buffer (concatenate with zeros), and with a kept prefix each single-row step returns a concatenated view. Measured on Qwen3.8-27B-4bit (text, 32k context): a 16k drafter ring cost 55.2 ms per round vs 47.0 ms for the whole-context drafter, although the block-5 proposal micro-benchmark shows equal drafter attention cost there (6.3 vs 5.8 ms). At 100k on the router path, windows cut rounds from 90 to 71 ms, so the ring overhead hides part of the attention saving.

The window default stays the whole context (acceptance falls with every window and outweighs the saving, see ^hkt3a09's sweep), so this only matters if a window is ever made default or used by callers.

- [x] Measure the per-round drafter commit + rewind cost by window (extend MTPRoundCostBreakdownTests).
- [x] Write multi-row ring commits in place (split at the ring end) and rewind a wrapped ring by moving `idx` back within the reserve instead of linearizing; keep the prefix-keeping mask semantics and the save/restore format.
- [x] Re-run MTPDrafterWindowSweepTests and record whether any window then beats the whole context.

#mtp #perf

## Result 2026-11-03 (785d0436, ae2230a7, 720adf2a, b5a9706f, 31c75c57; worktree 63dd0712, a22448b5, 61bd8de5, 7785c5cd, b5eb1afe)
- RotatingKVCache with a rewind reserve: multi-row commits up to reserve+1 rows are written in place (split at the ring end); a wrapped-ring rewind moves idx back across the wrap instead of linearizing, and the next writes refill the freed slots (no regrow). Old path kept for prefill chunks, rings without a reserve and the first wrap of a full-history cache; package switch `writesRingInPlace` (default true) selects it for tests. Masks, offsets, saved state and metaState unchanged; prompt-cache format unchanged.
- One documented difference: right after a multi-row commit the old path briefly held up to count-1 extra old rows (higher maxTrimCount, lower evictedTokenCount); the in-place ring holds exactly window + reserve, which covers every drafter rewind (reserve 2 x block size).
- RotatingKVCacheInPlaceRingTests: 8 XCTest cases compare in-place vs old path (keys, values, masks, offsets, views, saved rows, snapshot round trips) over keep 0/2/3, reserve 1/4/10, step 1/4/256, wraps, cross-wrap rewinds, post-prefill eviction.
- Ring-only microbench (load 26–28): in place 0.3–0.8 ms/round faster at most sizes, ~1–2% of a 35–50 ms round. MTPRoundCostBreakdownTests: path differences (−3.2 to +2.0 ms) below run noise (±3 ms).
- Window sweep (cut to 32k and 100k, text and router) vs whole context: text 32k 0.96–0.97x, router 32k 0.91–0.96x, text 100k 0.88–0.95x, router 100k 0.88–0.97x. At 32k the ring overhead is gone (text 16k 36.5 vs 35.9 ms/round, was 55.2 vs 47.0); at 100k windows save 1–6 ms/round but acceptance falls more (text 100k 16k: 0.875 vs 0.985, 2.81 vs 3.56 tokens/verify). Default stays whole context.
- Sweep now selected by test name (8 @Test functions, 100k split wide/narrow per lock hold), no environment filters.
- Release guards pass: qwen38FourBitMTPGeneration (bs4/pMin0.75 fastest), qwen38FourBitMTPSystemPromptCache (edit turn 123 tokens, equal to plain).
- review: 17 findings on the first 3 commits, fixed in 7785c5cd/b5eb1afe; re-review 0 findings. On mtp with ^zn3wdms merged: five bundles green (MLXLMTests 938 + 2123, 1 known issue).
