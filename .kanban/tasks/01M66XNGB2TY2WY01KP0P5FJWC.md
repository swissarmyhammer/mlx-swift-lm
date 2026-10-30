---
assignees:
- claude-code
position_column: todo
position_ordinal: 8b80
title: 'MTP: a windowed drafter ring copies O(window) rows several times per round (multi-row commit, wrapped-ring rewind, regrow)'
---
Found while measuring ^hkt3a09 (drafter attention window, `GenerateParameters.mtpDrafterAttentionWindow`). With a RotatingKVCache drafter of window W, each MTP round re-linearizes the ring: a multi-row commit (appendDrafterState of the accepted tokens) goes through `updateConcat` (temporalOrder + trim/concatenate of W + reserve rows), a rewind of rejected drafts on a wrapped ring linearizes it again (`trim` → temporalOrder), the next single-row write regrows the shortened buffer (concatenate with zeros), and with a kept prefix each single-row step returns a concatenated view. Measured on Qwen3.8-27B-4bit (text, 32k context): a 16k drafter ring cost 55.2 ms per round vs 47.0 ms for the whole-context drafter, although the block-5 proposal micro-benchmark shows equal drafter attention cost there (6.3 vs 5.8 ms). At 100k on the router path, windows cut rounds from 90 to 71 ms, so the ring overhead hides part of the attention saving.

The window default stays the whole context (acceptance falls with every window and outweighs the saving, see ^hkt3a09's sweep), so this only matters if a window is ever made default or used by callers.

- [ ] Measure the per-round drafter commit + rewind cost by window (extend MTPRoundCostBreakdownTests).
- [ ] Write multi-row ring commits in place (split at the ring end) and rewind a wrapped ring by moving `idx` back within the reserve instead of linearizing; keep the prefix-keeping mask semantics and the save/restore format.
- [ ] Re-run MTPDrafterWindowSweepTests and record whether any window then beats the whole context.

#mtp #perf