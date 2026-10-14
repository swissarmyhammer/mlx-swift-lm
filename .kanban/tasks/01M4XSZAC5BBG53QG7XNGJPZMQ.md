---
assignees:
- claude-code
position_column: todo
position_ordinal: 8b80
title: 'DeepSeek-V4 prefill speed: each query block reads only its own window rows, not all window keys of the block'
---
## What

Found on 2026-10-14 in a study of osaurus (scratchpad `osaurus/REPORT.md`, idea 1).

At prefill, each query scores all window keys of the block (127 earlier rows plus the full 2048-row block, 2175 keys), and the mask hides all but 128. This occurs in all 43 layers: `DeepSeekV4Attention.swift` about line 733, `attentionOverRankedTile` 1463-1487, `maskedAttention` 807-835, `windowAttention` 927. The query tiles cut the queries, but not the keys.

Better form: each 128-row query block reads one 256-key slab (the previous block plus its own block) with one mask, in one batched call (vmlx-swift commit 06185160, `NaiveN05FlashPrefillAttention.swift:19-53`). For the ranked-chunk path, join the slab with the picked pool rows of that query block.

Estimate: 8.5 times fewer window keys at 32k, prefill +15 to +30 %. No decode effect. Same math; probably bit-exact or very near. Risk: low.

Also measure a prefill step of 4096 after this change (osaurus idea 4: 0 to +5 %, more peak memory). Keep it only if it is faster and the peak memory is acceptable.

## Acceptance Criteria

- [ ] A unit test compares the slab path with the current path.
- [ ] Prefill tok/s, peak memory and decode ms/token at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance