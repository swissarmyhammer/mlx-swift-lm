---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4y6y38b7g7ntezpnrteghhb
  text: |-
    Research after ^f4ztxgj (HEAD 155eec4). The window slab cannot give speed at 32k now.

    1. Prefill. `DeepSeekV4SparseAttentionKernel.canRun` sends each block with 256 threadgroups or more to the fused kernel. With 64 heads and a head block of 16, that is each block of 64 tokens or more. The fused kernel already reads only the visible window rows of each query (one range of at most 128 rows). This is the goal of the slab, with no slab and no mask. The op paths that the card names (`windowAttention` slices, `maskedAttention`, `attentionOverRankedTile`) run only for blocks of 2 to 63 tokens.
    2. The 32k bench prompt is 32118 tokens with prefill step 2048 and `.balanced` chunking (`PrefillParameters.forEachChunk`, which reserves 1 token). The blocks are 15 x 2008 tokens, 1 x 1997 tokens and 1 token. All the multi-token blocks go to the fused kernel. Thus the op path runs for 0 blocks of 2 to 63 tokens at 32k. The time that the slab could remove is 0 ms.
    3. Decode and the reserved last token. The window cache is `RotatingKVCache(maxSize: slidingWindow = 128)`. Thus a 1-token block reads 128 window rows or fewer, and `readsWithNoMask` gives no mask. A slab of 256 keys cannot read fewer rows. There is no decode effect (the card also says this).
    4. The cache concat and the window buffer do not change with a slab: the fused kernel reads the rows that `RotatingKVCache.update` gives back.

    Thus I did not change the code. The unit-test and greedy-parity criteria do not apply because no code changed.

    Osaurus idea 4 (prefill step 4096). Its premise is that the attention cost per row does not grow with the block size. The fused kernel already makes that true, so I measured it. I changed only the scratchpad bench (`DSV4_PREFILL_STEP` env var; the default stays 2048). M3 Ultra, GPU idle before each run, args `<snap> 32768 128 8192 1024`:
    - step 2048 (f4z-after-full-1/2 + new control ngj-step2048-full-1): 559.9 / 559.5 / 559.5 tok/s, peak 143 GiB, decode 30.4 / 30.3 / 30.4 ms/token.
    - step 4096 (ngj-step4096-full-1/2): 579.6 / 579.0 tok/s, peak 145 GiB, decode 30.4 / 30.3 ms/token.
    - Result: prefill +3.5 % (inside the 0 to +5 % estimate), peak memory +2 GiB, no decode change. HARD reuse check PASS in all runs.
    The prefill step is a caller parameter (`GenerateParameters.prefillStepSize`; the library default is 512 for all models). The repository has no DeepSeek-V4 step to change. I added a new task for a model-preferred prefill step.
  timestamp: 2026-10-14T22:01:13.227048+00:00
- actor: claude-code
  id: 01m4y6ymdeaty5xr2tbpkcy756
  text: |-
    ### implement — no-change
    - evidence: The window slab can remove 0 ms at 32k. Every prefill block of the 32k run (15 x 2008, 1 x 1997 tokens) goes to the fused kernel of ^f4ztxgj, which already reads only the visible window rows (128 or fewer). The op path runs for 0 blocks of 2 to 63 tokens. Decode and the last token read the 128-row RotatingKVCache window, so a slab cannot read fewer rows. Repository files changed: 0. Prefill step 4096 measured: 579.6 / 579.0 tok/s, peak 145 GiB, decode 30.4 / 30.3 ms/token. Step 2048: 559.9 / 559.5 / 559.5 tok/s, peak 143 GiB, decode 30.4 ms/token. Logs: scratchpad ngj-step4096-full-1.log, ngj-step4096-full-2.log, ngj-step2048-full-1.log.
    - next: a person decides to close this card as overtaken by ^f4ztxgj. The step-size finding is in the new task ^jyfka2h.
  timestamp: 2026-10-14T22:01:30.798881+00:00
position_column: doing
position_ordinal: '80'
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