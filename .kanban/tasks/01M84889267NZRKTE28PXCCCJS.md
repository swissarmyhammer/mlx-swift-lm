---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m848mpdvesfbmag9dbxybysn
  text: |-
    ## Attribution (stable 24bd6047, M5 Ultra). Profile mode: synchronous eval of one decode step, median of 64
    Method:
    - Temporary env-gated skips (`INK_SKIP`), saved as scratchpad/ablation.patch and not committed.
    - Dispatch count: `MLX_MAX_OPS_PER_BUFFER=1` plus xctrace command-buffer submissions at 8 vs 40 steps, which gives **~525 dispatches per token (~11 per layer visit)**.
    - Measured read bandwidth: 725 GB/s (2 GiB sum).

    | component | ms/token 2k | ms/token 32k | weight/KV bytes per token | floor at 725 GB/s |
    |---|---|---|---|---|
    | whole step (GPU eval) | 10.71 | 11.97 | ~2.7 GB | ~3.7 ms |
    | MoE router (logits + top-k, 2 kernels) | 0.85 | 0.85 | 59 MB | 0.08 |
    | MoE experts (gate-up + down) | ~3.2 | ~3.3 | 1.72 GB | 2.4 |
    | attention projections (fused qkv + o_proj) | 1.66 | 1.74 | 0.70 GB | 0.97 |
    | attention prologue + KV cache write | 0.62 | 0.43 | small | – |
    | **decode attention kernel** | **1.55** | **2.90** | 0.25 GB / 1.3 GB | 0.35 / 1.8 |
    | lm_head + softcap | 0.26 | 0.37 | 0.26 GB | 0.36 |
    | rest: depth mix ×2, message controllers, convolutions, embed | **2.9** | 2.9 | small | ~0 |

    ## Findings
    1. **Decode attention is about 4x off its floor.** Sliding visits (≤1024 keys) launch one threadgroup per query head, only 24 threadgroups (`singleThreadgroupDecodeKeys`). The split path uses 256 keys per split, which gives 8×S/256 threadgroups. Both leave most of the GPU idle. This is now the top card, ^xdrv99b, with its scope widened to sliding layers.
    2. **The "rest" costs 2.9 ms for almost no bytes.** The depth-mix kernel runs one row per threadgroup, so at decode it is a single threadgroup, twice per visit. Add a card: spread the depth mix over threadgroups.
    3. **The router costs 0.85 ms for 59 MB.** Fusing its two kernels (^cyw9v2w) saves at most ~0.4 ms.
    4. **The projections reach ~58% of bandwidth.** ^2gcj3fq, three matvecs into one in loop passes 2 and 3, saves a little.
    5. **lm_head is 0.26 ms.** ^00csn7w can gain at most ~0.03 ms, so drop it.
    6. **The cache write is inside the 0.62 ms prologue+write.** Run ^8yq29yp last, if at all.

    New order: ^xdrv99b (decode attention occupancy), depth-mix card, ^cyw9v2w, ^2gcj3fq, ^s95r8b1, ^8yq29yp; then the prefill cards.
  timestamp: 2026-11-23T13:13:34.395383+00:00
depends_on:
- 01M8487ZP6NSQ1E0FEY1CEY7KX
position_column: done
position_ordinal: ffffde80
title: 'Inkling 2 decode: attribute the GPU time of a decode step by component (measure only)'
---
## Why
At 2k, a decode step is 10.8 ms on the GPU, against a bandwidth floor of ~2.5 ms (the weights read per token). Before any card changes code, find which components hold that time, so the later cards are ranked by evidence.

## What
- Time the components with temporary, env-gated skips (not committed), or with microbenchmarks in inklingcli: attention (prologue, cache write, decode attention, output projection), MoE (router logits, routing, gate-up, down), depth mix and message controllers, and lm_head with its softcap.
- Count the dispatches per layer visit and per token. For example, read them off the code path, or use MLX ops per command buffer together with the command buffers in an xctrace capture.
- Re-rank the later inkling-perf cards by the result, and add or drop cards as the numbers say.

## Acceptance
- [ ] The card records a table of ms per token and dispatches per token by component, at 2k and at 32k.
- [ ] No library change is committed. #inkling-perf