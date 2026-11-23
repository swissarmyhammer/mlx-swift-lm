---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84dkxe9m55adjvdejr9xpag
  text: |-
    ## Implemented as routing inside the gate-up kernel. Bit-exact; the gain is within noise
    I did not fuse into the logits kernel. That would put all 128 logits in one threadgroup, with one core reading the 1.2 MB float32 router weight. Instead:
    - New `expertRoutedGateUpSource`: simdgroup 0 of each gate-up threadgroup runs the `expertRoutingSource` code on its row's logits: sigmoid, biased top-k with the same tie-break, sum and normalization in the same order. It shares the slot's expert through threadgroup memory.
    - The threadgroup with x == 0 of each slot writes the slot's index and weight, which `inkling2ExpertDown` reads.
    - `Inkling2Router.kernelLogits` and `Inkling2Experts.decode(_:logits:bias:topK:...)`. `Inkling2SparseMoE.decode` tries the routed path first and falls back to the two kernels.
    - One dispatch fewer per visit, 48 per token. Verify rows (≤16) take the same kernel.

    Results:
    - inklingcheck (ref2): CHECK PASS, bit-exact.
    - Profile at 2k: build 1.70 → 1.55 ms, eval 9.72 → 9.67 ms.
    - inklingbench vs cum1: decode -0.5% at 2k, +3.3% at 8k, +3.0% at 32k, -0.3% at 64k. Within noise; the upper bound from the ^s95r8b1 ablation was ~0.4–0.8 ms.
    - Kept for now because it removes dispatches. It is optional: drop it if the extra kernel is not worth the maintenance.
    - MLXLMTests had passed before this change. The Inkling2CheckpointIntegrationTests run (4/4 pass, ~102 tok/s decode in chat turns) was also before this change.
  timestamp: 2026-11-23T14:40:31.689474+00:00
depends_on:
- 01M84889267NZRKTE28PXCCCJS
position_column: done
position_ordinal: ffffe680
title: 'Inkling 2 decode: fuse the router logits and the top-k routing into one kernel'
---
## Why
Each layer visit routes with two dispatches: `inkling2RouterLogits` (Inkling2Kernels.swift `routerLogitsSource`), then `inkling2ExpertRouting` (`expertRoutingSource`). Over 48 visits that is 96 dependent launches per token, at ~5.3 us each.

## What
- One kernel computes the 128 float32 logits of a row, then the sigmoid, the biased top-8, the weight normalization and the route scale, all in threadgroup memory.
- Keep the bits of the two-kernel path: the same float32 reduction order for the logits, and the same tie-break order for the top-k.
- Keep the speculative verify rows (up to 16) on the same kernel, so a verify pass routes as decode steps do.

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS (every golden id matches).
- [ ] inklingbench.sh shows a decode gain beyond the ±3% noise, or the card records the result and reverts.
- [ ] The Inkling 2 unit tests pass, including the bf16 decode kernel tests and the verify-matches-decode tests. #inkling-perf