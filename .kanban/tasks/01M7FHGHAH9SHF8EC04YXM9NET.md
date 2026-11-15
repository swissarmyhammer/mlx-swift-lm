---
assignees:
- claude-code
depends_on:
- 01M7FHFQT76AN9817CJJ4PQTWR
- 01M7FHFWPMBDPGDSD9BG68BHJ1
- 01M7FHG1KPDNCWYR3DCPG2HSB8
- 01M7FHG7TA2MGDAS4Q5NANQJ4P
position_column: todo
position_ordinal: '8580'
title: 'Flint: measure whether DFlash2 speculative decoding can pay off (8-token verify cost)'
---
## Why
Flint has no MTP head. `incoai/Flint-DFlash2` exists (block 8, target layers 5/15/26/36/46). On the M5, DFlash2 for Qwen3.8-27B lost because a round cost ~2.9 plain steps (see memory `dflash2-branch-status`; branch `origin/dflash2` is parked). An 8-token MoE verify touches up to 48 distinct experts per layer instead of 6, so it may cost far more than a 1-token step.

## What
- Do this after the fusion, cast, compile and sliding-attention cards land, so the baseline step is the fast one.
- Measure the wall time of an 8-token verify forward against a 1-token step on oQ4e, at 2k and 16k context.
- Measure the drafter forward cost.
- Estimate break-even acceptance. Build the integration only if the measured round cost leaves real headroom.

## Acceptance
- A written measurement on this card: verify/step ratio, drafter cost, and a go/no-go.
- If go: a follow-up card for the integration with ChatSession and the prompt cache, per memory `goal-speed-with-cache-reuse`. #flint-perf