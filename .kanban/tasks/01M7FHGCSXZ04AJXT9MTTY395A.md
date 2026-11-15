---
assignees:
- claude-code
position_column: todo
position_ordinal: '8480'
title: 'Flint: profile prefill and close the gap to mlx-vlm (5.6–6k vs 7–7.7k tok/s)'
---
## Why
On oQ4e, Swift prefill runs at 5.6–6.0k tok/s from 2k to 16k context. mlx-vlm 0.7.12 gets 7.0–7.7k tok/s on the same machine in 2048-token chunks (scratchpad `pyscripts/pybench.py`). Decode is already faster than Python (7.7 vs 13.9 ms/token), so prefill is the gap.

## What
- **Compare like with like.** Check chunk size first: our `prefillStepSize` versus 2048.
- **Find the cost** with ablations or a Metal trace. Suspects:
  - the MoE sort/gather path for many tokens (`projectSwitchExperts`; does `callAndWeightedReduce` / `weightedExpertUnsort` apply? Flint calls `switchMLP(x, indices)` plus `weightedExpertSum` and never uses the direct-reduction path);
  - the float32 casts of the routed output;
  - sliding vs full attention masks at prefill;
  - the attention kernel with sinks.
- **Port what mlx-vlm does better.**

## Acceptance
- Prefill tok/s at 2k, 8k and 16k improves against a same-session baseline.
- Float32 logit parity is unchanged.
- Greedy ids are identical.
- All five bundles are green. #flint-perf