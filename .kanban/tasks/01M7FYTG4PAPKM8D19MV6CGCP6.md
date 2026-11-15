---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'Flint: banded sliding-window prefill attention (the array mask costs ~8-9% of prefill)'
---
## Why
Found in ^tty395a. At 2048-token prefill chunks on oQ4e, an ablation that gave the sliding layers the plain causal mask (numerically wrong, timing only) raised prefill from 8021 to 8762 tok/s at 8k and from 7312 to 7911 at 16k. 35 of 49 Flint layers slide over a window of 513. Each sliding layer of a 2048-token chunk runs SDPA over 2048 x (512 + 2048) scores with an explicit array mask, so no key block is skipped, although each query needs only 513 keys. mlx-vlm 0.7.12 pays the same cost (RotatingKVCache.make_mask array), so this goes beyond the reference.

## What
- Measure first: a Metal trace of one prefill chunk, sliding SDPA vs full SDPA.
- Try splitting the query rows of a sliding layer into sub-blocks, each attending only to the keys of its own window (a smaller array mask per sub-block), or a kernel route that skips key tiles outside the band.
- Keep the attention sinks.

## Acceptance
- Prefill tok/s at 2k, 8k and 16k improves against a same-session baseline (scratchpad pfbench.sh with `@prompt_706.txt`).
- Greedy ids of flintcheck MATCH x4; float32 logit parity holds.
- Decode does not regress.
- All five bundles are green. #flint-perf