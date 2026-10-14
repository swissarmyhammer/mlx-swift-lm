---
assignees:
- claude-code
depends_on:
- 01M4XBWWKVGF366NQ6EX5NV4X8
position_column: todo
position_ordinal: '9280'
title: 'DeepSeek-V4 decode speed: put the comb softmax and the sigmoids into the Sinkhorn kernel'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

mlx-vlm does the comb softmax and the pre/post sigmoids inside its Sinkhorn kernel. Our decode graph has 86 `Softmax` and 172 `CompiledBroadcastAdd` nodes for these steps in each step. Move them into `DeepSeekV4SinkhornKernel.swift`. Keep a precise `exp` so that the output stays bit-exact.

Do this after the 4-lane Sinkhorn kernel task.

Expected gain: about 0.3-0.5 ms/token.

## Acceptance Criteria

- [ ] The kernel output is bit-exact with the op path (float32 and bfloat16 tests).
- [ ] Decode ms/token at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance