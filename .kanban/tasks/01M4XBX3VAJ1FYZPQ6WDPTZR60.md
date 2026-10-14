---
assignees:
- claude-code
position_column: todo
position_ordinal: '9080'
title: 'DeepSeek-V4 prefill speed: compile the mHC split and expand and the SwiGLU and combine steps at prefill'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

Chunk 0 of a prefill is 0.34 s slower than mlx-vlm (HISA off), about 5.4 s on a 32k prefill. Causes: the mHC split and expand are not compiled at prefill, and SwiGLU and combine run in float32. Places: `DeepSeekV4MathHelpers.swift` 124-161 and 391-400, `DeepSeekV4HyperConnection.swift` 221-235, `DeepSeekV4MoE.swift` 527-537.

First compile these steps and keep the dtypes (same numbers). Measure a change from float32 to bfloat16 as a separate step, because it changes the numbers.

Expected gain: up to about 5 s on a 32k prefill.

## Acceptance Criteria

- [ ] The compiled steps give the same bits as the op path (unit test).
- [ ] Prefill tok/s at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance