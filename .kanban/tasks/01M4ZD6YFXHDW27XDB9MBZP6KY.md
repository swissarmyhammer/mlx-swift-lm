---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: 'Qwen3.8 prefill speed: compile the elementwise steps of prefill'
---
## What

Only the decode step is compiled. In prefill, the elementwise steps (precise SwiGLU, `sigmoidMultiply`, beta, q/k scale, gated RMSNorm) run as separate ops. Both studies estimate about 1 % of prefill.

## Approach

1. Use shapeless `compile` for these helpers in prefill, as the decode segments do. The outputs must have the same bits.
2. Measure. Keep only if faster.

## Measure

scratchpad/qwen38bench at 32k: full mode for prefill, peak memory, decode and the reuse check (two runs each, idle GPU).

## Acceptance Criteria

- [ ] Prefill at 32k is faster, measured, or the card records why not.
- [ ] A test shows the compiled helpers give the same bits.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance