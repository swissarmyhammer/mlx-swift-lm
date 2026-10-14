---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'DeepSeek-V4 speed: one fused Metal lightning-indexer kernel (no per-head score tensor)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 2).

Our indexer (`DeepSeekV4Indexer.swift`, about lines 339-372) writes a per-head float32 score tensor (4 GiB for each 2048-token step at 32k) and then reduces it, in a compiled function since ^er1vnkw. llama.cpp uses one kernel that calculates `sum_h w_h * relu(q_h . k)` plus the mask with simdgroup matmul, and never writes the per-head scores (PR 25893, `kernels/fa_aux.metal:338`).

Use float32 accumulation so that the selected chunks stay as near as possible to the current path. llama.cpp uses f16 Q and K; do not copy that.

Estimate: prefill +4 to +8 % at 32k, decode -0.2 to -0.5 ms/token. Not bit-exact. Risk: medium.

## Acceptance Criteria

- [ ] A unit test compares the kernel scores with the op path (tolerance) and counts the changed selections.
- [ ] Prefill tok/s, peak memory and decode ms/token at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance