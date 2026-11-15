---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'Decode: measure MLX command-buffer sizing (MLX_MAX_OPS/MB_PER_BUFFER) for decode on M5'
---
## Why
- Found in ^nanqj4p. On oQ4e at 2k (flintcli graph), `MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200` gives 5.99 ms/tok against 6.35 at the default 50/50 (-6%); 400/400 gives 5.98, and 25/25 gives 7.04.
- A decode token makes ~41 command buffers of ~111 us. MLX allows 10 in flight (MAX_ACTIVE_TASKS), and the GPU idles ~1.1-1.5 ms at each token boundary.
- This is a process-wide MLX setting, not a model change, so it needs its own decision.

## What
1. Measure decode and prefill across buffer sizes for Flint and Qwen3.8 at 2k/16k, plus peak memory.
2. Decide whether the library should set it (and where), or document it for apps.

## Acceptance
- A recorded A/B table, and a decision with its evidence. Prefill and memory must not regress. #flint-perf