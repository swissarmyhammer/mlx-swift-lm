---
assignees:
- claude-code
position_column: todo
position_ordinal: '8680'
title: 'Loader: release checkpoint tensors that prepareCheckpoint replaces before materialization (Flint fusion load peak)'
---
## Why
^j4pqtwr fuses Flint q/k/v and gate/up projections in `FlintModel.prepareCheckpoint`. The concatenated tensors are new buffers, and the parts stay alive until materialization: `installWeights` (Libraries/MLXLMCommon/Load.swift) keeps its own `checkpoint` value while `prepareCheckpoint` runs, and the freed parts then sit in the MLX buffer cache until `materializeModelForInference` calls `Memory.clearCache()`.

Measured with `/usr/bin/time -l flintcli <dir> chat 200 hello` on the M5: oQ8e peak memory footprint 42.7 GB before fusion, 70.2 GB after (weights 39 GB); oQ4e 38.6 GB after (weights 22 GB). Steady-state RSS is unchanged. The allocator releases its cache near the GPU limit, but the transient peak is close to the 77.8 GiB working set on the M5 Ultra for oQ8e.

## What
- Let a model's checkpoint preparation free the tensors it replaces before the new ones evaluate (for example, hand `prepareCheckpoint` the only reference, or evaluate replaced tensors in bounded groups with a cache clear between them).
- Keep `ModelCheckpoint` semantics for other models.

## Acceptance
- oQ8e load peak footprint back near the unfused 42.7 GB (measure the same way).
- Flint greedy ids unchanged (flintcheck.sh MATCH on all four prompts).
- All five test bundles green.
#flint-perf