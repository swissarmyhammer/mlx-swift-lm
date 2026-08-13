---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: 'Wired memory: a ticket taken at generate time is too late, thus every large model pays a per-token residency cost'
---
## What

Task `^3gh7rb5` measured a 31-times decode speed-up on `mlx-community/DeepSeek-V4-Flash-4bit` (141 GB, 43 layers, M3 Ultra 512 GiB, release build):

- default Metal wired limit: **2.10 s** for one decode step
- wired limit raised BEFORE the weights are allocated: **0.068 s** for one decode step
- wired limit raised AFTER the load: **2.10 s**, that is no change at all

## Why

MLX gives its Metal residency set a capacity of zero unless the process raises the wired limit (`wired_limit_{0}` in `mlx/backend/metal/allocator.h`), and it takes only buffers of 1 MB or less from the one heap that set holds (`heap_size_ = 1 << 20`). Every model weight tensor is larger than 1 MB. Thus each weight buffer sits outside the residency set, and Metal makes the whole weight set resident again for each command buffer, which is once for each decode step. The cost is proportional to the total weight bytes, measured at about 70 GB/s. A 141 GB model pays about 2 s for each token; a 4 GB model pays about 60 ms.

A buffer joins the residency set when it is made, thus the limit must stand BEFORE the load. It must also never fall again, because a lower limit empties the set.

## The gap in this library

`MLXLMCommon.generate(... wiredMemoryTicket:)` takes the ticket at GENERATE time, which is after the load. The measurement above shows that this ticket cannot give the speed-up it looks like it gives. No load path of this repository raises the limit:

- `Libraries/MLXLMCommon/Load.swift`, `loadWeights(modelDirectory:model:quantization:perLayerQuantization:)`
- `Libraries/MLXLLM/LLMModelFactory.swift`, the `load(from:using:...)` path

## Decide, then do

The decision needs a person, because it changes global process state for every consumer:

1. Raise the wired limit inside `loadWeights`, sized from the weight bytes and clamped to `GPU.maxRecommendedWorkingSetBytes()`, and hold it for the life of the model.
2. Or give the model factory an explicit wired-memory argument, and document that a ticket taken at generate time gives no residency benefit.
3. Or document the limit as a consumer duty, and correct the doc comment of `wiredMemoryTicket` so that it does not promise what it cannot give.

## Acceptance Criteria

- [ ] The decision is recorded on this card, with the numbers above.
- [ ] The chosen path is implemented, or the `wiredMemoryTicket` documentation states the limit honestly.
- [ ] A measurement on a second, smaller model shows the size of the effect there.

#deepseek-v4 #performance