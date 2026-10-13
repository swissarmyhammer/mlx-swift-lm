---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'DeepSeek-V4 speed: one compiled trace from the tail of a layer to the head of the next layer'
---
## What

Task ^rnnxh8d runs each DeepSeek-V4 decode step through two compiled traces for each layer (`DeepSeekV4DecoderLayer.decodeStep` in `Libraries/MLXLLM/Models/DeepSeekV4.swift`): one before the cache stage and one after it. That is 86 trace calls for each step.

A Time Profiler run of 21 decode steps at 128 context (scratch `prof2/tp128.trace` of ^rnnxh8d) shows `mlx::core::detail::compile_replace` at 46 ms, about 2.2 ms for each step. Each call also passes every weight of the layer as a compile input.

Evaluate one trace that runs the tail of layer i and the head of layer i+1 (the pattern `CompiledDecodeSegment` of `Libraries/MLXLMCommon/CompiledDecodeSegments.swift` gives Qwen3.5). That halves the trace calls and lets the expand of layer i fuse with the collapse of layer i+1. Measure it before keeping it.

## Acceptance Criteria

- [ ] A decode-step measurement on the real checkpoint, GPU idle, at 128 and 16k context, before and after.
- [ ] The parity tests of ^rnnxh8d (same bits as the op path, float32 and bfloat16) still pass.
- [ ] Greedy ids on the mlx-vlm fixture do not change.
#deepseek-v4 #performance