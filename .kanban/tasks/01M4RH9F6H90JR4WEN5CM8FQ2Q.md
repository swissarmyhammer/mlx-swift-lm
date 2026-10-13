---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rzchkqgg8566y5qk2k205c
  text: |-
    ### Research 2026-10-12 (session scratchpad dl/, Python sources)
    - This is probably most of the decode growth from 104 ms at 128 context to about 128 ms at 16k.
    - `DeepSeekV4Cache.swift:210-213, 359-366` runs `concatenated([chunks, added])` on each step, also when no chunk is complete. `DeepSeekV4Attention.swift:516-518` concatenates all pooled chunks behind the window again. The buffer size changes, thus the MLX buffer cache cannot reuse it. At 16k that is about 0.4 GB of new buffers for each step.
    - Python adds a pooled row only when a chunk is complete (`mlx_vlm .../language.py:426-473, 1011-1028`; `pr1195 deepseek_v4.py:1073-1086`).
    - Project only the new token, in BF16. Python uses float32 only in the softmax (`mlx_vlm .../language.py:390-394, 618-654`). Remove the float32 cast before `wkv` / `wgate` (`DeepSeekV4Compressor.swift:223`). This changes numbers, so record the decision.
  timestamp: 2026-10-12T21:13:05.911731+00:00
position_column: todo
position_ordinal: '8480'
title: 'DeepSeek-V4 speed: compressor decode state that does not pool the raw rows again on each step'
---
## What

At each decode step, the pooled cache joins up to 127-255 raw hidden rows with the new token and pools them through the compressor again: `DeepSeekV4Cache.swift:201-221`. Each step also copies all pooled chunks: `DeepSeekV4Cache.swift:359-367`. This cost grows with the context length.

mlx-lm PR #1195 keeps the projected rows and makes a chunk only when a window is full (Metal kernel `dsv4_compressor_overlap_emit`).

## Approach

- Keep the projected (not raw) rows in a buffer that grows in steps, as `KVCacheSimple` does.
- Make a new pooled chunk only when a compress window is full.
- Write into a preallocated chunk buffer, not a new concatenation on each step.
- Keep the rule of `DeepSeekV4CacheTests`: the chunks a block pools must not depend on how the block was cut.
- Keep the prompt-cache state format, or move it to a new version with a restore check.

## Acceptance Criteria

- [ ] A decode step does no compressor work when no compress window is full.
- [ ] `DeepSeekV4CacheTests` and the other DeepSeek-V4 unit tests pass.
- [ ] A prompt cache saved before the change restores, or fails closed with a clear error.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after, at 4k and at 32k context.
#deepseek-v4 #performance