---
assignees:
- claude-code
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