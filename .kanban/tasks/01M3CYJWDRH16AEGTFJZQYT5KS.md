---
assignees:
- claude-code
position_column: todo
position_ordinal: '8e80'
title: Check the attention mask of the BaichuanM1 sliding-window layers
---
## What

Found during ^znyes82. `BaichuanM1ModelInner.callAsFunction` (`Libraries/MLXLLM/Models/BaichuanM1.swift`) makes one mask for all layers with `createAttentionMask(h: x, cache: cache?.first)`. It gives the `CacheList` and no `windowSize`. The layers in `sliding_window_layers` use a `RotatingKVCache` with the window, thus a prefill longer than the window possibly attends to more tokens than a decode on the same layer. Compare with `mlx_lm/models/baichuan_m1.py`, which can make a separate mask for the sliding-window layers.

## Acceptance Criteria

- [ ] A test with a tiny BaichuanM1 whose window is shorter than the prompt compares a cold prefill with a split prefill and decode.
- [ ] If they differ, the sliding-window layers get a windowed mask as in mlx-lm.
- [ ] All five unit bundles pass.

#prompt-cache