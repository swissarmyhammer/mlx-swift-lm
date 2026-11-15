---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'Flint: cut the L=1 KV cache write cost (MLX Swift subscript set, ~0.66 ms/token)'
---
## Why
- Measured in ^nanqj4p (oQ4e, 2k): keeping the cache write but attending over 1 key costs -0.30 ms against the baseline, while skipping the cache update entirely costs -0.96 ms. So the L=1 write path costs about 0.66 ms per token (13.5 us/layer), and 0.32 ms of that is Swift graph build.
- `RotatingKVCache.update` at L=1 costs 4.8 us CPU and `KVCacheSimple.update` costs 6.9 us. Each MLX Swift subscript set costs ~2.5 us (5x a plain op) and adds two Reshape nodes (`leadingSingletonDimensionsRemoved`, then a reshape back). The SliceUpdate itself is in place.
- MLX Swift has no public slice-update function, and `Cmlx` is not a product this package can import.

## What
1. Find a public path that writes a row with less overhead: an upstream mlx-swift API (`sliceUpdate`), or an index form that avoids `expandEllipsisOperations` and the reshape pair.
2. Keep RotatingKVCache/KVCacheSimple state, serialization, trim and restore exactly as now.

## Acceptance
- Bit-exact; greedy ids unchanged on oQ4e/oQ8e.
- Decode ms/token improves against a same-session baseline at 2k and 16k.
- Prompt-cache and ChatSession behaviour unchanged; all five bundles green. #flint-perf