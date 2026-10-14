---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'DeepSeek-V4 speed: radix-select top-k for the indexer, not argPartition (a full sort in MLX)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 3).

MLX `argPartition` is a full sort (`mlx/backend/metal/sort.cpp:342`: "We direct arg partition to sort for now"). The indexer uses it at `DeepSeekV4Indexer.swift:273` to pick 512 of up to 8192 chunks. llama.cpp uses a radix-select top-k kernel (`kernels/argsort.metal:250`). Also consider the MoE router (`DeepSeekV4MoE.swift:387`).

The selected set is the same except for ties. The index order changes, so sort the picks if a later step needs an order.

Estimate: prefill +1 to +3 % at 32k, decode -0.1 to -0.4 ms/token. Risk: low to medium.

## Acceptance Criteria

- [ ] A unit test shows that the selected set is the same as argPartition when there are no ties.
- [ ] Prefill tok/s and decode ms/token at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance