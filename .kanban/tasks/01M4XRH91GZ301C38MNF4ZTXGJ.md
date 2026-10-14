---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'DeepSeek-V4 speed: one fused Metal sparse attention kernel for head dim 512 (window + picked pool rows + sink, online softmax)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 1).

MLX has no fused SDPA for head dim 512 (only 64-256, `mlx/backend/metal/scaled_dot_product_attention.cpp:662-688`). Thus all our attention calls use matmul, softmax, matmul, with a gather copy and a float32 score tensor. llama.cpp uses one Metal kernel that reads the window rows and the picked pool rows by index, does online softmax with the sink, and uses K as V (PR 28098, PR 29377; `ggml-metal-ops.cpp:3171-3230`, `kernels/fa_vec_common.metal`). PR 28098 took prefill at 32k from 170 to 348 tok/s on M2 Ultra.

Our places: `DeepSeekV4Attention.swift` `attentionOverRankedChunks`, `attentionOverPickedChunks`, `attentionOverJoinedKeys`. A dense variant serves the window-only layers.

Estimate: prefill +5 to +12 % at 32k, decode -0.5 to -1.5 ms/token. Not bit-exact (sum order changes). Risk: medium to high.

## Acceptance Criteria

- [ ] A unit test compares the kernel with the op path (tolerance), with hidden picks and a query that sees no chunk.
- [ ] Prefill tok/s, peak memory and decode ms/token at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance