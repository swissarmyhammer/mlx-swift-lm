---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'Qwen3.8 decode speed: attention at 32k reads the KV cache at only 247 GB/s'
---
## What

At 32k, one decode token spends 4.34 ms in attention (16 full-attention layers, 24 query heads, 4 KV heads, head dim 256). It reads the 1.07 GB KV cache at only 247 GB/s. The quantized matmuls read at 662 GB/s. MLX uses the fused 2-pass decode SDPA with many blocks (about 512 on an Ultra at 32k), which writes about 0.2 GB of partial results for each token.

## Approach (measure each, keep only what is faster)

1. Sweep `MLX_SDPA_BLOCKS` (for example 32, 64, 128, 256) at 32k. If a value is faster, find how the library can select it for this shape.
2. If that is not enough: one custom Metal kernel for decode attention (as for DeepSeek-V4): each threadgroup reads each KV row one time for its 6 query heads (GQA 6), online softmax in float32, then one combine pass. Target about 2.7 ms per token.
3. Check the newer MLX decode SDPA (upstream 0e3ff364, osaurus study); note it, but do not update MLX in this task.

## Measure

scratchpad/qwen38bench at 32k: profile and step modes for decode ms/token (two runs each, idle GPU), plus full mode for prefill, peak memory and the reuse check.

## Acceptance Criteria

- [ ] Decode attention at 32k is faster, measured, or the card records why no change is faster.
- [ ] A tolerance test compares any new kernel with the MLX SDPA, with a positive control.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance