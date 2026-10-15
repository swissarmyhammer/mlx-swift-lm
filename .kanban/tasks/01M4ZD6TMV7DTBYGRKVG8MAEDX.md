---
assignees:
- claude-code
position_column: todo
position_ordinal: 8b80
title: 'Qwen3.8 speed: fuse the MLP gate and up projections, and the attention q/k/v projections'
---
## What

The MLP matmuls are the largest cost: 13.6 ms of a 30.5 ms decode token, and 2.96 s of a 5.84 s prefill block. The MLP `gate_proj` and `up_proj` read the same input in two quantized matmuls. The full-attention layers do q, k and v in separate matmuls. The GDN input projections are already fused.

## Approach

1. Join the gate and up weights (and q/k/v weights) into one quantized weight at load time (mxfp4, same group size), and split the output. Do the same for the scales.
2. Measure decode and prefill. Keep only what is faster.

## Measure

scratchpad/qwen38bench at 32k: profile and step modes for decode, full mode for prefill, peak memory and the reuse check (two runs each, idle GPU).

## Acceptance Criteria

- [ ] Decode or prefill at 32k is faster, measured, or the card records why not.
- [ ] A test shows the fused projection gives the same values as the separate projections.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance