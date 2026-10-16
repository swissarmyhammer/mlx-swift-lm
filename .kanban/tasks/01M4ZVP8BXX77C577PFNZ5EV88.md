---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5291nhvzc29yvsn43vraycf
  text: |-
    ### closed by the user
    - The user closed this task without implementation.
  timestamp: 2026-10-16T11:55:05.147348+00:00
position_column: done
position_ordinal: ffe080
title: Use DecodeAttentionKernel in the other GQA decode paths (Qwen3Next, generic attention route)
---
## What

^pqdshah added `DecodeAttentionKernel` (MLXLMCommon) and uses it only in the Qwen3.5/3.8 decode paths (`Qwen35Attention.attend`). At 32k it reads the KV cache at about 600 GB/s against about 465 GB/s for the MLX SDPA (Qwen3.8 shape). Other models with an even GQA group and a head width of a multiple of 32 can also gain at long context: Qwen3Next has its own `attentionCacheStep`, and most models call `attentionWithCacheUpdate`.

## Approach

1. Measure the kernel against the MLX SDPA for each target shape (GQA, head width) at 8k and 32k with the `qwen38bench attn` micro-benchmark. Keep only the shapes where it is faster.
2. Route those decode calls through `DecodeAttentionKernel.scaledDotProductAttention`.
3. Note: one kernel call costs about 13.5 us of CPU graph build against 0.4 us for the MLX SDPA. Check that decode stays GPU-bound.

## Acceptance Criteria

- [ ] Each routed shape is faster at 32k, measured.
- [ ] Tests show each routed decode path gives the kernel output.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #performance