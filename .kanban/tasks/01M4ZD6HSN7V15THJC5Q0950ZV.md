---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'Qwen3.8 decode speed: fuse the small GDN kernels of each linear-attention layer'
---
## What

At 32k, one decode token spends about 4.1 ms in the Gated DeltaNet (GDN) work that is not a matmul, over 48 layers. The token runs 1430 kernels, 416 of them compiled elementwise kernels. The GPU is busy 100 % of the time, so each small kernel costs time.

## Approach

1. Make a list of the kernels of one GDN layer in decode (conv state update, conv, SiLU, q/k norm and scale, beta sigmoid, decay gate, `gated_delta_step`, gated RMSNorm, output gate).
2. Put as many as possible into one Metal kernel (or into `gated_delta_step`), so that a GDN layer has the fused input projection, one kernel, and the output projection. Target: 2 to 3 ms less for each token.
3. Keep the prefill path as it is, unless the same kernel is also faster there.

## Measure

scratchpad/qwen38bench at 32k: profile and step modes for decode (two runs each, idle GPU), the dispatch count (dot mode), and full mode for prefill, peak memory and the reuse check.

## Acceptance Criteria

- [ ] Decode at 32k is faster, measured.
- [ ] A tolerance test compares the fused kernel with the op path, with a positive control.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance