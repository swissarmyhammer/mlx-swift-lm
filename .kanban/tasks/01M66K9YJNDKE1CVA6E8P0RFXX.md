---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'MTP: route the router path''s (MLXVLM Qwen35) causal verify mask as .causal so verify reaches the decode/verify attention kernels'
---
Found by ^wcmxg07. On the router path, MLXVLM Qwen35 (loadModelContainer → VLM Qwen35, the path the SWE-bench ACP agent runs) builds the causal mask with `createAttentionMask(..., returnArray: true)`. So a multi-row verify passes an ARRAY mask and never reaches DecodeAttentionKernel or VerifyAttentionKernel. Both read only `.causal` or no mask.
- Microbench (16 layers, MTPVerifyForwardProfileTests): MLX SDPA with an array mask is 35–44 ms at 32k and 107–122 ms at 100k for 2–5 rows. The verify route with `.causal` is ~6.5 ms and ~18 ms.
- Round cost after ^wcmxg07: router 100k verify rows 1–5 are 27.67/43.20/56.41/67.16/79.49 ms (unchanged). Text 100k is 27.33/37.22/46.19/51.42/57.03.
- Sweep after: router 100k guard off 1.21x plain, text 1.43x. Router 100k guard on dropped to 0.96x plain, below the 0.97 floor (a known issue).

- [ ] Find why the VLM Qwen35 language model asks for an array mask (position ids / mrope? padding?). When the mask is a plain causal one, pass `.causal` (or `.none` for one row), as the text Qwen35 does.
- [ ] Unit test: the VLM attention layer gives the route output bit for bit with the causal mode, and the model-level verify matches plain decode within the bf16 near-tie rule.
- [ ] Re-run MTPRoundCostBreakdownTests (router) and the sweep. Router per-row cost should match text.

#mtp #perf