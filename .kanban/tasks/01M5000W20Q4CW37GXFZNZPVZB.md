---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: Use the fused GDN decode kernel in Qwen3Next and in the MLXVLM Qwen 3.5 model
---
## What

^q0950zv added `GatedDeltaDecodeKernel` (Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift). It runs the body of one gated delta net layer on one decode token in one Metal kernel. Only `Qwen35GatedDeltaNet` in MLXLLM calls it. Two other GDN layers have the same op path and do not use it:

- `Qwen3NextGatedDeltaNet.forward` (Libraries/MLXLLM/Models/Qwen3Next.swift). Its conv input comes from a concatenation and its `z` is a strided view, thus the kernel copies `z` first.
- The GDN layer of Libraries/MLXVLM/Models/Qwen35.swift (no compiled decode, op path only).

## Approach

1. Call `GatedDeltaDecodeKernel.canRun` / `step` in the S = 1, no-mask, not-training branch of each layer, as `Qwen35GatedDeltaNet.decodeKernelStep` does.
2. Add a tolerance test against the op path of each layer, with a positive control.
3. Measure decode with a model of each kind; keep the change only if decode is faster.

## Acceptance Criteria

- [ ] Each layer uses the kernel on one decode token, with a tolerance test and a positive control.
- [ ] Decode is faster, measured, for each model.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #performance