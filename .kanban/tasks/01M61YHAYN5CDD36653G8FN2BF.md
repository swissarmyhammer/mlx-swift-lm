---
assignees:
- claude-code
position_column: todo
position_ordinal: '9980'
title: LLM Qwen35 kernel fixture decode is not bitwise deterministic across processes
---
## What
Found while doing ^kqvf7kp. `Qwen35CompiledVerifyTests.kernelModel()` (Tests/MLXLMTests/Qwen35MTPTests.swift, GDNDecodeKernelFixture.smallHeadCase, 4 layers) gives different logits from one `xcrun xctest` process to the next with no code change:

- bf16, a batched 10-token prompt on the general path: the prefill logits differ in some runs (first differing byte inside the first `[1, 10, 32]` logits).
- fp16 or bf16, the prompt fed one token at a time with `useCompiledDecode: false` for every step: the logits are identical through 7 tokens and differ from the 8th decode step on, in about half the runs.
- 20 runs of the fp16 sequence (plain decode, then compiled verifies of 2...mostTokens) gave 3 distinct byte outcomes (12/7/1).

The outcomes are discrete (the same 2-3 byte patterns recur), which points at a kernel route or reduction order chosen at run time rather than a data race. Qwen3Next (fp16/bf16) and VLM Qwen35 (fp16 dense/MoE, bf16) fixtures of the same shape are byte-identical over 20 runs, so the suspects are the Qwen35 text-only pieces: `GatedDeltaDecodeKernel` / the GDN prefill kernel, `DecodeAttentionKernel` / `PrefillAttentionKernel` at cache length 8.

- [ ] Find which op makes the run-to-run difference (dump per-layer outputs of step 8 across processes).
- [ ] Decide: make it deterministic, or document why not and pin tests to tolerances.

## Why it matters
Bit-identity proofs for refactors of the Qwen35 text decode path (^kqvf7kp) can only compare outcome sets, not single dumps.

#mtp #test