---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m633rrzz844q58w6bmajxn2b
  text: |-
    Research (implement): root cause found — out-of-vocabulary token ids, not a kernel.

    - Reproduced only with the input the ^kqvf7kp dump test used (recovered from its session log: TemporaryScheduleRunnerDumpTests): it fed `[3, 14, 15, 9, 2, 6, 5, 35, 7, 1, 4, 8, 12, 3, 9]` (the Qwen35 suite decode prompt, made for the 64-token Qwen35MTPFixture vocabulary) into `Qwen35CompiledVerifyTests.kernelModel()`, whose vocabulary is `GDNDecodeKernelFixture.vocabularySize` = 32. Token 35 is the 8th token: the embedding gather reads row 35 of a 32-row table, i.e. past the end of the buffer (MLX `take` has no bounds check on GPU). That is exactly "identical through 7 tokens, differs from the 8th step", and the batched 10-token prompt also holds 35.
    - Probe (temporary, 20 processes): `embedTokens([35])` gave 5 distinct hashes (14/2/1/1/1...); logits identical for steps 0-6 in all 20 runs, step 7 split 14/6. Discrete outcomes = whatever bytes sit past the embedding buffer in that process.
    - Ruled out with in-vocabulary seeded tokens (all identical): plain decode per token (useCompiledDecode false and true), bf16/fp16 batched 10-token prefill, 15 plain steps + compiled verifies 2...6 with rollbacks (hash of logits, hidden rows, every cache state, offsets): 60 sequential processes, 48 processes 6-way parallel under GPU contention, and a full MLXLMTests bundle run — one outcome each. Same at the card-time commit 08b56ad7 (temporarily checked out, then restored), so nothing landed since fixed it; the kernels (GDN decode/prefill, attention, router top-k) are deterministic.
    - Committed tests feed kernelModel only `seededTokens(..., vocabularySize: model.vocabularySize)`, so they were never affected; only the shared literal prompts (`Qwen35CompiledVerifyTests.prompt`, `Qwen35VerifyRewindTests.decodePrompt`, both `[3,14,15,9,2,6,5,35]`) do not fit the kernel model.
    - Plan: no library change (a bounds check on every forward would need a GPU sync; MTP speed untouched). Fix the fixture: the kernel model vocabulary takes `Qwen35MTPFixture.vocabularySize` (64) so every Qwen35 suite prompt fits every Qwen35 suite model, pinned by a new test that each shared decode prompt is inside the kernel model vocabulary.
  timestamp: 2026-10-29T05:57:47.135568+00:00
- actor: claude-code
  id: 01m6348807y5vsv9swd0t9d1qy
  text: |-
    ### implement — changed
    - Decision: deterministic. No kernel, reduction order or runtime route varies; the model is bitwise repeatable for in-vocabulary input. The only nondeterminism was the embedding gather of token 35 past the end of a 32-row table (the shared Qwen35 decode prompt fed to the kernel model). No library change (a per-forward bounds check would need a GPU sync), so MTP speed and caching are untouched and the Release guards are not needed.
    - Fix: `GDNDecodeKernelFixture.vocabularySize` is now `Qwen35MTPFixture.vocabularySize` (64), so every Qwen 3.5 suite prompt is a token of every Qwen 3.5 suite model; `Qwen35CompiledVerifyTests.prompt` is internal so the new suite reads it.
    - TDD: new `Qwen35KernelModelVocabularyTests` (each shared decode prompt — `Qwen35CompiledVerifyTests.prompt`, `Qwen35VerifyRewindTests.decodePrompt` — is in the kernel model vocabulary; the kernel model vocabulary is the MTP fixture vocabulary). RED: both failed (35 not in 0..<32; 32 != 64). GREEN after the fixture change.
    - Cross-process proof (temporary probe, deleted): the card input `[3,14,15,9,2,6,5,35,7,1]` one token at a time on kernelModel: before, step 7 split 14/6 over 20 processes and `embedTokens([35])` gave 5 distinct hashes; after, 20/20 sequential processes and 36/36 processes run 6-way parallel give one outcome for every probe (plain decode fp16/bf16, batched prefill, 15 steps + compiled verifies 2...6 with rollbacks, logits/hidden/cache states/offsets).
    - Affected suites (Qwen35CompiledVerifyTests, Qwen35VerifyRewindTests, Qwen35KernelModelVocabularyTests, Qwen35GDNDecodeKernelTests) pass in 5 separate processes: 20 tests each run.
    - `swift build --build-tests`: 0 compiler warnings (only the swiftbuild "missing creator" notice). `xcrun xctest`: MLXLMTests 920 XCTest + 1992 Swift Testing pass (1 pre-existing skip, benchmarkVerification, ^smn05mj), MLXGuidedGenerationTests 70, MLXFoundationModelsTests 404, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6 — 0 failures.
    - evidence: 3 files — Tests/MLXLMTests/GDNDecodeKernelFixture.swift, Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTests/Qwen35KernelModelVocabularyTests.swift (new)
    - next: /review
  timestamp: 2026-10-29T06:06:14.023050+00:00
- actor: claude-code
  id: 01m634aebfjx1kd7ze9bns9mqm
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (5fca3383) — counts findings 0, confirmed 0, refuted 0 (7 attempted, 0 failed); 3 files reviewed, 2 .kanban task files skipped (no validator matches)
    - next: none — task moved to done
  timestamp: 2026-10-29T06:07:26.063678+00:00
- actor: claude-code
  id: 01m634aqc7er1ph6t0xkq3tr7z
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 3 test files (fixture vocabulary 64, vocabulary suite)
    - test: green — 5 bundles 0 failures; card input 20/20 processes one outcome
    - commit: 5fca3383
    - review: clean — task moved to done
  timestamp: 2026-10-29T06:07:35.303036+00:00
position_column: done
position_ordinal: fffc80
title: LLM Qwen35 kernel fixture decode is not bitwise deterministic across processes
---
## What
Found while doing ^kqvf7kp. `Qwen35CompiledVerifyTests.kernelModel()` (Tests/MLXLMTests/Qwen35MTPTests.swift, GDNDecodeKernelFixture.smallHeadCase, 4 layers) gives different logits from one `xcrun xctest` process to the next with no code change:

- bf16, a batched 10-token prompt on the general path: the prefill logits differ in some runs (first differing byte inside the first `[1, 10, 32]` logits).
- fp16 or bf16, the prompt fed one token at a time with `useCompiledDecode: false` for every step: the logits are identical through 7 tokens and differ from the 8th decode step on, in about half the runs.
- 20 runs of the fp16 sequence (plain decode, then compiled verifies of 2...mostTokens) gave 3 distinct byte outcomes (12/7/1).

The outcomes are discrete (the same 2-3 byte patterns recur), which points at a kernel route or reduction order chosen at run time rather than a data race. Qwen3Next (fp16/bf16) and VLM Qwen35 (fp16 dense/MoE, bf16) fixtures of the same shape are byte-identical over 20 runs, so the suspects are the Qwen35 text-only pieces: `GatedDeltaDecodeKernel` / the GDN prefill kernel, `DecodeAttentionKernel` / `PrefillAttentionKernel` at cache length 8.

- [x] Find which op makes the run-to-run difference (dump per-layer outputs of step 8 across processes).
- [x] Decide: make it deterministic, or document why not and pin tests to tolerances.

## Why it matters
Bit-identity proofs for refactors of the Qwen35 text decode path (^kqvf7kp) can only compare outcome sets, not single dumps.

#mtp #test