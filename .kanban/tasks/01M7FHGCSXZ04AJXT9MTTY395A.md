---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7fy96rhc8qspg85pb9wy4sf
  text: |-
    Research + same-session measurements (oQ4e, M5, medians of 3 process runs, prefill tok/s at 2k / 8k / 16k; scratchpad pfbench.sh; flintcli bench and pybench.py now take `@idsfile` and a chunk size).

    Input note: prompt_706.txt holds 19728 comma-separated real token ids. Both flintbench and pybench tokenized it as TEXT (digit text). Real ids (`@prompt_706.txt`) are the fair input.

    | run | 2k | 8k | 16k |
    |---|---|---|---|
    | Swift, text, default step 512 (= flintbench today) | 5456 | 6400 | 5930 |
    | Swift, ids, step 512 (default) | 4766 | 5957 | 5553 |
    | Swift, ids, step 1024 | 5852 | 7385 | 6862 |
    | Swift, ids, step 2048 | 6082 | 7827 | 7357 |
    | Swift, ids, step 4096 | 6113 | 7849 | 7381 |
    | Swift, ids, step 8192 | 6128 | 5909 | 6168 |
    | mlx-vlm, text, 2048 chunks | 6361 | 8014 | 7266 |
    | mlx-vlm, ids, 2048 chunks | 7270 | 7802 | 7150 |
    | mlx-vlm, ids, 512 chunks | 4279 | 5266 | 4862 |

    Conclusions:
    1. The gap is the chunk size. At equal chunks Swift is equal or faster than mlx-vlm at 8k/16k (7827 vs 7802, 7357 vs 7150) and also faster at 512 (5957 vs 5266). Flint uses the generic LLMModel.prepare default of 512; mlx-vlm/mlx-lm prefill in 2048.
    2. Second cost: when the prompt is not longer than the step size, the generic prepare hands the WHOLE prompt to the iterator, whose forward runs the float32 lm_head (150144 vocab) over every prompt row. That is why Swift ids/2048 at ctx 2048 is 6082 vs mlx-vlm 7270 (mlx-vlm prefills 2047 tokens with no logits, then 1), and why step 8192 collapses at 8k. Same cost hits every short prompt / ChatSession follow-up message. Gemma3Text already avoids it by prefilling through the inner model and always reserving one token.

    Ablations at step 2048 (temporary FLINT_ABL switch, reverted), baseline 6246 / 8021 / 7312:
    - no sinks: 5739 / 7982 / 7489 -> noise, sinks cost nothing measurable.
    - bf16 combine weights (no float32 routed sum): 7364 / 8318 / 7617 -> ~4% at 8k/16k; NOT taken: the reference and mlx-vlm do the f32 combine and float32 parity must hold.
    - full causal mask on sliding layers: 6609 / 8762 / 7911 -> ~8-9%; the sliding-window array mask costs. mlx-vlm pays the same (create_attention_mask -> RotatingKVCache.make_mask array). Going beyond mlx-vlm here needs a banded sliding prefill attention -> filed as a separate task.
    - no routed experts: 9492 / 13894 / 12414 -> routed experts are ~42% of prefill time; same gather_qmm path as mlx-vlm (sorted at >=16 routes; weightedExpertUnsort is top-8 bf16 only, Flint is top-6 f32 weights, so it does not apply).

    Plan: Flint gets its own prepare: default step 2048 (the mlx-vlm/mlx-lm chunk), prefill through the inner model (no lm_head), always reserve exactly one token for the iterator.
  timestamp: 2026-11-15T15:47:43.505070+00:00
- actor: claude-code
  id: 01m7fz86nde8vxra2d3eyfffx4
  text: |-
    ### implement — changed
    - change: `FlintModel.prepare` (Libraries/MLXLLM/Models/Flint.swift): default prefill step `defaultPrefillStepSize = 2048` (the mlx-vlm/mlx-lm chunk; a caller's stepSize still wins), chunks run through the inner text model (no lm_head), and exactly one token is always reserved for the iterator (Gemma3Text pattern), so a prompt <= one step no longer computes float32 logits over every prompt row. Doc of `PrefillParameters.stepSize` names the Flint default. Tests: FlintTests `testPrepareLeavesOnlyTheLastTokenForTheIterator` (RED 11 != 1), `testPrepareReadsOneMLXVLMChunkInOneForward` (RED [512,1024,1536,2048] != [2048]), `testPreparedLogitsMatchOneForward` (characterization); model built under a seeded random state.
    - prefill, real ids `@prompt_706.txt`, medians of 3, cold-process 2k/8k/16k: Swift before (step 512) 4766 / 5957 / 5553 -> after 6172 / 7929 / 7353; mlx-vlm 2048 chunks 7270 / 7802 / 7150. Warm 2k (second 2048 in-process): Swift before 6100 -> after 8100; mlx-vlm 8220. Cold 2k in the Swift process carries first-call warmup (min/max 6045-7259).
    - flintbench.sh (text input): prefill 6614 / 6368 / 5943 -> 8200 / 8020 / 7438; decode 6.37 / 6.72 / 7.07 ms/tok (baseline 6.37 / 6.72 / 7.09, no regression).
    - flintcheck: MATCH x4 + TOOL CALL get_weather(Paris). oQ8e loads; greedy ids MATCH golden8_fused.json x4. Forward numerics untouched, so float32 logit parity is unchanged (one-shot golden bit-identical).
    - iterator path (TokenIterator greedy, new mode `flintcli greedy`): prompts 200-500 tokens: old (one-shot through iterator) vs new (n-1 prefill + 1) identical 128 ids. parity_ids 666: 512 vs 2048 identical. long_ids 3600 and prompt_706 19728: both old 512 and new 2048 already diverge from the one-shot golden, always at near-ties (gaps 0.377, 0.111, 0.004, 0.008) — bf16 chunking noise that existed before.
    - ChatSession: follow-up "Now name two more." prefilled 6 tokens with 80 cached.
    - tests: MLXLMTests 2339 Swift Testing passed with 3 known issues + XCTest 1095 with 4 LFM2 env skips, 0 failures; MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 XCTest. swift-format run; only this task's files changed.
    - new tasks: ^v6cgcp6 (banded sliding-window prefill attention; the array mask costs ~8-9%), ^dnsqaqc (guided generation pins prefill step to 512 and overrides the model default, so Flint guided requests still prefill at 512).
    - next: /review
  timestamp: 2026-11-15T16:04:39.213026+00:00
- actor: claude-code
  id: 01m7fzqsd1pgvf8g371j0wkhfp
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean; xcrun xctest per bundle: MLXLMTests XCTest 1095 run, 4 skipped (LFM2 env skips, baseline), 0 failures; Swift Testing 2339 tests in 222 suites passed, 3 known issues (baseline); GDNDecodeKernelLayerTests passed. MLXGuidedGenerationTests 76 passed. MLXFoundationModelsTests 494 passed. CXGrammarTests 7 passed. MLXHuggingFaceMacrosTests 11 passed (XCTest). swift-format lint -s on Flint.swift, PrefillParameters.swift, FlintTests.swift clean. flintcheck.sh: MATCH x4 (256 ids identical each) and TOOL CALL get_weather Paris.
    - next: review/commit
  timestamp: 2026-11-15T16:13:09.921920+00:00
- actor: claude-code
  id: 01m7fzrjev61gzf52d2b6m1yht
  text: |-
    ### commit — changed
    - evidence: commit subject `perf(flint): prefill in 2048-token steps without the output projection (^tty395a)`; files Libraries/MLXLLM/Models/Flint.swift, Libraries/MLXLMCommon/PrefillParameters.swift, Tests/MLXLMTests/FlintTests.swift, and all .kanban/ changes (including ^v6cgcp6, ^dnsqaqc). Local commit on stable, not pushed. The sha is in the commit itself.
    - next: none
  timestamp: 2026-11-15T16:13:35.579950+00:00
- actor: claude-code
  id: 01m7fzvwg4a6ts1bp7nh77g08t
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (6adfaa83) — 1 finding (1 confirmed, 2 refuted): Tests/MLXLMTests/FlintTests.swift:242 code-hygiene/disallowed-constructs-swift no_unchecked_sendable
    - next: implement the open item in the 2026-11-15 10:13 Review Findings section, then re-review
  timestamp: 2026-11-15T16:15:24.164054+00:00
- actor: claude-code
  id: 01m7fzw7bhjxvd2h94xne3zefp
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — Flint.swift (FlintModel.prepare), PrefillParameters.swift (doc), FlintTests.swift (+3 tests)
    - test: green — MLXLMTests 1095 XCTest + 2339 ST, Guided 76, FM 494, CXGrammar 7, Macros 11; flintcheck MATCH x4 + TOOL CALL
    - commit: 6adfaa83
    - review: findings — Tests/MLXLMTests/FlintTests.swift:242 (no_unchecked_sendable)
  timestamp: 2026-11-15T16:15:35.281667+00:00
- actor: claude-code
  id: 01m7g0aff15v1a66rdynpgdabv
  text: |-
    ### implement — changed
    - finding: `Tests/MLXLMTests/FlintTests.swift:242` no_unchecked_sendable. `ProgressLog` was a `final class: @unchecked Sendable` with a mutable `var processed`. It is now a plain `Sendable` struct around `OSAllocatedUnfairLock<[Int]>` (the pattern of `ClearLog` in GenerationBufferPoolClearingTests and `CallCounter` in SampleTests): `record(_:)` appends under the lock, and `processed` reads under the lock. The test closure calls `log.record(processed)`. Added `import os`. The file has no other `@unchecked Sendable`. Finding flipped to [x].
    - evidence: swift build --build-tests clean. xcrun xctest: MLXLMTests XCTest 1095 run, 4 skipped (LFM2 baseline), 0 failures; Swift Testing 2339 tests in 222 suites passed, 3 known issues (baseline). MLXGuidedGenerationTests 76, MLXFoundationModelsTests 494, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11, all passed. swift-format run; only FlintTests.swift and this card changed. Tests-only change, so flintcheck was not needed.
    - next: /review (task left in doing, not committed)
  timestamp: 2026-11-15T16:23:22.337003+00:00
- actor: claude-code
  id: 01m7g0r07e4qgbjjfhv04v7tz8
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` ok (the only build output warning is the SwiftPM bundle message "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS", which is tooling output, not a compiler warning). `git diff --stat` shows Libraries/ unchanged (only Tests/MLXLMTests/FlintTests.swift plus the task's .kanban files), so the real-model guard did not run.
    - MLXLMTests: XCTest 1095 run, 0 failures, 4 skipped (the 4 LFM2 env skips); Swift Testing 2339 tests in 222 suites passed with 3 known issues (the baseline). GDNDecodeKernelLayerTests passed.
    - MLXGuidedGenerationTests: 76 tests in 17 suites passed.
    - MLXFoundationModelsTests: 494 tests in 66 suites passed.
    - CXGrammarTests: 7 tests in 5 suites passed.
    - MLXHuggingFaceMacrosTests: 11 XCTest tests, 0 failures.
    - swift-format lint -s --configuration .swift-format Tests/MLXLMTests/FlintTests.swift: clean.
    - next: review/commit
  timestamp: 2026-11-15T16:30:45.486692+00:00
- actor: claude-code
  id: 01m7g0rpmgyfdcd81gyg9p69g7
  text: |-
    ### commit — changed
    - evidence: commit `test(flint): guard the prefill progress log with a lock instead of @unchecked Sendable, found in review of ^tty395a` (sha recorded in the returned step block); files: Tests/MLXLMTests/FlintTests.swift, .kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.jsonl, .kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.md
    - next: none
  timestamp: 2026-11-15T16:31:08.432820+00:00
- actor: claude-code
  id: 01m7g0tcmdzm09nyk3f464xmn4
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (875f4de9) — 0 findings (0 confirmed, 1 refuted, 7 attempted, 0 failed); 1 file reviewed (Tests/MLXLMTests/FlintTests.swift), 2 .kanban files not matched by any validator; prior finding (2026-11-15 10:13) checked
    - next: none — task moved to done
  timestamp: 2026-11-15T16:32:03.725240+00:00
- actor: claude-code
  id: 01m7g0txt84c4g90a6v2qrayx0
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — FlintTests.swift ProgressLog lock (1/1 finding)
    - test: green — MLXLMTests 1095 XCTest + 2339 ST, Guided 76, FM 494, CXGrammar 7, Macros 11
    - commit: 875f4de9
    - review: clean — 0 findings; task done
  timestamp: 2026-11-15T16:32:21.320569+00:00
position_column: done
position_ordinal: ffffa880
title: 'Flint: profile prefill and close the gap to mlx-vlm (5.6–6k vs 7–7.7k tok/s)'
---
## Why
On oQ4e, Swift prefill runs at 5.6–6.0k tok/s from 2k to 16k context. mlx-vlm 0.7.12 gets 7.0–7.7k tok/s on the same machine in 2048-token chunks (scratchpad `pyscripts/pybench.py`). Decode is already faster than Python (7.7 vs 13.9 ms/token), so prefill is the gap.

## What
- **Compare like with like.** Check chunk size first: our `prefillStepSize` versus 2048.
- **Find the cost** with ablations or a Metal trace. Suspects:
  - the MoE sort/gather path for many tokens (`projectSwitchExperts`; does `callAndWeightedReduce` / `weightedExpertUnsort` apply? Flint calls `switchMLP(x, indices)` plus `weightedExpertSum` and never uses the direct-reduction path);
  - the float32 casts of the routed output;
  - sliding vs full attention masks at prefill;
  - the attention kernel with sinks.
- **Port what mlx-vlm does better.**

## Acceptance
- Prefill tok/s at 2k, 8k and 16k improves against a same-session baseline.
- Float32 logit parity is unchanged.
- Greedy ids are identical.
- All five bundles are green. #flint-perf

## Review Findings (2026-11-15 10:13)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 3 file(s) reviewed, 8 not reviewed.

> 8 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7FHG1KPDNCWYR3DCPG2HSB8.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FHG1KPDNCWYR3DCPG2HSB8.md` — no validator matches this file
> - `.kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FHGCSXZ04AJXT9MTTY395A.md` — no validator matches this file
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.md` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.md` — no validator matches this file

- [x] `Tests/MLXLMTests/FlintTests.swift:242` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.
