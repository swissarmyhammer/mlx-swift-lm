---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xkq5xns53gnf63ggz23z4f
  text: |-
    Research (implement):
    - Baseline at HEAD 0c6b5c7, GPU idle, dsv4bench `32768 128 8192 1024`: prefill 401.9 tok/s (79.91 s), peak 147 GiB, decode 34.1 ms/token. Cache checks: HARD reuse PASS two times, tie-aware PASS two times.
    - The decode functions in DeepSeekV4CompiledDecode.swift compile with one trace for each input shape. MLX keeps each new shape as a new cache entry (compile.cpp `entries.push_back`), thus a prefill block of a new length adds one entry. Rule 3 of that file forbids that.
    - Shapeless compile is possible for the SwiGLU, the expand and the combine: they use only elementwise ops, ExpandDims, Transpose, Matmul, Reduce and Broadcast, and each of these has `output_shapes` in MLX.
    - Shapeless compile is NOT possible for the whole mHC split: `Slice` has no `output_shapes` in MLX (shapeless replay calls `output_shapes` on each node), and the split slices the mixes vector. The elementwise part of the split (sigmoid, softmax, scale and base) can be shapeless if the slices and the reshape stay outside the compiled function.
    - Estimate of GPU traffic saved for each 2048-token chunk: routed SwiGLU about 1.7 GB for each layer, expand about 0.6 GB for each call, combine about 0.45 GB for each layer. The split works on (T, 24) arrays and saves almost no GPU time.
  timestamp: 2026-10-14T16:25:23.637052+00:00
- actor: claude-code
  id: 01m4xmgrqpxa8f9hwhprxr0ej4
  text: |-
    Step 1 (compile only, same dtypes, same bits) landed in the working tree:
    - `DeepSeekV4CompiledDecode`: `clampedSwiGLU(limit:)`, `expand`, `combineWithSharedExpert` and `combineRoutedExperts` are now `compile(shapeless: true)`. New `splitActivations(hcEps:)`: three shapeless compiled field activations (pre, post, comb).
    - `DeepSeekV4Math`: the split is now `hcSplitOperands` (casts, slices, one reshape) plus `hcSplitActivations` (elementwise, plus the row softmax). `hcSplitFields` is the composition. The comb reshape now comes before the scale multiply; the bits are the same (elementwise).
    - MLP, SwitchGLU, MoE combine and the hyper-connection expand read the compiled functions for every block. The collapse of a prefill block reads the op path for the projection and the slices, and the compiled split activations. The decode collapse is unchanged (per-shape compiled `collapseFields`).
    - Unit tests: the compiled functions are called at block lengths 5, 3, 1 with one instance each, and give the bits of the op path. Full unit run: XCTest 716 tests, 0 failures; Swift Testing 1421 tests passed; build has 0 warnings.
    - Bench at 32k, GPU idle: prefill 401.9 -> 407.3 tok/s (79.91 -> 78.86 s), peak 147 -> 147 GiB, decode 34.1 -> 33.2 ms/token. Cache checks: HARD reuse PASS two times, tie-aware PASS two times.
    - Greedy parity `greedyFirstTokensMatchThePythonFixture()`: passed.
    - The gain is about 1 s, less than the 3.5 s estimate from the bandwidth count.
  timestamp: 2026-10-14T16:39:22.102414+00:00
- actor: claude-code
  id: 01m4xn1azkh4qg5kyfh2mhkpgt
  text: |-
    Step 2 (float32 -> bfloat16 for the SwiGLU and the combine) was measured and NOT kept:
    - Experiment: `clampedSwiGLU` without the float32 casts, and `combine` with the weighted sum and the shared add in the dtype of `routed`.
    - Bench at 32k, GPU idle: prefill 407.3 -> 412.7 tok/s (78.86 -> 77.82 s), peak 147 GiB, decode 33.2 ms/token. Cache checks PASS.
    - Greedy parity FAILED: the stream parts from the fixture at index 20 (Swift 2511, fixture 22896). The rule says keep it only when parity passes, thus the code is back to the step 1 state (checked with `cmp` against the step 1 diff).
    After step 2 was reverted: `DeepSeekV4MLP.swigluLimit` and `DeepSeekV4SwitchGLU.swigluLimit` had no reader any more (the op-path branches were their only readers), thus the two stored properties are deleted. The `swigluLimit:` init argument stays; it feeds the compiled activation.

    ### implement — changed
    - evidence: files Libraries/MLXLLM/Models/DeepSeekV4CompiledDecode.swift, DeepSeekV4MathHelpers.swift, DeepSeekV4HyperConnection.swift, DeepSeekV4MoE.swift, DeepSeekV4.swift (header note), Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift. 32k, GPU idle: prefill 401.9 -> 407.3 tok/s, peak 147 -> 147 GiB, decode 34.1 -> 33.2 ms/token; cache checks HARD PASS x2, tie-aware PASS x2. Unit: XCTest 716 tests 0 failures, Swift Testing 1421 tests passed, 0 build warnings. Greedy parity `greedyFirstTokensMatchThePythonFixture()` passed on the final code. Kept: step 1 (compile, same bits). Rejected: step 2 (bfloat16), parity failed.
    - next: /review
  timestamp: 2026-10-14T16:48:25.075843+00:00
- actor: claude-code
  id: 01m4xn5bx9fpdexf65tj44n1da
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (8dcabbc). 0 findings, 0 confirmed, 0 refuted, 7 attempted, 0 failed. 6 files reviewed. 2 kanban files not reviewed (no validator matches them).
    - next: The task is moved to done. All acceptance criteria are checked. The description has no earlier review findings sections.
  timestamp: 2026-10-14T16:50:37.097598+00:00
- actor: claude-code
  id: 01m4xn5ps8vhth18sd0w7ny285
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 6 files; 32k prefill 401.9 -> 407.3 tok/s, decode 34.1 -> 33.2 ms/token; bfloat16 step rejected (parity failed)
    - test: green — 716 XCTest + 1421 Swift Testing pass, 0 warnings, greedy parity pass (run by the implement step on the final tree)
    - commit: 8dcabbc
    - review: clean — 0 findings
  timestamp: 2026-10-14T16:50:48.232560+00:00
position_column: done
position_ordinal: ffc980
title: 'DeepSeek-V4 prefill speed: compile the mHC split and expand and the SwiGLU and combine steps at prefill'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

Chunk 0 of a prefill is 0.34 s slower than mlx-vlm (HISA off), about 5.4 s on a 32k prefill. Causes: the mHC split and expand are not compiled at prefill, and SwiGLU and combine run in float32. Places: `DeepSeekV4MathHelpers.swift` 124-161 and 391-400, `DeepSeekV4HyperConnection.swift` 221-235, `DeepSeekV4MoE.swift` 527-537.

First compile these steps and keep the dtypes (same numbers). Measure a change from float32 to bfloat16 as a separate step, because it changes the numbers.

Expected gain: up to about 5 s on a 32k prefill.

## Acceptance Criteria

- [x] The compiled steps give the same bits as the op path (unit test).
- [x] Prefill tok/s at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance