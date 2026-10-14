---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xqdpmf29h4ds6gstx30a1v
  text: |-
    Research (HEAD d42b780).
    - Decode graph (y7r/after.decode.dot) has 5 nodes for each of the 86 collapses ahead of the Sinkhorn `CustomKernel`: `CompiledBroadcastMultiplyBroadcastAddSigmoidBroadcastAdd` (pre), `CompiledBroadcastBroadcastMultiplyBroadcastAddSigmoidMultiply` (post), `CompiledBroadcastMultiplyBroadcastAdd` (comb logits), `Softmax` (comb rows) and `CompiledBroadcastAdd` (comb + eps). Thus 430 dispatches for each step can go into the kernel.
    - MLX `Softmax` (kernels/softmax.h, `softmax_single_row`) always uses `fast::exp`, also with `precise: true` (the flag only selects a float accumulator for half types). Thus a bit-exact kernel must use `fast::exp` for the softmax, not a precise exp. The card says "keep a precise exp"; that holds for the sigmoids only. MLX `Sigmoid` is `y = 1 / (1 + metal::exp(metal::abs(x))); (x < 0) ? y : 1 - y`.
    - The softmax kernel: max over the row; `exp(x - max)`; each thread adds 4 values (SOFTMAX_N_READS) in index order from 0, then `simd_sum` adds the partial sums; then `1 / sum` and `x * (1 / sum)`. For 8 or fewer copies, there are at most 2 nonzero partial sums, thus the `simd_sum` order cannot change the bits.
    - Plan: the kernel takes the float32 mixes, the scale vector and the bias, and gives `post`, `comb` and `collapsed`. Each thread computes the `pre` weights it needs; lane r of the first SIMD group computes row r of the softmax and `post[r]`. The op path off the GPU does not change.
  timestamp: 2026-10-14T17:30:07.375710+00:00
- actor: claude-code
  id: 01m4xrpmg4qvcgm965n45tyc91
  text: |-
    Implementation landed (not committed).
    - `DeepSeekV4SinkhornKernel.splitSinkhornCollapse(mixes:scale:base:stream:iters:eps:)` replaces `sinkhornCollapse(comb:pre:stream:iters:eps:)`. The kernel reads the float32 mixes, the scale vector and the bias, and gives `post`, `comb` and `collapsed`. Lane r of the first SIMD group takes the logits, the row softmax (+eps) and the Sinkhorn steps of row r, and `post[r]`. Every thread takes the `pre` sigmoids itself, thus no barrier.
    - `DeepSeekV4HyperConnection.collapse` on the GPU: float32 cast, `ManifoldStream.mixes`, then the kernel. Off the GPU it keeps the old path (compiled collapse fields / shapeless activations, `sinkhorn`, `weightedSum`).
    - `DeepSeekV4Math`: `splitFieldCount`, the three scale indices and `postGain` are now internal (the kernel reads them as template arguments). New `mixWidth(hcMult:)` is used by `hcSplitOperands`, the hyper-connection init and the kernel.
    - Bit-exactness discovery: the softmax must use `fast::exp` (MLX `softmax_single_row` uses `fast::exp` also with `precise: true`). The sigmoid must use `metal::exp`, the same function as MLX `Sigmoid`. With `metal::precise::exp` in the sigmoid, the parity test failed: `post` and `collapsed` moved by one unit in the last place in 4 of 5 cases. With `metal::exp` all cases give the same bits. Thus the card text "keep a precise exp" is not correct for these kernels; the same exp as the op path is what keeps the bits.
    - Tests: parity test now compares the kernel with `hcSplitSinkhorn` + `weightedSum` for post, comb and collapsed; cases (4,20,64,f32), (4,20,64,bf16), (3,1,6), (3,7,8), (8,3,4); constant-space case with 1 copy (mixes, bias, scale all in `constant`). New graph test `theCollapseTakesTheRowSoftmaxOfTheMixingMatrixInTheFusedKernel` (1 token and 4 tokens) counts `Softmax` nodes with a positive control (op path holds 1). New shared helper `DeepSeekV4TestArrays.nodeCount(_:primitive:)`; `gatherNodeCount` uses it.
    - RED: build failed (no `splitSinkhornCollapse`). Mutation RED for the graph test: with `canRun` false on the GPU, both cases failed. GREEN: suite 22 tests pass.
    - Full run: `swift build --build-tests` 0 compiler warnings (only the known `missing creator for mutated node` line); `xcrun xctest`: 716 XCTest 0 failures, 1426 Swift Testing tests in 112 suites passed.
  timestamp: 2026-10-14T17:52:28.676509+00:00
- actor: claude-code
  id: 01m4xsezh496hc51gjy3xtx0bh
  text: |-
    Measurements at 32k, GPU idle (0 %) before each run, bench `<snap> 32768 128 8192 1024` and `profile 32768`.
    - Before (HEAD d42b780): bench decode 33.0 / 32.9 ms/token, prefill 417.1 / 416.8 tok/s; profile window 34.6 / 35.4 ms/step, profile prefill 415.9 / 413.8 tok/s.
    - After: bench decode 31.7 / 31.7 ms/token, prefill 417.0 / 417.6 tok/s; profile window 33.2 / 33.3 ms/step, profile prefill 416.1 / 416.1 tok/s.
    - Gain: about 1.2-1.3 ms/token (bench) and 1.4-2.1 ms/step (profile window). The card expected 0.3-0.5 ms.
    - Decode graph (PROF4_DOT): Softmax 129 -> 43; `CompiledBroadcastMultiplyBroadcastAddSigmoidBroadcastAdd`, `CompiledBroadcastMultiplyBroadcastAdd`, `CompiledBroadcastBroadcastMultiplyBroadcastAddSigmoidMultiply` and `CompiledBroadcastAdd` 86 -> 0 each; thus 430 fewer kernel dispatches for each step. CustomKernel stays 86. Views: Slice 878 -> 104, Reshape 1197 -> 1025.
    - Generated text and every cache-check line of both bench runs are the same as before. Cache checks: HARD reuse PASS and tie-aware PASS for both thinking modes in each run.
    - Greedy parity `greedyFirstTokensMatchThePythonFixture()` through xcodebuild `-only-testing`: TEST SUCCEEDED.
    Files are in scratchpad `yyh/`.

    ### implement — changed
    - evidence: Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift, Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, Libraries/MLXLLM/Models/DeepSeekV4CompiledDecode.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift, Tests/MLXLMTests/DeepSeekV4TestArrays.swift. Decode graph Softmax 129 -> 43 and 344 compiled split kernels -> 0 (430 dispatches less per step). 32k decode 33.0/32.9 -> 31.7/31.7 ms/token; profile window 34.6/35.4 -> 33.2/33.3 ms/step; prefill unchanged. xctest 716 XCTest 0 failures, 1426 Swift Testing passed, 0 compiler warnings. Bit-exact float32 and bfloat16 parity tests pass. Greedy parity passed.
    - next: review
  timestamp: 2026-10-14T18:05:46.404825+00:00
- actor: claude-code
  id: 01m4xsmpn8m4npdpakqj5ds17p
  text: |-
    ### review — findings
    - evidence: `review sha HEAD~1..HEAD` (dfa3b28): 1 finding (7 attempted, 1 confirmed, 6 refuted). Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift:330 `code-hygiene/magic-numbers-swift`.
    - acceptance: all 3 items checked against the comments. (1) Parity test `theFusedSinkhornKernelAgreesWithTheOpPath` compares post, comb and collapsed with `==` on the floats, cases include (4,20,64,float32) and (4,20,64,bfloat16); suite passes. (2) 32k decode, GPU idle: 33.0/32.9 -> 31.7/31.7 ms/token. (3) `greedyFirstTokensMatchThePythonFixture()`: TEST SUCCEEDED.
    - note: the card text asks for a precise `exp`. The kernel uses `fast::exp` (softmax) and `metal::exp` (sigmoid), the same as the MLX ops. The implement step recorded that `metal::precise::exp` broke bit-exactness. Bit-exactness is the acceptance criterion, thus this change agrees with the criterion.
    - next: implement the finding (replace the output indices 0, 1, 2 with named constants, and remove the same cause from the whole file), then review again.
  timestamp: 2026-10-14T18:08:53.928643+00:00
- actor: claude-code
  id: 01m4xsn3ywmapdjzw2ky3x939t
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 6 files; 32k decode 33.0 -> 31.7 ms/token, prefill unchanged; same bits; greedy parity pass
    - test: green — 716 XCTest + 1426 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: dfa3b28
    - review: findings — DeepSeekV4SinkhornKernel.swift:330
  timestamp: 2026-10-14T18:09:07.548892+00:00
- actor: claude-code
  id: 01m4xsx2f69jc8dfh3ktbr08w2
  text: |-
    Finding `code-hygiene/magic-numbers-swift` (DeepSeekV4SinkhornKernel.swift:330, "Magic numbers should be replaced by named constants.") is done for the whole file.
    - RED: `swiftlint lint` with `only_rules: [no_magic_numbers]` and `allowed_numbers: [0, 1, -1, 100]` gave 1 violation at the `outputs[2]` line. GREEN: 0 violations.
    - Swift: new `private enum Output: Int, CaseIterable { post, comb, collapsed }`. The raw value is the place in the output list, the case name is the Metal output name, and `shape(leading:copies:width:)` gives the output shape. `outputNames`, `outputShapes`, `outputDTypes` and the return tuple all come from this enum. Thus the order is in one place.
    - Metal source: `constexpr int stepsPerPair = 2` replaces the two `2` literals of the Sinkhorn step loop. `constexpr uint vectorWidth = sizeof(float4) / sizeof(float)` replaces the two `4` literals of the weighted sum. The doc comment tells both names. The values are compile-time constants with the same value, thus the bits do not change.
    - Tests: GPU idle (0 %). `swift build --build-tests`: 0 compiler warnings (only the known `missing creator for mutated node` line). `xcrun xctest`: 716 XCTest 0 failures; 1426 Swift Testing tests in 112 suites passed. `theFusedSinkhornKernelAgreesWithTheOpPath` (5 cases, float32 and bfloat16) and `theCollapseTakesTheRowSoftmaxOfTheMixingMatrixInTheFusedKernel` pass.

    ### implement — changed
    - evidence: Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift; swiftlint no_magic_numbers 1 -> 0; xctest 716 XCTest 0 failures, 1426 Swift Testing passed, 0 compiler warnings.
    - next: review
  timestamp: 2026-10-14T18:13:28.166414+00:00
- actor: claude-code
  id: 01m4xsyz2rqvhf9ep295wcqar8
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (639b959): 0 findings, 0 confirmed, 0 refuted, 7 attempted, 0 failed. 1 file reviewed. The prior finding at Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift (code-hygiene/magic-numbers-swift) is checked.
    - next: none. The task moved to done.
  timestamp: 2026-10-14T18:14:30.232390+00:00
- actor: claude-code
  id: 01m4xszqzfkz3t5xs21a196bt5
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 file, 1 finding fixed (named kernel outputs and loop widths)
    - test: green — 716 XCTest + 1426 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 639b959
    - review: clean — 0 findings
  timestamp: 2026-10-14T18:14:55.727314+00:00
depends_on:
- 01M4XBWWKVGF366NQ6EX5NV4X8
position_column: done
position_ordinal: ffcb80
title: 'DeepSeek-V4 decode speed: put the comb softmax and the sigmoids into the Sinkhorn kernel'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

mlx-vlm does the comb softmax and the pre/post sigmoids inside its Sinkhorn kernel. Our decode graph has 86 `Softmax` and 172 `CompiledBroadcastAdd` nodes for these steps in each step. Move them into `DeepSeekV4SinkhornKernel.swift`. Keep a precise `exp` so that the output stays bit-exact.

Do this after the 4-lane Sinkhorn kernel task.

Expected gain: about 0.3-0.5 ms/token.

## Acceptance Criteria

- [x] The kernel output is bit-exact with the op path (float32 and bfloat16 tests).
- [x] Decode ms/token at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance

## Review Findings (2026-10-14 13:06)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 10 not reviewed.

> 10 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4XBXA418W76DBZV6YYHQKAK.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XBXA418W76DBZV6YYHQKAK.md` — no validator matches this file
> - `.kanban/tasks/01M4XRH91GZ301C38MNF4ZTXGJ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XRH91GZ301C38MNF4ZTXGJ.md` — no validator matches this file
> - `.kanban/tasks/01M4XRHDG8R5QF02SYQ5EFYZ6W.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XRHDG8R5QF02SYQ5EFYZ6W.md` — no validator matches this file
> - `.kanban/tasks/01M4XRHHA9FY9TSJHZWYCM4P15.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XRHHA9FY9TSJHZWYCM4P15.md` — no validator matches this file
> - `.kanban/tasks/01M4XRHN411A4RM83YWATJ552P.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XRHN411A4RM83YWATJ552P.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift:330` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
