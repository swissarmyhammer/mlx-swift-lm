---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4yajzkwz2fdrgn4ctmq5fkr
  text: |-
    Picked up. First step: measure the time of the mHC collapse and expand now (HEAD d3a3a4d).

    Micro bench (scratchpad dsv4bench `mhcops`, MICRO build with ENABLE_TESTABILITY, real config, random weights, GPU idle). Each run chains N halves (collapse, then expand) into one graph and evaluates it one time. Median of 9:
    - Decode, 1 token, 86 halves: op path 5.57 ms, compiled half 4.37 ms (0.051 ms per half). Collapse + 1 add: 4.55 ms. Floor of 1 add per half: 0.76 ms. Floor of 3 dependent dispatches per half (rmsNorm, rmsNorm, add): 2.55 ms.
    - Prefill, 2048 tokens, 8 halves: op path 2.64 ms per half, compiled 2.35 ms per half; collapse alone about 1.3 ms per half. At 32k that is 2.35 ms x 86 halves x 16 blocks = about 3.2 s of the 58.6 s prefill (about 5.5 %).

    Conclusion: the possible gain is more than the noise. Prefill: the fused path reads the bfloat16 stream about 2 times and writes the float32 copies 0 times; the op path writes and reads about 800 MB for each half. Decode: the compiled collapse plus expand is about 8 dispatches for each half; the fused design is 3. Thus I continue with the implementation.
  timestamp: 2026-10-14T23:05:03.356636+00:00
- actor: claude-code
  id: 01m4ydhedhv1bbnawkh1zfj88c
  text: |-
    Implementation landed (not committed). Design:
    - New `DeepSeekV4MixesKernel` (Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift): RMS reduction plus mixes projection in one dispatch. It reads the stream in its own dtype. Two layouts: decode step (1 token, 2 mix rows for each threadgroup, 512 threads) and prefill block (one token in each SIMD lane, 12 rows for each threadgroup, 256 threads), from 256 tokens up. A layout scan (scratchpad dsv4bench `mhcgen`, MICRO build) measured 0.022 ms for one token and 0.38 ms for 2048 tokens, against 0.036 ms and 1.13 ms for the op path.
    - `DeepSeekV4SinkhornKernel` now reads the stream and writes `collapsed` in the dtype of the stream (template `T`). Bits are the same as before (the parity test now compares `collapsed` with `opCollapsed.asType(dtype)`).
    - New `DeepSeekV4ExpandKernel` (Libraries/MLXLLM/Models/DeepSeekV4ExpandKernel.swift): the whole expand in one dispatch, float32 sums, vector loads of 4.
    - Collapse on the GPU = 2 dispatches, expand = 1 dispatch (a unit test counts the graph nodes). Off the GPU, and for widths that are not a multiple of 4, the op path stays.

    Things that did not work, or that a next agent must know:
    - A first prefill layout that gave 2 to 16 tokens to each threadgroup with strided loads was 0.55 to 1.3 ms for 2048 tokens. Contiguous spans for each SIMD group with one token in each lane gave 0.38 ms.
    - MLX gives an input of fewer than 8 values in the `constant` address space, thus each kernel refuses (canRun) a stream or block output of fewer than 8 values.
    - Metal: `min(uint, MIX_WIDTH - 1)` is ambiguous because a template argument is `int`; a `constexpr uint` fixes it.

    Measurements, 32k, M3 Ultra, GPU idle, DSV4_PREFILL_STEP default 2048 (logs in scratchpad dsv4bench, `*-b8v-before*.txt` / `*-b8v-after*.txt`):
    - full: before 558.4 / 559.7 tok/s prefill, 30.1 / 29.8 ms/tok decode, peak 143 / 143 GiB. After 588.5 / 589.8 tok/s (+5.3 %), 27.8 / 27.4 ms/tok, peak 143 / 143 GiB. Cache checks PASS in all runs.
    - profile: before 31.1 / 31.1 ms/step, after 28.9 / 28.5 ms/step.
    - step (B pipelined, 2 rounds each run): before 30.6, 30.9 / 30.4, 30.7 ms; after 28.0, 28.3 / 28.1, 28.1 ms. Step prefill 59.2 s before, 56.3 s after.
  timestamp: 2026-10-14T23:56:38.705037+00:00
- actor: claude-code
  id: 01m4ydwqtrsa6denxmv8st8t1q
  text: |-
    ### implement — changed
    - evidence: 7 files — Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift (new), Libraries/MLXLLM/Models/DeepSeekV4ExpandKernel.swift (new), Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift, Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Libraries/MLXLLM/Models/DeepSeekV4CompiledDecode.swift (doc only), Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift, Tests/MLXLMTests/DeepSeekV4TestArrays.swift. TDD: new tests watched red (stub kernels: canRun false, collapse 9 dispatches, collapsed float32), then green. `swift build --build-tests` 0 warnings; xctest all five bundles green (MLXLMTests 716 XCTest + 1458 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 367, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5), 0 failures; suite re-run after the last constant refactor 27/27. Greedy parity (xcodebuild -only-testing greedyFirstTokensMatchThePythonFixture) PASS on the real checkpoint. 32k: prefill 558.4/559.7 -> 588.5/589.8 tok/s, decode full 30.1/29.8 -> 27.8/27.4 ms/tok, profile 31.1/31.1 -> 28.9/28.5 ms/step, step B 30.4-30.9 -> 28.0-28.3 ms, peak 143 GiB both.
    - next: review. Not committed; task stays in doing.
  timestamp: 2026-10-15T00:02:48.792288+00:00
- actor: claude-code
  id: 01m4yefg9rh3ewyyj5wmcfj39r
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (f022390); 3 findings, 3 confirmed, 7 refuted. Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:93, Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:98, Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:238
    - next: Correct the 3 findings in the "Review Findings (2026-10-14 19:10)" section. The task stays in review.
  timestamp: 2026-10-15T00:13:03.672089+00:00
- actor: claude-code
  id: 01m4yefxfrdeerh3y6ndj2fqvs
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 7 files; 32k prefill 559 → 589 tok/s, decode 30.0 → 27.6 ms/token, peak 143 GiB no change
    - test: green — xcrun xctest MLXLMTests, 716 XCTest and 1458 Swift Testing passed, 0 failures, 0 warnings; greedy parity passed
    - commit: f022390
    - review: findings — Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:93, :98, :238
  timestamp: 2026-10-15T00:13:17.176039+00:00
- actor: claude-code
  id: 01m4yezreysjnatjreneqq99hb
  text: |-
    Review findings fixed (not committed). The cause was the same in all DeepSeek-V4 kernel files, thus I removed it from all of them.

    - New file Libraries/MLXLLM/Models/DeepSeekV4MetalKernel.swift, `enum DeepSeekV4MetalKernel`. It declares one time: `simdGroupWidth` (was in 5 kernel files), `vectorWidth` (Mixes, Expand, IndexerScore, and the Sinkhorn Metal `constexpr`), `smallestDeviceInput` with the fewer-than-8-values `constant` address-space rule (the rule was written again in Mixes, Expand, Sinkhorn, TopK, IndexerScore (2 times) and SparseAttention), `copyAxis`, `widthAxis`, `tokenAxisCount` (Sinkhorn, Expand, Mixes, IndexerScore, SparseAttention), `deviceIsGPU` (the `Device.defaultDevice().deviceType == .gpu` expression in all 6 kernels), `groupCount(covering:groupSize:)` (the ceiling division in Mixes, IndexerScore and SparseAttention, 6 places) and `tokenShape(of:)` (the leading-axes expression in Mixes, Sinkhorn and Expand).
    - Mixes: new private `features(of:)`; `canRun` and `mixes(of:projection:normEps:)` call it.
    - Each kernel doc now names ``DeepSeekV4MetalKernel/smallestDeviceInput`` and does not state the MLX rule again. The local limits that the old `simdGroupWidth` docs held (Sinkhorn threadsPerToken, TopK threadsPerRow, SparseAttention key tile) moved to the doc of the local constant.
    - Sinkhorn Metal: the weighted sum read `constexpr uint vectorWidth = sizeof(float4) / sizeof(float)` and `float4`. It now reads the new `VECTOR_WIDTH` template argument (the shared value 4) and `typedef vec<float, VECTOR_WIDTH> floatVector`. `vec<float, 4>` is `float4`, thus the numbers do not change; the bit-exact Sinkhorn tests and the greedy parity test pass. This is the one change to a Metal source text; it was necessary to remove the second declaration of the vector width. No other Metal text changed.
    - Not changed: TopK `rowAxis = -1` (another role: the row of the scores) and the `ScalarSlot` enums (per-kernel data).
    - New unit test Tests/MLXLMTests/DeepSeekV4MetalKernelTests.swift for `groupCount` and `tokenShape`. RED: the build failed with "cannot find 'DeepSeekV4MetalKernel' in scope". GREEN after the new file.
  timestamp: 2026-10-15T00:21:56.318536+00:00
- actor: claude-code
  id: 01m4yezw9ys00ybq6wnsd6c1na
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLLM/Models/DeepSeekV4MetalKernel.swift (new), DeepSeekV4MixesKernel.swift, DeepSeekV4ExpandKernel.swift, DeepSeekV4SinkhornKernel.swift, DeepSeekV4TopKKernel.swift, DeepSeekV4IndexerScoreKernel.swift, DeepSeekV4SparseAttentionKernel.swift, Tests/MLXLMTests/DeepSeekV4MetalKernelTests.swift (new). swift build --build-tests: 0 errors, 0 compiler warnings (only the known "missing creator for mutated node" note). xcrun xctest MLXLMTests: 716 XCTest, 0 failures; 1460 Swift Testing tests in 116 suites passed. xcodebuild -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/greedyFirstTokensMatchThePythonFixture(): passed, TEST SUCCEEDED. swift format lint: clean on the changed files. 3 of 3 findings checked.
    - next: /review
  timestamp: 2026-10-15T00:22:00.254113+00:00
- actor: claude-code
  id: 01m4yf3mban8rqrrsscs8bxzpy
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (commit 62540d7). 0 findings, 0 confirmed, 1 refuted, 7 attempted, 0 failed. All 3 findings in "Review Findings (2026-10-14 19:10)" are checked.
    - next: None. The task moved to done.
  timestamp: 2026-10-15T00:24:03.178754+00:00
- actor: claude-code
  id: 01m4yf408s6w8hvrxpz3egqzn7
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 8 files; shared DeepSeekV4MetalKernel constants and helpers
    - test: green — xcrun xctest MLXLMTests, 716 XCTest and 1460 Swift Testing passed, 0 failures, 0 warnings; greedy parity passed
    - commit: 62540d7
    - review: clean — 0 findings; 3/3 prior findings checked
  timestamp: 2026-10-15T00:24:15.385883+00:00
position_column: done
position_ordinal: ffd180
title: 'DeepSeek-V4 speed: mHC collapse in 2 dispatches and expand in 1 (fused norm, projection and expand kernels)'
---
## What

Found on 2026-10-14 in a study of osaurus (scratchpad `osaurus/REPORT.md`, idea 3).

Our collapse is a float32 cast, `rmsNorm`, a matmul, the Sinkhorn kernel and a cast (`DeepSeekV4HyperConnection.swift` about 399-419). Our expand is a batched 4x4 matmul (`DeepSeekV4CompiledDecode.swift` about 99-103). Osaurus does the collapse in 2 dispatches and the expand in 1 for GLM-5.3 (vmlx-swift commits 01abc39f, f8130127; `MHCFusedDecode.swift:28-154`, `MHCFusedPrefill.swift:33-187`). Much of their decode gain came from replacing a 1-lane Sinkhorn, which ^x5nv4x8 already did.

Estimate at 32k: decode -0.5 to -1.0 ms/token, prefill +1 to +3 %. The fused projection changes the sum order, thus it is not bit-exact. Risk: medium. Do this after ^yyhqkak.

## Acceptance Criteria

- [x] A unit test compares the fused kernels with the op path (tolerance), float32 and bfloat16.
- [x] Decode ms/token and prefill tok/s at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance

## Review Findings (2026-10-14 19:10)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 7 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4XSZJZDDR3KGCA30B8V2PQ4.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XSZJZDDR3KGCA30B8V2PQ4.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:93` `reuse/reuse` — The private constant vectorWidth = 4 is declared again. It is the same value and the same meaning as the constant in DeepSeekV4ExpandKernel, so the vector-load width is defined twice. Declare the vector-load width once in a shared place and read it in both kernels. A change to the load width then needs one edit.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:98` `reuse/reuse` — The private constant smallestDeviceInput = 8 is declared again, with the same comment, in both kernel files. The two copies can drift apart. Declare the minimum device-input size once in the shared place from the finding on the gate, and read it from both kernels.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4MixesKernel.swift:238` `reuse/reuse` — The flattened feature count is computed again. The same suffix-product expression already appears in canRun in the same file, so the rule for the feature count exists twice. Add one private helper, for example features(of:), that returns the flattened copy-and-width count. canRun and mixes(of:projection:normEps:) then call it.