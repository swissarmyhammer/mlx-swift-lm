---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xyzbn5e08eye5s8p3s9p79
  text: 'Picked up. Baseline at HEAD 61a13b1, GPU idle, 32k (`<snap> 32768 128 8192 1024`, two runs each): prefill 419.8 / 419.6 tok/s, decode 30.3 / 30.3 ms/token, peak 147 GiB, profile window 32.1 / 32.3 ms/step (profile prefill 417.5 / 417.8 tok/s). Cache checks: PASS in both runs. Logs: scratchpad 5ef-before-*.log.'
  timestamp: 2026-10-14T19:42:05.989248+00:00
- actor: claude-code
  id: 01m4xze3ewdtg4h7kj9v5ee8h8
  text: |-
    Research and prototype (scratchpad dsv4bench `fusedindexer` mode, 1 x 64 heads x 2048 tokens x 128, f32):
    - N=8192 chunks: op path (tiles of 512 rows, matmul + compiled tail + mask) 31.9 ms; the matmul alone 19.1 ms; thus the per-head score write and read costs about 13 ms for each indexer layer and step.
    - First prototype (scalar tile loads, loops not unrolled): 211 ms. The fragment arrays were not in registers. With `#pragma clang loop unroll(full)` on every fragment loop and float4 tile loads: 17.0 ms (depth 16, 1 token for each threadgroup), 16.0 ms (depth 32, 4 tokens). About 17 TFLOPS, near the f32 peak of the M3 Ultra, thus more tuning gives little.
    - Layout: 4 SIMD groups (2 head halves x 2 chunk halves), threadgroup tile = all 64 heads x 64 chunks, simdgroup_float8x8 MMA with f32 inputs and f32 sums, K loop over the head width in steps of 8 (the order of the MLX steel GEMM). The epilogue stores the per-head sums to threadgroup memory and one thread for each chunk adds the heads in order 0..63: (relu(s) * scale) * w, as the op path.
    - Numbers: not bit-exact. Max abs diff 1.4e-6 at scores up to 4.6 (a few ulp). Changed top-512 selections: 0 of 2048 rows at N = 2048, 4096, 8192.
    - The attend stage, which calls the indexer, runs outside the compiled decode traces (`DeepSeekV4DecoderLayer.decodeStep`), thus the offset can be a fresh 1-value int32 input for each call. MLX gives inputs of fewer than 8 values in constant address space; the kernel indexes them directly.
    - The vector tile loads need `headDim % 4 == 0`. Another width, and a device that is not the GPU, keep the tiled op path.
    - Feeding the top-k kernel directly: the kernel output is (batch, tokens, chunks) f32, 64 MiB for a 2048-token step at 32k. A fused selection would save one write and one read of 64 MiB (about 0.1 ms for each layer and step). Not done; the gain is too small.
  timestamp: 2026-10-14T19:50:09.116351+00:00
- actor: claude-code
  id: 01m4y0xwk88nymqajmpr2khfwf
  text: |-
    Implementation landed (not committed).

    Files:
    - NEW Libraries/MLXLLM/Models/DeepSeekV4IndexerScoreKernel.swift: one MLXFast Metal kernel. Threadgroup = 4 SIMD groups (2 head halves x 2 chunk halves), tile = all heads of one token x 64 chunks, 4 tokens one after the other, head width loaded in steps of 32 as float4 into threadgroup memory, simdgroup_float8x8 MMA with f32 inputs and f32 sums. Epilogue: per-head sums to threadgroup memory, one thread per chunk adds the heads in order 0..H-1 as (max(s,0)*scale)*w, then the masked score (-1e30) for a chunk past its query. A chunk tile hidden from a token takes no products. Heads and head width pad with zeros, so any head count and any head width that is a multiple of 4 run. canRun(headDim:) = GPU and headDim % 4 == 0.
    - Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift: rankedChunks calls the kernel then the top-k kernel; the tiled op path (tiledRankedChunks) stays for a device that is not the GPU or an unaligned head width. ScoreOperands now hold keys (batch, chunks, headDim) and head weights (batch, tokens, heads); chunkScores turns them with views for the op path (same bits as before).
    - NEW Tests/MLXLMTests/DeepSeekV4IndexerScoreKernelTests.swift: tolerance vs op path (4 shapes: published prefill with partial tile and hidden chunks, synthetic 2x4 with 2 sequences, published decode, synthetic decode with constant-address head weights), exact masked score, changed top-512 selections = 0 (published shape, 64 x 2048), positive control for the counter, canRun.
    - Tests/MLXLMTests/DeepSeekV4IndexerTests.swift: the tiled-ranking test now uses head width 6 (op path) and checks canRun is false; new kernel-ranking reference test; new graph test: kernel ranking holds 2 Matmul nodes (the projections), op path 3 (positive control).

    Problems found on the way:
    - `fragment` is a Metal keyword (fragment shader); a constexpr with that name does not compile. Renamed to matrixSide.
    - MLX hides the Metal compile log behind a fatalError in tests; `withError {}` in the bench prints it.
    - The synthetic indexer's head weights hold 2 values, thus MLX gives them in constant address space and a `const device float*` variable does not compile. The kernel now indexes inputs from base offsets, and load_four has one overload per address space.

    Micro-benchmark (dsv4bench `fusedindexer`, copy of the production source, 1 x 64 x 2048 x 128 f32, prefill block whose last token sees every chunk): N=2048: op path 8.09 ms -> kernel 3.79 ms; N=4096: 15.98 -> 7.74 ms; N=8192: 31.97 -> 15.73 ms (MLX matmul alone 19.3 ms). Max abs diff 1.2e-6 to 1.4e-6; changed top-512 rows 0 of 2048 at each N. Decode (21 layers x 1 token x 8192 chunks in one eval): 0.793 ms both.

    32k end-to-end (GPU idle, two runs each, `<snap> 32768 128 8192 1024` and `<snap> profile 32768`):
    - prefill 419.8 / 419.6 -> 439.2 / 439.7 tok/s (+4.7 %)
    - decode 30.3 / 30.3 -> 30.3 / 30.4 ms/token (no change)
    - profile window 32.1 / 32.3 -> 31.6 / 31.8 ms/step
    - peak 147 -> 145 GiB
    - cache checks: HARD reuse PASS, warm == cold (text) YES, tie-aware PASS, both runs.
    - The 32k generated text now differs from the baseline after a few words (not bit-exact, as the card says); greedy parity is the gate.

    Tests: swift build --build-tests, 0 compiler warnings; xcrun xctest: 716 XCTest + 1444 Swift Testing tests in 114 suites, 0 failures. Greedy parity `greedyFirstTokensMatchThePythonFixture()`: passed (10.1 s).

    Top-k fusion: not done. The kernel output is 64 MiB for a 2048-token step at 32k; a fused selection saves about 0.1 ms per layer and step.
  timestamp: 2026-10-14T20:16:14.952329+00:00
- actor: claude-code
  id: 01m4y0y6may0z58nxxc134jtsh
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLLM/Models/DeepSeekV4IndexerScoreKernel.swift (new), Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Tests/MLXLMTests/DeepSeekV4IndexerScoreKernelTests.swift (new), Tests/MLXLMTests/DeepSeekV4IndexerTests.swift. Score step micro-benchmark 31.97 -> 15.73 ms at 8192 chunks (decode equal). 32k: prefill 419.7 -> 439.5 tok/s (+4.7 %), peak 147 -> 145 GiB, decode 30.3 -> 30.35 ms/token, profile 32.2 -> 31.7 ms/step. Changed top-512 selections: 0 (unit test, 64 rows; bench, 2048 rows at 2048/4096/8192 chunks). xctest 716 + 1444 tests, 0 failures, 0 compiler warnings. Greedy parity passed. Cache checks PASS.
    - next: /review (task stays in doing, not committed)
  timestamp: 2026-10-14T20:16:25.226724+00:00
- actor: claude-code
  id: 01m4y12zjwmxm14w1frcc8mbta
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (commit 90ade34). counts: findings 0, confirmed 0, refuted 0, attempted 7, failed 0. 4 source files reviewed. 2 .kanban files have no validator. No prior Review Findings sections.
    - acceptance check: (1) Tests/MLXLMTests/DeepSeekV4IndexerScoreKernelTests.swift has the tolerance test (relativeTolerance 1e-5, scoreCases), the exact masked-score test, the changed-selection test (expects 0) and its positive control (expects 1). (2) 32k before/after numbers are in the comments: prefill 419.8/419.6 -> 439.2/439.7 tok/s, peak 147 -> 145 GiB, decode 30.3/30.3 -> 30.3/30.4 ms/token. (3) greedyFirstTokensMatchThePythonFixture is in IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift; the comment records a pass (10.1 s).
    - next: none. Task moved to done.
  timestamp: 2026-10-14T20:19:01.852045+00:00
- actor: claude-code
  id: 01m4y13dkse1vy79pvze9b0epv
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 4 files; 32k prefill 419.7 -> 439.5 tok/s, peak 147 -> 145 GiB, decode unchanged; 0 changed selections; greedy parity pass
    - test: green — 716 XCTest + 1444 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 90ade34
    - review: clean — 0 findings
  timestamp: 2026-10-14T20:19:16.217072+00:00
position_column: done
position_ordinal: ffce80
title: 'DeepSeek-V4 speed: one fused Metal lightning-indexer kernel (no per-head score tensor)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 2).

Our indexer (`DeepSeekV4Indexer.swift`, about lines 339-372) writes a per-head float32 score tensor (4 GiB for each 2048-token step at 32k) and then reduces it, in a compiled function since ^er1vnkw. llama.cpp uses one kernel that calculates `sum_h w_h * relu(q_h . k)` plus the mask with simdgroup matmul, and never writes the per-head scores (PR 25893, `kernels/fa_aux.metal:338`).

Use float32 accumulation so that the selected chunks stay as near as possible to the current path. llama.cpp uses f16 Q and K; do not copy that.

Estimate: prefill +4 to +8 % at 32k, decode -0.2 to -0.5 ms/token. Not bit-exact. Risk: medium.

## Acceptance Criteria

- [x] A unit test compares the kernel scores with the op path (tolerance) and counts the changed selections. (`DeepSeekV4IndexerScoreKernelTests`: relative tolerance 1e-5 on 4 shapes, exact masked score, 0 changed top-512 selections of 64 rows, with a positive control for the counter.)
- [x] Prefill tok/s, peak memory and decode ms/token at 32k, before and after, GPU idle. (Prefill 419.8 / 419.6 -> 439.2 / 439.7 tok/s; peak 147 -> 145 GiB; decode 30.3 / 30.3 -> 30.3 / 30.4 ms/token; profile window 32.1 / 32.3 -> 31.6 / 31.8 ms/step.)
- [x] The greedy parity test on the real checkpoint passes. (`greedyFirstTokensMatchThePythonFixture` passed, 10.1 s.) #deepseek-v4 #performance