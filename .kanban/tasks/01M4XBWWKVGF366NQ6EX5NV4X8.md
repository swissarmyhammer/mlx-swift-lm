---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xd2gs0b7dhqpf65wx7y1np
  text: 'Research done. The kernel is DeepSeekV4SinkhornKernel.swift; lane 0 does all 2*ITERS-1 normalizations. PAR_SRC in scratchpad scripts/prof4_sinkhorn_bench.py keeps row r of the matrix in the registers of lane r of simdgroup 0 and reads columns through simd_shuffle. This needs copies <= 32 (one simdgroup) and a threadgroup of 32 or more threads. canRun does not check the copy count now, thus a guard is necessary. The parity test is theFusedSinkhornKernelAgreesWithTheOpPath (cases (4,20,64), (3,1,6), (3,7,8)). Bench: scratchpad dsv4bench full mode `<snap> 32768 128 8192 1024`; the bench builds against the working tree of this repo.'
  timestamp: 2026-10-14T14:29:15.168591+00:00
- actor: claude-code
  id: 01m4xe9hn7evzhqw2hs225qabt
  text: |-
    Implementation landed (not committed).

    Kernel: lane r of the first SIMD group keeps row r of the mixing matrix in registers; a column step reads the column through simd_shuffle and adds rows 0..HC-1 in index order; divisions stay divisions (PAR_SRC). New template argument SIMD_WIDTH (Swift constant simdGroupWidth = 32). canRun now also needs copies <= 32, and sinkhornCollapse has a precondition for it.

    Discoveries:
    1. A `const device float*` pointer into comb_in does not compile when the input has fewer than 8 values: MLX metal_kernel gives such an input in the `constant` address space (mlx/backend/common/metal_kernel.cpp, max_constant_array_size = 8). One token with 2 copies hits this (DeepSeekV4CacheTests crashed the suite). The kernel now indexes comb_in directly. New test theFusedSinkhornKernelTakesAMixingMatrixInTheConstantAddressSpace failed (Fatal error, metal library build) before the fix and passes after.
    2. A parity case with 16 or 32 copies fails for the weighted sum too, which this change does not touch: the MLX op-path sum does not add in index order at those counts. Thus the parity cases stop at 8 copies; (8, 3, 4) was added.

    Tests: new theFusedSinkhornKernelDoesNotTakeMoreCopiesThanOneSimdGroup (RED before canRun change, GREEN after). Full unit run: 716 XCTest + 1411 Swift Testing tests, 0 failures. Greedy parity greedyFirstTokensMatchThePythonFixture: TEST SUCCEEDED (26.7 s).

    32k measurements, GPU idle, same method before and after (scratchpad dsv4bench, full mode `<snap> 32768 128 8192 1024` and `<snap> profile 32768`):
    - full mode generate: 37.2 -> 33.9 ms/token; prefill 326.2 -> 326.9 tok/s.
    - profile: 39.1 -> 35.4 ms/step; prefill 314.3 -> 323.9 tok/s.
    - Same first 160 characters of output text; cache checks PASS before and after.
    Logs: scratchpad dsv4bench/full-before-x5n.txt, full-after-x5n.txt, profile-before-x5n.txt, profile-after-x5n.txt.

    Acceptance criteria: all three are met (the description update failed to show output, see the checkbox state).
  timestamp: 2026-10-14T14:50:34.023642+00:00
- actor: claude-code
  id: 01m4xe9nsvm6pr2h448v3tj16f
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift. 32k decode 37.2 -> 33.9 ms/token (full mode), 39.1 -> 35.4 ms/step (profile); prefill 326.2 -> 326.9 tok/s. Unit: 716 XCTest + 1411 Swift Testing, 0 failures. Greedy parity PASS.
    - next: /review
  timestamp: 2026-10-14T14:50:38.267043+00:00
- actor: claude-code
  id: 01m4xeg520hcrn21fr397h0ffn
  text: |-
    ### test - green
    - evidence: GPU idle check (Device Utilization 0%). swift build --build-tests: build complete. xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest: XCTest 716 tests, 0 failures; Swift Testing 1411 tests in 111 suites passed. 0 skipped. The 4-lane Sinkhorn kernel tests pass.
    - warnings: touched the two changed files and built again. MLXLLM and MLXLMTests recompiled with zero compiler warnings. One build-system line remains: "missing creator for mutated node ... mlx-swift_Cmlx.bundle". It is not a compiler warning and it is not from this change.
    - next: review
  timestamp: 2026-10-14T14:54:10.496153+00:00
- actor: claude-code
  id: 01m4xejnp5475tttxpzs4exymn
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (commit 2266398). 0 findings, 0 confirmed, 0 refuted. 7 validator runs attempted, 0 failed. 2 source files reviewed. 12 .kanban task files not reviewed (no validator matches them).
    - next: none. The task moved to done.
  timestamp: 2026-10-14T14:55:33.061682+00:00
- actor: claude-code
  id: 01m4xejzn2z80761cc8ef0kqhw
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 2 files; 32k decode 37.2 -> 33.9 ms/token, prefill 326.2 -> 326.9 tok/s; greedy parity pass
    - test: green — 716 XCTest + 1411 Swift Testing pass, 0 warnings
    - commit: 2266398
    - review: clean — 0 findings
  timestamp: 2026-10-14T14:55:43.266746+00:00
position_column: done
position_ordinal: ffc680
title: 'DeepSeek-V4 decode speed: Sinkhorn kernel with 4 lanes in registers'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`, `prof4/sinkhorn_bench.txt`).

The fused collapse kernel in `DeepSeekV4SinkhornKernel.swift` takes about 57 us and runs 86 times in each decode step. Lane 0 does all 39 normalizations alone. A 4-lane variant with the same order of additions and the same divisions takes 13.9 us, with bit-identical outputs at 1 and 2048 tokens. The tested source is `PAR_SRC` in scratchpad `scripts/prof4_sinkhorn_bench.py`.

Expected gain: 3.4-4.0 ms/token at all contexts (decode at 32k is 39.6 ms/token; mlx-vlm 33.3).

## Acceptance Criteria

- [x] The kernel uses 4 lanes. The existing bit-exact tests of the kernel pass (float32 and bfloat16).
- [x] Decode ms/token at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance