---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jr30ckwagy4qefq8f3v9f2
  text: |-
    Research (scratch harness $S/mrbench: one Qwen3.8-27B step of quantized projections, affine 4-bit g64 bf16, distinct random packed weights per layer, variants interleaved run by run, median of 9-15, bit-mismatch count vs MLX row by row; kernels read from $S/mrbench/kernels).

    Findings:
    - The current kernel already decodes each weight block once and reuses it for every row, so "share the load and unpack" was done. The cost of rows 3+ is not the decode (a near-free wrong decode saved only ~0.8 ms at 4 rows).
    - Dead ends (all bit-exact, none faster): decoded weights as float instead of half; vectorized bf16 x loads; RESULTS 1/2/8 per SIMD group (8 spills, 1-2 re-read x too often); splitting the rows over threadgroups (2 rows per TG: 4 rows 31.5 vs 23.5 ms, weights re-read) or over SIMD groups of one TG (26 ms).
    - What works: (1) stage each block of x (as float, with the lane sums of the affine bias term) in threadgroup memory once per threadgroup, double-buffered with one barrier per block, and widen the threadgroup to 8 SIMD groups (32 output rows) so 8 SIMD groups share one load+convert+sum of x; (2) decode nibbles to half with the 0x6400 magic (two nibbles per half2 op), exact; (3) prefetch the next block's packed weights and scales/biases into registers before the barrier. Without (3) the barrier costs ~3 ms (a racy no-barrier variant was 3 ms faster); with it the barrier is hidden.
    - Step matmul time (ms) lib -> new: rows 1 14.5 (MLX) / 14.3; rows 2 17.6 (qmv_wide, not exact) / 14.6 exact; rows 3 20.2 -> 16.3; rows 4 22.7 -> 18.9; rows 5 27.5 -> 23.5; rows 6 37.3 -> 29.8; rows 8 45.8 -> 36.2. 4 SIMD groups (tile 16) is much worse (4 rows 25.8), so the kernel takes 8 SIMD groups when the output width is a multiple of 32, else 4 or 2.
    - Rows 2 through the new kernel are faster than qmv_wide AND bit-exact. Out of scope here (fewestRows stays 3; switching it changes the MTP greedy ids of the L=2 verifies vs the golden) -> follow-up task.

    qmv_nax (mlx #4781 merged 2026-11-01 as 91473241, #4787 merged 2026-11-02 as ba8c005c): NOT in this repo's MLX. mlx-swift 0.33.0 (Package.resolved 7ede4fd6) pins mlx 36969c09 = v0.32.4 (2026-10-14). Built mlx main fc344c98 (0.32.5.dev) as a Python wheel in $S/venv_nax with a local env override to force the qmv_nax row floor down to 3 (scratch only), same step benchmark ($S/naxbench/naxbench.py, results $S/naxbench/naxbench.txt), M5 Ultra applegpu_g17d:
      rows: 0.32.4 (qmv_wide/qmm) | main default | main forced qmv_nax
      3: 21.2 | 21.7 | 15.3   4: 30.4 | 30.7 | 16.3   5: 34.7 | 34.0 | 17.4   6: 51.3 | 17.7 (qmv_nax) | -   8: 65.6 | 20.2 | -   12: 54.9 | 22.9   16: 53.2 | 25.2
      qmv_nax is NOT bit-exact: ~45-55% of output values differ from the single-row matvec (e.g. rows 4: 374656 mismatches, rows 6: 635848). qmv_wide at 2-5 rows: 114-313 mismatches. Faster than the new exact kernel (3: 15.3 vs 16.3, 4: 16.3 vs 18.9, 6: 17.7 vs 29.8) but not exact -> recorded, not switched. #4787 (qmv_mma) is off on Ultra and NAX GPUs by its own dispatch.

    Verify-forward microbench before (qwencli verify, 15 reps, mtp-verify-state ms) $S/verify_before_ets0cjp.txt: 2k L1-4 19.16/21.20/27.05/31.07; 32k 21.59/23.87/29.78/33.61; 64k 24.13/27.28/33.61/37.40.
  timestamp: 2026-11-16T17:57:12.211494+00:00
- actor: claude-code
  id: 01m7jrpjq1a80ne0fc2hfmje0h
  text: |-
    Implementation landed in Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift: inputs of each block staged once per threadgroup in threadgroup memory (float, plus the lane sums of the affine bias term), two stages and one barrier per block; threadgroup of 8/4/2 SIMD groups by output width (`simdGroups(outputs:)`, widest whose tile of 4 rows per SIMD group divides N); next block's packed words + scales/biases (and the staged rows' inputs) prefetched before the barrier; affine fields decoded two at a time via the half-1024 mantissa trick. Each lane still sums the same inputs in the same order as qmv_fast, so every row keeps its bits.

    Gotcha found while porting: moving the per-block compute loop into a helper that takes the weights/result arrays by reference cost ~0.3 ms at 4 rows (the Metal compiler kept them out of registers); always_inline did not help. The loop body stays in project_rows. A plain helper with only scalar/1-D args (decode, fetch_row, load_vector) is free.

    TDD: RED = three tile-width tests (`simdGroups(outputs:)` stub returned the old constant 2 -> 4096 and 48 failed), GREEN after. New exact parity test `testEveryRowIsBitIdenticalToItsOwnMatvecForEveryThreadgroupWidth` (every format, bf16+f16, N = 64/48/40 for 8/4/2 SIMD groups, K = 1536 = 3 blocks at 4-bit / 6 at 8-bit, rows 1 and 3..8) passes on old and new kernel (regression guard for the exactness contract). Existing MultiRowQuantizedMatvecTests (11) and ExactProduct (2) pass.

    Step matmul microbench final ($S/mrbench_final_ets0cjp.txt; MLX 1 row 14.48 ms): rows 3: MLX 23.24 / old kernel 19.46 / new 17.05; rows 4: 30.99 / 24.87 / 19.65; rows 6: 50.89 / 37.24 / 29.94. (Lib numbers include ~0.4 ms of canRun/reshape CPU work per step that the raw-kernel variants don't pay.)

    Follow-up filed: ^q1yp3b1 (2-row verifies through the kernel: 14.6 ms exact vs qmv_wide 17.6 ms not exact; changes MTP L=2 ids, so needs a golden decision).
  timestamp: 2026-11-16T18:07:53.569347+00:00
- actor: claude-code
  id: 01m7jt8dgt0h85crcxtrz5af8h
  text: |-
    Results.

    Verify-forward microbench (qwencli verify, 15 reps, mtp-verify-state ms; $S/verify_before_ets0cjp.txt -> $S/verify_after_ets0cjp.txt):
    - 2k: L3 27.05 -> 24.24 (-2.81), L4 31.07 -> 28.22 (-2.85); L1/L2 unchanged (19.16/21.20 -> 19.15/21.20)
    - 32k: L3 29.78 -> 27.15 (-2.63), L4 33.61 -> 31.24 (-2.37)
    - 64k: L3 33.61 -> 31.04 (-2.57), L4 37.40 -> 35.04 (-2.36)
    L4 verify is now below the plain 4-token forward at every context (2k 28.22 vs 28.93).

    qwencheck.sh ($S/qwencheck_ets0cjp.log): all 8 golden runs MATCH (short/code/32k/toolchat x plain/mtp, 257 ids each), SESSION PASS (plain and mtp, cached 139/203 tok), TOOLCHAT plain+mtp PASS, CHECK PASS.

    qwenbench.sh ($S/qwen_bench_ets0cjp.txt, medians of 3):
    vs stable (qwen_bench_base): plain 18.26/20.60/23.44 -> 17.04/19.77/22.67 ms (-6.7/-4.0/-3.3%); MTP 10.17/13.02/14.20 -> 9.34/12.48/13.53 ms (-8.2/-4.1/-4.7%); tokens/verify same 3.29/2.40/2.65; prefill 1334/1152/722 -> 1359/1160/725 tok/s.
    vs ^ftj9mae: plain 17.03/19.78/22.68 -> 17.04/19.77/22.67 (same, as expected: plain decode is 1 row); MTP 9.94/12.86/14.02 -> 9.34/12.48/13.53 ms (-6.0/-3.0/-3.5%); prefill +1.5/0.0/-0.1%.

    Tests (swift build --build-tests, xcrun xctest, $S/testlogs_ets0cjp): MLXLMTests XCTest 1117 (0 failures, 4 skipped) + Swift Testing 2369 passed with 3 known issues; MLXGuidedGenerationTests 78; MLXFoundationModelsTests 496; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. All exit 0. The 4 skips (LFM2/LFM2Moe tests gated on MLX_LFM2_* env vars), the skipped VerifyAttentionBlockPassProfileTests suite and the 3 known issues (MiniCPM5 CDATA tool call x2, AttentionKernelTolerance near-tie) are identical at the ^ftj9mae baseline (1113 XCTest, same skips and known issues); this change adds 4 tests. No build warnings beyond SwiftPM's "missing creator for mutated node" bundle notice. swift-format run on both changed files; lint clean.

    DFlash2 (dflash2-4bit, 4-5 row verifies) should gain too: step matmuls at 5 rows 27.5 -> 23.5 ms in the microbench. Not measured on that branch.
  timestamp: 2026-11-16T18:35:06.650718+00:00
- actor: claude-code
  id: 01m7jt8h0cdjkbftzdxj1aekad
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift, Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift; qwencheck CHECK PASS; 5 bundles exit 0; verify L3/L4 2k 27.05/31.07 -> 24.24/28.22 ms; MTP decode vs ^ftj9mae -6.0/-3.0/-3.5%
    - next: /review (not committed, per instructions); follow-up ^q1yp3b1 filed
  timestamp: 2026-11-16T18:35:10.220308+00:00
depends_on:
- 01M7J83VR4YW1XXC8EAFTJ9MAE
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 MTP verify: make rows 3+ of the multi-row quantized matvec cheap, bit-exact'
---
## Why
Measured verify forward cost at 2k: L=1/2/3/4 is 22.5/24.1/29.4/33.4 ms. The second row costs 1.6 ms on MLX's 2-row kernel. Rows 3 and 4 add 5.3 and 4.0 ms on our bit-exact multi-row kernel, whose cost grows linearly with each row. Verify is about 28 ms of a 32 ms MTP round at 32k.

## What
In `Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift`:
- Share the weight load and unpack across rows. Keep each row's accumulation order identical to the single-row kernel, so the output stays bit-exact.
- Try more rows per threadgroup and other tilings.
- Also benchmark the merged MLX qmv_nax kernels for 6–16 rows (ml-explore/mlx,, and when it lands). If they are faster but not bit-exact, record that and don't switch.

## Acceptance
- [x] Microbench: L=3 and L=4 verify forward in ms, before and after.
- [x] `qwencheck.sh`: MTP greedy ids identical to the golden file, and cache reuse passes. The parity unit test (exact) passes. All five bundles are green.
- [x] `qwenbench.sh` delta on this card. #qwen-perf