---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m61mvv6qkv70nfnhpz0gqs4d
  text: |-
    Picked up on branch mtp (M5 Ultra, load avg ~13-21 from other projects' swift-test/swift-build plus mediaanalysisd; GPU otherwise idle).

    Research:
    - mlx-swift 0.32.3 (Package.resolved 19601207) already has `qmv_wide` (affine on gen>=15, fp modes everywhere): for 2 <= M < vector_limit, MLX routes to qmv_wide (tile of <=5 vectors, k_lanes 8 affine / 16 fp). Its per-row reduction order differs from `qmv_fast`, which is why 2-token verify is not bit-identical to 1-token decode today. M=1 stays on `qmv_fast` (N%8==0, K%512==0 for 4-bit).
    - Custom kernels compile with MathMode::Safe (default CompileOptions), MLX's own metallib with -fno-fast-math, so a kernel that replays qmv_fast's per-thread arithmetic per row can be bit-identical.

    Re-measured baseline here (new Release XCTest IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift, every matmul of one Qwen3.8-27B step with distinct random weights per layer, bf16 input, median of 7):
    4-bit affine g64: M=1 15.06 ms, M=2 19.72 (1.31x), M=3 21.46 (1.42x), M=4 30.49 (2.02x), M=5 35.68 (2.37x), M=6 47.39 (3.15x), M=7 61.18 (4.06x), M=8 63.82 (4.24x).
    mxfp4: M=1 14.03, M=2 16.87 (1.20x), M=3 23.06 (1.64x), M=4 29.48 (2.10x), M=6 50.36 (3.59x), M=8 63.75 (4.54x).
    So on the M5 Ultra even M=2 (today's MTP verify) costs 1.31x a plain step — worth targeting M=2 too.
  timestamp: 2026-10-28T16:18:04.631465+00:00
- actor: claude-code
  id: 01m61pa76j4a4qnj475waggkq6
  text: |-
    Kernel landed in Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift (public enum MultiRowQuantizedMatvec, quantizedMatmulRows(_:_:) / canRun(_:_:)); unit tests Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift (XCTest, run alone with `xcrun xctest -XCTest MLXLMTests.MultiRowQuantizedMatvecTests ...`) pass: every row bit-identical to quantizedMM of that row for affine 4/8-bit g32/64/128, mxfp4, bf16+f16, N=48/96/248320, M=1..8. Sanity: pointing the same test at MLX's own multi-row path (layer(x), qmv_wide) fails for every M>=2, so the test discriminates.

    What worked / did not (Release benchmark, M5 Ultra, ratios vs MLX M=1 in the same run; run-to-run noise about +-5-8%, load avg 8-25 from other projects):
    1. First cut (weights kept as packed words in a thread array, decoded per row via pointer casts as in qdot): bit-identical but slow - M=2 1.31x, M=4 2.67x, M=8 16x. The pointer casts of thread arrays push them to the stack.
    2. Decode each block's masked integers / fp4 values once to float, reuse for every row (exact, so still bit-identical): 4-bit M=2 1.16x, M=3 1.37x, M=4 1.80x, M=6 2.84x, M=8 5.62x (spills at ROWS 7-8).
    3. Decoded weights as half (all values exact in half): small gain.
    4. Probe: fma chain instead of qdot's mul/add tree (NOT bit-identical): only ~5-10% faster (M=4 1.62x vs 1.75x). Bit-identity is not the main cost; the per-row arithmetic qmv_fast fixes (x prep incl. bf16 partial sums, 16 mul+add per output row) makes the kernel ALU-bound beyond M=2. Reverted.
    5. Row groups of one output tile side by side on grid x (second group reads weights from cache) with at most 6 rows per threadgroup: M=7 2.68x, M=8 2.93x (was 3.7/5.1). Cap 4 was worse for M=5 (2.40x vs 1.88x) - kept 6.
    Current (cap 6): 4-bit g64 M=2 1.15x (MLX 1.18x), M=3 1.29x (1.61x), M=4 1.69x (2.02x), M=5 1.88x (2.28x), M=6 2.58x (3.43x), M=7 2.68x (3.73x), M=8 2.93x (4.17x); mxfp4 similar. Targets M=4<=1.3x / M=6<=1.6x are NOT reached; better than MLX at every M>=3 and bit-identical to plain decode. M=2 matmul time is about equal to MLX (within noise) but gains bit-identity with plain decode.
  timestamp: 2026-10-28T16:43:24.242996+00:00
- actor: claude-code
  id: 01m61qnqxj6yncm6k8dze5hbdw
  text: |-
    Wiring + end-to-end results (M5 Ultra, load avg 8-23 from other projects).

    Wired (via subagent, TDD: new tests went RED on bits first, then GREEN): Qwen35 GDN projectInputs (fused + unfused), new projectOutput (out_proj), attention projectPreRope (q/k/v) + mergeHeadsAndProject (o_proj), new Qwen35TextModel.projectVocabulary (lm_head; tied embedding unchanged), Qwen3NextMLP (gate/up/down). Also `Qwen3NextModelInner.layers` went fileprivate -> internal so the Qwen3Next fixture test can reach the MLPs. Tests: Tests/MLXLMTests/Qwen35MultiRowProjectionTests.swift, Tests/MLXLMTests/Qwen3NextMultiRowMLPTests.swift (plain decode logits bit-identical to an MLP whose projections keep MLX's matvec; verify rows bit-identical to single-row decode).

    M=2 through the kernel is a REGRESSION end to end: qwen38FourBitMTPGeneration same session, baseline (wiring stashed) MTP 90.3/81.0/78.9 tok/s (plain 58.7/59.1/59.2) -> with kernel at M>=2 80.5/72.5/70.4 (plain 58.5/58.9/59.2), output still equals plain. The microbenchmark parity at M=2 (independent matmuls overlap) does not hold in the dependent verify chain. So fewestRows = 3 (card: M=2 only if measured gain - it is not). With fewestRows=3: MTP 92.1/81.3/79.2 tok/s, plain 59.4/59.3/59.4, matches plain on all 3 -> within noise of baseline (MTP uses 2-row verify, which keeps MLX).

    Unit: swift build --build-tests clean; full MLXLMTests bundle via xcrun xctest: XCTest 910 tests 0 failures, Swift Testing 1913 tests in 168 suites passed.

    BLOCKER - qwen38FourBitMTPSystemPromptCache FAILS with the wiring (fewestRows=3): `GenerationBenchmarkTests.swift:263 Expectation failed: holds` - [thinking] turn 2 MTP diverged at token 0, gap 1.375. Baseline same session (wiring stashed) PASSES, but its thinking chat is on the tolerance edge too (turn 1 near-tie at token 52 gap 0.125, turn 2 near-tie at token 0 gap 0.375). With the wiring the PLAIN thinking turn 1 output also changes (decode 97 -> 125 tokens) and MTP turn 1 near-ties at token 0 (gap 0.25); the two histories then differ (prompts 2446 vs 2477) and turn 2 diverges. Edit chat still fine: edit turn recomputes 123 tokens (176.7 ms MTP, 264.5 ms plain), every follow-up cached. Cause (by inspection, not yet probed): some 3-8 row calls outside the verify take the kernel - most likely the short prompt tail ChatSession prefills after a checkpoint split, and the MTP drafter (Qwen35MTP uses Qwen35DecoderLayer + Qwen3NextMLP). Those rows now get decode-consistent bits instead of qmv_wide bits, which flips a near-tie.

    Decision needed: (a) restrict the kernel to the compiled speculative verify only (no effect on today's 2-token MTP; ready for deeper drafts), (b) keep it for all 3-8 row calls and accept/re-baseline the thinking-chat near-tie in the cache test, or (c) land only the kernel + tests without model wiring. Targets M=4<=1.3x / M=6<=1.6x are not met (kernel ALU-bound under qmv_fast's per-row arithmetic; see earlier comment).
  timestamp: 2026-10-28T17:07:10.386151+00:00
- actor: claude-code
  id: 01m61qnyebpykqka14kphnmwn3
  text: |-
    ### implement — stuck
    - evidence: 8 files — Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift (new), Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift (new), Tests/MLXLMTests/Qwen35MultiRowProjectionTests.swift (new), Tests/MLXLMTests/Qwen3NextMultiRowMLPTests.swift (new), IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift (new). MLXLMTests: XCTest 910/0 failures, Swift Testing 1913 passed. Release qwen38FourBitMTPGeneration 92.1/81.3/79.2 tok/s (baseline 90.3/81.0/78.9), matches plain. Release qwen38FourBitMTPSystemPromptCache FAILS (thinking turn 2 diverged, gap 1.375; baseline passes at the tolerance edge).
    - gaps: microbenchmark targets M=4<=1.3x (got 1.69x) and M=6<=1.6x (got 2.58x) not met; cache guard red.
    - next: human picks (a) kernel only in the compiled verify, (b) keep for all 3-8 row calls and re-baseline the cache test, or (c) land the kernel without model wiring.
  timestamp: 2026-10-28T17:07:17.067498+00:00
- actor: claude-code
  id: 01m61qq47nfx9sp45vkq95aqm4
  text: |-
    ### finish iteration 1 — stuck
    - implement: stuck — kernel bit-identical, M=4 1.69x / M=6 2.58x (targets 1.3x / 1.6x); M=2 via kernel slowed MTP, so M=2 stays on MLX; qwen38FourBitMTPGeneration pass 92.1/81.3/79.2; qwen38FourBitMTPSystemPromptCache FAIL (thinking chat turn 2 gap 1.375) because 3-8 row calls outside the verify (likely prefill tail or drafter) now take the kernel.
    - commit: f1a91bfa on branch `mtp-multirow-wip` (parked off `mtp` so mtp stays green)
    - review: not run
    - decision needed: (a) wire the kernel only into the compiled speculative verify (no effect until drafts deepen), (b) keep it for all 3-8 row calls and update the cache test, (c) land the kernel and tests without model wiring. Implementer recommends (a).
  timestamp: 2026-10-28T17:07:55.765492+00:00
- actor: claude-code
  id: 01m621fbw4vjza1serxfwmcm3r
  text: |-
    ### decision — user chose (a), 2026-10-28
    - Use the multi-row kernel only inside the speculative verify (3–8 rows); prefill, plain decode, the drafter and other 3–8-row calls stay on MLX. No effect at today's block size 2; ready for deeper drafts.
    - next: bring the kernel from branch mtp-multirow-wip (f1a91bfa) onto mtp with verify-only wiring; both Release guards must stay green (edit turn 123 tok, thinking chat passes).
  timestamp: 2026-10-28T19:58:27.204418+00:00
depends_on:
- 01M5SKBCEWG4D25SZZNGG7BVM7
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: multi-row quantized matvec that reads the weights one time for 2–8 verify rows'
---
## What
Measured on M5 Max (scratchpad rows.py, Qwen3.8-27B 4-bit g64 shapes, the matmuls of one step): M=1 25.0 ms, M=2 27.5 ms (1.10×), M=3 51.3 ms (2.05×), M=4 48.5 ms (1.94×). For M ≥ 3, MLX reads the weights again for each row, so a verify of 3–4 tokens costs about two plain steps. Deeper drafts are not useful until this is fixed. MLX issue 3661 reports the same problem (closed without a fix).

### References
- llama.cpp `kernel_mul_mv_ext_q4_f32_impl` (ggml/src/ggml-metal/kernels/mul_mv.metal ~603; dispatch in ggml-metal-ops.cpp ~2625). It is used for 2 ≤ M ≤ 8 for q4_0, q4_1, q8_0 and mxfp4. Layout: 2 SIMD groups for each threadgroup; `nxpsg` = 16 threads along a weight row when K % 256 == 0 and M < 3, else 8; each SIMD group does 32/nxpsg weight rows; each thread dequantizes 4 float4 chunks one time into registers and does a dot product with `r1ptg` activation rows (r1ptg = 2 for M=2, 3 for M=3 or 6, 4 for M=4, 7 or 8, 5 for M=5). The grid is (ceil(N / rows per TG), ceil(M / r1ptg)). With this kernel the cost for each row stays almost flat up to M = 4–8.
- osaurus vmlx-swift PR 613 `RowExactAffineQMV`, a copy of MLX `qmv_fast_impl` where all rows share one weight stream.

- [ ] Add Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift: an `MLXFast.metalKernel` for M ≤ 8 rows (r1ptg 2–5 as in llama.cpp), affine (bits 4 and 8, group 32/64/128, with biases) and mxfp4, bf16/f16 input. The output of each row must be bit-identical to `quantizedMM` of that row alone (same reduction order as `qmv_fast`). If bit-identical output costs too much speed, measure both and record the result.
- [ ] Add a dispatch helper `quantizedMatmulRows(_ x:, _ layer: QuantizedLinear)`. It uses the kernel when 3 ≤ M ≤ 8 (also M = 2 if the measurement shows a gain), and otherwise `QuantizedLinear.callAsFunction`. M = 1 always keeps the MLX kernel.
- [ ] Use the helper in the Qwen35 verify path: the fused GDN projection, out_proj, the MLP, the attention projections, and lm_head (Libraries/MLXLLM/Models/Qwen35.swift; Libraries/MLXLLM/Models/Qwen3Next.swift `Qwen3NextMLP`, which other models share).

## Acceptance Criteria
- [ ] A microbenchmark of all Qwen3.8 shapes: M=4 costs ≤ 1.3× M=1, and M=6 costs ≤ 1.6× M=1 (an XCTest in Release, or a Python harness that runs the same kernel source).
- [ ] Bit-identical row outputs against `quantizedMM` for every supported mode, bits, and group size.
- [ ] Plain decode (M=1) of Qwen3.8 and of a Qwen3Next model is unchanged: the same logits, and tok/s within noise.

## Tests
- [ ] Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift: bitwise equality for M = 1..8, all modes, odd output sizes (48, 96, 248320).
- [ ] Tests/MLXLMTests: a Qwen3Next fixture test where plain-decode logits are bit-identical before and after the `Qwen3NextMLP` change.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf