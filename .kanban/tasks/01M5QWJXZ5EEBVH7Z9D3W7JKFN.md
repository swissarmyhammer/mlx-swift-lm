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
- actor: claude-code
  id: 01m6248nwhv6mny9fh23mzddz3
  text: 'Picked up again on branch mtp for decision (a): bring kernel + tests from f1a91bfa, re-wire only on the speculative verify path (3–8 rows), prefill/plain decode/drafter stay on MLX.'
  timestamp: 2026-10-28T20:47:13.809767+00:00
- actor: claude-code
  id: 01m6252hrw0ckpg57jsxwbp7hc
  text: |-
    Verify-only wiring landed on mtp (from f1a91bfa: kernel file, its unit tests, Release benchmark; model wiring redone, not cherry-picked).
    - Gate: MultiRowQuantizedMatvec.run(asSpeculativeVerify:_:) sets a TaskLocal scope; MultiRowQuantizedMatvec.projection(_:by:) takes quantizedMatmulRows only inside it, else layer(x). Qwen35TextModel.forward (LLM) enters the scope only when emitDrafterState && mtpVerifyKeepsSnapshotsKey (the exact condition under which the backbone gets keepsSnapshots); VLM Qwen35Language.LanguageModel enters it when mtpVerifyKeepsSnapshotsKey is set. Compiled verify segments trace inside the scope (verifyStep is only reached from it); the decode traces never do.
    - Wired projections: Qwen35 (LLM+VLM) GDN in_proj fused+unfused, out_proj (new projectOutput), attention q/k/v/o, MLP (Qwen3NextMLP, VLM MLP), lm_head (new projectVocabulary). Qwen3Next has no verify path (gatedDeltaMixer preconditions !keepsSnapshots) so only its shared Qwen3NextMLP routes through the helper (no-op outside a verify). Qwen3NextModelInner.layers fileprivate -> internal for the fixture test.
    - Not covered by the gate: a verify of numDraft > maximumNativeTargetCacheRewind (5) takes the staged-round path with no keepsSnapshots flag, so a 7-8 token verify stays on MLX. Today block size is 2; up to 6 tokens are native.
    - Tests (RED first: verify-rows-vs-single-row logits failed 128..261 mismatches before wiring; new API failed to compile): Qwen35MultiRowProjectionTests (projection bits inside verify scope; prefill tail 3 rows plain + MTP-prefill bit-identical to an all-MLX-matvec reference model; verify S=3..6 compiled+general logits rows == single-row lm_head; compiled vs general within AttentionKernelTolerance.bfloat16), Qwen35VLMMultiRowProjectionTests (same for VLM), Qwen3NextMultiRowMLPTests, shared MultiRowProjectionFixture + MLXMatvecQuantizedLinear. Sanity: forcing projection() to always take the kernel makes both prefill-tail tests fail (164 / 171 mismatches) - they discriminate.
  timestamp: 2026-10-28T21:01:21.564972+00:00
- actor: claude-code
  id: 01m625km8gtea7kgpp5t41wah3
  text: |-
    ### implement — changed
    - evidence: 13 files — Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift (new, from f1a91bfa + verify scope/projection gate), Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift (new, from f1a91bfa), Tests/MLXLMTests/MultiRowProjectionFixture.swift (new), Tests/MLXLMTests/MLXMatvecQuantizedLinear.swift (new), Tests/MLXLMTests/Qwen35MultiRowProjectionTests.swift (new), Tests/MLXLMTests/Qwen35VLMMultiRowProjectionTests.swift (new), Tests/MLXLMTests/Qwen3NextMultiRowMLPTests.swift (new), IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift (new, from f1a91bfa), card .kanban files.
    - swift build --build-tests: 0 compiler warnings. xcrun xctest: MLXLMTests XCTest 920/0 failures + Swift Testing 1939 tests/172 suites passed; MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 376 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5/0. One pre-existing env-gated skip (MTPGreedyVerificationTests.benchmarkVerification, tracked by ^smn05mj).
    - Release qwen38FourBitMTPGeneration: PASS, plain 59.0/59.1/59.3, MTP 90.5/81.4/80.7 tok/s (within 3% of ~92/83/81), matches plain on all 3.
    - Release qwen38FourBitMTPSystemPromptCache: PASS, no known issues, thinking chat passes; edit turn recomputes 123 tokens with MTP (175.6 ms MTP vs 214.8 ms plain), follow-ups cached.
    - not met (recorded as measured): microbenchmark targets M=4 <= 1.3x (1.69x) and M=6 <= 1.6x (2.58x).
    - next: /review
  timestamp: 2026-10-28T21:10:41.168623+00:00
- actor: claude-code
  id: 01m62615yyjna07v4tzn3bymap
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (85e37abe), 14 findings (34 confirmed, 6 refuted) — IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:21,22,23,24,25,26,27,28,40,41; Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift:61,62,483; Tests/MLXLMTests/Qwen35VLMMultiRowProjectionTests.swift:40
    - next: implement the 14 open items in the 2026-10-28 16:11 Review Findings section, then re-review
  timestamp: 2026-10-28T21:18:05.278950+00:00
- actor: claude-code
  id: 01m6261mze5j3e2qa2yhdvmczd
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — verify-only wiring per user decision (a), 11 files
    - test: green (implementer run) — 5 bundles (MLXLMTests 920 + 1939); Release MTP 90.5/81.4/80.7 = plain; cache guard pass incl. thinking chat, edit turn 123 tok
    - commit: 85e37abe
    - review: findings — MultiRowQuantizedMatvecBenchmarkTests.swift:21-28,40,41; MultiRowQuantizedMatvec.swift:61,62,483; Qwen35VLMMultiRowProjectionTests.swift:40
  timestamp: 2026-10-28T21:18:20.654416+00:00
- actor: claude-code
  id: 01m626sq6nsjf0m0q15v4dj03f
  text: |-
    Review findings (2026-10-28 16:11) worked, all 14 flipped to [x].
    - magic numbers: MultiRowQuantizedMatvecBenchmarkTests now names the Qwen3.8-27B widths and layer counts in a private `Qwen38` enum (values checked against the cached mlx-community/Qwen3.8-27B-4bit config: hidden 5120, 48 GDN + 16 attention layers, 24x256 q heads with output gate, 4 kv heads, 48x128 values, vocab 248320) plus `checkpointBits`/`affineGroupSize`/`mxfp4GroupSize`. MultiRowQuantizedMatvec.swift: `narrowAffineBits`/`wideAffineBits`, `small/default/largeAffineGroupSize`, `weightAxes`. Metal header/source strings untouched (git diff hunks only in the Swift constants and fitsTiles), so the generated kernel is identical and the Release model guards were not rerun.
    - swiftlint 0.65 no_magic_numbers with only_rules + allowed_numbers [0, 1, -1, 100] (child config in scratchpad, no project .swiftlint.yml exists) over the 8 files of 85e37abe: 34 violations before (only the two flagged files), 0 after.
    - reuse: new `MultiRowProjectionFixture.seededQuantized(seed:matvecOfMLX:filter:prepare:_:)` (seed -> bf16 -> quantize -> prepare -> train(false) -> eval). Used by Qwen35MultiRowProjectionTests.makeModel, Qwen35VLMMultiRowProjectionTests.makeTarget and the per-module builds (VLM `quantized` helper deleted), and Qwen3NextMultiRowMLPTests.makeModel (third copy of the same setup; its private groupSize/bits and quantize closure removed, reference uses matvecOfMLX: true).
    - gotcha: the helper must be `throws`, not `rethrows` - a defaulted throwing closure parameter (`prepare`) makes every rethrows call need `try` anyway (compile error "call is to 'rethrows' function, but a defaulted argument function can throw").
    - gotcha: the files `edit file` op with replace_all replaced only 1 of 3 occurrences twice; edit each occurrence with unique context.
    - gotcha: Release build-for-testing of IntegrationTesting needs ENABLE_TESTABILITY=YES (GoldenReplayTests uses @testable imports); without it the build fails with "Unable to resolve Swift module dependency".
  timestamp: 2026-10-28T21:31:29.365920+00:00
- actor: claude-code
  id: 01m626sw894zvgp1saz5ygk7vh
  text: |-
    ### implement — changed
    - evidence: 6 files — IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift, Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift, Tests/MLXLMTests/MultiRowProjectionFixture.swift, Tests/MLXLMTests/Qwen35MultiRowProjectionTests.swift, Tests/MLXLMTests/Qwen35VLMMultiRowProjectionTests.swift, Tests/MLXLMTests/Qwen3NextMultiRowMLPTests.swift (+ card). swiftlint no_magic_numbers over the 8 files of 85e37abe: 34 -> 0. swift-format lint clean. swift build --build-tests: 0 compiler warnings (only the known build-system "missing creator ... mlx-swift_Cmlx.bundle"). xcrun xctest: MLXLMTests XCTest 920/0 failures + Swift Testing 1939 tests/172 suites passed (only skip: pre-existing env-gated benchmarkVerification, ^smn05mj); MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 376 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5/0. Totals: 3317 tests, 0 failures. Release xcodebuild build-for-testing IntegrationTesting (ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES): TEST BUILD SUCCEEDED, 0 warnings. Kernel source unchanged -> Release model guards not rerun.
    - findings: 14/14 checked.
    - next: /review
  timestamp: 2026-10-28T21:31:34.537782+00:00
depends_on:
- 01M5SKBCEWG4D25SZZNGG7BVM7
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: multi-row quantized matvec that reads the weights one time for 2–8 verify rows'
---
## What
Measured on M5 Max (scratchpad rows.py, Qwen3.8-27B 4-bit g64 shapes, the matmuls of one step): M=1 25.0 ms, M=2 27.5 ms (1.10×), M=3 51.3 ms (2.05×), M=4 48.5 ms (1.94×). For M ≥ 3, MLX reads the weights again for each row, so a verify of 3–4 tokens costs about two plain steps. Deeper drafts are not useful until this is fixed. MLX issue 3661 reports the same problem (closed without a fix).

### Decision (user, 2026-10-28, option a)
The kernel runs ONLY inside the speculative verify, for 3–8 rows. Prefill (the short prompt tail after a checkpoint among them), plain decode, the MTP drafter and every other 3–8-row call stay on MLX and keep their bits. At today's block size 2 (2-row verify, which stays on MLX `qmv_wide`) nothing changes; the kernel is ready for deeper drafts. The user's rule is that gains add up: a speedup only counts when multi-turn prompt caching and output-equal-plain still hold.

### References
- llama.cpp `kernel_mul_mv_ext_q4_f32_impl` (ggml/src/ggml-metal/kernels/mul_mv.metal ~603; dispatch in ggml-metal-ops.cpp ~2625). It is used for 2 ≤ M ≤ 8 for q4_0, q4_1, q8_0 and mxfp4. Layout: 2 SIMD groups for each threadgroup; `nxpsg` = 16 threads along a weight row when K % 256 == 0 and M < 3, else 8; each SIMD group does 32/nxpsg weight rows; each thread dequantizes 4 float4 chunks one time into registers and does a dot product with `r1ptg` activation rows (r1ptg = 2 for M=2, 3 for M=3 or 6, 4 for M=4, 7 or 8, 5 for M=5). The grid is (ceil(N / rows per TG), ceil(M / r1ptg)). With this kernel the cost for each row stays almost flat up to M = 4–8.
- osaurus vmlx-swift PR 613 `RowExactAffineQMV`, a copy of MLX `qmv_fast_impl` where all rows share one weight stream.

- [x] Add Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift: an `MLXFast.metalKernel` for M ≤ 8 rows, affine (bits 4 and 8, group 32/64/128, with biases) and mxfp4, bf16/f16 input. The output of each row is bit-identical to `quantizedMM` of that row alone (same reduction order as `qmv_fast`).
- [x] Dispatch helper `quantizedMatmulRows(_ x:, _ layer:)`: the kernel for 3 ≤ M ≤ 8, otherwise `QuantizedLinear.callAsFunction`. M = 1 keeps the MLX kernel; M = 2 keeps MLX (measured: M=2 through the kernel slowed end-to-end MTP 90.3/81.0/78.9 -> 80.5/72.5/70.4 tok/s).
- [x] Verify-only gate: `MultiRowQuantizedMatvec.run(asSpeculativeVerify:_:)` (TaskLocal scope) and `MultiRowQuantizedMatvec.projection(_:by:)`, which takes the kernel only inside the scope. Qwen35 text model enters the scope when the call emits drafter state with `mtpVerifyKeepsSnapshotsKey` (compiled and general verify); the Qwen3.5 VLM language model when `mtpVerifyKeepsSnapshotsKey` is set. A verify beyond the native rewind depth (> 6 tokens, staged-round path, no snapshot flag) stays on MLX.
- [x] Projections routed through the gate (Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, `Qwen3NextMLP` in Libraries/MLXLLM/Models/Qwen3Next.swift): fused/unfused GDN input projection, out_proj, attention q/k/v/o, MLP, lm_head. Qwen3Next itself has no verify path, so it never enters the scope.

## Acceptance Criteria
- [x] A microbenchmark of all Qwen3.8 shapes (IntegrationTesting MultiRowQuantizedMatvecBenchmarkTests, Release). Targets M=4 ≤ 1.3× M=1 and M=6 ≤ 1.6× M=1 are NOT met; recorded as measured (M5 Ultra, 4-bit g64, vs MLX M=1 in the same run): M=2 1.15× (MLX 1.18×), M=3 1.29× (1.61×), M=4 1.69× (2.02×), M=5 1.88× (2.28×), M=6 2.58× (3.43×), M=7 2.68× (3.73×), M=8 2.93× (4.17×). Better than MLX at every M ≥ 3; the kernel is ALU-bound under qmv_fast's per-row arithmetic (see comments).
- [x] Bit-identical row outputs against `quantizedMM` for every supported mode, bits, and group size.
- [x] Plain decode (M=1) of Qwen3.8 and of a Qwen3Next model is unchanged: the same logits, and tok/s within noise.
- [x] A 3–8-row call outside a verify (prompt tail, MTP prefill) keeps the matvec of MLX bit for bit; a verify of S=3..6 matches the general path within AttentionKernelTolerance.
- [x] Release guards: qwen38FourBitMTPGeneration (output equal plain, MTP ~92/83/81 within 3%) and qwen38FourBitMTPSystemPromptCache (passes with no known issues, thinking chat included; edit turn ~123 tokens with MTP on).

## Tests
- [x] Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift: bitwise equality for M = 1..8, all modes, odd output sizes (48, 96, 248320).
- [x] Tests/MLXLMTests/Qwen3NextMultiRowMLPTests.swift: plain-decode logits bit-identical to an all-MLX-matvec reference; MLP verify rows bit-identical to single-row decode inside the verify scope.
- [x] Tests/MLXLMTests/Qwen35MultiRowProjectionTests.swift and Qwen35VLMMultiRowProjectionTests.swift: projection rows inside the verify scope; prompt tail outside a verify bit-identical to an all-MLX-matvec reference; verify S=3..6 logits rows equal the single-row lm_head; compiled verify vs general path within tolerance.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf

## Review Findings (2026-10-28 16:11)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 11 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWJXZ5EEBVH7Z9D3W7JKFN.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWJXZ5EEBVH7Z9D3W7JKFN.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:21` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:22` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:23` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:24` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:25` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:26` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:27` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:28` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:40` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MultiRowQuantizedMatvecBenchmarkTests.swift:41` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift:61` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift:62` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift:483` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/Qwen35VLMMultiRowProjectionTests.swift:40` `reuse/reuse` — The `quantized` helper repeats the seeded-weights, bfloat16-cast, quantize, eval setup that `Qwen35MultiRowProjectionTests.makeModel` already performs (Qwen35MultiRowProjectionTests.swift:39-55). Two copies of the same setup will drift apart, for example when the quantization or seeding changes in one file only. Move the seeded, bfloat16, quantized, eval-ready build into `MultiRowProjectionFixture` as one generic helper that takes the `make` closure and the quantize filter, and call it from both the text and VLM test classes.
