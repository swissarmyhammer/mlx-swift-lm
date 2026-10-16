---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m529bkbp4q1mb80q1ewkwq2m
  text: |-
    Research (implement step).
    - MLXLLM `Qwen35GatedDeltaNet.decodeKernelStep` builds `GatedDeltaDecodeKernel.Weights` from `conv1d.weight`, `aLog`, `dtBias`, `norm.weight`, `norm.eps`, q/k eps 1e-6 and the scales 1/Dk and 1/sqrt(Dk), then calls `canRun` and `step`. Qwen3Next and the MLXVLM layer have the same fields and the same constants (written inline as `1e-6` and `pow(Float(headKDim), -0.5)`).
    - MLXVLM cannot import MLXLLM. Thus the shared code must go in MLXLMCommon.
    - Qwen3Next: `mixedQKV` is a concatenation in the order q, k, v; `z`, `b`, `a` are strided views. The kernel takes row-contiguous inputs (the metal kernel copies a strided input). The value head order of `z` and `v` agrees with the key head map of the kernel (`value_head / (Hv / Hk)`).
    - Qwen3Next norm is `Qwen3NextRMSNormGated` (the same norm as the MLXLLM Qwen 3.5 layer). The MLXVLM norm is `RMSNormGated`: it rounds `rmsNorm(x) * silu(gate)` in the input type, the kernel rounds one time in float32. The tolerance test must show if this gap is inside the tolerance.
    - The MLXVLM layer writes the cache inside `callAsFunction` (no `forward`). The kernel branch must write `cache[0]`, `cache[1]` and call `cache.advance(1)`.
    - Qwen3Next compiled decode runs only for float16; `linearLayerBody` calls `forward`, thus the kernel also runs inside the traced segments. The tiny test config of `Qwen3NextCompiledDecodeTests` has key head width 8, thus the kernel refuses it and those tests keep the op path.
    - Bench: `scratchpad/qwen38bench` now has `QWEN38_FACTORY=vlm` to load with `VLMModelFactory`. Base (HEAD b65b776) binary is in `qwen38bench/base-znzpvzb/`. Free disk is 1.4 TiB; the download of `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` runs (log in `scratchpad/qwen3next/download.log`).
  timestamp: 2026-10-16T12:00:30.582894+00:00
- actor: claude-code
  id: 01m52aff61srv4p1dmgaz65vbz
  text: |-
    Implementation landed (TDD).
    - MLXLMCommon: `GatedDeltaQueryKeyNorm` (GatedDelta.swift) holds the q/k RMSNorm eps and the scales 1/Dk and 1/sqrt(Dk). The op paths of the three layers and the kernel weights read it (no copies of `1e-6` and `-0.5`). `GatedDeltaDecodeKernel.Weights.init(conv:aLog:dtBias:norm:normEps:keyHeadDim:)`. `package protocol GatedDeltaDecodeLayer: Module` with one requirement `decodeKernelWeights`, and the shared `decodeKernelStep(qkv:z:b:a:convState:recState:)` in a protocol extension (moved from `Qwen35GatedDeltaNet`).
    - Qwen3Next: new `projectInputs(_:batch:sequence:)` (the projections, the q/k/v reorder and the concatenation); the kernel branch in `forward` after the mask branch.
    - MLXVLM Qwen 3.5: the kernel branch in `callAsFunction`; it writes `cache[0]`, `cache[1]` and calls `cache.advance(S)`.
    - Tests: the shared operands of the ^q0950zv tests moved to `Tests/MLXLMTests/GDNDecodeKernelFixture.swift` (one JSON config for the three config types, a table of three layer kinds). New `GDNDecodeKernelLayerTests.swift`: tolerance against the op path of each whole layer (the same layer under a mask that keeps the token), the positive control, and a wiring test, for each kind, head shape and dtype.
    - Discovery: for Qwen3-Next, on the small head shape the op path gives the same bits as the kernel, thus a bitwise test cannot show that the kernel runs. The wiring test also counts the graph nodes with `DeepSeekV4TestArrays.primitiveNodeCount`: 82 nodes on the op path against 25 with the kernel.
    - RED: the wiring test failed for Qwen3-Next and MLXVLM (graph count; for MLXVLM also the bits) before the branches. GREEN: `xcrun xctest MLXLMTests.xctest`: 721 XCTest and 1516 Swift Testing tests pass, 0 warnings.
  timestamp: 2026-10-16T12:20:05.953439+00:00
- actor: claude-code
  id: 01m52c4abh0y95jhjgtvp69w6s
  text: |-
    Benchmark (same-session A/B against the HEAD b65b776 bench binary; full mode, one prompt of about 32k tokens, 256 decode tokens; one GPU job at a time, GPU idle before each run). Report: scratchpad/qwen3next/REPORT.md.
    - MLXVLM Qwen3.8-27B-mxfp4 (`QWEN38_FACTORY=vlm`, log shows `model type: Qwen35`): decode base 30.91, 31.41, 31.60 ms/token (mean 31.31); new 30.05, 30.25, 30.24, 30.51 (mean 30.26): -1.05 ms, -3.4 %. One more base run (41.13) was an outlier with a second sourcekit-lsp at 116 % CPU; not used. Prefill 394-397 tok/s for both; peak 21.5 GiB for both.
    - Qwen3-Next-80B-A3B-Instruct-4bit (downloaded, 42 GB; 1.4 TiB was free): decode base 17.96, 16.94 (mean 17.45); new 14.45, 14.22 (mean 14.34): -3.11 ms, -17.8 %. Prefill same (1808 tok/s after the first loads); peak 46.2 GiB for both.
    - For each model the slowest new run is faster than the fastest base run. Both changes kept. Background CPU in all runs: sourcekit-lsp about 88 %, often fseventsd about 100 % and SourceKitService or swift-package about 80 %.
    - Bench: scratchpad/qwen38bench now has the MLXVLM product and `QWEN38_FACTORY=vlm`. Script: scratchpad/qwen3next/ab-znzpvzb.sh. Patch of the change: scratchpad/qwen3next/znzpvzb.patch.
    - Integration (xcodebuild -only-testing): Qwen35SessionPromptCacheTests and Qwen35CheckpointIntegrationTests, 8 tests in 2 suites pass. No Qwen3-Next and no MLXVLM Qwen 3.5 integration test exists.
    - Mutant run (GDNDecodeKernelLayerTests): the VLM branch without `cache[1] = step.recurrentState`, and the Qwen3-Next branch with `b` and `a` swapped: 18 issues, each mutant caught for each head shape and dtype; then restored.
  timestamp: 2026-10-16T12:48:57.713219+00:00
- actor: claude-code
  id: 01m52c4dsp5cck71mj52d48he4
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/GDNDecodeKernelFixture.swift (new), Tests/MLXLMTests/GDNDecodeKernelLayerTests.swift (new), Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift. `swift build --build-tests` 0 warnings; `xcrun xctest MLXLMTests.xctest` 721 XCTest + 1516 Swift Testing pass; integration 8/8 pass. Decode at 32k: Qwen3.8 MLXVLM -3.4 %, Qwen3-Next -17.8 %; prefill and peak unchanged.
    - next: /review (task stays in doing).
  timestamp: 2026-10-16T12:49:01.238309+00:00
- actor: claude-code
  id: 01m52cc4ed8pnx5vm6h1mxc4vz
  text: |-
    ### review — findings
    - evidence: `review sha HEAD~1..HEAD`, 6 findings (13 confirmed, 9 refuted). Libraries/MLXLLM/Models/Qwen3Next.swift:358, Libraries/MLXVLM/Models/Qwen35.swift:556, Libraries/MLXVLM/Models/Qwen35.swift:654, Tests/MLXLMTests/GDNDecodeKernelFixture.swift:131, Tests/MLXLMTests/GDNDecodeKernelFixture.swift:132, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:42.
    - next: /implement to correct each item in the section "Review Findings (2026-10-16 07:50)". The task stays in review.
  timestamp: 2026-10-16T12:53:13.805655+00:00
- actor: claude-code
  id: 01m52ccq6naef7y5peh57wm2wr
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 8 files; decode at 32k -3.4 % (Qwen3.8 MLXVLM), -17.8 % (Qwen3-Next)
    - test: green — run on the final tree by implement: MLXLMTests 721 + 1516, integration 8/8; 0 compiler warnings
    - commit: ab97919
    - review: findings — Libraries/MLXLLM/Models/Qwen3Next.swift:358, Libraries/MLXVLM/Models/Qwen35.swift:556, Libraries/MLXVLM/Models/Qwen35.swift:654, Tests/MLXLMTests/GDNDecodeKernelFixture.swift:131, Tests/MLXLMTests/GDNDecodeKernelFixture.swift:132, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:42
  timestamp: 2026-10-16T12:53:33.013403+00:00
- actor: claude-code
  id: 01m52dggh3x9aegrpwv3yyyzfx
  text: |-
    Review findings corrected (implement step, TDD).
    - q/k norm: new `GatedDeltaQueryKeyNorm.normalized(query:key:dtype:)` in GatedDelta.swift. It has the same ops in the same order as the old inline code (`MLXArray(scale).asType(dtype) * MLXFast.rmsNorm(x, weight: mlxNone, eps)`), thus the bits do not change. The three op paths call it: Qwen3Next.swift, MLXLLM Qwen35.swift and MLXVLM Qwen35.swift. Qwen3-Next gives `dtype: inputs.dtype` (as before); the two Qwen 3.5 layers give `q.dtype` (as before). The kernel is not changed.
    - Weights: `GatedDeltaDecodeLayer` now requires `headKDim`, `conv1d`, `aLog`, `dtBias` and `norm` (associated type `OutputNorm: GatedDeltaOutputNorm`, a new protocol with `weight` and `eps`). The protocol extension holds the one `decodeKernelWeights`. The three copies of the property and the convenience `Weights.init(...keyHeadDim:)` are deleted.
    - Access: to satisfy the package protocol, `Qwen3NextRMSNormGated` and MLXVLM `Qwen35Language.RMSNormGated` are now `package` (with `package` weight and eps), and the five witness members of `Qwen3NextGatedDeltaNet` and MLXVLM `GatedDeltaNet` are `package`. No public API changes.
    - Tests: fixture head shapes use named constants (`Qwen38Heads`, `SmallHeads`, `qwen38HeadCase`, `smallHeadCase`). The layer kinds are named (`mlxLLMQwen35`, `qwen3Next`, `mlxVLMQwen35`). `Qwen35GDNDecodeKernelTests.operands` calls `Fixture.layer(Fixture.mlxLLMQwen35, ...)` with `#require(... as? Qwen35GatedDeltaNet)`; the local `layer(_:dtype:)` is deleted. The `gatedNorm` stubs and the `headKDim` requirement of the test protocol are deleted (the shared protocol now gives `norm` and `headKDim`).
    - New test `GatedDeltaQueryKeyNormTests` (table: key head widths of the fixture x dtypes): q and k against a float32 reference with factors 1/D and 1/sqrt(D). RED: build error, no member `normalized`. GREEN: 4 cases pass.
    - Discovery: `replace_all` of the files tool replaced only one of three identical blocks; each stub needed its own edit.
  timestamp: 2026-10-16T13:13:05.827367+00:00
- actor: claude-code
  id: 01m52dgnqv90687q6xqne4rwb9
  text: |-
    Verification after the corrections.
    - `swift build --build-tests`: 0 warnings (only the known note "missing creator for mutated node").
    - `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 721 XCTest, 0 failures; Swift Testing 1517 tests in 125 suites pass. Log: scratchpad/qwen3next/znz-fix-xctest.log.
    - xcodebuild -only-testing Qwen35SessionPromptCacheTests and Qwen35CheckpointIntegrationTests: 8 tests in 2 suites pass, TEST SUCCEEDED. Log: scratchpad/qwen3next/znz-fix-integ.log.
    - Decode at 32k, Qwen3-Next-80B-A3B-Instruct-4bit (`ab-znzpvzb.sh`, GPU idle before each run, 32153 prompt tokens, 256 decode tokens). `fix` = bench built from this tree (scratchpad/qwen38bench/fix-znzpvzb), `new` = the bench of commit ab97919. Order fix new fix new, then new fix fix new. fix: 14.83, 14.75, 14.33, 14.50 ms/token (mean 14.60). new: 14.61, 14.54, 14.53, 14.38 (mean 14.52). The gap is inside the noise (the second set has fix faster). Base was 17.45. Prefill 1765-1805 tok/s, peak 46.2 GiB for all. Results: scratchpad/qwen3next/znz-nextfix-*.txt and znz-nextfix2-*.txt.
  timestamp: 2026-10-16T13:13:11.163992+00:00
- actor: claude-code
  id: 01m52dgs25cyrmsxrjjq1w7564
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLLM/Models/Qwen3Next.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/GDNDecodeKernelFixture.swift, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift, Tests/MLXLMTests/GatedDeltaQueryKeyNormTests.swift (new). 6 of 6 findings checked. Build 0 warnings; MLXLMTests 721 XCTest + 1517 Swift Testing pass; integration 8/8 pass; Qwen3-Next decode at 32k 14.60 ms/token (commit binary 14.52, base 17.45).
    - next: /review (task stays in doing).
  timestamp: 2026-10-16T13:13:14.565058+00:00
- actor: claude-code
  id: 01m52dm9p83c6a9hfedcpj3y2t
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (554c2d5). 0 findings, 0 confirmed, 0 refuted. 7 validator runs, 0 failed. All 6 findings of round 1 are checked.
    - next: none. The task is in done.
  timestamp: 2026-10-16T13:15:09.896493+00:00
- actor: claude-code
  id: 01m52dmr78596gm4b68hyv9yy9
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 8 files; the 6 findings fixed; Qwen3-Next decode at 32k stays at about 14.6 ms/token
    - test: green — run on the final tree by implement: MLXLMTests 721 + 1517, integration 8/8; 0 compiler warnings
    - commit: 554c2d5
    - review: clean — 0 findings; 6/6 prior findings checked; task moved to done
  timestamp: 2026-10-16T13:15:24.776161+00:00
position_column: done
position_ordinal: ffe180
title: Use the fused GDN decode kernel in Qwen3Next and in the MLXVLM Qwen 3.5 model
---
## What

^q0950zv added `GatedDeltaDecodeKernel` (Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift). It runs the body of one gated delta net layer on one decode token in one Metal kernel. Only `Qwen35GatedDeltaNet` in MLXLLM calls it. Two other GDN layers have the same op path and do not use it:

- `Qwen3NextGatedDeltaNet.forward` (Libraries/MLXLLM/Models/Qwen3Next.swift). Its conv input comes from a concatenation and its `z` is a strided view, thus the kernel copies `z` first.
- The GDN layer of Libraries/MLXVLM/Models/Qwen35.swift (no compiled decode, op path only).

## Approach

1. Call `GatedDeltaDecodeKernel.canRun` / `step` in the S = 1, no-mask, not-training branch of each layer, as `Qwen35GatedDeltaNet.decodeKernelStep` does.
2. Add a tolerance test against the op path of each layer, with a positive control.
3. Measure decode with a model of each kind; keep the change only if decode is faster.

## Acceptance Criteria

- [x] Each layer uses the kernel on one decode token, with a tolerance test and a positive control.
- [x] Decode is faster, measured, for each model.
- [x] All MLXLMTests pass, with 0 compiler warnings. #performance

## Review Findings (2026-10-16 07:50)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 8 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5000W20Q4CW37GXFZNZPVZB.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5000W20Q4CW37GXFZNZPVZB.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/Qwen3Next.swift:358` `reuse/reuse` — The q and k RMSNorm with scale is written inline again. It matches the block in MLXLLM Qwen35.swift and MLXVLM Qwen35.swift. A shared helper should own it. Move the q/k normalization into a shared GatedDeltaQueryKeyNorm function and call it here instead of the inline scale and rmsNorm.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:556` `reuse/reuse` — The decodeKernelWeights property repeats the body of the same property in the other two layer classes. The copies can diverge. Use one shared builder for GatedDeltaDecodeKernel.Weights, and call it from this property.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:654` `reuse/reuse` — The q and k RMSNorm with scale is written inline a third time. The same math is in MLXLLM Qwen35.swift and Qwen3Next.swift. One shared helper should hold it. Call a shared GatedDeltaQueryKeyNorm normalization function here instead of the inline scale and rmsNorm.
- [x] `Tests/MLXLMTests/GDNDecodeKernelFixture.swift:131` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/GDNDecodeKernelFixture.swift:132` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:42` `reuse/reuse` — The new test helper layer(_:dtype:) builds and prepares a Qwen35GatedDeltaNet by hand. The new shared fixture already does this for every layer kind. The two copies can drift. Replace the local layer(_:dtype:) with a call to Fixture.layer(Fixture.layerKinds[0], headCase, dtype:) or a direct Fixture.layer call, and cast the result to Qwen35GatedDeltaNet if the test needs the concrete type.