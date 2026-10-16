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
position_column: doing
position_ordinal: '80'
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