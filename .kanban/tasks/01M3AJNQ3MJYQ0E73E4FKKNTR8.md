---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3arg11n2zq8fn7pfs2p6bze
  text: |-
    ### Root cause (research)

    The NR0 flash kernel is not the cause. The seeds, the codebook table, the rotation and the shared codec cache are all deterministic. The cause is a data race in the Metal source `fusedEncodeWHTSource` (`Libraries/MLXLMCommon/TurboQuantKernels.swift`). The cache uses this kernel to encode all keys and values for power-of-2 head dimensions.

    Phase 2 of the kernel does the WHT butterfly stages that cross a SIMD group (stages 5 ..< LogDim; Dim=64 has 1 such stage, Dim=128 has 2, Dim=256 has 3). Each stage had ONE barrier only, after the write. Thread d read `shared_buf[d]` and `shared_buf[partner]` and then wrote `shared_buf[d]`. The partner is in a different SIMD group. Thus thread d could write its new value before the partner read the old value. The partner then used a value from the wrong stage. The encoded indices then changed from call to call. The attention output of testStandardAttentionRepeatFactors (d=64) was thus sometimes bad (cos 0.888). The GPU schedule decides if the race occurs, which is why other GPU work (the full bundle) made the failure more frequent.

    Proof before the fix (probe, 8192 rows, 50 repeated calls on the same input): all 50 calls gave output different from the first call, for dim 64, 128 and 256. Index mismatches against the MLX butterfly (`TurboQuantRotation.fwhtForward` + `boundaryQuantize`): 929-2246 (dim 64), 7058-9526 (dim 128), 59622-65678 (dim 256). After the fix: 0 changed calls, and 0 or 1 mismatches in 1M (float sum order of the norm only).

    ### Fix
    Each cross-SIMD stage now reads the partner value, has a barrier, writes the new value, and has a second barrier. The arithmetic is the same as before (lower = partner + own, upper = partner - own). The tolerance of testStandardAttentionRepeatFactors did not change.

    ### Other kernels checked
    `fusedEncodeSource`, `fusedQuantizePackScaledSource`, the single-pass kernel and pass 2 with fused rotation: each one writes threadgroup memory and has a barrier before other threads read it, and no stage reads and writes the same cells without a barrier. No other butterfly on threadgroup memory is in Libraries (VarianceNormalizedKVCache uses the MLX `hadamardTransform`).

    ### Regression test
    `TurboQuantIntegrationTests.testWHTEncodeKernelMatchesButterflyAndRepeats`: 20 repeated calls must be bit-identical, and at most 1e-5 of the indices can differ from the MLX butterfly. It fails on the old kernel (6 failures) and passes on the new kernel.
  timestamp: 2026-09-24T22:27:36.117207+00:00
- actor: claude-code
  id: 01m3argbv7qs37awjx1nmhebq7
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/TurboQuantKernels.swift (two barriers for each cross-SIMD WHT butterfly stage), Tests/MLXLMTests/TurboQuantTests.swift (new testWHTEncodeKernelMatchesButterflyAndRepeats). The new test fails on the old kernel with 6 failures. `for i in $(seq 20); do xcrun xctest -XCTest MLXLMTests.TurboQuantIntegrationTests ...; done`: 20 of 20 runs pass, 23 tests each, 0 failures.
    - next: /test (full MLXLMTests bundle)
  timestamp: 2026-09-24T22:27:47.175479+00:00
- actor: claude-code
  id: 01m3arn3zvzb47m5xtzsb8zpv2
  text: |-
    ### test — green (against the known baseline of ^fbhgd7k)
    - evidence: `swift build --build-tests`: only the known warning `missing creator for mutated node`. `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: XCTest 695 tests, 36 failures (all baseline: GlmOcrContinuationTests 6, NanbeigeTests 1, Qwen25VLContinuationTests 13, Qwen35ContinuationTests 8, Qwen3VLContinuationTests 8); Swift Testing 1207 tests, 101 issues (baseline). 0 TurboQuant failures. MLXGuidedGenerationTests 70 pass, MLXFoundationModelsTests 299 pass, CXGrammarTests 7 pass, MLXHuggingFaceMacrosTests 5 pass.
    - next: /commit
  timestamp: 2026-09-24T22:30:22.971817+00:00
position_column: doing
position_ordinal: '80'
title: TurboQuantIntegrationTests.testStandardAttentionRepeatFactors fails intermittently (cos 0.888 < 0.95 at rep=2)
---
## What

`MLXLMTests.TurboQuantIntegrationTests.testStandardAttentionRepeatFactors` (`Tests/MLXLMTests/TurboQuantTests.swift:2190`) fails on some runs and passes on others, with the same seeds.

Measured on 2026-09-24, on branch `stable`:
- Full MLXLMTests run: failed with `XCTAssertGreaterThan failed: ("0.88828874") is not greater than ("0.95") - rep=2: cos 0.88828874`.
- `xcrun xctest -XCTest MLXLMTests.TurboQuantIntegrationTests .build/out/Products/Debug/MLXLMTests.xctest`, 5 times: 1 failure, 4 passes.

The inputs use fixed `MLXRandom.key` seeds and `TurboQuantKVCache(seed: 5)`, thus the result must be deterministic. A result that changes from run to run points to a race in the compressed-attention kernel for GQA rep=2 (the NR0 fast path), or to uninitialized memory.

MLXLMTests does not depend on MLXFoundationModels. The failure was found during the test step of ^2mk47nr, which changes only MLXFoundationModels. It is not in the known baseline (the ...ContinuationTests suites and NanbeigeTests).

## Acceptance Criteria
- [x] Find the cause of the nondeterminism and correct it.
- [x] The test passes on 20 consecutive runs of the TurboQuantIntegrationTests suite.

## Tests
- [x] `for i in $(seq 20); do xcrun xctest -XCTest MLXLMTests.TurboQuantIntegrationTests .build/out/Products/Debug/MLXLMTests.xctest || break; done`