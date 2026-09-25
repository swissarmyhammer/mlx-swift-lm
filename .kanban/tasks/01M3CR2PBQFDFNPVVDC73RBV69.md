---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3crm6a7fqz63d9nhmc2wj13
  text: |-
    Research and implementation:
    - `MLXGuidedGenerationTests` has no float32 comparison (no tolerance, no `accuracy:`, no matmul, no float32 use). Thus only `MLXFoundationModelsTests` gets `MLXTestPrecision`. `CXGrammarTests` and `MLXHuggingFaceMacrosTests` use no MLX arrays.
    - `Package.swift`: `MLXTestPrecision` is a dependency of `MLXFoundationModelsTests`.
    - New `Tests/MLXFoundationModelsTests/Float32PrecisionTests.swift`: the guard test calls `mlx_test_precision_tf32_is_disabled()` (this call also keeps the constructor in the linked bundle). A second test compares a 48x32 by 32x48 float32 matmul on the GPU with the same matmul on the CPU stream (the CPU does not use TF32), within 1e-4. This test proves the effect directly, and does not need the unknown input arrays of the 502.14685 checksum.
    - `ExecutorPromptCacheQwenFileTests.swift`: the cold check is `<= coldTolerance` (1e-3) again. `decodeLogits`, `coldNoiseFactor` and the noise-floor text are deleted, because nothing uses them now.
    - `CLAUDE.md` "Float32 tests and TF32" names `MLXLMTests` and `MLXFoundationModelsTests`, and names the bundles without float32 comparisons.
    - Red proof: with `MLX_ENABLE_TF32=1`, the FM bundle fails with 4 issues: both Qwen cases fail the 1e-3 cold check, the guard test fails, and the GPU/CPU matmul test fails. Without the variable: 307 tests pass, 3 runs.
  timestamp: 2026-09-25T17:08:21.447181+00:00
- actor: claude-code
  id: 01m3crm8ba4wxmvjtvxjg5mp4z
  text: |-
    ### implement — changed
    - evidence: 4 files — Package.swift, CLAUDE.md, Tests/MLXFoundationModelsTests/Float32PrecisionTests.swift (new), Tests/MLXFoundationModelsTests/ExecutorPromptCacheQwenFileTests.swift. FM bundle 307 tests pass x3; with MLX_ENABLE_TF32=1 it fails with 4 issues (red proof).
    - next: /test
  timestamp: 2026-09-25T17:08:23.530904+00:00
- actor: claude-code
  id: 01m3crsfek94thknzp9b8y0hat
  text: |-
    ### test — green
    - evidence: swift build --build-tests (only the known warning `missing creator for mutated node`); xcrun xctest: MLXLMTests 695 XCTest + 1208 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 307 (4 runs), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. 0 failures, 0 skipped.
    - next: /commit
  timestamp: 2026-09-25T17:11:14.643019+00:00
position_column: doing
position_ordinal: '80'
title: Give MLXFoundationModelsTests the MLX_ENABLE_TF32=0 test setting, and restore the 1e-3 cold bound of the Qwen file tests
---
#prompt-cache

## What

The same float32 `matmul` gives different values in two test bundles on this machine (Apple M5 GPU with NAX). In `MLXLMTests` a checksum of a 48x32 by 32x48 product is 502.14685. In `MLXFoundationModelsTests` it is 501.8337. With `MLX_ENABLE_TF32=0` the second bundle also gives 502.14685.

Known cause (task ^fbhgd7k, commit a691739): on a GPU with NAX, MLX computes float32 matmul in TF32 unless `MLX_ENABLE_TF32=0`. The C target `Tests/MLXTestPrecision` sets `MLX_ENABLE_TF32=0` in a load-time constructor when the variable is not set. `Package.swift` gives this target to `MLXLMTests` only. `MLXFoundationModelsTests` does not depend on it, thus that bundle runs in TF32. The note in `CLAUDE.md` ("Float32 tests and TF32") names `MLXLMTests` only.

Effect: in `MLXFoundationModelsTests` a split forward and a single forward of a tiny Qwen model differ by approximately 3e-3, not 1e-6. `Tests/MLXFoundationModelsTests/ExecutorPromptCacheQwenFileTests.swift` (task ^4q07zch) thus uses a noise-floor bound of approximately 3e-2 for its cold check, not the 1e-3 that task asked for.

Work:
- In `Package.swift`, add `MLXTestPrecision` to the dependencies of `MLXFoundationModelsTests`. Examine whether the other float32-comparing test targets (`MLXGuidedGenerationTests`) need it too, and add it where a float32 comparison exists.
- Make sure the linker keeps the constructor in each bundle (the MLXLMTests fix calls a C symbol from a test for this reason; see `Tests/MLXLMTests/Float32PrecisionTests.swift`). Add the same guard test in each bundle that gets the target.
- In `ExecutorPromptCacheQwenFileTests.swift`, set the cold check back to 1e-3 against a cold prefill, as ^4q07zch asked.
- Update the `CLAUDE.md` section "Float32 tests and TF32" to name every bundle that has the setting.

## Acceptance Criteria
- [ ] In `MLXFoundationModelsTests` the float32 matmul checksum equals the `MLXLMTests` value (502.14685) without an environment variable set by the caller.
- [ ] `ExecutorPromptCacheQwenFileTests` checks the cold prefill within 1e-3 and passes.
- [ ] A guard test in each bundle that gets `MLXTestPrecision` asserts that the setting is in effect.
- [ ] `CLAUDE.md` names each bundle that has the setting.

## Tests
- [ ] Guard test in `Tests/MLXFoundationModelsTests/` (same pattern as `Tests/MLXLMTests/Float32PrecisionTests.swift`).
- [ ] `swift build --build-tests`, then run all five bundles with `xcrun xctest .build/out/Products/Debug/<Bundle>.xctest`: 0 failures, 0 skipped.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.