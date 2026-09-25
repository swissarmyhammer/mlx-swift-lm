---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: Find why MLX uses TF32 matmul in the MLXFoundationModelsTests process and not in MLXLMTests
---
## What

The same float32 `matmul` gives different values in two test bundles on the same machine. In `MLXLMTests` a sum of a 48x32 by 32x48 product is 502.14685. In `MLXFoundationModelsTests` it is 501.8337. With `MLX_ENABLE_TF32=0` the second bundle gives 502.14685 also.

Cause found so far: `metal::is_nax_available()` (`mlx/backend/metal/device.h:268`) is true in the MLXFoundationModelsTests process, thus `matmul.cpp:366` selects the NAX kernel in TF32 for float32. It is false in the MLXLMTests process. Both bundles hold the same `default.metallib` (same SHA-1). The check uses `__builtin_available(macOS 26.2, ...)` and the GPU architecture generation. A possible cause is the deployment target or the SDK version of the binary that holds Cmlx in each bundle.

Effect: in MLXFoundationModelsTests a split forward and a single forward of a tiny Qwen model differ by approximately 3e-3, not 1e-6. `Tests/MLXFoundationModelsTests/ExecutorPromptCacheQwenFileTests.swift` thus uses the noise-floor rule `max(10 * noiseFloor, 1e-3)` for its cold check (task ^4q07zch).

## Acceptance Criteria

- [ ] The cause of the different `is_nax_available()` result is known and written on this task.
- [ ] A decision: either the bundles use the same matmul path, or the difference is documented in CLAUDE.md.

## Tests

- Run the probe from task ^4q07zch: a float32 matmul checksum in each bundle, with and without `MLX_ENABLE_TF32=0`.

#prompt-cache