---
assignees:
- claude-code
depends_on:
- 01KX3MZDHS31HAGZ36N34EPSM4
position_column: todo
position_ordinal: '8880'
title: Establish CPU-device MLX tensor unit-test pattern (metallib workaround)
---
## What
Chunk-store work needs unit tests that actually evaluate MLXArrays (slice/concat/compare), but `swift test` crashes on GPU eval ("Failed to load the default metallib" — see memory swiftpm-test-gpu-metallib-limit). Establish the pattern: pin tensor-evaluating test suites to the CPU device (e.g. `Device.withDefaultDevice(.cpu)` or `MLX.Device.setDefault(device: .cpu)` in suite init — implementer verifies the current mlx-swift API). Add a helper in Tests/MLXFoundationModelsTests/PromptCacheTestSupport.swift (e.g. `withCPUDevice { }` or suite-level setup) and one proving test.

## Acceptance Criteria
- [ ] A test that slices and concatenates small random MLXArrays and asserts element equality passes under plain `swift test` (no GPU, no metallib crash)
- [ ] Pattern documented in PromptCacheTestSupport.swift doc comments for the following tasks to reuse

## Tests
- [ ] New test in Tests/MLXFoundationModelsTests (e.g. PromptCacheTestSupport or a small suite): create [1,2,8,4] random arrays, slice [..., a..<b, ...], concat, assert allClose/equal on CPU
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.