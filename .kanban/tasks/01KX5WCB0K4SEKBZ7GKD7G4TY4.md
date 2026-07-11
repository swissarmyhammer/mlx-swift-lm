---
assignees:
- claude-code
position_column: todo
position_ordinal: 9b80
title: Apply the swift-test metallib bootstrap fix to MLXLMTests and MLXGuidedGenerationTests
---
## What
`23ff1zx` fixed `swift test`'s "Failed to load the default metallib" crash for `MLXFoundationModelsTests` by adding a once-only bootstrap (`Tests/MLXFoundationModelsTests/TestBootstrap.swift`, `MetalLibraryTestBootstrap.ensureColocatedMetallib`) that symlinks `mlx.metallib` next to the running test binary, satisfying mlx-swift's `Cmlx` loader's first probe path.

Confirmed during that task's verification: `MLXLMTests` and `MLXGuidedGenerationTests` still crash identically under plain `swift test` (untouched, out of scope for that task's title/acceptance criteria which scoped it to `MLXFoundationModelsTests`).

## Acceptance Criteria
- [ ] Apply the same (or a shared/extracted) bootstrap mechanism to `Tests/MLXLMTests` and `Tests/MLXGuidedGenerationTests` so `swift test --filter MLXLMTests` and `swift test --filter MLXGuidedGenerationTests` no longer abort on GPU eval.
- [ ] Consider extracting the bootstrap logic to a shared location (e.g. a small internal test-support target, or duplicated per-target static like the original if a shared target is overkill) rather than copy-pasting verbatim across 3 test targets.
- [ ] No regression to the existing `xcodebuild build-for-testing` + `xcrun xctest` safe pattern already used elsewhere in this project.

## Scope
`Tests/MLXLMTests/`, `Tests/MLXGuidedGenerationTests/`. Not urgent/blocking — this is a test-infrastructure convenience (enables plain `swift test` for these bundles), the project's mandated safe test-invocation pattern already works around this via `xcodebuild`/`xcrun xctest`.