---
depends_on:
- 01KWYGHAWKQ1ADJMSKNGNVK4D0
position_column: todo
position_ordinal: '8880'
title: Migrate consumer pins (Router + Multitool) from mlx-foundationmodels to foundationmodels-fixes
---
## What
Once `gnvk4d0` (the `SamplingMode.Kind` case-rename fix) lands and is pushed to `origin/foundationmodels-fixes`, downstream consumers need to move their SwiftPM pins from the old `mlx-foundationmodels` branch to the new `foundationmodels-fixes` branch — **together, in one coordinated change**, not independently.

**Constraint discovered while investigating this (confirmed by hitting it directly):** SwiftPM resolves one branch requirement per package across the whole dependency graph. `FoundationModelsMultitool` depends on `mlx-swift-lm` both directly AND transitively through `FoundationModelsRouter` (which has its own `.package(url: .../mlx-swift-lm, branch: "mlx-foundationmodels")` pin in its `Package.swift`). Changing only one consumer's pin produces:
```
error: mlx-swift-lm is required using two different revision-based requirements (foundationmodels-fixes and mlx-foundationmodels), which is not supported
```
So this is a single coordinated edit across at least two repos, not two independent tasks.

## Acceptance Criteria
- [ ] `gnvk4d0`'s fix is committed and pushed to `origin/foundationmodels-fixes` first (this task is blocked on it).
- [ ] `../FoundationModelsRouter/Package.swift`'s `mlx-swift-lm` dependency pin changed from `branch: "mlx-foundationmodels"` to `branch: "foundationmodels-fixes"`, its `Package.resolved` re-locked, and its own build/test suite verified green.
- [ ] `../FoundationModelsMultitool/Package.swift`'s `mlx-swift-lm` dependency pin (and its doc comment referencing the branch name) changed the same way, `Package.resolved` re-locked, `swift build`/`swift build --build-tests`/`swift test` verified green.
- [ ] `swift package resolve` succeeds in Multitool with no "two different revision-based requirements" error (i.e. both pins agree).