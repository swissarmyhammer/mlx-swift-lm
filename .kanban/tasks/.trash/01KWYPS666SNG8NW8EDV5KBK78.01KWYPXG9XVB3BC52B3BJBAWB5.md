---
position_column: todo
position_ordinal: '8880'
title: Fix GenerationOptions.SamplingMode.Kind rename breaking build against current Xcode-beta SDK
---
## What
Discovered while working on FoundationModelsMultitool's task `h6rqz4v` (Retire callTool/DirectToolCall escape hatch), which vendors this repo at the pinned commit `e6ccd272` on branch `mlx-foundationmodels`. Building FoundationModelsMultitool against the currently-installed Xcode-beta SDK fails inside this repo's own source because Apple renamed `GenerationOptions.SamplingMode.Kind`'s cases: `.top`/`.nucleus` are now `.randomTopK`/`.randomProbabilityThreshold`.

Confirmed (via `git stash` in the consuming repo) this is pre-existing and unrelated to any change there — it is purely this repo's source vs. the newer SDK's renamed API. It was worked around locally only inside the consumer's gitignored `.build/checkouts/` directory (not committed anywhere) to unblock verification builds.

## Acceptance Criteria
- [ ] Update every reference to `GenerationOptions.SamplingMode.Kind.top`/`.nucleus` in this repo's source to the current SDK names `.randomTopK`/`.randomProbabilityThreshold` (or add a compatibility shim if this repo needs to support both old and new SDKs).
- [ ] This repo builds cleanly against the current Xcode-beta SDK with no manual workaround.
- [ ] Existing test suites pass with no regressions.
- [ ] Once fixed, FoundationModelsMultitool's pin can be bumped past `e6ccd272` to pick this up (tracked there separately if needed).