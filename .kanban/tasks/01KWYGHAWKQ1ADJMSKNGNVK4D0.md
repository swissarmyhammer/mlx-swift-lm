---
comments:
- actor: claude-code
  id: 01kwz7e00v3fkcppkeee034cpd
  text: |-
    Fixed and pushed. Updated Executor.samplingMode(from:) in MLXLanguageModel.swift:694-697 to match the current SDK's GenerationOptions.SamplingMode.Kind cases (.randomTopK(_:seed:) / .randomProbabilityThreshold(_:seed:)), replacing the old .top/.nucleus pattern matches. Verified: swift build succeeds (clean), and the full test suite passes via xcodebuild build-for-testing + xcrun xctest (391 tests, 52 suites, 0 failures across CXGrammarTests/MLXGuidedGenerationTests/MLXFoundationModelsTests/MLXLMTests).

    Committed as 280869d on branch foundationmodels-fixes (not mlx-foundationmodels — per team decision, foundationmodels-fixes is now the working branch for this batch of fixes, to be merged back later) and pushed to origin/foundationmodels-fixes.

    Note: this task's acceptance criteria mentioned pushing to origin/mlx-foundationmodels specifically — that branch is intentionally being kept as a clean, unmodified mirror of upstream for easy syncing. The fix instead landed on the new foundationmodels-fixes branch, which is what task y6cbj25 (consumer pin migration) is now blocked on and expects.
  timestamp: 2026-07-07T21:23:34.043503+00:00
position_column: done
position_ordinal: '80'
title: Fix SamplingMode.Kind case rename for current beta SDK (.top/.nucleus -> .randomTopK/.randomProbabilityThreshold)
---
MLXFoundationModels/MLXLanguageModel.swift's SamplingModeMapper.samplingMode(from:) pattern-matches FoundationModels.GenerationOptions.SamplingMode.Kind cases .top(let k, _) / .nucleus(let threshold, _). The current beta toolchain (Xcode 27 beta, Swift 6.4, macOS 27.0 SDK) renamed these cases to .randomTopK(_: Int, seed: UInt64?) and .randomProbabilityThreshold(_: Double, seed: UInt64?), so this file fails to compile against that SDK -- breaking every downstream consumer (e.g. FoundationModelsRouter) that builds against it.

Confirmed against the real FoundationModels.framework arm64e-apple-macos.swiftinterface on this machine: .top/.nucleus no longer exist as case names.

A local one-off fix (renaming the two case matches) was verified working on branch mlx-foundationmodels, commit e6ccd2721ab3b236b92e436bee2130086f48041c as the parent -- but that fix currently sits uncommitted/unpushed in a downstream consumer's .build/checkouts sandbox and was never landed here. This task is to land the real fix in this repo's own history and push it.

Acceptance criteria:
- SamplingModeMapper.samplingMode(from:) uses the SDK's current case names (.randomTopK/.randomProbabilityThreshold), or is written to be resilient to this kind of beta-SDK rename going forward.
- swift build succeeds against the current FoundationModels beta SDK.
- Fix is committed and pushed to origin/mlx-foundationmodels so downstream consumers can bump their Package.resolved pin to it.