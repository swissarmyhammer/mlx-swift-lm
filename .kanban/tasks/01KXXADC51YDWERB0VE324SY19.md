---
assignees:
- claude-code
depends_on:
- 01KXX2JDFWQ79CXZE1P1FJMY9F
position_column: todo
position_ordinal: '8880'
title: 'Fix regression: IntegrationTestingTests xcodeproj target failed to build after makeXgTokenizer signature change (^r9rf5g7)'
---
## What

Task ^r9rf5g7 (hybrid Mamba/attention prompt-cache work, commits `7bb20a8`/`a1c1385`/`7c0522f`) added a required `configuration: ModelConfiguration` parameter to `MLXLanguageModel.makeXgTokenizer(modelID:tokenizer:configuration:)` (`Libraries/MLXFoundationModels/MLXLanguageModel.swift`), and updated the one SwiftPM call site (`Tests/MLXFoundationModelsTests/XgTokenizerStopRegistrationTests.swift`), but missed all ~12 call sites inside the hand-authored `IntegrationTesting/IntegrationTesting.xcodeproj` `IntegrationTestingTests` target. That target does not build under SwiftPM (`swift test` never compiles it), so `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting` has been silently broken since that merge and nobody noticed.

Discovered and fixed while implementing ^1fjmy9f (hybrid Qwen3.6 prompt-cache integration test) because that task's own required verification command needed the target to build at all.

## Fix already applied (as part of ^1fjmy9f's branch work)

Added the missing `configuration: context.configuration` argument at each call site:
- `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/GuidedGeneration/GuidedGenerationTests.swift` (2 call sites)
- `.../GuidedGeneration/GuidedGenerationBenchmarkTests.swift` (2)
- `.../GuidedGeneration/HardReserveStressTests.swift` (1)
- `.../GuidedGeneration/MaxTokenTruncationTests.swift` (2)
- `.../GuidedGeneration/MultiModelGuidedGenerationTests.swift` (3)
- `.../GuidedGeneration/GenerableRoundTripTests.swift` (1)
- `.../GuidedGeneration/Grammar/EmitStopSignalTests.swift` (1)

`xcodebuild -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' build-for-testing` now succeeds.

## Acceptance Criteria

- [ ] Confirm the fix above is present on `main` (it currently lives on the `foundationmodels-fixes` branch as part of ^1fjmy9f's changes) -- review/merge as appropriate.
- [ ] Consider adding a CI check (or at minimum a documented manual step) that runs `xcodebuild build-for-testing` for the `IntegrationTesting` scheme, since this class of break is invisible to `swift test` and could recur silently after any signature change touched only in SwiftPM call sites.

## Tests

- [ ] `xcodebuild -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' build-for-testing` succeeds