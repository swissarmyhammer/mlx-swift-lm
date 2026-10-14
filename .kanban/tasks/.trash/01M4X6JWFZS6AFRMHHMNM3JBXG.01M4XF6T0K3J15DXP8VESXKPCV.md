---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: 'IntegrationTesting: remove the "no calls to throwing functions occur within ''try'' expression" warnings'
---
## What

`xcodebuild test` of the IntegrationTesting scheme (HEAD bc35319) shows Swift warnings `no calls to throwing functions occur within 'try' expression [#UnnecessaryEffectMarker]` in test files that the DeepSeek-V4 work did not change:

- `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/GuidedGeneration/Grammar/TokenizerVocabExtractorTests.swift`
- `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/GuidedGeneration/HardReserveStressTests.swift` (5 places)

Found during ^tj8ds2m (log in the session scratchpad `it-parity-tj8.log`).

## Acceptance Criteria

- [ ] Each `try` that has no throwing call is removed, or the call is a throwing call again.
- [ ] `xcodebuild build-for-testing` of the IntegrationTesting scheme shows zero Swift warnings in the IntegrationTestingTests target. #testing