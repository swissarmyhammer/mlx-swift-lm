---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: IntegrationTesting target does not compile on qwen-perf (DeepseekV4 switch, Qwen38FourBitMTPModel ContainerLoader)
---
Found while working ^q1yp3b1. `xcodebuild build-for-testing -skipPackagePluginValidation -skipMacroValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` fails with TEST BUILD FAILED on files that change did not touch:

- `IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift`: `error: switch must be exhaustive`.
- `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/Qwen38FourBitMTPModel.swift`: `error: cannot convert return expression of type 'MLXLanguageModel.ContainerLoader' ... to return type 'MLXLanguageModel.ContainerLoader'` (same printed type: likely a `@concurrent` / `@Sendable` attribute mismatch of the closure).

## Acceptance
- [ ] The IntegrationTesting scheme builds for testing with zero errors (CI `integration_tests.yml` runs it).
#qwen-perf