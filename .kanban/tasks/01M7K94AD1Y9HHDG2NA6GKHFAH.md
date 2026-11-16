---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7kb51vgseevf1sr04m5cs4r
  text: |-
    Origin: neither error comes from qwen-perf. `git diff --quiet stable HEAD -- Libraries/MLXLMCommon/GenerationResults.swift Libraries/MLXFoundationModels IntegrationTesting` is empty, and stable == origin/stable == 87460a3f (the merge-base). Both come from the upstream squash merge 6dc1e4e7 on stable, which added `Generation.steering(SteeringEvent)` and changed `ContainerLoader` to a deprecated alias with a new `ProgressContainerLoader`.

    - The printed ContainerLoader-vs-ContainerLoader error is misleading. The real mismatch is `case .router: testLoad()`: `testLoad()` (FMTestHelpers) returns `ProgressContainerLoader`, but `LoadPath.containerLoader` declared the deprecated `ContainerLoader`. Fix: declare `ProgressContainerLoader` and forward `progress:` in the `.text` case.
    - `.steering`: the first build stopped at one switch. After the first fix, a second unhandled switch appeared in `DeepseekV4IntegrationTests.aShortToolPromptEmitsOneDSMLToolCall`. Neither test sends a steering instruction, so both treat a steering event as a fault: the assessment's `collect` throws `IntegrationTestFailure` (as it does for `.rejectedToolCall`), and the @Test records an Issue.
    - The same merge left 5 deprecation warnings in IntegrationTesting: GenerationBenchmarkTests (`progressHandler:` loadContainer), GuidedGenerationBenchmarkTests (deprecated `init(...load:)` and `progressHandler:`), and Qwen38FourBitMTPModel. All are moved to the progress-aware API.
  timestamp: 2026-11-16T23:30:22.192842+00:00
- actor: claude-code
  id: 01m7kb9gb75v4j98yqp5b7e025
  text: |-
    ### implement — changed
    - evidence: 5 files, all under IntegrationTesting/IntegrationTestingTests: DeepseekV4AgenticPromptCacheAssessmentTests.swift, DeepseekV4IntegrationTests.swift, GenerationBenchmarkTests.swift, MLXFoundationModelsIntegration/GuidedGeneration/GuidedGenerationBenchmarkTests.swift, MLXFoundationModelsIntegration/TextGeneration/Qwen38FourBitMTPModel.swift. The CI command `xcodebuild build-for-testing ... -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64'` gives TEST BUILD SUCCEEDED, 0 errors, and 0 source warnings (the only line is the toolchain's appintentsmetadataprocessor notice). `swift build --build-tests` succeeds. `xcrun xctest MLXLMTests.xctest`: XCTest 1134 executed, 0 failures, 4 skipped. Swift Testing: 2372 tests passed, with 3 known issues. swift-format has run on the changed files.
    - The 4 skips are LFM2 checkpoint tests that an environment variable gates. They came from 6dc1e4e7 and were filed as a new task. The real-model IntegrationTesting suites were not run, as instructed.
    - next: /review
  timestamp: 2026-11-16T23:32:48.103895+00:00
position_column: doing
position_ordinal: '80'
title: IntegrationTesting target does not compile on qwen-perf (DeepseekV4 switch, Qwen38FourBitMTPModel ContainerLoader)
---
Found while working ^q1yp3b1. `xcodebuild build-for-testing -skipPackagePluginValidation -skipMacroValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` fails with TEST BUILD FAILED on files that change did not touch:

- `IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift`: `error: switch must be exhaustive`.
- `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/Qwen38FourBitMTPModel.swift`: `error: cannot convert return expression of type 'MLXLanguageModel.ContainerLoader' ... to return type 'MLXLanguageModel.ContainerLoader'` (same printed type: likely a `@concurrent` / `@Sendable` attribute mismatch of the closure).

## Acceptance
- [ ] The IntegrationTesting scheme builds for testing with zero errors (CI `integration_tests.yml` runs it).
#qwen-perf