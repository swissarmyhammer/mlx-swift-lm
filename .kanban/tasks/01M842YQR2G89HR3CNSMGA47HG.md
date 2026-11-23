---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84cev71js6c17xppx3ndv49
  text: |-
    Verified the uncommitted fix against the card.
    - `MLXLanguageModelGeneration.swift`: new `Executor.answering(_:config:thinkingEnabled:tokenizer:)` appends `ReasoningConfig.answerOpening` via `ThinkingRender.openingAnswer` only when thinkingEnabled == false. `openingAnswer` returns the input unchanged when `answerOpening` is nil, so non-Inkling reasoning models (Qwen etc.) are not affected.
    - `MLXLanguageModel.makeReasoningPlan`: both the `suppressed` (reasoning undeclared) and `reasoning` (declared; no_think gives false) prepared inputs are wrapped. Only the no-tools text path. Tool turns keep the template prompt, as the card says.
    - Scripted tests `InklingExecutorThinkingOffPromptTests` (3 tests): no_think, undeclared, and thinking-on control. The dispatcher reported they fail without the fix.
    - Real-checkpoint test `Inkling2ThinkingOffSummaryTests`: the dispatcher reported 6/6 runs with no reasoning and a full response. I did not run it (no GPU jobs in this step).
    Fixes to the rules (integration test only): the `guard #available … else { return }` in the test is replaced by `@available` on the test function (the optionals rule says never `guard` in a test). Doc comments added to the suite, `prompt`, `promptTranscript`, `run` and the test. The literals 24/3/100/3/400 are now named static constants.
    Discovered: pre-existing build warning `class 'MTPDrafterRegistry' must restate inherited '@unchecked Sendable' conformance` in MTPDrafterModelFactory.swift (not touched here). Filed as ^4rbbqvb.
    Still to do after landing: tell foundationmodelsrouter-23 so it can remove `ReasoningControl.reducible` (Router ci/inkling-matrix 18ad4a19).
  timestamp: 2026-11-23T14:20:16.993977+00:00
- actor: claude-code
  id: 01m84cexpqty96r3jy8hey09x8
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXFoundationModelsTests/InklingExecutorThinkingOffPromptTests.swift, IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift; swift-format applied; `swift build --build-tests` Build complete, no warnings in changed files (one pre-existing warning in MTPDrafterModelFactory.swift, filed as ^4rbbqvb)
    - next: test step runs MLXFoundationModelsTests (InklingExecutorThinkingOffPromptTests), then /review
  timestamp: 2026-11-23T14:20:19.543058+00:00
- actor: claude-code
  id: 01m84d4tf6vebjm0s5q31b9xcq
  text: |-
    ### test — green
    - evidence: swift build --build-tests (exit 0; one warning "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS" from the build system; the MTPDrafterRegistry @unchecked Sendable warning (^4rbbqvb) was not re-emitted because the build was incremental). xcrun xctest per bundle, all exit=0: MLXLMTests Test run with 2604 tests in 249 suites passed, 3 known issues (expected), Executed 1260 tests with 0 failures; MLXGuidedGenerationTests Test run with 95 tests in 22 suites passed; MLXFoundationModelsTests Test run with 605 tests in 84 suites passed; CXGrammarTests Test run with 7 tests in 5 suites passed; MLXHuggingFaceMacrosTests Executed 11 tests with 0 failures. The 3 new InklingExecutorThinkingOffPromptTests tests pass. 0 skipped. IntegrationTesting not run.
    - next: review/commit
  timestamp: 2026-11-23T14:32:17.126709+00:00
position_column: doing
position_ordinal: '80'
title: Inkling thinking-off text calls still reason and can return an empty answer
---
Reported by the Router session (CI run 45078982291, `.emptySummary`). With `.custom("no_think")`, Inkling 2 Mini still opens a thinking frame: `reasoning_effort: none` is only a hint. On a blank transcript with a long summary prompt and maxTokens 400, three runs reasoned for 817, 287 and 1233 chars; the last answer was cut to 50 chars. One run with reasoning undeclared came back fully empty.

Fix: on the no-tools text path, a thinking-off prompt (no_think, or reasoning undeclared) gets the `answerOpening` header appended after it, the way schema passes already do, so thinking off is a hard off. Tool turns keep the template prompt.

Tests: `InklingExecutorThinkingOffPromptTests` (scripted) and `Inkling2ThinkingOffSummaryTests` (real checkpoint).

When this lands, tell foundationmodelsrouter-23 so it can remove `ReasoningControl.reducible` (Router ci/inkling-matrix 18ad4a19).

#inkling #bug