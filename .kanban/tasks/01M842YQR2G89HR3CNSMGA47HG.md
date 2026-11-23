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
- actor: claude-code
  id: 01m84d5hw3xm81ztfjg441k04t
  text: |-
    ### commit — changed
    - evidence: 672e699a fix(fm): hard thinking-off for Inkling text calls via the answer opening (^mga47hg)
    - next: none (local commit only, not pushed)
  timestamp: 2026-11-23T14:32:41.091323+00:00
- actor: claude-code
  id: 01m84d9yfn66ndbn95aa9tv15f
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (672e699a) — 4 findings (4 confirmed, 2 refuted) — IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift:132, :135, :136 (code-hygiene/disallowed-constructs-swift); Tests/MLXFoundationModelsTests/InklingExecutorThinkingOffPromptTests.swift:51 (reuse/reuse)
    - next: implement the 4 items in `## Review Findings (2026-11-23 08:32)`; task stays in review
  timestamp: 2026-11-23T14:35:05.077024+00:00
- actor: claude-code
  id: 01m84dae0v5vv63qh2cm891t21
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 4 files (fix already in tree, test doc/constants tidy)
    - test: green — 5 bundles: MLXLMTests 2604 (3 known), Guided 95, FM 605, CXGrammar 7, Macros 11
    - commit: 672e699a
    - review: findings — Inkling2ThinkingOffSummaryTests.swift:132, :135, :136 (no_direct_standard_out_logs); InklingExecutorThinkingOffPromptTests.swift:51 (reuse effortToken)
  timestamp: 2026-11-23T14:35:20.987377+00:00
- actor: claude-code
  id: 01m84de1t90k0j9crp9ze3h1t8
  text: |-
    Fixed the 4 review findings.
    - no_direct_standard_out_logs: all three `print` calls in Inkling2ThinkingOffSummaryTests are gone. The suite now writes its result lines through `Inkling2ResultLog.report` (tag `INKLING2 SUMMARY`, unified log `com.apple.FoundationModels-MLX`/`Inkling2`). Every other Inkling 2 integration suite uses this, including the ones under MLXFoundationModelsIntegration/. The literal 300 is now `reportedPrefixLength`. The reasoning `#expect` message now carries the char count and a prefix of the reasoning, so a CI failure shows what the model wrote. I avoided `+` concatenation inside the `#expect` comment (`Comment` has no `+`). IntegrationTesting is not compiled by `swift build`, so I checked this edit by reading it only.
    - reuse: added `public func effortToken(_ level: String) -> Int` on `InklingFrameTokenizer` (Tests/MLXLMTestSupport). It is public because MLXLMTestSupport is a separate module and its other members are public. I deleted the private copies in InklingExecutorThinkingOffPromptTests and InklingExecutorSchemaPromptTests. I also folded in the two inline `tokenizer.token(InklingFrameTokenizer.effortPrefix + …thinkingOffEffort)` copies in InklingExecutorToolLoopTests and Tests/MLXLMTests/Inkling2ToolLoopSessionTests.
  timestamp: 2026-11-23T14:37:19.561731+00:00
- actor: claude-code
  id: 01m84de4mwy8wgre50vbg7mzjv
  text: |-
    ### implement — changed
    - evidence: 6 files — IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift, Tests/MLXLMTestSupport/InklingFrameTokenizer.swift, Tests/MLXFoundationModelsTests/InklingExecutorThinkingOffPromptTests.swift, Tests/MLXFoundationModelsTests/InklingExecutorSchemaPromptTests.swift, Tests/MLXFoundationModelsTests/InklingExecutorToolLoopTests.swift, Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift; swift-format applied to each; `swift build --build-tests` Build complete (only the build-system "missing creator for mutated node" warning); 4/4 findings checked
    - next: test (MLXFoundationModelsTests, MLXLMTests), then /review
  timestamp: 2026-11-23T14:37:22.460382+00:00
- actor: claude-code
  id: 01m84e6nvx6w5fefrnpdjshh51
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0 (only a SwiftPM "missing creator for mutated node" bundle note, no compiler warnings). xcrun xctest on 5 bundles all exit 0: MLXLMTests 2604 Swift Testing (3 expected known issues) + 1260 XCTest, 0 failures; MLXGuidedGenerationTests 95; MLXFoundationModelsTests 605; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 XCTest, 0 failures. xcodebuild IntegrationTesting Inkling2ThinkingOffSummaryTests exit 0: "Test run with 1 test in 1 suite passed after 69.413 seconds", TEST SUCCEEDED; INKLING2 SUMMARY lines show reasoning 0 chars for no_think and nil-level calls.
    - next: review/commit
  timestamp: 2026-11-23T14:50:46.525121+00:00
- actor: claude-code
  id: 01m84e7c7et83a310rjnaw9akr
  text: |-
    ### commit — changed
    - evidence: 0e5d9dc6 test(fm): log the Inkling summary suite and share effortToken (^mga47hg review)
    - next: none
  timestamp: 2026-11-23T14:51:09.422595+00:00
- actor: claude-code
  id: 01m84e8zreyt0ndbg2rathf02w
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (0e5d9dc6) — 0 findings, 0 confirmed, 0 refuted; 7 attempted, 0 failed; 6 files reviewed, 2 kanban files skipped (no validator); all 4 prior findings checked
    - next: none — card moved to done. Tell foundationmodelsrouter-23 so it can remove `ReasoningControl.reducible` (Router ci/inkling-matrix 18ad4a19).
  timestamp: 2026-11-23T14:52:02.190516+00:00
- actor: claude-code
  id: 01m84e981vzkypd58tqd2zwr91
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 6 files, 4/4 findings checked
    - test: green — 5 bundles (2604/95/605/7/11) + Inkling2ThinkingOffSummaryTests 6/6 no reasoning
    - commit: 0e5d9dc6
    - review: clean — 0 findings
  timestamp: 2026-11-23T14:52:10.683883+00:00
position_column: done
position_ordinal: ffffdd80
title: Inkling thinking-off text calls still reason and can return an empty answer
---
Reported by the Router session (CI run 45078982291, `.emptySummary`). With `.custom("no_think")`, Inkling 2 Mini still opens a thinking frame: `reasoning_effort: none` is only a hint. On a blank transcript with a long summary prompt and maxTokens 400, three runs reasoned for 817, 287 and 1233 chars; the last answer was cut to 50 chars. One run with reasoning undeclared came back fully empty.

Fix: on the no-tools text path, a thinking-off prompt (no_think, or reasoning undeclared) gets the `answerOpening` header appended after it, the way schema passes already do, so thinking off is a hard off. Tool turns keep the template prompt.

Tests: `InklingExecutorThinkingOffPromptTests` (scripted) and `Inkling2ThinkingOffSummaryTests` (real checkpoint).

When this lands, tell foundationmodelsrouter-23 so it can remove `ReasoningControl.reducible` (Router ci/inkling-matrix 18ad4a19).

#inkling #bug

## Review Findings (2026-11-23 08:32)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M842YQR2G89HR3CNSMGA47HG.jsonl` — no validator matches this file
> - `.kanban/tasks/01M842YQR2G89HR3CNSMGA47HG.md` — no validator matches this file
> - `.kanban/tasks/01M84CEHAPVWFM1KJY04RBBQVB.jsonl` — no validator matches this file
> - `.kanban/tasks/01M84CEHAPVWFM1KJY04RBBQVB.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift:132` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift:135` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Reasoning/Inkling2ThinkingOffSummaryTests.swift:136` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `Tests/MLXFoundationModelsTests/InklingExecutorThinkingOffPromptTests.swift:51` `reuse/reuse` — The new `effortToken(_:)` helper is an exact copy of a private helper already in the sibling Inkling test suite. Two identical copies will drift apart, and a fix to the effort-token lookup would have to be made twice. Delete the private `effortToken` here and share one helper for both suites: move it to a shared test-support file (for example beside `InklingFrameTokenizer`) as an internal static on `InklingFrameTokenizer` or a free function, and have both suites call it.
