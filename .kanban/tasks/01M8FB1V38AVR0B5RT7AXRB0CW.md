---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8fvvzhxs4xt0rcascjnr5fx
  text: 'Research + RED: confirmed the cause on the card. New tests in ClusterExecutorTests (aSchemaTurnWhoseForwardLosesAMemberThrowsTheLossPromptly, aRequiredToolTurnWhoseForwardLosesAMemberThrowsTheLossPromptly) threw memberLost but after 60.12 s / 60.01 s (RED on the elapsed < 10 s expectation). Discovery: the required-tool pass of the fixture (ToolCallAlphabetTokenizer) runs ONE forward (the prompt) and ends incompleteOutput, so failForwards(after: 8) never strikes; the required-tool test uses after: 0. Fix approach: ClusterLanguageModel.withoutCancellingOnFailure(_:) (TaskLocal) keeps a failed forward from cancelling the current task; FM''s withGuidedGenerationSection runs the pass in it (runGuidedPass); GuidedTarget.checkFailure() throws the kept failure after each forward (prefill, feedOneAtATime, verify) and after a draft round, so the guided loop throws the unwrapped ClusterModelError at the failed forward. Not cancelling the respond task also keeps a caller (FoundationModelsRouter retry) from seeing its task cancelled. Rejected alternative: making the drain coordinator skip a section held by the cancelling thread — it would still cancel the respond task.'
  timestamp: 2026-11-28T01:21:14.813733+00:00
- actor: claude-code
  id: 01m8fwbxepnyt2z37bbdq8m4g8
  text: |-
    ### implement — changed
    - evidence: time to throw went from 60.12 s to 0.095 s (schema turn) and from 60.01 s to 0.009 s (required-tool turn); both throw memberLost("m5u2.local") unwrapped. Files: Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift (new withoutCancellingOnFailure(_:) with a TaskLocal; failGeneration skips the cancel inside it), Libraries/MLXGuidedGeneration/GuidedTarget.swift (checkFailure(); prefill/feedOneAtATime/verify throw the kept failure), GuidedSpeculation.swift and GuidedGenerationLoop.swift (try at the call sites, plus a check after each draft round), Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift (runGuidedPass wraps the guided section body), MLXLanguageModel.swift (doc), Documentation.docc/distributed-inference.md (doc). Tests: ClusterExecutorTests gets 2 new tests plus the extracted requiredToolRequest/schemaRequest helpers. ClusterLanguageModelTests gets aForwardThatFailsWithoutCancellingOnFailureLeavesItsTaskUncancelled, which failed when the guard was mutated to always cancel. Full runs: MLXFoundationModelsTests 656 passed, MLXGuidedGenerationTests 96 passed, MLXLMTests 2845 Swift Testing tests passed (3 known issues) and 1261 XCTest tests passed, 0 failures. swift-format applied.
    - next: /review
  timestamp: 2026-11-28T01:29:56.950823+00:00
- actor: claude-code
  id: 01m8fwx8knhvg593262kvckwww
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 warnings, 0 errors); xcrun xctest per bundle, all exit 0: MLXLMTests 2845 Swift Testing + 1261 XCTest, 0 failures; MLXGuidedGenerationTests 96 tests; MLXFoundationModelsTests 656 tests; CXGrammarTests 7 tests; MLXHuggingFaceMacrosTests 11 XCTest; 0 skipped. MLXLMTests reports 3 known issues, all intentional withKnownIssue negative checks in files this change does not touch (AttentionKernelToleranceTests, ToolTests MiniCPM5). swift-format lint on changed Swift files clean.
    - next: review
  timestamp: 2026-11-28T01:39:25.429554+00:00
- actor: claude-code
  id: 01m8fx1gj1qy7yhf939vw782rm
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (a33396b7): 0 findings (14 attempted, 0 failed); 8 files reviewed, 5 not matched by any validator (.kanban task files, Documentation.docc/distributed-inference.md)
    - next: none; task moved to done
  timestamp: 2026-11-28T01:41:44.641776+00:00
- actor: claude-code
  id: 01m8fx1v6yy9zrn15xm4c3b8h2
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 9 files, 3 tests added; time-to-throw schema 60.12 s → 0.095 s, required-tool 60.01 s → 0.009 s
    - test: green — all 5 bundles (MLXLMTests 2845+1261, MLXFoundationModelsTests 656, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: a33396b7
    - review: clean — 0 findings
  timestamp: 2026-11-28T01:41:55.550197+00:00
position_column: done
position_ordinal: ffffff8c80
title: 'Clustered models: a guided pass whose forward fails waits 60 s on its own drain section before it throws'
---
## What
A schema or required-tool turn (the guided loop) on a `ClusterLanguageModel` whose forward fails mid-answer takes 60 s to throw. Found while implementing ^rrkhywq: a schema turn over `InProcessRemoteModelWorker.failForwards(after: 8, with: InProcessMemberLoss(...))` (the shape of `ClusterExecutorTests.aSchemaTurnWritesTheLocalResponse` plus the fault) failed after 60.103 s.

## Cause
`GuidedGenerationLoop.run` runs synchronously on the respond task, inside `withGuidedGenerationSection` (MLXLanguageModelGeneration.swift), so the drain coordinator counts one active section. `ClusterLanguageModel.forward` catches the failure and calls `withUnsafeCurrentTask { $0?.cancel() }`, which here cancels the respond task itself. The respond's `withTaskCancellationHandler` (MLXLanguageModel.Executor.respond) then runs `GenerationDrainCoordinator.cancelActiveAndAwaitQuiescence()` synchronously on that same thread, and that waits for `activeSections == 0`, a section the waiting thread holds. It waits until `quiescenceTimeout` (60 s), then the pass ends and the turn throws (since ^rrkhywq: the kept `ClusterModelError`; before it: `CancellationError`).

The text and allowed-tool paths do not hang: their forward runs on the generation task, not the respond task.

## Fix
A failed cluster forward on the guided loop must end the pass without the respond task waiting on its own section, for example: stop the guided loop on a recorded cluster failure (the `onText` callback can also check `ClusterLanguageModel.checkFailure()`), and keep the forward from cancelling a task whose cancellation handler drains the section it runs in.

Test: a unit test in ClusterExecutorTests: a schema turn whose worker fails the Nth forward throws `ClusterModelError.memberLost` within a few seconds (well under `quiescenceTimeout`). #cluster