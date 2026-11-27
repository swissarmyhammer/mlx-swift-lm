---
comments:
- actor: claude-code
  id: 01m8fb20x4f4k4rrx3zm7pmtq5
  text: |-
    Research + implementation notes:
    - Chokepoint: every generation path (text, reasoning, allowed/required tool, schema) runs inside `container.perform` in `MLXLanguageModel.Executor.runRespond` via `dispatchGeneration`. The fix wraps that call in a new `Executor.reportingClusterFailure(of:_:)`, which calls the new public `ClusterLanguageModel.checkFailure()` (backed by `RemoteModelSession.checkFailure()`) after the generation ends, and also in place of an error the generation threw (a guided pass the failed forward cancelled throws `CancellationError`). The outer catch of runRespond passes a non-GrammarError through unchanged, and LanguageModelSession.respond/streamResponse surface it unwrapped: the session-level tests assert `ClusterModelError.memberLost(...)` by value and type (coordinator requirement).
    - Test support: `InProcessRemoteModelWorker.failForwards(after:with:)` fails only forward ops after N forwards (the existing `failOps(after:)` counts every op).
    - Discovery, filed as a new task: a guided (schema / required-tool) pass whose cluster forward fails hangs 60 s (`GenerationDrainCoordinator.quiescenceTimeout`): the forward cancels the respond task itself, whose cancel handler waits on the section that same thread holds. Measured: schema turn failed after 60.103 s. Because of that, the catch branch of the helper is unit-tested directly (`aGenerationCancelledByALostMemberThrowsTheLoss`) rather than through a 60 s guided turn.
    - Not done here (needs the 2-Mac setup): FoundationModelsRouter IntegrationTests `lostMemberFallsBackToThisMac` on m5u1+m5u2.
  timestamp: 2026-11-27T20:27:27.012062+00:00
- actor: claude-code
  id: 01m8fbbbp25x8mpqvqksm7cscg
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift (public `checkFailure()`), Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift (`RemoteModelSession.checkFailure()`), Libraries/MLXFoundationModels/MLXLanguageModel.swift (`Executor.reportingClusterFailure(of:_:)` around `dispatchGeneration`), Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift (`failForwards(after:with:)`), Tests/MLXFoundationModelsTests/ClusterExecutorTests.swift (3 tests: aResponseWhoseForwardLosesAMemberThrowsTheLoss, aStreamWhoseForwardLosesAMemberThrowsTheLossAfterItsSnapshots, aGenerationCancelledByALostMemberThrowsTheLoss; each watched RED first). Full bundles: MLXLMTests 2788 Swift Testing + 1261 XCTest passed (3 pre-existing known issues, unrelated); MLXFoundationModelsTests 621 passed. swift-format run on the changed files.
    - open: FoundationModelsRouter `lostMemberFallsBackToThisMac` on the 2-Mac setup is still to run (needs m5u1+m5u2). Guided-path 60 s hang filed as ^axrb0cw.
    - next: /review
  timestamp: 2026-11-27T20:32:32.962973+00:00
- actor: claude-code
  id: 01m8fbx2vkvf4r7mcqbkr4cgn4
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (no warnings from project sources); swift-format lint on the 5 changed files clean; xcrun xctest, all exit 0: MLXLMTests 2788 Swift Testing tests / 279 suites + 1261 XCTest, 0 failures; MLXGuidedGenerationTests 96 tests; MLXFoundationModelsTests 621 tests; CXGrammarTests 7 tests; MLXHuggingFaceMacrosTests 7+3+11 XCTest suites, 0 failures. No skips.
    - note: MLXLMTests reports 3 known issues, all deliberate withKnownIssue pins in untouched files (ToolTests.swift MiniCPM5 CDATA, 2 issues; AttentionKernelToleranceTests.swift near-tie, 1 issue), last changed in merge 99f1f242, not by this task.
    - next: review
  timestamp: 2026-11-27T20:42:13.747729+00:00
- actor: claude-code
  id: 01m8fc12eah7smyt8b10edwwz1
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (6468c3f0) — 0 findings (0 confirmed, 1 refuted, 7 attempted, 0 failed); 5 files reviewed, 4 .kanban files not matched by any validator; no prior Review Findings sections
    - next: task moved to done
  timestamp: 2026-11-27T20:44:24.394168+00:00
- actor: claude-code
  id: 01m8fc1f4td289vsz38vq17j07
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 5 files, 3 tests added in ClusterExecutorTests
    - test: green — all 5 bundles (MLXLMTests 2788+1261, MLXGuidedGenerationTests 96, MLXFoundationModelsTests 621, CXGrammarTests 7, MLXHuggingFaceMacrosTests 21 suites)
    - commit: 6468c3f0
    - review: clean — 0 findings
  timestamp: 2026-11-27T20:44:37.402436+00:00
position_column: done
position_ordinal: ffffff8780
title: 'Clustered models: a generation whose forward fails returns its partial text instead of throwing the failure'
---
## What
FoundationModelsRouter's 2-Mac test lostMemberFallsBackToThisMac (2026-11-27, Qwen3.8-27B-mxfp4 split over m5u1+m5u2) kills the worker on the member mid-answer. The streamed answer ends 1.17 s after the kill with no error, and its text is "\n". The Router then can't tell the answer failed, so it never moves the session to a local copy (^hvygt9p). The next request of another session throws `ClusterModelError.memberLost` from `beginRequest`, and that session does fail over and answers on the local copy.

## Cause
`ClusterLanguageModel.forward` (Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift:204) catches the failure, records it (`session.recordFailure`), cancels the current task and returns zero logits. The decode loop then ends as if the generation had finished, and MLXLanguageModel returns the partial response. Nothing reads the recorded failure when the generation ends: `RemoteModelSession.hasFailed` is read only by `RemoteKVCache` at deinit, and the failure surfaces only at the next `beginRequest`.

## Fix
When a generation ends and its ClusterLanguageModel session has a recorded failure, the respond/stream of MLXLanguageModel throws that failure (`ClusterModelError.memberLost` for a lost member) and doesn't return the partial text. A stream throws after the snapshots it already gave. For example, expose the recorded failure on ClusterLanguageModel (a throwing `checkFailure()`), and call it in the executor after the generation loop ends and before the response completes.

Tests: a unit test with a stub transport that fails the Nth forward: `respond` throws `memberLost`, and `streamResponse` throws after its earlier snapshots. Then FoundationModelsRouter IntegrationTests lostMemberFallsBackToThisMac passes on the 2-Mac setup.

Router side: FoundationModelsRouter ^hvygt9p (ce284eab) already retries a submission that throws a ClusterModelError on a copy on this Mac.
Evidence: FoundationModelsRouter run 2026-11-27 11:35 on m5u1, output "E2E lost member: the answer ended 1.173709875 seconds after the kill" / "E2E lost member answer: \"\\n\"". #cluster