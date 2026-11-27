---
assignees:
- claude-code
position_column: todo
position_ordinal: '9080'
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