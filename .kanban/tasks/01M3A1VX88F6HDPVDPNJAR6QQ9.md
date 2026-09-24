---
assignees:
- claude-code
depends_on:
- 01M3A1RHPV3CV6Q7Q59W0S77DT
- 01M3A29W6YK4F0VMGBCB28DXZ2
position_column: todo
position_ordinal: '8580'
title: Restore a spilled prompt cache in MLXLanguageModel.Executor, and prove warm restores on a real model
---
#prompt-cache

(Plan item F3, executor side, and the real-model acceptance of F4.)

## What

In `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, `runRespond` checks out the session's entry (about `:973-980`) and checks it in (about `:1107-1109`). After ^w0s77dt, `checkOut` can return `.spilled(handle)`, which that task handles as a cold start. This task restores it.

- When `checkOut` gives `.spilled`, the key has left the store's `onDisk`, and the executor owns the file. Delete the file in a `defer` that runs on EVERY exit path of `runRespond`: success, throw, cancellation, and a failed `container.perform` before the restore runs.
- The restore needs `model.newCache(parameters:)` as templates, and it evaluates arrays, thus it runs INSIDE `container.perform`. The slot is made before `perform` today. Give `ExecutorPromptCacheSlot` a pending-restore state: `init(spilled: handle, key:)`, plus a method `restoreIfPending(model:parameters:)` that runs ONCE, before the first `plan`. The executor calls `plan` at three sites; call `restoreIfPending` at each (it is a no-op after the first call).
- `restoreIfPending` calls `ExecutorPromptCacheFile.read(from:key:templates:)` (^z6av3ep). Any error (class mismatch, corrupt file, offset mismatch, missing file) gives a cold start and one log line, never a failed turn.
- The plan log line (`ExecutorPromptCacheReport.planLine`) names the source (`source=memory|disk|none`) and, for `disk`, the restore time.
- Test isolation: bind a per-test store with `ExecutorPromptCacheStore.$current.withValue(...)` (from ^ddjhenh), never the shared store.

## Acceptance Criteria
- [ ] A session whose entry went to disk comes back warm: the reused token count is greater than 0, and under greedy sampling the output equals a cold run of the same transcript.
- [ ] The file is deleted after a successful restore.
- [ ] A corrupt spilled file gives a cold turn that succeeds, and the file is deleted.
- [ ] A turn cancelled after `.spilled` and before the restore leaves no file.
- [ ] A turn whose `container.perform` throws after `.spilled` leaves no file.
- [ ] Qwen3.5 (`MambaCache` + `KVCacheSimple`) comes back warm from disk (integration suite passes).
- [ ] The spool integration suite passes: more sessions than the memory budget holds, interleaved, and each later turn is warm.

## Tests
- [ ] Unit: new `Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift`. Drive `Executor.respond` with a small test model inside `ExecutorPromptCacheStore.$current.withValue(store)`, where `store` has a tiny memory budget, thus every entry spills. One test for each unit criterion.
- [ ] Integration: new `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheSpoolIntegrationTests.swift`, with the method of `PromptCacheReuseChannelTests`.
- [ ] Integration: extend `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift` with a spill and a restore.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.
- [ ] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` compiles (`swift build` does not see `IntegrationTesting/`), and the two suites pass with `-only-testing:`.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.
- After the run, write the integration results as a comment on this task.