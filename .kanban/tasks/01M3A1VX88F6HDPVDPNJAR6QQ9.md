---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3amga0zwvtzyhgqn1ad7b8k
  text: |-
    ### Research
    - `runRespond` (MLXLanguageModel.swift ~1069) calls `checkOutPromptCache`, which today logs `spilledColdStartLine`, deletes the file and returns nil. The slot is made before `container.perform`.
    - The three plan sites are `runAllowedToolGeneration`, `runUnconstrained` and `runReasoning`. The guided path calls `carriesNoCache` and never plans: a spilled cache of a guided turn is not restored, and the `defer` deletes its file.
    - `ExecutorPromptCacheFile.read(from:key:templates:)` evaluates every array before it returns, thus the executor can delete the file after the read.
    - `loadPromptCacheSnapshot(url:into:)` throws on a layer-count mismatch. The scripted model with 0 cache layers (used by `respondOnce`) thus gets a restore failure and a cold turn, and the existing spool test stays valid.
    - `ScriptedLanguageModel` with `cacheLayerCount: 1` and `PromptBytesInputProcessor` gives real warm turns (see `PromptCacheScopeTests`).
    - Integration models are in the local Hugging Face cache: `mlx-community/Llama-3.2-1B-Instruct-4bit` and `mlx-community/Qwen3.8-27B-mxfp4`.
    - IntegrationTesting uses synchronized folders: a new file needs no project edit.
  timestamp: 2026-09-24T21:17:51.007945+00:00
- actor: claude-code
  id: 01m3ansgd4v92e7mfr221gy48z
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift (new), Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, Tests/MLXFoundationModelsTests/PromptCacheScopeTests.swift, Tests/MLXFoundationModelsTests/ScriptedModelTestSupport.swift, IntegrationTesting/.../TextGeneration/PromptCacheSpoolIntegrationTests.swift (new), IntegrationTesting/.../TextGeneration/PromptCacheReuseChannelTests.swift, IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift
    - The slot holds a private `Carried` enum (none / memory / restored / pendingRestore). `init(spilled:key:)` and `restoreIfPending(model:parameters:)` are new; the executor calls `restoreIfPending` before each of the three `plan` calls. `runRespond` deletes the spilled file in a `defer`.
    - The plan line now has `source=none|memory|disk`, and `restoreSeconds=` for disk. `spilledColdStartLine` is replaced by `restoreFailureLine`.
    - What did not work: the first spool integration run asserted that the third session is on disk before its second turn. It was not: the grown entry of the first session (9.9 MB) is larger than the one-session budget (8.4 MB), thus the store spills that entry itself and the third session stays in memory. The assertion now applies to each session but the last.
    - xcodebuild needs `-skipPackagePluginValidation` (the CudaBuild plugin of mlx-swift fails validation without it), as CI uses.
    - next: /test
  timestamp: 2026-09-24T21:40:21.028284+00:00
- actor: claude-code
  id: 01m3ap3nvkhc8sa0kjh8warqj2
  text: |-
    ### test — red (the failures were there before this change; this change adds no failure and no warning)
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`.
    - MLXFoundationModelsTests: 299 tests in 34 suites passed, 3 runs (291 baseline + 8 new).
    - MLXGuidedGenerationTests: 70 passed. CXGrammarTests: 7 passed. MLXHuggingFaceMacrosTests: 5 passed.
    - MLXLMTests: 36 XCTest failures and 101 Swift Testing issues, the same set in 2 runs. They are numeric tolerance failures in MLXLMCommon/MLXLLM/MLXVLM tests (DeepSeek-V4 Python reference, GLM-OCR, Qwen VL continuation, SSM). The MLXLMTests target graph does not contain MLXFoundationModels, and this change touches no file of that graph, thus the failures were there before this change. New task: ^fbhgd7k.
    - Integration: `xcodebuild build-for-testing -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64'` — TEST BUILD SUCCEEDED. PromptCacheSpoolIntegrationTests passed (1 test). Qwen35SessionPromptCacheTests passed (2 tests).
    - next: /commit
  timestamp: 2026-09-24T21:45:54.291560+00:00
- actor: claude-code
  id: 01m3ap3v8panp2176x2079vn41
  text: |-
    ### Integration results (2026-09-24)

    **PromptCacheSpoolIntegrationTests** (`mlx-community/Llama-3.2-1B-Instruct-4bit`, greedy, 3 sessions, memory budget = the bytes of one session, 8 388 608): passed in 7.6 s.

    | session | on disk before turn 2 | prompt | cached | cold control cached | same answer |
    |---|---|---|---|---|---|
    | 1 | yes | 63 | 47 | 0 | yes |
    | 2 | yes | 67 | 51 | 0 | yes |
    | 3 | no (memory) | 62 | 46 | 0 | yes |

    Executor plan lines: `source=disk restoreSeconds=0.003 rendered=63 reused=47 fed=16 rule=extend`, and `source=disk restoreSeconds=0.005 rendered=67 reused=51 fed=16 rule=extend`. Spill writes took 0.009 to 0.015 s. Session 3 stays in memory because the grown entry of session 1 (9 928 704 bytes) is larger than the budget, thus the store spills that entry itself.

    **Qwen35SessionPromptCacheTests** (`mlx-community/Qwen3.8-27B-mxfp4`, `MambaCache` + `KVCacheSimple`, greedy): 2 tests passed in 54 s.
    - From memory (existing test): turn 2 `source=memory rendered=126 reused=101 fed=25 rule=splice`.
    - From disk (new test, memory budget 0): turn 1 spilled 170 722 160 bytes in 0.298 s (file 160 581 757 bytes). Turn 2 `source=disk restoreSeconds=0.061 rendered=126 reused=101 fed=25 rule=splice`; cachedTokenCount 101 of 126, turn 1 rendered 73; answer contains "teal".
  timestamp: 2026-09-24T21:45:59.830027+00:00
depends_on:
- 01M3A1RHPV3CV6Q7Q59W0S77DT
- 01M3A29W6YK4F0VMGBCB28DXZ2
position_column: doing
position_ordinal: '80'
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
- [x] A session whose entry went to disk comes back warm: the reused token count is greater than 0, and under greedy sampling the output equals a cold run of the same transcript.
- [x] The file is deleted after a successful restore.
- [x] A corrupt spilled file gives a cold turn that succeeds, and the file is deleted.
- [x] A turn cancelled after `.spilled` and before the restore leaves no file.
- [x] A turn whose `container.perform` throws after `.spilled` leaves no file.
- [x] Qwen3.5 (`MambaCache` + `KVCacheSimple`) comes back warm from disk (integration suite passes).
- [x] The spool integration suite passes: more sessions than the memory budget holds, interleaved, and each later turn is warm.

## Tests
- [x] Unit: new `Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift`. Drive `Executor.respond` with a small test model inside `ExecutorPromptCacheStore.$current.withValue(store)`, where `store` has a tiny memory budget, thus every entry spills. One test for each unit criterion.
- [x] Integration: new `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheSpoolIntegrationTests.swift`, with the method of `PromptCacheReuseChannelTests`.
- [x] Integration: extend `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift` with a spill and a restore.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.
- [x] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` compiles (`swift build` does not see `IntegrationTesting/`), and the two suites pass with `-only-testing:`.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.
- After the run, write the integration results as a comment on this task.