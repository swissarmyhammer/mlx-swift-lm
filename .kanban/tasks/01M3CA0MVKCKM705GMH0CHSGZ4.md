---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cvtgjjkgp2fngpessb7kf7
  text: |-
    Research: the executor takes `runReasoning` when the model declares `.reasoning`, the request has no tools and no schema, and `context.configuration.reasoningConfig` is set (the resolver reads the configuration of the ModelContext, not of MLXLanguageModel). With `toolCallFormat == nil` the format is `.json`, which gives no protocol decoder, thus the detokenizer branch of the stream closure runs. The scripted byte model writes `<think>plan</think>A`; the reasoning path streams `plan` as reasoning text and `A` as response text. A turn off the reasoning path would stream the delimiters as response text, thus the reasoning-text assertion proves that `runReasoning` ran.

    Changes (tests only, no library bug found):
    - `ScriptedModelTestSupport.swift`: `makeScriptedContainer` and `ScriptedSessionModel.make` take an optional `reasoningConfig` (the model then declares `.reasoning` and thinks before its answer); `ScriptedPassResult` records `reasoningText`.
    - `ExecutorPromptCacheRestoreTests.swift`: `SpilledSession.make` takes `reasoningConfig`; the disk-restore and corrupt-file checks take `reasoningConfig` and return the turn; `CacheEchoSession`/`CacheEchoRun` became `TwoTurnSession`/`TwoTurnRun` with a `makeModel` parameter (default: the cache-echo model). Three new tests: memory warm turn, disk warm turn with the spilled file deleted, corrupt file gives a cold turn that succeeds. All three assert the reasoning text and the response text.

    MLXFoundationModelsTests: 312 tests passed (309 + 3 new).
  timestamp: 2026-09-25T18:04:14.290222+00:00
- actor: claude-code
  id: 01m3cvtjbv74ccv1qeth3h2gab
  text: |-
    ### implement — changed
    - evidence: 2 files — Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift, Tests/MLXFoundationModelsTests/ScriptedModelTestSupport.swift; `swift build --build-tests` clean (only the known `missing creator for mutated node` warning); MLXFoundationModelsTests 312 passed, 0 failed.
    - next: /test
  timestamp: 2026-09-25T18:04:16.123206+00:00
- actor: claude-code
  id: 01m3cvvw8pw5h1m7n4stwrrkfc
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` clean (only the known `missing creator for mutated node` warning); `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` 3 runs — each 312 tests in 36 suites passed, 0 failed, 0 skipped. Only MLXFoundationModelsTests files changed.
    - next: /commit
  timestamp: 2026-09-25T18:04:59.030449+00:00
position_column: doing
position_ordinal: '80'
title: Test the prompt cache path of runReasoning (restore, plan, commit)
---
## What

File: `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, function `MLXLanguageModel.Executor.runReasoning(input:reasoningConfig:primedInside:...)`, lines 2444-2603.

Coverage (unit bundles, merged): 0% (0/160 lines). Its three closures are also 0%: the generation closure 2471-2480 (0/10), the stream closure 2484-2544 (0/61), and the completion closure 2550-2561 (0/12).

Uncovered ranges: 2444-2603 (all).

This is the path of a model that declares `.reasoning` and has no tools and no schema. Qwen3.5 and Qwen3.8 use this path. The path calls `promptCache.restoreIfPending` (2450), `promptCache.plan` (2451-2454) and `withPromptCacheCommit` (2482-2484). No unit test sends a reasoning pass through the executor, thus no unit test proves that a reasoning turn restores a spilled cache or checks a cache in.

What to test:
- A scripted model that declares `.reasoning` and a reasoning config, with a store bound through `ExecutorPromptCacheStore.$current`.
- Turn 2 of a session that stays in memory reuses the prompt of turn 1.
- Turn 2 of a session whose cache went to disk (memory budget 0) restores the file, reuses the prompt of turn 1, and the spilled file is deleted.
- A corrupt spilled file gives a cold reasoning turn that succeeds.

## Acceptance Criteria

- [ ] A test runs one reasoning turn through `Executor.respond` and the coverage of `runReasoning` is more than 0.
- [ ] A test proves that turn 2 in memory has `reusedTokenCount > 0` on the reasoning path.
- [ ] A test proves that turn 2 from disk has `reusedTokenCount > 0` and that no spilled file stays after the turn.
- [ ] A test proves that a corrupt spilled file gives `reusedTokenCount == 0` and a successful turn.
- [ ] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift` (add the reasoning cases here; use `ScriptedModelTestSupport.swift` for the scripted model).

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache