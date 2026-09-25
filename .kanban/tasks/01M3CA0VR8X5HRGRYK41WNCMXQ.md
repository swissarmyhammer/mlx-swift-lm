---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dcxp4wh88bpbv0g3rn7j5c
  text: |-
    Research done.
    - `withPromptCacheCommit` body throws only from `Task.checkCancellation()` and from `RejectedToolCallError` in `runUnconstrained` and `runReasoning`. A cancel does not reach the catch in a reliable way: the `for await` of an AsyncStream returns nil on cancel, thus the body ends with no throw. The allowed-tool path throws its rejection AFTER the body. Thus the test uses the unconstrained path: a scripted turn writes an unclosed `<tool_call>{...`. The `.json` tool processor rejects it as incomplete output at the end of the stream, and the body throws.
    - `ExecutorPromptCacheSlot.carriesNoCache()` sets `reusedTokenCount = 0` but does NOT clear `carried`. For a session whose cache is in memory, a guided pass thus checks the OLD entry back in at the end of `respond` (`checkIn(key, promptCache.entry)`). The task requires that the store holds nothing for the session after a guided pass. Expect the schema test to fail first; the fix is `carried = .none` in `carriesNoCache()`.
    - A spilled file is deleted by the `defer` in `respond` (checkout.spilledHandle), for every path.
    - Scripted doubles: `ScriptedSessionModel` gives the same script to each pass. The new tests need a different script for each pass, thus `ScriptedSessionModel.make` gets a `scripts:` form, and `ScriptedExecutorPass.respond` gets a form that takes a full request (schema, tools, tool mode).
  timestamp: 2026-09-25T23:03:04.092354+00:00
- actor: claude-code
  id: 01m3ddbqb2695cc1g59my90n7k
  text: |-
    Implementation landed (TDD). Two real bugs were found and fixed.

    Bug 1 (cause): `ExecutorPromptCacheSlot.carriesNoCache()` set `reusedTokenCount = 0` but kept `carried`. A schema pass or a required tool pass over a session whose cache was in memory thus checked the OLD entry back in at the end of `respond`. RED: the schema test failed with `store.retainedByteCount == 0` and `store.checkOut(key) == .none` false. Fix: `carried = .none` in `carriesNoCache()`.

    Bug 2 (cause): `emitRequiredToolCallEvent` called `JSONSerialization.data(withJSONObject: arguments)` with no options. The arguments of a tool whose schema is a scalar (here `String.generationSchema`) are a JSON fragment, and Foundation raises an Objective-C `NSInvalidArgumentException` ("Invalid top-level type in JSON write"). `try?` cannot catch it, and the whole test process stopped (abort trap 6) inside `runRequiredToolTurn`. Fix: `options: [.fragmentsAllowed]`.

    Tests (Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, suite "A session carries its prompt cache between turns"):
    - a turn whose stream body throws checks in nothing, and the next turn starts cold (unclosed `<tool_call>` on the plain path -> RejectedToolCallError inside the body of `withPromptCacheCommit`).
    - a schema pass reports no reuse and leaves no cache of its session in the store.
    - a required tool pass reports no reuse and deletes the spilled file of its session.
    - a required tool pass reports no reuse and leaves no cache of its session in the store.
    Support: `ScriptedSessionModel.make(weights:scripts:...)`, `ScriptedExecutorPass.respond(to:model:inside:)`, `ScriptedPassResult.toolCallNames`.

    Break checks (each reverted, `rg BREAK` is empty): (a) commit in the catch of `withPromptCacheCommit` -> the failed-body test fails (`peek == nil`, `next.reusedTokenCount == 0`); (b) remove `carriesNoCache()` from `runSchemaGeneration` -> the schema test fails; (c) remove `carriesNoCache()` from `runRequiredToolTurn` -> the required memory test fails. The spilled required test crashed inside `runRequiredToolTurn` before bug 2 was fixed, which proves it reaches that path.

    Note: the scripted model does not replay the arguments of a guided tool call exactly (xgrammar fast-forward feeds several tokens in one forward pass), thus the test checks the tool name only.

    MLXFoundationModelsTests: 323 passed, 0 failed.

    ### implement — changed
    - evidence: 4 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, Tests/MLXFoundationModelsTests/ScriptedModelTestSupport.swift
    - next: /test
  timestamp: 2026-09-25T23:10:44.066639+00:00
- actor: claude-code
  id: 01m3ddm1e7p570q2xgggz9xvnt
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` exit 0, one known warning `missing creator for mutated node` only. `xcrun xctest`: MLXLMTests 696 XCTest + 1218 Swift Testing passed; MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 323 passed (3 runs, all green); CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed. 0 failed, 0 skipped.
    - next: /commit
  timestamp: 2026-09-25T23:15:16.551766+00:00
- actor: claude-code
  id: 01m3ddyszc4s0har96qtabxkwg
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` — 0 findings, 0 confirmed, 0 refuted, 7 attempted, 0 failed (only the two .kanban files not reviewed: no validator matches them).
    - next: done

    ### finish iteration 1 — clean
    - implement: changed — 4 files (ExecutorPromptCache.swift, MLXLanguageModel.swift, ExecutorPromptCacheTests.swift, ScriptedModelTestSupport.swift); 2 library bugs fixed with TDD
    - test: green — MLXLMTests 696 + 1218, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 323 (3 runs), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5; 0 failed, 0 skipped; only the known `missing creator for mutated node` warning
    - commit: d841f85
    - review: clean (0 findings)
  timestamp: 2026-09-25T23:21:09.356800+00:00
position_column: done
position_ordinal: ffb380
title: Test that a failed stream body and a guided pass check in no prompt cache
---
## What

File: `Libraries/MLXFoundationModels/MLXLanguageModel.swift`.

1. `MLXLanguageModel.Executor.withPromptCacheCommit(task:plan:state:promptCache:body:)`, lines 2012-2026. Coverage: 77.8% (7/9 lines). Uncovered: 2022-2023. When `body` throws, the function cancels and drains the task and does not call `promptCache.commit`. No test proves that the session then starts cold on its next turn.
2. `runSchemaGeneration(...)`, lines 2220-2311. Coverage: 0% (0/92). It calls `promptCache.carriesNoCache()` at line 2224.
3. `runRequiredToolTurn(...)`, lines 1644-1811. Coverage: 0% (0/168). It calls `promptCache.carriesNoCache()` at line 1645.
4. `runRespond` lines 1149-1157 (the schema branch) are uncovered.

What to test (prompt cache behavior only):
- A turn whose stream consumer throws (for example a cancellation or a rejected tool call on the allowed-tool path) checks in nothing. The next turn of the session has `reusedTokenCount == 0`.
- A guided pass (schema, and required tool mode) through `Executor.respond` reports no reuse, checks in nothing, and deletes a spilled file of its session.

## Acceptance Criteria

- [x] A test makes `body` throw inside `withPromptCacheCommit`, and lines 2022-2023 are covered.
- [x] That test proves the next turn of the session starts cold.
- [x] A test runs a schema pass through `Executor.respond` with a checked-in cache, and proves the pass reports `cachedTokenCount == 0` and that the store holds nothing for the session after it.
- [x] A test runs a required-tool pass with a spilled cache, and proves that the spilled file is deleted.
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`.

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache