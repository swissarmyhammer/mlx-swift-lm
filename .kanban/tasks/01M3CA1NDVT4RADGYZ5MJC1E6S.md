---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dcbwp3dmyewqbe0se8sc8f
  text: |-
    Research: `reconcilePromptCache` gives kind `.prefill` only when `RewindToCommonPrefixRule` returns `.prefillAll`, thus when the first cache is at offset 0. `ExtendCachedPrefixRule` refuses an empty ledger. Thus an entry with one `KVCacheSimple()` at offset 0 and an empty ledger reaches lines 1049-1052. No library bug found.

    New test `aCarriedEntryWhoseCachesHoldNothingPlansAsCold` in `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift` checks the decision, the reused count and the full plan line. Break check: `.prefill` changed for a short time to return `.extend`; the test failed with 2 issues (decision and plan line). The break is removed; the library file has no change.

    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift; MLXFoundationModelsTests 319 tests passed
    - next: test
  timestamp: 2026-09-25T22:53:20.963917+00:00
position_column: doing
position_ordinal: '80'
title: Test that a carried cache that holds nothing plans as cold (decision .prefill)
---
## What

File: `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`, function `static ExecutorPromptCachePlan.decision(of:render:ledger:)`, lines 1045-1060. Coverage: 78.6% (11/14 lines). Uncovered 1050-1052: the `.prefill` case of `PromptCacheReuse.kind`, which gives `.cold`.

This case occurs when a session carries an entry, but the reuse rules give a prefill from position 0 (the caches hold nothing that the render can use). The plan line must then name `rule=cold`, and `reusedTokenCount` must be 0, although an entry was carried.

What to test:
- Make an entry whose caches are at offset 0 with an empty ledger, or a ledger that the reconcile answers with `.prefill`. Call `ExecutorPromptCachePlan.make(reusing:input:model:parameters:)`.
- Expect `plan.decision == .cold`, `plan.reusedTokenCount == 0`, and that the plan line ends with `rule=cold`.

## Acceptance Criteria

- [x] Lines 1050-1052 are covered.
- [x] The test checks the decision, the reused count and the plan line text.
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