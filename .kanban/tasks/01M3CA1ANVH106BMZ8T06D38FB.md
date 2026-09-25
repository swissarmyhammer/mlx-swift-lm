---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dbrhz99ek90yd2n87j21kz
  text: 'Research: `spill(_:_:)` catches the error of `ExecutorPromptCacheFile.prepare`, logs a failed spill line and returns before it changes a counter or enqueues a write. `prepare` throws `tokenOutOfRange` for a ledger token that is not in Int32 (`TokenLedger.encode`). `writeDidEnd` gives `.failed(error)` for each failed write and records no file; `finishPendingWrite` runs in `defer`, thus `waitForSpills()` returns. The store already takes an injected writer (`ExecutorPromptCacheFileWriter`), thus no production seam is necessary. `PromptCacheSpoolFixtures.expectNothingStored` checks the three byte totals, the spool folder and the check-out.'
  timestamp: 2026-09-25T22:42:47.401731+00:00
- actor: claude-code
  id: 01m3dbw0jwryfethsg67n8fb7e
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift. Three new tests in "Failed spills": prepare fails (ledger token Int32.max + 1), writer throws (`FailingWriter` counts calls and throws `SpillWriteFailure`), and the spill line of `.failed(error)`. No production change: the injected writer was sufficient, and no library bug was found.
    - break checks: (1) keep the entry in memory in the catch of `spill` → the prepare test failed with 2 issues; (2) `recordDiskFile(handle)` in the failure case of `writeDidEnd` → the write test failed; (3) outcome text "failed" without the error → the spill-line test failed. Each break was removed before the commit (`git diff` shows only the test file).
    - MLXFoundationModelsTests: 318 Swift Testing tests (315 + 3), 0 failures.
    - next: /test
  timestamp: 2026-09-25T22:44:40.668280+00:00
- actor: claude-code
  id: 01m3dc3wxa2t7n7phwv58c7qwk
  text: |-
    ### test — green
    - evidence: swift build --build-tests (only the known warning `missing creator for mutated node`); MLXLMTests 696 XCTest + 1218 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 318 (3 runs, all pass), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5; 0 failures, 0 skipped.
    ### commit — changed
    - evidence: db43a28 test(prompt-cache): cover the failed spill outcomes when prepare fails and when the write fails
    - next: /review
  timestamp: 2026-09-25T22:48:59.050490+00:00
- actor: claude-code
  id: 01m3dc805spxfxp01jq8yb5bsb
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD — 0 findings (7 validator runs attempted, 0 failed); the two kanban files have no matching validator. No prior Review Findings section.
    - next: none; task moved to done.

    ### finish iteration 1 — clean
    - implement: changed — Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (3 new tests; each break check failed as expected, then the breaks were removed)
    - test: green — MLXLMTests 696 + 1218, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 318 (3 runs), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5; 0 failures, 0 skipped; only the known build warning
    - commit: db43a28
    - review: clean (0 findings)
  timestamp: 2026-09-25T22:51:13.465489+00:00
position_column: done
position_ordinal: ffb180
title: Test the failed spill outcomes of ExecutorPromptCacheStore (prepare fails, write fails)
---
## What

File: `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`. File coverage: 96.83% (610/630 lines).

- `ExecutorPromptCacheStore.spill(_:_:)`, lines 831-860: 25/30. Uncovered 838-842. When `ExecutorPromptCacheFile.prepare` throws, the store logs a `failed` spill line and drops the entry. The next check-out of that key must give `.none`, and no write goes into the queue.
- `ExecutorPromptCacheStore.writeDidEnd(...)`, lines 892-912: 20/21. Uncovered 907. When the writer throws, the outcome is `.failed(error)`. The store must record no disk file, `spillingByteCount` must go back to 0, and `waitForSpills()` must return.

What to test:
- Check in an entry whose ledger holds a token larger than `Int32.max` into a store with memory budget 0. Expect `checkOut(key) == .none`, `spillingByteCount == 0`, `diskByteCount == 0`, and no file in the spool folder.
- Make a store with a writer that throws. Check in an entry with memory budget 0. After `waitForSpills()`, expect `checkOut(key) == .none`, all three byte totals 0, and no file.
- Check the spill line text of a failed write with `ExecutorPromptCacheReport.spillLine(... outcome: .failed(error))` ends with `result=failed: <error>`.

## Acceptance Criteria

- [x] Lines 838-842 and 907 are covered.
- [x] Each test checks the three byte totals and the check-out result.
- [x] `waitForSpills()` returns in the failed-write test (no hang).
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift` (use `PromptCacheSpoolTestSupport.swift` for the store and the entry).

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache