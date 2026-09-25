---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dbcdpamv3k4y6kznxy6xe3
  text: |-
    Research and implementation (iteration 1).

    - No change to production code. No injection seam is necessary. All three paths are reachable from a test.
    - Line 567: a root that does not exist. The test gives a new temporary path that is not made.
    - Lines 574-576: a root that is a regular file. `fileExists` is true, then `contentsOfDirectory` throws (not a folder). The test writes a file at the root path.
    - Lines 497 and 501-505: `init(directory:)` with a non-file URL (`https://example.invalid/prompt-cache-spool`). The loop walks up to `/`, then `resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])` returns a nil value (it does not throw). Line 497 throws `CocoaError(.fileReadUnknown)`, and the catch at 501-505 logs and returns 0. Thus `diskBudgetBytes == 0`.
    - Measured with a scratch script: the https URL gives nil, `/dev` and `/System/Volumes/VM` give 0 (not nil), thus a file URL cannot reach line 497 on this machine.
    - Proof that the tests reach the paths: a temporary mutation of each path (probe once in each early return; return Int.max in the catch) made each of the three new tests fail at its own expectation. The mutation was reverted.
    - The three tests pass at once on the unchanged code, because this is a coverage-gap task and the behavior is correct.
    - File: Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (3 tests, 1 helper `probeCount(ofCleanUpIn:)`, 2 named fixture values).
  timestamp: 2026-09-25T22:36:09.802415+00:00
- actor: claude-code
  id: 01m3dbcj5e8wrsdx57abkcatzk
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (+47 lines, 3 new tests). `swift build --build-tests` clean except the known `missing creator for mutated node` warning. MLXFoundationModelsTests: 315 Swift Testing tests passed.
    - next: /test
  timestamp: 2026-09-25T22:36:14.382559+00:00
- actor: claude-code
  id: 01m3dbkxv9j81ck2w1cb5adp7n
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` clean (only the known `missing creator for mutated node` warning). `xcrun xctest`: MLXLMTests 696 XCTest + 1218 Swift Testing passed; MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 315 passed (312 + 3 new), run 3 times, all passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed. 0 failures, 0 skipped. TextToolCallRecoveryBenchmark passed.
    - next: /commit
  timestamp: 2026-09-25T22:40:15.721716+00:00
- actor: claude-code
  id: 01m3dbpatvtbvt42b3tzhx7h1a
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (b187399) — 0 findings, 0 confirmed, 0 refuted; 7 validators attempted, 0 failed. The two .kanban files had no matching validator.
    - next: done
  timestamp: 2026-09-25T22:41:34.555970+00:00
- actor: claude-code
  id: 01m3dbpd3tyx1gkekwty6wk4x9
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (3 new tests, no production change)
    - test: green — all five bundles pass (MLXLMTests 696 + 1218, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 315 x3, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5), 0 failures, 0 skipped
    - commit: b187399
    - review: clean (0 findings)
  timestamp: 2026-09-25T22:41:36.890788+00:00
position_column: done
position_ordinal: ffb080
title: Test the file system edge paths of the spool clean-up and the default disk budget
---
## What

File: `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`.

- `static ExecutorPromptCacheStore.removeStaleSpoolFolders(in:probe:)`, lines 563-590: 22/26 (84.6%). Uncovered 567 (the root folder does not exist, thus the function returns) and 574-576 (the root cannot be read, thus the function logs and returns).
- `static ExecutorPromptCacheStore.availableCapacity(ofVolumeOf:)`, lines 486-506: 15/21 (71.4%). Uncovered 497 and 501-505 (the free space cannot be read, thus the disk budget is 0).

What to test:
- `removeStaleSpoolFolders(in:)` with a root that does not exist does not throw and calls the probe zero times.
- `removeStaleSpoolFolders(in:)` with a root that is a regular file (not a folder) does not throw, calls the probe zero times, and keeps the file.
- For `availableCapacity`: it is private. Test it through `diskBudgetBytes` of a store made with `init(directory:)`. When no path gives the error, record the reason on this task and leave only the clean-up criteria.

## Acceptance Criteria

- [x] Lines 567 and 574-576 are covered.
- [x] Lines 497 and 501-505 are covered, or this task records why a test cannot reach them without a change to the code under test.
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift`.

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache