---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3af11sg8gzbh2jwnkwfmdwf
  text: |-
    Research (implement, iteration 1):
    - The store already has a private `remove(_:) -> ExecutorPromptCacheEntry?` (memory tier only). The new internal `remove(_ key:)` has the same name, thus the private function gets the name `removeEntry(_:)`.
    - `checkIn` already removes memory, spill and disk state of its key at its start. It will call the new `remove(_:)`.
    - `onDisk` records only a handle. The disk budget needs the byte count of each file and an LRU order, thus each disk record gets the file size (read with `URLResourceValues.fileSize` when the write ends) and the store keeps `diskUsageOrder` and `diskByteCount`.
    - `ExecutorPromptCacheTests.makeStore()` uses a writer that writes no file. A disk record for such a write has a size that cannot be read; the store logs that and records zero bytes.
    - The stale-folder clean-up must not run for test stores (their folder is directly in the temporary directory). Thus only the default `init()` gives a root (`<tmp>/mlx-prompt-cache/`) to clean. The clean-up function takes a process probe, thus tests cover ESRCH, EPERM and kill == 0 without real processes.
  timestamp: 2026-09-24T19:42:08.176143+00:00
- actor: claude-code
  id: 01m3af8bkh2gxfe49q2kz274ta
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift (disk budget with LRU disk order and known disk total, `configure(diskBudgetBytes:)`, `defaultDiskBudgetBytes(availableCapacity:)`, `isStaleProcess(killResult:errorNumber:)`, `removeStaleSpoolFolders(in:probe:)` at first use of the default store, `remove(_:)`, `evict(sessionID:)`, `evict(modelID:)` over all three tiers, `diskEvictionLine`), Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (13 new tests). Red first (compile errors), then green: MLXFoundationModelsTests 279 tests passed.
    - note: the private memory-tier `remove(_:) -> Entry?` is now `removeEntry(_:)`.
    - next: /test
  timestamp: 2026-09-24T19:46:07.601545+00:00
- actor: claude-code
  id: 01m3affa3mkw6qgag3n1bmpnbq
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. MLXFoundationModelsTests 3 runs: 279/279 passed each time. MLXGuidedGenerationTests 70 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 5 passed. MLXLMTests: 694 XCTest with 36 failures (GlmOcr 6, Qwen25VL 13, Qwen35 8, Qwen3VL 8 ContinuationTests, NanbeigeTests 1) and 1207 Swift Testing with 101 issues, which is the baseline of 2026-09-24. No new failure, no new warning.
    - next: /commit
  timestamp: 2026-09-24T19:49:55.444461+00:00
- actor: claude-code
  id: 01m3afrasj54ndevnfhkgs6757
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 2 findings (2 confirmed, 2 refuted) — Libraries/MLXFoundationModels/ExecutorPromptCache.swift:584 (code-hygiene/magic-numbers-swift), Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift:528 (code-hygiene/data-driven)
    - next: /implement fzvh9gx

    ### finish iteration 1 — findings
    - implement: changed (ExecutorPromptCache.swift, ExecutorPromptCacheSpoolTests.swift)
    - test: green (MLXFoundationModelsTests 279/279 x3; MLXLMTests at baseline 36 XCTest failures, 101 issues; other 3 bundles pass)
    - commit: e71bbe1
    - review: findings — `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:584` Magic numbers should be replaced by named constants.; `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift:528` The switch statement over pid is a table written as control flow.
  timestamp: 2026-09-24T19:54:51.058546+00:00
- actor: claude-code
  id: 01m3ag0a3t6q6dk8d1spmtwp4e
  text: |-
    ### implement — changed
    - evidence: 2 files. ExecutorPromptCache.swift: the literal 2 in `processIdentifier(ofSpoolFolder:)` is now `spoolFolderNamePartCount`; the `-` separator is now `spoolFolderNameSeparator`, used where the folder name is written and where it is read. ExecutorPromptCacheSpoolTests.swift: the probe `switch` is now the table `probeResults` with the default `liveProbeResult`.

    ### test — green
    - evidence: build has only the known `missing creator for mutated node` warning. MLXFoundationModelsTests 279/279 passed 3 times. MLXLMTests at baseline (36 XCTest failures in the same 5 suites, 101 Swift Testing issues). MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 passed.
    - next: /commit
  timestamp: 2026-09-24T19:59:12.506105+00:00
depends_on:
- 01M3A1RHPV3CV6Q7Q59W0S77DT
position_column: review
position_ordinal: '80'
title: 'Disk spool, part 2: disk budget, clean-up of dead-process folders, and removal of one key or one session'
---
#prompt-cache

(Plan item F3, store side, part 2, and the store half of F5.)

## What

In `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`, on top of the spool of ^w0s77dt:

- Disk budget: `diskBudgetBytes` with an internal `configure(diskBudgetBytes:)`. After each change of `onDisk`, delete least-recently-used files until the known total is at or below the budget. Default: 25% of the free space of the volume of the spool folder, read at the first use (`URLResourceValues.volumeAvailableCapacityForImportantUsage`), computed in a pure static function that tests call.
- Clean-up at the first use: delete each folder under `mlx-prompt-cache/` whose process does not run. A folder is stale ONLY when `kill(pid, 0) == -1 && errno == ESRCH`. `EPERM` means a live process that this process cannot signal: keep that folder. Put the check in a function that takes the `kill` result and `errno`, thus a test covers each case.
- `func remove(_ key: ExecutorPromptCacheKey)` — removes that one key from memory, from a running spill (its completion then adds nothing and deletes its own file, by the generation rule of ^w0s77dt), and from disk. A no-op for an unknown key.
- `func evict(sessionID: String)` — `remove` for every key of that session, for all models.
- `evict(modelID:)` also removes spilling entries and files of that model; `evict(modelID: nil)` removes all.

## Acceptance Criteria
- [x] The disk budget deletes files, least recently used first, and the known disk total stays at or below the budget.
- [x] The default-disk-budget function gives 25% of the value it receives.
- [x] Stale-folder check: `ESRCH` → stale (deleted); `EPERM` → live (kept); `kill` returns 0 → live (kept).
- [x] `remove(key)` of an entry in memory, of an entry being spilled (slow writer), and of an entry on disk leaves no memory entry, no `onDisk` record, and no file for that key after all writes end.
- [x] `remove(key)` of an unknown key does nothing and does not throw.
- [x] `evict(sessionID:)` removes that session for every model and leaves other sessions.
- [x] `evict(modelID:)` removes memory, spilling and disk state of one model and leaves other models.

## Tests
- [x] Extend `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift` (from ^w0s77dt) with one test for each criterion, each with its own `ExecutorPromptCacheStore(directory:)` in a temporary folder.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.

## Review Findings (2026-09-24 14:50)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 2 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3A2BAA6N6SF1647TFZVH9GX.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3A2BAA6N6SF1647TFZVH9GX.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:584` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift:528` `code-hygiene/data-driven` — The switch statement over pid is a table written as control flow. Each arm differs only in the constants returned. This should be expressed as data rather than as parallel code paths. Extract the pid-to-result mapping into a dictionary and use dictionary lookup with a default: Create a static dictionary `let probeResults: [pid_t: ProcessProbeResult] = [Self.stalePID: (killResult: -1, errorNumber: ESRCH), Self.unsignalablePID: (killResult: -1, errorNumber: EPERM)]` and replace the switch with `probeResults[pid] ?? (killResult: 0, errorNumber: 0)`.