---
assignees:
- claude-code
depends_on:
- 01M3A1RHPV3CV6Q7Q59W0S77DT
position_column: todo
position_ordinal: 8a80
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
- [ ] The disk budget deletes files, least recently used first, and the known disk total stays at or below the budget.
- [ ] The default-disk-budget function gives 25% of the value it receives.
- [ ] Stale-folder check: `ESRCH` → stale (deleted); `EPERM` → live (kept); `kill` returns 0 → live (kept).
- [ ] `remove(key)` of an entry in memory, of an entry being spilled (slow writer), and of an entry on disk leaves no memory entry, no `onDisk` record, and no file for that key after all writes end.
- [ ] `remove(key)` of an unknown key does nothing and does not throw.
- [ ] `evict(sessionID:)` removes that session for every model and leaves other sessions.
- [ ] `evict(modelID:)` removes memory, spilling and disk state of one model and leaves other models.

## Tests
- [ ] Extend `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift` (from ^w0s77dt) with one test for each criterion, each with its own `ExecutorPromptCacheStore(directory:)` in a temporary folder.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.