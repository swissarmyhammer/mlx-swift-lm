---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3ac1nn2hgxahs5jkw5acpmw
  text: |-
    Research before the change:
    - `ExecutorPromptCacheStore.spill(_:_:)` only logs `evictionLine` now. `checkIn` calls it for an entry larger than the budget, and `evictToBudget` calls it for each LRU entry. `evict(modelID:)` does not use it.
    - `ExecutorPromptCacheFile.prepare` makes new array handles (`$0[.ellipsis]`) and does not evaluate. `ExecutorPromptCacheFile.write` calls `writePromptCache` -> `save(arrays:)`, which evaluates the arrays under `evalLock`. Thus the writer task evaluates, not the actor.
    - `write` does not make the folder. The spool must make the folder before the first write.
    - Package is Swift 6 language mode (tools 6.2).
    - Existing store tests use `ExecutorPromptCacheStore()`. With a spool, all of those stores share one process folder, and file names are `<digest>-<generation>`. Two stores with a per-store generation counter can write the same file name for the same key. Thus the existing test file must give each store its own folder through `init(directory:)`.
    - The test "the least recently used session loses its cache past the byte budget" expects `checkOut == nil` for the evicted key. After this change the evicted key is spilling or on disk, thus that test must read `peek` (memory only).
    - A stale write completion (spill A, then check-out, check-in, spill B) must delete ITS OWN file (A's generation name), else B is not the only file. It never changes `onDisk` and never deletes B's file.
  timestamp: 2026-09-24T18:50:02.786385+00:00
- actor: claude-code
  id: 01m3acgp3epa1gk3erexpw8wr8
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift (spool: `ExecutorPromptCacheCheckout`, `ExecutorPromptCacheSpilledHandle`, `ExecutorPromptCacheSpoolWriter` with one AsyncStream consumer task, `spilling`/`onDisk`/`spillingByteCount`, generation numbers, `init(directory:writer:)`, `waitForSpills()`, `spillLine`), Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift (`removePartialFile` becomes internal `removeFile(at:)`), Libraries/MLXFoundationModels/MLXLanguageModel.swift (`checkOutPromptCache`: `.spilled` starts cold and deletes the file), Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift (call sites; each store has its own folder and a writer that writes nothing; executor test for `.spilled`), Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (new, one test for each criterion).
    - Notes: the folder is made by the default writer `ExecutorPromptCacheStore.writeSpillFile`, not by the consumer, thus a test writer that writes nothing makes no folder. A stale write (its generation is not the current generation of its key) deletes only its own file and changes no record.
    - `swift build --build-tests`: only the known `missing creator for mutated node` warning. MLXFoundationModelsTests: 267 Swift Testing tests in 31 suites pass, 6 runs in a row.
    - next: /test
  timestamp: 2026-09-24T18:58:14.766609+00:00
- actor: claude-code
  id: 01m3ad8xx1bfar2zg86gjwz8yz
  text: |-
    ### test — green (baseline only)
    - evidence: `swift build --build-tests` — only the known `missing creator for mutated node` warning. `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — 267 tests in 31 suites pass (7 runs, no flaky test). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` — 694 XCTest with 36 failures (GlmOcr/Qwen25VL/Qwen35/Qwen3VL ContinuationTests + NanbeigeTests) and 1207 Swift Testing tests with 101 issues: equal to the 2026-09-24 baseline. The change touches only MLXFoundationModels, thus the other three bundles do not change.
    - next: /commit
  timestamp: 2026-09-24T19:11:29.185268+00:00
- actor: claude-code
  id: 01m3adyprvsp231wg089s7mdz3
  text: |-
    ### finish iteration 1 — findings
    - implement: changed (6 files: ExecutorPromptCache.swift, ExecutorPromptCacheFile.swift, MLXLanguageModel.swift, ExecutorPromptCacheTests.swift, new ExecutorPromptCacheSpoolTests.swift, task files)
    - test: green against the baseline (MLXFoundationModelsTests 267/267 pass; MLXLMTests 36 XCTest failures + 101 Swift Testing issues, equal to the baseline; only the known build warning)
    - commit: 7c1aac5 feat(prompt-cache): spill evicted entries to disk with one serial writer
    - review: findings — 2: `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:299` swift/access-control (`directory` must be `private let`); `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift:483` test-integrity/test-partitioning (the executor test with file I/O must move to `ExecutorPromptCacheSpoolTests.swift`)
  timestamp: 2026-09-24T19:23:22.779441+00:00
depends_on:
- 01M3A1QAFD56F5TEPPADDJHENH
- 01M3A1QND68R1PN6K74Z6AV3EP
position_column: review
position_ordinal: '80'
title: 'Disk spool, part 1: spill evicted entries with one serial writer, and hand out a spilled handle on check-out'
---
#prompt-cache

(Plan item F3, store side, part 1. Part 2 — disk budget, folder clean-up, per-session removal — is a separate task.)

## What

In `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`, give `ExecutorPromptCacheStore` a spool. An entry has three states: in memory, spilling, on disk.

- Folder: `FileManager.default.temporaryDirectory/mlx-prompt-cache/<pid>-<process start UUID>/`. An internal `init(directory:)` lets tests use their own folder.
- `spill(_:_:)` (the one eviction path from ^ddjhenh), ON THE ACTOR and synchronously: call `ExecutorPromptCacheFile.prepare(entry, key:)` (task ^z6av3ep). This copies the array handles and metadata, thus the writer holds no reference to the caches. The race it prevents: a check-out during a spill gives the same entry back to a turn, and `KVCacheSimple.update` changes its arrays in place (`self.keys?[...] = keys`), while a writer that reads `cache.state` would race on those Swift properties.
- Give each spill a new generation number (`UInt64`, from a counter on the actor) and write to `fileName(for: key, generation:)`. A completion may change `onDisk` or delete a file ONLY if its generation is still the current generation of that key. This prevents the case spill A → check-out → check-in → spill B, where A's completion would delete B's file.
- ONE serial writer (for example an `AsyncStream` of jobs consumed by one task): at most one write runs at a time, because `save(arrays:)` holds MLX's process-wide `evalLock` for the whole write (`.build/checkouts/mlx-swift/Source/MLX/IO.swift:61-77`). The writer, not the actor, evaluates the arrays (`eval` waits on `evalLock`). Log the write duration in each spill's log line.
- `spilling[key]` keeps the entry, its prepared input, its generation, and its bytes. Spilling bytes count in a separate total `spillingByteCount` (they are still resident until the write ends).
- `checkOut` returns
  ```swift
  enum ExecutorPromptCacheCheckout {
      case memory(ExecutorPromptCacheEntry)
      case spilled(ExecutorPromptCacheSpilledHandle)   // url + key
      case none
  }
  ```
  Order: memory → spilling (take the entry back as `.memory`; the write's completion then adds nothing to `onDisk` and deletes its own file) → on disk (`.spilled`, and remove the key from `onDisk`) → `.none`.
- `checkIn` of a key deletes any file of that key: the entry in memory is newer.
- Update every caller of `checkOut`: the executor in `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (`carriedPromptCache = await ...checkOut(...)`, about `:976`) and the call sites in `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`. Until task ^jar6qq9 lands, the executor treats `.spilled` as a cold start and deletes the file.

## Acceptance Criteria
- [x] An entry evicted by the byte budget is on disk, and a later `checkOut` returns `.spilled`.
- [x] A check-out during a spill of the same key returns `.memory` with the same entry; after the write ends, no `onDisk` record and no file remain for that key.
- [x] A check-out during a spill, followed by a cache `update` on the returned entry, does not change the bytes of the file that the writer writes.
- [x] Spill A (slow writer), check-out, check-in, spill B: after both writes end, B's file exists and is the only file for the key.
- [x] At most one write runs at a time (assert with an injected writer that counts concurrent calls).
- [x] Other check-outs and check-ins complete while a write runs (the actor is free).
- [x] `checkIn` of a key removes that key's older file.
- [x] `swift build --build-tests` compiles with no new warnings.

## Tests
- [x] New `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift`, one test for each criterion, each with its own `ExecutorPromptCacheStore(directory:)` in a temporary folder. For in-flight cases, inject a slow writer (an internal writer closure that tests replace).
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.

## Review Findings (2026-09-24 14:13)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3A1RHPV3CV6Q7Q59W0S77DT.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3A1RHPV3CV6Q7Q59W0S77DT.md` — no validator matches this file

- [ ] `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:299` `swift/access-control` — Internal storage detail should be explicitly declared private, not implicitly internal. The `directory` property is only used within the actor and is not intended as part of the API surface; all other similar properties on this actor explicitly declare their access level. Change line 299 to `private let directory: URL`.
- [ ] `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift:483` `test-integrity/test-partitioning` — The test `anExecutorPassThatFindsItsCacheOnDiskStartsColdAndDeletesTheFile` uses real file I/O to disk (a real external system). According to test-partitioning rules, integration tests that use real external systems belong in a separate integration test target, not the unit test target. The documented convention for this file (lines 37-41) explicitly states: 'These tests read the memory tier alone, and `ExecutorPromptCacheSpoolTests` reads the files.'. Either move this test to `ExecutorPromptCacheSpoolTests.swift` (if it fits the spool testing scope), create a separate integration test target for it, or refactor to test the executor behavior without direct file I/O assertions. The test name suggests it belongs with executor behavior, not file I/O tests — consider if the assertion can be rephrased to test executor behavior rather than file system side effects.