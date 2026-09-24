---
assignees:
- claude-code
depends_on:
- 01M3A1QAFD56F5TEPPADDJHENH
- 01M3A1QND68R1PN6K74Z6AV3EP
position_column: todo
position_ordinal: '8380'
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
- [ ] An entry evicted by the byte budget is on disk, and a later `checkOut` returns `.spilled`.
- [ ] A check-out during a spill of the same key returns `.memory` with the same entry; after the write ends, no `onDisk` record and no file remain for that key.
- [ ] A check-out during a spill, followed by a cache `update` on the returned entry, does not change the bytes of the file that the writer writes.
- [ ] Spill A (slow writer), check-out, check-in, spill B: after both writes end, B's file exists and is the only file for the key.
- [ ] At most one write runs at a time (assert with an injected writer that counts concurrent calls).
- [ ] Other check-outs and check-ins complete while a write runs (the actor is free).
- [ ] `checkIn` of a key removes that key's older file.
- [ ] `swift build --build-tests` compiles with no new warnings.

## Tests
- [ ] New `Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift`, one test for each criterion, each with its own `ExecutorPromptCacheStore(directory:)` in a temporary folder. For in-flight cases, inject a slow writer (an internal writer closure that tests replace).
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.