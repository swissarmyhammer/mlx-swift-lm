---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3akekt8n4tcfvx67a5cd62c
  text: |-
    Research:
    - `ExecutorPromptCacheStore` (ExecutorPromptCache.swift:251) has `configure(memoryBudgetBytes:)` (:508), `configure(diskBudgetBytes:)` (:519), `remove(_:)` (:704), and the byte totals `retainedByteCount`, `spillingByteCount`, `diskByteCount` (private(set)). `current` is a task-local (:291).
    - Defaults: memory = one quarter of (recommended working set - active memory) at first use; disk = one quarter of the free space of the spool volume at first use (`defaultBudgetDivisor = 4`).
    - `evict()`/`evictAll()` are at MLXLanguageModel.swift:243/:258.
    - ExecutorPromptCacheSpoolTests.swift has private fixtures (`entry()`, `HeldWriter`, `fileNames(in:)`, `expectNothingStored`). The new tests need the same fixtures, thus I move them to a shared test support file and do not copy them.
    - The store has no single read of the three totals. I add `usage` on the store so that `promptCacheUsage` reads the three totals in one actor hop (one consistent snapshot).
  timestamp: 2026-09-24T20:59:26.920575+00:00
- actor: claude-code
  id: 01m3akq5ng0jrfgjpbs3j65c3t
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXFoundationModels/MLXLanguageModel.swift (4 public members), Libraries/MLXFoundationModels/ExecutorPromptCache.swift (`usage` snapshot), Tests/MLXFoundationModelsTests/PromptCachePublicAPITests.swift (7 new tests), Tests/MLXFoundationModelsTests/PromptCacheSpoolTestSupport.swift (shared fixtures: protocol `PromptCacheSpoolFixtures` + `HeldWriter`), Tests/MLXFoundationModelsTests/ExecutorPromptCacheSpoolTests.swift (uses the shared fixtures, no copy).
    - TDD: the new tests did not compile before the API existed (red). After: `swift build --build-tests` has only the known `missing creator` warning; MLXFoundationModelsTests 291/291 passed on 4 runs.
    - Note: the files tool `replace_all` made only one replacement per call in this session. Check the result count after each replace_all.
    - next: /test
  timestamp: 2026-09-24T21:04:07.344508+00:00
depends_on:
- 01M3A2BAA6N6SF1647TFZVH9GX
- 01M3A1W91WSM28W94MS2MK47NR
position_column: doing
position_ordinal: '80'
title: 'Public API: set the prompt cache budgets, and release the cache of one session'
---
#prompt-cache

(Plan items F2, the public part, and F5.)

## What

In `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, next to `evictAll()` (`:243`) and `evict()` (`:258`), add:

```swift
/// Sets the memory budget of the shared prompt cache, in bytes. Entries
/// past the budget go to the disk spool, least recently used first.
public static func configurePromptCache(memoryBudgetBytes: Int) async
/// Sets the disk budget of the spool, in bytes.
public static func configurePromptCache(diskBudgetBytes: Int) async
/// Releases the cache of one session of this model: in memory, while it
/// spills, and on disk. A no-op for an unknown session.
public func releasePromptCache(sessionID: String) async
/// The bytes the prompt cache holds now, for a host that sizes its pool.
/// `spillingBytes` are still resident in memory until their write ends.
public static var promptCacheUsage: (memoryBytes: Int, spillingBytes: Int, diskBytes: Int) { get async }
```

- These act on `ExecutorPromptCacheStore.current` (the shared store unless a test binds its own, from ^ddjhenh).
- `releasePromptCache(sessionID:)` calls the store's `remove(ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID))` (task ^fzvh9gx).
- Doc comment of `sessionID`: it is the identifier the host binds with `MLXLanguageModel.promptCacheScope = .session(id)` (task ^2mk47nr), or, when the host binds nothing, the id of the first transcript entry.
- Keep `evict()`/`evictAll()` as they are.
- Document the default budgets (from ^ddjhenh and ^fzvh9gx) on the functions.

## Acceptance Criteria
- [ ] `configurePromptCache(memoryBudgetBytes:)` changes the budget of the current store and evicts at once when the new budget is smaller.
- [ ] `configurePromptCache(diskBudgetBytes:)` changes the disk budget and deletes files at once when the new budget is smaller.
- [ ] After `releasePromptCache(sessionID:)`, the store holds nothing for that key in memory, in the spill, or on disk, and the file is gone; other sessions and other models are unchanged.
- [ ] A release of an unknown session is a no-op and does not throw.
- [ ] `promptCacheUsage` reports the store's three byte totals.

## Tests
- [ ] New `Tests/MLXFoundationModelsTests/PromptCachePublicAPITests.swift`. Each test binds its own store with `ExecutorPromptCacheStore.$current.withValue(store)` (never the shared store): release of an entry in memory, on disk, and during a spill (slow writer); an unknown session; each budget change; the usage totals.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.