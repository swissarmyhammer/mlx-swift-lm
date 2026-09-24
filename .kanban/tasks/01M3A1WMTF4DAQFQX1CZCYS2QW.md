---
assignees:
- claude-code
depends_on:
- 01M3A2BAA6N6SF1647TFZVH9GX
- 01M3A1W91WSM28W94MS2MK47NR
position_column: todo
position_ordinal: '8780'
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