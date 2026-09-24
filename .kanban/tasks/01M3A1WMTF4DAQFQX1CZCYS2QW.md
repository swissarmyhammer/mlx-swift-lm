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
- actor: claude-code
  id: 01m3am1n8c41adbjywysq89azw
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 1 finding: Tests/MLXFoundationModelsTests/PromptCacheSpoolTestSupport.swift:16 `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
    - next: /implement to remove the cause from the file.

    ### finish iteration 1 — findings
    - implement: changed (5 files; 7 new tests)
    - test: green (swift build --build-tests: only the known `missing creator` warning; MLXFoundationModelsTests 291/291 passed on 4 runs)
    - commit: 19f65cd feat(prompt-cache): add public API to set the prompt cache budgets, release one session and read the usage
    - review: findings (1) — PromptCacheSpoolTestSupport.swift:16 code-hygiene/magic-numbers-swift
  timestamp: 2026-09-24T21:09:50.988255+00:00
- actor: claude-code
  id: 01m3am8m9222kr13q85z3g1xmw
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD — 0 findings, 0 confirmed, 0 refuted; the one earlier finding is checked.

    ### finish iteration 2 — clean
    - implement: changed (Tests/MLXFoundationModelsTests/PromptCacheSpoolTestSupport.swift — named tokenCount, batchSize, and a named product; no literal ledger)
    - test: green (swift build --build-tests: only the known `missing creator` warning; MLXFoundationModelsTests 291/291 passed on 3 runs)
    - commit: 6a30c0c test(prompt-cache): name the numbers of the spool fixture shape
    - review: clean — task moved to done
  timestamp: 2026-09-24T21:13:39.362201+00:00
depends_on:
- 01M3A2BAA6N6SF1647TFZVH9GX
- 01M3A1W91WSM28W94MS2MK47NR
position_column: done
position_ordinal: ff9d80
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
- [x] `configurePromptCache(memoryBudgetBytes:)` changes the budget of the current store and evicts at once when the new budget is smaller.
- [x] `configurePromptCache(diskBudgetBytes:)` changes the disk budget and deletes files at once when the new budget is smaller.
- [x] After `releasePromptCache(sessionID:)`, the store holds nothing for that key in memory, in the spill, or on disk, and the file is gone; other sessions and other models are unchanged.
- [x] A release of an unknown session is a no-op and does not throw.
- [x] `promptCacheUsage` reports the store's three byte totals.

## Tests
- [x] New `Tests/MLXFoundationModelsTests/PromptCachePublicAPITests.swift`. Each test binds its own store with `ExecutorPromptCacheStore.$current.withValue(store)` (never the shared store): release of an entry in memory, on disk, and during a spill (slow writer); an unknown session; each budget change; the usage totals.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.

## Review Findings (2026-09-24 16:05)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3A1WMTF4DAQFQX1CZCYS2QW.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3A1WMTF4DAQFQX1CZCYS2QW.md` — no validator matches this file

- [x] `Tests/MLXFoundationModelsTests/PromptCacheSpoolTestSupport.swift:16` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.