---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3a4c73x99110d5j09xptmxd
  text: |-
    Research:
    - `Array<MLXArray>.totalByteCount` in MLXLMCommon is internal, thus MLXFoundationModels cannot use it. The entry sums `KVCache.residentByteCount` (public) and `LMOutput.State.residentByteCount` (public).
    - `KVCacheSimple` allocates in steps of 256 positions, thus a fed cache of 1..256 tokens holds one step of bytes. Tests compare with `entry.byteCount`, not with fixed byte numbers.
    - `GPU.deviceInfo()` is not deprecated; `Memory.activeMemory` is the current name. `Int(clamping:)` converts the `UInt64` working set with no trap.
    - The scripted model (`ScriptedLanguageModel`) has no KV heads, thus a pass commits nothing (`noCaches`) and checks in nil. The task-local test seeds a bound store, runs `Executor.respond` inside `$current.withValue(store)`, and expects the check-out to empty that store.
    - `MLXLanguageModel.evict()` / `evictAll()` also read the store; a second task-local test proves `evict()` reads `current`.
  timestamp: 2026-09-24T16:35:59.741640+00:00
- actor: claude-code
  id: 01m3a4hcc31tvj8ykxe2p42ebd
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift (byteCount on the entry; byte budget, `configure(memoryBudgetBytes:)`, `retainedByteCount`, `defaultMemoryBudgetBytes(workingSet:active:)`, `@TaskLocal current`, one `spill(_:_:)` for every budget eviction, eviction log line; `maximumRetainedSessions` removed; the shared logger moved to `ExecutorPromptCacheLog`), Libraries/MLXFoundationModels/MLXLanguageModel.swift (4 call sites read `ExecutorPromptCacheStore.current`), Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift (entry helper builds real arrays; 17 new tests). MLXFoundationModelsTests: 244 tests passed.
    - note: `evict(modelID:)` is a release the host asks for, not a budget eviction, thus it does not go through `spill`. An entry larger than the whole budget goes to `spill` at once and evicts no other entry.
    - next: /test
  timestamp: 2026-09-24T16:38:48.963560+00:00
- actor: claude-code
  id: 01m3a4qn304kmgcsvj3xqr6dxr
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — Build complete; one warning `missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS` is also present on HEAD without this change (checked with a stash build), thus it is baseline. `xcrun xctest MLXFoundationModelsTests.xctest` — 244 tests in 29 suites passed. `xcrun xctest MLXLMTests.xctest` — 694 XCTest with 36 failures (the baseline continuation failures: GlmOcr/Qwen25VL/Qwen35/Qwen3VL ContinuationTests and NanbeigeTests.testWarmContinuationMatchesFullPrefill) and 1184 Swift Testing tests with 101 issues, equal to the baseline. No new failure, no new warning.
    - next: /commit
  timestamp: 2026-09-24T16:42:14.496406+00:00
depends_on:
- 01M3A1T3CK0SMNZKTTP375ZMCS
position_column: doing
position_ordinal: '80'
title: Limit the executor prompt cache memory tier by bytes, with a limit the host sets, and remove maximumRetainedSessions
---
#prompt-cache

(Plan item F2.)

## What

`ExecutorPromptCacheStore` (`Libraries/MLXFoundationModels/ExecutorPromptCache.swift:88`) keeps at most `maximumRetainedSessions = 4` sessions (`:99`) and drops the least recently used entry in `checkIn` (`:134-136`). A count of sessions is the wrong unit: one 32k-token session can be larger than twenty short ones. The FoundationModelsRouter host sizes the limit from its pool memory budget and can change it while it runs. Remove the count completely. Do not replace it with another fixed count.

- Add `let byteCount: Int` to `ExecutorPromptCacheEntry`: the sum of `residentByteCount` (task ^375zmcs) over `caches`, plus `state?.residentByteCount ?? 0`. Compute it once in `init`.
- Replace `maximumRetainedSessions` with `private(set) var memoryBudgetBytes: Int`.
- Default budget, when no host sets one: 25% of `max(0, maxRecommendedWorkingSetSize - Memory.activeMemory)`, read once at the first use of the store. `GPU.deviceInfo().maxRecommendedWorkingSetSize` is `UInt64` and `Memory.activeMemory` is `Int`: convert both to `Int` BEFORE the subtraction, else the `UInt64` subtraction traps. Put the computation in a pure `static func defaultMemoryBudgetBytes(workingSet: Int, active: Int) -> Int` that tests call.
- Add `func configure(memoryBudgetBytes: Int)`. It applies the budget at once: it evicts least-recently-used entries until the total is at or below the new budget.
- `checkIn` keeps a running `retainedByteCount`. After the insert, it evicts least-recently-used entries until `retainedByteCount <= memoryBudgetBytes`. One entry larger than the whole budget is not kept in memory.
- Every eviction goes through ONE private function `spill(_ key:, _ entry:)`. In this task it only drops the entry. Task ^w0s77dt replaces its body.
- Log each eviction at `info` level in the `ExecutorPromptCache` category with the key and its bytes.
- Test isolation: the executor uses `ExecutorPromptCacheStore.shared`, and Swift Testing runs suites in parallel. Add an internal task-local `@TaskLocal static var current: ExecutorPromptCacheStore = .shared`, and make every executor use of the store read `ExecutorPromptCacheStore.current` (the call sites in `Libraries/MLXFoundationModels/MLXLanguageModel.swift` at `:245`, `:260`, `:976`, `:1108`). A test binds its own store with `$current.withValue(store) { ... }`.
- `retainedSessionCount` stays for tests. Add `var retainedByteCount: Int`.
- No public API in this task.

## Acceptance Criteria
- [ ] `maximumRetainedSessions` and every other session-count limit are gone from `Libraries/` and `Tests/`.
- [ ] With a budget of N bytes, the store never holds more than N bytes after `checkIn` returns.
- [ ] Many small sessions stay in memory together; a few large sessions push each other out.
- [ ] Eviction order is least recently used first; `checkOut` then `checkIn` makes an entry most recently used.
- [ ] `configure(memoryBudgetBytes:)` to a smaller value evicts at once; to a larger value evicts nothing.
- [ ] `evict(modelID:)` keeps `retainedByteCount` correct.
- [ ] `defaultMemoryBudgetBytes(workingSet:active:)` gives 0, and does not trap, when `active` is larger than `workingSet`.
- [ ] An executor call inside `ExecutorPromptCacheStore.$current.withValue(store)` uses `store` and not `.shared`.

## Tests
- [ ] In `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`, rewrite the test at `:152-163` (it reads `maximumRetainedSessions`) for bytes, and make the `entry(tokens:)` helper build real arrays: it now builds an empty `KVCacheSimple`, which is 0 bytes.
- [ ] Add tests: total within budget; LRU by bytes; one oversize entry; many small entries fit together; budget lowered at run time; `byteCount` includes `LMOutput.State` arrays; `evict(modelID:)` keeps the byte total; the default-budget function; the task-local store. Each test uses its own `ExecutorPromptCacheStore()` instance.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.