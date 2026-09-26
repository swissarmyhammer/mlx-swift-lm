---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dj9kctjhserytmbb07mtw3
  text: |-
    Research: the lines moved. Current places in Libraries/MLXLMCommon/KVCache.swift: `ArraysCache.restoreFromMetaState` 1604-1622 (legacy branch 1618-1621), `preparePromptCacheSave` 2236-2293 (model state guard 2243-2245, merge closure 2273-2275), `PromptCacheFileContents.init(url:)` 2450-2484 (part-count guard 2457-2459, cache-count guard 2473-2475), `loadPromptCacheState` 2799-2840 (count guard 2809-2813, extra metadata guard 2831-2833), `validateRotatingCache` 3056-3079 (wrapped flag 3073-3078), `validateVarianceNormalizedCache` 3087-3112 (tile count 0 branch 3093-3094), `unflattenArrays` 3477-3513 (key guard 3486-3494, contiguous guard 3503-3508).

    Discoveries:
    - The part-count guard 2457-2459 ("Invalid cache metadata format") cannot fail. `unflattenMetadata` always gives back a literal array of three parts. Thus no file reaches it. It is dead code; this task removes it (and its constant `metadataPartCount`) so that no uncovered line stays. The behavior does not change.
    - A file without the `2.*` class names still has its offset records `1.__mlx_lm_offset_<i>`. With a class count of 0, `PromptCacheOffsetRecord.read` rejects those records first. Thus the test of removed class names also removes the offset records to reach "Mismatch in cache counts".
    - Removing `0.0.*` of a two-layer file does not shorten the cache info: `unflattenMetadata` fills index 0 with []. Remove `0.1.*` (the last layer) to reach "Mismatch in cache counts".
    - `validateArraysCache` accepts the legacy meta state [""], thus a tampered MambaCache file reaches the legacy branch of `restoreFromMetaState` through `loadPromptCacheSnapshot(url:)`.
    - The merge closure 2273-2275 is not reachable: `validateUserMetadata` rejects the reserved offset prefix first. Not tested, as the task says.
  timestamp: 2026-09-26T00:36:57.370398+00:00
- actor: claude-code
  id: 01m3djh5w11ywytv2ht98faqvk
  text: |-
    Implementation landed.
    - Library: removed the dead part-count guard ("Invalid cache metadata format") and its constant `metadataPartCount` from `PromptCacheFileContents.init(url:)`. `unflattenMetadata` always gives back three parts, thus the guard could not fail. No behavior change.
    - Tests (PromptCacheTemplateRestoreTests): cacheCountMismatchThrows (class names removed, or the second layer info removed), invalidModelStateCountThrows (nil, "0", "not-a-count"), unknownModelStateKeyThrows, invalidArrayKeyThrows ("x.0", "9.0"), nonContiguousArrayIndicesThrow, invalidRotatingWrappedFlagThrows, varianceNormalizedTileArraysWithoutTilesThrow, legacyArraysMetaStateRestoresSlotsInOrder. Each rejection test checks the exact message. `tamperedFile` takes an optional model state.
    - Test (PromptCacheSaveInputTests): prepareRefusesAModelStateWithNoCache checks "Model state requires at least one prompt cache".
    - Break checks: each path was broken for a short time (message changed, or condition forced), and each new test failed. The wrapped-flag break with the check removed stops the process in the `RotatingKVCache` setter (fatal error), thus the check is necessary. All breaks were removed; the library file is the saved copy.
    - No library bug found.

    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/KVCache.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, Tests/MLXLMTests/PromptCacheSaveInputTests.swift
    - next: test
  timestamp: 2026-09-26T00:41:05.665893+00:00
- actor: claude-code
  id: 01m3dk26dtgrffdfsmw3dynf9d
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` exit 0, one known warning (`missing creator for mutated node`). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` 3 runs: each 696 XCTest 0 failures + 1238 Swift Testing passed (1229 + 9 new). MLXGuidedGenerationTests 70 passed, MLXFoundationModelsTests 329 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 5 passed. 0 skipped. TextToolCallRecoveryBenchmark passed.
    - next: commit
  timestamp: 2026-09-26T00:50:23.290070+00:00
- actor: claude-code
  id: 01m3dk74r8zmd5tyc77e7j22n7
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` — 0 findings, 0 confirmed, 0 refuted; 7 validators attempted, 0 failed. No prior Review Findings section.
    - next: done
  timestamp: 2026-09-26T00:53:05.416816+00:00
- actor: claude-code
  id: 01m3dk77219tzv3rk4j1rackjf
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Libraries/MLXLMCommon/KVCache.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, Tests/MLXLMTests/PromptCacheSaveInputTests.swift
    - test: green — MLXLMTests 3 runs, each 696 XCTest + 1238 Swift Testing, 0 failures; the four other bundles pass (70, 329, 7, 5); 0 skipped; one known build warning
    - commit: 1e4ccbd
    - review: clean — review sha HEAD~1..HEAD, 0 findings
  timestamp: 2026-09-26T00:53:07.777684+00:00
position_column: done
position_ordinal: ffb780
title: Test the prompt cache loader rejections of damaged files and the legacy ArraysCache meta state
---
## What

File: `Libraries/MLXLMCommon/KVCache.swift`.

- `PromptCacheFileContents.init(url:)`, lines 2420-2454: 33/35. Uncovered 2428 (fewer than 3 metadata parts) and 2444 (the cache info count is not the class count).
- `loadPromptCacheState(arrays:metadata:)`, lines 2738-2779: 38/40. Uncovered 2751 (the state count is missing, not a number, or 0) and 2771 (a reserved state metadata key that the reader does not know).
- `unflattenArrays(_:cacheCount:)`, lines 3414-3450: 28/34. Uncovered 3429-3430 (an array key that is not `i.j`, or an index out of range) and 3441-3444 (the array indices of one cache are not contiguous).
- `validateRotatingCache(state:metaState:)`, lines 2993-3016: 20/24. Uncovered 3011-3014 (the wrapped flag is not `true` or `false`).
- `validateVarianceNormalizedCache(state:metaState:)`, lines 3024-3049: 25/26. Uncovered 3031 (tile count 0 with tile arrays present).
- `preparePromptCacheSave(cache:metadata:state:)`, lines 2210-2267: 51/54. Uncovered 2218 (a model state with no cache throws). Lines 2248-2249 are the merge closure for a key collision; `validateUserMetadata` rejects the reserved prefix first, thus no input reaches them. Record this and do not test them.
- `ArraysCache.restoreFromMetaState(state:savedMetaState:)`, lines 1595-1613: 16/19. Uncovered 1610-1612: the legacy meta state (no slot count) restores the state arrays in order.

What to test (use `tamperedFile(_:edit:)`):
- Remove the `2.*` class metadata: throws (2428 or 2444).
- Remove one `0.<i>.*` cache info entry of a two-layer file: throws "Mismatch in cache counts".
- Set the state count to `0` or `x`: throws (2751). Add one extra reserved state key: throws (2771).
- Rename an array key to `x.0` or `9.0`: throws (3429-3430). Remove the array `0.0` of a cache that has `0.1`: throws (3441-3444).
- Set the wrapped flag of a saved `RotatingKVCache` to `maybe`: throws (3011-3014).
- Set the tile count of a saved `VarianceNormalizedKVCache` to 0 while it has tile arrays: throws (3031).
- `preparePromptCacheSave(cache: [], state: <state with one array>)` throws (2218).
- A `MambaCache` whose meta state is `[""]` restores its state arrays into slots 0 and 1 (1610-1612).

Note: line 2428 (the part-count guard, "Invalid cache metadata format") could not fail, because `unflattenMetadata` always gives back three parts. The change removes that dead guard, thus no uncovered line stays there. Lines 2248-2249 (the merge closure) are not reachable, as recorded above.

## Acceptance Criteria

- [x] Each uncovered line above is covered, except 2248-2249, which the task records as not reachable.
- [x] Each rejection test checks `KVCacheError`.
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift` (loader rejections) and `Tests/MLXLMTests/PromptCacheSaveInputTests.swift` (line 2218).

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache