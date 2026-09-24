---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3ak45qmndms9jxwvk34yva4
  text: |-
    Research: the case `.none` has references only in `MLXLanguageModel+PromptCacheScope.swift` (declaration and doc), `MLXLanguageModel.swift` (`sessionCacheKey` switch and its doc) and `PromptCacheScopeTests.swift`. No docs outside Libraries/Tests mention the scope. Each check-in makes a new `ExecutorPromptCacheEntry` (`commitOutcome`), thus a test can use `!==` to prove that a pass checked its cache in.

    TDD: the tests used `.uncached` first; the build failed with "type 'MLXLanguageModel.PromptCacheScope?' has no member 'uncached'" (red). Then the rename made the build pass.

    The new test binds `nil` inside an outer `withValue(.uncached)`. This proves that an explicit `withValue(nil)` gives the first-entry-id rule, and not the outer uncached scope. The earlier test "with nothing bound" already passed nil through the helper, but only at the top level.
  timestamp: 2026-09-24T20:53:44.820142+00:00
- actor: claude-code
  id: 01m3ak5s8pbqssh10mz5dg1ttg
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXFoundationModels/MLXLanguageModel+PromptCacheScope.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXFoundationModelsTests/PromptCacheScopeTests.swift. `swift build --build-tests` passes; only the known warning `missing creator for mutated node`.
    - next: test
  timestamp: 2026-09-24T20:54:37.590178+00:00
- actor: claude-code
  id: 01m3ak6de8aeg8crrtv3gh5mq5
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` (only the known warning `missing creator for mutated node`); `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — Swift Testing: 284 tests in 32 suites passed, 0 failed, 0 skipped (baseline 283 + 1 new); XCTest: 0 tests in this bundle.
    - next: commit
  timestamp: 2026-09-24T20:54:58.248842+00:00
- actor: claude-code
  id: 01m3ak701n7bg3kx8tntva8t1f
  text: |-
    ### commit — changed
    - evidence: ad3d1d3 refactor(prompt-cache): rename PromptCacheScope.none to .uncached
    - next: review HEAD~1..HEAD
  timestamp: 2026-09-24T20:55:17.301523+00:00
- actor: claude-code
  id: 01m3akavja50s69jdc6s51v0zc
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (ad3d1d3) — findings 0, confirmed 0, refuted 0; 7 validator runs attempted, 0 failed. No earlier Review Findings sections.
    - next: done
  timestamp: 2026-09-24T20:57:23.786271+00:00
- actor: claude-code
  id: 01m3akaxev73d0wa3wky3fa1zt
  text: |-
    ### finish iteration 1 — review clean
    - implement: changed — 3 files (MLXLanguageModel+PromptCacheScope.swift, MLXLanguageModel.swift, PromptCacheScopeTests.swift)
    - test: green — MLXFoundationModelsTests 284 passed, 0 failed, 0 skipped; build has only the known warning
    - commit: ad3d1d3
    - review: clean — 0 findings
  timestamp: 2026-09-24T20:57:25.723195+00:00
depends_on:
- 01M3A1W91WSM28W94MS2MK47NR
position_column: done
position_ordinal: ff9c80
title: Rename PromptCacheScope.none to .uncached, because .none collides with Optional.none
---
#prompt-cache

## What

`MLXLanguageModel.promptCacheScope` (task ^2mk47nr, file `Libraries/MLXFoundationModels/MLXLanguageModel+PromptCacheScope.swift`) has the type `PromptCacheScope?`. A host that writes `MLXLanguageModel.$promptCacheScope.withValue(.none) { ... }` gets `Optional.none` (no scope: the first-entry-id rule), not `PromptCacheScope.none` (no cache). The compiler gives no error. A summarizer that makes this mistake silently adds a key to the store, which is the bug the case exists to prevent.

- Rename the case `PromptCacheScope.none` to `PromptCacheScope.uncached`. Keep `.session(String)`.
- Remove the doc-comment warning that tells hosts to write the full name; it is no longer necessary.
- Update `sessionCacheKey` in `Libraries/MLXFoundationModels/MLXLanguageModel.swift` and every test that uses the case.
- No deprecated alias: the API is new and not released.

## Acceptance Criteria
- [x] No case named `none` exists in `PromptCacheScope`.
- [x] `withValue(.uncached)` gives a pass that takes no cache and leaves none.
- [x] `withValue(nil)` gives the first-entry-id rule.

## Tests
- [x] Update `Tests/MLXFoundationModelsTests/PromptCacheScopeTests.swift` to use `.uncached`, and add one test that binds `nil` and asserts the first-entry-id rule.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` — all pass.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.