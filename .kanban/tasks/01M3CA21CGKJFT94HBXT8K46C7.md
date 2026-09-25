---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3de2sg43v5y99r1d6sksks5
  text: 'Research: the lines moved. Current places in Libraries/MLXLMCommon/KVCache.swift: `CacheList.savedChildren` 1920-1941 (child count guard 1923-1925), `CacheList.savedChild` 1952-1970 (truncated header guard 1956-1958, count guard 1959-1965), `PromptCacheTemplateRestore.prepareCacheList` 2589-2611 (child count mismatch guard 2595-2600). A removed metadata key "0.0.0" does not remove the value: `unflattenMetadata` fills the gap with "", thus `Int("")` is nil and the child count guard throws. The file keys of the meta state of layer 0 are "0.0.<index>". The saved CacheList meta state is [childCount, (className, stateCount, metaStateCount, metaState...)*].'
  timestamp: 2026-09-25T23:23:19.940262+00:00
- actor: claude-code
  id: 01m3deqhbt6csjc46r8q157sf8
  text: |-
    Implementation: 4 new Swift Testing tests (9 cases) in Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, section "CacheList rejections". Each test restores a tampered file into a CacheList template through the helper `expectRejected(_:into:message:)`. The helper checks KVCacheError, the message of the check that must throw, and that the template list keeps its own empty children.

    Discovery: a check of the error type alone does not prove the path. When I broke each guard (the throw became a return), another later check (the final "values that no child owns" guard, the child-count guard of prepareCacheList, or the offset record) still threw KVCacheError. Thus the tests also compare the message.

    Break checks (each break was removed; git checkout of KVCache.swift):
    - A: child count guard returns [] -> 3 cases of badCacheListChildCountThrows fail.
    - C: truncated header guard returns an empty child -> truncatedCacheListChildHeaderThrows fails.
    - B: count guard returns an empty child -> 2 cases of overrunCacheListChildCountThrows fail.
    - D (run alone): prepareCacheList count mismatch returns `{ list }` -> 2 cases of cacheListChildCountMismatchThrows fail.
    A first run with A+B+C+D together crashed on the replaceChildren precondition, because A and D interact; thus D ran alone.

    No library bug found. Baseline run with the new tests: MLXLMTests 696 XCTest + 1222 Swift Testing, 0 failures.

    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift
    - next: /test
  timestamp: 2026-09-25T23:34:39.738905+00:00
- actor: claude-code
  id: 01m3df7k7errvd4e04scdpe3m6
  text: |-
    ### test — green
    - evidence: swift build --build-tests (only the known warning "missing creator for mutated node"); MLXLMTests 3 runs, each 696 XCTest + 1222 Swift Testing, 0 failures, 0 skipped; MLXGuidedGenerationTests 70, MLXFoundationModelsTests 323, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5, all pass. TextToolCallRecoveryBenchmark passed.
    - next: /commit
  timestamp: 2026-09-25T23:43:25.934491+00:00
- actor: claude-code
  id: 01m3dfdn96kktgnxkd1y9v2p65
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 2 findings (2 confirmed, 0 refuted): Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift:316 swift/fluent-usage (filledCacheList -> makeFilledCacheList), :324 swift/fluent-usage (cacheListTemplate -> makeCacheListTemplate)
    - next: /implement

    ### finish iteration 1 — findings
    - implement: changed (Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift)
    - test: green (MLXLMTests 3 runs 696 + 1222, other four bundles pass)
    - commit: 9b192fb
    - review: findings (2, swift/fluent-usage at :316 and :324)
  timestamp: 2026-09-25T23:46:44.646123+00:00
- actor: claude-code
  id: 01m3dfzw44z3e9bj9avzy01sg4
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift: renamed `filledCacheList()` to `makeFilledCacheList()` and `cacheListTemplate()` to `makeCacheListTemplate()`, with all call sites (swift/fluent-usage findings). No other new factory function in the file needed the `make` prefix.
    - next: /test

    ### test — green
    - evidence: swift build --build-tests (only the known warning "missing creator for mutated node"); MLXLMTests 3 runs, each 696 XCTest + 1222 Swift Testing, 0 failures, 0 skipped; MLXGuidedGenerationTests 70, MLXFoundationModelsTests 323, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5, all pass.
    - next: /commit
  timestamp: 2026-09-25T23:56:41.476848+00:00
position_column: doing
position_ordinal: '80'
title: Test the rejections of the saved CacheList parser and of a CacheList template restore
---
## What

File: `Libraries/MLXLMCommon/KVCache.swift`.

- `static CacheList.savedChildren(state:metaState:)`, lines 1894-1915: 19/20 (95%). Uncovered 1898: the meta state has no child count, or the count is negative.
- `static CacheList.savedChild(state:metaState:metaIndex:stateIndex:)`, lines 1926-1944: 15/17 (88.2%). Uncovered 1931 (the child header is truncated) and 1938 (a child array count or meta count is out of range).
- `static PromptCacheTemplateRestore.prepareCacheList(_:into:)`, lines 2553-2575: 17/21 (81%). Uncovered 2560-2563: the file holds a `CacheList` with another number of children than the template list.

A Qwen model does not use `CacheList` today, but the template restore of the executor spool reads every `CacheList` layer through these functions. A damaged file must throw `KVCacheError` before a setter writes a template.

What to test (use the tampered-file helper `tamperedFile(_:edit:)` of the test file):
- A saved `CacheList(KVCacheSimple(), MambaCache())` whose child count metadata is removed, is `-1`, or is not a number: `loadPromptCacheSnapshot(url:into:)` throws `KVCacheError`.
- A saved list whose meta state is cut inside the header of the second child throws `KVCacheError`.
- A saved list whose child array count is larger than the arrays of the file throws `KVCacheError`.
- A saved list of 2 children restored into `CacheList(KVCacheSimple())` (1 child) or into a list of 3 children throws `KVCacheError`, and the template list keeps its children.

## Acceptance Criteria

- [x] Lines 1898, 1931, 1938 and 2560-2563 are covered.
- [x] Each test checks `KVCacheError` and that the template was not written.
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift`.

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache

## Review Findings (2026-09-25 18:43)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 1 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3CA21CGKJFT94HBXT8K46C7.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3CA21CGKJFT94HBXT8K46C7.md` — no validator matches this file

- [x] `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift:316` `swift/fluent-usage` — Factory method should follow the `make*` naming convention; `filledCacheList()` creates and returns a `CacheList` instance. Rename to `makeFilledCacheList()` and update call sites at lines 598, 609, 627, 647.
- [x] `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift:324` `swift/fluent-usage` — Factory method should follow the `make*` naming convention; `cacheListTemplate()` creates and returns a `CacheList` instance. Rename to `makeCacheListTemplate()` and update call sites at lines 604, 620, 638, 651.