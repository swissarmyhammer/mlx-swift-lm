---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3dg6zp5e3bc1kwvpzbeqkfr
  text: |-
    Research (the lines moved):
    - `KVCacheSerializationRegistry.className(for:)` is now at KVCache.swift:2036. Only `cacheClassName` (2129) calls it, and only for a cache that is not built in and not `PromptCacheRestorable`. `MiniMaxM3KVCache` is registered but also `PromptCacheRestorable`, thus no current cache reaches 2136 through the registry. A test cache class is necessary.
    - `KVCacheSerializationRegistry.validate(state:metaState:for:)` is at 2065; its throw is 2071.
    - `PromptCacheTemplateRestore.prepare` is at 2528; the throw for a template that is not built in is 2544-2546. A custom template reaches it only when the saved class name equals the name of the template: `"KVCache"` for an unregistered class, or the registered name.
    - `validateFixedConfiguration` is at 2619; its throw is 2628-2632.
    - `restoreCacheFromMetaState` unknown class throw is at 2898.
    - `validateBuiltInCache` `default` throw is at 3021.

    Possible bug: `fixedConfigurationIndices` (2504) has no `QuantizedKVCache` entry. The `metaState` setter of `QuantizedKVCache` (1273) writes `groupSize` and `bits`. Thus a saved `QuantizedKVCache(groupSize: 64, bits: 8)` restores into a `QuantizedKVCache(groupSize: 64, bits: 4)` template without an error, and changes the template to 8 bits. I will write the test first and confirm it fails.

    Unreachable: `validateBuiltInCache` `default` (3021). Each caller passes only a name in `builtInLeafClassNames`: `restoreCacheFromMetaState` guards with that set (2892), `KVCacheSerializationRegistry.validate` guards with that set (2068-2069), and `prepare` calls it only when the saved name equals the name of a built-in template that is not `CacheList` (2530, 2533, 2544). Every name in `builtInLeafClassNames` has a `case`. An unknown saved name on a template restore of a `KVCacheSimple` template stops at the class check in `prepareConverted` (2569-2575).
  timestamp: 2026-09-26T00:00:34.501972+00:00
- actor: claude-code
  id: 01m3dgmk7a7dreyp4thg724cqb
  text: |-
    Bug found and fixed (TDD). Cause: `PromptCacheTemplateRestore.fixedConfigurationIndices` had no `QuantizedKVCache` entry, but the `metaState` setter of `QuantizedKVCache` writes `groupSize` and `bits`. A saved `QuantizedKVCache(groupSize: 64, bits: 8)` thus restored into a `QuantizedKVCache(groupSize: 32 or 64, bits: 8 or 4)` template without an error, and changed the template to the saved quantization. Red: `quantizedConfigurationMismatchThrows` failed for both cases (no error, metaState changed). Fix: add `"QuantizedKVCache": [quantizedGroupSize, quantizedBits]` to the table, and state the rule in the two doc comments.

    Break checks: with `className(for:)` changed to give `nil`, and with a changed message in each of `validate` (2072), `prepare` (2553), `prepareConverted` (2580), `restoreCacheFromMetaState` (2905) and `validatePromptCache` (3456), each of the six new tests failed. The breaks were then removed (the file was restored from a copy, and `git diff` shows only the fix).

    Not reachable from public input: `validateBuiltInCache` `default` (3021), as the research comment states.

    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/KVCache.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift
    - next: test
  timestamp: 2026-09-26T00:08:00.490303+00:00
- actor: claude-code
  id: 01m3dh75jpwb176g83mgz9z1he
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. `xcrun xctest` MLXLMTests 3 runs: each 696 XCTest 0 failures + 1229 Swift Testing passed (was 1222; 7 new cases). MLXGuidedGenerationTests 70 passed, MLXFoundationModelsTests 323 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 5 XCTest 0 failures. 0 skipped.
    - next: commit
  timestamp: 2026-09-26T00:18:09.110626+00:00
- actor: claude-code
  id: 01m3dh8273zdz2ms2gjrfshkn1
  text: |-
    ### commit — changed
    - evidence: 7c49a44 fix(prompt-cache): refuse a template restore into a QuantizedKVCache of another quantization
    - next: review HEAD~1..HEAD. The three acceptance boxes are ticked.
  timestamp: 2026-09-26T00:18:38.435504+00:00
- actor: claude-code
  id: 01m3dhd7z405hysjs1v4n8921m
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD — 0 findings, 0 confirmed, 0 refuted (7 validators attempted, 0 failed). No prior findings.
    - next: done
  timestamp: 2026-09-26T00:21:28.164312+00:00
- actor: claude-code
  id: 01m3dhdb237mwg2p1d292kjezr
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — fix `fixedConfigurationIndices` (add `QuantizedKVCache` group size and bit width) in Libraries/MLXLMCommon/KVCache.swift; 7 new test cases and a message check on the ring mismatch test in Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift
    - test: green — MLXLMTests 3 runs (696 XCTest + 1229 Swift Testing, 0 failures), MLXGuidedGenerationTests 70, MLXFoundationModelsTests 323, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5; 0 skipped; only the known warning `missing creator for mutated node`
    - commit: 7c49a44
    - review: clean (0 findings)
  timestamp: 2026-09-26T00:21:31.331651+00:00
position_column: done
position_ordinal: ffb580
title: Test the template restore rejections and the registry class name fallback
---
## What

File: `Libraries/MLXLMCommon/KVCache.swift`.

- `static KVCacheSerializationRegistry.className(for:)`, lines 2010-2014: 0/5 (0%). Uncovered 2010-2014.
- `cacheClassName(_:)`, lines 2103-2111: 8/9. Uncovered 2110: a cache that is not built in and not `PromptCacheRestorable` gets its name from the registry, or `"KVCache"`.
- `static KVCacheSerializationRegistry.validate(state:metaState:for:)`, lines 2039-2049: 7/9. Uncovered 2045-2046: a cache that is not a built-in leaf class throws `KVCacheError`.
- `static PromptCacheTemplateRestore.prepare(_:into:)`, lines 2492-2519: 26/28. Uncovered 2509-2510: a template that is not built in, not `CacheList` and not `PromptCacheRestorable` throws.
- `static PromptCacheTemplateRestore.validateFixedConfiguration(_:template:)`, lines 2583-2598: 10/14. Uncovered 2593-2596: a saved fixed configuration value (for example the `maxSize` of a `RotatingKVCache`, or the group size or bits of a `QuantizedKVCache`) that is not the value of the template throws.
- `restoreCacheFromMetaState(className:state:metaState:)`, lines 2830-2850: 16/17. Uncovered 2844: an unknown class name that the registry does not know throws.
- `validateBuiltInCache(className:state:metaState:)`, lines 2943-2967: 22/23. Uncovered 2965: the `default` case throws `Unknown cache class`.

The executor restores a Qwen hybrid into `model.newCache(parameters:)` templates. When the window or the quantization of the model changes between the save and the restore, the restore must throw and the turn must start cold. Line 2593-2596 is that check.

What to test:
- Register a test cache class with `KVCacheSerializationRegistry.register`, save it, and check that the file names that class. Load it with `loadPromptCacheSnapshot(url:)`.
- Save a cache of an unregistered custom class: the class name in the file is `"KVCache"`, and the load throws `KVCacheError`.
- `KVCacheSerializationRegistry.validate(state:metaState:for:)` with a `CacheList` or a custom cache throws.
- Restore a saved `RotatingKVCache(maxSize: 4)` into `RotatingKVCache(maxSize: 8)`, and a saved `QuantizedKVCache(groupSize: 64, bits: 8)` into `QuantizedKVCache(groupSize: 64, bits: 4)`: both throw `KVCacheError`.
- Restore into a custom template that is not `PromptCacheRestorable`: throws.
- A file whose class name is changed to an unknown name throws on `loadPromptCacheSnapshot(url:)` (line 2844) and on the template restore of a `KVCacheSimple` template (line 2965 when reached).

## Acceptance Criteria

- [x] Each uncovered line above is covered, or the task records which line no public input reaches.
- [x] Each rejection test checks `KVCacheError` and that the template keeps its state.
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