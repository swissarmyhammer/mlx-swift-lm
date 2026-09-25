---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3ct09cqdrm881p295wyjhgv
  text: |-
    Research:
    - `ExecutorPromptCacheFile.read` calls `loadPromptCacheSnapshot(url:into:)` and then checks that each offset equals the ledger length. Thus one read covers the restore into the `newCache` templates and the offset check.
    - `maybeQuantizeKVCache(kvScheme: "turbo4")` does not give `TurboQuantKVCache` layers in the tiny Qwen3.5 model. It has two attention layers, and the boundary protection (`KVCacheTree.turboQuantProtectedPaths`) changes both to 8-bit `QuantizedKVCache` for a fragile scheme. The non-fragile scheme `turbo0v4` gives `TurboQuantKVCache` layers.
    - The head dimension of the tiny model is 32. The default kv group size 64 does not divide it, thus `toQuantized` fails and `maybeQuantizeKVCache` keeps the `KVCacheSimple`. The test uses group size 32.
    - `ExecutorPromptCacheFileTests.expectEqualCaches` is private. The new test makes it internal and uses it again.
  timestamp: 2026-09-25T17:32:26.391323+00:00
- actor: claude-code
  id: 01m3ctax1q098ba88g949mq3a8
  text: |-
    Bug found and fixed (TDD). Cause: `PromptCacheTemplateRestore.fixedConfigurationIndices` in `Libraries/MLXLMCommon/KVCache.swift` had no entry for `RotatingKVCache`. The restore thus wrote the saved `metaState` into the template, and the `metaState` setter changed the window (`maxCacheSize`) and the kept prefix (`keep`) of the ring that the model made. A ring of window 8 restored without error into `newCache(parameters: GenerateParameters(maxKVSize: 16))`, and the model then got a window of 8 and not the window of 16 that it asked for. The red test: the negative control in `ExecutorPromptCacheQwenFileTests` and the new `ringConfigurationMismatchThrows` (window and keep) in `PromptCacheTemplateRestoreTests` (both recorded "an error was expected but none was thrown", and the template metaState changed).
    Fix: `fixedConfigurationIndices` now has `"RotatingKVCache": [rotatingKeep, rotatingMaxSize]` (new index `SavedMetaStateIndex.rotatingKeep = 0`). The check runs before any setter, thus the template stays as it was. The executor catches the error of `ExecutorPromptCacheFile.read` and reports a restore failure, thus a ring of another window gives a cold prefill.
    Tests added:
    - `ExecutorPromptCacheQwenFileTests.aConvertedEntryContinuesAsTheLiveCaches` (kvBits, TurboQuant `turbo0v4`, wrapped ring of window 8 after a 40-token prefill and 10 single-token decode steps): class names, template identity for kept layers, offsets, state and metaState, and continuation logits within 1e-6 of the live caches, all through `ExecutorPromptCacheFile.prepare/write/read`.
    - `ExecutorPromptCacheQwenFileTests.aWrappedRingIntoAnotherWindowThrows`: KVCacheError.
    - `PromptCacheTemplateRestoreTests.ringConfigurationMismatchThrows` (window, keep).
    Refactors: `ExecutorPromptCacheFileTests.expectEqual/expectEqualCaches` are internal now (reused). `writePrefill` uses the new helpers `writeEntry` and `writeFile`.
  timestamp: 2026-09-25T17:38:14.199323+00:00
- actor: claude-code
  id: 01m3ctb0qv22rn7qeg2hrxtpsm
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/KVCache.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheQwenFileTests.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheFileTests.swift
    - next: test
  timestamp: 2026-09-25T17:38:17.979254+00:00
- actor: claude-code
  id: 01m3ctsdkmvdnvg5e62t6qnnm0
  text: |-
    ### test — green
    - evidence: swift build --build-tests (only the known warning "missing creator for mutated node"); xcrun xctest of all five bundles: MLXLMTests 695 XCTest + 1211 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 309, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures, 0 skipped. MLXLMTests and MLXFoundationModelsTests ran 3 times, all green.
    - next: commit
  timestamp: 2026-09-25T17:46:09.908974+00:00
position_column: doing
position_ordinal: '80'
title: Test converted (kvBits, TurboQuant) and wrapped rotating layers inside a tiny Qwen3.5 hybrid restore
---
## What

Path gap. Measured coverage of the code:
- `Libraries/MLXLMCommon/KVCache.swift` `PromptCacheTemplateRestore.prepareConverted(_:into:templateClassName:)` 2530-2543: 12/12.
- `Libraries/MLXVLM/Models/Qwen35.swift` `makeCache(capacity:)` (1035-1045, the `capacity.makeRotatingCache()` branch at 1041) is part of `Qwen35.newCache`.

What the tests prove today:
- A converted layer restores into a `KVCacheSimple` template, one layer alone: `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift:298-313` (`.quantized`, `.turboQuant`).
- A converted child of a `CacheList(KVCacheSimple(), MambaCache())`: same file, 315-333.
- A wrapped `RotatingKVCache`, one layer alone: same file, `CacheKind.rotatingAfterWrap`.
- The executor file test uses an unwrapped ring only: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheFileTests.swift:43-45`.

No test restores a Qwen hybrid layer list (`[MambaCache, ..., QuantizedKVCache or TurboQuantKVCache or RotatingKVCache]`) whose attention layers generation converted, into the `newCache` templates of that model. No test sends such an entry through `ExecutorPromptCacheFile`, whose offset check reads every layer.

What to test, with the tiny Qwen3.5 model of `Qwen35ContinuationTests.makeTinyModel()`:
- kvBits: prefill, then convert the attention layers with `maybeQuantizeKVCache` (or `toQuantized(groupSize:bits:)`). Save, restore into `model.newCache(parameters: nil)`. Expect the attention layers to come back as `QuantizedKVCache`, the `MambaCache` layers as their templates, all offsets equal, and continuation logits equal (1e-6) to the live converted caches.
- TurboQuant: the same with `TurboQuantKVCache` attention layers.
- Rotating: make the caches with `GenerateParameters(maxKVSize: 8)`, prefill and decode past the window (the ring wraps). Save and restore into `newCache(parameters: GenerateParameters(maxKVSize: 8))`. Expect equal state, meta state and continuation logits.
- Send each of the three entries through `ExecutorPromptCacheFile.prepare/write/read` with a ledger of the right length, and expect no `offsetMismatch`.
- Negative control: restore the rotating file into `newCache(parameters: GenerateParameters(maxKVSize: 16))`. Expect `KVCacheError`.

## Acceptance Criteria

- [ ] A test proves each of kvBits, TurboQuant and a wrapped ring inside the Qwen3.5 hybrid layer list.
- [ ] Each test compares continuation logits, not only offsets.
- [ ] Each case also passes through `ExecutorPromptCacheFile.read`.
- [ ] The window-mismatch control throws.
- [ ] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheQwenFileTests.swift` (the file of the Qwen3.5 hybrid task; create it when it is not there).

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache