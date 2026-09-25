---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cs2bsaj1jcqk2arnggb84r
  text: |-
    Research:
    - `Qwen3NextGatedDeltaNet.callAsFunction` (Qwen3Next.swift:255) and the compiled decode step (Qwen3Next.swift:693) call `MambaCache.advance(_:)`. That call moves only the batch bookkeeping. It does not move `offset`. Thus the `MambaCache` layers of a live Qwen3-Next cache stay at offset 0. `Qwen3NextCompiledDecodeTests` records this as `[0, 5, 0, 5]`.
    - Qwen3.5 (Qwen35.swift:349, 790, 1047) calls `MambaCache.advancePosition(by:)`, which moves `offset` also (KVCache.swift:1650).
    - Result: `savePromptCache` writes offset 0 for each linear layer in the `__mlx_lm_offset_` record, and the restored `MambaCache` has offset 0. `ExecutorPromptCacheFile.read` refuses a file whose offsets are not the ledger length, thus a Qwen3-Next entry can never warm the executor.
    - Expected: the new round-trip test fails on the offset check of the `MambaCache` layers. Fix: use `advancePosition(by:)` in both places, as Qwen3.5 does.
    - Reuse: `PromptCacheTemplateRestoreTests` has `temporaryURL()`, `expectSameContents(_:_:_:)` and `isSameInstance(_:_:)` as `fileprivate`. Make them internal and call them, instead of a third copy.
  timestamp: 2026-09-25T17:16:05.802493+00:00
- actor: claude-code
  id: 01m3csde3va4et0t5ffmnkt2tm
  text: |-
    Bug found and fixed (TDD).
    - RED: the new `Qwen3NextPromptCacheRestoreTests` failed with 4 issues. All were offsets: the live `MambaCache` layers had offset 0 after a 10-token prefill, the restored `MambaCache` layers had offset 0, and after 4 decode steps they had offset 4 and not 14. The state, the meta state and the logits (1e-6 against live, 1e-3 against cold) already agreed.
    - Cause: `Qwen3NextGatedDeltaNet.callAsFunction` and the compiled `decodeStep` of `Qwen3NextModelInner` called `MambaCache.advance(_:)`, which moves only the batch bookkeeping. The save thus recorded offset 0 for each linear layer, and `ExecutorPromptCacheFile.read` refuses such a file because its offsets are not the ledger length. A Qwen3-Next entry could never warm the executor.
    - Fix: both places now call `MambaCache.advancePosition(by:)`, as Qwen3.5 does. `Qwen3NextCompiledDecodeTests` recorded the defect as `[0, 5, 0, 5]`; it now expects `[5, 5, 5, 5]`.
    - Reuse: `Qwen3NextCompiledDecodeTests.configuration(headDim:quantizable:)` is now an internal static function, and `PromptCacheTemplateRestoreTests.temporaryURL()`, `expectSameContents(_:_:_:)` and `isSameInstance(_:_:)` are now internal. The new test calls them.
    - GREEN: MLXLMTests 695 XCTest + 1210 Swift Testing, 0 failures.

    ### implement — changed
    - evidence: 4 files — Libraries/MLXLLM/Models/Qwen3Next.swift, Tests/MLXLMTests/Qwen3NextPromptCacheRestoreTests.swift (new), Tests/MLXLMTests/Qwen3NextCompiledDecodeTests.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift
    - next: /test
  timestamp: 2026-09-25T17:22:08.635457+00:00
- actor: claude-code
  id: 01m3cspk7hz4jhfekbzaqa6zwk
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` (one known warning `missing creator for mutated node`, no new warning). `xcrun xctest` MLXLMTests 3 runs: each 695 XCTest + 1210 Swift Testing, 0 failures, 0 skipped. MLXGuidedGenerationTests 70, MLXFoundationModelsTests 307, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5: 0 failures, 0 skipped.
    - next: /commit
  timestamp: 2026-09-25T17:27:08.785969+00:00
position_column: doing
position_ordinal: '80'
title: Test a tiny Qwen3-Next hybrid cache through the template restore and a warm decode
---
## What

Path gap. File: `Libraries/MLXLLM/Models/Qwen3Next.swift`, `Qwen3NextModel.newCache(parameters:)`, lines 773-780: 8/8 covered. File coverage: 87.87% (594/676).

No test saves a Qwen3-Next cache to a file. No test restores one into `newCache` templates. `grep` finds no `savePromptCache` or `loadPromptCacheSnapshot` call with a Qwen3-Next model in `Tests/` or `IntegrationTesting/`. The executor spool restores every layer of this model through `loadPromptCacheSnapshot(url:into:)` (`Libraries/MLXLMCommon/KVCache.swift:2352-2370`). The linear layers use `MambaCache`, whose offset comes back only through the `__mlx_lm_offset_` record (`KVCache.swift:2611-2710`), and `ExecutorPromptCacheFile.read` refuses a file whose offsets are not the ledger length.

What to test:
- Make a tiny `Qwen3NextModel` with random weights (use the configuration style of `Tests/MLXLMTests/Qwen3NextCompiledDecodeTests.swift`), at least one linear layer and one full-attention layer.
- Prefill a prompt into `newCache(parameters: nil)`. Save with `savePromptCache`. Restore with `loadPromptCacheSnapshot(url:into: model.newCache(parameters: nil))`.
- Expect: each restored cache is its template instance, each offset equals the prompt length (the `MambaCache` layers also), and the state and meta state equal the saved caches.
- Decode 4 greedy tokens on the restored caches and on the live caches. The logits must agree within 1e-6 at each step. The first-step logits must agree within 1e-3 with a cold prefill of the prompt plus the first token.

## Acceptance Criteria

- [ ] A test proves the Qwen3-Next round trip into `newCache` templates.
- [ ] The test checks the offset of every layer, the `MambaCache` layers included.
- [ ] The test checks logits over 4 decode steps, and not only one argmax token.
- [ ] All five unit bundles pass.

## Tests

Test file: `Tests/MLXLMTests/Qwen3NextPromptCacheRestoreTests.swift` (new file).

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache