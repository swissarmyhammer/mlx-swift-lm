---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3ck4ftsd66pq7n2xb012b64
  text: |-
    Research:
    - `ExecutorPromptCacheFile.read(from:key:templates:)` (Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift:134) already throws `offsetMismatch` when an offset is not the ledger length. The test also expects the offsets itself.
    - The store writes each spill file directly in the directory of `ExecutorPromptCacheStore(directory:)`, with the name `<sha256>-<generation>.safetensors`. A partial file has `.partial.safetensors`.
    - `MLXLanguageModel.$promptCacheScope.withValue(.uncached)` makes `sessionCacheKey` nil: the pass checks out no cache and checks in no cache. Thus a cold control does not touch the store.
    - FoundationModels has `LanguageModelSession(model: some LanguageModel, tools:, transcript:)`. The cold control uses a new session made from the transcript of turn 1.
    - The key of the M-RoPE state is private in Qwen35.swift (`qwen35.ropeDeltas`). The test makes its own `LMOutput.Key<MLXArray>("qwen35.ropeDeltas")`.
    - `PromptCacheSpoolCostAssessmentTests.SpoolCostProbe` has `prefill` and `contextTokens`. The new logit comparison reuses them.
  timestamp: 2026-09-25T15:32:24.025620+00:00
- actor: claude-code
  id: 01m3cm8nakfbs42td7brg9tmhj
  text: |-
    ### implement — changed
    - evidence: 2 files — IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift, IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift. `xcodebuild build-for-testing` succeeded. `xcodebuild test-without-building -only-testing:.../Qwen35SessionPromptCacheTests -only-testing:.../PromptCacheSpoolCostAssessmentTests`: 8 tests in 2 suites passed.
    - measurements: spilled file of turn 1 = ledger 101 tokens, render 73 tokens, 64 cache offsets all 101, ropeDeltas present. Restored turn 2 cached 101 of 126; uncached turn 2 cached 0; both texts "teal". Max absolute logit difference over 8 steps = 0.0 at every step for qwen3-4b and for qwen3.8-27b (the restore is exact).
    - no library bug: the new tests pass on the first run.
    - next: /test
  timestamp: 2026-09-25T15:52:09.299778+00:00
- actor: claude-code
  id: 01m3cnxwa69qdbbk5gym56t1cp
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. xctest: MLXLMTests 695 XCTest + 1208 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 299, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures, 0 skipped. Integration: Qwen35SessionPromptCacheTests + PromptCacheSpoolCostAssessmentTests — 8 tests in 2 suites passed.
    - next: /commit
  timestamp: 2026-09-25T16:21:13.158034+00:00
position_column: doing
position_ordinal: '80'
title: 'Integration: prove cold equivalence and the M-RoPE state of a Qwen3.5 session restored from disk'
---
## What

Weak assertions in the real-model disk tests of Qwen.

1. `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:134-173` (`expectASecondTurnComesBackWarmFromDisk`, model `mlx-community/Qwen3.8-27B-mxfp4`). It asserts that turn 1 left memory (165), that the disk holds bytes (166), that turn 2 cached at least the render of turn 1 (167-169), and that the answer contains `"teal"` (170). It has no cold control: it does not prove that the restored turn gives the answer of an uncached turn. It does not prove that the model state `qwen35.ropeDeltas` was in the file. It does not prove which rule (`splice` or `extend`) reused the cache.
2. `IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift:174-177, 320-322`. The check compares one greedy argmax token after one decode step. Two different logit vectors can give the same argmax, thus the check can pass with a wrong restore.

Model in the local cache: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-mxfp4` (`model_type` `qwen3_5`, with `vision_config`, 4 safetensors). `models--mlx-community--Qwen3-4B-4bit` is present for the pure-attention control.

What to test (add new tests; do not change the existing ones):
- In `Qwen35SessionPromptCacheTests`: after turn 1 and `waitForSpills()`, read the one spilled file with `ExecutorPromptCacheFile.read(from:key:templates: context.model.newCache(parameters: nil))` inside `container.perform`, and expect `state?[LMOutput.Key<MLXArray>("qwen35.ropeDeltas")] != nil`, `renderTokens` not empty, and every cache offset equal to `tokens.count`. Then run turn 2 twice from the same file copy: one warm run, and one run under `MLXLanguageModel.$promptCacheScope.withValue(.uncached)`. Under greedy sampling, expect the same text.
- In `PromptCacheSpoolCostAssessmentTests`: decode at least 8 greedy steps on the restored caches and on the original caches, and compare the last logits of each step within a tolerance (for example max absolute difference 1e-3 for mxfp4), not only the argmax token.

## Acceptance Criteria

- [ ] A test proves that the file of a Qwen3.5 turn holds the `qwen35.ropeDeltas` state, the render ledger and offsets equal to the ledger length.
- [ ] A test proves that the disk-restored second turn gives the text of an uncached second turn.
- [ ] A test compares restored and original logits over at least 8 decode steps for the hybrid model and for `Qwen3-4B-4bit`.
- [ ] Each test fails with a clear message when its model is not in the local cache.

## Tests

Test files: `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift` and `IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift`.

Run:

```sh
xcodebuild build-for-testing -skipPackagePluginValidation \
    -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting \
    -destination 'platform=macOS'
xcodebuild test-without-building -skipPackagePluginValidation \
    -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting \
    -destination 'platform=macOS' \
    -only-testing:IntegrationTestingTests/Qwen35SessionPromptCacheTests \
    -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache