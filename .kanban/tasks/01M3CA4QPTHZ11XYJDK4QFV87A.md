---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cdbsq7cgx6d0xcw3ga6j7q
  text: |-
    Research:
    - `SessionDriver.runRound` reads `ExecutorPromptCacheStore.shared`. The executor reads `ExecutorPromptCacheStore.current` (a `@TaskLocal`, default `.shared`). Thus the driver must read `.current` to see a store that a test binds.
    - `ExecutorPromptCacheStore.peek` reads only the memory entries. With `configure(memoryBudgetBytes: 0)`, `checkIn` spills each entry, thus `peek` gives nil after the round. The ledger seam of a disk round is thus empty; the proof of reuse is `cachedTokenCount >= previous.renderedTokenCount - cacheSeamSlack`.
    - `MissingLocalModelError` and `requireLocalWeights` are private in `PromptCacheSpoolCostAssessmentTests.swift`. Plan: move them to `FMTestHelpers.swift` as an `MLXLanguageModel` extension, so that the two suites use one copy. The same for the `unsupportedSystem` comment (a `Comment` extension).
    - Plan for the new test: run the 5 rounds (4 tool rounds) two times with the same driver: one time with a store at budget 0 (disk), one time with a store at budget `Int.max` (memory). Compare the generated text and the tool call of each round.
  timestamp: 2026-09-25T13:51:32.071249+00:00
- actor: claude-code
  id: 01m3cerjydc4ra1vpb6ds0smhc
  text: |-
    ### implement — changed
    - evidence: 3 files — IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift (new test `hybridModelRestoresEachToolRoundFromDisk`, driver reads `ExecutorPromptCacheStore.current`, waits for spills, records `isInMemoryAfterRound` and `diskByteCountAfterRound`), IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Support/FMTestHelpers.swift (`MissingLocalModelError`, `MLXLanguageModel.requireLocalWeights()`, `Comment.unsupportedSystem` moved here), IntegrationTesting/IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests.swift (uses the shared helpers).
    - `xcodebuild build-for-testing`: TEST BUILD SUCCEEDED, no new warning in the changed files.
    - `-only-testing:.../hybridModelRestoresEachToolRoundFromDisk()`: passed after 829 s. The new test found no library bug: the disk run and the memory run gave the same output in all 5 rounds (4 tool calls, bays 3, 7, 11, 15, then text). Memory run: round 2..5 cached 27581/27666/27751/27838 of the 27434/27616/27701/27787 tokens that the round before rendered (the seam is 35-36 tokens under the next render, inside `cacheSeamSlack`).
    - next: /test
  timestamp: 2026-09-25T14:15:59.693332+00:00
- actor: claude-code
  id: 01m3cjhf6fks0vfgj6sxqxzq97
  text: |-
    ### test — green
    - evidence: `xcodebuild build-for-testing` (IntegrationTesting): TEST BUILD SUCCEEDED; no warning in the 3 changed files (the other warnings are in files that this task did not touch). `xcodebuild test-without-building -only-testing:IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests`: 6 tests, 5 passed. `PromptCacheSpoolCostAssessmentTests` 3/3 passed; `hybridModelRestoresEachToolRoundFromDisk` passed (1012 s); `controlModelCarriesThePromptCacheAcrossToolRounds` passed.
    - `hybridModelCarriesThePromptCacheAcrossToolRounds` failed one time at `roundFour.prefillSeconds < roundTwo.prefillSeconds * prefillGrowthLimit` (round 2 2.80 s, round 4 6.65 s, for 35 and 36 fed tokens). Cause: an external GPU load. A game (CivilizationVII, 1100 % CPU) started on this machine during the run; round 1 prefill took 231 s in place of the usual 85-99 s. The test passed on a re-run with no code change (371 s; prefill of round 2..5 = 0.35 / 0.38 / 0.44 / 0.39 s). The change in this task does not touch that timing: for the shared store, `waitForSpills()` returns at once when no write is in the queue, and it runs after the round, not in the prefill.
    - SwiftPM bundles: this task changes no SwiftPM target (only `IntegrationTesting/`), thus the baseline stays.
    - next: /commit
  timestamp: 2026-09-25T15:22:00.783858+00:00
position_column: doing
position_ordinal: '80'
title: 'Integration: Qwen3.5 hybrid agentic tool rounds with a spill to disk between each round'
---
## What

Real-model gap. `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift` runs the agentic shape (a prompt, tool calls, tool outputs, more rounds) on `mlx-community/Qwen3.8-27B-mxfp4` (line 521-524). It reads `ExecutorPromptCacheStore.shared` (lines 376 and 386) and never sets a memory budget. Thus each round finds its cache in memory, and no round restores from disk. No test runs tool rounds with a spill between the rounds.

The disk tier changes three things for a tool round: the spilled file must hold the ledger of the round and the render of the round (`renderTokens`) for `QwenCommittedTurnRule`; the `MambaCache` offsets come back only through the offset record; and the Qwen VL state `qwen35.ropeDeltas` must come back with the caches. (The model `Qwen3.8-27B` has `model_type` `qwen3_5` with a `vision_config`, and `ModelFactoryRegistry` tries MLXVLM first, `Libraries/MLXLMCommon/ModelFactory.swift:489-500`, thus the executor gets the MLXVLM `Qwen35` model and its state.)

Model in the local cache: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-mxfp4` (4 safetensors, no incomplete blob). `models--mlx-community--Qwen3.8-27B-4bit` is also present.

What to test:
- Run the same tool rounds as `hybridModelCarriesThePromptCacheAcrossToolRounds`, inside `ExecutorPromptCacheStore.$current.withValue(store)` with a store in a temporary folder and `configure(memoryBudgetBytes: 0)`.
- After each round, `waitForSpills()`, and expect `store.peek(key) == nil` and `store.diskByteCount > 0`.
- Expect each round `cachedTokenCount >= previous.renderedTokenCount - cacheSeamSlack` (the same bound as line 593-598).
- Run the same rounds a second time with a memory-only store (large budget). Expect the same answers and tool calls under greedy sampling in both runs.

## Acceptance Criteria

- [x] A new test runs at least 3 tool rounds with a spill to disk before each round after the first.
- [x] The test proves that each round after the first reuses the previous render from disk.
- [x] The test proves that the disk run and the memory run give the same output.
- [x] The test fails with a clear message when the model is not in the local cache (the pattern of `PromptCacheSpoolCostAssessmentTests.requireLocalWeights`).

## Tests

Test file: `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift`.

Run:

```sh
xcodebuild build-for-testing -skipPackagePluginValidation \
    -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting \
    -destination 'platform=macOS'
xcodebuild test-without-building -skipPackagePluginValidation \
    -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting \
    -destination 'platform=macOS' \
    -only-testing:IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache