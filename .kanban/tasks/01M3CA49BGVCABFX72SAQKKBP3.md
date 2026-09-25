---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cp7ep2w0f3x0rgz38p6sp2
  text: |-
    Research:
    - `ScriptedLanguageModel` writes zero keys and values and emits its script. Thus no existing executor restore test can see a wrong restored value.
    - A spill file is a safetensors file. The arrays are the cache layers ("<layer>.<index>"). The offsets, the ledgers, the key and the format are in the metadata. `loadArraysAndMetadata(url:)` and `save(arrays:metadata:url:)` of MLX read and write it. Thus a test can change the array values and keep the header valid.
    - Plan: a new scripted model in the test file that writes each input token ID into its cache as key and value, and emits, at step N of a round, the token that the cache holds at position N (the start of the rendered prompt). The first positions come from the restored prefix, thus the output depends on the restored values. A value change of +1 in the file changes each emitted letter.
  timestamp: 2026-09-25T16:26:26.882892+00:00
- actor: claude-code
  id: 01m3cpadgq40e2xewkmdtqffqv
  text: |-
    Implementation landed in Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift:
    - New model `CacheEchoLanguageModel`: it writes the input token IDs into its cache, and emits the tokens that the cache holds at positions 0..3.
    - New helper `CacheEchoSession.run(budget:changesSpilledValues:)`: two-turn session, memory budget 1 GiB (memory) or 0 (disk). The negative control loads the spill file with `loadArraysAndMetadata`, adds 1 to each array (same dtype and shape), and saves it with the same metadata.
    - New tests: "a turn restored from disk gives the output of a turn restored from memory" and "a turn restored from a spilled file with changed values gives a different output".
    - Measured with a temporary print (removed): memory text "firs", changed-file text "gjst", reused 10 tokens in both runs. No library bug found: the disk restore gives the memory output.

    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift
    - next: test
  timestamp: 2026-09-25T16:28:03.991315+00:00
- actor: claude-code
  id: 01m3cpfgfghx79vpypc1zd23ar
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` (only the known warning `missing creator for mutated node`); `xcrun xctest` for each bundle: MLXLMTests 695 XCTest + 1208 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 301 (3 runs, all pass), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. 0 failures, 0 skipped.
    - next: commit
  timestamp: 2026-09-25T16:30:50.864460+00:00
- actor: claude-code
  id: 01m3cpjrrwsjcnraz076n7dya4
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` — 0 findings, 0 confirmed, 0 refuted, 7 validators attempted, 0 failed. No prior review findings on the task.
    - next: move to done
  timestamp: 2026-09-25T16:32:37.660995+00:00
- actor: claude-code
  id: 01m3cpjvd07qbyd99xcfr8a64n
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift (new cache-echo model, disk-vs-memory test, negative control). No library bug found.
    - test: green — MLXLMTests 695 + 1208, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 301 (3 runs), CXGrammarTests 7, MLXHuggingFaceMacrosTests 5; 0 failures, 0 skipped; only the known build warning.
    - commit: 4818182 test(prompt-cache): prove that a disk restore gives the output of a memory restore
    - review: clean — review sha HEAD~1..HEAD, 0 findings
  timestamp: 2026-09-25T16:32:40.352800+00:00
position_column: done
position_ordinal: ffa480
title: Add an executor disk-restore test whose output depends on the restored cache content
---
## What

Weak assertion. `Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift:178-189` (`expectASpilledSessionComesBackWarm`) asserts `warm.reusedTokenCount > 0` and `warm.responseText == cold.responseText`. The model is `ScriptedLanguageModel` (`Tests/MLXFoundationModelsTests/ScriptedModelTestSupport.swift:117-225`). It writes zeros into its caches (203-211) and emits the scripted text `"A"` whatever the caches hold (214-224, `ScriptedSessionModel.scriptedResponse` at 310). Thus the text check passes when the restored cache holds wrong values. The test proves reuse, but not a correct restore.

The same weakness is in `ExecutorPromptCacheRestoreTests.swift:198-204` (reuse and file count only).

What to test (add a new test; do not change the existing tests):
- Run the executor two times over the same two-turn session, with the same tiny model: one run with a store whose memory budget is large (the cache stays in memory), and one run with memory budget 0 (the cache goes to disk).
- Use a model whose output depends on the cache content: the tiny Qwen3.5 model of `Qwen35ContinuationTests.makeTinyModel()` wrapped in a `ModelContainer` with a byte tokenizer, or a scripted model that writes the token IDs into its caches and emits a token that it reads from the cache.
- Expect: the same `reusedTokenCount` in both runs, the same response text, and a response text that is different when the restored cache is changed on purpose (negative control: write a corrupt value into the file arrays, keep the header valid).

## Acceptance Criteria

- [x] A test proves that the disk-restored turn gives the output of the memory-only turn with a model whose output depends on the cache.
- [x] The negative control proves that the test fails when the restored values are wrong.
- [x] All five unit bundles pass.

## Tests

Test file: `Tests/MLXFoundationModelsTests/ExecutorPromptCacheRestoreTests.swift`.

Run:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest
```

## Workflow

- Use `/tdd` #coverage-gap #prompt-cache