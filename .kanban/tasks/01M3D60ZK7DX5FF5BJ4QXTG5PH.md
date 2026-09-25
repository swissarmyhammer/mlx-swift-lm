---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3da99ngkn16cwv198zz8e7s
  text: |-
    Research:
    - The branch is reachable. `generate(input:...draftModel:...)` (Evaluate.swift near line 1944 and 2206) and `ChatSession` (line 1372) make a `SpeculativeTokenIterator` for any `LanguageModel`. Many MLXVLM models (for example Qwen2VL, Qwen25VL, Qwen3VL, Gemma3 VLM, Idefics3, Paligemma) return `.logits` from `prepare`.
    - Main `.logits`: `y` = sampled first token T0. The first verify pass feeds T0 and emits only the tokens after T0. T0 is never put into `pendingTokens`. The speculative stream thus loses its first token.
    - Draft `.logits`: `draftY` = the draft sample d0. The draft loop feeds d0 and then proposes the token after d0. When the main model returned `.tokens`, the first proposal must be the prediction for the first generated position, not for the second. The proposals are thus one position late, and the verify pass rejects them.
    - Main `.logits` with draft `.tokens`: `draftY` = the prompt remainder. The first proposal is then the draft prediction for T0, but the verify pass compares it with the main prediction after T0. The proposals are again one position off.
    - Plan: after the two prefills, if the main model gave `.logits`, or if the draft gave `.logits`, bring both models to one state: the main cache and the draft cache hold the full prompt, T0 is sampled from the main logits, T0 is the first pending token (committed to no cache), and `y` = `draftY` = T0. When the main model gave `.tokens` and the draft gave `.logits`, run the main model on its remainder to sample T0. When the draft gave `.tokens` and the main model gave `.logits`, feed the draft remainder to the draft model. When both give `.tokens`, keep the current behavior.
  timestamp: 2026-09-25T22:16:58.800385+00:00
- actor: claude-code
  id: 01m3dagsk705z19fyd744evsef
  text: |-
    Red: the new test `Speculative decoding matches TokenIterator for each prepare result` (Tests/MLXLMTests/SpeculativeDecodingTests.swift) fails before the fix with 8 issues.
    - main `.logits`: the speculative stream is [19, 96, 83, ...] and the `TokenIterator` stream is [52, 19, 96, 83, ...]. The first main token 52 is lost (both for draft `.tokens` and draft `.logits`).
    - main `.tokens` + draft `.logits`: the stream is correct, but the verify pass rejects each draft (the proposals are one position late), and after `finalizeGeneration` the main and draft caches do not hold the same count of tokens.
    - main `.logits` + draft `.tokens`: all drafts are rejected and the caches end misaligned also.
    - main `.tokens` + draft `.tokens`: passes.
    The defect is real and reachable.
  timestamp: 2026-09-25T22:21:04.487254+00:00
- actor: claude-code
  id: 01m3daqxag2pjjh8v07x806544
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/Evaluate.swift, Tests/MLXLMTests/SpeculativeDecodingTests.swift
    - fix: `prepare` now uses one helper `prefill(_:with:into:state:prefill:role:)` for both models, then `startDecoding(main:draft:)`. When a model gives `.logits`, both caches get the whole prompt (the main model runs on its remainder, or the draft model runs on its remainder), the main logits give the first token, that token is the first pending token, and `y` = `draftY` = that token. When both models give `.tokens`, the behavior does not change. `feedDraft` now calls the new `feedDraftModel(_:)`.
    - green: the new test passes for the 4 combinations (tokens/logits for main and draft): equal streams, all drafts accepted, caches aligned after `finalizeGeneration`. MLXLMTests: 696 XCTest 0 failures, 1218 Swift Testing passed.
    - next: /test
  timestamp: 2026-09-25T22:24:57.680541+00:00
- actor: claude-code
  id: 01m3db286mm3dtabjnsns40v61
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` has only the known warning `missing creator for mutated node`. `xcrun xctest` for each bundle: MLXLMTests 3 runs, each 696 XCTest 0 failures + 1218 Swift Testing passed (1217 baseline + 1 new); MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 312 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 XCTest 0 failures. 0 skipped. `TextToolCallRecoveryBenchmark` did not fail.
    - next: /commit
  timestamp: 2026-09-25T22:30:36.500453+00:00
- actor: claude-code
  id: 01m3db6rxmq2emazybww6xj0hg
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (d080716) — 0 findings, 0 confirmed, 0 refuted, 7 validators attempted, 0 failed. The two kanban files matched no validator.
    - next: done
  timestamp: 2026-09-25T22:33:04.692722+00:00
- actor: claude-code
  id: 01m3db6v24bvbt5fx3yetf7bk7
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Libraries/MLXLMCommon/Evaluate.swift, Tests/MLXLMTests/SpeculativeDecodingTests.swift
    - test: green — 5 bundles, 0 failures, 0 skipped; MLXLMTests 3 runs (696 XCTest + 1218 Swift Testing each); only the known build warning
    - commit: d080716
    - review: clean — 0 findings (review sha HEAD~1..HEAD)
  timestamp: 2026-09-25T22:33:06.884421+00:00
position_column: done
position_ordinal: ffaf80
title: SpeculativeTokenIterator.prepare drops the first main token when a model prepare returns .logits
---
## What

Found during ^k8hm41j. In `SpeculativeTokenIterator.prepare(input:prefill:)` (`Libraries/MLXLMCommon/Evaluate.swift`), the `.logits` branch of the main prefill samples a token into `y`, and the `.logits` branch of the draft prefill samples the draft's own token into `draftY`.

Facts:
- `y` is fed to the verify pass, but it is never put into `pendingTokens`, thus the first main token is not emitted.
- `draftY` is the draft sample, not `y`. The draft cache then holds a different token at that position than the main cache, and the drafts continue from the draft's own token.
- `Gemma3TextModel.prepare` returns `.tokens`, thus the Gemma3 tests do not reach this branch.

Find if a model that returns `.logits` from `prepare` can reach `SpeculativeTokenIterator`. If it can, emit `y`, and set `draftY` to `y`.

## Acceptance Criteria

- [x] A test with a model whose `prepare` returns `.logits` shows that speculative decoding emits the same tokens as `TokenIterator` at temperature 0, or the task records why the branch is not reachable.