---
comments:
- actor: claude-code
  id: 01m0j3n8yc4eznh0hrkjcy38xm
  text: |-
    ### research

    - `ExecutorPromptCachePlan.make` in `Libraries/MLXFoundationModels/ExecutorPromptCache.swift` holds the guard. No unit test calls `make` today; `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift` tests the key, the store, the slot and `committed(generatedTokens:)` only.
    - `MuseGlimmer.prepare(_:cache:state:prefill:)` (model side) accepts tokens of rank 1 or rank 2 and reads no text mask. The processor gives rank 2 tokens `[1, N]` and an `int8` all-ones mask of the same shape.
    - `MLXLanguageModel.Executor` feeds `plan?.input ?? input` to generation. A narrowed input must keep the rank and the mask presence of the input it came from, so the model sees the shape its processor makes.
    - `ScriptedLanguageModel` in `ScriptedModelTestSupport.swift` (same test target) is a `LanguageModel` with `kvHeads == []`, thus `newCache` gives `[]`. It serves as the `model` argument of `make` in unit tests.
    - Real-weights pattern: `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseChannelTests.swift` reads `cachedTokenCount` from the channel with `makeTestModel` + `makeMLXExecutor`. `mlx-community/Muse-Glimmer-30B-4bit` is in the local Hugging Face cache. The integration test is run with `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -only-testing:<Suite>/<test>`, one test at a time.
    - Review rules read: the swift rule set (casing, naming-clarity, optionals, idioms, access-control), missing-docs-swift, magic-numbers-swift, function-length-swift, dead-code-swift, duplication, reuse, test-integrity, completeness.
  timestamp: 2026-08-21T12:10:17.932730+00:00
- actor: claude-code
  id: 01m0j4ctgezgkqbdp84qcn1fet
  text: |-
    ### implementation landed

    What changed:
    - `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`: `ExecutorPromptCachePlan.make` now reads its ledger through `ledgerTokens(of:)`. A text-only input passes when its tokens hold one sequence (rank 1, or rank 2 with one row) and its mask, when present, has the shape of the tokens and holds no zero. Media, a batch of more than one row and a mask that holds a zero still give nil. `narrowed(_:to:)` gives the tail of the prompt back in the rank of the input, with an all-ones mask of the input's mask dtype when the input carried a mask, thus the model sees the shape its processor makes.
    - `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`: five unit tests under "Planning one pass over a processor's input". Two went red first (nil plan) and green after the change; three guard the refusals (mask with a zero, two rows, an image).
    - `IntegrationTesting/.../PromptCacheReuseChannelTests.swift`: real-weights test `aSecondTurnOfAVLMProcessorModelReusesThePromptOfItsFirstTurn` on `mlx-community/Muse-Glimmer-30B-4bit`. Measured: round 1 prompt 64, cached 0; round 2 prompt 75, cached 63, fed 12.

    What did not work at first:
    - The VLM test threw `This model always reasons; .reasoning must be declared at MLXLanguageModel init`. Muse-Glimmer is an always-reasoning model, thus the test builds it with `makeReasoningTestModel`. With 24 response tokens its first turn holds reasoning alone, the response text is empty and `TranscriptConverter` skips that entry on turn 2; the measurement of the prompt cache is not affected.
    - `xcodebuild -only-testing:` needs the Swift Testing name with parentheses: `.../PromptCacheReuseChannelTests/aSecondTurnOfAVLMProcessorModelReusesThePromptOfItsFirstTurn()`. Without them the run selects 0 tests.

    Open:
    - The last acceptance criterion (`FoundationModelsRouter`'s `secondTurnReusesFirstTurnsKVCache`) lives in `~/github/swissarmyhammer/FoundationModelsRouter`, which depends on this fork by URL on branch `stable`. It needs a pushed commit, which this step does not make. The box stays unchecked.
  timestamp: 2026-08-21T12:23:09.582696+00:00
- actor: claude-code
  id: 01m0j4ddqm70adfd9b8qk5t8pr
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseChannelTests.swift. Build: `xcodebuild build-for-testing -skipPackagePluginValidation -scheme mlx-swift-lm-Package -destination 'platform=macOS'` exit 0. Unit bundles with `xcrun xctest`: MLXFoundationModelsTests 167 passed (5 new, 0 failed); MLXLMTests 919 Swift Testing + 493 XCTest passed; MLXGuidedGenerationTests 70 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed. Integration: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -parallel-testing-enabled NO '-only-testing:IntegrationTestingTests/PromptCacheReuseChannelTests/aSecondTurnOfAVLMProcessorModelReusesThePromptOfItsFirstTurn()'` 1 passed: round 2 cached 63 of 75. swift-format run. Three of four acceptance boxes checked; the FoundationModelsRouter box stays open because it needs a pushed commit on `stable`.
    - next: /review
  timestamp: 2026-08-21T12:23:29.268652+00:00
- actor: claude-code
  id: 01m0j4t94a45qcjh0hzh9pdt1f
  text: |-
    ### test — green
    - command: `xcodebuild build-for-testing -skipPackagePluginValidation -scheme mlx-swift-lm-Package -destination 'platform=macOS'` — exit 0, 0 warnings, 0 errors. I touched `ExecutorPromptCache.swift` and `PromptCacheReuseChannelTests.swift` and built again to force a fresh compile of the changed file. The log shows the compile step and the build result. No warning line is in the log.
    - bundle results, each run with `xcrun xctest "$D/<Bundle>.xctest"`:
      - MLXFoundationModelsTests: 167 tests passed, 0 failed, 0 skipped.
      - MLXLMTests: 919 Swift Testing tests + 493 XCTest tests = 1412 tests passed, 0 failed, 0 skipped.
      - MLXGuidedGenerationTests: 70 tests passed, 0 failed, 0 skipped.
      - CXGrammarTests: 7 tests passed, 0 failed, 0 skipped.
      - MLXHuggingFaceMacrosTests: 5 tests passed, 0 failed, 0 skipped.
      - Total: 1661 tests passed, 0 failed, 0 skipped.
    - I did not run the IntegrationTesting real-weights suite. The implement step ran the one new real-weights test one time. A full run of that suite is not needed here.
    - No fix was needed. All bundles were green on the first run.
    - next: /review
  timestamp: 2026-08-21T12:30:30.538142+00:00
- actor: claude-code
  id: 01m0j4vkxfm9x1nf6gg62wf0tk
  text: |-
    ### commit — changed
    - evidence: 239b41e fix(foundationmodels): let the plan guard accept a batched VLM text-only input (7 files changed, 344 insertions(+), 11 deletions(-))
    - next: none. This is a local commit only. No push was done.
  timestamp: 2026-08-21T12:31:14.351321+00:00
- actor: claude-code
  id: 01m0j52z7r6f15rjjgcvpv1qsf
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (commit 239b41e). The engine reviewed 3 files and attempted 7 validator-file pairs: `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`, `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`, `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseChannelTests.swift`. 0 findings, 0 confirmed, 0 refuted, 0 files skipped. The description holds no prior `## Review Findings` section.
    - next: The task moved `doing → review → done`. One acceptance criterion stays unchecked in the description: `FoundationModelsRouter`'s `secondTurnReusesFirstTurnsKVCache` passes against the corrected revision. That check lives in the `FoundationModelsRouter` repository (card ^de1yq0p), not in this one.
  timestamp: 2026-08-21T12:35:15.320045+00:00
- actor: claude-code
  id: 01m0j53xvc187z165fkc8kf8ga
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 3 files (Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, IntegrationTesting/.../PromptCacheReuseChannelTests.swift)
    - test: green — xcodebuild build-for-testing exit 0, 0 warnings; five bundles 1661 passed, 0 failed, 0 skipped; real-weights VLM test 1 passed (round 2 cached 63 of 75)
    - commit: 239b41e (local only, not pushed)
    - review: clean — review sha HEAD~1..HEAD, 3 files, 7 validator-file pairs, 0 findings
    - column: done
    - open: acceptance box 4 (FoundationModelsRouter `secondTurnReusesFirstTurnsKVCache`) needs a pushed commit on `stable`; the push is the user's step
  timestamp: 2026-08-21T12:35:46.668616+00:00
position_column: done
position_ordinal: ff9280
title: 'The executor prompt cache never engages for a VLM-processor model: a batched, masked text-only input fails the plan guard'
---
The executor prompt cache never engages for a model that loads through `VLMModelFactory`, even on a turn that has no image. The turn thus processes the whole prompt again, and `usage.input.cachedTokenCount` stays 0.

## The cause

`ExecutorPromptCachePlan.make` in `Libraries/MLXFoundationModels/ExecutorPromptCache.swift` opens with this guard:

```swift
guard input.image == nil, input.video == nil, input.audio == nil,
    input.text.mask == nil, input.text.tokens.ndim == 1
else {
    return nil
}
```

`MLXLanguageModel.Executor` renders every prompt with `context.processor.prepare(input:)`. For a VLM the processor is the model's own, and it gives a batched, masked input even when the caller supplies no image. `MuseGlimmerProcessor.prepare(input:)` in `Libraries/MLXVLM/Models/MuseGlimmer.swift` shows the text-only branch:

```swift
guard !input.images.isEmpty else {
    let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
    let mask = ones(like: promptArray).asType(.int8)
    return LMInput(text: .init(tokens: promptArray, mask: mask))
}
```

`expandedDimensions(axis: 0)` makes `ndim == 2`, and the mask is not nil. The guard thus fails on two counts for every text-only turn of the model.

One cause gives two results:

- `make` gives nil, so `MLXLanguageModel.Executor` calls `generateProtocolTokensTask(input: plan?.input ?? input, cache: plan?.caches, ...)` with `cache: nil`. The turn builds a new cache and processes the full prompt again. The key/value cache is NOT reused between the turns of one session.
- `ExecutorPromptCacheSlot.plan` sets `reusedTokenCount` to 0, so the executor stamps `cachedTokenCount: 0` on the usage of every turn.

The mask this processor makes is all ones, and the batch holds one row, so the input carries nothing that a token ledger cannot describe. The guard refuses it only because of its shape.

## How it was measured

`FoundationModelsRouter` (its card ^de1yq0p) drives `mlx-community/Muse-Glimmer-30B-4bit` through `LanguageModelSession`. Its integration test `secondTurnReusesFirstTurnsKVCache` is red, and it repeats:

```
Expectation failed: turn2Usage.input.cachedTokenCount > 0
  turn2Usage.input.cachedTokenCount -> 0
cachedTokenCount (0) should approximate turn 1's total processed tokens (154) within 38
```

The checkpoint's `config.json` gives `model_type: muse_glimmer`, and its `processor_config.json` gives `processor_class: MuseGlimmerProcessor`. `VLMModelFactory` and `MuseGlimmerProcessor` are thus the path in use.

Revision under test: this fork, branch `stable`, `ba8ff43b9040ceec43c84f28637a250f33590633`. That revision carries `ExecutorPromptCache.swift`, so the prompt cache is present; it only never engages.

Wall clock is not proof by itself, and this card does not use it as proof. In the same run turn 1 took 7.40 s and turn 2 took 2.52 s, but turn 1 also holds the first-generation warm-up of the process.

## What to do

Let a text-only input through the guard when its shape carries no content a token ledger cannot describe:

- Accept `input.text.tokens.ndim == 2` when the first dimension is 1. Read the token ledger from that one row, and give the narrowed input back in the shape the model expects.
- Accept a mask that marks every token as present. Refuse a mask that holds a zero, because the ledger cannot describe padding.
- Keep the refusal for image, video and audio input, and for a batch of more than one row.

## Acceptance Criteria

- [x] `ExecutorPromptCachePlan.make` gives a plan for a text-only input that a VLM processor batched to one row and masked with all ones
- [x] A unit test proves that plan reuses the prefix, and that `make` still gives nil for a mask that holds a zero and for a batch of more than one row
- [x] An integration test on a real VLM checkpoint shows `usage.input.cachedTokenCount > 0` on turn 2 of one session
- [ ] `FoundationModelsRouter`'s `secondTurnReusesFirstTurnsKVCache` passes against the corrected revision #defect #performance