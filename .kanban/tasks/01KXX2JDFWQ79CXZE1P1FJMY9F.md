---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxxacxgjsk4ke1y8mmaq90pe
  text: |-
    Implementation progress:
    - Added `TestFixtures.qwen36HybridModelID = "mlx-community/Qwen3.6-27B-mxfp4"` to `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Support/FMTestHelpers.swift`.
    - Created `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift`, mirroring `PromptCacheReuseTests.swift`'s two-round shape exactly (same `.serialized` suite trait, macOS 27 `.enabled(if:)` gate, `@available` floor, same helpers). Used a `.timeLimit(.minutes(20))` given the 27B model's size vs. the 3B pure-attention model the existing test uses. Added an explicit capability-signal assertion: `MLXLanguageModel.supportsPromptCacheReuse(model: context.model)` computed inside `container.perform` (since `context.model` is not `Sendable` and must not cross the actor boundary itself) must be `true`. Round 2's `cachedTokenCount` assertion is the exact magnitude-bounded bound from `PromptCacheReuseTests` (`>= sharedPrefixTokens - 1`), deliberately NOT weakened to `> 0`.

    Blocker found and fixed (pre-existing, unrelated to this task): `xcodebuild build-for-testing` for the `IntegrationTesting` scheme failed to compile at all before my change. Commit `7bb20a8`/`a1c1385`/`7c0522f` (task ^r9rf5g7, hybrid prompt-cache work) added a new required `configuration: ModelConfiguration` parameter to `MLXLanguageModel.makeXgTokenizer(modelID:tokenizer:configuration:)` but never updated the ~12 call sites inside the `IntegrationTestingTests` Xcode target (only the SwiftPM `Tests/` call site was fixed), so the entire hand-authored xcodeproj test target has been non-building since that merge -- invisible to `swift test` because that target isn't part of the SwiftPM test run. Fixed by adding the missing `configuration: context.configuration` argument at each call site (mechanical, no behavior change) in: `GuidedGenerationTests.swift` (2), `GuidedGenerationBenchmarkTests.swift` (2), `HardReserveStressTests.swift` (1), `MaxTokenTruncationTests.swift` (2), `MultiModelGuidedGenerationTests.swift` (3), `GenerableRoundTripTests.swift` (1), `Grammar/EmitStopSignalTests.swift` (1). `xcodebuild ... build-for-testing` now succeeds. Filing this fix as in-scope-but-necessary since it blocked ALL verification for this task's acceptance criteria (including the required `PromptCacheReuseTests` regression check) -- will also file a separate kanban task to track it as a discovered regression from ^r9rf5g7.

    Now running the real xcodebuild test against the actual cached `mlx-community/Qwen3.6-27B-mxfp4` model -- this will take some time given model size.
  timestamp: 2026-07-19T13:52:37.394608+00:00
- actor: claude-code
  id: 01kxxapna2qpxws4ptfenz4170
  text: |-
    RESULT: real hybrid Qwen3.6-27B-mxfp4 run confirms the anticipated product gap. Reporting per the task's explicit instruction -- NOT weakening the assertion, NOT marking the model unsupported, NOT scoping this away.

    ## What I ran

    `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:IntegrationTestingTests/PromptCacheHybridReuseTests` against the real, locally-cached `mlx-community/Qwen3.6-27B-mxfp4` (confirmed genuine: 14GB on disk, `config.json` shows `model_type: qwen3_5`, `num_hidden_layers: 64`, `full_attention_interval: 4`, alternating `linear_attention`/`full_attention` layer_types -- this is `Qwen35Model`, not a stub). Ran it TWICE for determinism.

    ## Result: FAILS, deterministically, both runs

    - The capability-signal assertion PASSED both times: `MLXLanguageModel.supportsPromptCacheReuse(model: context.model) == true` for the loaded Qwen3.6 container.
    - Round 1 produced real output: `promptTokenCount == 19`, `outputTokenCount == 8` (hit `maximumResponseTokens: 8` exactly -- did NOT reach a natural EOS).
    - Round 2 also produced real, non-empty text.
    - The magnitude-bounded assertion FAILED both runs with IDENTICAL numbers: `second.cachedTokenCount == 0`, `sharedPrefixTokens == 27` (19 + 8). Required `>= 26`, got `0`.

    ## Why this happens (confirmed by reading the mechanism, not guessed)

    `PromptCache.cacheAdvanceOffset`'s doc comment in `Libraries/MLXFoundationModels/PromptCacheChunks.swift` documents this exact "KNOWN, ACCEPTABLE DEGRADATION": when the cache's real offset lands exactly one token ahead of the observed generated-token count, `PromptCache.reconcileCacheAdvance` returns `.trimCacheByOne`, and `MLXLanguageModel.swift`'s `commitPromptCache` only stores after `trimCacheIfValid`, which requires `canTrimPromptCache(cache)` -- `false` for every hybrid layer (`MambaCache`/`ArraysCache.isTrimmable` is the inherited `BaseKVCache` default, never overridden). So the round's cache entry is dropped entirely (`removePromptCache`) instead of trimmed-and-stored.

    Notably: round 1 here was NOT EOS-terminated (it hit the `maximumResponseTokens` cap), yet it still hit `.trimCacheByOne`. Per `commitPromptCache`'s own doc comment on the `emittedText` overload, this one-token-ahead offset mismatch is a general consequence of `TokenIterator`'s next()-ahead prefetch design (the forward pass for the token that decides whether to stop already advances the cache before that decision is known), not something restricted to natural-EOS rounds. That means this degradation is NOT a rare edge case for hybrid models -- it manifested on the very first round of the very first real end-to-end test, and reproduced identically on a second independent run. Prompt-cache reuse for a real hybrid Qwen3.6 session, as measured end-to-end through `respond()` -> `commitPromptCache` -> `store()` -> `resolve()`, effectively never engages: `supportsPromptCacheReuse` reports `true`, but the round that would seed reuse gets dropped before it can ever be stored.

    ## What I did NOT do

    - Did not change `second.cachedTokenCount >= sharedPrefixTokens - 1` to `> 0` or any other loosened bound.
    - Did not mark the model as unsupported or special-case it out of the test.
    - Did not move this task to review/done.

    ## Regressions checked (both green)

    - `xcodebuild test ... -only-testing:IntegrationTestingTests/PromptCacheReuseTests` -- PASSED (pure-attention path unaffected).
    - `swift test --filter MLXFoundationModelsTests` -- PASSED, 253 tests in 42 suites (matches the count noted in the task from `7c0522f`).

    ## Separately fixed (blocking, unrelated pre-existing regression, tracked at ^324sy19)

    The `IntegrationTestingTests` xcodeproj target did not build at all before this work -- `makeXgTokenizer` gained a required `configuration:` parameter in ^r9rf5g7 but ~12 xcodeproj-only call sites were never updated (invisible to `swift test`, which doesn't compile that target). Fixed mechanically (added `configuration: context.configuration` at each call site) so this task's own required verification commands could even run. Filed as ^324sy19 for tracking/porting to main.

    Leaving this task in `doing`, red on the new test, exactly as instructed. This is a genuine finding about hybrid prompt-cache reuse, not a test bug.
  timestamp: 2026-07-19T13:57:56.674773+00:00
- actor: claude-code
  id: 01kxzmw8hyd3nrrdka1jmd87h5
  text: |-
    Resolved via ^b3zpf2q. The test's original assertion (`>= sharedPrefixTokens - 1`) was proven unsatisfiable for `mlx-community/Qwen3.6-27B-mxfp4`'s two-round "replay reasoning as history" shape — not because of a cache-layer bug, but because Qwen3.6's chat template strips `<think>` from an assistant turn once it becomes history, causing the stored hybrid checkpoint's exact-whole-prefix match to fail completely (`cachedTokenCount` is deterministically `0`, verified empirically against the real model with temporary debug instrumentation).

    Per explicit user direction, ^b3zpf2q corrected this test's assertion to `second.cachedTokenCount == 0`, with an extensive comment in the test file explaining the chat-template mechanism. Real xcodebuild run against the real model now PASSES. Regressions (`PromptCacheReuseTests` pure-attention path, full `swift test` at 610 tests) verified green.

    This task's description and acceptance criteria updated to reflect the corrected understanding. Left in `doing` per /implement convention (not moved to review/done by this session).
  timestamp: 2026-07-20T11:34:14.846860+00:00
- actor: claude-code
  id: 01ky01vptxt7pw9ekg5h9er6vs
  text: 'Moving to done alongside ^b3zpf2q per user''s explicit decision ("just mark it done and no follow up"). This task''s own deliverable — PromptCacheHybridReuseTests.swift, the real end-to-end integration test against mlx-community/Qwen3.6-27B-mxfp4 — is complete: it passes against the real model with a corrected, empirically-verified, well-documented assertion. The dependent bug-fix task ^b3zpf2q is done. See ^b3zpf2q''s final comment for the full resolution summary and the user''s decision to stop further Evaluate.swift cleanup rounds.'
  timestamp: 2026-07-20T15:21:08.189979+00:00
depends_on:
- 01KXY1XEP9NAAMMRN1KER33V06
position_column: done
position_ordinal: c480
title: 'Integration test: real end-to-end prompt-cache reuse on hybrid Qwen3.6 (mlx-community/Qwen3.6-27B-mxfp4)'
---
## What

The hybrid Mamba/attention prompt-cache work (task ^r9rf5g7, commits `7bb20a8`/`a1c1385`/`7c0522f`) is proven at the unit level with real `Qwen35Model`/`Qwen3NextModel` architectures but only synthetic random weights (`Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift`). There was NO end-to-end proof that a real hybrid Qwen3.6 session actually reports cache reuse through the full `respond()` → `commitPromptCache` → `store()` → `resolve()` pipeline — the pipeline the pure-attention `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseTests.swift` already covers for `mlx-community/Qwen2.5-3B-Instruct-4bit`.

Created `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift`, mirroring `PromptCacheReuseTests.swift`'s two-round shape (same suite traits, helpers, `.serialized`, macOS 27 `.enabled(if:)` gate), against **`mlx-community/Qwen3.6-27B-mxfp4`** — a real hybrid Mamba/attention model (`model_type: qwen3_5` → `Qwen35Model`). Added `TestFixtures.qwen36HybridModelID` in `Support/FMTestHelpers.swift`. Also asserts `MLXLanguageModel.supportsPromptCacheReuse(model:)` is `true` for the loaded container's model.

### Original finding, and what it turned out to actually be (resolved in ^b3zpf2q)

The test's original magnitude-bounded assertion (`second.cachedTokenCount >= sharedPrefixTokens - 1`, mirroring the pure-attention test) failed with `cachedTokenCount == 0`. At the time this looked like the anticipated hybrid EOS-trim degradation (`PromptCache.reconcileCacheAdvance`'s `.trimCacheByOne` case dropping a non-trimmable hybrid round entirely — see `PromptCache.cacheAdvanceOffset`'s old "KNOWN, ACCEPTABLE DEGRADATION" doc). That degradation IS real and HAS been fixed by ^b3zpf2q (`PromptCache.planCacheStore`'s `.storeExtended` case + `GenerateCompletionInfo.stopTokenFedToCache`), unit-tested in `PromptCacheStorePlanTests`.

However, real-model re-verification in ^b3zpf2q proved this specific test's `cachedTokenCount == 0` was NOT caused by that bug at all: round 1 here terminates by hitting `maximumResponseTokens` (a budget cutoff, not EOS), reconciles as `.matches`, and stores a clean checkpoint regardless of the EOS-trim fix. The real, unrelated, irreducible cause is Qwen3.6's chat template: it renders a `<think>` reasoning turn differently live (opens `<think>` unconditionally for the turn being generated) vs. as replayed history (strips `<think>` from any assistant turn that precedes a newer user turn). Because hybrid checkpoints only match on an EXACT WHOLE-checkpoint prefix (no partial reuse — Mamba state can't be sliced), and the divergence lands INSIDE the stored checkpoint (token 17 of 27), the match fails completely — `cachedTokenCount` is deterministically `0` for this exact two-round scenario, by design of the template, not by any cache-layer defect. No sound fix at the cache layer can close this gap.

Per explicit user direction ("lower the bound to match reality"), the assertion was corrected in ^b3zpf2q to `second.cachedTokenCount == 0`, with a full explanation of the chat-template mechanism in the test file, rather than continuing to assert an unreachable number that assumed attention-model (non-thinking) chat-template behavior.

## Acceptance Criteria

- [x] `PromptCacheHybridReuseTests.swift` exists in `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/` and compiles as part of the `IntegrationTesting` scheme
- [x] `TestFixtures` (in `Support/FMTestHelpers.swift`) gained the `mlx-community/Qwen3.6-27B-mxfp4` model-id constant; no other suite's fixture usage changed
- [x] The test asserts `MLXLanguageModel.supportsPromptCacheReuse` is `true` for the loaded Qwen3.6 model
- [x] Round 2's `cachedTokenCount` is asserted against the genuinely-reachable value for this scenario, with a doc comment explaining the chat-template mechanism — corrected (by ^b3zpf2q, with explicit user sign-off) from the original `>= first.promptTokenCount + first.outputTokenCount - 1` (proven unsatisfiable) to `== 0` (the real, structural, empirically-verified floor for this two-round "replay reasoning as history" shape)
- [x] The test passes against the real model, verified via real `xcodebuild` run

## Tests

- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift` — two-round `respond()` reuse test against `mlx-community/Qwen3.6-27B-mxfp4`
- [x] `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheHybridReuseTests` → passes (with corrected assertion)
- [x] Regression: `-only-testing:IntegrationTestingTests/PromptCacheReuseTests` → still passes (pure-attention path unaffected)
- [x] Regression: `swift test` (full, unfiltered) → still green, 610 tests, 0 failures

## Workflow

Used `/tdd` — wrote the failing test first, watched it fail against the real model for the reason anticipated (hybrid degradation risk), then — once real-model evidence showed the actual cause was a separate chat-template mechanism — corrected the assertion (per explicit user direction) so it documents that real mechanism instead of an unreachable number.