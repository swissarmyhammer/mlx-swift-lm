---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'GuidedGeneration: completion-reserve policy lets trivial schemas run unbiased for most of the token budget, allowing runaway repetition'
---
## What
Root-caused while investigating kanban 68bc1rt (`MultiModelGuidedGenerationTests.intRoundTrip(modelID: gemma-3-270m-it-4bit)` DecodingError.dataCorrupted).

`Executor.ConstraintSetup.reserves(forMaxTokens:)` in `Libraries/MLXFoundationModels/MLXLanguageModel.swift`:
```swift
func reserves(forMaxTokens maxTokens: Int) -> (completionReserve: Int, hardReserve: Int) {
    (Swift.max(structuralReserve * 3, maxTokens / 4), structuralReserve * 8)
}
```
`completionReserve` determines how many tokens before `maxTokens` the soft-zone closing bias (`GuidedGenerationLoop`'s `applyBiasAndSample`, +200 EOS / +100 closing-tier bias) starts nudging generation toward completion. The soft zone begins at token index `maxTokens - completionReserve`.

For a schema whose minimal JSON skeleton is tiny (e.g. bare `Int` -> `CompletionReserve.estimate` returns `"0"` tokenized, ~1-3 tokens), `structuralReserve * 3` is negligible, so `max(...)` always picks the `maxTokens / 4` floor. With the Executor's `defaultMaxTokens = 4096`, that floor is `1024`, so the soft zone doesn't begin until token `3072` -- 75% of the ENTIRE budget runs with zero completion bias, regardless of how trivially small the schema actually is.

Reproduced directly: `mlx-community/gemma-3-270m-it-4bit` asked "What is 2+2? Reply with just the number." under an `Int` grammar constraint. The grammar makes `EOS` legal after every single digit (since `"2"`, `"22"`, `"222"`, ... are all valid, complete JSON integers), but the model's own raw preference (a known small-model repetition-degeneration failure mode) kept greedily selecting the digit `2` over `EOS` for the entire unbiased normal zone. At token 3072 the soft-zone `+200 EOS` bias FINALLY tipped the decision to `EOS` on the very first biased sample -- but by then the accumulated output was 3072 copies of `"2"`, a JSON-valid but astronomically large integer literal that overflows Swift's `Int` and fails `JSONDecoder` with `dataCorrupted`.

This is a genuine adapter-policy gap, not a grammar-correctness bug (the grammar mask worked exactly as designed) and not pure "model incapability" (the bias mechanism that's supposed to prevent exactly this only engages far too late for trivial schemas). Confirmed via a real, deterministic (greedy/temp=0) run -- not a flake.

## Why not fixed directly under 68bc1rt
`ConstraintSetup.reserves` is the single call site (`Executor.runGuidedGenerationLoop` and `reasoningAdjustedMaxTokens`) driving reserve sizing for EVERY guided-generation and tool-calling request that goes through `MLXLanguageModel.Executor.respond` -- i.e. most of `Tests/MLXFoundationModelsTests` and `IntegrationTesting/.../GuidedGeneration/*` (GenerableRoundTripTests, GuidedGenerationIntegrationTests, GuidedGenerationUsageTests, PrewarmGrammarTests, StopTokenRegressionIntegrationTests, MultiModelGuidedGenerationTests, tool-calling phase-2 budgeting, etc.). `HardReserveStressTests.swift` is NOT affected (it recomputes the same formula locally rather than calling `ConstraintSetup.reserves`), but the rest of the suite routes through this one function. Retuning it safely requires designing a new policy (e.g. anchoring the "free normal-zone" runway to schema complexity rather than a flat fraction of `maxTokens`, without regressing the deliberately-generous behavior `HardReserveStressTests`-style large/nested schemas rely on) AND rerunning the full real-model guided-generation suite to confirm no regressions -- a properly-scoped, separately-budgeted task, not a drive-by fix bundled into 68bc1rt's checkpoint-hygiene/4-test-triage scope.

## Suggested direction
Make the "unbiased normal zone" length scale with `structuralReserve` (the schema's actual minimal-content need) with a modest cap, instead of an unconditional `maxTokens / 4` floor that dominates for trivial schemas regardless of `maxTokens` size. Needs empirical tuning + full regression pass across the guided-generation integration suite (real models, real generations) before landing.

## Acceptance Criteria
- [ ] New reserve-sizing policy designed and documented (why it doesn't regress large/nested-schema behavior)
- [ ] `MultiModelGuidedGenerationTests.intRoundTripGemma3_270m` (currently `.disabled`, added under 68bc1rt) re-enabled and passing
- [ ] Full `GuidedGeneration`-suite real-model regression run attached/referenced, zero unexplained failures
