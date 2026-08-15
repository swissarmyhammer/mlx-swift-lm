---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx8qgpq0txhvxqgks7a7a298
  text: Re-flagged (with shifted line numbers, ~2138) in a later review pass on commit de41a70, confirmed pre-existing/untouched by that commit — same function, same reasoning/suppression/guidedInput gate complexity already tracked here. No new scope, just confirming this task's relevance persists.
  timestamp: 2026-07-11T13:57:49.920439+00:00
- actor: claude-code
  id: 01kxakjbzax8g79k3438gyzebj
  text: |-
    Implemented. Extraction shape (after reading the actual 105+ line function): split into two helpers rather than the suggested single PromptVariants-with-eager-field shape, to preserve exact throw ordering.

    1. `RespondConfigResolution` struct (resolved config + contextLength) + `resolveRespondConfiguration(modelID:context:declaresReasoning:configurationResolver:) throws -> RespondConfigResolution` -- owns the config.json read/decode/resolve + validateReasoningCapability call (the "configuration-resolution" half).
    2. `PromptVariants` struct (suppressed/reasoningSetup/guided fields, deliberately NOT including the eager render) + `preparePromptVariants(request:messages:context:declaresReasoning:resolved:needsEagerInput:input:schemaJSON:) async throws -> PromptVariants` -- owns mayRunReasoningPath + the suppression/reasoning/guided-input gates (the "rendering" half).

    `prepareRespondSetup` itself now: renders the eager input directly (unchanged, must stay first), calls resolveRespondConfiguration, calls preparePromptVariants, computes effectiveInput = variants.suppressed ?? input, returns RespondSetup. ~85 lines -> ~55 lines, and each gate's condition now lives in exactly one place instead of being interleaved with the config-resolution chain.

    Kept eager render OUTSIDE the PromptVariants struct/method (unlike the task's suggested shape) specifically because it must execute strictly before config resolution to preserve the original throw order (eager render's error must still win over a later capability-validation error if a request would trigger both) -- moving it into a method gated on `resolved` would have reordered that.

    Verification: `swift build --target MLXFoundationModels` clean (only a pre-existing unrelated deprecation warning at line 843). Full `swift test`: 207 tests / 37 suites in MLXFoundationModelsTests all green, plus MLXGuidedGeneration's 7 tests green -- no failures anywhere. `git diff --stat` confirms only MLXLanguageModel.swift (plus kanban bookkeeping files) touched. Local `review working` pass: 0 findings. runToolCalling/executeThinkThenCallPhase1/executeToolCallingPhase2 untouched, confirmed out of scope.

    Adversarial double-check agent dispatched to verify no subtle behavior change before handoff.
  timestamp: 2026-07-12T07:27:19.018665+00:00
position_column: done
position_ordinal: b180
title: Reduce prepareRespondSetup's cognitive complexity (state-dependency across multiple prompt-rendering gates)
---
## What
Surfaced by review pressure on task `kr3zkap`'s prewarm commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — zero matches, untouched by that commit.

`Libraries/MLXFoundationModels/MLXLanguageModel.swift`'s `prepareRespondSetup` function is 105+ lines with multiple independent conditional branches (tooling gate, suppression gate, reasoning gate, guided-input gate) that each decide whether to re-render or re-tokenize the prompt. High cognitive load from state-dependency complexity even though each branch individually is ≤3 levels — readers must track which branches execute together, what they depend on, and whether their side effects conflict.

Suggested approach from the review: introduce a struct to bundle prompt-rendering decisions (e.g. `struct PromptVariants { let eager: LMInput?; let suppressed: LMInput?; let guided: LMInput? }`), compute all variants in a separate method (e.g. `prepareAllPromptVariants(...)`), then have `prepareRespondSetup` become: unpack variants → combine/select → return — separating rendering complexity from configuration-resolution logic.

## Acceptance Criteria
- [ ] Reduce `prepareRespondSetup`'s complexity via extraction (the suggested `PromptVariants` approach, or an equivalent the implementer judges cleaner after reading the actual current code — the suggestion is a starting point, not gospel).
- [ ] No behavior change — pure refactor.
- [ ] Build clean, full test suite green.
- [ ] A local review pass confirms zero remaining findings of this class.

## Scope
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only. Not urgent/blocking — pre-existing complexity debt, not a correctness bug.

## Note on a companion finding, REJECTED as stale
The same review round also flagged `runToolCalling` for supposedly needing extraction into `executeThinkThenCallPhase1`/`executeToolCallingPhase2` helper methods — this was rejected as a stale/hallucinated citation: both functions ALREADY EXIST as separate, already-extracted private methods in this file (confirmed via `grep -n "func executeThinkThenCallPhase1\|func executeToolCallingPhase2"`). If `runToolCalling` still has genuine residual complexity beyond what's already extracted, that would need a fresh, accurate assessment — not this stale finding's specific (already-done) suggestion.