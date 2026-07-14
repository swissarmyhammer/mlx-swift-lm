---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxg9ay9w2d9wt65s6mr159kg
  text: |-
    Root cause found via TDD (unit-level repro first, per the task's workflow instruction):

    `Libraries/MLXGuidedGeneration/CompletionReserve.swift`'s `synthesizeMinimalJSON` never handled the JSON Schema `"const"` keyword. `SchemaConverter.encodeToolCallingEnvelopeJSON` names every `oneOf` alternative's tool via `{"name": {"const": "<tool name>"}}` -- not `enum`/`type`. Since `const` wasn't recognized, synthesis of that required `name` property returned `nil`, which propagated up through the `object` and `oneOf` cases, so the WHOLE tool-calling envelope's `structuralReserve` silently fell back to the 64-token default regardless of the schema's real (much smaller) minimal-JSON size -- verified empirically with a probe unit test before touching any code (`CompletionReserve.estimate` returned exactly the injected `defaultReserve` for the real 2-tool envelope shape).

    With `structuralReserve` stuck at 64, `hardReserve = structuralReserve * 8 = 512` exceeded `maxTokens = 256` (the tool round's `maximumResponseTokens`) outright. `GuidedGenerationLoop.applyBiasAndSample`'s hard-zone check (`tokenCount >= maxTokens - hardReserve`) was then true from token 0, forcing the ENTIRE 256-token budget through the hard-closing bias. That bias still permits single-digit tokens (`ClosingTokenBias` treats `0`-`9` as closing-tier, needed so numeric fields' own digits aren't suppressed) -- so `mlx_final_answer`'s open-ended `response` string field could ramble in digits for the full budget without ever selecting the closing quote. This exactly matches the observed failure signature (digit-spam to `done tokens=256`, `toolCallName == nil`).

    Note: this const-handling gap predates ad4c4cf and is orthogonal to the schema-complexity-scaling formula itself -- ad4c4cf's formula change didn't cause it, but it did NOT protect against it either (both old and new formulas compute `hardReserve = structuralReserve * 8` identically, so `hardReserve` could already exceed `maxTokens` before ad4c4cf too, given this fallback). I could not fully reconcile that with the bisection's "ad4c4cf^ passes 2/2" claim through static analysis alone (the zone-priority logic in `GuidedGenerationLoop.applyBiasAndSample` makes the hard zone dominate unconditionally once `hardReserve > maxTokens`, independent of `completionReserve`'s value, so old and new formulas should behave identically for this schema under that reading) -- flagging this as a residual open question in case it matters, but did not spend further unbounded time on it per the task's explicit anti-hang guidance, since the fix is verified correct and sufficient against the real model regardless.

    Reconciled fix (two parts, per the task's explicit ask -- verified BOTH are needed, not just one):
    1. `CompletionReserve.swift`: added a `const` case to `synthesizeMinimalJSON` (checked before `enum`), so `const`-typed properties synthesize their real minimal JSON.
    2. `MLXLanguageModel.swift`'s `ConstraintSetup.reserves(forMaxTokens:)`: capped `hardReserve` at `maxTokens / 2` (`Swift.min(structuralReserve * 8, maxTokens / 2)`). Empirically confirmed fix #1 ALONE is insufficient: with only the const fix, the real envelope's `structuralReserve` becomes ~55, giving `hardReserve = 440`, which still exceeds `maxTokens = 256` without the cap. The cap guarantees the hard zone can never swallow the whole budget regardless of how `structuralReserve` was derived (a legitimately large schema, or any future estimation gap), while `completionReserve`'s existing `Swift.max(_, hardReserve)` still keeps `completionReserve >= hardReserve` always, preserving t3nynaj's zone-ordering invariant.

    Tests added (TDD RED confirmed before the fix, GREEN after):
    - `Tests/MLXGuidedGenerationTests/CompletionReserveTests.swift`: `objectWithRequiredConstProperty`, `toolCallingEnvelopeSchemaSynthesizesMinimalJSON` (real envelope shape).
    - `Tests/MLXFoundationModelsTests/ToolEnvelopeReserveZoneTests.swift` (new file): `toolEnvelopeHardZoneLeavesRealRunway` (builds a real `ConstraintSetup` via a byte-fallback `GrammarTokenizer`/cheap `GrammarConstraint`, asserts hard/completion reserve stay under budget and `hardReserve <= maxTokens/2`), `hardReserveCapHoldsEvenForAnInflatedStructuralReserve` (hardcodes `structuralReserve: 64` to simulate the old fallback directly, asserts `hardReserve == 128` not 512). Confirmed both would fail without the respective fix (temporarily reverted the cap and re-ran to confirm RED).

    Verification (real evidence, all fresh-run this session):
    - `swift test` (all local targets): 254+65+209+7 = 535 tests, 0 failures.
    - Real-model xcodebuild integration: `PromptCacheToolReasoningReuseTests` 2/2 pass (`[GuidedGen] done tokens=20`, not 256). Full PromptCache integration set: 11 tests / 6 suites pass (PromptCacheMultimodalBoundaryTests skipped by its own gate, not a failure -- matches acceptance criteria's "6 suites, 11 tests" exactly).
    - `MultiModelGuidedGenerationTests.intRoundTripGemma3270m` (t3nynaj's own regression test) still passes: produced `"222222"` (6 digits), consistent with t3nynaj's documented single-digit-magnitude convergence expectation -- confirms the new cap doesn't reintroduce t3nynaj's bug.

    Adversarial double-check agent launched for a second pair of eyes on the diff; will report its verdict next.
  timestamp: 2026-07-14T12:23:56.476151+00:00
- actor: claude-code
  id: 01kxgb5fmgn611rypw6wwaktsc
  text: |-
    First double-check pass found a real edge case: capping `hardReserve` at `maxTokens / 2` (same ceiling as `normalZoneLength`) let the two collide and collapse the soft zone to zero width for the tool-envelope schema (`completionReserve == hardReserve == 128` for `maxTokens=256`, `structuralReserve=55`). Fixed by capping `hardReserve` at `maxTokens / 4` instead (a strictly tighter ceiling than `normalZoneLength`'s `maxTokens / 2`), which guarantees `completionReserve - hardReserve >= maxTokens / 4` whenever the caps bind. Added `completionReserve > hardReserve` assertions to both new tests in `ToolEnvelopeReserveZoneTests.swift`, confirmed RED before this correction and GREEN after (re-ran `swift test`: 535 tests green).

    Also discovered a SEPARATE, unrelated intermittent flake while re-verifying against the real model: `PromptCacheToolReasoningReuseTests.toolCallingRoundReusesPromptCacheOnRoundTwo`'s `toolOptions` didn't force `.greedy` sampling, so it ran at the framework's default non-zero temperature and the model would nondeterministically (~40% of runs in my sample) answer in plain prose instead of invoking any tool at all -- a completely different failure mode from the reserve-sizing regression (clean short completion at ~40 tokens, no digit-spam, no full-budget consumption). Confirmed this is NOT caused by my reserve fix: (a) it also reproduces on the original unfixed code (git stash tested), and (b) 40 tokens is far from any zone boundary my fix touches (hard zone starts at token 192 with the corrected formula). Fixed by adding `samplingMode: .greedy` to `toolOptions`, matching the sibling `reasoningRoundReusesPromptCacheOnRoundTwo` test's own established `greedyOptions` pattern in the same file. Verified 4/4 consecutive real-model runs now pass deterministically with `done tokens=20` every time.

    Re-ran full verification after both corrections: swift test (535 tests, 0 failures), full PromptCache integration set + t3nynaj's `intRoundTripGemma3270m` regression test (12 tests / 8 suite invocations, all green). Second double-check round launched on the corrected diff.
  timestamp: 2026-07-14T12:55:54.768736+00:00
- actor: claude-code
  id: 01kxgbejv4p87qwwjxrqz9j4y9
  text: |-
    Second double-check round returned PASS (regime analysis confirmed `completionReserve > hardReserve` holds across all structuralReserve/maxTokens combinations including small-maxTokens edge cases; confirmed the greedy-sampling test fix is legitimate and in-scope; confirmed no duplication/stray-code issues).

    Final fresh verification (this session, all green):
    - `swift test` (all local targets): 535 tests, 0 failures.
    - Real-model xcodebuild: full PromptCache integration set (11 tests / 6 suites, PromptCacheMultimodalBoundaryTests skipped by its own gate) + `PromptCacheToolReasoningReuseTests` 2/2 + t3nynaj's `intRoundTripGemma3270m` regression test = 12 tests / 8 suite invocations, all passing, run twice for confidence (once right after the reserve-formula correction, once again as final confirmation).
    - `PromptCacheToolReasoningReuseTests.toolCallingRoundReusesPromptCacheOnRoundTwo` specifically re-run 4 consecutive times after the greedy-sampling fix: 4/4 pass, `done tokens=20` every time (previously flaky ~40% at default temperature, for a reason unrelated to this task's reserve-sizing fix -- see prior comments).

    Summary of all changes (diff: 4 modified files + 1 new test file):
    1. `Libraries/MLXGuidedGeneration/CompletionReserve.swift` -- `synthesizeMinimalJSON` now handles JSON Schema `const` (root cause of the regression).
    2. `Libraries/MLXFoundationModels/MLXLanguageModel.swift` -- `ConstraintSetup.reserves(forMaxTokens:)` caps `hardReserve` at `maxTokens / 4` (defense-in-depth against any future reserve-estimation gap; deliberately a tighter cap than `normalZoneLength`'s existing `maxTokens / 2` to avoid a zero-width soft zone).
    3. `Tests/MLXGuidedGenerationTests/CompletionReserveTests.swift` -- two new unit tests for `const` handling.
    4. `Tests/MLXFoundationModelsTests/ToolEnvelopeReserveZoneTests.swift` (new) -- two unit tests pinning the tool-envelope schema's reserve-zone boundaries directly against `ConstraintSetup.reserves(forMaxTokens:)`.
    5. `IntegrationTesting/.../PromptCacheToolReasoningReuseTests.swift` -- `toolOptions` now forces `.greedy` sampling (separate, unrelated flake fix discovered during re-verification, matching the sibling test's own established pattern).

    All acceptance criteria met:
    - [x] PromptCacheToolReasoningReuseTests 2/2 pass at HEAD (real model, deterministic -- confirmed with greedy sampling now forced)
    - [x] Full PromptCache integration set (6 suites, 11 tests) passes at HEAD
    - [x] ad4c4cf's original review finding remains satisfied -- re-ran `intRoundTripGemma3270m`, still converges to single-digit-magnitude output ("222222")
    - [x] Unit regression test pins the envelope schema's reserve-zone boundary (`ToolEnvelopeReserveZoneTests.swift`)

    Work is complete and green. Leaving task in `doing` for `/review`.
  timestamp: 2026-07-14T13:00:52.964348+00:00
position_column: done
position_ordinal: b880
title: 'REGRESSION ad4c4cf: completion-reserve resizing breaks tool-calling round (guided envelope degenerates to digit-spam)'
---
## What
Deterministic regression, bisected (greedy sampling, reproducible): PromptCacheToolReasoningReuseTests' tool-calling round test fails at ad4c4cf ("fix(mlx-fm): size guided-generation completion-reserve zone off schema complexity") and at the upstream merge d1b5e62, but PASSES at ad4c4cf^ (2/2). The upstream merge is NOT the cause.

Failure signature (postmerge-run.log in the session scratchpad): round 1's guided tool-envelope phase produces `mlx_final_answer` with an unbounded digit string ("100000000...") and runs to the full 256-token budget ("[GuidedGen] done tokens=256"), the `<tool_call>` envelope leaks into responseText, toolCallName == nil — the test's `expected round 1 to produce a real tool call` #require fails. Hypothesis: the schema-complexity-based reserve sizing changed the budget-zone boundaries for the tool-calling envelope schema, so the grammar's completion-forcing zone no longer engages where it used to — the model rambles inside a string field until the budget kills the round (previously the hard-reserve zone forced JSON closure).

Repro: cd IntegrationTesting && xcodebuild test-without-building ... -only-testing:IntegrationTestingTests/PromptCacheToolReasoningReuseTests (single worker; build flags as in ^5ra1wzm). Verify the fix ALSO keeps ad4c4cf's original intent (whatever review finding it addressed — see task t3nynaj) satisfied: reserve sized off schema complexity may be right for large guided schemas but must not shrink the envelope's completion-forcing zone.

## Acceptance Criteria
- [ ] PromptCacheToolReasoningReuseTests 2/2 pass at HEAD (real model, deterministic)
- [ ] The full PromptCache integration set (6 suites, 11 tests) passes at HEAD
- [ ] ad4c4cf's original review finding remains satisfied (no revert-and-forget: reconcile both)
- [ ] A unit-level regression test pins the envelope schema's reserve-zone boundary (so this cannot silently regress again)

## Tests
- [ ] Unit regression test in the guided-generation suite for reserve sizing on the tool-envelope schema
- [ ] `swift test` all targets green; integration set green

## Workflow
- Use `/tdd` — reproduce at unit level first (reserve-zone boundary for the envelope schema), then fix.