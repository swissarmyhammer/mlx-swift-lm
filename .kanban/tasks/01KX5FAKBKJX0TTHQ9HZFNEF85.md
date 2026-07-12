---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxa91zy0dt0hj7z473wq8fsq
  text: |-
    Documentation sweep complete on Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift.

    Fixed 16 functions (relocated by name/signature since line numbers had shifted):
    - prefillAndGetLogits: added Parameters/Returns/Throws
    - applyBiasAndSample: rewrote Parameters directly (7 params), no longer cross-referenced to run()
    - emitToken: added Parameters
    - processFastForwardTokens: rewrote cross-referenced Parameters group directly, added Throws
    - cacheOrNil: added Parameter/Returns
    - feedTokenThroughModel: added Parameters/Returns (this is the function w7nvvg8 touched; it wasn't one of the 16 original findings, but was still missing formal blocks, and the acceptance criterion is zero remaining missing-doc findings for the whole file, so I documented it too)
    - advanceSingleSampledToken: rewrote Parameters directly (10 params), added Throws
    - logMaskSnapshot, logStopReason, logProgress: added Parameters
    - durationToMilliseconds: added Parameter/Returns
    - updateMaskAfterForwardPass: rewrote cross-referenced Parameters group directly, added Throws
    - advanceMaskOnCpuWhileGpuRuns, computeMaskAndArray: added Throws only (Parameters/Returns were already fully individual)
    - buildStopTokenIDs, buildMaskArray, bitmaskToMLXArray: added Parameters/Returns

    hardZoneBias, processFastForwardTokens's core params, and applyMaskAndSample/run were already fully compliant and left untouched where already correct.

    Test file finding: added a single additive `- Throws:` doc comment above FastForwardSampledTokenKVCacheTests.swift's `sampledTokenIsFedThroughModelWhenFastForwardFires()` test function. No other test restructuring.

    Verification:
    - `swift build`: clean.
    - `swift test --filter MLXGuidedGenerationTests`: no assertion failures; process aborts partway through on a pre-existing "Failed to load the default metallib" crash that also reproduces identically on stashed/original code (confirmed via git stash) -- this is the known infra gap tracked separately as kanban d7g4ty4 ("Apply the swift-test metallib bootstrap fix to MLXLMTests and MLXGuidedGenerationTests"), not something introduced by this doc-only change.
    - `swift test --filter MLXFoundationModelsTests`: 207/207 pass.
    - `review working` (swift + missing-docs validators): 0 missing-doc findings for this file. Two unrelated pre-existing findings surfaced (run()'s stale prematureEOS throws-doc claim, and a repeated `32` literal in bitmaskToMLXArray) -- both are pre-existing code/doc issues I did not introduce and are out of scope for this doc-only task; left as-is per scope.
  timestamp: 2026-07-12T04:23:36.640622+00:00
- actor: claude-code
  id: 01kxa970gg35mz0e1pgxt9akmh
  text: |-
    Adversarial double-check (double-check agent): PASS, no findings. Confirmed every diff line is a `///` doc-comment (no production code/signature/control-flow changes), every new Parameters/Returns/Throws block matches the actual function signature, the Throws chain traces correctly to GrammarError.maskComputationFailed, and only the two expected source files (plus the two kanban task files) changed.

    Leaving task in `doing` for `/review` per the implement skill's contract (implement doesn't move tasks to review).
  timestamp: 2026-07-12T04:26:21.072440+00:00
position_column: doing
position_ordinal: '80'
title: Add full Parameters/Returns/Throws doc blocks across GuidedGenerationLoop.swift's pre-existing functions
---
## What
Surfaced by review pressure on `w7nvvg8`'s KV-cache correctness fix, but confirmed genuinely pre-existing — `git diff HEAD~1..HEAD` on the reviewed commit (`3506e85`) shows the ONLY function added/changed was `feedTokenThroughModel` (which already has a substantive multi-paragraph doc comment, just not in the exact formal `- Parameters:`/`- Returns:`/`- Throws:` block format this validator wants). All 16 findings are on OTHER, untouched, pre-existing functions.

Mirrors the same class of debt already tracked for `MLXLanguageModel.swift` (`9jtbtkd`, done) and `Chat.swift` (`2yyn7f7`, done) — a whole-file `///` doc-comment sweep is needed for `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`.

## Review Findings (2026-07-10 02:25) — full list to address

- [ ] `prefillAndGetLogits` (~line 177, 3 params, throws, non-Void return) — needs `- Parameters:`, `- Returns:`, `- Throws:`.
- [ ] Function at ~line 330 (throws) — needs `- Throws:`.
- [ ] Function at ~line 351 (3 params, non-Void return, likely `advanceSingleSampledToken` or similar) — needs `- Parameters:` (has `- Returns:` already).
- [ ] Function at ~line 409 (non-Void return) — needs `- Returns:`.
- [ ] Function at ~line 498 (4 params, non-Void return) — needs `- Parameters:` and `- Returns:`.
- [ ] `applyBiasAndSample` (~line 521, 9 params, throws) — currently uses vague cross-references ("see run's parameter of the same name") instead of documenting each parameter directly; needs a proper `- Parameters:` block listing all 9 individually, plus `- Throws:`.
- [ ] Function at ~line 568 (2 params, likely `hardZoneBias`) — needs `- Parameters:` for `state`/`vocabSize` (or its actual params).
- [ ] Function at ~line 584 (2 params) — needs `- Parameters:`.
- [ ] Function at ~line 595 (3 params, likely `logProgress` or similar) — needs `- Parameters:` for `state`/`clock`/`startInstant`.
- [ ] Function at ~line 605 (non-Void return, likely `durationToMilliseconds`) — needs `- Returns:`.
- [ ] Function at ~line 620 (throws, likely `computeMaskAndArray`) — needs `- Throws:`.
- [ ] Function at ~line 647 (throws) — needs `- Throws:`.
- [ ] Function at ~line 685 (throws, likely `advanceMaskOnCpuWhileGpuRuns`) — needs `- Throws:`.
- [ ] `buildStopTokenIds` (~line 753, 2 params, non-Void return) — needs `- Parameters:` for `tokenizer`/`configuration` and `- Returns:`.
- [ ] `buildMaskArray` or similar (~line 810, 3 params, non-Void return) — needs `- Parameters:` for `maskPtr`/`maskBitCount`/`totalCount` and `- Returns:`.
- [ ] `Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift:126` — throwing test function missing `- Throws:`. (Per the review skill's test-refactor exception, adding a doc comment to a test may be acceptable since it's additive documentation, not restructuring — but confirm this test predates or postdates recent commits before deciding whether it's in scope here or should be dropped as pre-existing test code.)

Line numbers are from the review pass at commit `3506e85` — the file will have shifted by the time this task is picked up; re-locate each function by name/signature rather than trusting exact line numbers.

## Acceptance Criteria
- [ ] Every function above gets full `- Parameters:`/`- Returns:`/`- Throws:` sections as applicable, matching the style already used elsewhere in the file (e.g. `hardZoneBias`, `processFastForwardTokens` after `w7nvvg8`'s fix).
- [ ] `applyBiasAndSample`'s 9-parameter block specifically needs each parameter documented directly, not cross-referenced to `run()`'s parameters of the same name.
- [ ] No behavior change — documentation only.
- [ ] Build clean, full test suite green (MLXGuidedGenerationTests, MLXFoundationModelsTests).
- [ ] A local review pass (`review sha` scoped to the commit) confirms zero remaining missing-doc findings for this file.

## Scope
`Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift` (and the one test-file finding, if in scope). Not urgent/blocking — pre-existing documentation debt, not a correctness bug.