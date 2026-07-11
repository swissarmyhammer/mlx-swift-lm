---
assignees:
- claude-code
position_column: done
position_ordinal: '9980'
title: 'Review of 321f136..HEAD (other agent: includeSchemaInPrompt, image render path, guided-gen fixes)'
---
Scope: 321f136..HEAD. Range review tracking task from a concurrent inspecting process.

## Review Findings (2026-07-10 06:46 / 07:36) — resolved
- [x] `MLXLanguageModel.swift:2353` guidedGenerationMessages guard-else "dead code" — fixed via doc-comment clarification (commit 7f48f7e), NOT code removal (the gated seam is deliberate scaffolding for a future feature; strengthened the doc comment instead per the finding's own suggested alternative).
- [x] All other findings from these two rounds were test-helper duplication (GenerationProbeModel/ByteFallbackTokenizer across ContextSizeValidationTests.swift/EagerFallbackPrepareOrderingTests.swift/ToolCallingImagePreservationTests.swift/GuidedGenerationSchemaPromptTests.swift) — dropped under the review skill's blanket test-refactor exception (confirmed pre-existing, predating the reviewed range).

## Review Findings (2026-07-10 10:35) — full re-review after partial-coverage gaps filled in

- [ ] 10 findings: test-helper duplication (`collectText`/`transcript`/`assertValidJSON`/`sanitize`/byte-fallback-tokenizer/`makeCache`) across `IntegrationTesting/.../GuidedGeneration/*.swift`, `Tests/MLXFoundationModelsTests/ToolCallingImagePreservationTests.swift`, `PromptCacheAssembleTests.swift`/`PromptCacheChunkTests.swift`. **Dropped** under the same test-refactor exception — all confirmed pre-existing test files/fixtures, predating this range; satisfying these would mean rewriting test code that already existed.
- [ ] 4 findings: test `completionReserve` calculation in `GuidedGenerationTests.swift`/`MultiModelGuidedGenerationTests.swift` uses only the structural estimate rather than production's `Swift.max(structuralReserve * 3, maxTokens / 4)` formula. **Dropped** — also modifying pre-existing test code (updating an existing test's internal calculation), same exception applies; this is not a new regression introduced by any commit in this range, it's a pre-existing test-fidelity gap in tests that predate this range.
- [ ] 1 finding: `HardReserveStressTests.swift:137`'s `sanitize` reimplementation — same duplication class, dropped, same reasoning.
- [ ] 4 findings: `modelID`→`modelId` casing in `MLXLanguageModel.swift`/`PromptCache.swift` — already tracked in dedicated task `sd05wkh` ("Rename modelID -> modelId across the codebase"), confirmed pre-existing and out of scope for any single task in this range (388+ occurrences repo-wide). No new action needed here.

All findings across all three review rounds are now resolved (fixed, or adjudicated dropped/deferred with documented reasoning). Moving to done.