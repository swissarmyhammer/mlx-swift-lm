---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx6m7426rh92fydfmxgxza02
  text: 'User ruling (2026-07-10): uppercase-acronym convention (entryID, modelID, tokenID) wins going forward, not the lowercase form (entryId, modelId, tokenId). This directly reverses this task''s premise — `modelID`''s existing casing is CORRECT and must NOT be renamed to `modelId`. Confirmed the governing validator (`swift`/`casing` rule at /Users/wballard/.validators/swift, likely sourced from /Users/wballard/.local/share/validators/swift/rules/casing.md) already states this explicitly: "ID and IDs are acronyms... DO: entryID, generatedTokenIDs, schemaJSON... This rule wins over local file prevalence... never flag correct uniform casing toward a nearby mixed-case majority." No validator fix needed — it was already correct; recent work (12d8p71 and portions of tba2jnb/5h314en) went the wrong direction against it. The actual reversal work (renaming entryId/tokenId/stopTokenIds/whitespaceTokenIds/schemaJson/eosId back to uppercase-ID form across the codebase) is tracked in a new comprehensive task. Closing this task as obsolete/invalid — do not implement.'
  timestamp: 2026-07-10T18:21:41.318616+00:00
position_column: todo
position_ordinal: 9a80
title: Rename modelID -> modelId across the codebase (large cross-file acronym-casing rename)
---
## What
Surfaced by review pressure on `12d8p71`'s acronym-casing rename in `MLXLanguageModel.swift`, but confirmed genuinely pre-existing and untouched by that commit's diff. Per this project's established acronym-casing convention (interior acronyms down-cased in lowerCamelCase — `Xg`, `tokenId`, `entryId`, `eosId`, etc., all already applied elsewhere), `modelID` should be `modelId`.

**This is a MUCH larger rename than the tasks it was found alongside** — confirmed via repo-wide grep (`grep -rn "\bmodelID\b" --include=*.swift . --exclude-dir=.build`) to appear across ~25+ files, spanning:
- Production code: `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, `PromptCache.swift`, `MLXDownloadProgress.swift`, `MLXLanguageModel+Availability.swift`, `Libraries/MLXLMCommon/ReasoningHeuristics.swift`.
- Test code: numerous files under `Tests/MLXFoundationModelsTests/` (PromptCache*, TokenizerBiasCacheTests, MLXLanguageModelTests, ModelCacheEvictionTests, ToolCallingImagePreservationTests, UnsupportedTranscriptContentMappingTests, TestHelpers, etc.).
- `IntegrationTesting/`: at least a dozen test files across GuidedGeneration, TextGeneration, Reasoning suites.

Note: the codebase is ALREADY inconsistent here — `modelId` (lowercase-d) also independently exists ~96 times elsewhere, so this rename would be unifying toward the already-more-common form, not introducing a brand new style.

## Acceptance Criteria
- [ ] Rename every `modelID` occurrence to `modelId` across all production and test files (including `IntegrationTesting/`), updating every call site, declaration, doc comment, and string literal reference.
- [ ] No behavior change — pure rename.
- [ ] Build clean, full test suite green (all bundles that reference this identifier: MLXFoundationModelsTests, and any others).
- [ ] Attempt to compile-verify `IntegrationTesting` too (a pre-existing, unrelated SDK-mismatch failure in `FMTestHelpers.swift` is expected/documented — confirm no NEW errors).
- [ ] A local review pass (`review sha` scoped to the commit) confirms zero remaining findings of this class.

## Scope
Whole-repo rename, not confined to one or two files — this is larger in blast radius than the typical "pre-existing debt" tracking task from this session, so budget accordingly. Not urgent/blocking — pre-existing naming-convention debt, not a correctness bug.