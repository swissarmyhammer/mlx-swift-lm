---
assignees:
- claude-code
position_column: doing
position_ordinal: '80'
title: Rename ClosingTokenBias.compute's eosTokenId parameter to eosTokenID (8 call sites incl. production)
---
## What
Surfaced by review pressure on `eg5tedh`'s ID-casing reversal, but this specific instance is bigger than a simple local rename: `ClosingTokenBias.compute(tokenizer:eosTokenId:)`'s `eosTokenId` parameter is a shared production function used at 8 call sites repo-wide (including production code, not just the 2 IntegrationTesting test files that originally surfaced it — `GenerableRoundTripTests.swift`, `MultiModelGuidedGenerationTests.swift`). Renaming only the 2 test call sites would break the build; a real fix means renaming the function's parameter itself and cascading through all 8 call sites.

## Acceptance Criteria
- [ ] Rename `ClosingTokenBias.compute`'s `eosTokenId:` parameter to `eosTokenID:` and update all 8 call sites (locate them via `git grep 'eosTokenId'` — includes both the 2 known IntegrationTesting test files and production code).
- [ ] No behavior change — pure rename.
- [ ] Build clean, full test suite green across all affected bundles.
- [ ] `IntegrationTesting` compile-checked (pre-existing unrelated `FMTestHelpers.swift` SDK-mismatch failure expected/documented).
- [ ] A local review pass confirms zero remaining findings of this class.

## Scope
Wherever `ClosingTokenBias.compute` and its 8 call sites live — likely spans `Libraries/MLXGuidedGeneration/` and multiple test/IntegrationTesting files. Budget for a genuine cross-file rename, not a 2-file fix. Not urgent/blocking — pure naming-convention correction, part of the broader ID/IDs acronym-casing correction effort (continuing `12d8p71`/`tba2jnb`/`38p9f6e`/`eg5tedh`).