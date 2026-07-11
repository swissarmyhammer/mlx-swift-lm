---
assignees:
- claude-code
position_column: todo
position_ordinal: 9d80
title: Mark resolvePromptCache/storePromptCache/removePromptCache/isDownloadingInCache as private in MLXLanguageModel.swift
---
## What
Surfaced by review pressure on `cthbfmw`'s PromptCache cutover commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — that commit's only change to `MLXLanguageModel.swift` was deleting `setPromptCacheSlotLimit`; these 4 functions were untouched.

`resolvePromptCache`, `storePromptCache`, `removePromptCache`, `isDownloadingInCache` (static functions in `MLXLanguageModel.swift`) are internal implementation details not meant for cross-module use, but default to `internal` access rather than being explicitly marked `private`.

## Acceptance Criteria
- [ ] Mark each of the 4 functions `private static func ...` — verify first that nothing outside this file/type actually calls them (grep the whole repo) before narrowing access, since `internal` may be relied upon by something not yet checked.
- [ ] No behavior change — pure access-level tightening.
- [ ] Build clean, full test suite green.

## Scope
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only. Not urgent/blocking — pre-existing cleanliness debt, not a correctness bug.