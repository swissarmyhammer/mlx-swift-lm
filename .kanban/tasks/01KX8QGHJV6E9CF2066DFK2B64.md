---
assignees:
- claude-code
position_column: todo
position_ordinal: a180
title: Reduce commitPromptCache's do-catch/cache-management complexity
---
## What
Surfaced by review pressure on a standalone `getOrCreateCached`/`Set()` fix commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — zero matches, untouched by that commit.

`Libraries/MLXFoundationModels/MLXLanguageModel.swift` (~line 2855 as of the flagging commit — will have shifted): a function with complex do-catch error handling with specific exception-type catching, conditional cache-slot mutation, and nested if logic for cache management. The do block has conditional cache mutation and the catch block catches a specific exception type.

## Acceptance Criteria
- [ ] Extract the cache management logic (slot initialization, mutation, validation) into a separate helper function, reducing conditional density in the main function.
- [ ] No behavior change — pure refactor.
- [ ] Build clean, full test suite green.
- [ ] A local review pass confirms zero remaining findings of this class.

## Scope
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only — likely `commitPromptCache` or a neighboring function; relocate by content (do-catch with cache-slot mutation + specific exception-type catch), not line number. Not urgent/blocking — pre-existing complexity debt, not a correctness bug.