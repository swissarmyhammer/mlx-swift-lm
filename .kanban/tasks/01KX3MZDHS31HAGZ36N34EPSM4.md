---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: 'Disposition gate: land or cancel in-flight PromptCache work before the chunk rewrite starts'
---
## What
Five tasks sit in `doing` awaiting review with uncommitted code on this branch that the chunk rewrite partially deletes: ^k5zjwjd (resolve/applyDecision tests), ^t36npt2 (slot LRU tests), ^nxh4cqp (evictAll/remove tests), ^7fw2apn (multi-session tests), ^64jm412 (configurable slot limit incl. setPromptCacheSlotLimit). Starting the rewrite while these await review risks merge conflicts or silently deleting just-reviewed work.

Resolve the overlap FIRST:
- Run /review and commit the in-flight work as-is (it documents and guards CURRENT behavior, which ships until the rewrite lands; PromptCacheTestSupport.swift and much of the multi-session suite carry over).
- Comment on ^64jm412 and ^k5zjwjd that their surfaces (setPromptCacheSlotLimit/maxSlotsPerModel; decide/selectSlot/applyDecision tests) are slated for removal by the chunk rewrite — reviewed for correctness-on-current-code only.

## Acceptance Criteria
- [ ] All five `doing` tasks are in `done` (or explicitly cancelled with a comment), and the branch has no uncommitted PromptCache changes
- [ ] ^64jm412 and ^k5zjwjd carry supersession comments naming the chunk-rewrite tasks

## Tests
- [ ] `git status --short` clean for Libraries/ and Tests/ before the first chunk task starts; `swift test --filter 'PromptCache'` green at the commit point

## Workflow
- Use `/review` + `/commit` on the in-flight tasks; no new code in this task.