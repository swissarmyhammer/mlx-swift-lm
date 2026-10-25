---
assignees:
- claude-code
depends_on:
- 01M5SK8FDMF5XH4FDH12H0FPX1
position_column: todo
position_ordinal: '80'
title: 'MTP: drafter cold start at a cached position, and a general trim/append API (resume is done by PR 581)'
---
## What
Upstream PR 581 (commit 843854ec) added `ResumableMTPDrafterModel` with `resumeDrafterState(...)` and `finalizeDrafterState(...)` in both Qwen35MTP.swift files (shared code `finalizeShiftedMTPDrafterState`), and `MTPDrafterContinuation { state, targetBoundaryHidden, targetProcessedTokenCount }` (Libraries/MLXLMCommon/MTPDrafterModel.swift). Thus a drafter continues from a cached prefix when a continuation is present.

What remains:
- [ ] Cold start: today a warm main cache with no aligned continuation makes the iterator run target-only for the turn (and for later turns on that cache). Add a cold start: an empty drafter state at position P (`positionOffset` sets RoPE before `cache.ropeOffset`, Libraries/MLXLLM/Models/Qwen35.swift ~597), fed from the first new token with the boundary hidden row of the current prefill. Correctness does not depend on the drafter. Use it in the iterator in place of the target-only fallback for `unavailableContinuation`.
- [ ] A general drafter-state API to trim the last n tentative entries and append entries with given hidden rows. `finalizeShiftedMTPDrafterState` requires `proposalAppended == 0`; make it work when tentative entries remain (needed for chained drafts).

## Acceptance Criteria
- [ ] A turn on a warm cache with no continuation runs MTP (no passthrough) and gives the same greedy tokens as plain decode; the next turn continues the drafter.
- [ ] Trim and append give the same drafter cache contents as a cold run over the same tokens (within float tolerance).

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift and ChatSessionQwenTextMTPTests.swift: cold start, then continuation; trim/append.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38