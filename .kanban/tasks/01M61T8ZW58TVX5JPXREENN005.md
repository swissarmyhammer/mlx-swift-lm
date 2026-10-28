---
assignees:
- claude-code
depends_on:
- 01M5SKBV27GED43EB3T8MCSVWE
position_column: todo
position_ordinal: '9880'
title: 'MTP: finalizeGeneration cannot rewind the committed lookahead of a partially accepted hybrid round'
---
## What
Found while doing ^8mcsvwe. With blockSize > 2 (^hcfzyem), a round can accept some drafts and reject the rest. The iterator then rewinds the main cache by `numDraft - accepted` with `rewindSpeculativePromptCache`, and `MambaCache.selectSpeculativeSlot(rollingBack:)` releases all snapshot slots (decision of ^cst7wan: a round with a rejection releases its slots). The accepted drafts stay in the cache as committed lookahead (`committedPendingTokenCount = accepted`).

If the stream stops before the caller consumes those accepted drafts (stop token, maxTokens, cancel), `finalizeGeneration` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift) calls `mainCacheStorage.trim(lookahead)` (refused: MambaCache is not trimmable) and then `mainCacheStorage.rewindSpeculative(lookahead)`, which returns 0 because no slots remain. `rewound != lookahead`, so `targetBoundaryHidden = nil`, the drafter continuation is dropped, and the main cache keeps tokens that were never emitted. At blockSize 2 this cannot happen (accepted = 1 implies no rejection, and the slots stay).

- [ ] Keep enough recurrent state after a partial-accept rewind to rewind the accepted drafts again (for example: keep the slots of tokens 1..accepted after `selectSpeculativeSlot`, and release them at the next round / passthrough / finalize, as an all-accepted round does), or replay — choose and document; measure the memory and tok/s cost.
- [ ] `finalizeGeneration` rewinds the committed lookahead of a partially accepted round exactly.

## Acceptance Criteria
- [ ] With a scripted drafter at blockSize 3..6 (see Tests/MLXLMTests/Qwen35VerifyRewindTests.swift `ScriptedRejectionDrafter`), a stream that stops inside the accepted drafts of a partially accepted round leaves the main cache at the emitted ledger count, and a continued second call equals a cold call and plain decode.

## Tests
- [ ] Tests/MLXLMTests: stop-inside-accepted-drafts test at blockSize 3..6.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache