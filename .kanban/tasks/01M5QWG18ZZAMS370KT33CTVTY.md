---
assignees:
- claude-code
depends_on:
- 01M5SK8FDMF5XH4FDH12H0FPX1
position_column: todo
position_ordinal: '80'
title: 'MTP: let the Qwen drafter state continue from a cached prefix (model side)'
---
## What
Today `Qwen35MTPDraftModel.prepareDrafterState` sets `state.nextPosition = 0` and fills the drafter cache from the prompt of the current call only. The same reset is in both Libraries/MLXLLM/Models/Qwen35MTP.swift and Libraries/MLXVLM/Models/Qwen35MTP.swift. If the caller gives a main cache that already holds a prefix, the drafter does not have that prefix. This task changes the drafter model and its state type only. The iterator side is the next task.

The alignment contract, as the code does it today:
- **During decode:** drafter entries == main entries. Drafter entry j is `(token j+1, target hidden j)`. `prepareDrafterState` stores `prompt[1...] + bonus`, and `commitDrafterState` adds `accepted + 1` entries.
- **At a call boundary (after `finalizeGeneration`):** drafter entries == main entries − 1, plus the target hidden row of the last kept main position (the "pending hidden"). With this row, the next call adds drafter entry P−1 from its first new token, without a new run of the target.

- [ ] Add `MTPDrafterState.processedTokenCount` (== `nextPosition`) and `pendingHidden: MLXArray?` in Libraries/MLXLMCommon/MTPDrafterModel.swift (this task owns these two fields; the drafter-state task wraps them). The pending hidden is NOT part of `LMOutput.State` (there it sets `carriesModelState`, and `PromptCacheReusePolicy` then blocks every rewind).
- [ ] In both Qwen35MTP.swift files, make `prepareDrafterState` append at `state.nextPosition`, and use `pendingHidden` as the hidden of the first new entry when it is there.
- [ ] Cold start: when the state is empty but the main cache holds P tokens, start the drafter at position P (`positionOffset` sets RoPE before `cache.ropeOffset`, Libraries/MLXLLM/Models/Qwen35.swift ~597). Make sure `createAttentionMask` and the KVCacheSimple offset agree with this start. Do not replay the target over the prefix.
- [ ] Add a drafter-state API to trim the last n tentative entries and to append entries with given hidden rows (used by `commitDrafterState` and by the iterator finalize).

## Acceptance Criteria
- [ ] Two drafter calls, where the second continues the state of the first, give the same drafter cache contents (within float tolerance) and the same draft tokens as one cold call on the full prompt.
- [ ] A cold-started drafter at position P produces drafts with the same RoPE positions as a warm drafter (compare the attention keys of the new entries).

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared Qwen35 + MTP fixture: continuation against cold, cold start at P, trim and append.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38