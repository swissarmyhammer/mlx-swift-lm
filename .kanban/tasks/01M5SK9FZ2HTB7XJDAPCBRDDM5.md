---
assignees:
- claude-code
depends_on:
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SJNWMNA5D8N2VWEJJPT9A2
position_column: todo
position_ordinal: '9380'
title: 'MTP: iterator takes a drafter state in and gives the final drafter state out (all end paths)'
---
## What
`MTPSpeculativeTokenIterator.init` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift) always makes a new `MTPDrafterState` (~151-152). `MTPDrafterState` is a struct (MTPDrafterModel.swift ~112), and the iterator moves into the private `generateLoopTask` (Libraries/MLXLMCommon/Evaluate.swift ~2946); `preparedState` reads state only before the loop (~2511, ~2849). Thus no caller can get the final drafter state today.

- [ ] Let `MTPSpeculativeTokenIterator.init` and the `generate` / `generateTokens(...mtpDrafter:)` overloads (Evaluate.swift ~2690-2790) accept an optional existing `MTPDrafterState`.
- [ ] Export the final drafter state with a completion callback (or a reference box) that `generateLoopTask` calls after `finalizeGeneration`, on every exit: stop token, `maxTokens`, cancellation, and error (nil on error).
- [ ] `finalizeGeneration`: rewind the main lookahead as today, trim the drafter by `lookahead + 1`, and set `pendingHidden` to the verify hidden row `accepted − lookahead` of the last round. Define the two paths that the current code misses: (a) no round ran (`guard lookahead > 0` at ~738): use the prefill hidden row N−1; (b) the tail `passthroughStep()` when `draftBudget == 0` (~410-416) moves the main cache without the drafter: feed the drafter for that step from its hidden row, or export nil.
- [ ] After `switchToPassthrough`, export nil (passthrough steps emit no hidden rows); the next call then uses the cold start.

## Acceptance Criteria
- [ ] Two calls, where the second continues the main cache and the exported drafter state of the first, give the same greedy tokens as one cold MTP call on the full prompt and as plain decode, and the same drafter cache contents (within float tolerance) as the cold call.
- [ ] The drafter-count invariant holds after each exit path: drafter == main − 1 with a pending hidden row, or nil.
- [ ] `qwen38FourBitMTPGeneration` (Release integration benchmark on the M5 Max) MTP tok/s stays within ±3% of the baseline (math 52.3, code 43.4, prose 26.3).

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared fixture: continue after a stop on an accept-all round, on a reject round, at `maxTokens` with the tail passthrough step, after zero rounds, and after cancellation.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38