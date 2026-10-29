---
assignees:
- claude-code
position_column: todo
position_ordinal: 9c80
title: 'MTP: keep speculating past the window of a maxKVSize ring that keeps a prefix'
---
## What
Found in ^crdd8sf. A `RotatingKVCache` built from legacy `maxKVSize` keeps a prefix (`keep = min(4, maxKVSize - 1)`). A multi-row write (an MTP verify) masks a plain window over the rows it presents (`RotatingKVCache.makeMask` n > 1), so a query past the window loses the kept prefix, while single-token decode keeps the prefix in the ring. MTP verify tokens then differ from plain decode. ^crdd8sf made the Qwen in-place round shrink to the window and then pass through with `MTPSpeculativeTokenIterator.keptPrefixWindowReason`, so `maxKVSize` streams stop speculating once the window fills (typed capacity with `preservedPrefixTokens: 0` keeps speculating).

- [ ] Give the multi-row write of a ring that keeps a prefix a keep-aware mask: query row r sees rows `< keep` and the `maxSize - keep` rows up to r (as single-row decode does), at least for a speculative verify.
- [ ] Decide whether chunked prefill past the window should use the same mask (plain prefill semantics change) or only the verify.
- [ ] Remove the passthrough in `MTPSpeculativeTokenIterator.drafts(_:fittingWindowOf:)` once the verify attends as decode.

## Acceptance Criteria
- [ ] With `maxKVSize` smaller than the generated length, Qwen MTP greedy output equals plain greedy output and `passthroughReason == nil`.

## Tests
- [ ] Extend Tests/MLXLMTests/Qwen35MTPCacheKindTests.swift (`prefixKeepingRingStopsSpeculatingAtTheWindow` becomes a speculate-past-the-window test).
- [ ] Release guards: `qwen38FourBitMTPGeneration` and `qwen38FourBitMTPSystemPromptCache` unchanged.

#mtp #kv-cache #qwen38