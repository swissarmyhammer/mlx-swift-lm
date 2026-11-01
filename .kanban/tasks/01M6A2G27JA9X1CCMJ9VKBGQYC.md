---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6asxc29qzgdw22am5bxmnag
  text: 'The same small change fixed the code here, under ^84b33jx (uncommitted, branch mtp): `SpeculativeTokenIterator` stores `GenerationBufferPoolClearing` and clears through one `emitPendingToken()`, which both hand-out sites now call, so it counts emitted tokens. Test `draftModelIteratorClearsEveryIntervalOfEmittedTokens` (RED then GREEN; 2 mutants, no clear and round-start only, both fail it). The second box is checked on that basis. The first box stays open: no long draft-model decode was measured, so whether the pool grew on this path, and the tok/s effect, are not known.'
  timestamp: 2026-11-01T05:39:27.433037+00:00
position_column: todo
position_ordinal: '9080'
title: SpeculativeTokenIterator never clears the MLX buffer pool (the draft-model path, as MTP did before ^y1hk1vz)
---
Found in ^y1hk1vz. `TokenIterator` and, after ^y1hk1vz, `MTPSpeculativeTokenIterator` clear the MLX pool every 256 emitted tokens through `GenerationBufferPoolClearing`. `SpeculativeTokenIterator` in `Libraries/MLXLMCommon/Evaluate.swift` (draft-model speculation) has no clear, by reading the code. It is not measured. A long draft-model decode may grow the pool the same way, for its target cache and its draft cache.

- [ ] Measure a long draft-model decode at long context: pool, swap, tok/s.
- [x] If the pool grows, store `GenerationBufferPoolClearing` in the iterator and count emitted tokens, with a test as in `GenerationBufferPoolClearingTests`.

#perf