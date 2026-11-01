---
assignees:
- claude-code
position_column: todo
position_ordinal: '9080'
title: SpeculativeTokenIterator never clears the MLX buffer pool (the draft-model path, as MTP did before ^y1hk1vz)
---
Found in ^y1hk1vz. `TokenIterator` and, after ^y1hk1vz, `MTPSpeculativeTokenIterator` clear the MLX pool every 256 emitted tokens through `GenerationBufferPoolClearing`. `SpeculativeTokenIterator` in `Libraries/MLXLMCommon/Evaluate.swift` (draft-model speculation) has no clear, by reading the code. It is not measured. A long draft-model decode may grow the pool the same way, for its target cache and its draft cache.

- [ ] Measure a long draft-model decode at long context: pool, swap, tok/s.
- [ ] If the pool grows, store `GenerationBufferPoolClearing` in the iterator and count emitted tokens, with a test as in `GenerationBufferPoolClearingTests`.

#perf