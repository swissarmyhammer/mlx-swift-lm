---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'DeepSeek-V4 decode speed: keep a float32 copy of the indexer pool in the cache, not a cast for each step'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 5).

`DeepSeekV4Indexer.swift:351` casts the whole indexer pool to float32 for each prefill block and for each decode step. At 32k the pool has 8192 rows. Keep a float32 copy in the cache and add only the new chunks. ^er1vnkw measured no prefill gain for this (the cast is 4 MiB), thus measure decode first. This changes the cache layout (state format version). If the fused indexer kernel task is done first, this task can become unnecessary.

Estimate: decode -0.05 to -0.2 ms/token. Bit-exact. Risk: low.

## Acceptance Criteria

- [ ] Same bits as before (unit test).
- [ ] Decode ms/token at 32k, before and after, GPU idle. If there is no gain, record the numbers and close with no code change.
- [ ] The cache state save and load tests pass. #deepseek-v4 #performance