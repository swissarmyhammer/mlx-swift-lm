---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
title: 'DeepSeek-V4 prefill speed: cheaper flat indexer with the same math (compile, query tiles, no f32 cast for each chunk)'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

In `DeepSeekV4Indexer.swift` (about lines 185-217 and 336-356), the prefill score tensor is 4 GiB of float32 at 32k, and the steps after it are not compiled. The pooled keys are cast to float32 again for each chunk. The flat indexer takes 58.2 ms for each layer at the last chunk.

Compile the steps after the scores, tile the queries, and keep a float32 copy of the pooled keys (or stop the cast) so that each chunk does not cast again. Do not use HISA: it changes which chunks are selected.

Expected gain: about 3-6 s on a 32k prefill.

## Acceptance Criteria

- [ ] The selected chunks are the same as before (unit test).
- [ ] Prefill tok/s and peak memory at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance