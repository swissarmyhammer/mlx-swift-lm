---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xhjkcq357fre0h735ybvtf
  text: 'Research (HEAD b90edec). `DeepSeekV4Indexer.chunkScores` still builds the full `(batch, 64, tokens, chunks)` float32 score tensor: 4 GiB at a 2048-token step with 8192 chunks. Then `maximum`, `* scale`, `* headWeights` run as separate primitives (not compiled), each one reads and writes 4 GiB, and `sum(axis: 1)` reads it again. The pooled keys `(1, chunks, 128)` are cast to float32 on each call: 4 MiB at 8192 chunks, thus a small cost. `rankedChunks` is called once per prefill step from `DeepSeekV4Attention.indexerPicks`; the ranked attention then tiles the query rows (`attentionOverRankedChunks`, 1 GiB gather limit). Baseline at 32k (GPU idle, this build): prefill 382.6 tok/s, peak 148 GiB, decode 33.8 ms/token, both HARD cache checks PASS.'
  timestamp: 2026-10-14T15:47:56.439048+00:00
- actor: claude-code
  id: 01m4xjkpzdfykw1855g2xkh6sr
  text: |-
    Implementation landed (not committed).

    - `DeepSeekV4Indexer.rankedChunks` now goes in query tiles of `queryRowsPerScoreTile` rows (1 GiB of per-head float32 scores for each tile: 512 rows at 8192 chunks and 64 heads). New argument `scoreByteLimit` (default `oneTileScoreBytes`). A block that fits in one tile (each decode step) takes one call as before.
    - The steps after the score matmul (`maximum`, `* scale`, `* headWeights`, `sum` over the heads) are one shapeless compiled function, `compiledHeadWeightedSum`, shared by every layer. Op path: `headWeightedSum`.
    - The pooled keys are cast to float32 one time for each block, in `scoreOperands`, and every tile reads that one copy.
    - New shared helper `DeepSeekV4Math.rowTiles(count:rowsPerTile:)`. The two tile loops of `DeepSeekV4Attention` (`attentionInQuerySlices`, `attentionOverRankedChunks`) now call it, so the stride loop is not written three times.

    What did not work: the first compiled function captured `scale` as a Swift `Float`. The fused kernel then gave sums that differ from the op path in the last bit (the new parity test `theCompiledHeadSumGivesTheBitsOfTheOpPath` failed). With `scale` as an `MLXArray` argument the bits are equal. A micro-benchmark (scratchpad `er1/indexerops1.txt`, 64 heads, 2048 rows) gave the same bits for every tile size 128 to 1024, and the same top-512 picks.

    Micro-benchmark, one layer, one prefill step: 8192 chunks 56.2 ms -> 31.7 ms; 2048 chunks 14.4 ms -> 8.2 ms. A persistent float32 copy of the pooled keys across steps gave no measured gain (31.80 ms with the cast, 31.70 ms with a kept copy: the cast is 4 MiB), so the cache layout did not change; the cast now runs one time for each block instead of each tile.
  timestamp: 2026-10-14T16:06:01.453852+00:00
- actor: claude-code
  id: 01m4xjkwkq15r958m0yvt180pb
  text: |-
    ### implement — changed
    - evidence: files `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, `Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift`, `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`, `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift`, `Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift`. Bench at 32k (`dsv4bench <snap> 32768 128 8192 1024`, GPU idle, Release): before prefill 382.6 tok/s (83.94 s), peak 148 GiB, decode 33.8 ms/token; after prefill 401.7 tok/s (79.96 s, -4.0 s), peak 147 GiB, decode 33.5 ms/token. Both HARD cache reuse checks PASS and both tie-aware checks PASS, before and after. Same generated text prefix. Unit: `swift build --build-tests` 0 compiler warnings; `xcrun xctest MLXLMTests.xctest` 716 XCTest 0 failures, 1418 Swift Testing tests passed. Greedy parity `greedyFirstTokensMatchThePythonFixture()` PASS (10.8 s), TEST SUCCEEDED. New tests: `aTiledRankingReadsTheChunksOfTheHighestReferenceScore` (selection equals the old math), `theCompiledHeadSumGivesTheBitsOfTheOpPath`, `aScoreTileHoldsTheRowsThatFitTheByteLimit`, `aScoreTileHoldsOneRowWhenOneRowBreaksTheByteLimit`, `rowTilesCoverEveryRowInOrderAndEndOnAShortTile`, `rowTilesGiveOneTileWhenEveryRowFits`. Logs: scratchpad `er1/before.log`, `er1/after.log`, `er1/unit2.log`, `er1/parity.log`.
    - next: /review
  timestamp: 2026-10-14T16:06:07.223104+00:00
- actor: claude-code
  id: 01m4xjrg8ghs3tkb1j26901vvg
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (b58361b). 2 findings, 2 confirmed, 1 refuted. Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:1043, Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:256 (duplication/duplication).
    - next: implement — extract one shared tiling helper and use it at the three sites. The task stays in review.
  timestamp: 2026-10-14T16:08:38.416179+00:00
- actor: claude-code
  id: 01m4xjrzahrjvfve2xx898rcc7
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 files; 32k prefill 382.6 -> 401.7 tok/s, peak 148 -> 147 GiB, decode 33.8 -> 33.5 ms/token; greedy parity pass
    - test: green — 716 XCTest + 1418 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: b58361b
    - review: findings — DeepSeekV4Attention.swift:1043, DeepSeekV4Indexer.swift:256
  timestamp: 2026-10-14T16:08:53.841142+00:00
- actor: claude-code
  id: 01m4xk15hc5wzwxyz59sxj5zgj
  text: |-
    Review findings fixed (not committed). New helper `DeepSeekV4Math.tiledRows(count:rowsPerTile:axis:whole:tile:)` in `DeepSeekV4MathHelpers.swift`. It holds the guard, the `rowTiles` map and the `concatenated(axis:)` join. `attentionInQuerySlices`, `attentionOverRankedChunks` and `DeepSeekV4Indexer.rankedChunks` now call it. A search of the three files finds no other copy of the wrapper.

    Why the helper takes a `whole` closure and does not only take `tile`: the one-tile path of each site calls its body with the arguments of the caller (no slice, the full mask, the `visible` that is not sliced). If that path called `tile(0 ..< count)`, it would add slice ops. Then the ops would not be the same. With `whole`, every site runs the same ops in the same order as before.

    New tests in `DeepSeekV4MathHelpersTests`: `tiledRowsJoinTheTilesInOrderOnTheAxis` (several tiles, the order, the join on axis 1) and `tiledRowsTakeTheWholePathWhenEveryRowFits` (one tile: `whole` runs and `tile` does not). RED: the test target did not compile ("type 'DeepSeekV4Math' has no member 'tiledRows'"). Note: `xcrun xctest --testing-library swift-testing --filter` does not work with this xctest; it printed only its environment. Thus GREEN came from the one full-suite run.
  timestamp: 2026-10-14T16:13:22.348096+00:00
- actor: claude-code
  id: 01m4xk18xd4yfz8c4w548x76de
  text: |-
    ### implement — changed
    - evidence: 5 files — `Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift`, `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`, `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, `Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift`, plus the task card. GPU idle (0%) before the run. `swift build --build-tests`: 0 compiler warnings (the only `warning:` line is the SwiftPM message "missing creator for mutated node", which was also there before). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest tests, 0 failures; 1420 Swift Testing tests in 111 suites pass (1418 + 2 new). Log: scratchpad `er1-unit3.log`. This is a refactor only (same ops in the same order), thus no bench. Both findings are checked.
    - next: /review
  timestamp: 2026-10-14T16:13:25.805609+00:00
- actor: claude-code
  id: 01m4xk3swfsvywz12t1hxz5kwh
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (0c0ef4f) — 0 findings, 0 confirmed, 0 refuted; 7 pairs attempted, 0 failed; 4 files reviewed, 2 .kanban files not reviewed (no validator matches). The 2 prior findings are checked.
    - next: none. The task is in done.
  timestamp: 2026-10-14T16:14:48.719857+00:00
- actor: claude-code
  id: 01m4xk45x9pd0k9cr1tbggvkdc
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 files, 2 findings fixed (shared tiledRows helper)
    - test: green — 716 XCTest + 1420 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 0c0ef4f
    - review: clean — 0 findings
  timestamp: 2026-10-14T16:15:01.033831+00:00
position_column: done
position_ordinal: ffc880
title: 'DeepSeek-V4 prefill speed: cheaper flat indexer with the same math (compile, query tiles, no f32 cast for each chunk)'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`).

In `DeepSeekV4Indexer.swift` (about lines 185-217 and 336-356), the prefill score tensor is 4 GiB of float32 at 32k, and the steps after it are not compiled. The pooled keys are cast to float32 again for each chunk. The flat indexer takes 58.2 ms for each layer at the last chunk.

Compile the steps after the scores, tile the queries, and keep a float32 copy of the pooled keys (or stop the cast) so that each chunk does not cast again. Do not use HISA: it changes which chunks are selected.

Expected gain: about 3-6 s on a 32k prefill.

## Acceptance Criteria

- [x] The selected chunks are the same as before (unit test).
- [x] Prefill tok/s and peak memory at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance

## Review Findings (2026-10-14 11:06)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4XBX091HWH4QJ1R5ER1VNKW.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XBX091HWH4QJ1R5ER1VNKW.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:1043` `duplication/duplication` — The guard, rowTiles map, and concatenated(axis: queryAxis) wrapper is repeated in attentionInQuerySlices and attentionOverRankedChunks. Copies of this loop can drift apart. Extract one shared helper, such as DeepSeekV4Math.tiledRows(count:rowsPerTile:axis:tile:), that holds the guard, the rowTiles map, and the concatenation. Each of the three sites passes its per-tile closure and its row limit.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:256` `duplication/duplication` — The rowsPerTile guard, rowTiles map, and concatenation in rankedChunks copy the same wrapper used in the two attention functions. Use the same shared tiling helper as the attention paths. Pass tokenAxis for the concatenation axis and keep the chunk-score closure as the per-tile body.
