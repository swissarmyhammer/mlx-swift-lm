---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xw6kz29ch164wczqyzt1b4
  text: |-
    Measured the cost of the cast first. Micro-benchmark `dsv4bench castops` (scratchpad, loads no model): one decode query, 64 heads, head width 128, 21 indexer layers (the checkpoint has 21 layers with compress ratio 4), each layer with its own bfloat16 pool. 200 steps, median.
    - N=2048 chunks (8k context): cast only 0.329 ms; score with cast 0.617 ms; score with a kept float32 pool 0.547 ms. Gain 0.07 ms/step.
    - N=8192 chunks (32k context): cast only 0.453 ms; score with cast 1.008 ms; score with a kept float32 pool 0.725 ms. Gain 0.28 ms/step.
    - Same bits for both variants: true (a bfloat16 to float32 cast is exact).
    The gain at 32k is more than the 0.1 ms/token limit, thus the code change goes ahead.
  timestamp: 2026-10-14T18:53:38.146690+00:00
- actor: claude-code
  id: 01m4xy771xzmn7jz28tmm30v0m
  text: |-
    Implementation landed (not committed).

    What changed:
    - `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`: new `static let scoreDType = DType.float32`. The three float32 casts of the score operands use it. MLX `astype` gives back the same array when the dtype does not change (checked in `mlx/ops.cpp`), thus a float32 pool takes no cast. A call with no cache still casts its bfloat16 pool one time for each block.
    - `Libraries/MLXLLM/Models/DeepSeekV4Cache.swift`: `DeepSeekV4ChunkCache` has a new `poolDType: DType?`. When it is set, each new chunk is cast one time when it arrives, before the append. The indexer branch of `DeepSeekV4Cache` uses `DeepSeekV4Indexer.scoreDType`; the attention branch keeps the activation dtype (`nil`). `copy()` keeps the dtype. The empty-pool fallback uses the pool dtype. State format bumped: `DeepSeekV4Cache.projectedRows.v2` -> `DeepSeekV4Cache.scoreDTypeIndexerPool.v3`; a v2 prompt cache fails closed with `earlierFormatMessage`.
    - `Tests/MLXLMTests/DeepSeekV4CacheTests.swift`: 4 new tests (7 cases), and one shared helper `pooled(_:cuts:through:compressor:layer:)` that `pooledInCuts` now also uses.

    Discoveries:
    - On the real checkpoint the indexer pool was bfloat16 (bench `pooldtype` mode: layer 2 state slot 6, shape [1, 16, 128], bfloat16 before; float32 after).
    - With float32 compressor weights (the synthetic test checkpoint), the compressor norm and rotation promote the chunks to float32, thus a test of the dtype must cast the compressor weights to bfloat16 first. The first RED run showed this.
    - I did not add a dtype check to `validateRestore`: `PromptCacheTemplateRestoreTests` fills the indexer branch with float16 chunks through `restore(state:)`, and the indexer casts any pool dtype, so the result stays correct.
    - Memory: the indexer pool is 2x larger (at 32k: 21 layers x 8192 x 128 x 4 B = 84 MiB, was 42 MiB). Peak in the bench stays 147 GiB.
  timestamp: 2026-10-14T19:28:54.845264+00:00
- actor: claude-code
  id: 01m4xy7enmnax045agf07b607d
  text: |-
    Measurements at 32k (dsv4bench, args `<snap> 32768 128 8192 1024` and `<snap> profile 32768`, two runs each, GPU idle before each run; logs in the scratchpad as `atj-before-*.log` and `atj-after-*.log`).

    Before (HEAD 64a2d51 + only the scoreDType rename, which has no behavior change):
    - full bench decode: 31.0 and 30.8 ms/token; prefill 419.3 and 420.0 tok/s.
    - profile window: 32.3 and 31.9 ms/step; prefill 417.4 and 417.5 tok/s.

    After:
    - full bench decode: 30.4 and 30.5 ms/token; prefill 419.9 and 419.6 tok/s.
    - profile window: 32.1 and 32.1 ms/step; prefill 417.8 and 417.8 tok/s.

    Result: full bench decode -0.45 ms/token (mean 30.90 -> 30.45). Profile window: mean 32.1 -> 32.1, no change that the noise lets me see. Prefill: no change. The micro-benchmark predicted -0.28 ms/step. The generated text and the cache-check text are the same in all four full runs (same hash). Bench cache checks: HARD reuse check PASS and tie-aware PASS, both thinking modes, all four full runs.

    Tests: `swift build --build-tests` clean (no compiler warning; one SwiftPM "missing creator for mutated node" line that the build had before this change too). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest, 0 failures; 1437 Swift Testing tests in 113 suites, all passed. Greedy parity `greedyFirstTokensMatchThePythonFixture()` (xcodebuild -only-testing): passed.
  timestamp: 2026-10-14T19:29:02.644004+00:00
- actor: claude-code
  id: 01m4xy7h6gyk65awdc8y0ybsnc
  text: |-
    ### implement — changed
    - evidence: cast cost (micro-benchmark, 21 layers, 8192 chunks) 0.28 ms/step; 3 files — Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Tests/MLXLMTests/DeepSeekV4CacheTests.swift; decode at 32k 31.0/30.8 -> 30.4/30.5 ms/token (profile window 32.3/31.9 -> 32.1/32.1 ms/step); prefill ~420 tok/s before and after; state format v2 -> v3; MLXLMTests 716 XCTest + 1437 Swift Testing, 0 failures; greedy parity passed.
    - next: /review
  timestamp: 2026-10-14T19:29:05.232515+00:00
- actor: claude-code
  id: 01m4xyavhe68t68q3145p883d1
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (993f890): 0 findings, 0 confirmed, 0 refuted; 7 validator runs, 0 failed; 3 files reviewed (DeepSeekV4Cache.swift, DeepSeekV4Indexer.swift, DeepSeekV4CacheTests.swift); 2 .kanban files have no validator. Acceptance: (1) same bits — test theIndexerBranchKeepsItsPoolInTheScoreDTypeWithTheBitsOfACast (7 cuts) and theIndexerBranchKeepsItsScoreDTypePoolAcrossAStateRoundTrip; (2) decode at 32k 31.0/30.8 -> 30.4/30.5 ms/token, recorded in a comment; (3) cache save and load — round-trip test, v2 fail-closed test, and the full suite pass (716 XCTest, 1437 Swift Testing).
    - next: none. The task is in done.
  timestamp: 2026-10-14T19:30:54.126472+00:00
- actor: claude-code
  id: 01m4xybbdp318hq50y95s527qp
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 3 files; cast cost 0.28 ms/step; 32k decode 31.0/30.8 -> 30.4/30.5 ms/token; state format v3; greedy parity pass
    - test: green — 716 XCTest + 1437 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 993f890
    - review: clean — 0 findings
  timestamp: 2026-10-14T19:31:10.390325+00:00
position_column: done
position_ordinal: ffcd80
title: 'DeepSeek-V4 decode speed: keep a float32 copy of the indexer pool in the cache, not a cast for each step'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 5).

`DeepSeekV4Indexer.swift:351` casts the whole indexer pool to float32 for each prefill block and for each decode step. At 32k the pool has 8192 rows. Keep a float32 copy in the cache and add only the new chunks. ^er1vnkw measured no prefill gain for this (the cast is 4 MiB), thus measure decode first. This changes the cache layout (state format version). If the fused indexer kernel task is done first, this task can become unnecessary.

Estimate: decode -0.05 to -0.2 ms/token. Bit-exact. Risk: low.

## Acceptance Criteria

- [x] Same bits as before (unit test).
- [x] Decode ms/token at 32k, before and after, GPU idle. If there is no gain, record the numbers and close with no code change.
- [x] The cache state save and load tests pass. #deepseek-v4 #performance