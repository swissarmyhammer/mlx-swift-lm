---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xf9kwyfexswne318njjqsq
  text: |-
    Research done.
    - mlx-vlm (`language.py`, `_sparse_pooled_attention`) uses the gather for every block length when the pool holds more than `index_topk` chunks. It gathers `(B, L, topK, D)` rows with `mx.take`, scores the window and the gathered rows apart, masks the picks that the query may not read (`take_along_axis` of the pool mask), and adds the two weighted sums. It does not tile the queries.
    - MLX SDPA has no fused kernel for a head of 512, thus the masked path is the op fallback: `q*scale @ k^T`, `where(mask, s, finfo.min)`, sink column first, `softmax(precise)`, `@ v`. The gather path can do the same math on `[sink | window scores | picked scores]`, thus only the order of the output sum changes.
    - Plan: for a block of more than one token on an indexer layer whose pool holds more than topK chunks, use `rankedChunks` (it already ranks the invisible chunks last) and mask each pick by the block-causal rule. Tile the queries so that one tile gathers at most 1 GiB and holds fewer than 2^31 scores. Decode stays on its present path.
    - After this change, `DeepSeekV4Indexer.callAsFunction` (selection mask) and `membershipMask` are reached only when the pool fits the budget, where the answer is the plain visibility. Thus they become dead code; the causal-contract tests move to `rankedChunks` plus the pick visibility.
    - Existing end-to-end checks that read the new path: `DeepSeekV4CacheTests` (prefill cut, decode steps against one call).
  timestamp: 2026-10-14T15:08:04.894028+00:00
- actor: claude-code
  id: 01m4xfkcb1j932myn82xhdvrrk
  text: 'Baseline at HEAD 4321eb9, GPU idle (0 %), Release dsv4bench, `32768 128 8192 1024`: prefill 32118 tokens in 98.16 s = 327.2 tok/s; decode 34.3 ms/token; peak 151 GiB. Both cache checks PASS (reuse, warm == cold). Log: scratchpad `dsv4bench/gather-before-32768.txt`.'
  timestamp: 2026-10-14T15:13:24.833703+00:00
- actor: claude-code
  id: 01m4xgfmzk35143d9ybr3cqyk7
  text: |-
    Implementation landed (not committed).
    - `DeepSeekV4Attention.attentionOverRankedChunks`: a block that needs a mask, on an indexer layer whose pool holds more than topK chunks, now gathers the topK ranked rows of each query (`takeAlong`, shape `(B, L, topK, D)`) and attends over `[sink | window scores | ranked scores]` with one precise softmax. This is the op path of MLX SDPA for a head of 512; only the output sum adds the two halves apart. Query tiles: `queryRowsPerTile` keeps one gather at or below `oneTileGatherBytes` (1 GiB = 2048 rows at the real checkpoint) and the joined scores under `oneCallScoreLimit`. No SDPA call is in this path.
    - Ranked chunks that a query may not read (fewer visible chunks than topK) are hidden with the new `DeepSeekV4Math.pickedChunkVisibility`; `pooledChunkVisibility` now shares the same rule.
    - Decode (one token, no window mask) keeps `attentionOverPickedChunks`. The masked path stays for a layer with no indexer and for a pool inside the budget.
    - Dead code removed: `DeepSeekV4Indexer.callAsFunction` (selection mask), `DeepSeekV4Indexer.membershipMask`, the attention `chunkVisibility`. Indexer tests now read `rankedChunks` plus `pickedChunkVisibility`; the membership-mask test and the ranked-versus-mask test went with the code.
    - New unit tests: `theRankedChunksGiveTheAnswerOfTheMaskedChunks` (one tile and forced tiles, hidden picks, a query with no visible chunk) against masked SDPA, tolerance 1e-4; `aTileHoldsTheMostRowsThatStayUnderTheGatherBudget`; `pickedChunkVisibilityHidesEveryPickThatReachesPastItsQuery`.
    - 32k, GPU idle, Release dsv4bench `32768 128 8192 1024`: prefill 327.2 -> 382.5 tok/s (98.16 s -> 83.97 s); decode 34.3 -> 34.0 ms/token; peak 151 -> 148 GiB. Logs: scratchpad `dsv4bench/gather-before-32768.txt`, `dsv4bench/gather-after-32768.txt`.
    - Cache checks: both HARD reuse checks PASS and both tie-aware checks PASS. With thinking=false, warm and cold turn 2 now differ in text ("Gull." against "Gulls."): first token, cold top-1 43.25 against warm 41.0, gap 2.25, inside the cut-spread margin 4.0 of the bench. Before the change the text was equal. The order of additions changed, thus a near tie can move.
    - Unit suite: `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` 716 XCTest + 1412 Swift Testing tests, 0 failures. Greedy parity `greedyFirstTokensMatchThePythonFixture()` PASS (10.6 s).
  timestamp: 2026-10-14T15:28:51.187139+00:00
- actor: claude-code
  id: 01m4xgfs4kx9rh9jene0zrktc1
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, Tests/MLXLMTests/DeepSeekV4AttentionTests.swift, Tests/MLXLMTests/DeepSeekV4IndexerTests.swift, Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift. 32k prefill 327.2 -> 382.5 tok/s, peak 151 -> 148 GiB, decode 34.3 -> 34.0 ms/token. Unit: 716 XCTest + 1412 Swift Testing, 0 failures. Greedy parity PASS. Bench cache checks: HARD PASS x2, tie-aware PASS x2.
    - next: /review
  timestamp: 2026-10-14T15:28:55.443959+00:00
- actor: claude-code
  id: 01m4xgpy24qvxv80m5bpqcpejz
  text: |-
    ### test - green
    - evidence: GPU idle (Device Utilization 0 percent). Deleted the DeepSeekV4 object files, then swift build --build-tests: Build complete, 0 compiler warnings, 0 errors. The only warning is a build-system message about the mlx-swift_Cmlx.bundle, not from the source. xcrun xctest MLXLMTests.xctest: XCTest 716 tests, 0 failures. Swift Testing 1412 tests in 111 suites passed. 0 skipped.
    - next: review
  timestamp: 2026-10-14T15:32:49.860275+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 prefill speed: attend over the top-k pooled chunks by gather, not masked SDPA over all chunks'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`, `prof4/prefill_attn_bench.txt`).

At prefill, `DeepSeekV4Attention.swift` (about lines 739-860 and 955-1006) gives SDPA the window keys and ALL pooled rows, and a mask selects the indexer top-k. At the last 2048-token chunk of a 32k prefill, this takes 137.4 ms for each layer. A top-k gather path takes 48.2 ms at all sizes. Thus each chunk gets slower as the context grows (4.5 s to 8.1 s for each chunk; mlx-vlm 4.6 s to 4.7 s).

Use the gather that decode already uses (about line 1229), in query tiles. Each 2048-query tile needs about 1 GiB. Keep the ^f24gprg rule: no SDPA call holds 2^31 or more scores.

Expected gain: about 12-15 s on a 32k prefill (323 to about 380 tok/s).

## Acceptance Criteria

- [x] Prefill at 32k uses the gather path. A unit test compares it with the masked path (tolerance, because the order of additions changes).
- [x] Prefill tok/s and peak memory at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance