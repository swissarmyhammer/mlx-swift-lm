---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4x2dvmkt6kewprp2sd5zaa9
  text: |-
    Research done. SDPA call sites in `DeepSeekV4Attention.swift`:
    1. Layer with no compressor (layers 0 and 1): `attentionWithCacheUpdate` with the window mask. The cache is a `RotatingKVCache`, and a block of L tokens gives about L keys, thus the 8018-token evidence on layer 0 reaches this site.
    2. Layer with a compressor and no pooled chunk yet: one direct SDPA over the window keys.
    3. `maskedAttention` -> `attentionOverJoinedKeys`: window keys joined to every pooled row, array mask.
    4. Decode (L = 1): `attentionOverEveryChunk` and `attentionOverPickedChunks`, both through `attentionOverJoinedKeys`, mask `.none`.

    Plan: one pure function `queryRowsPerCall(batch:heads:keys:scoreLimit:)` that gives the largest row count with `batch * heads * rows * (keys + 1) < 2^31`, and one `attentionInQuerySlices` that cuts the queries and the query axis of the mask into those slices. A block whose rows fit (every decode step) makes the same one call as before. A `.causal` mask becomes an explicit bottom-right causal array for each slice, because MLX aligns `.causal` to the last key. Site 1 keeps `attentionWithCacheUpdate` for L = 1 and for a cache that owns its own attention (quantized, TurboQuant, `KVCacheAttentionProtocol`); those routes do not use the MLXFast op path.
  timestamp: 2026-10-14T11:23:12.403478+00:00
- actor: claude-code
  id: 01m4x2rgpyyj3ea3dzey93k1xs
  text: |-
    Draft upstream issue for ml-explore/mlx (subtask 3). Not posted. The user posts it.

    Title: `fast.scaled_dot_product_attention` with `sinks` and an array mask gives wrong values when `B * H * L * (S + 1)` reaches 2^31

    Version: MLX 0.32.3 (Python and mlx-swift give the same result), Apple silicon, float32.

    Repro shape: `q` of shape (1, 64, n, 512), `k = v` of shape (1, 1, n, 512) (one key head, head dim 512, thus no fused kernel and the op path), `sinks` of shape (64,), an array mask of shape (n, n) for a causal window of 128 keys. Compare the last query row against a float32 reference that does the same math with plain ops (scores, prepend the sink column, softmax, drop column 0, matmul with v).

    Evidence (relative error of the last query row):
    - n = 5792: 1e-6 (64 x 5792 x 5793 = 2,147,395,584, less than 2^31).
    - n = 5793: 3e-6 (64 x 5793 x 5794 = 2,148,137,088, more than 2^31).
    - n = 6000: 0.024.
    - n = 8018: 0.061.
    - The same call without `sinks`: 1e-6 for every n up to 8193.

    Effect on a real model: DeepSeek-V4-Flash layer 0 (64 heads, head dim 512, sinks), one prefill block of 8018 tokens changes the attention output of the last token by 3.3 % in float32. Blocks of 128 to 4096 tokens: 0.0 %.

    Suspected cause: the op path with sinks joins one column to the scores, thus it holds `B * H * L * (S + 1)` values, and an index of 32 bits on that path overflows at 2^31.

    Workaround used in mlx-swift-lm: split the queries of one call so that each call holds fewer than 2^31 scores (`DeepSeekV4Attention.attentionInQuerySlices`).
  timestamp: 2026-10-14T11:29:01.662973+00:00
- actor: claude-code
  id: 01m4x2s6nb91bkgkqprtnbn5p0
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Tests/MLXLMTests/DeepSeekV4AttentionTests.swift. RED: `swift build --build-tests` failed on the missing `queryRowsPerCall` and `attentionInQuerySlices`. GREEN: `xcrun xctest -XCTest MLXLMTests.DeepSeekV4AttentionTests` 24 tests (5 new, the sliced test with 3 mask cases) passed. Full `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: XCTest 716 tests 0 failures, Swift Testing 1408 tests in 111 suites passed. Build: 0 compiler warnings (the one SwiftPM "missing creator for mutated node" line is from the build system, not from the code).
    - what changed: every L > 1 SDPA site (no-compressor window path, compressor path with no pooled chunk, `attentionOverJoinedKeys`) goes through `attentionInQuerySlices`. A block whose rows fit makes the same one call as before, thus each decode step is unchanged. The no-compressor path keeps `attentionWithCacheUpdate` for L = 1 and for a quantized, TurboQuant or `KVCacheAttentionProtocol` cache, and otherwise reads the arrays that `cache.update` returned (the issue 1662 rule holds; `attentionReadsTheArraysTheCacheReturned` still passes).
    - not run here: the real-weights greedy parity test and the scratch bench full mode (acceptance item 3); the orchestrator runs them.
    - next: /review
  timestamp: 2026-10-14T11:29:24.139698+00:00
- actor: claude-code
  id: 01m4x3bzzkwq073r91rndtd530
  text: |-
    ### test — green
    - evidence: `swift build --build-tests`: 0 compiler warnings (only a SwiftPM "missing creator for mutated node" build-system notice). `xcrun xctest MLXLMTests.xctest`: XCTest 716 run, 0 failures. Swift Testing 1408 tests in 111 suites passed. DeepSeekV4AttentionTests suite passed. No skipped tests.
    - evidence: xcodebuild `-only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/greedyFirstTokensMatchThePythonFixture()`: 1 test passed (26.5 s). TEST SUCCEEDED.
    - evidence: bench full mode (Release). Prefill: ctx 32 = 58.5 tok/s; ctx 2048 = 429.4 tok/s; ctx 8192 = 400.0 tok/s; ctx 32768 = 290.0 tok/s. Generate ms/token: 33.2 / 34.7 / 36.0 / 37.8 (baseline 32.9 / 35.0 / 36.3 / 37.8). Peak memory 141 / 143 / 147 / 158 GiB.
    - evidence: cache checks, thinking=false and thinking=true: warm == cold turn 2 (text) YES, tie-aware PASS, HARD reuse check PASS.
    - note: bench build shows 3 warnings in the scratch file main.swift only (deprecated prefillStepSize and peakMemory, optional interpolation). They are not in project code.
    - next: ready for review/commit. No commit made.
  timestamp: 2026-10-14T11:39:39.891495+00:00
- actor: claude-code
  id: 01m4x3ggxb60e6q0kwksybgkmh
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (commit e11e8e7). 0 findings, 0 confirmed, 1 refuted, 7 attempted, 0 failed. Reviewed: Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Tests/MLXLMTests/DeepSeekV4AttentionTests.swift. Not reviewed (no validator matches): the two `.kanban/tasks` files.
    - acceptance item 3 checked: the test step comment shows that the real-weights greedy parity test and the scratch bench full mode passed.
    - open for the user: post the draft upstream MLX issue (see the draft comment). Agents do not post it.
    - next: task moved to done. No commit, no push.
  timestamp: 2026-10-14T11:42:08.299486+00:00
- actor: claude-code
  id: 01m4x3gxqvfjz7d7nmbjmtj0xf
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 2 files (DeepSeekV4Attention.swift, DeepSeekV4AttentionTests.swift)
    - test: green — 716 XCTest + 1408 Swift Testing pass; greedy parity pass; bench generate ms/token 33.2/34.7/36.0/37.8, cache checks pass
    - commit: e11e8e7
    - review: clean — 0 findings
  timestamp: 2026-10-14T11:42:21.435277+00:00
position_column: done
position_ordinal: ffc380
title: 'DeepSeek-V4: one attention call over too many scores gives wrong values (MLX 32-bit index overflow with sinks)'
---
## What

Found on 2026-10-13 during ^sebtwba.

`MLXFast.scaledDotProductAttention` with `sinks` and an array mask gives wrong values when `heads x queries x (keys + 1)` is more than 2^31. Head dim 512 has no fused kernel, thus MLX uses the op path. That path holds a 32-bit index. MLX 0.32.3, Python (mlx-vlm venv) and mlx-swift, same result.

Evidence (session scratchpad `scripts/sdpa_long_mask.py` and `dsv4bench sdpa`): 64 query heads, 1 key head, head dim 512, float32, 128-key window mask, the last query row against a float32 reference:
- n = 5792: relative error 1e-6 (64 x 5792 x 5793 is less than 2^31).
- n = 5793: 3e-6. n = 6000: 0.024. n = 8018: 0.061.
- Without sinks: 1e-6 for all n up to 8193.

On the real checkpoint, one block of 8018 tokens changes the layer-0 attention output of the last token by 3.3 % in float32 (blocks of 128 to 4096 tokens: 0.0 %). Both ports show it.

## Where this port can reach the limit

`DeepSeekV4Attention.attentionOutput` gives SDPA the window keys AND every pooled row (the mask selects the indexer top-k). The score count of one call is `64 x L x (windowKeys + pooledRows + 1)`.
- Unchunked prefill (`PrefillParameters.Chunking.unchunked`) or a direct `model(tokens)` call: every block of more than 5792 tokens.
- Prefill in 2048-token chunks: a ratio-4 layer holds `context / 4` pooled rows, thus a context of about 57k tokens or more.
- Prefill in 512-token chunks (the default step): about 259k tokens or more.

## Approach

1. Write a unit test that compares one attention call over a long block with the same block cut into query slices (small shapes are not enough; the test must cross 2^31 scores, or the split rule must be tested as a pure function).
2. Split the queries of one SDPA call so that each call holds less than 2^31 scores. Keep the decode path as it is.
3. Report the MLX issue upstream (ml-explore/mlx). Status: draft written, the user posts it (see the comment that holds the draft issue text).

## Acceptance Criteria

- [x] No SDPA call in DeepSeek-V4 attention holds 2^31 or more scores.
- [x] A unit test proves the split.
- [x] The real-weights greedy parity test and the scratch bench full mode still pass. #deepseek-v4