---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4vnpheydb536am8cqv16dkk
  text: 'Note from ^9xmds28 (not committed yet, in review): `DeepSeekV4Indexer.rankedChunks(_:queryResidual:pooledKeys:cos:sin:offset:)` now gives the top-k chunk INDICES, shape (batch, tokens, topK), for any block length; `callAsFunction` builds its membership mask from the same ranking (`topChunks(of:visible:)`). ^9xmds28 uses the indices only for decode (L = 1). The prefill path still builds the [B, L, topK, chunks] broadcast compare, so this card is unchanged. An alternative to the scatter: mlx-vlm gathers the top-k rows for L > 1 as well (`_gather_pooled`, `_sparse_pooled_attention` with a split softmax and `take_along_axis` of the pool mask), which removes the membership mask for prefill. Memory of that gather at a 2048-token prefill step: 2048 x 512 x 512 x 2 bytes = 1 GiB for each ratio-4 layer, against about 2 GiB of Booleans for the current compare at 8192 chunks (32k context). Measure before you choose.'
  timestamp: 2026-10-13T22:21:30.974251+00:00
- actor: claude-code
  id: 01m4vnpns3gsd1jbjfzjt8r48j
  text: 'Correction to the note above: the current broadcast compare at a 2048-token prefill step with 8192 chunks is 2048 x 512 x 8192 Booleans = 8 GiB for each ratio-4 layer, not 2 GiB. The gather alternative is 1 GiB (bf16 rows).'
  timestamp: 2026-10-13T22:21:35.395217+00:00
- actor: claude-code
  id: 01m4x608zjyq5vx2213wv7ybez
  text: 'Research (HEAD bc35319). The prefill membership mask is in `DeepSeekV4Indexer.callAsFunction`: `topChunks(of:visible:)` gives indices (batch, tokens, topK), then `(ranked .expandedDimensions(-1) .== chunkIndices).any(axis: -2)` makes a (batch, tokens, topK, chunks) Boolean array, then `&& visible`. The decode path (`rankedChunks`, top-k gather in `DeepSeekV4Attention`) does not use this mask. `argPartition` gives distinct indices in [0, chunks), thus the ranking has no duplicate or out-of-range index. When fewer chunks are visible than topK, the ranking gives indices of chunks that are not visible; the `&& visible` step removes them, and it stays. When chunks <= topK, the `guard` path returns the visibility and does no ranking. Plan: a static `membershipMask(of:chunkCount:)` that does `putAlong` of `true` into a (batch, tokens, chunks) zero Boolean array. Before measurement at HEAD bc35319 (GPU idle, full mode args): prefill 62.9/430.6/400.4/290.1 tok/s, peak 141/143/147/158 GiB, generate 33.0/35.1/36.1/37.6 ms/token at context 32/2k/8k/32k.'
  timestamp: 2026-10-14T12:25:41.618754+00:00
- actor: claude-code
  id: 01m4x6ke73e6axef062522vrrq
  text: |-
    Implementation done (not committed). `DeepSeekV4Indexer.membershipMask(of:chunkCount:)` (internal static) does `putAlong` of `true` into a (batch, tokens, chunks) zero Boolean array on the last axis. `callAsFunction` calls it in place of the broadcast compare; the `&& visible` step stays. The private `rankedChunkAxis` (-2) is removed; a new private `maskChunkAxis` (-1) replaces it. RED: the new test did not compile (no member `membershipMask`). GREEN: `theScatteredMembershipMaskIsTheBroadcastCompare` passes. It compares the scatter against an independent broadcast compare in the test file, for 2 x 7 x 5 random uint32 indices at 3, 5, 16, 64 chunks (3 chunks with 5 indices for each row gives repeated indices). MLX scatter on a Boolean array works on the GPU. Out-of-range indices are not tested: `argPartition` cannot make them, and the doc comment states the range as a requirement.

    Measurement, M3 Ultra, GPU idle, full mode args `32,2048,8192,32768 128 8192 1024`, same build steps:
    - Before (HEAD bc35319): prefill 62.9/430.6/400.4/290.1 tok/s; peak 141/143/147/158 GiB; generate 33.0/35.1/36.1/37.6 ms/token.
    - After: prefill 64.6/429.9/409.3/326.3 tok/s; peak 141/143/145/151 GiB; generate 33.0/34.9/36.1/37.8 ms/token.
    - At 32k: prefill +12.5% (290.1 -> 326.3 tok/s), peak -7 GiB (158 -> 151). At 8k: +2.2%, peak -2 GiB. 2k and 32 are the same (noise). Decode is the same. The first 160 characters of each bench answer are the same before and after. The cache check passes in both runs.

    Greedy parity: `xcodebuild test -project IntegrationTesting.xcodeproj -scheme IntegrationTesting ... -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/greedyFirstTokensMatchThePythonFixture()`, run alone: PASS after 25.9 s, TEST SUCCEEDED.

    Discovery: that xcodebuild run shows `no calls to throwing functions occur within 'try' expression` warnings in IntegrationTesting files this task did not touch (TokenizerVocabExtractorTests.swift, HardReserveStressTests.swift). New task ^nm3jbxg records them.
  timestamp: 2026-10-14T12:36:09.571194+00:00
- actor: claude-code
  id: 01m4x6kjadwn3v2sprwat0m8kx
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Tests/MLXLMTests/DeepSeekV4IndexerTests.swift. 32k prefill 290.1 -> 326.3 tok/s, peak 158 -> 151 GiB (GPU idle). `swift build --build-tests` exit 0 (only the known "missing creator for mutated node" note). `xcrun xctest MLXLMTests.xctest`: 716 XCTest, 0 failures; 1409 Swift Testing in 111 suites passed; 0 skipped. greedyFirstTokensMatchThePythonFixture (xcodebuild, alone): PASS. `swift format lint` on the 2 files: clean.
    - next: /review. Not committed. Task stays in doing.
  timestamp: 2026-10-14T12:36:13.773538+00:00
- actor: claude-code
  id: 01m4x6t09h9spkd2bthtcp98jq
  text: |-
    ### test — green
    - evidence: GPU idle before the run (Device Utilization 0%). `swift build --build-tests` after touch of the two changed files: Build complete, 0 compiler warnings. The only build output warning is a build-system note, "missing creator for mutated node" for the mlx-swift_Cmlx.bundle path. `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: XCTest 716 tests, 0 failures; Swift Testing 1409 tests in 111 suites, passed; 0 skipped. Suite DeepSeekV4IndexerTests passed.
    - next: review. No commit was made.
  timestamp: 2026-10-14T12:39:44.689435+00:00
- actor: claude-code
  id: 01m4x6wgfsjnpxm2hctcpqmncz
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (9fea5ff). 0 findings, 0 confirmed, 0 refuted. 7 validators ran, 0 failed. The engine reviewed 2 files: Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift and Tests/MLXLMTests/DeepSeekV4IndexerTests.swift. It did not review 6 .kanban files because no validator matches them.
    - next: none. The task is in done.
  timestamp: 2026-10-14T12:41:06.809908+00:00
- actor: claude-code
  id: 01m4x6wqpre5jnwa51ddmqk4zr
  text: 'Correction to the review comment above: the engine counts are attempted 7, failed 0, skipped 0. The value 7 is the number of review attempts. It is not necessarily the number of validators.'
  timestamp: 2026-10-14T12:41:14.200213+00:00
- actor: claude-code
  id: 01m4x6x6qe2avdbk2rf9qs4czt
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 2 files (DeepSeekV4Indexer.swift, DeepSeekV4IndexerTests.swift); 32k prefill 290.1 -> 326.3 tok/s, peak 158 -> 151 GiB; greedy parity pass
    - test: green — 716 XCTest 0 failures + 1409 Swift Testing pass, 0 warnings
    - commit: 9fea5ff
    - review: clean — 0 findings
  timestamp: 2026-10-14T12:41:29.582699+00:00
position_column: done
position_ordinal: ffc580
title: 'DeepSeek-V4 prefill speed: indexer membership mask by scatter, not a broadcast compare'
---
## What

`Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:216-220` builds the membership mask of the selected chunks with a `[B, L, topK, chunks]` broadcast compare.

At prefill, L is large. In mlx-lm PR 1189 the same method needed about 34 GB for each layer at 128K context. A scatter (`put_along_axis`) gives the same mask with O(L * topK) writes. That change, with a fused scorer, took prefill at 64K context from 191 to 335 tok/s (PR 1189 comments, `c1189.md:1096-1124` in the session scratchpad `dl/`).

Also consider the mlx-vlm decode path, which gathers only the top-k pooled rows above 512 rows (`mlx_vlm .../language.py:426-473, 1011-1028`).

Baseline on 2026-10-12, M3 Ultra, GPU idle: prefill 397 tok/s at 2k, 375 at 8k, 276 at 32k.

## Acceptance Criteria

- [x] The mask comes from a scatter, with a parity test against the broadcast compare.
- [x] Peak memory and prefill speed at 32k context, before and after, on an idle GPU.
- [x] Greedy output on the real checkpoint is the same as before.
#deepseek-v4 #performance