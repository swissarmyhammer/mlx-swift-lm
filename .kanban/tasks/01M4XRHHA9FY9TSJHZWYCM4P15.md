---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xt676s4x6rng45561w7242
  text: |-
    Research and ceiling (micro-benchmark `dsv4bench topkops`, Release, GPU idle, float32 scores, per call):
    - Indexer prefill tile 2048 x 8192: argPartition 3.31 ms; the negate alone 0.39 ms. 2048 x 4096: 2.22 ms. 2048 x 2048: 0.78 ms.
    - Indexer decode 1 x 8192 (21 calls in one eval): 0.044 ms per call; negate alone 0.012 ms. Ceiling about 0.03 ms x 21 CSA layers = about 0.67 ms per token (2 % of 31.7 ms).
    - Prefill ceiling at 32k: about 0.6 s of about 78.6 s (below 1 %).
    - MoE router 1 x 256 (43 calls): 0.009 ms per call; negate alone 0.006 ms. Ceiling about 0.13 ms per token, and the router is in a compiled trace. Router prefill 2048 x 256: 0.24 ms. Thus the router stays on argPartition.
    - MLX sort: 4 values per thread, 512 threads, thus 2048 values per block. 8192 values go to the multi-block sort (several dispatches).
    - The callers of `rankedChunks` do not need an order: the decode path gathers the picks and the prefill path masks them by index. The softmax sums change order only.
    - Plan: one Metal kernel (MLXFast.metalKernel), one threadgroup for each row: 4 radix passes of 8 bits on an order-preserving uint key, then one compaction pass in index order. The picks come out in ascending index order. On ties at the threshold, the kernel takes the lowest indices. NaN maps to the lowest key (argPartition puts NaN last), -inf next.
  timestamp: 2026-10-14T18:18:27.929410+00:00
- actor: claude-code
  id: 01m4xvqneatcbyx23235ba485y
  text: |-
    Implementation landed (not committed):
    - New `Libraries/MLXLLM/Models/DeepSeekV4TopKKernel.swift`: one MLXFast.metalKernel, one threadgroup of 1024 threads for each row, 4 radix passes of 8 bits on an order-preserving uint key, then one compaction pass. Picks come out in ascending index order. Ties at the cut take the lowest indices. NaN takes the lowest key; -0 folds into +0 (the argPartition order). No float math in the kernel, thus no fp contract pragma. The row width comes from `scores_shape`, thus a growing pool does not compile a new specialization.
    - `DeepSeekV4Indexer.topChunks` calls the kernel on the GPU; on another device it keeps argPartition and sorts the picks (same contract).
    - Pitfall found: MLX gives an input of fewer than 8 values in the `constant` address space, thus the kernel must index `scores` and keep no `device` pointer (same as the Sinkhorn kernel).
    - Tests: new `Tests/MLXLMTests/DeepSeekV4TopKKernelTests.swift` (6 tests; set parity with argPartition for 4 shapes, ascending order, NaN/-inf/hidden, ties, ties inside the run of one thread, signed zero). Each tie and signed-zero test was seen to fail on a mutated kernel. New indexer test `theRankedChunksOfEachQueryComeInAscendingIndexOrder` failed on argPartition (16 issues) and passes now.

    Bench at 32k (Release, GPU idle, two runs each, `<snap> 32768 128 8192 1024` and `profile 32768`):
    - Before (HEAD 64e0feb): prefill 417.0 / 417.3 tok/s, decode 31.7 / 31.4 ms/token; profile prefill 416.4 / 416.3 tok/s, window 33.4 / 33.2 ms/step.
    - After: prefill 419.6 / 419.7 tok/s, decode 30.9 / 30.9 ms/token; profile prefill 417.6 / 417.1 tok/s, window 32.2 / 32.2 ms/step.
    - Cache checks: HARD reuse PASS and tie-aware PASS in all four full runs (both thinking modes).
    - The 32k sample text changes after about 70 characters (the attention sums over the picks now run in index order, thus the numbers are not bit-exact with the argPartition order). The 8k warm == cold text check is now YES in both thinking modes (before: NO at char 4 with thinking off, tie-aware PASS).
  timestamp: 2026-10-14T18:45:28.138337+00:00
- actor: claude-code
  id: 01m4xvye44tjhv8nk81y74jq1h
  text: |-
    ### implement — changed
    - evidence: files Libraries/MLXLLM/Models/DeepSeekV4TopKKernel.swift (new), Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift, Tests/MLXLMTests/DeepSeekV4TopKKernelTests.swift (new), Tests/MLXLMTests/DeepSeekV4IndexerTests.swift. argPartition ceiling: prefill tile 2048x8192 3.31 ms (negate 0.39 ms), decode 1x8192 0.044 ms per call x 21 layers. 32k before: prefill 417.0/417.3 tok/s, decode 31.7/31.4 ms/token (profile 33.4/33.2 ms/step). After: prefill 419.6/419.7 tok/s, decode 30.9/30.9 ms/token (profile 32.2/32.2 ms/step). Bench cache checks: HARD reuse PASS, tie-aware PASS in all runs. `swift build --build-tests` exit 0, no compiler warning (only the known "missing creator" build-system note). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest, 0 failures; 1433 Swift Testing tests in 113 suites passed; 0 skipped. Greedy parity `greedyFirstTokensMatchThePythonFixture()` via xcodebuild -only-testing: PASS, TEST SUCCEEDED (scratchpad/it-parity-ycm.log). MoE router not changed: ceiling about 0.13 ms per token and it runs in a compiled trace.
    - next: /review
  timestamp: 2026-10-14T18:49:10.020219+00:00
- actor: claude-code
  id: 01m4xw28484znsek6th0p8650z
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (commit 979276f): 0 findings, 0 confirmed, 0 refuted, 7 validator runs, 0 failed. 4 files reviewed. 2 .kanban files not reviewed (no validator matches). Acceptance check: (1) set parity with argPartition: `theKernelPicksTheSetOfArgPartitionWhenNoScoresTie` in Tests/MLXLMTests/DeepSeekV4TopKKernelTests.swift, 4 shapes, test run passed (1433 Swift Testing tests, 0 failures). (2) 32k before/after, GPU idle: prefill 417.0/417.3 to 419.6/419.7 tok/s, decode 31.7/31.4 to 30.9/30.9 ms/token. (3) Greedy parity: `greedyFirstTokensMatchThePythonFixture()` passed, TEST SUCCEEDED in scratchpad/it-parity-ycm.log.
    - next: none. Task moved to done.
  timestamp: 2026-10-14T18:51:14.952582+00:00
- actor: claude-code
  id: 01m4xw2s4e2gt2mhxgbqb354h0
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 4 files; 32k decode 31.7 -> 30.9 ms/token, prefill 417 -> 420 tok/s; greedy parity pass
    - test: green — 716 XCTest + 1433 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 979276f
    - review: clean — 0 findings
  timestamp: 2026-10-14T18:51:32.366060+00:00
position_column: done
position_ordinal: ffcc80
title: 'DeepSeek-V4 speed: radix-select top-k for the indexer, not argPartition (a full sort in MLX)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 3).

MLX `argPartition` is a full sort (`mlx/backend/metal/sort.cpp:342`: "We direct arg partition to sort for now"). The indexer uses it at `DeepSeekV4Indexer.swift:273` to pick 512 of up to 8192 chunks. llama.cpp uses a radix-select top-k kernel (`kernels/argsort.metal:250`). Also consider the MoE router (`DeepSeekV4MoE.swift:387`).

The selected set is the same except for ties. The index order changes, so sort the picks if a later step needs an order.

Estimate: prefill +1 to +3 % at 32k, decode -0.1 to -0.4 ms/token. Risk: low to medium.

## Acceptance Criteria

- [x] A unit test shows that the selected set is the same as argPartition when there are no ties.
- [x] Prefill tok/s and decode ms/token at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance