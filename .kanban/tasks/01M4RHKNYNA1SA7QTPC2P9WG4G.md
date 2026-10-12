---
assignees:
- claude-code
position_column: todo
position_ordinal: '8580'
title: Unit test that reproduces the cross-thread lazy cache state crash after prefill
---
## What

`TokenIterator.prepare` now evaluates the cache state together with the first token (`Libraries/MLXLMCommon/Evaluate.swift`, `evaluateTokenAndLogProbabilities`). Before that change, every DeepSeek-V4 generation on `stable` stopped the process with "There is no Stream(gpu, N) in current thread" at the first decode step.

The fix has proof only on the real weights (`mlx-community/DeepSeek-V4-Flash-4bit`, 2026-10-12, a scratch probe in the session scratchpad): prefill on one thread, then `eval` of each cache state array on a new thread.
- Without the fix: 21 arrays fail. All of them are slot 5 of `DeepSeekV4Cache` on the ratio-4 layers, which is the indexer pool, shape `[1, 3, 128]`. With a short prompt, the indexer selects no chunks, so the first token does not read that pool.
- With the fix: 0 arrays fail.

No unit test reproduces the crash yet. Both attempts passed with and without the fix:
1. A `KVCacheSimple` subclass that keeps a lazy sum of its keys in `state`.
2. A tiny float DeepSeek-V4 (compress ratios `[0, 0, 4, 16]`, `index_topk` larger than the chunk count), with prefill in `Task.detached` and decode through `generateTask`.

## Mechanism (from mlx-swift 0.32.3 sources)

- A Swift pool stream comes from `new_thread_unsafe_stream`. Its Metal encoder goes in the global map (`backend/metal/eval.cpp:22-26`), thus every thread can use it.
- A C++ default stream comes from `new_stream` through `default_stream(Device)`, which is `thread_local` (`stream.cpp:18-50`). Its encoder goes in the thread-local map of the thread that made it (`eval.cpp:15-20`).
- Thus an op that the C++ side puts on its own default stream makes an array that only the prefill thread can evaluate. Find which op in the real path does this. Candidates: the quantized layers (the tiny model is not quantized), `depends` (`ops.cpp:6415`), `compile` (`compile.cpp:373`), and `metal_kernel` without a stream (`metal_kernel.cpp:26`).

## Acceptance Criteria

- [ ] A unit test that fails (the process stops, or `withError` catches the error) when the cache-state evaluation in `prepare` is removed, and passes with it.
- [ ] The op that puts work on a thread-local stream is named in the test comment.
- [ ] If the op is in mlx-swift and not intended, record an upstream issue.
#deepseek-v4