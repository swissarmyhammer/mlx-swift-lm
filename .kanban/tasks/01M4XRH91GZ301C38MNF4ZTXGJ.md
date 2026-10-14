---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xszn3ptjxc94jpefkk0ca6
  text: |-
    Note from the osaurus study (scratchpad `osaurus/REPORT.md`) on 2026-10-14:
    - The kernel must read only the visible window range of each query (see the new window-slab task), or it keeps the waste of scoring all window keys of the block.
    - Reference kernels in vmlx-swift (scratchpad `osaurus/vmlx/`): `vmlx_dsv4_heads16_prefill` at `DeepseekV4MathHelpers.swift:1604-1805` loads one row at a time and is slow. The 32-key-tile sparse kernel at `NaiveN05FlashPrefillAttention.swift:79-140` is a better start.
  timestamp: 2026-10-14T18:14:52.790069+00:00
- actor: claude-code
  id: 01m4y1x37a9m9z2p7cw76j0yqm
  text: |-
    Picked up. Research done.
    - Window keys at prefill: `RotatingKVCache.updateConcat` gives `[earlier rows (min(liveRows, window - 1)) | block rows]` in time order. The window mask is `createCausalMask(n, offset: earlier rows, windowSize)`. Thus query `t` sees the window rows `[max(0, earlier + t - window + 1), earlier + t]`: one range of at most 128 rows. The kernel can compute the range from the shapes and needs no mask array. At decode the ring holds at most `window` rows, and each row is visible (mask `.none`).
    - HCA layers (no indexer, or pool <= topK): query `t` sees the pooled rows `[0, (offset + t + 1) / ratio)`. Also one range.
    - Baseline at HEAD 5f3a3c8 (GPU idle, 32k, `<snap> 32768 128 8192 1024`): run 1 prefill 439.1 tok/s, peak 145 GiB, decode 30.4 ms/token; run 2 prefill 439.6 tok/s, peak 145 GiB, decode 30.3 ms/token. Cache checks PASS (both). Profile window: 31.7 and 31.7 ms/step (prefill 436.1 and 436.0 tok/s). Logs: scratchpad `f4z-before-*.log`.
  timestamp: 2026-10-14T20:33:17.546933+00:00
- actor: claude-code
  id: 01m4y28zv7eap4rvz4c3dmzam6
  text: 'Micro-benchmark before (HEAD 5f3a3c8, Release, one attention call, last 32k prefill chunk: 2048 queries at offset 30720, 2175 window keys, CSA pool 8192 with 512 picks, HCA pool 256; bf16): CSA ranked gather 43.0 ms (peak +2.57 GiB), HCA masked joined 34.7 ms, window-only 30.8 ms. Decode (1 query, offset 32768): CSA picked gather 0.075 ms, HCA every chunk 0.080 ms, window SDPA 0.051 ms. Thus attention is about 1.65 s of the 4.66 s of one 2048-token prefill step at 32k. Bench: scratchpad `dsv4bench/Sources/dsv4bench/MicroAttention.swift`, built with `ENABLE_TESTABILITY=YES` and the `MICRO` condition into `build-micro`; log `f4z-micro-before.log`.'
  timestamp: 2026-10-14T20:39:47.303350+00:00
- actor: claude-code
  id: 01m4y2zefw5cdcb79pgawmhhmw
  text: |-
    Kernel landed in `Libraries/MLXLLM/Models/DeepSeekV4SparseAttentionKernel.swift`; tolerance tests in `Tests/MLXLMTests/DeepSeekV4SparseAttentionKernelTests.swift` pass (12 cases float32 and bfloat16, with positive controls).
    Design: one threadgroup = one query token x 16 heads, 8 SIMD groups, each SIMD group holds a 64-column slice of Q and of the output; 32-key tiles (one lane for each key), partial scores meet in threadgroup memory, online softmax from the sink, K read as V. Window rows: only `[row - window + 1, row]` from the shapes.
    What did not work:
    - First version without `#pragma clang loop unroll(full)`: correct but 628 ms for one CSA prefill call (op path 43 ms). The register arrays of SIMD-group matrices spill when the loops are not unrolled. With the pragmas and paired bf16 loads: 21.4 ms.
    - 128-column slice with 4 SIMD groups: slower (CSA 33.9 ms, HCA 17.7 ms), more register pressure. Kept 64.
    Micro-benchmark (64-column version, bf16, last 32k prefill chunk): CSA 21.4 ms (op 43.0), HCA 12.1 ms (op 34.7), window 4.0 ms (op 30.8), peak +0.13 GiB (op CSA +2.57 GiB). Decode: CSA 0.052 ms (op 0.072), HCA 0.034 ms (op 0.079), window 0.024 ms (op 0.050). Max |kernel - op| at prefill (bf16 outputs): 0.008 / 0.012 / 0.016; decode 0.002 / 0.003 / 0.004.
  timestamp: 2026-10-14T20:52:03.196287+00:00
- actor: claude-code
  id: 01m4y3yt643fk4ztm673sk2kxn
  text: |-
    Wired into DeepSeekV4Attention (prefill and decode) and ran the 32k bench one time: prefill 553.1 tok/s (+26 % from 439.1), peak 143 GiB, cache checks PASS, but decode 41.3 ms/token (from 30.4). Cause: a decode call gives only 4 threadgroups (64 heads / 16), thus it runs on 4 GPU cores. The first micro-benchmark measured 20 independent calls in one eval (throughput), not latency; a chained micro-benchmark (each call reads the output of the call before it) shows the real decode latency: kernel CSA 0.412 ms against op 0.078 ms.
    Decode attempt 2 (flash-decoding split): one threadgroup for each 32-key tile writes its partial softmax state, a second small kernel joins the states and the sink. Tests passed (float32 and bfloat16). Chained latency: CSA 0.084 ms (op 0.079), HCA 0.082 ms (op 0.088), window 0.073 ms (op 0.059). About 3.55 ms for each token on both paths, thus not faster. Removed again: decode keeps the op path. A next step could fuse the combine pass into the key pass (the last threadgroup joins the states), which saves one dispatch.
    Decision: the kernel runs only when a call gives at least 256 threadgroups (prefill blocks); decode and short blocks keep the op path.
  timestamp: 2026-10-14T21:09:10.980579+00:00
- actor: claude-code
  id: 01m4y4vqvhgvyyg6mwttnzwhkf
  text: |-
    ### implement — changed
    - files: `Libraries/MLXLLM/Models/DeepSeekV4SparseAttentionKernel.swift` (new: one Metal kernel, three pool forms: window only, every visible chunk, picked chunks; `canRun(queries:)` needs the GPU, an even head width and at least 256 threadgroups), `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift` (prefill blocks of both layer kinds read the kernel; decode, short blocks and the CPU keep the op paths; docs), `Tests/MLXLMTests/DeepSeekV4SparseAttentionKernelTests.swift` (new), `Tests/MLXLMTests/DeepSeekV4AttentionTests.swift` (routing test: a 256-token prefill block dispatches 1 custom kernel on the GPU, the decode step 0, the CPU 0).
    - The kernel reads the window values the cache gives back (a DeepSeek-V4 cache gives the keys again), thus `attentionReadsTheArraysTheCacheReturned` holds; each pooled chunk is its own value.
    - No op-path code became dead: decode, blocks of fewer than 256 threadgroups and the CPU still read it.
    - Micro-benchmark (one call, bf16, last 32k prefill chunk): CSA 21.5 ms (op 43.3), HCA 12.1 ms (op 34.9), window 4.5 ms (op 31.0); peak +0.13 GiB (op CSA +2.57 GiB). Decode (chained latency): the kernel was not faster (see the comment above), thus decode keeps the op path. Max |kernel - op| on bf16 outputs: 0.012 / 0.012 / 0.016 (the op path rounds in bf16 too); unit tests: float32 within 1e-4, bfloat16 within 1e-2, positive controls show the hidden keys move the output past 1e-2.
    - 32k bench `<snap> 32768 128 8192 1024`, GPU idle: before prefill 439.1 / 439.6 tok/s, peak 145 GiB, decode 30.4 / 30.3 ms/token; after prefill 559.9 / 559.5 tok/s (+27 %), peak 143 / 143 GiB, decode 30.4 / 30.3 ms/token. Cache checks PASS (both modes, both runs). Profile window: before 31.7 / 31.7 ms/step (prefill 436.1 / 436.0 tok/s), after 31.7 / 31.9 ms/step (prefill 553.3 / 553.4 tok/s). Logs: scratchpad `f4z-*.log`.
    - Tests: `swift build --build-tests` 0 compiler warnings; `xcrun xctest` XCTest 716 tests 0 failures, Swift Testing 1449 tests passed. `swift format lint` clean on the four files. Greedy parity `greedyFirstTokensMatchThePythonFixture()` passed.
    - next: review. Follow-up idea for decode: fuse the combine pass of a split-key decode kernel into the key pass.
  timestamp: 2026-10-14T21:24:58.865187+00:00
- actor: claude-code
  id: 01m4y51vvm5drdasjd6106evhr
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (e0ad34d). 0 findings, 0 confirmed, 1 refuted, 7 attempted, 0 failed. 4 files reviewed. 2 kanban files had no validator.
    - next: The task is in done. No work remains.
  timestamp: 2026-10-14T21:28:19.572399+00:00
- actor: claude-code
  id: 01m4y52655j5wrzy6kgsb6azx7
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 4 files; 32k prefill 439.1 → 559.9 tok/s, peak 145 → 143 GiB, decode 30.4 ms/token (no change)
    - test: green — xcrun xctest MLXLMTests, 716 XCTest and 1449 Swift Testing passed, 0 failures; greedy parity passed
    - commit: e0ad34d
    - review: clean — 0 findings
  timestamp: 2026-10-14T21:28:30.117709+00:00
position_column: done
position_ordinal: ffcf80
title: 'DeepSeek-V4 speed: one fused Metal sparse attention kernel for head dim 512 (window + picked pool rows + sink, online softmax)'
---
## What

Found on 2026-10-14 in a study of llama.cpp (scratchpad `llamacpp/REPORT.md`, item 1).

MLX has no fused SDPA for head dim 512 (only 64-256, `mlx/backend/metal/scaled_dot_product_attention.cpp:662-688`). Thus all our attention calls use matmul, softmax, matmul, with a gather copy and a float32 score tensor. llama.cpp uses one Metal kernel that reads the window rows and the picked pool rows by index, does online softmax with the sink, and uses K as V (PR 28098, PR 29377; `ggml-metal-ops.cpp:3171-3230`, `kernels/fa_vec_common.metal`). PR 28098 took prefill at 32k from 170 to 348 tok/s on M2 Ultra.

Our places: `DeepSeekV4Attention.swift` `attentionOverRankedChunks`, `attentionOverPickedChunks`, `attentionOverJoinedKeys`. A dense variant serves the window-only layers.

Estimate: prefill +5 to +12 % at 32k, decode -0.5 to -1.5 ms/token. Not bit-exact (sum order changes). Risk: medium to high.

## Acceptance Criteria

- [x] A unit test compares the kernel with the op path (tolerance), with hidden picks and a query that sees no chunk.
- [x] Prefill tok/s, peak memory and decode ms/token at 32k, before and after, GPU idle.
- [x] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance