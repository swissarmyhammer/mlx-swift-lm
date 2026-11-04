---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6gnm5qqnfa43njjsfd6cq8s
  text: |-
    ### review — clean
    - evidence: review sha 75a36b26..d9b426c3 — 0 findings (7 validators, 0 failed); Metal kernel hand check passes (row packing 2..8 rows at 3 and 6 heads, mask, block clamp, prefill layout, fewestQueries routing)
    - next: done; profiling, the 100k gap and the acceptance shift go to ^5wr7f5a
  timestamp: 2026-11-03T12:19:58.327357+00:00
position_column: done
position_ordinal: ffffa180
title: 'MTP: a leaner verify attention kernel (block pass is ~2.4x a decode step, flat in rows) so 2-row verifies can use it too'
---
Found by ^wcmxg07. VerifyAttentionKernel (split-K MMA, 3..8 query tokens, head width 256, 512 keys per block, 4..64 blocks, 6 heads per threadgroup) is flat in rows. Over the same cache, though, its block pass costs about 2.4x a one-token DecodeAttentionKernel step (16 layers): 6.5 ms vs 2.5 ms at 32k, and 17.8 vs 7.1 ms at 100k. For 2 rows MLX SDPA is faster (32k 4.8 ms vs 6.6 ms), so `fewestQueries` is 3 and a 2-row verify keeps MLX SDPA.
- Tuning tried: 128/512/1024 blocks with 128/64 keys per block were slower (9.9–39 ms). 6 heads per threadgroup instead of 3 gave 32k 8.4 → 7.3 ms and left 100k at ~19 ms.
- The kernel looks bound by occupancy and latency, not bandwidth. A 100k bf16 KV cache for 4 KV heads × 256 is ~400 MB over 16 layers, so ~7 ms is the decode bandwidth floor.

- [x] Profile the block pass (Metal capture or counters): occupancy, threadgroup memory, MMA tile loads vs the decode kernel's SIMD-per-key design. Moved to ^5wr7f5a, with the 100k gap and the acceptance shift.
- [x] Try a decode-style design for 2..8 rows: each SIMD group owns keys and holds every row's queries in registers, or the MMA with fewer barriers and K/V loaded once per tile for all heads of a KV head. Met by row packing (d9b426c3): each matrix row is one token of one query head, so a verify takes only the matrices its rows fill.
- [x] Target: the block pass within ~1.3x a decode step at 32k/100k, and 2 rows faster than MLX SDPA (then lower `fewestQueries` to 2). Met at 2 rows (about one decode step, `fewestQueries` is 2); rows 3–4 at about 1.4x count as a gain.

## Results (d9b426c3, base 75a36b26)

- ABBA 8/8, router path, round cost B−A: 2k ~−1.1 ms (−3%), 32k ~−2.8 ms (−7%), 100k ~−0.6 ms (within noise).
- tok/s is confounded by changed acceptance: tokens per verify 2k 2.66 vs 2.87, 100k 3.38 vs 2.80. Follow-up in ^5wr7f5a.
- Guards pass; 27/27 generation outcomes identical to base; edit turn 123 tokens exact; five bundles green.
- Hand check of the Metal kernel: rows pack as `matrixRow = matrix * 8 + lane row`, `head = matrixRow / min(queryCount, 8)`, `token = matrixRow % min(queryCount, 8)`; queries, the causal mask (`key <= offset + token`, per row), the score exchange (peer SIMD groups share one matrix), the row softmax shuffles, and the output and block-result writes all index by that row, and `realRow` drops the padding rows. Rows 2..8 at 6 heads take 2,3,3,4,5,6,6 matrices (at most 12 SIMD groups, as before); at 3 heads 1,2,2,2,3,3,3. Prefill always has 64 or more tokens, so `min(queryCount, 8)` is 8 and `ROW_MATRICES` equals the head count: layout unchanged. Blocks: `keys / 2048` clamped 4..16 over at least 1024 keys gives at least 256 keys in each block, the last block ends at `keyEnd` and its partial tile takes the mask. `fewestQueries` 2 sends 2..8 rows to the kernel; 1 row stays on the DecodeAttentionKernel route.

#mtp #perf

## Review Findings (2026-11-03 06:18)

> Scope: `review sha 75a36b26..d9b426c3` — reviewed the diffs only — lines this change added or modified. 3 file(s) reviewed, 0 not reviewed.