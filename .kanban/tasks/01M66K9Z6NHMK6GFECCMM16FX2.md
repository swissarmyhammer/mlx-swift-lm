---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'MTP: a leaner verify attention kernel (block pass is ~2.4x a decode step, flat in rows) so 2-row verifies can use it too'
---
Found by ^wcmxg07. VerifyAttentionKernel (split-K MMA, 3..8 query tokens, head width 256, 512 keys per block, 4..64 blocks, 6 heads per threadgroup) is flat in rows. Over the same cache, though, its block pass costs about 2.4x a one-token DecodeAttentionKernel step (16 layers): 6.5 ms vs 2.5 ms at 32k, and 17.8 vs 7.1 ms at 100k. For 2 rows MLX SDPA is faster (32k 4.8 ms vs 6.6 ms), so `fewestQueries` is 3 and a 2-row verify keeps MLX SDPA.
- Tuning tried: 128/512/1024 blocks with 128/64 keys per block were slower (9.9–39 ms). 6 heads per threadgroup instead of 3 gave 32k 8.4 → 7.3 ms and left 100k at ~19 ms.
- The kernel looks bound by occupancy and latency, not bandwidth. A 100k bf16 KV cache for 4 KV heads × 256 is ~400 MB over 16 layers, so ~7 ms is the decode bandwidth floor.

- [ ] Profile the block pass (Metal capture or counters): occupancy, threadgroup memory, MMA tile loads vs the decode kernel's SIMD-per-key design.
- [ ] Try a decode-style design for 2..8 rows: each SIMD group owns keys and holds every row's queries in registers, or the MMA with fewer barriers and K/V loaded once per tile for all heads of a KV head.
- [ ] Target: the block pass within ~1.3x a decode step at 32k/100k, and 2 rows faster than MLX SDPA (then lower `fewestQueries` to 2).

#mtp #perf