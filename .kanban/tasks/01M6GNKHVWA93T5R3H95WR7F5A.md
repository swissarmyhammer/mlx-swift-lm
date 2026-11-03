---
assignees:
- claude-code
position_column: todo
position_ordinal: '9880'
title: 'MTP: profile the verify block pass and explain why the 100k round barely moves'
---
Follows ^mm16fx2 (row packing in VerifyAttentionKernel, `fewestQueries` 2, 2048 keys per block, 4..16 blocks). The ABBA 8/8 router-path round cost B−A was 2k ~−1.1 ms (−3%), 32k ~−2.8 ms (−7%), 100k ~−0.6 ms (within noise). The microbench predicted ~8 ms less attention at 100k.

- [ ] Profile the block pass (Metal capture or counters): occupancy, threadgroup memory, MMA tile loads vs the decode kernel's SIMD-per-key design.
- [ ] Explain the gap at 100k: the microbench predicted ~8 ms less attention, the real round dropped ~0.6 ms.
- [ ] Explain the acceptance shift between base and change: tokens per verify 2k 2.66 vs 2.87, 100k 3.38 vs 2.80, which confounds tok/s.

#mtp #perf