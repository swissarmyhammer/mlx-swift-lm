---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'ShardCheck comparator: statistical gate for long runs'
---
The 1024-step 2-Mac runs (17p77c9) show no drift, but isolated single-step outliers trip the per-step gate: Inkling KL 0.0764 at one step of 1025; a Qwen argmax flip at a 3-ulp margin with KL 0.017. Keep the strict gate for short runs and add a statistical gate for long ones.
- [ ] For runs over 64 steps, gate on all of these: mean KL ≤ 0.005; p99 KL ≤ 0.05; max KL ≤ 0.2; at most 0.5% of steps above 0.05.
- [ ] Argmax: a mismatch counts only when the reference margin is above the tie rule AND KL at that step is above 0.02. Mismatches that don't count are reported as "noise flip". At most 0.5% of steps may be noise flips.
- [ ] Short runs (≤ 64 steps) keep today's gate unchanged.
- [ ] Print per-window max/mean (0–32, 33–128, 129–256, 257–512, 513–1024) so drift is visible.
- [ ] Extend scripts/test_shard_check_compare.py: an isolated outlier passes in long mode and fails in short mode; a steady rise in KL fails on mean or p99; a confident mismatch fails. #cluster