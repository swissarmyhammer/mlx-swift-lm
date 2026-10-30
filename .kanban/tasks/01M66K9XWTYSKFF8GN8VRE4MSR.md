---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'MTP: cut the per-row cost of the multi-row feed-forward and GDN-projection matvec in the verify forward'
---
Found by ^wcmxg07 (Release, M5 Ultra, Qwen3.8-27B-4bit; MTPVerifyForwardProfileTests, text path). The verify attention kernel has removed most of the attention growth. Now the largest per-row cost of a multi-row verify is the 4-bit multi-row matvec of the feed-forward and the GDN projections. It adds about 2.6 ms per verify row at every context:
- feed-forward over 64 layers, rows 1 → 5: 10.0 → 20.6 ms (32k after the kernel change: 9.77/12.81/14.55/17.48/20.08)
- GDN projections, rows 1 → 5: 3.4 → 7.6 ms (100k: 3.85/4.25/5.91/6.51/7.53)
In theory a 4-bit matvec with a few rows reads the weights once, so 5 rows should cost close to 1 row.

- [ ] Find which quantized matmul kernel MLX picks for M = 2..5 rows (qmv vs qmm) for the FF and GDN projection shapes. Measure each one alone.
- [ ] Try a multi-row qmv that reads each weight tile once for up to 8 rows, or a better split. Keep the output within bf16 tolerance of the current path.
- [ ] Re-run MTPVerifyForwardProfileTests, MTPRoundCostBreakdownTests and the sweep before and after.

#mtp #perf