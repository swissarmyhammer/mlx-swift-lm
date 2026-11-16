---
assignees:
- claude-code
depends_on:
- 01M7J83VR4YW1XXC8EAFTJ9MAE
position_column: todo
position_ordinal: '8780'
title: 'Qwen3.8 MTP verify: make rows 3+ of the multi-row quantized matvec cheap, bit-exact'
---
## Why
Measured verify forward cost at 2k: L=1/2/3/4 is 22.5/24.1/29.4/33.4 ms. The second row costs 1.6 ms on MLX's 2-row kernel. Rows 3 and 4 add 5.3 and 4.0 ms on our bit-exact multi-row kernel, whose cost grows linearly with each row. Verify is about 28 ms of a 32 ms MTP round at 32k.

## What
In `Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift`:
- Share the weight load and unpack across rows. Keep each row's accumulation order identical to the single-row kernel, so the output stays bit-exact.
- Try more rows per threadgroup and other tilings.
- Also benchmark the merged MLX qmv_nax kernels for 6–16 rows (ml-explore/mlx,, and when it lands). If they are faster but not bit-exact, record that and don't switch.

## Acceptance
- [ ] Microbench: L=3 and L=4 verify forward in ms, before and after.
- [ ] `qwencheck.sh`: MTP greedy ids identical to the golden file, and cache reuse passes. The parity unit test (exact) passes. All five bundles are green.
- [ ] `qwenbench.sh` delta on this card. #qwen-perf