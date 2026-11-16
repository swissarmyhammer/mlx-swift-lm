---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'Decode-heads kernels: divide by the norm width the way MLX fast math does'
---
## Why
Found on ^ftj9mae. SwiftPM builds the MLX `rms_*` kernels with fast math, which turns `acc / axis_size` into `acc * (1 / axis_size)`. For a width that is not a power of two this rounds differently from a true division: a probe of 256 float32 rows found 50-65 rows whose inverse differed by one ulp at widths 72, 5120 and 6000 (0 rows at 2048), and `total * (1.0f / WIDTH)` matched every row.

`Qwen35DecodeHeadsKernel` (`simd_sum(groupSum) / DIM`) and `FlintDecodeHeadsKernel` (`squares / DIM`) divide. They are bit-exact today only because their head widths are powers of two (256). A head width such as 384, which both init guards accept, would not be bit-exact.

## What
Write `* (1.0f / DIM)` in both kernels, as `ResidualRMSNormKernel` does. Add a parity case with a head width that is not a power of two (if the kernels accept it), or narrow the init guards to powers of two.

## Acceptance
- [ ] Both kernels multiply by the reciprocal, or reject such widths.
- [ ] A parity case covers it. All five bundles are green. #qwen-perf