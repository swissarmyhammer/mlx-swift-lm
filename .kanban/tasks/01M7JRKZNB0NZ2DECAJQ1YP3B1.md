---
assignees:
- claude-code
depends_on:
- 01M7J840QAGWRNMGV5HETS0CJP
position_column: todo
position_ordinal: '8e80'
title: 'Qwen3.8 MTP verify: run 2-row verifies through the bit-exact multi-row kernel'
---
## Why
^ets0cjp rebuilt `MultiRowQuantizedMatvec` (inputs staged in threadgroup memory across 8 SIMD groups, next-block weight prefetch, half2 nibble decode). In the step matmul microbench ($S/mrbench, Qwen3.8-27B 4-bit g64, M5 Ultra) the new kernel runs 2 rows in 14.6 ms, bit-exact, while MLX `qmv_wide` takes 17.6 ms and differs from a row alone in the last bits. `fewestRows` is still 3 because the old kernel made the 2-token MTP verify ~11% slower end to end.

## What
- Set `MultiRowQuantizedMatvec.fewestRows` to 2 and update its doc comment and the tests that assume 3 (`MultiRowQuantizedMatvecTests`, `MultiRowProjectionFixture`, `Qwen3NextMultiRowMLPTests`, `MultiRowQuantizedMatvecExactProductTests`, the IntegrationTesting benchmark's `qmvWideRows`).
- The MTP greedy ids of L=2 verifies will change (they become those of a plain decode). Regenerate the MTP golden on purpose and record why, or confirm the ids stay identical.

## Acceptance
- [ ] `qwencli verify` L=2 mtp-verify-state before/after at 2k/32k/64k.
- [ ] `qwencheck.sh` passes (golden policy decided and recorded); parity tests exact; all five bundles green.
- [ ] `qwenbench.sh` delta on the card. #qwen-perf