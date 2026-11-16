---
assignees:
- claude-code
depends_on:
- 01M7J83R03FRVF0510R6JF4G0C
position_column: todo
position_ordinal: '8680'
title: 'Qwen3.8 decode: fuse residual add + RMSNorm'
---
## Why
The decode graph has 162 Add and 161 RMSNorm operations per token. Most of the adds are residual adds that feed the next RMSNorm directly.

## What
Fuse `h = x + r; n = rmsnorm(h)` into one kernel that returns both `h` and `n`. Do it with a custom Metal kernel, or with mx.compile of the pair if that fuses, whichever measures faster. Use it in the decoder layers of Qwen35 (VLM and LLM) for decode and verify.

The result must match the current path exactly: bf16 rounding of `h` happens before the norm, exactly as it does today.

## Acceptance
- [ ] `qwencheck.sh`: greedy ids are identical in plain and MTP modes, and cache reuse passes.
- [ ] A kernel parity unit test passes. All five bundles are green.
- [ ] `qwenbench.sh` delta on this card. If there is no measurable gain, record it as a no-go and revert. #qwen-perf