---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'DeepSeek-V4 speed: mHC collapse in 2 dispatches and expand in 1 (fused norm, projection and expand kernels)'
---
## What

Found on 2026-10-14 in a study of osaurus (scratchpad `osaurus/REPORT.md`, idea 3).

Our collapse is a float32 cast, `rmsNorm`, a matmul, the Sinkhorn kernel and a cast (`DeepSeekV4HyperConnection.swift` about 399-419). Our expand is a batched 4x4 matmul (`DeepSeekV4CompiledDecode.swift` about 99-103). Osaurus does the collapse in 2 dispatches and the expand in 1 for GLM-5.3 (vmlx-swift commits 01abc39f, f8130127; `MHCFusedDecode.swift:28-154`, `MHCFusedPrefill.swift:33-187`). Much of their decode gain came from replacing a 1-lane Sinkhorn, which ^x5nv4x8 already did.

Estimate at 32k: decode -0.5 to -1.0 ms/token, prefill +1 to +3 %. The fused projection changes the sum order, thus it is not bit-exact. Risk: medium. Do this after ^yyhqkak.

## Acceptance Criteria

- [ ] A unit test compares the fused kernels with the op path (tolerance), float32 and bfloat16.
- [ ] Decode ms/token and prefill tok/s at 32k, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance