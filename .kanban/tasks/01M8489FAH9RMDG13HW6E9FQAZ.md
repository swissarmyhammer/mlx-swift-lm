---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84dqfs5dfjyfzb6e25g82sa
  text: |-
    ## Dropped after analysis: prefill is GPU-bound, and the expert GEMMs and attention dominate
    - xctrace Metal System Trace, 32k prefill: the GPU is busy **99.8%**. Unlike decode, the host is not the limit, so a compile only helps if the fused elementwise passes are a large share of GPU time.
    - Per 2048-token chunk the routed experts do 2048 × 8 × (2304×2304 + 1152×2304) × 2 ≈ 261 GFLOP per visit, ~12.5 TFLOP over 48 visits. At 4775 tok/s (0.43 s per chunk) that is ~29 TFLOP/s in MLX's sorted `gather_qmm`.
    - The elementwise passes (RMS norms, block readouts, residual adds) run over `[2048, 2304]` float32, ~19 MB each, ~15 passes per visit. That is ~0.3 GB per visit, ~0.3 ms against ~9 ms per visit: ≤3%.
    - The prefill levers are long-context attention (^rcf6183) and the expert GEMM efficiency, which is MLX's `gather_qmm` kernel. No card here.
  timestamp: 2026-11-23T14:42:28.773433+00:00
depends_on:
- 01M8487ZP6NSQ1E0FEY1CEY7KX
position_column: done
position_ordinal: ffffe980
title: 'Inkling 2 prefill: compile the elementwise steps of a prefill chunk'
---
## Why
Qwen gained by compiling the elementwise steps of prefill (2e42b386). The Inkling prefill layer path runs several elementwise steps on full `[2048, 2304]` float32 tensors, each a separate pass over memory:
- `unweightedRMSNorm` of the residual
- the block-attention residual readouts
- `completedBlocks` concatenation
- the post-attention and post-MLP convolutions, and the residual adds

## What
- Profile one prefill chunk to find the elementwise share.
- Fuse it with `compile(shapeless:)`, or fold it into the existing prefill kernels (`inkling2PrefillPrologue`, `inkling2RowConvolution`).

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS.
- [ ] inklingbench.sh shows a prefill gain beyond the noise. #inkling-perf