---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84ap0tzr75rjmtvsjqt2yta
  text: 'Already implemented. `Inkling2Experts.outputs(_:_:)` sorts the routed rows by expert with `gatherSort` once `indices.size >= sortedRowsThreshold`, calls `gatherQuantizedMM(..., sortedIndices: true)` and unsorts with `scatterUnsort`, or hands the order to `inkling2RowConvolution` on the prefill path. Nothing to do.'
  timestamp: 2026-11-23T13:49:14.975293+00:00
depends_on:
- 01M8487ZP6NSQ1E0FEY1CEY7KX
position_column: done
position_ordinal: ffffe280
title: 'Inkling 2 prefill: sort the tokens by expert before the gathered expert matmuls'
---
## Why
With 2048 tokens × 8 experts per chunk, each of the 128 experts sees ~128 rows. In mlx-lm, SwitchGLU sorts the routed rows by expert (`sorted_indices=True`) once a chunk holds more than ~64 tokens. `gatherQuantizedMatmul` then reads each expert's weight once per contiguous run, not once per row.

## What
- Check whether `Inkling2Experts` prefill (`mlp.prefill`) sorts already.
- If it does not, sort the rows by expert, call gather_qmm with `sortedIndices: true`, and scatter the results back.
- Measure on the prefill bench at 2k (MoE-bound) and at 64k.

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS. If bf16 rounding moves ids, the card shows that the differences are near-ties, as ^5smnpg3 did.
- [ ] inklingbench.sh shows a prefill gain beyond the noise. #inkling-perf