---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: 'Flint: stop casting the attention sinks and the expert bias on every forward'
---
## Why
Two casts run on every forward pass and repeat the same work each time:
- `FlintAttention` passes `sinks.asType(queries.dtype)`, one kernel per layer (49 per token).
- `FlintSparseMoeBlock` computes `expertBias.asType(.float32)`, one per MoE layer (47 per token).

Decode is kernel-count bound (~7.7 ms/token against a ~2.4 ms bandwidth floor on oQ4e), so ~96 dispatches per token matter.

## What
- Cast once, when weights load or on first use, keyed on dtype. For example, store the bias in float32 in `sanitize`, and cache the sinks in the activation dtype.
- Look for other per-step casts in the decode path at the same time (`routed.asType(.float32)`, `out.asType(x.dtype)`). Fold them into the compiled combine of the decode-compile card if they can't be dropped.
- Keep the float32 routing math the reference uses.

## Acceptance
- Greedy token ids are identical on oQ4e and oQ8e.
- Float32 logit parity is unchanged.
- ms/token is better than a same-session baseline (flintcli `bench`).
- All five test bundles are green. #flint-perf