---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'Flint: decode after a chunked prefill differs from a cold prefill by up to 1.3 float32 logits'
---
## Why
- Found in ^nanqj4p and present at HEAD c8746a30: scratchpad `flintcli <oQ4e> warm long_ids.txt` with `MLX_ENABLE_TF32=0` and float32 parameters gives `decode x300 pos 1700..2000 max|d| 1.3005` against a cold one-shot prefill of the same 3600 ids. The 512-token prefill chunks before it stay at 6e-4..5e-3.
- A float32 decode should agree with the cold forward to ~1e-3, so either the cold banded prefill or the rotating-cache decode reads the wrong keys somewhere after the window wraps.
- The HEAD and ^nanqj4p binaries print the same numbers to 4 decimals, so the kernel of ^nanqj4p does not cause this.

## What
1. Find the first decode position where |d| jumps, and which layers (sliding vs full) diverge first. Compare against the mlx-lm Python reference (scratchpad pyref) on the same ids.
2. Fix the path that is wrong. Write the failing test first, with a tiny config whose window wraps.

## Acceptance
- `warm` decode x300 at float32 has max|d| in the range of the prefill chunks (~1e-2 or below).
- Greedy ids on oQ4e/oQ8e are unchanged, or any change is explained against the Python reference.
- All five bundles are green. #flint-perf