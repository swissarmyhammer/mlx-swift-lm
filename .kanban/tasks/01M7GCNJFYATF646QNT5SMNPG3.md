---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7gf95nybytfa144g0pr9kvb
  text: |-
    Research (HEAD 4838f8db, oQ4e, float32 params, MLX_ENABLE_TF32=0, long_ids.txt):
    - Added scratchpad flintcli modes `decpos` (per-position decode |d| vs a cold one-shot of N ids) and `coldsave`.
    - `decpos 2000 1700 512 300` (cold = one-shot of the first 2000 ids): 512-chunk prefill rows max|d| 0.0000/0.0000/0.0000/0.0014, decode 1700..2000 max|d| 0.002 at every position. The warm path (chunked prefill + rotating-cache decode after the 513-window wraps) agrees with a cold prefill of the same length.
    - `decpos 3600 1700 512 300` (cold = one-shot of 3600 ids, as in `warm`): only rows 1975:1.300 1976:0.282 1977:0.106 1978:0.105 exceed 0.05.
    - Root cause: MoE routing near-tie. At position 1975 the router of layer 30 has a 6th-vs-7th expert choice margin of 2e-6 (naive3600) / 9e-8 (naive2000). Float32 rounding that depends on the GEMM row count (2000 vs 3600 rows) flips the selected expert, and the flip cascades (routing then differs in layers 30,35,36,40,44,45) to 1.3 logits at row 1975, decaying to 0.04 by 1979.
    - The ground truth shows the same flip: scratchpad pyscripts/flip.py (mlx-vlm, float32, naive attention = explicit softmax+sinks+window): naive2000 vs naive3600 rows 1970..1979 max|d| [0,0,0,0,0,1.3,0.282,0.106,0.105,0.044]. Swift cold2000 == naive2000 (0.0 every row), Swift cold3600 == naive3600 == mlx-vlm fast2000 == fast3600 (0.0 every row). Swift warm decode == Swift cold2000. So neither Swift path is wrong; the 1.3 is a discontinuity of top-k routing at a tie, not a cache/mask bug.
    - The later `warm` phases (35 logits) follow `fed N junk tokens, trimmed 0`: a RotatingKVCache without a rewind reserve cannot trim after it wraps, so the junk keys stay and the comparison against a cold run without them is not meaningful.
  timestamp: 2026-11-15T20:44:48.190675+00:00
- actor: claude-code
  id: 01m7gfdr1jnmpdqg0x50jdc2nf
  text: |-
    Bisect (a): stable 71692319 (before any flint-perf change; scratchpad flintcli716 built against a temporary `git worktree`, since removed) gives the same picture. With `decpos 2000 1700 512 300`, the decode max|d| is 0.001 and no row is above 0.05. With `decpos 3600 1700 512 300`, rows 1975:1.3, 1976:0.282, 1977:0.106 and 1978:0.105 are above 0.05, and no other row is. So the flint-perf work (SlidingWindowBands 6535f367, FlintModel.prepare 6adfaa83, ^nanqj4p) did not introduce this.
    (b) The chunk size and cache wrap do not matter. The warm schedule (512-chunk prefill to 1700, then single-token decode after the 513 window wraps) matches a cold one-shot of the same 2000 ids to 0.002, both at HEAD and at 71692319. Only the length of the cold reference (3600 vs 2000 rows) moves row 1975.
    (c) This is not an attention problem in sliding or full layers. The first change is in expert selection at MoE layer 30, which has a top-6/7th margin of 2e-6 at pos 1975. After that, routing also differs in layers 35, 36, 40, 44 and 45.
    Against cold3600, every decode row other than 1975..1979 has a max|d| of 0.024.
    Conclusion: there is no defect. The difference comes from top-k routing at a near-tie: float32 rounding differs between GEMMs with 2000 rows and with 3600 rows. The naive ground truth (explicit attention, float32) has the same 1.3-logit flip between its 2000-row and 3600-row runs. There is no code change and no test, because no code path is wrong for a failing test to pin. The acceptance "warm decode x300 max|d| ~1e-2" is not reachable against a reference of a different length when the reference is a coin flip at this row. The warm decode meets it against a cold reference of the same length (0.002).
  timestamp: 2026-11-15T20:47:18.066838+00:00
- actor: claude-code
  id: 01m7gfdvp9em10gkyjc0zjsrzd
  text: |-
    ### implement — no-change
    - evidence: root cause is an MoE router near-tie (layer 30, pos 1975, 6th/7th margin 2e-6). The expert choice flips with float32 GEMM rounding that depends on row count. The naive float32 ground truth shows the same 1.3 between its 2000-row and 3600-row runs. Swift cold2000 == naive2000 and Swift cold3600 == naive3600 == mlx-vlm (0.0 per row). Warm decode vs a cold reference of the same length (2000 ids): max|d| 0.002 at HEAD and 0.001 at 71692319. Against cold3600 it is 1.3 at both commits, only at rows 1975..1978, and 0.024 elsewhere. No repo files changed, so no tests, flintcheck or flintbench were run: greedy ids and speed are unchanged by construction. Scratchpad: pyscripts/flip.py, flintcli modes decpos/coldsave.
    - next: a human decides whether to close the task or to relax its acceptance (compare against a cold reference of the same length, or exclude router-tie rows).
  timestamp: 2026-11-15T20:47:21.801780+00:00
- actor: claude-code
  id: 01m7gfep9k8k1wth6t2jge0m5k
  text: |-
    ### Resolution — not a defect
    The 1.3-logit gap at rows 1975–1978 is an MoE router near-tie (layer 30, 6th/7th expert margin 2e-6) that flips with row-count-dependent float32 rounding. The reference (mlx-vlm, naive attention, float32) shows the identical 1.3 between its 2000- and 3600-id runs; Swift matches the reference bit-exactly at both lengths (0.0/row), and warm decode matches a 2000-id cold reference to 0.002. Same numbers at stable 71692319, so flint-perf did not cause it. The card's acceptance (max|d| ~1e-2 vs a 3600-id cold reference) cannot be met by any correct implementation at that row; it holds against the equal-length reference. No code change; test step not needed (empty diff).
  timestamp: 2026-11-15T20:47:49.043993+00:00
- actor: claude-code
  id: 01m7gfeyqk3q1ss2y92mzre226
  text: |-
    ### commit — changed
    - evidence: local commit on stable (no push), subject: `chore(kanban): record that the Flint chunked-prefill logit gap is an MoE router near-tie, not a defect (^5smnpg3)`; files: .kanban/tasks/01M7FHG7TA2MGDAS4Q5NANQJ4P.{jsonl,md}, .kanban/tasks/01M7GCNJFYATF646QNT5SMNPG3.{jsonl,md}. The sha is the commit with this subject (see git log).
    - next: none, investigation record only; no code changed.
  timestamp: 2026-11-15T20:47:57.683986+00:00
position_column: doing
position_ordinal: '80'
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