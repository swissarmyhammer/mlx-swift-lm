---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m886k85vm4ekhr29qfdm9rw0
  text: |-
    Implemented in scripts/shard_check_compare.py: `run_passes(steps, max_kl)` picks the gate by length (≤ 64 steps: the unchanged per-step `gate_passes`; longer: `long_gate_failures`, which returns the failing reasons). `counts_as_mismatch` = verdict MISMATCH (outside the tie rule) and KL > 0.02; `is_noise_flip` = any other argmax disagreement. Choice: a tie in a long run also counts toward the 0.5% noise-flip budget (the card says every mismatch that doesn't count is a noise flip). Long runs print per-window max/mean (0-32, 33-128, 129-256, 257-512, 513-1024, then doubling past 1024) and the gate statistics. max-kl (CLI arg) applies only to short runs.

    Tests (RED first: 12 errors for missing run_passes/is_noise_flip/window_stats; then GREEN): 14 new cases — isolated 0.0764 outlier passes 1025 steps, fails 64 steps; linear drift 0→0.02 fails (mean 0.01); one step at 0.3 fails max; 6/1025 above 0.05 fails; confident mismatch (KL 0.03) fails; noise flip (KL 0.017) passes and is labelled; 6 noise flips fail; window bounds and maxima. `python3 -I scripts/test_shard_check_compare.py`: 23 tests OK. ruff format --line-length 100 applied.

    ### implement — changed
    - evidence: 2 files — scripts/shard_check_compare.py, scripts/test_shard_check_compare.py; 23/23 self-tests OK
    - next: commit, then review
  timestamp: 2026-11-25T01:54:47.611711+00:00
- actor: claude-code
  id: 01m887qmswqxd8ysmds5ymbnq5
  text: |-
    ### commit — changed
    - evidence: ad8ac35d feat(shard-check): statistical gate for runs over 64 steps (not pushed). Later commit f525bcc9 (^ffn1stg) adds a load-peak gate to the same script; the long-run gate is unchanged.
    - next: review
  timestamp: 2026-11-25T02:14:40.188303+00:00
position_column: doing
position_ordinal: '80'
title: 'ShardCheck comparator: statistical gate for long runs'
---
The 1024-step 2-Mac runs (17p77c9) show no drift, but isolated single-step outliers trip the per-step gate: Inkling KL 0.0764 at one step of 1025; a Qwen argmax flip at a 3-ulp margin with KL 0.017. Keep the strict gate for short runs and add a statistical gate for long ones.
- [x] For runs over 64 steps, gate on all of these: mean KL ≤ 0.005; p99 KL ≤ 0.05; max KL ≤ 0.2; at most 0.5% of steps above 0.05.
- [x] Argmax: a mismatch counts only when the reference margin is above the tie rule AND KL at that step is above 0.02. Mismatches that don't count are reported as "noise flip". At most 0.5% of steps may be noise flips.
- [x] Short runs (≤ 64 steps) keep today's gate unchanged.
- [x] Print per-window max/mean (0–32, 33–128, 129–256, 257–512, 513–1024) so drift is visible.
- [x] Extend scripts/test_shard_check_compare.py: an isolated outlier passes in long mode and fails in short mode; a steady rise in KL fails on mean or p99; a confident mismatch fails. #cluster