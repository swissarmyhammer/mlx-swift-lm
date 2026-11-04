---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66d3c427r3fkgv3m1gd2kb3
  text: |-
    ### closed — user decision 2026-10-30
    User does not want the adaptive guard at all; it is being removed, so reworking it is no longer wanted.
  timestamp: 2026-10-30T12:38:34.882361+00:00
position_column: done
position_ordinal: ffff9080
title: 'MTP adaptive guard: stop costing more than it saves (pause/probe transitions, short-turn probe, block steps at long context)'
---
The `.enabled` adaptive guard (^tdj3waa iteration 3: window verdicts at rate ± 2·SE) failed its shipping condition in the 2026-10-30 Release measurements, so `MTPAdaptivePolicy.default` stays `.disabled`. Data (router path unless noted; loads 5–38 from other sessions):

- 2k–100k sweep: guard off ≥ plain at all 10 points (1.06–1.80x); guard on below guard off beyond noise at text 8k (1.58 vs 1.80), text 100k (1.08 vs 1.31), router 32k (1.04 vs 1.15), 64k (0.97 vs 1.24), 100k (0.95 vs 1.27); 2–4 pauses and 460–691 plain tokens of 1024 at those points.
- Synthetic CPU load, 32k: guard off 1.05 / 0.93 / 1.10 of plain at 0/50/100% busy cores; guard on 0.99 / 0.83 / 1.01 with 662–735 plain tokens of 1024. At 50% the guarded mix (72% plain tokens) ran at 38.9 tok/s, below both pure plain (46.8) and pure MTP (43.9).
- ChatSession follow-ups (64–192-token turns, quiet): guard on never paused but ran ~0.96 of guard off — one 4-token probe per turn.
- Round cost breakdown (32k quiet): verify by rows 1..5 = 21.9/26.9/36.1/41.7/49.2 ms, drafter proposal 1.4–5.6 ms, host sync 0.24 ms (0.55–1.05 ms with 30 busy threads). Guard-off rounds measure 55 ms at effective block 2.71 in the iterator vs ~37 ms of verify + draft: ~18 ms per round of iterator overhead, far above the host syncs.

To do, deterministic replay tests first (extend Tests/MLXLMTests/MTPAdaptiveReplayTests.swift with measured per-unit costs):
- [ ] Measure where the guarded mix loses: per-unit timing of paused plain steps vs plain decode, drafter catch-up at resume, first round after a probe/pause. Expose a verdict trace (window rate, SE, plain estimate, context, block size) so a real run shows why each pause happened.
- [ ] Do not probe on short turns before a loss is plausible (or reuse the plain estimate across turns of one session) so follow-ups stay at guard-off speed.
- [ ] Make a transition pay for itself: count the transition costs in the verdict, or require a loss larger than those costs before pausing.
- [ ] Re-run `MTPSWEBenchScaleTests/decodeRateFromTwoToOneHundredThousandContextTokens`, `decodeRateUnderSyntheticCPULoad`, and `qwen38FourBitMTPSystemPromptCache(adaptivePolicy:)`; ship `.enabled` as default only when guard on ≥ guard off within noise and ≥ plain at every point.

#mtp #perf