---
assignees:
- claude-code
position_column: todo
position_ordinal: '8680'
title: 'MTP: remove the ~18 ms per-round iterator overhead at 32k context (55 ms measured vs ~37 ms of verify + draft GPU work)'
---
From ^tdj3waa iteration 4 (Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, router path): at 32k context an MTP round measures ~55 ms while its GPU work (verify forward + drafter steps, from MTPRoundCostBreakdownTests) is ~37 ms; host syncs are only 0.22–0.25 ms quiet (0.55–1.05 ms fully loaded). The ~18 ms gap is iterator overhead between GPU work, and it is what decides MTP vs plain at long context and under CPU load (unguarded MTP 0.93x plain at 32k with half the cores busy).

- [ ] Profile one MTP round at 32k end to end (signposts / os_signpost + Instruments Metal System Trace, or GPU-synced timers around each phase): graph building, cache bookkeeping (snapshot slots, rewind, KVCacheStorage), drafter commit, sampling/acceptance, array conversions, lazy-eval breaks that serialize CPU and GPU.
- [ ] Remove the dominant overhead (keep the GPU busy: build the next drafter step / verify graph while the previous evaluates; avoid redundant evals and host copies) with output identical to today (greedy up to the near-tie rule, sampled distribution unchanged), cache/drafter alignment intact.
- [ ] Re-measure: MTPRoundCostBreakdownTests, the 2k–100k sweep, and MTPCPULoadTests at 0/50/100% load; target round time close to the GPU work and MTP ≥ plain at 32k under 50% load.

#mtp #perf