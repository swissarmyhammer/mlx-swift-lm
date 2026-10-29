---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m644s8cp0arjvb6a1c4yev6w
  text: |-
    Phase 1 picked up in worktree agent-a24c62ee3c2f35eec (reset to mtp 268dab08).

    Research:
    - Per round host syncs today (Qwen chained, greedy, floor 0.75): seed top-prob readback (chainsAfterSeed .item), one readback per chained step (chainedMTPDrafts lowestTopProbability .item, up to blockSize-2 = 3), verify readback (BatchedMTPVerification). Up to 5 syncs/round; commit/seed draw stay lazy.
    - Plain TokenIterator pipelines (asyncEval next token before returning the previous one); the MTP sticky passthroughStep does NOT (eval + item each step). A pause that must not be slower than plain therefore needs a pipelined plain step.
    - Drafter catch-up after plain steps can be batched: ResumableMTPDrafterModel.commitDrafterState(targetHidden: rows, draftTokens: fed[1...], acceptedCount: n-1, finalToken) appends n entries in one drafter forward (the guided commitDrafter already does exactly this for forced-token feeds).
    - Guided path (GuidedSpeculation) per-step syncs are inherent (grammar matcher needs each draft on CPU), so the batched floor readback applies only to processor == nil chains.

    Measure first (new Tests/MLXLMTests/MTPHostSyncBenchmarkTests.swift, tiny fixture drafter, debug build, host load avg 11-14 + synthetic busy threads on all 30 cores), ms per 4-draft chain, two runs:
    - ungated (0 syncs): quiet 2.46 / 2.84, loaded 3.68 / 5.43
    - gated, per-step readback (4 syncs): quiet 3.58 / 4.10, loaded 18.1 / 5.53
    - processor path gated (4 syncs): quiet 3.76 / 4.13, loaded 5.80 / 6.21
    => ~0.3 ms per sync on a quiet host, with heavy-tailed outliers under load (18 ms once).
  timestamp: 2026-10-29T15:34:45.910931+00:00
position_column: doing
position_ordinal: '8380'
title: 'MTP: stay faster than plain under CPU load (adaptive block size / plain fallback, fewer host syncs)'
---
Final verification of stable 148f227e at load average 10–14 showed MTP losing to plain on some turns (ChatSession edit turn 34.6 vs 52.3 tok/s; guided JSON 11.8 vs 14.3) while other turns kept 1.4–1.7x. MTP does host syncs every round (verify readback, plus one per chained step for the pMin gate), so CPU contention costs it more than plain. The ACP SWE-bench run executes CPU-heavy work (django test runs, code-context index, other sessions' builds) next to generation, so this matters in production.

Wait for data first: the quiet-machine rerun of the two failed benchmarks, mtpsmoke1 telemetry (tok/s per call + bench/load.mtpsmoke1.log in FoundationModelsACPAgent), and the SWE-bench-scale benchmark (^task 01M641KG6YT4FXK19NRJ2TNZPN).

- [ ] Quantify: MTP/plain ratio vs load average (from the telemetry and a controlled run with a synthetic CPU load, e.g. N busy threads).
- [ ] If MTP falls below plain under load: measure per-round host-sync time vs GPU time; cut syncs (batch the pMin check into the verify readback, async eval of the seed probability).
- [ ] Adaptive policy: track emitted tokens per second of speculative rounds vs a periodic plain estimate (or round latency vs acceptance) and lower the block size or pause speculation when it isn't paying; resume when it is. Output must stay identical (greedy) / distribution-preserving (sampled), and cache/drafter state must stay consistent when pausing/resuming.
- [ ] Release guards at quiet load stay ≥ current; under synthetic load MTP ≥ plain.

#mtp #perf