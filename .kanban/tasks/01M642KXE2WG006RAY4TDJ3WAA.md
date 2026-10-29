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
- actor: claude-code
  id: 01m647p1y5q4c2pjb42b58r2q2
  text: |-
    Phase 1 landed: 94787c95 on worktree-agent-a24c62ee3c2f35eec (base mtp 268dab08).

    Design:
    - MTPAdaptivePolicy (public, on by default; .disabled to opt out) on SpeculativeDecodingConfig (adaptivePolicy:), MLXSpeculativeDecoding, MTPSpeculation, MTPSpeculativeTokenIterator init. Validated; invalid -> SpeculativeDecodingConfigurationError.invalidMTPAdaptivePolicy.
    - MTPAdaptiveController (clock-free, unit-tested) + MTPAdaptiveGuard (injectable clock). Each unit (round or plain step) is timed from its start to the next unit's start, so consumer time counts for both. Speculative tok/s (EWMA) vs plain tok/s from short probes (4 steps, 1 warm-up discarded) taken in-stream at the current position. Rounds losing (< requiredSpeedup) lower block size by 1 per judgment, at block 2 pause (64 tokens, doubling to 1024 while resumes keep losing); winning by growthSpeedup grows back after a hold. Probe interval 64 rounds, doubling to 512 while spec leads by margin. A plain estimate is re-probed when older than 32 rounds before a slow verdict, and (after the coordinator's SWE-bench data) whenever the context grew >10% and >=4096 tokens since it was measured, so long-context plain cost is measured at the current KV length.
    - Pause = pipelined plain decode (TokenIterator-style lookahead, emit flag on). Drafter catch-up: one batched commitDrafterState over the paused tokens + hidden rows at resume or finalize; drafter stays nextPosition == processed. Only for stateless or resumable drafters.
    - Guided (GuidedSpeculation): same guard; paused decisions feed one-token verifies, drafter catches up in one commit.
    - Telemetry: adaptivePauseCount, adaptiveProbeCount, adaptivePlainTokenCount, adaptiveBlockSizeChangeCount, adaptiveBlockSize (nil without a guard). Guided has no telemetry surface (not added).
    - Host syncs: chained pMin read one step ahead (argmax, no processor) so the GPU runs step k+1 while the host reads p_k; identical drafts/entries (trim one entry on stop). Seed + seed probability asyncEval'd at commit.

    Measure (tiny fixture, debug, host load 8-14 + 30 busy threads): 4-draft chain ms ungated/read-ahead/per-step: quiet 3.55/4.38/5.28, loaded 3.67/6.29/7.43 (earlier run 4.94/5.18/9.10).

    Not done / notes: generate(...mtpDrafter:) free functions keep the default policy (no new param, to avoid DocC link churn). One edit was made with a python one-liner instead of the files tool (test file only). The guided pause test and the context-growth controller test were written before the code but not run RED separately.

    Open thresholds needing real-model data: minimumRoundSamples 16, probeLength 4, probeInterval 64/512, stalePlainEstimateRounds 32, stalePlainContextGrowth 0.1 + 4096-token floor, requiredSpeedup 1.0, growthSpeedup 1.2, growthHoldRounds 128, pauseLength 64/1024, smoothing 0.25; read-ahead waste (<=1 drafter step) vs sync saving at 32k/100k.

    ### implement — changed
    - evidence: swift build --build-tests (0 compiler warnings; SwiftBuild 'missing creator for mutated node' bundle notice only); xctest MLXLMTests 2060 pass, MLXGuidedGenerationTests 72, MLXFoundationModelsTests 428, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6; 0 failures, 0 skipped. Commit 94787c95.
    - next: phase 2 real-model runs on user's go.
  timestamp: 2026-10-29T16:25:26.725091+00:00
- actor: claude-code
  id: 01m648fnj0g1v1sqngwwf1ep8b
  text: |-
    Rebased onto mtp cef34758 -> e1bdd285 (one commit). Conflict only in MLXSpeculativeDecoding.swift: kept upstream drafterBytes/memoryPolicy/admits* intact and added adaptivePolicy (param after memoryPolicy, validated, passed to MTPSpeculation). GuidedSpeculation/MLXLanguageModel auto-merged; SpeculativeDecodingOutcome metadata keys untouched.
    Guided sink: GuidedGenerationDiagnosticSink gains adaptivePauseCount/adaptiveProbeCount/adaptivePlainTokenCount/adaptiveBlockSize (package recordAdaptiveDecisions, reported once at end of GuidedSpeculation.decode). Paused decisions never call recordSpeculativeRound, so drafted/accepted counts exclude paused plain tokens (test asserts accepted + paused <= generated). generate(...mtpDrafter:) free functions left unchanged (4 public signatures + DocC link churn = not cheap).

    ### implement — changed
    - evidence: swift build --build-tests 0 compiler warnings; xctest MLXLMTests 2060, MLXGuidedGenerationTests 72, MLXFoundationModelsTests 455, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7 — 0 failures, 0 skipped; IntegrationTesting xcodebuild build-for-testing Release: TEST BUILD SUCCEEDED. Commit e1bdd285.
    - next: phase 2 on go.
  timestamp: 2026-10-29T16:39:26.016244+00:00
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