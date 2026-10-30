---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66njxj8rqvqnnnrkg4esxf5
  text: 'Picked up. Research: guard = commits ed863693 (controller, iterator pause/probe/catch-up, guided paused decisions, telemetry/sink fields, adaptivePolicy plumbing), 31febc53 (window SE judging + replay tests), ee74cc93 (off by default), 3b1e122e (stand-down fallback + shared helpers + named constants), 91b8f357 (IntegrationTesting guard-on runs, MTPCPULoadTests, Benchmark adaptivePolicy, MTPChatTurn.speculativeDecodingTelemetry). Kept (non-guard): QwenMTP one-step-late chained pMin read (readsAhead), asyncEval(seedArrays) after drafter commit, guided standDown → .unavailableContinuation (+ MTPDrafterFallback.init, sink fallback doc wording), resetPendingBuffer(), forwardTarget. Removed as guard-only: forwardTarget''s reconcilingSharedKV param (only feedPlainStep passed true), MTPChatTurn.speculativeDecodingTelemetry (only fed the guard description + guard test). Guided decode loop restored to pre-guard shape (no decideNext/catchUp; draftsThisRound uses maximumDrafts). Docs: git grep finds no README/using.md guard mention — nothing to remove there.'
  timestamp: 2026-10-30T15:06:52.872695+00:00
- actor: claude-code
  id: 01m66q7ajrf85ssvxab8m7vn88
  text: |-
    Implementation landed (not committed). Library: MTPAdaptiveSpeculation.swift deleted; iterator guard props/pause/probe/catch-up extension, adaptivePolicy/adaptiveClock params, forwardTarget.reconcilingSharedKV (guard-only) removed; ChatSession.swift now byte-identical to ed863693^; MTPSpeculation/SpeculativeDecoding/MLXSpeculativeDecoding back to pre-guard; guided decode loop back to pre-guard; sink adaptive fields gone; MTPChatTurn.speculativeDecodingTelemetry + guard description removed. Kept: QwenMTP readsAhead, asyncEval(seedArrays), standDown→.unavailableContinuation, resetPendingBuffer, forwardTarget. Tests: 5 guard test files git rm'd; guided guard test aSchemaWritesThePlainOutputAcrossPauses deleted; GuidedGenerationMTPStandDownTests reworked — HiddenWithholdingTarget now withholds hidden state on multi-token verify forwards (mtpVerifyKeepsSnapshotsKey set), so stand-down fires via commitDrafter without the guard; guard-only paused-forward test removed. IntegrationTesting: SWE sweep is {plain, mtp} (guarded runner, withKnownIssue ^w7cgbhd, guardedRateFloorOfPlain, describeGuard gone); MTPCPULoadTests plain vs MTP only; qwen38FourBitMTPSystemPromptCache unparameterized; Qwen38FourBitMTPModel.make / RoundCost / VerifyForwardProfile drop adaptivePolicy. Docs: no README/using.md guard text existed. Note: sweBenchScaleTimeLimitMinutes=150 left unchanged though each point now runs 2 passes instead of 3.

    Verification: swift build --build-tests 0 Swift warnings (only SwiftPM llbuild "missing creator for mutated node" on Cmlx bundle). xctest: CXGrammarTests 7, MLXFoundationModelsTests 455, MLXGuidedGenerationTests 72, MLXHuggingFaceMacrosTests 7 XCTest, MLXLMTests 920 XCTest + 2054 swift-testing — 0 fail, 0 skip. IntegrationTesting Release build-for-testing: TEST BUILD SUCCEEDED, 0 repo Swift warnings. scripts/verify-docs.sh: all passed. rg adaptive in Libraries/Tests/IntegrationTesting: only xgrammar AdaptiveTokenMask comments in GoldenReplayTests.

    Release run under bench.lock (owner "guard-removal <pid>", released; uptime load 4.80 at start → 10.04/15.13 at end; a concurrent swift-build + swift-test from another session ran at ~100% CPU each during the sweep, outside the lock): qwen38FourBitMTPSystemPromptCache PASSED — edit turn prefilled 123 tok (cached 2169), MTP 103.9 vs plain 55.9 tok/s; follow-ups 105.9/96.8/92.1, thinking 85.6/85.8/80.6 vs plain ~56. qwen38FourBitMTPGeneration FAILED one expectation: sweep.isNoSlowerThanSingleDraft(shipped) — shipped bs5 pMin0.75 mean 83.4 (math 100.6, code 77.0, prose 72.5) vs bs2 mean 85.1 (prose 79.9), noise 7.9%; plain rates in the shipped row dropped to 54.8/54.9 (vs ~58.5 elsewhere), consistent with the contention. The sweep does not exercise the guard (it was off by default on HEAD), so this is not a behaviour change from the removal; a clean rerun with the machine idle is needed to confirm.
  timestamp: 2026-10-30T15:35:30.136922+00:00
- actor: claude-code
  id: 01m66q7b8bjgfsbfazep2tkbqq
  text: |-
    ### implement — changed
    - evidence: 25 files — deleted Libraries/MLXLMCommon/MTPAdaptiveSpeculation.swift, Tests/MLXLMTests/MTPAdaptive{Controller,Speculation,Replay}Tests.swift, Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift, Tests/MLXLMTests/MTPChatBenchmarkAdaptivePolicyTests.swift; edited MTPSpeculativeTokenIterator, ChatSession, MTPSpeculation, SpeculativeDecoding, MLXSpeculativeDecoding, GuidedSpeculation, GuidedGenerationDiagnosticSink, MTPChatBenchmark, GenerationBenchmarks, GuidedGenerationMTP{Tests,StandDownTests,TestSupport}, GenerationBenchmarkTests, Qwen38FourBitMTPModel, MTPSWEBenchScaleTests, MTPCPULoadTests, MTPRoundCostBreakdownTests, MTPVerifyForwardProfileTests. Build 0 warnings; 5 bundles 0 fail/0 skip; IT Release build-for-testing OK; verify-docs OK. Release run: SystemPromptCache pass (edit turn 123 tok); MTPGeneration failed isNoSlowerThanSingleDraft under external CPU contention.
    - next: rerun qwen38FourBitMTPGeneration on an idle machine to confirm the sweep check, then /review.
  timestamp: 2026-10-30T15:35:30.827568+00:00
- actor: claude-code
  id: 01m66qapjgrsf4m94npchp7vrp
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (937390bc) — counts findings 0, confirmed 0, refuted 0 (7 validators attempted, 0 failed); 24 files reviewed, 2 .kanban files skipped (no validator). Declined items only for files the commit deleted. Repo-wide grep for MTPAdaptive|adaptivePolicy|invalidMTPAdaptivePolicy|adaptivePause|adaptiveProbe|adaptivePlainToken|adaptiveBlockSize: no matches.
    - next: none — task moved to done.
  timestamp: 2026-10-30T15:37:20.720951+00:00
- actor: claude-code
  id: 01m66qbcs8wphqn61af8h5aab2
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — guard removed (6 files deleted, 19 edited)
    - test: green — build 0 warnings; 5 bundles 0 fail/0 skip; IntegrationTesting Release build ok; verify-docs ok; Release qwen38FourBitMTPSystemPromptCache pass (edit turn 123); qwen38FourBitMTPGeneration single-draft check is the pre-existing ^p2bbykc
    - commit: 937390bc
    - review: clean — task moved to done
  timestamp: 2026-10-30T15:37:43.464411+00:00
position_column: done
position_ordinal: ffff9280
title: 'MTP: remove the adaptive guard entirely (user decision)'
---
User decision (2026-10-30): "I don't want that guard at all." Real-model measurement under ^tdj3waa showed the guard never pays (slower than guard-off at 5/10 sweep points, below plain at router 64k/100k and under 50% CPU load). Remove it completely rather than keep it opt-in.

- [x] Delete MTPAdaptivePolicy, the adaptive controller (MTPAdaptiveSpeculation.swift), the pause/probe/catch-up paths in MTPSpeculativeTokenIterator and GuidedSpeculation, the `adaptivePolicy:` parameters (SpeculativeDecodingConfig, MLXSpeculativeDecoding, the iterator, ChatSession plumbing), the `invalidMTPAdaptivePolicy` error, the adaptive telemetry fields (adaptivePauseCount/ProbeCount/PlainTokenCount/BlockSizeChangeCount/BlockSize) and the guided diagnostic-sink adaptive fields.
- [x] Keep what is not the guard: the one-step-late chained pMin floor read and async seed probability (host-sync reduction), the guided stand-down → `.unavailableContinuation` fallback reporting, resetPendingBuffer()/forwardTarget shared helpers — unless they only exist to serve the guard.
- [x] Delete the guard's tests (MTPAdaptiveControllerTests, MTPAdaptiveSpeculationTests, MTPAdaptiveReplayTests, MTPAdaptiveTestPolicy, guard parts of guided tests) and the guard-on columns/assertions in IntegrationTesting (MTPSWEBenchScaleTests sweep's guarded runner and its withKnownIssue check, MTPCPULoadTests guard-on runs, qwen38FourBitMTPSystemPromptCache guard-on variant, Benchmark.mtpSystemPromptCache adaptivePolicy param) — keep plain vs MTP comparisons.
- [x] Docs: remove guard mentions (README, using.md, doc comments).
- [x] Build 0 warnings; all 5 bundles green; IntegrationTesting Release build-for-testing; verify-docs passes. Greedy output unchanged vs before the guard landed (existing MTP tests).

#mtp