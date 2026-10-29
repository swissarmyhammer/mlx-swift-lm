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
- actor: claude-code
  id: 01m64902h574pax0h29kydqqvc
  text: |-
    ### review — findings
    - evidence: review sha ed863693~1..ed863693 — 9 findings (11 confirmed, 3 refuted) — Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:253, Libraries/MLXLMCommon/MTPAdaptiveSpeculation.swift:416, Libraries/MLXLMCommon/MTPAdaptiveSpeculation.swift:447, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:1518, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:1560, Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift:12, :13, :15, :16
    - next: implement the 9 findings in the 2026-10-29 11:40 section; the card's real-model subtasks (quantify, host-sync measurement, release guards) are also still open pending measurement, so the task stays in review.
  timestamp: 2026-10-29T16:48:23.589767+00:00
- actor: claude-code
  id: 01m6490p09vtmxyyjwwkx3edvt
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — adaptive guard + one-step-late floor read (worktree e1bdd285, applied to mtp)
    - test: green — 5 bundles 0 fail/0 skip, Release build-for-testing ok; real-model measurement pending GPU slot
    - commit: ed863693
    - review: findings — GuidedSpeculation.swift:253, MTPAdaptiveSpeculation.swift:416, :447, MTPSpeculativeTokenIterator.swift:1518, :1560, MTPAdaptiveTestPolicy.swift:12, :13, :15, :16
  timestamp: 2026-10-29T16:48:43.529519+00:00
- actor: claude-code
  id: 01m649yy029zfr9t10d8a1kdn8
  text: |-
    Review findings (2026-10-29 11:40) worked, commit 71cc7c66 on top of mtp ed863693. No GPU work.

    - GuidedSpeculation stand-down: `standDown(target:)` replaces both silent `stoodDown = true` branches (`recordPausedRow`, `commitDrafter`). It records `.unavailableContinuation` with its stated reason on the guided diagnostic sink; SpeculativeDecodingOutcome reads that sink for the `speculativeDecodingFallbackReason` metadata. `MTPDrafterFallbackLog.shared` logs it once per target. The stand-down is recorded at most once per generation. Added `package init(reason:statedReason:)` to MTPDrafterFallback.
    - New test file: Tests/MLXFoundationModelsTests/GuidedGenerationMTPStandDownTests.swift. A Qwen35 target subclass drops the hidden state on one-token forwards. Two tests:
      - One covers the paused path, on the schemas whose first round verifies several tokens, and asserts a probe ran.
      - One covers all schemas: whichever forward stands down first, the output is the plain output and exactly one fallback is recorded.
    - Shared code in MTPSpeculativeTokenIterator:
      - `resetPendingBuffer()` is used by `next()` and `nextGuardedPlainToken()`.
      - `forwardTarget(_:state:reconcilingSharedKV:)` runs the forward, commits, sets mainState, optionally reconciles shared KV, and samples. It is used by `passthroughStep` and `feedPlainStep`; each site keeps its own state setup, eval/item or asyncEval, drafter commit and kvCachePlan.
    - Named constants:
      - `MTPAdaptiveController.backoffFactor = 2` for the probe interval and pause doubling.
      - The `losingEveryRound` test policy now names its probe length, probe interval, and pause lengths.
      - swiftlint no_magic_numbers (allowed 0, 1, -1, 100): 0 violations in both files.

    ## Step record
    - Build: `swift build --build-tests`. 0 compiler warnings (only the "missing creator for mutated node" build-system notice).
    - Tests, 0 failures and 0 skipped:

      | Bundle | Tests |
      |---|---|
      | MLXLMTests | 2060 |
      | MLXGuidedGenerationTests | 72 |
      | MLXFoundationModelsTests | 457 (+2 new) |
      | CXGrammarTests | 7 |
      | MLXHuggingFaceMacrosTests | 7 |

    - IntegrationTesting Release build-for-testing: TEST BUILD SUCCEEDED.
    - Commit: 71cc7c66 (source only, no .kanban).
    - Real-model subtasks left unchecked for the GPU slot.
  timestamp: 2026-10-29T17:05:14.754233+00:00
- actor: claude-code
  id: 01m64a33ff13z003s4cf450qxs
  text: |-
    ### review — clean
    - evidence: review sha 3b1e122e~1..3b1e122e — 0 findings (0 confirmed, 1 refuted, 7 files attempted, 0 failed); all 9 items of ## Review Findings (2026-10-29 11:40) are checked. No new section appended.
    - next: task left in review, not moved to done, because the 4 real-model subtasks (quantify ratio vs load, host-sync measurement, adaptive policy verification, release guards under load) are still unchecked pending the GPU slot. Move to done once those are checked and a re-review is clean.
  timestamp: 2026-10-29T17:07:31.439545+00:00
- actor: claude-code
  id: 01m64a3jtp9c7cbq7fm3z13zyt
  text: |-
    ### finish iteration 2 — clean (code); task held in review
    - implement: changed — guided stand-down recorded + logged, shared helpers, named constants
    - test: green — 5 bundles 0 fail/0 skip, Release build-for-testing ok
    - commit: 3b1e122e
    - review: clean — held in review until the real-model subtasks are measured in the GPU slot (guard on vs off vs plain, 2k–100k). Not on stable yet: the guard is on by default, so stable waits for that measurement.
  timestamp: 2026-10-29T17:07:47.158266+00:00
- actor: claude-code
  id: 01m64exjcjn4kvmjs74a0qn0z3
  text: |-
    Coordinator steps 1–2 after the real-model step-4 sweep. On the sweep, guard on was below guard off at all 10 points, and it went to block 2 and paused even though guard off beat plain everywhere.

    **Step 1: d51f7990** (on mtp fb641ddf)
    - `MTPAdaptivePolicy.default` is now `.disabled`.
    - A new `MTPAdaptivePolicy.enabled` keeps the tuned settings, for callers that opt in.
    - Every `adaptivePolicy: = .default` parameter is now off: ChatSession, MTPSpeculation, the MTP iterator, MLXSpeculativeDecoding and the guided test support.
    - Docs are updated. The policy test now asserts that `.default == .disabled`.
    - All 5 bundles passed with 0 failures and 0 skips.

    **Step 2: 95034052. Cause: the decision rule, not the timing.**
    - I checked how time is charged to units, and it is sound:
      - Plain steps are pipelined like TokenIterator, so a step lasts about one forward.
      - The warm-up step is dropped.
      - Consumer time per token is a real cost for both rounds and plain steps.
      - Drafter catch-up after a pause lands in the first round after the resume. That round is 1 of at least 16 in its window, a few percent at most.
    - The bias was in the old controller:
      - It judged after every round, on an average weighted 0.25 toward the newest round, which in effect covers about the last 7 rounds.
      - Any short run of rejected drafts, or one stalled round, pulled that average below plain even while the overall rate was 1.19–1.75x plain.
      - Example, text 2k: four 1-token rounds followed by one 3x round drop the average to about 44 tok/s against plain 55.
      - It lowered the block size immediately, but growing back needed 1.2x and a 128-round hold. So each dip ratcheted down to block 2, and then pauses followed.
    - The fix: judge windows of rounds instead.
      - Each window covers the rounds since the last verdict, probe, pause or block change. Its rate is total tokens over total seconds, with the standard error of that ratio estimate.
      - Loss means `rate + significance·SE < plain·requiredSpeedup`. Win means `rate − significance·SE ≥ plain·requiredSpeedup`.
      - An undecided window doubles, from minimumRoundSamples (16) up to the new `maximumRoundSamples` (128). At the maximum it keeps the block size.
      - The new `significance` setting defaults to 2.
      - A window whose rate beats plain can therefore never lose.
      - A probe now starts a new window.
    - New test, `Tests/MLXLMTests/MTPAdaptiveReplayTests.swift`:
      - It replays the 10 points (guard-off ms/round, tokens per verify, plain ms/token from ^j2tnzpn) through the `.enabled` controller for 1024 tokens.
      - Each 20-round cycle has a 4-round burst of 1-token rounds followed by one 3x stall, keeping each point's mean cost; a companion test checks the means within 0.01.
      - It asserts no block change and no pause. It was RED at all 10 points before the fix and is GREEN after.
    - Controller tests:
      - Four tests updated to need a window of rounds per verdict: slowWindowLowersTheBlockSize, fastRoundsGrowTheBlockSizeBack, staleEstimateIsProbedFirst, grownContextIsProbedAgain.
      - Added insignificantDipKeepsTheBlockSize, plus validation of the two new settings.
    - The guard stays off by default. Turning it back on needs a GPU rerun of the sweep and the ChatSession follow-ups, not done here.

    ### implement — changed
    - evidence:
      - d51f7990: MTPAdaptiveSpeculation.swift, MLXSpeculativeDecoding.swift, MTPAdaptiveControllerTests.swift.
      - 95034052: MTPAdaptiveSpeculation.swift, MTPAdaptiveControllerTests.swift, MTPAdaptiveReplayTests.swift (new).
      - `swift build --build-tests`: 0 warnings. swiftlint no_magic_numbers: 0 violations.
      - Bundles, 0 failures and 0 skipped:

        | Bundle | Tests |
        |---|---|
        | MLXLMTests | 2063 |
        | Guided | 72 |
        | FM | 457 |
        | CXGrammar | 7 |
        | Macros | 7 |
    - next: on the GPU slot, rerun the step-4 sweep and the ChatSession follow-ups with `adaptivePolicy: .enabled`.
  timestamp: 2026-10-29T18:31:52.978823+00:00
position_column: review
position_ordinal: '80'
title: 'MTP: stay faster than plain under CPU load (adaptive block size / plain fallback, fewer host syncs)'
---
Final verification of stable 148f227e at load average 10–14 showed MTP losing to plain on some turns (ChatSession edit turn 34.6 vs 52.3 tok/s; guided JSON 11.8 vs 14.3) while other turns kept 1.4–1.7x. MTP does host syncs every round (verify readback, plus one per chained step for the pMin gate), so CPU contention costs it more than plain. The ACP SWE-bench run executes CPU-heavy work (django test runs, code-context index, other sessions' builds) next to generation, so this matters in production.

Wait for data first: the quiet-machine rerun of the two failed benchmarks, mtpsmoke1 telemetry (tok/s per call + bench/load.mtpsmoke1.log in FoundationModelsACPAgent), and the SWE-bench-scale benchmark (^task 01M641KG6YT4FXK19NRJ2TNZPN).

- [ ] Quantify: MTP/plain ratio vs load average (from the telemetry and a controlled run with a synthetic CPU load, e.g. N busy threads).
- [ ] If MTP falls below plain under load: measure per-round host-sync time vs GPU time; cut syncs (batch the pMin check into the verify readback, async eval of the seed probability).
- [ ] Adaptive policy: track emitted tokens per second of speculative rounds vs a periodic plain estimate (or round latency vs acceptance) and lower the block size or pause speculation when it isn't paying; resume when it is. Output must stay identical (greedy) / distribution-preserving (sampled), and cache/drafter state must stay consistent when pausing/resuming.
- [ ] Release guards at quiet load stay ≥ current; under synthetic load MTP ≥ plain.

#mtp #perf

## Review Findings (2026-10-29 11:40)

> Scope: `review sha ed863693~1..ed863693` — reviewed the diffs only — lines this change added or modified. 15 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M642KXE2WG006RAY4TDJ3WAA.jsonl` — no validator matches this file
> - `.kanban/tasks/01M642KXE2WG006RAY4TDJ3WAA.md` — no validator matches this file

- [x] `Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:253` `swift/preconditions` — When a paused forward returns no hidden row, the speculation ends through `stoodDown = true` with no log and no recorded reason. The end of the stream is then silent, and the `GuidedGenerationDiagnosticSink` shows nothing about why speculation stopped. Record why the speculation stopped, for example by setting a stand-down reason that the completion information reports, or by pairing the branch with `assertionFailure` and a production log, as the rule requires for an unexpected but recoverable condition.
- [x] `Libraries/MLXLMCommon/MTPAdaptiveSpeculation.swift:416` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLMCommon/MTPAdaptiveSpeculation.swift:447` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:1518` `duplication/duplication` — `nextGuardedPlainToken` repeats the three-line buffer reset that `next()` already performs before a speculation round. A later change to how the pending buffer is cleared would have to be made in both places. Move the three resets into one private method, such as `resetPendingBuffer()`, and call it from both `next()` and `nextGuardedPlainToken()`.
- [x] `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:1560` `duplication/duplication` — `feedPlainStep` repeats the forward-feed sequence of `passthroughStep`: build the state with the emit flag, call `mainModel` on the one-token input, commit the processed tokens, sample from the logits, apply the KV cache plan, and start evaluation. The copies differ only in the input token shape, whether the snapshot key is cleared, and whether shared-KV reconciliation runs. Those differences are what a shared helper should take as parameters, so the two can drift apart. Extract one private helper that takes the token array, and an optional flag for the shared-KV reconciliation. Have `passthroughStep` and `feedPlainStep` both call it, and keep only the differing bookkeeping at each site.
- [x] `Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift:12` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift:13` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift:15` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTestSupport/MTPAdaptiveTestPolicy.swift:16` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.