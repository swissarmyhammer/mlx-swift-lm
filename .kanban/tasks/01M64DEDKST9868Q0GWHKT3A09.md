---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66qrewbqam33kr2f0v8m8bn
  text: |-
    Picked up (implement). Research:
    - Drafter caches come only from `ShiftedMTPDrafter.makeCache(layerCount:parameters:maximumBlockSize:)` (QwenMTP.swift), reached by both drafters' `makeState(parameters:)` (MLXLLM Qwen35MTPDraftModel, MLXVLM Qwen35VLMNextNDraftModel). ^crdd8sf already makes it a RotatingKVCache (main window + keep, rewind reserve 2×mostTokens) when the main caches rotate; rewinds, isAligned (logical offsets), multi-row masks (prefix-keeping) all already work for a ring drafter.
    - Every drafter state source passes the stream's GenerateParameters: iterator (`makeState(parameters:)`), MTPDrafterPrefillFeed, `SerializedMTPDrafterContinuation.restored(by:parameters:)` from ChatSession and ExecutorPromptCacheFile. `restorePromptCacheSnapshot(into: templates)` already refuses another class or another window/keep → cold start. So a window decided from GenerateParameters round-trips and mismatches cold-start with no new restore code.
    - Plan: `GenerateParameters.mtpDrafterAttentionWindow: KVCacheConfiguration.Capacity?` (nil = full context), default `SpeculativeDecodingConfig.defaultMTPDrafterAttentionWindow` (picked by measurement); makeCache takes the narrower of the main capacity and the drafter window. A ring below its window attends to everything, so the window applies only past W tokens (the threshold is the window itself; no runtime switch).
    - FM executor and ChatSession build GenerateParameters with defaults, so they get the default window; measurements set the window per run through `generateTokensTask(parameters:)` on both load paths (router = loadModelContainer VLM Qwen35 + .matchingTarget drafter; text = LLMModelFactory).
    - Prior data (^j2tnzpn): chained drafter steps 1.4–1.6 ms at 32k, 3–12 ms at 100k; block-5 proposal 5.9/6.0 ms at 32k vs 24.5/22.3 ms at 100k (text/router).
  timestamp: 2026-10-30T15:44:51.595414+00:00
- actor: claude-code
  id: 01m66s2s70avjb29qx58113mf9
  text: |-
    Drafter proposal cost by window (MTPRoundCostBreakdownTests, Release, lock held 10:55–11:07, load avg 5.9→6.8; contention: swift-package ~79% CPU, replayd ~73%, two other swiftpm-testing-helper ~23% each). Block 2/3/4/5 proposal ms:
    - text 32k: full 1.51/3.78/4.48/5.82; 16k 2.93/3.26/4.61/6.30; 8k 2.94/3.03/4.05/5.26; 4k 2.25/2.67/4.09/5.11; 2k 1.64/2.57/3.62/4.83
    - text 64k: full 2.13/7.26/15.04/17.29; 16k 3.95/3.34/4.58/6.55; 8k 2.97/2.88/4.10/5.44; 4k 2.35/2.79/3.76/5.03; 2k 2.00/2.57/3.66/4.85
    - text 100k: full 2.07/9.56/21.10/25.38; 16k 2.06/3.15/4.71/7.19; 8k 2.86/2.93/4.15/5.50; 4k 2.63/2.81/3.99/5.04; 2k 1.73/2.53/3.68/4.79
    - router 64k: full 2.09/7.40/14.56/18.59; 16k 6.99 (block 5); 8k 6.10; 4k 5.37; 2k 4.93
    - router 100k: full 2.05/10.26/21.40/25.69; 16k 2.31/3.43/5.07/6.71; 8k 2.13/3.10/4.41/5.91; 4k 2.12/2.80/4.08/5.35; 2k 1.87/2.79/4.01/5.10
    - router 32k block 5: 2k 5.06 (other router-32k lines were before the log window; text 32k covers it)
    Verify forward rows 1..5 (unchanged by the window): text 100k 27.3/36.0/46.2/51.0/57.1 ms, router 100k 28.6/45.3/59.1/68.7/81.6 ms; plain one-token decode text 21.4/24.3/27.2, router 25.1 (64k)/28.8 (100k) ms.
    So a windowed drafter cuts a block-5 proposal at 100k from ~25 ms to 5–7 ms, and at 64k from ~17–19 ms to 5–7 ms; at 32k it is a wash. Acceptance/tok/s sweep (MTPDrafterWindowSweepTests) runs next.
  timestamp: 2026-10-30T16:07:58.432263+00:00
- actor: claude-code
  id: 01m66xh8jd306ntgbz53jrj3vd
  text: |-
    Decode sweep by drafter window (new MTPDrafterWindowSweepTests, Release, lock held 11:08–12:23, 75 min; load avg 5.5→5.1; contention at start swift-package ~81% CPU, fseventsd 27%; at end idle). Greedy, 1024 tokens, bs5/pMin0.75, plain decode reads top-2 candidates. tok/s (x full drafter), acceptance, tokens/verify, ms/round:
    TEXT 32k (plain 45.1): full 75.1 (1.00) 0.926 3.53 47.0 | 16k 58.5 (0.78) 0.885 3.23 55.2 | 8k 62.9 (0.84) 0.889 3.25 51.6 | 4k 67.2 (0.89) 0.848 3.09 46.0 | 2k 72.7 (0.97) 0.863 3.23 44.4
    TEXT 64k (plain 38.3): full 63.5 (1.00) 0.933 3.66 57.6 | 16k 52.1 (0.82) 0.899 3.37 64.6 | 8k 52.7 (0.83) 0.861 2.97 56.3 | 4k 60.9 (0.96) 0.898 3.41 56.0 | 2k 55.2 (0.87) 0.831 2.87 51.9
    TEXT 100k (plain 32.8): full 54.8 (1.00) 0.948 3.84 69.9 | 16k 49.6 (0.90) 0.908 3.71 74.7 | 8k 47.3 (0.86) 0.791 3.40 71.9 | 4k 52.6 (0.96) 0.890 3.51 66.6 | 2k 50.5 (0.92) 0.790 3.26 64.6
    ROUTER 32k (plain 46.9): full 67.6 (1.00) 0.922 3.37 49.8 | 16k 54.6 (0.81) 0.869 2.81 51.3 | 8k 61.6 (0.91) 0.814 3.41 55.4 | 4k 60.8 (0.90) 0.822 2.84 46.7 | 2k 62.9 (0.93) 0.799 3.26 51.8
    ROUTER 64k (plain 37.9): full 47.2 (1.00) 0.907 3.41 72.2 | 16k 42.8 (0.91) 0.894 3.32 77.6 | 8k 49.3 (1.04) 0.894 3.47 70.3 | 4k 44.4 (0.94) 0.796 2.94 66.2 | 2k 47.1 (1.00) 0.852 2.93 62.2
    ROUTER 100k (plain 32.5): full 40.0 (1.00) 0.943 3.62 90.3 | 16k 38.1 (0.95) 0.893 3.66 96.0 | 8k 36.5 (0.91) 0.843 2.76 75.6 | 4k 37.2 (0.93) 0.847 2.63 70.7 | 2k 38.2 (0.95) 0.808 2.73 71.5
    Every MTP pass speculated (no passthrough) and matched plain up to a near tie (MTPChatTokenMatch holds at all 30 MTP points).
    Verdict: no window is never-slower-than-full; 29 of 30 windowed points are slower than the whole-context drafter (only router 64k 8k is +4%, inside run noise, and 8k loses 9–17% elsewhere). Why: (1) acceptance falls with every window (tokens/verify −3% to −27%), and that outweighs the drafter time saved; (2) the ring adds per-round work the micro-benchmark does not see (text 32k: 16k ring 55.2 ms/round vs full 47.0 though the drafter attention is equal there) — multi-row commits re-linearize the ring (updateConcat temporalOrder+concat), a wrapped-ring rewind linearizes it again and the next write regrows it, so each round copies O(window) rows several times. Even at router 100k, where windows cut 15–20 ms/round, the acceptance loss leaves them 5–9% slower. Default stays the whole context (`defaultMTPDrafterAttentionWindow = nil`); the window stays available as an opt-in GenerateParameters setting. Caveat: after the first near tie each pass writes different text, so per-window acceptance is not on identical tokens; the direction is consistent across all 25 windowed points.
  timestamp: 2026-10-30T17:25:47.213839+00:00
- actor: claude-code
  id: 01m66z73ycj8nzaze51bmnpp39
  text: |-
    Final default = whole context (`SpeculativeDecodingConfig.defaultMTPDrafterAttentionWindow = nil`), documented with the measurement in its doc comment and in Documentation.docc/using.md (KV cache settings). Window remains an opt-in `GenerateParameters.mtpDrafterAttentionWindow`. Follow-up for the ring's per-round O(window) copies filed as ^0p5fjwc.

    qwen38FourBitMTPSystemPromptCache() rerun with the final default (Release, lock held 12:27:57–12:28:45, load avg 3.2→3.5; contention: another swiftpm-testing-helper 16.5% CPU at start, 66.9% at end): passed in 44.9 s. Edit turn: cached 2169, prefilled 123 tokens (plain the same), MTP 102.0 tok/s vs plain 55.3, acceptance 92.9%, text equal to plain. Follow-ups prefill 24–38 tokens, MTP 77.8–100.7 tok/s vs plain 55.2–55.7.

    Unit suites (Debug, `swift build --build-tests`: 0 compiler warnings; the only build warning is the pre-existing SwiftPM "missing creator for mutated node ... mlx-swift_Cmlx.bundle"): `xcrun xctest` all 5 bundles exit 0 — CXGrammarTests 7, MLXGuidedGenerationTests 72, MLXHuggingFaceMacrosTests 7 XCTest, MLXFoundationModelsTests 457 (incl. ExecutorPromptCacheDrafterWindowTests), MLXLMTests 920 XCTest + 2068 Swift Testing in 195 suites (incl. Qwen35MTPDrafterWindowTests 14); 0 failures, 0 skipped.

    New test `windowedDrafterAttendsToItsWindowAlone` checks the ring attention itself (a windowed drafter fed in 1- and 3-row chunks past its wrap gives the seed hidden row of a whole-context drafter fed only the last W positions), so the acceptance loss in the sweep is the window's effect, not a ring bug.
  timestamp: 2026-10-30T17:55:11.948562+00:00
- actor: claude-code
  id: 01m66z7apz6j0b883f8ycqkwm5
  text: |-
    ### implement — changed
    - evidence: 11 files — Libraries/MLXLMCommon/QwenMTP.swift, Libraries/MLXLMCommon/Evaluate.swift, Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLMCommon/Documentation.docc/using.md, Tests/MLXLMTests/Qwen35MTPDrafterWindowTests.swift (new), Tests/MLXFoundationModelsTests/ExecutorPromptCacheDrafterWindowTests.swift (new), IntegrationTesting/.../TextGeneration/MTPDrafterWindowSweepTests.swift (new), MTPRoundCostBreakdownTests.swift, Qwen38FourBitMTPModel.swift, MTPSWEBenchScaleTests.swift (+ new task ^0p5fjwc). `swift build --build-tests` 0 compiler warnings; 5 bundles via xcrun xctest exit 0, 0 failures, 0 skipped; Release: MTPRoundCostBreakdownTests pass (688 s), MTPDrafterWindowSweepTests pass (4493 s), qwen38FourBitMTPSystemPromptCache pass (edit turn 123 tokens). Default window = whole context (no window wins).
    - next: /review
  timestamp: 2026-10-30T17:55:18.879774+00:00
- actor: claude-code
  id: 01m66zk419p5y00zyf9bx540f2
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (39d23442) — 4 findings (confirmed 9, refuted 5, attempted 14): MTPDrafterWindowSweepTests.swift:35, MTPDrafterWindowSweepTests.swift:38, MTPRoundCostBreakdownTests.swift:159, Qwen38FourBitMTPModel.swift:243
    - next: implement the 4 open items in the 2026-10-30 12:56 Review Findings section, then re-review
  timestamp: 2026-10-30T18:01:45.257796+00:00
- actor: claude-code
  id: 01m66zm1zhjvjhm7pewk0dqqn0
  text: |-
    ### finish iteration 1 — findings (held for a user decision)
    - implement: changed — opt-in mtpDrafterAttentionWindow; every measured window slower (0.78–0.96x of whole context), default whole context
    - test: green — build 0 warnings; 5 bundles 0 fail/0 skip; Release sweep/round-cost/system-prompt-cache pass
    - commit: 39d23442
    - review: findings — MTPDrafterWindowSweepTests.swift:35, :38, MTPRoundCostBreakdownTests.swift:159, Qwen38FourBitMTPModel.swift:243
    - held: asked the user whether to remove the opt-in window entirely (it never wins). The fix pass depends on that answer.
  timestamp: 2026-10-30T18:02:15.921259+00:00
- actor: claude-code
  id: 01m6728dccsar12q0sqd0eqs4g
  text: 'User decision 2026-10-30: keep the opt-in drafter attention window for now (default stays whole context). Proceed with the 4 open review findings (MTPDrafterWindowSweepTests.swift:35, :38, MTPRoundCostBreakdownTests.swift:159, Qwen38FourBitMTPModel.swift:243) after the current ^p2bbykc step.'
  timestamp: 2026-10-30T18:48:20.108534+00:00
- actor: claude-code
  id: 01m678sqrsgqx4ty21y7xammen
  text: |-
    Fix pass for the 2026-10-30 12:56 findings (user kept the opt-in window; default whole context):
    - MTPDrafterWindowSweepTests.swift: the sweep contexts (32_768/65_536/100_000) and windows (16_384/8_192/4_096/2_048) are now private named constants of `DrafterWindowSweep` (short/middle/longContextTokenCount, widest/wide/narrow/narrowestWindowTokenCount); `contextTokenCounts`/`windowTokenCounts` list them. swiftlint no_magic_numbers (allowed [0, 1, -1, 100]) over all three files: 0 violations (was 7, all in the sweep file).
    - MTPRoundCostBreakdownTests.swift: `for drafts in cost.drafts` -> `for drafterCost in cost.drafts`; the same element-of-`drafts` closure in `RoundCost.summary` (`drafts.map { cost in`, which also reused the RoundCost name) -> `drafterCost`.
    - Qwen38FourBitMTPModel.swift: deleted `PerformBox<Value>: @unchecked Sendable`. `LoadedWithDrafter.drafter` is now the `MTPDrafterContainer` (Sendable) itself, and `LoadedWithDrafter.perform { context, drafter in ... }` runs the body inside the drafter container's `perform`, handing the drafter model to the target's public `ModelContainer.perform(nonSendable:)`. The drafter model never leaves its container's isolation; both benchmarks call `loaded.perform`. Lock order is always drafter then target, and nothing else in these suites takes them the other way round. (MLXLMCommon's `SendableBox` is `package`, so the integration target cannot use it, and it would be the same unchecked box anyway.)
    - Block size: both suites read `speculation.blockSize` (the default is now 4 per ^p2bbykc), so nothing was hard-coded to 5.
    - swift-format lint shows 3 warnings in lines this pass did not touch (sweep log line LineLength, round-cost `rewindDepth` AddLines, round-cost log line LineLength). They were left as they are.
  timestamp: 2026-10-30T20:42:39.257669+00:00
- actor: claude-code
  id: 01m678srf3xk8ex4dtem0c3cn1
  text: |-
    ### implement — changed
    - evidence: 3 files — IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPDrafterWindowSweepTests.swift, MTPRoundCostBreakdownTests.swift, Qwen38FourBitMTPModel.swift; all 4 findings checked. swiftlint no_magic_numbers 0 violations; xcodebuild build-for-testing Release (IntegrationTesting scheme, arm64, ENABLE_TESTABILITY/ONLY_ACTIVE_ARCH) TEST BUILD SUCCEEDED, no warnings in the 3 files (objects rebuilt); `swift build --build-tests` 0 compiler warnings (only the pre-existing SwiftPM "missing creator for mutated node ... mlx-swift_Cmlx.bundle"); xcrun xctest 5 bundles exit 0, 0 failures, 0 skipped: CXGrammarTests 7, MLXGuidedGenerationTests 72, MLXHuggingFaceMacrosTests 7, MLXFoundationModelsTests 457, MLXLMTests 920 XCTest + 2078 Swift Testing. No real-model runs.
    - next: /review
  timestamp: 2026-10-30T20:42:39.971044+00:00
- actor: claude-code
  id: 01m678wn6j6yag43qjhbwvd3pq
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (173e5f7a) — 0 findings (0 confirmed, 0 refuted; 7 attempted, 0 failed); 3 Swift files reviewed, 2 .kanban files unmatched. All 4 items of Review Findings (2026-10-30 12:56) checked. LoadedWithDrafter.perform takes the drafter container then the target container; the only callers (MTPDrafterWindowSweepTests, MTPRoundCostBreakdownTests) take no further lock inside the body, and no library path takes them in the reverse order, so the order is fixed. The drafter model crosses only via perform(nonSendable:) into a @Sendable body whose result is Sendable, so it does not leave the drafter's isolation.
    - next: none — task moved to done
  timestamp: 2026-10-30T20:44:14.930834+00:00
- actor: claude-code
  id: 01m678xqfbt28d7e9ytterbzca
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — named sweep constants, drafterCost, Sendable LoadedWithDrafter.perform
    - test: green — build 0 warnings; 5 bundles 0 fail/0 skip; Release build-for-testing ok
    - commit: 173e5f7a
    - review: clean — task moved to done (opt-in window kept per user)
  timestamp: 2026-10-30T20:44:50.027447+00:00
position_column: done
position_ordinal: ffff9580
title: 'MTP: cheaper drafting at long context — window the drafter''s attention (drafter steps cost 3–12 ms at 100k)'
---
Step-4 breakdown (^j2tnzpn): the drafter's cache holds the full context (32768 / 100000 entries) and its attention runs over all of it on every chained step. Chained-step cost is ~1.5 ms at 32k but 3–12 ms at 100k (block-5 proposal 22–24 ms vs 6 ms at 32k).

The drafter only proposes tokens; the target verifies them, so a drafter that attends to a window (e.g. the last N tokens plus the kept prefix) cannot change the output — only acceptance.

- [x] Add a drafter attention window (rotating drafter cache with a bounded window, reusing the rotating-ring support from ^crdd8sf) that applies only past a context threshold; keep the drafter continuation/prompt-cache save/restore consistent. — `GenerateParameters.mtpDrafterAttentionWindow: KVCacheConfiguration.Capacity?` (default `SpeculativeDecodingConfig.defaultMTPDrafterAttentionWindow`); `ShiftedMTPDrafter.makeCache` builds a RotatingKVCache of the narrower of the main window and the drafter window (rewind reserve 2×6 rows), so the window applies only once the context passes it. Save/restore round-trips through the same parameters; another window or the whole context cold-starts.
- [x] Measure acceptance and tok/s at 32k/64k/100k for windows 2k/4k/8k/16k vs full; pick the default by measured speed (perf-gains rule), never slower than full-context drafting. — MTPRoundCostBreakdownTests (drafter ms per proposal by window) and new MTPDrafterWindowSweepTests (both load paths). Windows cut a block-5 proposal at 100k from ~25 ms to 5–7 ms, but 29 of 30 windowed decode points were slower than full (0.78–0.96x; one +4% point inside noise) because acceptance falls; default stays the whole context (nil). Ring per-round overhead filed as ^0p5fjwc.
- [x] Output identity tests (greedy equals plain up to the near-tie rule) and ChatSession/ExecutorPromptCache continuation tests with the window. — Qwen35MTPDrafterWindowTests (14 tests: ring shape, bounds, exact plain identity text+VLM, prefix-keeping, wrap rejects, windowed attention equals a whole-context drafter fed the window, continuation restore/refuse, ChatSession file resume / cold start) and ExecutorPromptCacheDrafterWindowTests (spill file reads back for the same window, cold for another window or full); the Release sweep holds MTPChatTokenMatch at all 30 MTP points.

#mtp #perf

## Review Findings (2026-10-30 12:56)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 9 file(s) reviewed, 5 not reviewed.

> 5 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M64DEDKST9868Q0GWHKT3A09.jsonl` — no validator matches this file
> - `.kanban/tasks/01M64DEDKST9868Q0GWHKT3A09.md` — no validator matches this file
> - `.kanban/tasks/01M66XNGB2TY2WY01KP0P5FJWC.jsonl` — no validator matches this file
> - `.kanban/tasks/01M66XNGB2TY2WY01KP0P5FJWC.md` — no validator matches this file
> - `Libraries/MLXLMCommon/Documentation.docc/using.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPDrafterWindowSweepTests.swift:35` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPDrafterWindowSweepTests.swift:38` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPRoundCostBreakdownTests.swift:159` `swift/naming-clarity` — The loop variable `drafts` names one `DrafterWindowCost` element, but `drafts` is also the name of the collection it iterates (`cost.drafts`). The singular element is named after the plural collection, so the role of each iteration value is unclear at the use site. Rename the loop variable to its role, for example `for drafterCost in cost.drafts {`, and update lines 160-161 to `drafterCost.entryCount` and `drafterCost.draft`.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/Qwen38FourBitMTPModel.swift:243` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.
