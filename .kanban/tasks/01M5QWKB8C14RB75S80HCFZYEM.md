---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzsqtgm76c5cdrd9n8pry
  text: 'Change after the PR 581 port (commit 843854ec): `finalizeShiftedMTPDrafterState` requires `proposalAppended == 0`. Chained drafts leave tentative entries, so use the general trim/append API of task 33ctvty.'
  timestamp: 2026-10-25T13:46:18.746946+00:00
- actor: claude-code
  id: 01m62939gw6h0wgqjkm7jc2aqv
  text: |-
    Research (implement pickup):
    - `draftMTPTokenBlock` no longer exists; the chain is `MTPDraftBlockOutput.chainedMTPDrafts` (Libraries/MLXLMCommon/QwenMTP.swift). Both Qwen35MTP drafters (MLXLLM `Qwen35MTPDraftModel`, MLXVLM `Qwen35VLMNextNDraftModel`) return only the seed when `state.takeSeed()` hits, which is every decode round (appendDrafterState always seeds). `maximumBlockSize = 2` on both clamps the iterator.
    - Commit is the shared `ResumableMTPDrafterModel.commitDrafterState` default (MTPDrafterModel.swift): today it KEEPS accepted tentative entries (drafter-hidden rows) and appends only the rest. Card wants accepted ones appended with target hidden rows -> trim all tentative entries, re-append accepted drafts + final with verifier rows in one MTP forward (same cost, entries equal a cold prefill).
    - Verify path at S = blockSize tokens <= 6: iterator takes native hybrid rewind (`maximumNativeTargetCacheRewind` = GatedDeltaDecodeKernel.mostRewoundTokens = 5), sets `mtpVerifyKeepsSnapshotsKey`, so Qwen35TextModel.forward runs inside `MultiRowQuantizedMatvec.run(asSpeculativeVerify: true)` (kernel rows 3..8) and the compiled verify schedule (CompiledVerifyShape.tokenRange 2...6). A >6-token round would take the staged-round path, which sets no keepsSnapshots and so skips both; maximumBlockSize 6 keeps every Qwen round on the native path.
    - pMin early stop makes the draft count per round variable; the iterator must use the drafts the drafter returned rather than numDraft.
    - Sampled modes: sample-and-match uses ArgMax draftSampler -> stateful `draftBlock` (chained steps argmax); rejection mode -> `draftBlockWithLogits` re-draws the seed from seedHidden with the sampler, chain must continue from that token with the processor copy told of it.
    - Per-depth acceptance: SpeculativeDecodingTelemetry has only totals; add per-depth counters for the benchmark print.
  timestamp: 2026-10-28T22:11:40.188549+00:00
- actor: claude-code
  id: 01m62c4xp09fw0jhtrahhd7cna
  text: |-
    Implementation landed (uncommitted): ShiftedMTPDrafter (QwenMTP.swift) drafts seed + chained steps from seedHidden for both Qwen drafters (token and with-logits paths; rejection mode re-draws the seed with the sampler and tells the processor copy of it), floor `MTPDrafterState.minimumDraftProbability` (pMin) gates chained steps (top-1 = exp(max - logSumExp), one sync per gated step), `appendingTentativeProposal` counts appended entries from the cache offset. Shared commit trims ALL tentative entries and re-appends accepted drafts + final with verifier rows (state == cold prefill; drafter-level test proves it with the independent drafter). maximumBlockSize = GatedDeltaDecodeKernel.mostTokens (6). Iterator: `minimumDraftProbability` init param (validated 0...1), verify uses the drafts actually returned. SpeculativeDecodingConfig(mtpDrafter:blockSize:minimumDraftProbability:), generate*/generateTokens* gain `minimumDraftProbability`. Telemetry: draftedRoundCountsByDepth / acceptedRoundCountsByDepth / acceptanceRate(atDepth:). BenchmarkHelpers: MTPBlockSweep.swift (sweep blockSize 2..6 x pMin 0/0.75, per-depth acceptance, plain-candidate MTPChatTokenMatch, load average, noise from plain spread).

    Release sweep #2 (M5 Ultra, noise 1.2%, load 21 -> 5, thermal 1), MTP tok/s math/code/prose (plain ~59):
    - bs2: 92.3 / 81.7 / 79.4 (mean 84.5)
    - bs3 p0: 96.2/72.8/68.8 (79.3); bs3 p.75: 93.2/81.4/76.1 (83.6)
    - bs4 p0: 105.7/65.8/68.4 (80.0); bs4 p.75: 99.1/81.7/74.2 (85.0)
    - bs5 p0: 99.6/61.1/62.1 (74.3); bs5 p.75: 99.6/84.6/72.9 (85.7)
    - bs6 p0: 97.2/50.6/55.0 (67.6); bs6 p.75: 94.7/84.9/74.1 (84.6)
    Per-depth acceptance code bs4 p0: d1 69% d2 37% d3 30%; math d1 94% d2 88% d3 79%. Sweep #1 (heavier load, noise 4.4%) agreed (bs2 84.2, best bs3 p.75 84.4).
    Why deeper blocks do not win broadly: each extra draft costs ~6-7 ms per round (round time code bs2 21.3 ms -> bs4 36.4 ms): the multi-row verify kernel is ALU-bound past 2 rows (^3w7jkfn measured M=4 ~1.7x M=1), plus one chained MTP step (+LM head) and, with a floor, one host sync per step. Prose/code chained acceptance (~55% conditional at d2) is below break-even. Every pMin 0.75 setting is 4-8% slower than bs2 on prose (beyond noise). DEFAULT RULE result: no deeper setting is both faster combined and no slower on every category -> default stays blockSize 2, pMin 0 (`SpeculativeDecodingConfig.defaultMTPBlockSize`/`defaultMTPMinimumDraftProbability`). The benchmark test now encodes this rule (fastest among settings no slower than one draft on any prompt).
  timestamp: 2026-10-28T23:04:59.328175+00:00
- actor: claude-code
  id: 01m62dskm22m0shphpr50z1b6p
  text: |-
    Final verification (M5 Ultra, Release, xcodebuild IntegrationTesting, -destination 'platform=macOS,arch=arm64' ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES):
    - qwen38FourBitMTPGeneration() PASS (sweep #3, noise 0.3%, load avg 4.76 -> 5.68, thermal state 1). MTP tok/s math/code/prose, plain 59.4-59.6 on every row: bs2 92.6/83.1/80.7 (mean 85.5); bs3 p0 97.5/75.6/73.6; bs3 p.75 94.0/82.5/77.1; bs4 p0 106.6/69.7/68.6; bs4 p.75 99.6/82.8/74.9; bs5 p0 100.5/63.7/62.5; bs5 p.75 100.4/85.4/73.9 (mean 86.6, fastest combined, but prose -8.4%); bs6 p0 97.6/51.8/55.1; bs6 p.75 94.7/85.0/74.2. Fastest no slower than one draft on any prompt: blockSize 2 pMin 0 -> shipped default stays 2/0. Shipped tokens exact or near-tie (prose bs3 near-tie at 114 gap 0.125 is a non-default row).
    - qwen38FourBitMTPSampledGeneration() PASS: default sample-and-match math 93.0 vs plain 58.5 (code 78.6, prose 71.7); rejection math 81.5, code 79.3, prose 73.6.
    - qwen38FourBitMTPSystemPromptCache() PASS, no known issues: follow-ups cached (prefill 38/29 tok, drafter prefills only suffix), edit turn recomputes 123 tokens with MTP on (172.0 ms MTP vs 185.2 ms plain), MTP decode 1.53-1.63x plain, all near-ties within tolerance.
    - swift build --build-tests: 0 compiler warnings (only the pre-existing build-system "missing creator" note). xcrun xctest all 5 bundles green: MLXLMTests 1966 Swift Testing + 920 XCTest, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. Pre-existing env-gated skip noticed (MTPGreedyVerificationTests.benchmarkVerification, MTP_GREEDY_BENCHMARK) - not touched.
    - Follow-up filed: ^c6fhxx6 (gate chained drafts on the seed's confidence / batch the floor sync) to remove the prose regression that keeps deeper blocks off by default.
  timestamp: 2026-10-28T23:33:45.730654+00:00
- actor: claude-code
  id: 01m62dsz4s1xvp106g9ththha9
  text: |-
    ### implement — changed
    - correction: the follow-up task filed above is ^hbq0f3n (01M62C57EC6FHXX6CBWHBQ0F3N), not ^c6fhxx6.
    - evidence: 24 files — Libraries/MLXLMCommon/{QwenMTP,MTPDrafterModel,MTPSpeculativeTokenIterator,ChatSession,Evaluate,MTPDrafterContinuation,SpeculativeDecoding}.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Libraries/BenchmarkHelpers/{BenchmarkHelpers,MTPBlockSweep(new)}.swift, IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift, Tests/MLXLMTests/{Qwen35ChainedMTPDraftTests(new),Qwen35ChainedMTPCachingTests(new),ChatSessionQwenChainedMTPTests(new),SpeculativeDecodingDepthTelemetryTests(new),SpeculativeDecodingConfigTests,Qwen35MTPSamplingTests,Qwen35MTPTests,Qwen35ScriptedMTPSupport}.swift, plus kanban cards. swift build --build-tests 0 compiler warnings; xcrun xctest 5 bundles green (MLXLMTests 1966+920); Release qwen38FourBitMTPGeneration / SampledGeneration / SystemPromptCache PASS. Default stays blockSize 2 pMin 0 per DEFAULT RULE (numbers in comments).
    - next: /review
  timestamp: 2026-10-28T23:33:57.529230+00:00
- actor: claude-code
  id: 01m62e8s0dekxx0m5m771sr5sm
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (a9985516) — 6 findings (6 confirmed, 6 refuted): Libraries/BenchmarkHelpers/MTPBlockSweep.swift:113, Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift:63, Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift:133, Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:78, Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:131, Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:309
    - next: /implement ^hcfzyem to resolve the 2026-10-28 18:34 Review Findings checklist
  timestamp: 2026-10-28T23:42:02.765834+00:00
- actor: claude-code
  id: 01m62e9ckbvrxhy7b1pqetxxfc
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 24 files (chained drafts, pMin, depth telemetry; default stays bs2)
    - test: green — 5 xctest bundles; Release MTPGeneration/SampledGeneration/SystemPromptCache pass, edit turn 123 tokens
    - commit: a9985516
    - review: findings — Libraries/BenchmarkHelpers/MTPBlockSweep.swift:113, Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift:63, :133, Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:78, :131, :309
  timestamp: 2026-10-28T23:42:22.827348+00:00
- actor: claude-code
  id: 01m62eqbmsxhdb1dt34ecjetzd
  text: |-
    Resolved the 2026-10-28 18:34 findings (pure refactor, no library behaviour change beyond the constant):
    - MTPBlockSweep.swift: the 0.75 floor is now `MTPBlockSetting.reportedFastestMinimumDraftProbability` and `sweptMinimumDraftProbabilities` reads `[0, reportedFastestMinimumDraftProbability]`. Ran swiftlint 0.65.1 `no_magic_numbers` (allowed_numbers [0, 1, -1, 100], the validator's config) on the whole file: 0 violations, down from 1.
    - Qwen35ScriptedMTPSupport.swift (8.6 KB -> 12.5 KB) gains `Qwen35MTPFixture.mtpIterator(over:blockSize:parameters:minimumDraftProbability:cache:mtpContinuation:)`, `greedyMTPIterator(over:blockSize:maxTokens:minimumDraftProbability:cache:mtpContinuation:)`, `prefillContinuation(of:blockSize:)`, plus global `heldTokenCount(of:)` and `drain(_:limit:)`, which is generic over `IteratorProtocol where Element == Int`.
    - Both chained suites now use these helpers. Their private `iterator`, `drain` and `prefillContinuation` copies and the inline while-let loops are gone. The bounded loop in the caching suite's `stoppedCall` now uses `drain(&decode, limit: stop)`.
    - The shared prefill reference uses maxTokens 1. The caching suite used 16 before; `promptBoundaryContinuation` is read before the first token, so maxTokens does not affect it.
    - Left alone: Qwen35MTPSamplingTests and Qwen35PartialRoundFinalizeTests keep their own `drain` copies. Those are older tests outside this change.
  timestamp: 2026-10-28T23:50:00.601625+00:00
- actor: claude-code
  id: 01m62eqejqqjsc35nt1yh726j4
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/BenchmarkHelpers/MTPBlockSweep.swift, Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift, Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift, Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift. `swift build --build-tests`: 0 compiler warnings (only the existing build-system "missing creator" note). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 1966 Swift Testing tests in 177 suites passed and 920 XCTest tests with 0 failures; Qwen35ChainedMTPDraftTests, Qwen35ChainedMTPCachingTests and ChatSessionQwenChainedMTPTests passed. All 6 findings checked.
    - next: /review
  timestamp: 2026-10-28T23:50:03.607426+00:00
- actor: claude-code
  id: 01m62etwaqjt5brj1yhgdqh0kq
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (c9ef346f) — 0 findings (0 confirmed, 3 refuted, 7 attempted); all 6 items of `## Review Findings (2026-10-28 18:34)` checked and verified in the diff (named pMin constant `reportedFastestMinimumDraftProbability`; shared `heldTokenCount(of:)`, generic `drain(_:limit:)`, `mtpIterator`/`greedyMTPIterator` with optional `mtpContinuation`, and `prefillContinuation(of:blockSize:)` in Qwen35ScriptedMTPSupport.swift, used by both chained suites)
    - next: task moved to done
  timestamp: 2026-10-28T23:51:55.991301+00:00
- actor: claude-code
  id: 01m62evagnng5g3ksb2d1fsmnx
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 files (shared test helpers, named sweep constant)
    - test: green — swift build --build-tests 0 warnings; MLXLMTests 1966 Swift Testing + 920 XCTest pass
    - commit: c9ef346f
    - review: clean — task moved to done
  timestamp: 2026-10-28T23:52:10.517002+00:00
depends_on:
- 01M5SKBV27GED43EB3T8MCSVWE
- 01M5SKBCEWG4D25SZZNGG7BVM7
- 01M5QWJXZ5EEBVH7Z9D3W7JKFN
- 01M5QWJ40KYCBZ9SW5CR9AH2BJ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: fff580
title: 'MTP: draft more than one token per round by chaining the Qwen MTP head'
---
## What
`Qwen35MTPDraftModel.maximumBlockSize = 2` (Libraries/MLXLLM/Models/Qwen35MTP.swift and Libraries/MLXVLM/Models/Qwen35MTP.swift), so each round drafts one token (measured: 1.70–1.96 tokens per verify). vLLM PR 62961 runs Qwen3.8-27B with 3 MTP tokens and reports a mean acceptance length of ~3.1. The multi-token rewind is the previous task; this task only adds the chained drafts.

`draftMTPTokenBlock` (now `MTPDraftBlockOutput.chainedMTPDrafts`, Libraries/MLXLMCommon/QwenMTP.swift) already loops `blockSize − 1` times and chains the MTP hidden. The real limit: `draftBlock` returns `state.seedToken` (one token) every round, because `commitDrafterState` always sets a seed.

### Findings from llama.cpp (common/speculative.cpp, `common_speculative_impl_draft_mtp`)
- Chained drafts 2+ use the MTP head's own output hidden after the shared head norm. Our `Qwen35MTPPredictor` already returns `norm(hidden)`.
- Default draft count `n_max = 3`. Qwen3.6-27B, 3 drafts: acceptance 72–80%, 1.7–2.3× decode on CUDA. On a small GPU 2 drafts was the fastest. The best count depends on the verify cost.
- A confidence gate (`--spec-draft-p-min`) stops the chain when the drafter top-1 probability < p_min. Users report that p_min 0.75 makes 5 drafts the fastest setting.

- [x] In `draftBlock` (both Qwen35MTP.swift files), return the seed plus up to `blockSize − 2` chained argmax steps from `seedHidden` through `draftMTPTokenBlock`. Stop the chain when the drafter top-1 probability < `pMin` (a new parameter of `SpeculativeDecodingConfig` and `generate(...mtpDrafter:)`, default 0). Track `proposalAppended` for each tentative drafter entry. Chained drafts work in both sampled modes (sample-and-match: argmax chain; rejection sampling: sampled chain with its distribution).
- [x] In `commitDrafterState`, trim the rejected tentative entries and append the accepted ones with the target hidden rows, so that drafter == main during decode stays true.
- [x] Raise `maximumBlockSize` to 6 and allow `blockSize` and `pMin` in `SpeculativeDecodingConfig(mtpDrafter:blockSize:)`. DEFAULT RULE (user, speed first): the shipped default blockSize/pMin is the fastest measured setting on the M5 Ultra across math/code/prose combined, and never slower than blockSize 2 on any category by more than noise. If no deeper setting wins, the default stays blockSize 2 and the numbers are recorded.

## Acceptance Criteria
- [x] Greedy output equals plain decode for blockSize 2..6, with pMin 0 and pMin 0.75, on the shared fixture (accept-mode and reject-mode drafters).
- [x] The drafter-count invariant holds after partial acceptance and after a pMin early stop.
- [x] Caching keeps working with deeper blocks: ChatSession checkpoints, drafter continuation (aligned boundary), prompt-cache save/load with drafter state, finalize mid-round. Tests at the new default blockSize for ChatSession multi-turn reuse (no full recompute) and edit/rewind.
- [x] Release benchmark on the M5 Ultra (`qwen38FourBitMTPGeneration` sweeping blockSize 2..6 and pMin 0, 0.75): no hard % threshold — gains add up; the goal is the fastest setting, measured, per the DEFAULT RULE above. Baseline before this task at blockSize 2: MTP ~91/81/79 tok/s (math/code/prose), plain ~59. The benchmark prints the acceptance for each draft depth. Record plain tok/s and machine load alongside.
- [x] The verify of S tokens goes through `MultiRowQuantizedMatvec.run(asSpeculativeVerify:)` and the compiled verify schedule (S = 2..6); a >6-token staged round skips them, so maximumBlockSize 6 keeps Qwen rounds on the native path.
- [x] `qwen38FourBitMTPGeneration()` at the new default: output equals plain or near-tie per the MTPChatTokenMatch rule. `qwen38FourBitMTPSampledGeneration()` and `qwen38FourBitMTPSystemPromptCache()` pass with no known issues (edit turn ~123 tokens with MTP on).

## Tests
- [x] New test files (Qwen35MTPTests.swift is ~104 KB; each new file well under 40 KB; reuse Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift): blockSize 3..6 equality, drafter-count invariant, pMin early stop, sampled modes, ChatSession reuse and edit at the default.
- [x] `swift build --build-tests` with 0 compiler warnings; all 5 test bundles via `xcrun xctest`; then the Release benchmarks via xcodebuild (IntegrationTesting scheme).

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf

## Review Findings (2026-10-28 18:34)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 20 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWKB8C14RB75S80HCFZYEM.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWKB8C14RB75S80HCFZYEM.md` — no validator matches this file
> - `.kanban/tasks/01M62C57EC6FHXX6CBWHBQ0F3N.jsonl` — no validator matches this file
> - `.kanban/tasks/01M62C57EC6FHXX6CBWHBQ0F3N.md` — no validator matches this file

- [x] `Libraries/BenchmarkHelpers/MTPBlockSweep.swift:113` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift:63` `reuse/reuse` — The expression `KVCacheStorage(cache, plan: .disabled).processedTokenCount` that counts the tokens a cache holds is written out again in the draft suite. The two copies can drift if the way a held count is read changes. Add a small shared helper, such as `heldTokenCount(of cache: [KVCache]) -> Int`, in the test support file and use it in both suites.
- [x] `Tests/MLXLMTests/Qwen35ChainedMTPCachingTests.swift:133` `reuse/reuse` — The `while let token = second.next()` loop that collects every token of the restored iterator repeats the drain loop `drain(_:)` already implements in the draft suite. Each suite keeps its own copy, so the token-collection logic exists in two places. Move the drain helper into shared test support, next to `collectChatTurn` in Qwen35ScriptedMTPSupport.swift, and call it from both suites. Replace the inline loop at this line with that helper.
- [x] `Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:78` `reuse/reuse` — The private `iterator(_:blockSize:minimumDraftProbability:cache:parameters:)` builds a greedy `MTPSpeculativeTokenIterator` over the fixture. The caching suite builds the same iterator through its own private `iterator(_:input:cache:continuation:blockSize:maxTokens:)`, with the same initializer and greedy parameters. The two construction paths can drift apart. Put one shared greedy-iterator builder in the test support file and have both suites call it. The caching suite needs a continuation parameter, so the shared builder should take an optional `mtpContinuation`.
- [x] `Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:131` `reuse/reuse` — The inline `while let token = plain.next()` loop collects the plain `TokenIterator`'s tokens by hand. The same accumulation is already written as `drain(_:)` in this suite, but that helper is typed for `MTPSpeculativeTokenIterator`, so the loop is duplicated instead of shared. Make the shared drain helper generic over any iterator whose `next()` returns `Int?`, and use it for both the plain and the speculative iterators in this test.
- [x] `Tests/MLXLMTests/Qwen35ChainedMTPDraftTests.swift:309` `reuse/reuse` — `chainedDecodeLeavesThePrefillContinuation` builds its reference prefill continuation inline, constructing an `MTPSpeculativeTokenIterator` over the held tokens and reading `.promptBoundaryContinuation`. That is the same construction `prefillContinuation` already performs in the sibling caching suite. Two copies of the reference-building logic can drift apart. Move the prefill-continuation helper into shared test support, next to `collectChatTurn` in Qwen35ScriptedMTPSupport.swift, and call it from both suites. Alternatively, have this test call the caching suite's helper if it is made internal.