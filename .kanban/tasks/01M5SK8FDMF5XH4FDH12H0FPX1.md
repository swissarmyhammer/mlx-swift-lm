---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzr0ttt5993g35vame3b6
  text: 'Change after the PR 581/459 port (commits 843854ec, 82f273fd): the fake models and drafters of Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift, ChatSessionMTPEarlyTerminationTests.swift and MTPSpeculativeTokenIteratorTests.swift (MockDrafter, StatefulMockDrafter) are private to their files. Move them into the shared fixture target.'
  timestamp: 2026-10-25T13:46:16.986966+00:00
- actor: claude-code
  id: 01m5zg46skr3cwagwp6xmyr5qj
  text: |-
    Research (picked up on mtp @ 400c7488):
    - The `makeFixture` the card cites (~929 in c7724fef, now ~1321) is the checkpoint-loading fixture of `Qwen35CheckpointLoadingTests`: it writes a drafter config.json + safetensors and returns an `MTPDrafterModelFactory`; needs MLXVLM for the vision drafter. The in-memory target+drafter tests (Qwen35MTPMetalTests) build `Qwen35TextModel(cfg)` + `Qwen35MTPDraftModel(cfg)` inline on UNSEEDED random weights from `qwen35TextConfigJSON(mtpLayers: 1)` (1 layer, all attention, vocab 16) — the two ChatSession tests there are the flaky-prone shape.
    - Plan: new non-product target `MLXLMTestSupport` at Tests/MLXLMTestSupport (path under Tests/ so validator test-code carve-outs apply), deps MLXLMCommon/MLXLLM/MLXVLM/MLX/MLXNN, public API only (no @testable; `package` access is fine). Linked by MLXLMTests and MLXFoundationModelsTests. The FM builder lives in MLXFoundationModelsTests (only FM tests need MLXFoundationModels).
    - Seeded weights: `withRandomState(MLXRandom.RandomState(seed:)) { Model(cfg) }` (pattern from ExecutorPromptCacheQwenFileTests).
    - Copy mode design: the Qwen MTP drafter shares the target's embedding and LM head. With the drafter's fc set to [I, 0] (embedding half) and its layer outputs silenced (o_proj, down_proj = 0), it proposes argmax head(norm(E[x])) — the target's token path. The target's embedding is scaled so the token path outweighs the context path in the residual, so the target agrees with that proposal on most steps but context still flips some (both accept and reject rounds). Tied embeddings would make the token path predict x itself (self dot product), so the fixture unties them.
    - The JSON builders (qwen35TextConfigJSON etc., 57 call sites in 4 files) move to the support target as statics on an enum namespace (idioms rule: a function belongs to a type).
    - The card comment asks to move the private fakes of ChatSessionQwenTextMTPTests, ChatSessionMTPEarlyTerminationTests and MTPSpeculativeTokenIteratorTests (MockDrafter, StatefulMockDrafter, MockMainModel, ...) into the support target too; no name collisions found in Tests/.
    - Single-suite runs: `swift test` cannot load the metallib (CLAUDE.md); runs go through `xcrun xctest`.
  timestamp: 2026-10-27T20:16:46.899869+00:00
- actor: claude-code
  id: 01m5zgn2d3njqdzmxhnawcc7ms
  text: |-
    Progress: MLXLMTestSupport target + Qwen35MTPFixture + ScalarTokenizer + Qwen35FixtureJSON landed; Qwen35MTPFixtureSmokeTests (MLXLMTests) went RED (no such module) then GREEN (6 tests).

    What did NOT work for the copy mode: a drafter that reads only the token embedding (fc=[I,0], layer outputs zeroed) against a target whose embeddings are scaled up (gain). Gain 4: acceptance 0.12; gain 6-8: 0.84-1.0 but the target becomes a bigram model that falls into a short cycle; gain >= 10: the target no longer depends on the context at all (two prompts with the same last token decode the same). Rejected: it trades away the context sensitivity that checkpoint/restore tests need.

    What works: target untouched except its embedding rows normalized to RMS 1 (so the drafter's pre-fc RMSNorm is a no-op on them); copy mode gives the drafter's one MTP layer the weights of the target's full-attention layer (model.layers.1.* -> mtp.layers.0.*) and fc = [I, 0]. The drafter cache holds the same tokens one position earlier, which RoPE does not see, so the drafter runs the target minus the GDN layer and minus the first prompt token. Measured acceptance over 48 greedy tokens: seed 35 -> 0.64, seeds 7/101/2024 -> 0.81/0.52/0.62; independent drafter -> 0.0 on all four. Target stays context-sensitive in both modes.

    Tool note: the `files` edit op ignores `replace_all` (one site per call), so the 43-site rename to Qwen35FixtureJSON.* was done with one perl substitution.

    Filter tip: `xcrun xctest -XCTest 'MLXLMTests.<SuiteName>' <bundle>` filters Swift Testing suites too (~5 s per suite run).
  timestamp: 2026-10-27T20:25:59.459847+00:00
- actor: claude-code
  id: 01m5zhf2ndmkaz3tsw9h59m30a
  text: |-
    Implementation landed (uncommitted, branch mtp):
    - New non-product target `MLXLMTestSupport` (Tests/MLXLMTestSupport, deps MLXLMCommon/MLXLLM/MLXVLM/MLX/MLXNN, public API + package access only, no @testable), linked by MLXLMTests and MLXFoundationModelsTests.
    - `Qwen35MTPFixture(drafter: .copiesTarget | .independent, seed:)`: hybrid target (GDN layer 0, full attention layer 1), hidden 32, vocab 64, untied head, seeded via withRandomState; `makeModelContext/makeModelContainer/makeSpeculativeDecodingConfig/makeChatSession(speculative:maxTokens:)`, `decodeGreedily(prompt:maxTokens:speculative:)`, `text(for:)`. Hidden 16 made the greedy decode fall into a 3-cycle within 6 tokens; 32 does not (copy acceptance 0.88 at hidden 32 vs 0.64 at 16).
    - `ScalarTokenizer` + `ScalarTokenizerInputProcessor`: token n <-> U+E000+n, so replies re-render to the same tokens (multi-turn cache reuse).
    - `Qwen35FixtureJSON` (moved qwen35*ConfigJSON builders, + tieWordEmbeddings), `Qwen35MTPCheckpointFixture` (moved Qwen35MTPTests.makeFixture/fixtureConfiguration/makeDrafter), `UnusedTokenizerLoader`.
    - MTPFakes/: the fakes of MTPSpeculativeTokenIteratorTests (StatefulMockDrafter, MockDrafter, MockMainModel, CountingKVCache, NonTrimmableCountingKVCache, MutableOffsetKVCache, PositionScriptedMainModel, preservedStateKey), ChatSessionQwenTextMTPTests (MTPStateEmittingTarget, QwenStyleMTPDrafter, MTPStagingProbe, SpeculativeStagingCache, StatelessMTPDrafter, NonResumableMTPDrafter, stagingTargetMarkerKey) and ChatSessionMTPEarlyTerminationTests (EarlyTerminationMTPTarget/Drafter), made public; their literals named in `SyntheticMTPValues` / per-fake statics. Tokenizers and input processors of those files stayed private (not models/drafters).
    - Qwen35MTPTests: checkpoint tests use Qwen35MTPCheckpointFixture; testQwen35TextChatSessionResumesActualDrafterState now runs on Qwen35MTPFixture(drafter: .independent) (seeded, hybrid); the VLM ChatSession test uses ScalarTokenizer.
    - FM builder: `Qwen35MTPFixture.makeLanguageModel(weights:)` in Tests/MLXFoundationModelsTests/Qwen35MTPFixture+LanguageModel.swift. Its smoke test needs `GenerationOptions(samplingMode: .greedy, ...)`: the FM default sampling is not greedy and the response differed from plain decode until greedy was set.
    - Discovery: the FM executor renders a bare user prompt to exactly the tokens of its text through the fixture processor, so LanguageModelSession output == fixture.decodeGreedily on the same tokens.
    - Note for later tasks: the fixture is text-only; the VLM target asked for by ^pv3c64m / ^vgma4mq-era tests is not in it yet (Qwen35FixtureJSON.visionLanguage and Qwen35MTPCheckpointFixture(vision:) exist).
  timestamp: 2026-10-27T20:40:11.693467+00:00
- actor: claude-code
  id: 01m5zhf7m00mbx00ag7tbc7b56
  text: |-
    ### implement — changed
    - evidence: `swift build --build-tests` clean (only the pre-existing "missing creator ... mlx-swift_Cmlx.bundle" build-system warning); `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` 895 XCTest 0 failures + 1842 Swift Testing passed (baseline 1836 + 6 new); `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` 376 Swift Testing passed. RED observed for both smoke files (missing module / missing makeLanguageModel) before GREEN. 20 files: Package.swift; Tests/MLXLMTestSupport/{Qwen35FixtureJSON,Qwen35MTPFixture,Qwen35MTPCheckpointFixture,ScalarTokenizer}.swift; Tests/MLXLMTestSupport/MTPFakes/{MTPIteratorFakes,ChatSessionMTPFakes,EarlyTerminationMTPFakes,SyntheticMTPValues}.swift; Tests/MLXLMTests/{Qwen35MTPFixtureSmokeTests,Qwen35MTPTests,MTPSpeculativeTokenIteratorTests,ChatSessionQwenTextMTPTests,ChatSessionMTPEarlyTerminationTests,HybridRecurrentCacheOffsetTests,Qwen35RecurrentCachePositionTests,ChatSessionRecurrentCheckpointTests}.swift; Tests/MLXFoundationModelsTests/{Qwen35MTPFixtureSmokeTests,Qwen35MTPFixture+LanguageModel}.swift
    - next: /review
  timestamp: 2026-10-27T20:40:16.768072+00:00
- actor: claude-code
  id: 01m5zhqmewf79b3q5119smppem
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0; xcrun xctest MLXLMTests: 895 XCTest, 0 failures + 1842 Swift Testing passed; MLXFoundationModelsTests: 376 passed (43 suites), 0 failures.
    - build warning: "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS" is the only warning. It references only the Cmlx bundle, no file this change touched, so it is the pre-existing SwiftPM build-graph message. No other warnings.
    - skipped: Swift Testing logs "benchmarkVerification() skipped" (Tests/MLXLMTests/MTPGreedyVerificationTests.swift, gated on MTP_GREEDY_BENCHMARK=1). The file is committed and untouched by this change; it is already inside the 1842 baseline. Reported, not changed (out of scope).
    - next: review
  timestamp: 2026-10-27T20:44:52.060510+00:00
- actor: claude-code
  id: 01m5zjb42jcagwg7weg9eqb8sq
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 6 findings (6 confirmed, 12 refuted) — Tests/MLXLMTestSupport/MTPFakes/EarlyTerminationMTPFakes.swift:158, Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:167 (x2), Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:168, Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:510 (x2)
    - next: implement the Review Findings (2026-10-27 15:45) checklist, then re-review
  timestamp: 2026-10-27T20:55:30.642992+00:00
- actor: claude-code
  id: 01m5zjbqtz99s026ezg30hcv4g
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 20 files (MLXLMTestSupport target, fakes moved, FoundationModels fixture)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1842 Swift Testing; MLXFoundationModelsTests 376
    - commit: be7f0e6d
    - review: findings — Tests/MLXLMTestSupport/MTPFakes/EarlyTerminationMTPFakes.swift:158, Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:167, :168, :510
  timestamp: 2026-10-27T20:55:50.879613+00:00
- actor: claude-code
  id: 01m5zjps79qrvexxdxfa90w87x
  text: |-
    Review findings (2026-10-27 15:45) worked:
    - Added `SyntheticMTPValues.makeRepeatedDraft(of:batchSize:blockSize:)` (SyntheticMTPValues.swift now imports MLX). Named with `make` per the swift/fluent-usage factory rule. Every drafter that built the repeated `[batch, blockSize - 1]` block now routes through it: EarlyTerminationMTPDrafter (EOS on queryOffset 1, else ordinary token), QwenStyleMTPDrafter and StatelessMTPDrafter (ChatSessionMTPFakes.swift), and StatefulMockDrafter (MTPIteratorFakes.swift, its draftedTokenValue) — the last was a fourth copy of the same block not named in the finding.
    - MTPIteratorFakes.swift: `[MLXArray]()` x2 -> `var tokens: [MLXArray] = []` / `var logitsRows: [MLXArray] = []`; `[String: (MLXArray, MLXArray)]()` -> `var sharedKV: [String: (MLXArray, MLXArray)] = [:]`, exactly as the findings wrote. No other constructor-style empty collection is left in Tests/MLXLMTestSupport/MTPFakes/.
    - Left as is (outside the MTPFakes files, and not a variable declaration): Tests/MLXLMTestSupport/Qwen35MTPCheckpointFixture.swift `root["vision_config"] = [String: Any]()` — a bare `[:]` there has no element type to infer from the `[String: Any]` subscript value; noting it for the next reviewer.
    - Tool note: `update task` / `move task` return the whole board (~78 KB) and overflow the tool-result limit; the writes still land (verified in .kanban/tasks/01M5SK8FDMF5XH4FDH12H0FPX1.md).
  timestamp: 2026-10-27T21:01:52.745547+00:00
- actor: claude-code
  id: 01m5zjpw07r58atjhnwxzd83r1
  text: |-
    ### implement — changed
    - evidence: 4 files — Tests/MLXLMTestSupport/MTPFakes/{SyntheticMTPValues,EarlyTerminationMTPFakes,ChatSessionMTPFakes,MTPIteratorFakes}.swift. `swift build --build-tests` exit 0 (only the pre-existing "missing creator ... mlx-swift_Cmlx.bundle" warning); `xcrun xctest MLXLMTests.xctest` 895 XCTest 0 failures + 1842 Swift Testing passed; `xcrun xctest MLXFoundationModelsTests.xctest` 376 passed. All 6 Review Findings (2026-10-27 15:45) checked.
    - next: /review
  timestamp: 2026-10-27T21:01:55.591354+00:00
- actor: claude-code
  id: 01m5zjyyhg38z17aj4sh8ke1b7
  text: |-
    ### test — green
    - evidence: swift build --build-tests (up to date; only the known "missing creator ... mlx-swift_Cmlx.bundle" SwiftPM message); xcrun xctest MLXLMTests.xctest — 895 XCTest, 0 failures, plus 1842 Swift Testing passed; MLXFoundationModelsTests.xctest — 376 passed, 0 failures. Only skip is the known env-gated benchmarkVerification() (upstream PR 724). No new warnings.
    - next: review/commit
  timestamp: 2026-10-27T21:06:20.336719+00:00
- actor: claude-code
  id: 01m5zk10fg83tp23p8jrn3g59w
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (94d74e41) — 0 findings, 0 confirmed, 0 refuted (7 attempted, 4 files reviewed). All 6 items of Review Findings (2026-10-27 15:45) checked and verified: SyntheticMTPValues.makeRepeatedDraft is used by EarlyTerminationMTPFakes, ChatSessionMTPFakes (x2), MTPIteratorFakes; no `[MLXArray]()` / `[String: (MLXArray, MLXArray)]()` constructors remain in MTPIteratorFakes.swift.
    - next: none — task moved to done.
  timestamp: 2026-10-27T21:07:27.856622+00:00
- actor: claude-code
  id: 01m5zk1ew1xjpsfy3tkvyybvke
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 files (shared makeRepeatedDraft helper, empty literals)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1842 Swift Testing; MLXFoundationModelsTests 376
    - commit: 94d74e41
    - review: clean — 0 findings, prior 6/6 verified; task done
  timestamp: 2026-10-27T21:07:42.593423+00:00
depends_on:
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffe680
title: 'MTP: shared Qwen35 + MTP test fixture for MLXLMTests and MLXFoundationModelsTests'
---
## What
Many MTP tasks need a small Qwen35 target with an MTP head that runs on random weights in unit tests. Today `makeFixture` is private in Tests/MLXLMTests/Qwen35MTPTests.swift (~929). The shared scripted ChatSession builder (Tests/MLXLMTests/ScriptedChatSession.swift, commit 478abaa1) takes a `ScriptedTokenModel`, which fails `Qwen35MTPDraftModel.isCompatible(with:)` (Libraries/MLXLLM/Models/Qwen35MTP.swift ~117-119). `MLXFoundationModelsTests` cannot see helpers in `MLXLMTests`.

- [x] Add a test-support target (for example `MLXLMTestSupport` in Package.swift, linked by `MLXLMTests` and `MLXFoundationModelsTests`) with `Qwen35MTPFixture`: a tiny Qwen35 text config (GDN + attention layers, small vocabulary), a matching `Qwen35MTPDraftModel`, a seeded weight init, a tokenizer stub, and a `ModelContainer` builder.
- [x] Give the fixture a mode where the drafter copies the target (high acceptance) and a mode with independent random weights (low acceptance), so that tests can force accept and reject rounds.
- [x] Move the private `makeFixture` users in Qwen35MTPTests.swift to the shared fixture.
- [x] Add a FoundationModels builder that makes an `MLXLanguageModel` on the fixture container, so that executor tests can run `LanguageModelSession` on it.

## Acceptance Criteria
- [x] The existing Qwen35MTPTests pass on the shared fixture with no change in results.
- [x] A smoke test in each of `MLXLMTests` and `MLXFoundationModelsTests` builds the fixture and runs one greedy MTP stream that equals plain decode.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPFixtureSmokeTests.swift and Tests/MLXFoundationModelsTests/Qwen35MTPFixtureSmokeTests.swift.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #testing

## Review Findings (2026-10-27 15:45)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 19 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5SK8FDMF5XH4FDH12H0FPX1.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SK8FDMF5XH4FDH12H0FPX1.md` — no validator matches this file

- [x] `Tests/MLXLMTestSupport/MTPFakes/EarlyTerminationMTPFakes.swift:158` `reuse/reuse` — EarlyTerminationMTPDrafter.draftedTokens reimplements the repeated-token draft block that the drafters in ChatSessionMTPFakes.swift already build. It is a parallel copy instead of one shared helper. Route this drafter through the shared repeated-draft helper proposed for SyntheticMTPValues, passing the end-of-sequence or ordinary token as the parameter.
- [x] `Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:167` `swift/idioms` — The empty array is built with a constructor call, `[MLXArray]()`, instead of an annotated empty literal. The idiom is `var tokens: [MLXArray] = []`. Write `var tokens: [MLXArray] = []`.
- [x] `Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:167` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[MLXArray]()' with ': [MLXArray] = []'.
- [x] `Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:168` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[MLXArray]()' with ': [MLXArray] = []'.
- [x] `Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:510` `swift/idioms` — The empty dictionary is built with a constructor call, `[String: (MLXArray, MLXArray)]()`, instead of an annotated empty literal. Write `var sharedKV: [String: (MLXArray, MLXArray)] = [:]`.
- [x] `Tests/MLXLMTestSupport/MTPFakes/MTPIteratorFakes.swift:510` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[String: (MLXArray, MLXArray)]()' with ': [String: (MLXArray, MLXArray)] = [:]'.
