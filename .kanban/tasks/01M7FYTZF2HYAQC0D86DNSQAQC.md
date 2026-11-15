---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7g47a0n0yhg600wstp3ecpq
  text: 'Research: only two library sites pin 512 — `GuidedGenerationLoop.run` (both overloads, default arg `.init(stepSize: PrefillParameters.defaultStepSize)`) and `Executor.runGuidedLoop` (`Self.prefillStepSize ?? PrefillParameters.defaultStepSize`), which serves both schema output and required tool calls. `GuidedTarget.prefill` hands the parameters straight to `model.prepare`, so nil reaches each model''s own `forEachChunk(defaultStepSize:)`. Every model without a default resolves nil to `PrefillParameters.defaultStepSize` (512) via `resolvedStepSize()`/`forEachChunk` defaults, so their effective step is unchanged; Gemma3Text goes 512 -> 128 and Flint 512 -> 2048. Other FM sites (`makeParameters`, preload `.init(stepSize: prefillStepSize)`) already pass nil through. Test helpers `GuidedFixtureRun.generate` and `GuidedGenerationMTPRouterPathTests.Comparison` mirror the old default and get `.init()` (Qwen3.5 fixtures have no own default, so same effective step). Reranker''s own `prefillStepSize: Int = 512` option is not a guided path and is left alone.'
  timestamp: 2026-11-15T17:31:32.757395+00:00
- actor: claude-code
  id: 01m7g4tafhmhst641we1xyt3mp
  text: |-
    Implementation landed (TDD). RED: `GuidedPrefillStepTests.aModelWithItsOwnDefaultStepPrefillsWithIt` (MLXGuidedGenerationTests) and `GuidedPrefillStepSizeTests.aGuidedPassWithoutAStepPrefillsWithTheModelDefault` (MLXFoundationModelsTests) both failed with the model resolving 512 instead of its own default; GREEN after `GuidedGenerationLoop.run` defaults `prefill` to `.init()` and `runGuidedLoop` passes `.init(stepSize: Self.prefillStepSize)`. Guards that pass before and after: a model without a default still resolves `PrefillParameters.defaultStepSize`; a served-model step (8) still wins. `ByteTokenizer`/`UnusedProcessor` moved from GuidedForwardStateTests into GuidedLoopTestSupport.swift (as `GuidedByteTokenizer`, named EOS/vocab constants) so both guided-loop suites share them.

    Real model (Flint oQ4e, scratchpad flintcli `guided` mode, JSON schema, time to first emit = prefill): 8017-token prompt 512-step ~5900 tok/s (1.35 s) -> default 2048 ~8750 tok/s (0.92 s), ~+48%; 12017-token prompt 5645 -> 8366 tok/s (~+48%). `auto` output is byte-identical to explicit 2048, proving the default reaches Flint. All 15 outputs valid against the schema. NOT byte-unchanged vs 512: at 8k the 512 run adds a third topic; the unchunked reference (`.unchunked`) gives exactly the new 2048 output. At 12k, 512/2048/unchunked each differ slightly (confidence 95 vs 100, topic wording) — Flint oQ4e guided output is sensitive to the prefill chunk boundaries (256, 1024, 4096 also each differ), same numerics trade ^tty395a already accepted for plain generation. flintcheck: MATCH x4 + TOOL CALL get_weather(Paris).
  timestamp: 2026-11-15T17:41:55.825208+00:00
- actor: claude-code
  id: 01m7g573tfg5g1cen3x2thnb57
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift (docs), Tests/MLXGuidedGenerationTests/{GuidedPrefillStepTests,GuidedLoopTestSupport,GuidedForwardStateTests}.swift, Tests/MLXFoundationModelsTests/{PrefillStepSizeTests,GuidedGenerationMTPTestSupport,GuidedGenerationMTPRouterPathTests}.swift. Tests: MLXLMTests 2344 Swift Testing + 1098 XCTest (4 LFM2 env skips, 3 known issues = baseline), MLXGuidedGenerationTests 78, MLXFoundationModelsTests 496, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11, 0 failures. Flint oQ4e guided JSON prefill 8017 tok: 512 ~5900 -> 2048 ~8750 tok/s; 12017 tok: 5645 -> 8366 tok/s. Outputs all schema-valid. They are not byte-identical to the 512 run, because the output depends on where the chunk boundaries fall. At 8k the new output equals the unchunked reference. flintcheck MATCH x4 + TOOL CALL.
    - next: /review
  timestamp: 2026-11-15T17:48:54.991906+00:00
- actor: claude-code
  id: 01m7g5nxeppzw1aprnr78rx8j8
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 warnings). xctest per bundle: MLXLMTests 1098 XCTest (4 LFM2 env skips, 0 failures) + 2344 Swift Testing passed, 3 known issues (baseline); GDNDecodeKernelLayerTests passed. MLXGuidedGenerationTests 78 passed. MLXFoundationModelsTests 496 passed. CXGrammarTests 7 passed. MLXHuggingFaceMacrosTests 29 XCTest, 0 failures. swift-format lint -s clean on all 9 changed files. flintcheck (run alone, after MLXLMTests finished): MATCH x4, TOOL CALL get_weather.
    - next: review
  timestamp: 2026-11-15T17:56:59.990137+00:00
- actor: claude-code
  id: 01m7g5ptbybj8m9wfw9z4eac54
  text: |-
    ### commit — changed
    - evidence: commit subject `perf(guided): let the model's prefill step apply to guided passes (^dnsqaqc)`; files: GuidedGenerationLoop.swift, MLXLanguageModelGeneration.swift, MLXLanguageModel.swift, GuidedPrefillStepTests.swift, GuidedLoopTestSupport.swift, GuidedForwardStateTests.swift, PrefillStepSizeTests.swift, GuidedGenerationMTPTestSupport.swift, GuidedGenerationMTPRouterPathTests.swift, .kanban/
    - next: review
  timestamp: 2026-11-15T17:57:29.598802+00:00
- actor: claude-code
  id: 01m7g5we1texz9r7phwcsm5x8p
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (792cd657); 1 finding (1 confirmed, 5 refuted) — Tests/MLXFoundationModelsTests/PrefillStepSizeTests.swift:116
    - next: implement the Review Findings (2026-11-15 11:57) checklist item, then re-review
  timestamp: 2026-11-15T18:00:33.594565+00:00
- actor: claude-code
  id: 01m7g5wyqkzw4vk8y8j5my5c2e
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — GuidedGenerationLoop.swift, MLXLanguageModelGeneration.swift, MLXLanguageModel.swift, 6 test files
    - test: green — MLXLMTests 1098 XCTest + 2344 ST, Guided 78, FM 496, CXGrammar 7, Macros pass; flintcheck MATCH x4 + TOOL CALL
    - commit: 792cd657
    - review: findings — Tests/MLXFoundationModelsTests/PrefillStepSizeTests.swift:116 (duplicate PrefillStepRecordingModel, reuse)
  timestamp: 2026-11-15T18:00:50.675194+00:00
- actor: claude-code
  id: 01m7g6kyr5ejr1j9da6qbxgbj2
  text: 'Fixed the reuse finding. I moved one `PrefillStepRecordingModel` (the form with an optional `ownDefaultStepSize`) and `GuidedByteTokenizer` into MLXLMTestSupport as `public` (Tests/MLXLMTestSupport/PrefillStepRecordingModel.swift and GuidedByteTokenizer.swift). MLXGuidedGenerationTests now depends on MLXLMTestSupport (Package.swift). MLXFoundationModelsTests already did. I deleted both copies of the model and the tokenizer copy in GuidedLoopTestSupport.swift; `UnusedProcessor` stays there. GuidedForwardStateTests and GuidedPrefillStepTests now import MLXLMTestSupport. `GuidedByteTokenizer` fits the FM guided pass: the JSON boolean grammar is built from its vocabulary through `guidedGrammar`, and both FM prefill tests pass. `ToolCallAlphabetTokenizer` is no longer used in PrefillStepSizeTests, and the unused MLX/MLXNN imports there are removed. The `missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS` line from `swift build` is a swiftbuild build-system warning. It also shows on the stashed baseline, so it does not come from this change.'
  timestamp: 2026-11-15T18:13:24.357050+00:00
- actor: claude-code
  id: 01m7g6m1d3wpzmc13yr3ytb518
  text: |-
    ### implement — changed
    - evidence: Package.swift, Tests/MLXLMTestSupport/{PrefillStepRecordingModel,GuidedByteTokenizer}.swift (new), Tests/MLXFoundationModelsTests/PrefillStepSizeTests.swift, Tests/MLXGuidedGenerationTests/{GuidedPrefillStepTests,GuidedLoopTestSupport,GuidedForwardStateTests}.swift. swift build --build-tests clean (only the baseline swiftbuild node warning). xctest: MLXLMTests 1098 XCTest (4 skipped, 0 failures) + 2344 ST passed, 3 known issues (baseline). MLXGuidedGenerationTests 78 passed. MLXFoundationModelsTests 496 passed. CXGrammarTests 7 passed. MLXHuggingFaceMacrosTests 11 XCTest, 0 failures. swift-format applied.
    - next: /review
  timestamp: 2026-11-15T18:13:27.075780+00:00
- actor: claude-code
  id: 01m7g71vq0c45hq1httcfp6dkp
  text: |-
    ### test — green
    - Libraries/ unchanged (git diff --stat lists only Package.swift, Tests/, .kanban/).
    - swift build --build-tests: complete, no compiler warnings (only the swiftbuild "missing creator for mutated node" tool message).
    - MLXLMTests: XCTest 1098 executed, 4 skipped (LFM2 env), 0 failures; Swift Testing 2344 tests in 223 suites passed, 3 known issues; GDNDecodeKernelLayerTests passed.
    - MLXGuidedGenerationTests: 78 tests in 18 suites passed.
    - MLXFoundationModelsTests: 496 tests in 67 suites passed.
    - CXGrammarTests: 7 tests in 5 suites passed.
    - MLXHuggingFaceMacrosTests: 11 XCTest, 0 failures.
    - swift-format lint -s on the 7 changed Swift files: clean.
    - Real-model guard not run (Libraries/ unchanged).
    - next: review/commit
  timestamp: 2026-11-15T18:21:00.000562+00:00
- actor: claude-code
  id: 01m7g72m751j9rd2tpxhxd5wg2
  text: |-
    ### commit — changed
    - evidence: test(guided): share the prefill-recording model and the byte tokenizer through MLXLMTestSupport, found in review of ^dnsqaqc (files: Package.swift, Tests/MLXLMTestSupport/GuidedByteTokenizer.swift, Tests/MLXLMTestSupport/PrefillStepRecordingModel.swift, Tests/MLXFoundationModelsTests/PrefillStepSizeTests.swift, Tests/MLXGuidedGenerationTests/*, .kanban/)
    - next: review
  timestamp: 2026-11-15T18:21:25.093508+00:00
position_column: doing
position_ordinal: '80'
title: Guided generation pins the prefill step to 512 and overrides the model default
---
## Why
Found in ^tty395a. `PrefillParameters.stepSize == nil` lets each model pick its own prefill step (Gemma3Text 128, Flint 2048 since ^tty395a). Guided generation never passes nil: `GuidedGenerationLoop.run` defaults `prefill` to `.init(stepSize: PrefillParameters.defaultStepSize)`, and `MLXLanguageModelGeneration` passes `Self.prefillStepSize ?? PrefillParameters.defaultStepSize`. So a guided (schema / grammar) FoundationModels request on Flint prefills in 512-token chunks, which measured ~25% slower than 2048 at 8k-16k on oQ4e, and Gemma 3 gets 512 instead of its tuned 128.

## What
- Pass `stepSize: nil` (or the caller's value) through both sites, so `prepare` picks the model default.
- Check every other site that writes `PrefillParameters.defaultStepSize` explicitly where nil was meant.

## Acceptance
- A test shows a guided-generation prefill on a model with its own default reads that default.
- All five bundles are green. #flint-perf

## Review Findings (2026-11-15 11:57)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 9 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTG4PAPKM8D19MV6CGCP6.md` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FYTZF2HYAQC0D86DNSQAQC.md` — no validator matches this file

- [x] `Tests/MLXFoundationModelsTests/PrefillStepSizeTests.swift:116` `reuse/reuse` — This change adds a second `PrefillStepRecordingModel` that duplicates the one already added at `Tests/MLXGuidedGenerationTests/GuidedPrefillStepTests.swift:14`. Both have the same `prepare` that records `prefill.resolvedStepSize(...)`, the same `newCache`, and the same `callAsFunction` that builds EOS-preferring logit rows. The copy here is a strict subset: it takes a non-optional `ownDefaultStepSize` and does not handle a model with no default. The near-match was not generalized, so the two copies can drift apart. Keep one `PrefillStepRecordingModel`. The two files are in different test targets, so put it in a shared test-support target that both link (for example `Tests/MLXLMTestSupport`, which already holds the shared MTP fakes), and use the optional `ownDefaultStepSize` form. Then delete the copy in `PrefillStepSizeTests.swift`, and reuse `GuidedByteTokenizer` from `GuidedLoopTestSupport.swift` in place of `ToolCallAlphabetTokenizer` if that fits.
