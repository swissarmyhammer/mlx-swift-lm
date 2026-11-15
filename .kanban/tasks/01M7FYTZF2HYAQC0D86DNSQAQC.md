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