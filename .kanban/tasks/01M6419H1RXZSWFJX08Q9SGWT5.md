---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m641p75b325p03grfd9payxd
  text: |-
    Picked up on a worktree reset to mtp 268dab08. Research:
    - Every non-guided pass of MLXLanguageModel (text, reasoning, allowed tools, think-then-call phase 1) ends in `Executor.observeCompletion(info)`; GenerateCompletionInfo carries `speculativeDecodingTelemetry` (draftTokenCount / acceptedDraftTokenCount, nil when roundCount == 0) and `speculativeDecodingFallbackReason` (.incompatibleDrafter on the FM paths).
    - Guided passes (schema, required-tool phase 2) run `GuidedGenerationLoop.run`, which returns only a token count; its MTP rounds reach only `GuidedGenerationDiagnosticSink` (recordSpeculativeRound, recordSpeculationFallback with a stated reason string, no case). The sink is task-local and test-bound; binding a full sink in production would record a mask + bias copy per decision and run a top-two sort + `.item` sync per token (GuidedDecoder.recordDecisionMargin), so production needs a speculation-only sink mode. Tests (GuidedGenerationMTPTestSupport) bind an outer sink across an executor call, so the executor must read deltas from an outer sink rather than shadow it.
    - `Transcript.Response.metadata` is public ([String: GeneratedContent]) on the 27 SDK, so tests can assert the real transcript entry of a session turn.
    Plan: one per-call `SpeculativeDecodingOutcome` (task-local, bound in respond when speculativeDecoding != nil) fed by observeCompletion and by a guided-pass wrapper reading sink deltas; emitted once on the response entry at the end of runRespond. Fallback reason = the first fallback a pass of the call reported.
  timestamp: 2026-10-29T14:40:40.619371+00:00
- actor: claude-code
  id: 01m642y61p42j9s59n91nwhrpj
  text: |-
    Implementation landed in 6354610a on worktree-agent-a96eeb70bc03d7a68 (on top of mtp 268dab08).
    - New Libraries/MLXFoundationModels/SpeculativeDecodingOutcome.swift: per-call tally, task-local `current`, bound in Executor.respond only when model.speculativeDecoding != nil; fed by Executor.observeCompletion (GenerateCompletionInfo.speculativeDecodingTelemetry + speculativeDecodingFallbackReason) and by withGuidedGenerationSection -> recordingGuidedPass (sink deltas). Emitted once on the response entry after the perform block via emitMetadata.
    - Multi-pass rule: counts summed; fallback reason = first fallback any pass reported, in pass order; speculativeDecoding = drafted > 0. Documented in README MTP section and the type doc.
    - MLXGuidedGeneration: sink gets `recordsDecoding` (init param, default true; GuidedDecoder ignores a sink that does not record decoding) and `speculationFallbacks: [SpeculativeDecodingFallbackReason]`; `recordSpeculationFallback` now takes `reason:`. An outer test sink stays bound and the outcome reads only the delta of the pass.
    - Discovery: existing tests compared `metadataKeys` of the MTP turn with the plain turn (MTPExecutorTests think-then-call, GuidedGenerationMTPTests x2); they now compare `outputMetadataKeys` (all keys minus the four outcome keys).
    - Test helpers de-duplicated: FixtureReasoningResolver.closingAtGreedyToken, Qwen35MTPFixture.makeIncompatibleSpeculativeDecoding, GuidedExecutorModels(resolver:speculativeDecoding:).
    - Process note: one edit (the lookupTool constant in SpeculativeDecodingOutcomeMetadataTests.swift) was made through a python one-liner in the shell instead of the files tool; all other edits used the files tool.
    Tests: SpeculativeDecodingOutcomeTests (9) + SpeculativeDecodingOutcomeMetadataTests (10, session transcript metadata for text/streamed text/reasoning/allowed tools/incompatible/no-MTP; executor metadata for schema, required tool, think-then-call sum, incompatible schema). RED: 10 tests 29 issues before the change; GREEN after. All 5 bundles: MLXLMTests 920 XCTest + 2035 ST, MLXGuidedGenerationTests 72, MLXFoundationModelsTests 446, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6 — 0 failures, 0 skipped. IntegrationTesting MTPGenerationTests now asserts the real-model response has speculativeDecoding == true and acceptedDraftTokenCount > 0 (Release run pending, done with task ^jhgc7r4).
  timestamp: 2026-10-29T15:02:30.198238+00:00
- actor: claude-code
  id: 01m645hmzdamqx25eyavgq9jhk
  text: |-
    ### implement — changed
    - evidence: commit 6354610a on worktree-agent-a96eeb70bc03d7a68; files: Libraries/MLXFoundationModels/{SpeculativeDecodingOutcome.swift (new), MLXLanguageModel.swift, README.md}, Libraries/MLXGuidedGeneration/{GuidedGenerationDiagnosticSink.swift, GuidedDecoder.swift, GuidedSpeculation.swift}, Tests/MLXFoundationModelsTests/{SpeculativeDecodingOutcomeTests.swift (new), SpeculativeDecodingOutcomeMetadataTests.swift (new), MTPExecutorTestSupport.swift, MTPExecutorTests.swift, GuidedGenerationMTPTests.swift, GuidedGenerationMTPTestSupport.swift, MTPDrafterFallbackExecutorTests.swift, Qwen35MTPFixture+LanguageModel.swift}, IntegrationTesting/.../MTPGenerationTests.swift. swift build --build-tests: 0 compiler warnings; 5 bundles 0 failures 0 skipped; verify-docs exit 0; IntegrationTesting Release build-for-testing TEST BUILD SUCCEEDED.
    - next: real-model Release run of MTPGenerationTests + MTPPromptCacheTests (asserts speculativeDecoding == true, acceptedDraftTokenCount > 0) waits for coordinator "GPU go"; then /review.
  timestamp: 2026-10-29T15:48:05.229305+00:00
position_column: doing
position_ordinal: '80'
title: 'FoundationModels: report each MLXLanguageModel call''s MTP outcome in Transcript.Response metadata'
---
Requested by the Router session (foundationmodelsrouter-ce): Router turns MTP on (Qwen3.8-27B-4bit + Qwen3.8-27B-MTP-4bit) for SWE-bench and its telemetry must prove MTP ran. On stable 148f227e the MTP outcome (GenerateCompletionInfo.speculativeDecodingFallbackReason, accepted/drafted tokens) is only visible through the internal test-only `generationObserver`. Router already reads `Transcript.Response.metadata["incompleteOutput"]` emitted through `emitMetadata`; use the same channel.

On each response entry of a call made with `speculativeDecoding != nil`, emit metadata:
- [x] `"speculativeDecoding"`: Bool — true when MTP drafted in at least one round of the call's passes.
- [x] `"speculativeDecodingFallbackReason"`: String — the `SpeculativeDecodingFallbackReason` case name when a pass fell back to plain decode; key absent when there was no fallback.
- [x] `"acceptedDraftTokenCount"`: Int and `"draftedTokenCount"`: Int — summed over all passes of the call (think-then-call phases, tool turns, guided generation).
- [x] A call with `speculativeDecoding == nil` emits none of these keys.
- [x] Covers text, reasoning, allowed/required tools, guided generation, streaming and non-streaming paths.
- [x] Tests in a new file: each key present/absent as above, sums across passes, fallback case (incompatible drafter), no-MTP call emits none.
- [x] Document the keys in Libraries/MLXFoundationModels/README.md (MTP section).

#mtp #foundation-models