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
- actor: claude-code
  id: 01m645y5wnevn15vszwy6avsc1
  text: |-
    ### review — findings
    - evidence: review sha d6bdd4a2~1..d6bdd4a2 — 3 findings (3 confirmed, 0 refuted) — IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGenerationTests.swift:121, Libraries/MLXFoundationModels/MLXLanguageModel.swift:1296, Tests/MLXFoundationModelsTests/SpeculativeDecodingOutcomeMetadataTests.swift:150
    - next: implement the 3 open items in the 2026-10-29 10:49 Review Findings section
  timestamp: 2026-10-29T15:54:55.765733+00:00
- actor: claude-code
  id: 01m645ywrdqnvtx83qkq0zkytb
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — MTP outcome metadata (worktree 6354610a, applied to mtp)
    - test: green — 5 bundles 0 fail/0 skip, verify-docs exit 0, Release build-for-testing ok; real-model check pending GPU slot
    - commit: d6bdd4a2
    - review: findings — MTPGenerationTests.swift:121, MLXLanguageModel.swift:1296, SpeculativeDecodingOutcomeMetadataTests.swift:150
  timestamp: 2026-10-29T15:55:19.181414+00:00
- actor: claude-code
  id: 01m647gft5yhx893h6qbnfve5e
  text: |-
    Review findings (2026-10-29 10:49) addressed on worktree-agent-ad8b635eacd770cf2 (reset to mtp 15260451), commit 09f2fd91.
    - MLXLanguageModel.runRespond: emitSpeculativeDecodingOutcome moved out of the success path to after the do/catch (after the prompt-cache check-in), so the response entry, which the do block opens with its modelID metadata, gets the four keys on success, CancellationError and every other throw. Key names are unchanged.
    - Discovery: a generation-loop pass that is cancelled mid-stream throws at Task.checkCancellation before its .info, so it records nothing of its own. A guided pass still records the rounds it ran before the throw (the defer in recordGuidedPass). This is now written down in the SpeculativeDecodingOutcome doc and the README MTP section.
    - Dead end: GuidedGenerationDiagnosticSink(cancelAfterEmitCount:) cancels the respond task from inside the guided section, so the cancel handler waits on its own section and stalls for the 60 s quiescence timeout. It is not used here.
    - Tests: aCancelledPassReportsItsDrafts runs think-then-call and the observer cancels the pass task at phase 1's .completion; the response reports phase 1's drafts (4 drafted / 2 accepted in RED output). aFailedPassReportsItsOutcome uses a .alwaysOnThinking resolver without .reasoning, so the call throws unsupportedCapability inside the do block; the response reports false/0/0 with no fallback. MTPExecutorTurns.respond(to:model:cancellingAt:) now runs the pass on its own Task, which inherits the task-locals. Added FixtureReasoningResolver(reasoningConfig:) and GuidedExecutorModels.respondSpeculatively(to:cancellingAt:).
    - Shared helper: Transcript.lastResponse (public extension, guarded by the trait and SDK) in Libraries/IntegrationTestHelpers/Transcript+LastResponse.swift. MLXFoundationModelsTests now depends on IntegrationTestHelpers (Package.swift), following the BenchmarkHelpers pattern for MLXLMTests. Both the unit test and MTPGenerationTests use it.
    - MTPGenerationTests: draftedTokenCount and acceptedDraftTokenCount are compared with completion.speculativeDecodingTelemetry, and the test expects speculativeDecodingFallbackReason to be absent. It was not run (no GPU).
  timestamp: 2026-10-29T16:22:24.325506+00:00
- actor: claude-code
  id: 01m647gq3yce8s8jjazkt5dx0h
  text: |-
    ### implement — changed
    - evidence: branch worktree-agent-ad8b635eacd770cf2 (on mtp 15260451), commit 09f2fd91. 9 files: Libraries/MLXFoundationModels/{MLXLanguageModel.swift, SpeculativeDecodingOutcome.swift, README.md}, Libraries/IntegrationTestHelpers/Transcript+LastResponse.swift (new), Package.swift, Tests/MLXFoundationModelsTests/{SpeculativeDecodingOutcomeMetadataTests.swift, MTPExecutorTestSupport.swift, GuidedGenerationMTPTestSupport.swift}, IntegrationTesting/.../MTPGenerationTests.swift. RED: 2 new tests failed with 7 issues (keys absent); GREEN after. swift build --build-tests: 0 Swift warnings (only the SwiftPM build-system line "missing creator for mutated node", which a no-op build also prints). xctest: MLXFoundationModelsTests 454 ST, MLXGuidedGenerationTests 72, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6, MLXLMTests 920 XCTest + 2035 ST; 0 failures, 0 skipped. IntegrationTesting Release build-for-testing: TEST BUILD SUCCEEDED. swift-format lint clean. All 3 review findings checked.
    - next: /review; the real-model MTPGenerationTests run is still waiting for a GPU slot.
  timestamp: 2026-10-29T16:22:31.806007+00:00
- actor: claude-code
  id: 01m647mg9safqymb6z59bzhcnc
  text: |-
    ### review — clean
    - evidence: review sha d3086858~1..d3086858 — 0 findings (0 confirmed, 0 refuted, 7 attempted, 0 failed); all 3 items of `## Review Findings (2026-10-29 10:49)` checked
    - next: none — task moved to done
  timestamp: 2026-10-29T16:24:35.897022+00:00
position_column: done
position_ordinal: ffff8d80
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

## Review Findings (2026-10-29 10:49)

> Scope: `review sha d6bdd4a2~1..d6bdd4a2` — reviewed the diffs only — lines this change added or modified. 14 file(s) reviewed, 1 not reviewed.

> 1 file(s) not reviewed — no validator matched:
> - `Libraries/MLXFoundationModels/README.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGenerationTests.swift:121` `completeness/invariant-propagation` — The real-model test checks the reported `acceptedDraftTokenCount` against zero but never checks `draftedTokenCount` against the completion's own count, nor the fallback reason. The same invariant the metadata reports (the drafts the call proposed) is therefore unproven on the real model. A regression that reports the wrong draft total, for example summing only one of the passes, would pass this test. Add `#expect(try mtp.metadata["draftedTokenCount"]?.value(Int.self) == mtp.completion.proposedDraftTokens)` and the matching assertion for `acceptedDraftTokenCount` against `mtp.completion.acceptedDraftTokens`, so the metadata is tied to the completion source on the real model.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1296` `completeness/public-output-contract` — The MTP outcome is emitted only on the success path, after the `do` block completes. When a pass throws, the `CancellationError` and generic `catch` branches set `outcome` without calling `emitSpeculativeDecodingOutcome`. A call of a model with a drafter that drafted and then failed or was cancelled therefore writes no `speculativeDecoding`, `draftedTokenCount` or `acceptedDraftTokenCount` into its response entry, even though the drafts were proposed and partly accepted. The intended side-effect (reporting the MTP outcome of the call) is silently dropped on the error path. Emit the outcome on every exit of the pass that the response entry is written for, for example in a `defer`-like step after the `do/catch`, or call `emitSpeculativeDecodingOutcome` in the failure branches too. Add one test where a speculating call is cancelled or throws mid-generation and assert what the response entry reports, or document that failed calls report nothing.
- [x] `Tests/MLXFoundationModelsTests/SpeculativeDecodingOutcomeMetadataTests.swift:150` `reuse/reuse` — The helper re-implements the "last response entry of the session transcript" extraction inline. The same compactMap-then-last pattern is written again in the integration test's `MTPGenerationTests.swift` (lines 89-92), so the transcript-reading logic has two copies that can drift apart. Extract it once beside `MTPTurnLog` in `MTPExecutorTestSupport.swift` and call it from both places. Add a shared helper, for example `func lastResponse(of transcript: Transcript) -> Transcript.Response?`, in `MTPExecutorTestSupport.swift`, and call it from `respond` here. The integration target cannot import that test support file, so either accept the copy there or move the helper into a module both targets can import.
