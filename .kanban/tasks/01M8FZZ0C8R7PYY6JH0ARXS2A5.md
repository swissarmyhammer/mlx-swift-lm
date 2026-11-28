---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g03829sthdqkhehxetj8ya
  text: |-
    Research:
    - Enum: Libraries/MLXLMCommon/SpeculativeDecoding.swift. No exhaustive `switch` over SpeculativeDecodingFallbackReason exists anywhere in Libraries/Tests/IntegrationTesting (EmbeddingGemma2's `.unsupportedMedia` is another enum); consumers use `String(describing:)` (FM metadata, MTPChatBenchmark) or `==`.
    - Guided rejection site: GuidedSpeculation.drafter(of:for:) guard `as? any GuidedMTPDrafter, !requiresSharedTargetKV` -> currently `.incompatibleDrafter` with "cannot draft through a grammar mask". Same code path serves local and cluster (RemoteMTPDrafter passes isCompatible, fails the cast).
    - Reason publishers: (1) SpeculativeDecodingOutcome.metadata (FM response metadata key `speculativeDecodingFallbackReason`), fed by loop passes (GenerateCompletionInfo.speculativeDecodingFallbackReason + passthroughReason already carries the stated text), guided passes (GuidedGenerationDiagnosticSink.speculationFallbacks, stated text only as last-wins `speculationFallbackReason`), and call-level record(fallbackReason:) in MLXLanguageModel (cluster rank 0 without drafter -> incompatibleDrafter; memory policy -> memoryBudgetExceeded) which carry no stated text today. (2) GenerateCompletionInfo / BenchmarkHelpers MTPChatBenchmark already publish passthroughReason beside the reason.
    - Existing test GuidedGenerationMTPRouterPathTests.aDrafterWithoutGuidedDraftsRecordsItsFallback expects `.incompatibleDrafter` for StatelessMTPDrafter; it changes to `.unsupportedGuidance` per the card.
  timestamp: 2026-11-28T02:35:07.209808+00:00
- actor: claude-code
  id: 01m8g100x3a0fxwvpyhbk62c1j
  text: |-
    Implementation landed (not committed).

    PUBLIC NAMES for the FoundationModelsRouter consumer:
    - enum case `SpeculativeDecodingFallbackReason.unsupportedGuidance` -> metadata value string "unsupportedGuidance" under key "speculativeDecodingFallbackReason".
    - new response-metadata key "speculativeDecodingFallbackDetail" (String; `SpeculativeDecodingOutcome.MetadataKey.fallbackDetail`): the first fallback's stated reason, emitted beside speculativeDecodingFallbackReason; absent when no fallback or none stated. For the bench case it reads "the MTP drafter RemoteMTPDrafter cannot draft through a grammar mask, so the generation decodes without speculation".
    - `.incompatibleDrafter` unchanged in meaning (no usable drafter).

    Changes:
    - GuidedSpeculation.drafter(of:for:): the guided-conformance rejection reports `.unsupportedGuidance` (local and cluster share this path).
    - GuidedGenerationDiagnosticSink: new `package private(set) var speculationFallbackRecords: [MTPDrafterFallback]`; `speculationFallbacks` and `speculationFallbackReason` are now computed from it (same public API).
    - SpeculativeDecodingOutcome: tally keeps a Fallback(reason, detail); loop passes take detail from GenerateCompletionInfo.passthroughReason, guided passes from the sink record; `record(fallbackReason:detail:)` now requires a detail. MLXLanguageModel states details for cluster-rank-0-without-drafter and memory-budget fallbacks.
    - Telemetry: GenerateCompletionInfo.passthroughReason and BenchmarkHelpers MTPChatBenchmark already carry the stated reason beside the reason; no change needed there. No exhaustive switch over the enum exists anywhere.

    TDD: 7 new or changed tests failed RED as expected; the plain-after-schema guard test passed in RED (it guards existing behavior). After the change all of them pass.
  timestamp: 2026-11-28T02:50:50.147738+00:00
- actor: claude-code
  id: 01m8g104yvgy678av2p1fy99a3
  text: |-
    ### implement — changed
    - evidence: 9 files — Libraries/MLXLMCommon/SpeculativeDecoding.swift, Libraries/MLXGuidedGeneration/GuidedSpeculation.swift, Libraries/MLXGuidedGeneration/GuidedGenerationDiagnosticSink.swift, Libraries/MLXFoundationModels/SpeculativeDecodingOutcome.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXFoundationModelsTests/{ClusterExecutorTests,SpeculativeDecodingOutcomeMetadataTests,GuidedGenerationMTPRouterPathTests,MTPExecutorTestSupport}.swift. xctest: MLXFoundationModelsTests 663 pass, MLXGuidedGenerationTests 96 pass, CXGrammarTests 7 pass, MLXHuggingFaceMacrosTests 11 pass, MLXLMTests 1261 XCTest + 2845 Swift Testing pass (3 pre-existing known issues). swift-format applied.
    - public names: `.unsupportedGuidance`, metadata key `speculativeDecodingFallbackDetail`
    - next: review
  timestamp: 2026-11-28T02:50:54.299709+00:00
- actor: claude-code
  id: 01m8g1hass2qz4zrxs1mxg2h51
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (no warnings from sources); xcrun xctest per bundle, all exit 0: MLXLMTests 2845 swift-testing tests/281 suites passed + 1261 XCTest, 0 failures; MLXGuidedGenerationTests 96 tests/23 suites passed; MLXFoundationModelsTests 663 tests/88 suites passed; CXGrammarTests 7 tests/5 suites passed; MLXHuggingFaceMacrosTests 21 XCTest, 0 failures. swift-format lint on all 9 changed files clean. 0 skipped.
    - pre-existing: MLXLMTests reports "3 known issues", all from the committed withKnownIssue block in Tests/MLXLMTests/ToolTests.swift (MiniCPM5 CDATA close tag), a file this change does not touch.
    - next: review/commit
  timestamp: 2026-11-28T03:00:17.337368+00:00
position_column: doing
position_ordinal: '80'
title: 'Speculative decoding: a guided call whose drafter can''t draft through a grammar mask reports its own fallback reason, with the stated reason in the response metadata'
---
## What
On bench cluster4i (Inkling with MTP on 2 Macs, stable b330697f), 66 of 452 calls fell back with `incompatibleDrafter`. All 66 were guided calls: `respond(to:generating: Selection.self)` from FoundationModelsRanker's SelectionTier. GuidedSpeculation.drafter(of:for:) (Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:95) requires `GuidedMTPDrafter = ResumableMTPDrafterModel & StatefulMTPDistributionDrafterModel`. RemoteMTPDrafter conforms to neither, so the call reports `.incompatibleDrafter` with "cannot draft through a grammar mask". `incompatibleDrafter` reads as "this model has no usable drafter", yet a plain call on the same model speculates (61% acceptance, 94.9 tok/s). Reported by foundationmodelsrouter-22; Router card ^j7thp86 records the reason on its span.

## Fix
- Add a new `SpeculativeDecodingFallbackReason` case, e.g. `.unsupportedGuidance`, for "the model has a drafter, but it can't draft under a grammar mask". Use it wherever the guided path rejects a drafter for lacking the guided conformances, both local and cluster. `incompatibleDrafter` keeps its meaning: no usable drafter at all.
- Expose the stated reason text (the "cannot draft through a grammar mask" message) in the response metadata next to `speculativeDecodingFallbackReason`, e.g. `speculativeDecodingFallbackDetail: String?`, wherever the fallback reason is already exposed. Find every place that publishes the reason (MLXFoundationModels response metadata, telemetry) and add the detail beside it.

## Tests
- A guided (schema) turn on a clustered model with a RemoteMTPDrafter reports `.unsupportedGuidance` and the detail text; a plain turn on the same model still speculates.
- A guided turn on a local model whose drafter lacks the guided conformances also reports `.unsupportedGuidance`.
- A model with no drafter still reports its existing reason.
#cluster