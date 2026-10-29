---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6313rzccvcbgaa5fwwx860e
  text: 'Also covers the guided-generation path (folded in from duplicate ^8hjz9t1): in ^9xe94tr the Release guided JSON benchmark drafted 0 tokens when the target loaded through loadModelContainer as a VLM with `.qwen35Text`. The fallback report must cover GuidedGenerationLoop / GuidedSpeculation as well as ChatSession and MLXLanguageModel.'
  timestamp: 2026-10-29T05:11:21.836590+00:00
- actor: claude-code
  id: 01m63cxgh1ym03x9fs6bdxdfv3
  text: 'Research: ChatSession already reports `.incompatibleDrafter` in `GenerateCompletionInfo.speculativeDecodingFallbackReason` (test `incompatibleDrafterFallsBackToPlainDecode`), but logs nothing and gives no stated reason. `keptPrefixWindowReason` (removed in ^8f954cv) was a `passthroughReason` string on `MTPSpeculativeTokenIterator`, so "the way it was exposed" = `GenerateCompletionInfo.passthroughReason`. Silent paths: (1) `MTPSpeculation.makeTokenIterator` (all MLXLanguageModel text/protocol/tokens paths via generateTaskRecordingTokens / generateProtocolTokensTask / generateTokensTask) returns a plain `TokenIterator` with no reason; `TextToolTokenLoopHandler.infoEvent` also overwrites `speculativeDecodingFallbackReason` with its own (nil outside ChatSession). (2) `GuidedSpeculation.drafter(of:for:)` returns nil on `!isCompatible`; `GuidedGenerationLoop.run` returns only a token count, so the only observers are logs and `GuidedGenerationDiagnosticSink`. Registration: `.qwen35Text` (MLXLLM, factory .shared) and `.qwen35VLM` (MLXVLM, factory .visionLanguage) live in modules that do not depend on each other, so no library can ship one combined static; the target model is in scope at drafter-load time in ChatSession (inside the turn) and MLXLanguageModel (container loads before `loadMTPSpeculation`). Plan: shared once-per-pair reporter in MLXLMCommon; TokenIterator carries the fallback so generateLoopTask fills reason + passthroughReason; ChatSession sets the same; guided logs + records on the sink; `MTPDrafterRegistration` gains a target predicate and `.matchingTarget(...)` combinator that ChatSession/MLXLanguageModel resolve against the loaded target before the load.'
  timestamp: 2026-10-29T08:37:39.489995+00:00
- actor: claude-code
  id: 01m63eqd2wtk9m4e4xwvtd9962
  text: |-
    Implementation landed (uncommitted).
    - New `Libraries/MLXLMCommon/MTPDrafterFallback.swift`: `MTPDrafterFallback.incompatibility(of:with:)` returns `.incompatibleDrafter` plus a stated reason ("the MTP drafter X cannot draft for the target Y, so the generation decodes without speculation: <validateCompatibility error>") and reports it to `MTPDrafterFallbackLog.shared`, which writes one `mtp-fallback` warning per distinct stated reason (= one per drafter/target pair).
    - `TokenIterator.speculationFallback` (package): `MTPSpeculation.makeTokenIterator` and ChatSession's plain iterator carry it; `generateLoopTask` puts `reason` into `speculativeDecodingFallbackReason` and the stated reason into `passthroughReason` (where `keptPrefixWindowReason` used to surface). `TextToolTokenLoopHandler.infoEvent` no longer clobbers the iterator's reason with its own nil. Covers every MLXLanguageModel text/protocol/tokens path and ChatSession.
    - Guided: `GuidedSpeculation.drafter(of:for:)` reports the fallback and records it on the new `GuidedGenerationDiagnosticSink.speculationFallbackReason`.
    - Registration (subtask 3, done rather than only considered): `MTPDrafterRegistration` gains `draftsForTarget:` (init param), `drafts(for:)`, `registration(for:)` and `static matchingTarget(_:_:)`. `.qwen35Text` drafts for `Qwen35Model`/`Qwen35TextModel`, `.qwen35VLM` for MLXVLM `Qwen35`. MLXLLM and MLXVLM do not depend on each other, so no single `.qwen35` static is possible; callers write `.matchingTarget(.qwen35Text, .qwen35VLM)`. ChatSession: new `DrafterSource.deferredMatchingTarget` case, `load()` replaced by `loader(for: target)` (and `loadDraftModel()` by `draftModelLoader(for:)`); the registration init resolves against the session's model in the turn. MLXLanguageModel: `MLXSpeculativeDecoding.drafterLoader(for:)`, `loadMTPSpeculation(for: container)` picks the loader inside `container.perform`.
    - Test support: `UnusedDownloader` moved from ChatSessionDeferredMTPDrafterTests to MLXLMTestSupport (public) for reuse; `GuidedFixtureRun.generate` takes an optional `drafter:`.
    - Left IntegrationTesting call sites on `.qwen35Text` with `LLMModelFactory` (already correct since ^h91ttb2/^9xe94tr).
    TDD: new tests failed to compile (missing API) before, pass after: Tests/MLXLMTests/MTPDrafterFallbackTests.swift (9), Tests/MLXFoundationModelsTests/MTPDrafterFallbackExecutorTests.swift (3, text turn + guided + registration).
  timestamp: 2026-10-29T09:09:16.508654+00:00
- actor: claude-code
  id: 01m63eqp320k3tb10atq71k059
  text: |-
    Verification. Build: `swift build --build-tests` 0 compiler warnings (only SwiftPM's pre-existing "missing creator for mutated node" note for the Cmlx bundle). Full unit run with `xcrun xctest`: MLXLMTests 2010 swift-testing + 920 XCTest, MLXFoundationModelsTests 407, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6 — 0 failures, 0 skipped. Release guards (xcodebuild Release, `-destination 'platform=macOS,arch=arm64' ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES`, load ~3.4 before, 3.03 after): qwen38FourBitMTPGeneration pass, default blockSize 5 pMin 0.75 math/code/prose 100.3/87.5/83.4 tok/s (baseline 100.4/87.4/83.4), plain 59.4-59.5; qwen38FourBitMTPSystemPromptCache pass, edit turn prefilled 123 tokens, no known issues, no `mtp-fallback` line in the log.

    ### implement — changed
    - evidence: 20 files — new Libraries/MLXLMCommon/MTPDrafterFallback.swift, Tests/MLXLMTests/MTPDrafterFallbackTests.swift, Tests/MLXFoundationModelsTests/MTPDrafterFallbackExecutorTests.swift; modified Libraries/MLXLMCommon/{ChatSession,Evaluate,MTPDrafterModelFactory,MTPSpeculation,TokenLoopHandler}.swift, Libraries/MLXLLM/Qwen35TextMTPRegistration.swift, Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift, Libraries/MLXFoundationModels/{MLXLanguageModel,MLXSpeculativeDecoding}.swift, Libraries/MLXGuidedGeneration/{GuidedSpeculation,GuidedGenerationDiagnosticSink}.swift, Tests/MLXLMTestSupport/Qwen35MTPCheckpointFixture.swift, Tests/MLXLMTests/{ChatSessionDeferredMTPDrafterTests,SpeculativeDecodingConfigTests}.swift, Tests/MLXFoundationModelsTests/GuidedGenerationMTPTestSupport.swift; 5 bundles 0 failures 0 skipped; Release guards within 3%.
    - next: /review
  timestamp: 2026-10-29T09:09:25.730015+00:00
- actor: claude-code
  id: 01m63f1kv5fsxn2xzbmp2fbdjj
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (c23c785f) — 1 finding (1 confirmed, 5 refuted, 21 attempted): Libraries/MLXFoundationModels/MLXSpeculativeDecoding.swift:119
    - next: implement the finding (share one loader-building helper on MTPDrafterRegistration used by both ChatSession and MLXLanguageModel), then re-review
  timestamp: 2026-10-29T09:14:51.109120+00:00
- actor: claude-code
  id: 01m63f21zvtg0fqzdav15tn53k
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 20 files (fallback reason + log, matchingTarget registration)
    - test: green — 5 bundles 0 failures 0 skipped; Release 100.3/87.5/83.4 vs plain 59.5; edit turn 123 tokens
    - commit: c23c785f
    - review: findings — Libraries/MLXFoundationModels/MLXSpeculativeDecoding.swift:119
  timestamp: 2026-10-29T09:15:05.595992+00:00
- actor: claude-code
  id: 01m63fha4m1sfqafkcmg8z34yr
  text: |-
    Finding fix (reuse/reuse at MLXSpeculativeDecoding.swift:119). Added `MTPDrafterRegistration.drafterLoader(for:from:using:)` (public, MLXLMCommon/MTPDrafterModelFactory.swift): picks the family with `registration(for: target)` and returns a `@Sendable (ModelConfiguration) async throws -> MTPDrafterContainer` that runs that family's `loadContainer`. `MLXSpeculativeDecoding`'s registration init now passes `{ target in registration.drafterLoader(for: target, from:, using:) }`; `SpeculativeDecodingConfig`'s matching-target init calls the same helper and only binds its `configuration` (`return { try await load(configuration) }`), since ChatSession's `DrafterLoader` takes no argument. Both inline pick-and-load copies are gone. TDD: new parameterized test `matchingRegistrationLoaderLoadsTheDrafterOfItsTarget` in Tests/MLXLMTests/MTPDrafterFallbackTests.swift failed to compile (no member `drafterLoader`) before, passes after. It shares a private `expectLoadsTheDrafterOfItsTarget(vision:load:)` helper with the existing ChatSession-config test, so the two tests do not copy the checkpoint/target/class-check block. No decode-path change, so no Release rerun.

    Flipping the finding checkbox via `update task` returned an output too large to show (82 KB board dump), so check the description: the finding should read `- [x]`.
  timestamp: 2026-10-29T09:23:25.460634+00:00
- actor: claude-code
  id: 01m63fhjm4vvczpaexbnnqtthr
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift (new `MTPDrafterRegistration.drafterLoader(for:from:using:)`), Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXFoundationModels/MLXSpeculativeDecoding.swift, Tests/MLXLMTests/MTPDrafterFallbackTests.swift; `swift build --build-tests` 0 compiler warnings (only SwiftPM's existing Cmlx-bundle "missing creator" note); `xcrun xctest` MLXLMTests 2011 swift-testing + 920 XCTest, MLXFoundationModelsTests 407, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6: 0 failures, 0 skipped (MTPDrafterFallbackTests, MTPDrafterFallbackExecutorTests, ChatSessionDeferredMTPDrafterTests passed); finding flipped to [x]
    - next: /review
  timestamp: 2026-10-29T09:23:34.148845+00:00
- actor: claude-code
  id: 01m63fkvre955mj91sb3esvwxq
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (cd6eca89) — 0 findings (0 confirmed, 0 refuted, 7 attempted, 0 failed); 4 files reviewed, 2 kanban files skipped (no validator matched). Prior finding (MLXSpeculativeDecoding.swift:119 reuse/reuse) checked and addressed by the shared MTPDrafterRegistration.drafterLoader(for:from:using:).
    - next: task moved to done
  timestamp: 2026-10-29T09:24:49.038606+00:00
- actor: claude-code
  id: 01m63fm95f38ycdhhagvp1dnpc
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 files (shared MTPDrafterRegistration.drafterLoader)
    - test: green — 5 bundles 0 failures 0 skipped
    - commit: cd6eca89
    - review: clean — task moved to done
  timestamp: 2026-10-29T09:25:02.767565+00:00
position_column: done
position_ordinal: ffff8180
title: 'MTP: report when a configured drafter silently falls back to plain decode'
---
When `MLXLanguageModel` (and `ChatSession`) are given an MTP drafter that is not compatible with the loaded target (for example the generic `loadModelContainer` loads Qwen3.8 as a VLM while the drafter is the text drafter), generation quietly runs the plain iterator and proposes 0 drafts. Found during ^h91ttb2: the first integration run silently lost MTP.

The user priority is that MTP stays on and fast, so losing it must be visible.

- [x] Surface the fallback with a stated reason (log once per model/drafter pair, and expose it in completion info / speculative telemetry the way `keptPrefixWindowReason` is exposed)
- [x] Test: an incompatible drafter yields plain output plus the reported reason, in both ChatSession and MLXLanguageModel
- [x] Consider making the drafter registration pick the matching text/VLM drafter for the loaded target

#mtp #foundation-models

## Review Findings (2026-10-29 04:10)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 18 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M62QEHRC0M7NQDMWFPY115GV.jsonl` — no validator matches this file
> - `.kanban/tasks/01M62QEHRC0M7NQDMWFPY115GV.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/MLXSpeculativeDecoding.swift:119` `reuse/reuse` — The `pickDrafterLoader` closure reimplements the same registration-pick-and-load closure that `ChatSession`'s matching-target initializer builds. It is the second copy of that logic in this change. Share one loader-building helper on `MTPDrafterRegistration`, as in the ChatSession finding, and have this initializer call it instead of repeating the closure.
