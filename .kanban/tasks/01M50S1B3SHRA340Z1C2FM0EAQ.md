---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m51r89s9gmm0d76srdy71ytz
  text: |-
    Research done. Design: move `ThinkingRender` (now nested in `MLXLanguageModel.Executor`, MLXFoundationModels) to MLXLMCommon as a public type, so that MLXFoundationModels and ChatSession call one function. ChatSession gets `public var thinkingEnabled: Bool?` (nil = the template default, no change for old callers). Precedence: when the switch is set, its template key replaces the same key of `additionalContext`. The prompt render and the checkpoint probe (`PromptCacheCheckpoint.lastMessageEnd`) use one render closure, thus the probe sees the same render as the prompt. A strategy that cannot turn thinking off (`alwaysOn`, `none`) throws `ReasoningError.cannotDisableReasoning` for the switch off, as the shared function does.

    TDD RED (stub property only, no behavior): `swift build --build-tests` 0 warnings; `xcrun xctest MLXLMTests.xctest` 1513 Swift Testing tests, 12 issues, all in `ChatSessionReasoningHistoryTests` and all in new rows: session switch off (history and reuse), the hybrid row with the switch off, switch rows (Qwen 3.5 off, tagged off, switch wins over flag), and the alwaysOn error test. The old rows and the control rows pass.
  timestamp: 2026-10-16T07:01:36.681672+00:00
- actor: claude-code
  id: 01m51rxxa3jr0ast0j1n36ncrm
  text: |-
    GREEN. New `Libraries/MLXLMCommon/ThinkingRender.swift` (public `ThinkingRender`, moved from MLXFoundationModels; `init(thinkingSwitch:config:)`, `templateContext(over:)`, `prepare(_:processor:tokenizer:)`; the closed prompt now also keeps `audio`). `Libraries/MLXFoundationModels/ThinkingRender.swift` is now an extension with `prepare(messages:tools:context:)` and `transcriptBoundary`, which call the shared function. The batched token rank constant moved to `LMInput.Text.batchedTokenRank` (the executor constant `batchedPromptRank` is gone). ChatSession: `public var thinkingEnabled: Bool?`, one `render` closure for the prompt and the checkpoint probe.

    Test note: with the switch off, the prompt is not the template text (the closed block replaces the open block). The hybrid row first compared the cache with the template text and failed for that reason; the test now builds the cold prompt from the messages of the cold render plus the generation prompt of the row (`promptThinks` column).

    Results: `swift build --build-tests` 0 warnings (only the known "missing creator" note). MLXLMTests: 721 XCTest + 1513 Swift Testing, 0 failures. MLXFoundationModelsTests: 368 Swift Testing, 0 failures (includes the Qwen 3.5 thinking-off executor suite).

    Mutant (one run): `templateContext(over:)` merged with the caller key winning. Result: row "the switch on and the template flag off: the switch wins" failed (2 issues), all else pass. Code restored.
  timestamp: 2026-10-16T07:13:24.803161+00:00
- actor: claude-code
  id: 01m51swcnpb0jgj0z4wj6r4r16
  text: |-
    Real-weights results (Qwen3.8-27B-mxfp4, idle GPU checked before each GPU job, one job at a time):

    - `xcodebuild test -only-testing:` Qwen35SessionPromptCacheTests, Qwen35CheckpointIntegrationTests, Qwen35ThinkingOffTemplateTests: 14 tests in 3 suites pass, TEST SUCCEEDED. New test `aChatSessionTurnWithTheThinkingSwitchOffAndOnAgainReusesTheTurnBefore` (ChatSession, switch on, off, on): turn 2 cached 101 of 128 (turn 1 rendered 73), turn 3 cached 131 of 154 (turn 2 rendered 128). These are the same numbers as the framework executor test with `reasoningLevel` no_think, thus the two paths render the same prompts. The integration target has 25 old warnings in other files (`no calls to throwing functions occur within 'try'`, GuidedGeneration tests); none is in a changed file.
    - qwen38bench (now uses `ChatSession.thinkingEnabled` and `ThinkingRender` for the reference renders; `QWEN38_THINKING_KEY` is gone), full mode, 32k, 48 decode tokens, binary scratchpad/qwen38bench/new-2fm0eaq, log scratchpad/qwen38/2fm-new-48-1.txt:
      - thinking off: turn 1 prefilled 32196 in 80.18 s; turn 2 warm prefilled 20, cached 32231, prompt 0.33 s, wall 0.587 s; turn 3 0.33 s; reuse YES, tie-aware warm == cold PASS, HARD reuse check PASS.
      - thinking on: turn 2 warm prefilled 22, cached 32242, prompt 0.26 s; turn 3 0.26 s; reuse YES, tie-aware PASS, HARD reuse check PASS.
      - Same as ^ef2zynk (off 0.34 s, on 0.26 s). With the switch off, the turn-1 prompt is 32196 tokens (the flag stays on, so the template keeps its thinking text), where the old flag-off render was 32154.

    Other changes for reuse, not copies: `PassOutcome` and `collect(_:)` in DeepseekV4AgenticPromptCacheAssessmentTests.swift are now internal, so the new Qwen 3.8 test uses them. `expectTheTurnReusesTheTurnBefore` in Qwen35SessionPromptCacheTests now has a numeric form; the `Usage` form forwards to it.

    Not done here (on purpose): DeepSeek-V4 has no reasoning config (^ggn6kfj), thus the switch does nothing for it now. The doc of `thinkingEnabled` states this.

    ARCHITECTURE.md does not exist; no update.
  timestamp: 2026-10-16T07:30:03.574672+00:00
- actor: claude-code
  id: 01m51swgmzjsjeymfxbxf6eh17
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/ThinkingRender.swift (new), Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXFoundationModels/ThinkingRender.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Tests/MLXLMTests/ChatSessionReasoningHistoryTests.swift, IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift, plus the card. `swift build --build-tests`: 0 warnings (only the known "missing creator" note). MLXLMTests: 721 XCTest + 1513 Swift Testing, 0 failures. MLXFoundationModelsTests: 368, 0 failures. TDD: RED 12 issues in the new rows only; one mutant (caller key wins over the switch) killed by the precedence row. Integration (real weights): 14/14 pass. Bench 32k: reuse PASS for thinking off and on, turn 2 prompt 0.33 s / 0.26 s. swift-format lint clean. Not committed.
    - next: /review. Task stays in doing.
  timestamp: 2026-10-16T07:30:07.647904+00:00
- actor: claude-code
  id: 01m51t2xkres98eww6hs7v9f6y
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (commit 62ec866). 7 files reviewed. 0 findings, 0 confirmed, 6 refuted. The two .kanban files had no validator.
    - next: none. The task moved to done.
  timestamp: 2026-10-16T07:33:37.528434+00:00
- actor: claude-code
  id: 01m51t38gemex7zzsdymqd1est
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 7 source files; ChatSession.thinkingEnabled, shared ThinkingRender in MLXLMCommon
    - test: green — run on the final tree by implement: MLXLMTests 721 + 1513, MLXFoundationModelsTests 368, real-weights 14; 0 compiler warnings
    - commit: 62ec866
    - review: clean — 0 findings (6 refuted); task moved to done
  timestamp: 2026-10-16T07:33:48.686912+00:00
position_column: done
position_ordinal: ffde80
title: 'ChatSession: no model-independent thinking switch for MLXLMCommon callers'
---
## What

Found in ^yp1jxca. `ChatSession` and `UserInput` have no thinking switch. A caller must know the template key of each model (`enable_thinking` for Qwen, `thinking` for DeepSeek-V4). The qwen38bench used the DeepSeek-V4 key on Qwen3.8, and thinking stayed on. `ReasoningConfig.promptStrategy` is the shared switch, but only `MLXLanguageModel` (MLXFoundationModels) applies it. For Qwen 3.5 the switch keeps `enable_thinking` true and needs `closingReasoning(in:with:tokenizer:)` after the render; `ChatSession` does not do that step.

## Decision for the user

Must the MLXLMCommon path get one thinking switch (for example on `ChatSession` or `UserInput`) that applies `reasoningConfig.promptStrategy`, closed block included? Or do callers of that path keep the model keys?

## Decision

The user decided: yes. Add one shared, model-independent thinking switch to the MLXLMCommon path. The switch applies `reasoningConfig.promptStrategy`. MLXFoundationModels and MLXLMCommon call one shared function for this step. Do not copy the logic.

## Acceptance Criteria

- [x] The MLXLMCommon path has one model-independent thinking switch. When the switch is not set, the template default does not change.
- [x] The switch applies `reasoningConfig.promptStrategy`. When the strategy needs it, the switch adds the closed reasoning block after the render (`closingReasoning(in:with:tokenizer:)`).
- [x] The doc of the switch tells that the switch applies only when the model declares a reasoning config.
- [x] Tests for Qwen 3.5 and Qwen 3.8 with thinking on and with thinking off.
- [x] The ChatSession prompt cache reuse continues to work with thinking on and with thinking off.
- [x] All tests pass with 0 warnings.

#qwen3-8