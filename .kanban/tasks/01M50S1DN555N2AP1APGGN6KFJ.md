---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m51tb00zb4gc7m3deandng7m
  text: |-
    Research (implement step).

    Why DeepSeek-V4 has no reasoningConfig: commit 12223aa (prompt cache reuse on a DeepSeek-V4 tool round) wrote the comments in DeepSeekV4ChatEncoder.swift. They say that no decoder splits the reasoning out, thus ChatSession keeps the whole text (reasoning + `</think>` + answer) as the content, and `holdsItsOwnReasoning` writes no second `</think>`. DeepSeekV4.swift says the model controls thinking through the encoder mode and not through a delimiter pair.

    What a reasoningConfig changes, by consumer:
    - ChatSession: it reads `reasoningConfig` only in ThinkingRender (the switch) and in ReasoningHistory, which needs `replaysReasoningIntoHistory == true`. With the default `false`, ChatSession keeps the whole text as content. Thus the encoder rule `holdsItsOwnReasoning` and the DSML committed-turn splice do not change. DSMLCommittedTurnRule does not read reasoningConfig.
    - No other MLXLMCommon generate path reads `configuration.reasoningConfig`.
    - MLXFoundationModels: the executor now applies the prompt strategy (reasoningLevel, and thinking off when `.reasoning` is not declared), and it splits `<think>`/`</think>` text into reasoning entries. Before, a DeepSeek-V4 answer held the reasoning text and `</think>`. This is the same behavior as every other templateFlag model (Qwen). History drops the reasoning (no replay); the DSML splice keeps the live cache for the next round.
    - IntegrationTesting DeepSeek-V4 tests use ChatSession with `additionalContext["thinking"]` and no switch. That path does not change.

    Decision: add the row. DeepSeekV4Model declares a ReasoningConfig: `<think>`/`</think>`, `.templateFlag(key: "thinking", defaultOn: true)`, no history replay. The key and the default come from DeepSeekV4EncodingTokenizer, so the doc is true by construction.
  timestamp: 2026-10-16T07:38:02.143463+00:00
- actor: claude-code
  id: 01m51ty2ystfmkbedmb0r1vew3
  text: |-
    Implementation landed (the row is added; the doc and the code agree).

    Production:
    - Libraries/MLXLMCommon/Tokenizer.swift: `DeepSeekV4EncodingTokenizer.reasoningConfig` (public static). Delimiters `<think>`/`</think>` from DeepSeekV4ChatEncoder.SpecialToken. Prompt strategy `.templateFlag(key: thinkingKey, defaultOn: thinksByDefault)`. New private `thinksByDefault = true`, which `applyChatTemplate` also reads. Thus the key and the default have one source. `replaysReasoningIntoHistory` stays false. The doc of `thinkingKey` now names this config.
    - Libraries/MLXLLM/Models/DeepSeekV4.swift: `DeepSeekV4Model.reasoningConfig` returns that config. The old comment ("declares no reasoningConfig") is removed.
    - Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift: two comments now give the true reason why ChatSession keeps the whole text as content (no history replay), not "no reasoningConfig".

    Tests:
    - Tests/MLXLMTests/DeepSeekV4EncoderWiringTests.swift: `theSessionThinkingSwitchSelectsTheGenerationMode`, a table of 3 rows (control: no switch gives thinking; on gives thinking; off gives chat). The configuration comes from the production factory load of the synthetic checkpoint, thus the test reads the model declaration. RED before the change: only the off row failed (prompt ended with `<think>`). The session setup is now one shared helper `makeRecordingSession`, also used by `chatSessionUsesTheEncoder`.
    - Tests/MLXFoundationModelsTests/DeepSeekV4ReasoningLevelRenderTests.swift: the executor with the DeepSeek-V4 config and a processor that renders through DeepSeekV4EncodingTokenizer. Rows: no level gives the thinking prompt (control); `no_think` gives the chat prompt. The recorder uses `Mutex`, no `@unchecked Sendable`.
    - Mutant run (one): the config key changed to "enable_thinking". Result: the off row of the MLXLMTests test and the no_think row of the MLXFoundationModels test failed; the control rows passed. Mutant reverted.
    - Comments in DeepSeekV4ToolEncodingTests.swift and DeepSeekV4EncoderWiringTests.swift that said "declares no reasoningConfig" are corrected.

    Behavior change to know (MLXFoundationModels only): for DeepSeek-V4 the executor now applies the prompt strategy. When `.reasoning` is not declared, the prompt is chat mode. When it is declared, `<think>`...`</think>` text goes to reasoning entries, not to the response. This is the same as for Qwen. ChatSession output does not change (no history replay, so no split), and the DSML committed-turn rule does not read the config.

    Real weights: IntegrationTestingTests/DeepseekV4IntegrationTests/chatAndThinkingModesBothGenerate() passed on mlx-community/DeepSeek-V4-Flash-4bit (thinking output: reasoning + `</think>` + answer; chat output: answer only).
  timestamp: 2026-10-16T07:48:27.737281+00:00
- actor: claude-code
  id: 01m51tyd0kptkdq43qjceqyqd9
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXLMCommon/Tokenizer.swift, Libraries/MLXLLM/Models/DeepSeekV4.swift, Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift, Tests/MLXLMTests/DeepSeekV4EncoderWiringTests.swift, Tests/MLXLMTests/DeepSeekV4ToolEncodingTests.swift, Tests/MLXFoundationModelsTests/DeepSeekV4ReasoningLevelRenderTests.swift (new). `swift build --build-tests`: 0 warnings (only the known "missing creator for mutated node" note). xctest: MLXLMTests 721 XCTest + 1514 Swift Testing pass; MLXFoundationModelsTests 369 pass; MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 pass. One mutant run killed in both new tests. Real weights: DeepseekV4IntegrationTests/chatAndThinkingModesBothGenerate() pass.
    - next: /review. Task stays in doing. Not committed.
  timestamp: 2026-10-16T07:48:38.035817+00:00
- actor: claude-code
  id: 01m51v2edqfeb4n1mypzqtps6h
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (ce4e4ea). 0 findings, 0 confirmed, 1 refuted, 7 attempted. 6 files reviewed. The 2 .kanban files have no validator.
    - next: The task is in done. No more work.
  timestamp: 2026-10-16T07:50:50.551215+00:00
- actor: claude-code
  id: 01m51v2wf44h3gj1hgty5tb56p
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 6 files; DeepSeek-V4 declares a reasoning config
    - test: green — run on the final tree by implement: MLXLMTests 721 + 1514, MLXFoundationModelsTests 369, other bundles pass, real-weights DeepSeek-V4 test pass; 0 compiler warnings
    - commit: ce4e4ea
    - review: clean — 0 findings (1 refuted); task moved to done
  timestamp: 2026-10-16T07:51:04.932170+00:00
position_column: done
position_ordinal: ffdf80
title: 'DeepSeek-V4: the doc of thinkingKey names a ReasoningConfig row that does not exist'
---
## What

Found in ^yp1jxca. The doc of `DeepSeekV4EncodingTokenizer.thinkingKey` (Libraries/MLXLMCommon/Tokenizer.swift) says the key is "the same key the family's `ReasoningConfig` row injects". DeepSeek-V4 declares no `reasoningConfig` (DeepSeekV4ChatEncoder.swift says so), and no resolver row gives one. Thus the shared thinking switch of MLXFoundationModels does not reach DeepSeek-V4.

## Acceptance Criteria

- [x] The doc and the code agree: correct the doc, or add the row with a test.

#bug