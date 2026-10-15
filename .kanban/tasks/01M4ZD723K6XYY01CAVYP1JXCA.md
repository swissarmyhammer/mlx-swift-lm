---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4zmgf1zx1xjewzqps69r2vt
  text: 'Finding from ^95mp4sw: the qwen38bench passes `additionalContext: ["thinking": ...]`. `thinking` is the DeepSeek-V4 key (`DeepSeekV4EncodingTokenizer.thinkingKey`). The Qwen 3.5/3.8 template reads `enable_thinking`. With `QWEN38_THINKING_KEY=enable_thinking` (a new bench switch) thinking off works at 32k: turn-1 prompt 32154 tokens instead of 32194, the answer has no reasoning, and turn 2 reuses the cache through `QwenCommittedTurnRule`. Logs: scratchpad/qwen38/reuse-*-enable-*.txt. Thus the flag gets to the template when the caller uses the correct key; the question for this card is whether the library must also accept `thinking` for Qwen (or the reasoning config), not a lost flag.'
  timestamp: 2026-10-15T11:17:41.055927+00:00
- actor: claude-code
  id: 01m50rqmaqz6aw1mk341a8cyg2
  text: |-
    Research. The library has one model-independent thinking switch: `ReasoningConfig.promptStrategy` (`additionalContext(forThinkingEnabled:)` and `closedBlock(forThinkingEnabled:)`). Each model declares its `reasoningConfig`; Qwen 3.5 / Qwen3.8 (`model_type` `qwen3_5`, `Qwen35TextModel`, VLM `Qwen35`) declares `QwenReasoningProtocol.qwen35` (ChatConventionsTests checks this). `MLXLanguageModel` (MLXFoundationModels) maps `ContextOptions.reasoningLevel` to that switch through `ThinkingRender`. For Qwen 3.5 the switch keeps `enable_thinking` true and puts the closed block `<think>\n\n</think>\n\n` in place of the open block, so that the system block (and the prompt cache) stays the same. The real template test `IntegrationTesting/.../Qwen35ThinkingOffTemplateTests` already renders the Qwen3.8 snapshot template this way.
    The MLXLMCommon path (`ChatSession`, `UserInput.additionalContext`) has no thinking switch: the caller passes the template keys of the model. The qwen38bench used that path with the DeepSeek-V4 key `thinking`, so Qwen ignored it. This is a caller error, not a lost flag. A caller of that path can get the keys from `configuration.reasoningConfig?.promptStrategy.additionalContext(forThinkingEnabled:)`, but for Qwen 3.5 that gives `enable_thinking: true` and the caller must also apply `closingReasoning(in:with:tokenizer:)`; `ChatSession` does not do this.
    Other discovery: DeepSeek-V4 declares no `reasoningConfig` (DeepSeekV4ChatEncoder.swift says so), but the doc of `DeepSeekV4EncodingTokenizer.thinkingKey` says the key is "the same key the family's ReasoningConfig row injects". That doc does not agree with the code.
    Plan: the shared switch exists and Qwen3.8 honours it, thus the fix is the test. Add to `Qwen35ThinkingOffRenderTests` a test that a pass with thinking off and a pass with thinking on render different prompts, with a positive control (two passes with thinking on render the same prompt).
  timestamp: 2026-10-15T21:50:44.567196+00:00
- actor: claude-code
  id: 01m50s1x1zvh52jr0q3f0yt1bb
  text: |-
    Implementation. The flag is not lost: the shared switch (`ContextOptions.reasoningLevel` -> `ReasoningConfig.promptStrategy`) gets to the Qwen 3.5 render. Thus no production change. Added the test "a turn with thinking off renders a prompt other than a turn with thinking on" (both the no-tool path and the tool path) to Tests/MLXFoundationModelsTests/Qwen35ThinkingOffRenderTests.swift. It renders one pass with thinking on two times (control: the two renders must be equal), and one pass with thinking off; the off render must differ, and the on render must end with the open think block. New shared helpers `renderOfOnePass(thinking:withTools:)` and `script(thinking:)`; the existing system-block check and the reuse check now use them.
    RED: with a temporary mutation in `ReasoningPromptStrategy.closedBlock(forThinkingEnabled:)` (return nil for thinking off, that is a lost flag), the new test failed at `thinkingOff != thinkingOn` for both arguments, and the control passed. The mutation is removed; ReasoningConfig.swift is the same as HEAD.
    GREEN: `swift build --build-tests` exit 0, the one warning is the known "missing creator for mutated node". MLXFoundationModelsTests: 368 tests in 40 suites passed. MLXLMTests: 1503 Swift Testing tests in 122 suites passed, XCTest 721 tests 0 failures.
    Note: I ran the MLXFoundationModelsTests bundle two times in the RED step with no code change between, because the first run output went to a filter that showed nothing useful.
    New tasks for the discoveries: ^2fm0eaq (ChatSession has no model-independent thinking switch; a design decision for the user) and ^ggn6kfj (the doc of DeepSeek-V4 `thinkingKey` names a ReasoningConfig row that does not exist).
  timestamp: 2026-10-15T21:56:21.183058+00:00
- actor: claude-code
  id: 01m50s1ypwyda2rshhry6v3w29
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/Qwen35ThinkingOffRenderTests.swift; build 0 warnings (only the known build note); MLXFoundationModelsTests 368 passed; MLXLMTests 1503 + 721 passed, 0 failures
    - next: review
  timestamp: 2026-10-15T21:56:22.876399+00:00
- actor: claude-code
  id: 01m50s4fwzhektfgw6z69gc72g
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (e1a66ef). 0 findings, 0 confirmed, 0 refuted. 1 file reviewed: Tests/MLXFoundationModelsTests/Qwen35ThinkingOffRenderTests.swift. 6 .kanban files not reviewed (no validator matches).
    - next: none. The task is in done.
  timestamp: 2026-10-15T21:57:46.015687+00:00
- actor: claude-code
  id: 01m50s4tantwq8are8ec9h28w8
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 1 test file; the flag is not lost; new cards ^2fm0eaq and ^ggn6kfj
    - test: green — run on the final tree by implement: MLXFoundationModelsTests 368 passed; MLXLMTests 1503 + 721 passed; 0 compiler warnings
    - commit: e1a66ef
    - review: clean — 0 findings; task moved to done
  timestamp: 2026-10-15T21:57:56.693235+00:00
position_column: done
position_ordinal: ffda80
title: 'Qwen3.8: the thinking flag possibly does not get to the chat template'
---
## What

In the qwen38bench runs at 32k, the answers with thinking off and thinking on are the same text, and both show reasoning. It is possible that the `thinking` flag (enable_thinking) does not get to the Qwen3.8 chat template. This is not a speed task.

## Approach

1. Render the Qwen3.8 template with thinking off and on, and compare the prompt tokens.
2. Find where the flag is lost, and fix it. Qwen35ThinkingOffRenderTests and Qwen35ThinkingOffTemplateTests are references.

## Acceptance Criteria

- [x] A test shows that thinking off and thinking on give different prompts for Qwen3.8.
- [x] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #bug