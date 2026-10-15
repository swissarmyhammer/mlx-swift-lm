---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4zmgf1zx1xjewzqps69r2vt
  text: 'Finding from ^95mp4sw: the qwen38bench passes `additionalContext: ["thinking": ...]`. `thinking` is the DeepSeek-V4 key (`DeepSeekV4EncodingTokenizer.thinkingKey`). The Qwen 3.5/3.8 template reads `enable_thinking`. With `QWEN38_THINKING_KEY=enable_thinking` (a new bench switch) thinking off works at 32k: turn-1 prompt 32154 tokens instead of 32194, the answer has no reasoning, and turn 2 reuses the cache through `QwenCommittedTurnRule`. Logs: scratchpad/qwen38/reuse-*-enable-*.txt. Thus the flag gets to the template when the caller uses the correct key; the question for this card is whether the library must also accept `thinking` for Qwen (or the reasoning config), not a lost flag.'
  timestamp: 2026-10-15T11:17:41.055927+00:00
position_column: todo
position_ordinal: 8d80
title: 'Qwen3.8: the thinking flag possibly does not get to the chat template'
---
## What

In the qwen38bench runs at 32k, the answers with thinking off and thinking on are the same text, and both show reasoning. It is possible that the `thinking` flag (enable_thinking) does not get to the Qwen3.8 chat template. This is not a speed task.

## Approach

1. Render the Qwen3.8 template with thinking off and on, and compare the prompt tokens.
2. Find where the flag is lost, and fix it. Qwen35ThinkingOffRenderTests and Qwen35ThinkingOffTemplateTests are references.

## Acceptance Criteria

- [ ] A test shows that thinking off and thinking on give different prompts for Qwen3.8.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #bug