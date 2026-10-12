---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'DeepSeek-V4 thinking mode: make the prompt cache reuse the prefix across turns'
---
## What

In thinking mode, a `ChatSession` with `mlx-community/DeepSeek-V4-Flash-4bit` never reuses the prompt cache. Each turn prefills the full conversation again.

Measured on 2026-10-12 (M3 Ultra, 8k-token context, `scratchpad/dsv4bench`):
- Chat mode: turn 2 reused 8043 tokens and prefilled 10 tokens in 0.69 s. The warm output was the same as the cold output.
- Thinking mode: turns 1, 2 and 3 each prefilled 8016, 8075 and 8135 tokens with `cached=0`, about 30 s for each turn.

Cause: `Tokenizer.applyChatTemplate` (`Libraries/MLXLMCommon/Tokenizer.swift:349-359`) calls `DeepSeekV4ChatEncoder().encode(messages:thinkingMode:)` with the default `dropsEarlierReasoning`. In thinking mode the encoder then removes the reasoning of earlier assistant turns (`DeepSeekV4ChatEncoder.swift`, `removingEarlierReasoning`). Thus the render of turn 1 is not a prefix of the render of turn 2.

This is the same class of problem as the Qwen 3.6 case. Task `01KXZP6CTSETM3YJS3E05ZT40G` fixed it with `preserve_thinking` and a replay of the reasoning content, so that the history stays a prefix. Also see commit `bcf4f54`, which keeps the system prompt the same for Qwen 3.5.

## Approach

- Make the session render keep the earlier reasoning (`dropsEarlierReasoning: false`, or a context key that selects it), and replay the reasoning of each assistant turn into the history.
- Make sure that the model sees the same text as the published reference encoder when the reasoning is kept. Record the decision.

## Acceptance Criteria

- [ ] In thinking mode, turn 2 of a `ChatSession` reports `cachedPromptTokenCount > 0` and prefills only the new turn.
- [ ] The warm turn 2 output is the same as a cold run from the full history (greedy).
- [ ] A unit test shows that the thinking-mode render of turn 1 is a prefix of the render of turn 2.
- [ ] Chat mode still reuses the cache.
#deepseek-v4