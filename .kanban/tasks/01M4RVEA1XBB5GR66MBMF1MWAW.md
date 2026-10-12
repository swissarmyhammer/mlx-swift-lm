---
assignees:
- claude-code
position_column: todo
position_ordinal: '8680'
title: 'DeepSeek-V4 thinking mode: a turn cut by maxTokens before </think> renders differently cold and warm'
---
## What

A thinking-mode turn that stops at `maxTokens` before `</think>` has no close in its content. ChatSession records that content as the assistant turn. A later render of the history (a cold session, or any re-render) writes `</think>` in front of the content: `<Assistant><think></think>partial-reasoning<eos>`. The model wrote `<Assistant><think>partial-reasoning`. Thus the cold history is not the text the model wrote.

Measured on 2026-10-12 with `scratchpad/dsv4bench` (mlx-community/DeepSeek-V4-Flash-4bit, 4k context, `maxTokens` 48, thinking mode), after ^6ytd2tg:
- turn 1 stopped at length, content without `</think>`.
- turn 2 warm: cached=4065, prefilled=11 (the committed-turn splice keeps the live trajectory).
- turn 2 cold from `[.user(p1), .assistant(t1.text)]`: the output differs from warm at character 55.
With `maxTokens` 1024 every turn closes and warm == cold.

## Cause

`DeepSeekV4ChatEncoder.body` writes `(reasoning ?? "") + </think>` before the content when `holdsItsOwnReasoning` is false. A content without `</think>` from a live thinking turn is reasoning, not an answer. A hand-built history `.assistant("answer")` is an answer. The encoder cannot tell the two apart from the content alone.

## Approach (to decide)

- Option A: ChatSession records a turn that stopped inside the reasoning block with the content as `reasoning` (or closes it in the record), so that the render writes `partial</think><eos>`, which is what the reference renders for `reasoning_content=partial, content=""`.
- Option B: mark the recorded turn so the encoder renders the content unchanged.
Record the decision and compare with the reference parser (`parse_message_from_completion_text` asserts a `</think>` in thinking mode).

## Acceptance Criteria

- [ ] A thinking-mode turn cut by `maxTokens` re-renders as a prefix-compatible text, and warm turn 2 equals a cold run from the full history (greedy) on real weights.
- [ ] A unit test pins the render of such a turn.
- [ ] A hand-built answer-only history renders as before.
#deepseek-v4