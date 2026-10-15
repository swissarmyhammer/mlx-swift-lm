---
assignees:
- claude-code
position_column: todo
position_ordinal: '9480'
title: 'Qwen3.8: after a turn that stops inside its reasoning, the splice of ChatSession omits the close of the think block'
---
## What

Found in ^z4aax33. ChatSession now records the reasoning of a Qwen 3.8 turn in `reasoning_content`. When turn 1 stops on the token budget inside its reasoning (no `</think>`), the render of turn 2 writes `<think>\n` + reasoning + `\n</think>\n\n` + `<|im_end|>`. The cache holds the reasoning without `\n</think>\n\n`.

`QwenCommittedTurnRule` (CommittedTurnSplice) splices at the commit: the warm model reads its unclosed reasoning, then `<|im_end|>`. The cold render has the closed block. In the qwen38bench at 32k with 48 decode tokens (scratchpad/qwen38/z4a-new-48-1.txt), turn 2 warm and cold gave different first tokens ("We" against "User"). The tie-aware check passed (gap 2.625, margin 4.0).

Before ^z4aax33 this case went back to the checkpoint at the end of the last message and prefilled the canonical render (warm == cold, 73 tokens, 0.42 s). Now it prefills 19 tokens (0.25 s).

## Options

1. Keep the splice (the documented behavior of CommittedTurnSplice for a budget stop).
2. Decline the splice when the generation stopped inside reasoning, and go back to the checkpoint at the end of the prompt. That checkpoint is now a prefix of the turn-2 render. The cost is a prefill of the generated tokens again.
3. Feed the tokens of the close of the think block before the commit.

A person must select the option.

## Acceptance Criteria

- [ ] The card records the selected option and the reason.
- [ ] A test shows the turn-2 reuse after a turn that stops inside its reasoning. #qwen3-8