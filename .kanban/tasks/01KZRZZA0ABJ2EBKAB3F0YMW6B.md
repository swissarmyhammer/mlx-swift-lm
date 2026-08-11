---
assignees:
- claude-code
position_column: todo
position_ordinal: '9780'
title: 'Decision needed: keep or drop the DSV4Minimal.jinja fallback template'
---
## What

A person must decide. Do not write code before the decision.

The card `^gbsaqc2` (Port DeepseekV4ChatEncoder core rendering) named a second,
Jinja path for DeepSeek-V4 prompts: `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`,
from `osaurus-ai/vmlx-swift-lm`. The port did not carry that file. This card
records why, and puts the choice to a person.

## The technical finding

DeepSeek-V4 ships **no** `chat_template`, neither in `tokenizer_config.json` nor
in `tokenizer.json`. DeepSeek gives prompt construction as a separate Python
file, `encoding/encoding_dsv4.py`. The Swift encoder
`Libraries/MLXLMCommon/DeepSeekV4ChatEncoder` follows that Python.

A Jinja template cannot state two of the rules that the Python states:

1. **The `drop_earlier_reasoning` rule.** The rule looks at the whole
   conversation and removes the reasoning block of every turn before the last
   one. It also counts the context messages to compute `context_len`. Jinja has
   no way to state this multi-message rule.
2. **The task markers.** The six task markers (token ids 128829 to 128845) and
   the action-task path choose a different marker order for the assistant turn.
   The choice depends on the task type, not on the message role, thus a template
   that only walks the message list cannot make it.

Therefore a Jinja fallback cannot agree with the Swift encoder. A second path
that disagrees is worse than no second path: it gives a wrong prompt with no
error.

## The choice

Pick one:

- **A. Keep a partial template.** Carry `DSV4Minimal.jinja`, but only for simple
  turns: user and assistant messages, no reasoning drop, no task markers, no
  tools. The template must fail loudly, or the loader must refuse it, when the
  conversation needs a rule the template cannot state.
- **B. Drop the fallback.** DeepSeek-V4 prompts always come from
  `DeepSeekV4ChatEncoder`. A DeepSeek-V4 model with no encoder is an error, not
  a fall back to a template.

## Acceptance Criteria

- [ ] A person writes the choice, A or B, on this card.
- [ ] If A: a new task states the exact set of turns the template handles, and
      the error the loader gives for every other turn.
- [ ] If B: a new task, or this card, records the refusal path and its error
      message.

## Provenance

- Fallback template: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`.
- Source of truth: `deepseek-ai/DeepSeek-V4-Flash` — `encoding/encoding_dsv4.py`.
- Moved from `^gbsaqc2` on 2026-08-11.
#deepseek-v4