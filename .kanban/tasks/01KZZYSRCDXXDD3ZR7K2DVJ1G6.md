---
assignees:
- claude-code
position_column: todo
position_ordinal: '9880'
title: DeepSeek-V4 writes its tool calls as plain JSON, which DSMLToolCallParser does not read
---
Measured on 2026-08-13 against `mlx-community/DeepSeek-V4-Flash-4bit` with the
real weights.

A short prompt that offers one tool made the model write:

```
{"function": "get_stock_level", "params": {"bay_id": "bay_7"}}
```

`DSMLToolCallParser` reads none of that, thus `toolCalls` stays empty and no
agentic round completes. Two things differ from the schema the prompt gave:

- The syntax is plain JSON, not the DSML the parser reads.
- The argument name is `bay_id`, and the tool schema names it `bay`.

## Why it blocks the agentic goal

`chatModeToolRoundReusesThePromptCache` and
`thinkingModeToolRoundReusesThePromptCache` both stop at
`#require(roundOne.toolCalls.first)`. Every tool-round measurement is thus out
of reach, which is why card `^3hb0rq4` had to measure a conversation with no
tools.

## The work

- [ ] Read the `## Tools` section `DeepSeekV4ChatEncoder` renders, and compare
      it with the section of the published reference. A prompt that does not
      match the training shape explains a model that answers in a different
      syntax
- [ ] Tell whether the defect is the render, the parser, or the model
- [ ] Correct what the answer names, and make one tool round complete

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process, or
the machine runs out of memory. #deepseek-v4