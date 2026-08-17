---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'DeepSeek-V4: the tool round reuses NO prompt cache, although its render extends round 1 whole'
---
Found on 2026-08-17 while card `^z5xrzg6` made DeepSeek-V4 tool calling work. The
tool round now completes, thus the prompt-cache assertions of
`DeepseekV4AgenticPromptCacheAssessmentTests` are REACHABLE for the first time.
Fact (a) passes. Fact (b) fails.

## The measurement, real weights, one load

`chatModeToolRoundReusesThePromptCache` and
`thinkingModeToolRoundReusesThePromptCache`, one process, 1034 s.

| quantity | chat | thinking |
| --- | ---: | ---: |
| round 1 rendered tokens | 3814 | 3814 |
| round 1 tool calls | `["get_stock_level"]` | `["get_stock_level"]` |
| tool round rendered tokens | 3906 | 3918 |
| tool round FED tokens | 3906 | 3918 |
| tool round tokens skipped by reuse | **0** | **0** |
| tool round prefill seconds | 15.71 | 18.14 |
| cold control prefill seconds | 15.63 | 21.39 |
| follow-up round tokens skipped by reuse | 3951 | **0** |
| follow-up round prefill seconds | 2.69 | 19.91 |

## Fact (a) PASSES, thus the seam is not the cause

- `chat tool round extends round 1 = true`, common prefix 3814 of 3814.
- `thinking tool round extends round 1 = true`, common prefix 3814 of 3814.

Round 1's divergent tail is EMPTY in both modes, thus the tool round's render
holds round 1's render as a whole prefix. `ExtendCachedPrefixRule` is reachable,
and it skips nothing.

## Fact (b) FAILS

- `toolRound.skippedTokenCount > 0` fails in both modes: 0 skipped.
- `toolRound.prefillSeconds < control.prefillSeconds * noReuseTimeFloorFraction`
  fails in both modes. The tool round takes as long as a COLD session on the same
  prompt, which is what 0 skipped tokens predicts.
- `followUp.skippedTokenCount > 0` fails in THINKING mode alone: 0 skipped.

## The one clue the numbers carry

Chat mode's FOLLOW-UP round DOES reuse: it skips 3951 of 3967 tokens and takes
2.69 s in place of about 15 s. Thus the reuse machinery works on that path. Only
the TOOL round resets it, in both modes, and in thinking mode the follow-up round
after the tool round stays reset as well.

The divergent tail of the thinking tool round is:

```
</think>I've read through the stock report. Now I need to call the get
```

Thus the tool round's render adds a `</think>` close and the assistant text of
round 1 before the tool result. Read whether the session drops or rebuilds the
cache when it appends a tool-result turn, rather than extending it.

## Acceptance criteria

- [ ] Name why the tool round feeds every token although its render extends
      round 1 whole
- [ ] `chatModeToolRoundReusesThePromptCache` passes on the real weights
- [ ] `thinkingModeToolRoundReusesThePromptCache` passes on the real weights

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process. The two
tests above run in one process, because the suite awaits one shared load. #deepseek-v4