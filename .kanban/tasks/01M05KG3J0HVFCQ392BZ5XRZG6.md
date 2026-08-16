---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'DeepSeek-V4 generation drops one token: the model writes `</｜DSML｜inv>` for `</｜DSML｜invoke>`'
---
Measured on 2026-08-16 against `mlx-community/DeepSeek-V4-Flash-4bit` with the
real weights, after card `^2dvj1g6` wired the DeepSeek-V4 prompt path into
`LLMModelFactory._load`.

The model now writes a nearly correct DSML tool call. One token is absent:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_stock_level">
<｜DSML｜parameter name="bay" string="true">bay 7</｜DSML｜parameter>
</｜DSML｜inv>
</｜DSML｜tool_calls>
```

The closing tag reads `</｜DSML｜inv>` where the syntax states
`</｜DSML｜invoke>`. Every other byte of the 179 is correct: the block, the
invoke element, the tool name, the parameter name `bay`, and the closing
`</｜DSML｜tool_calls>`.

`DSMLToolCallParser` refuses the payload for that one tag, thus no tool round
completes.

## Acceptance criteria

This card owns the round-completion criterion. Card `^2dvj1g6` held it while the
cause looked like the prompt. That card corrected the prompt path and proved the
model now reads the `## Tools` section, thus the criterion moved here with the
defect that holds it back.

- [ ] Tell whether the 4-bit weights or the DeepSeek-V4 attention port omits the
      identifier, with the experiment below
- [ ] Correct the cause the experiment names
- [ ] Make one tool round complete:
      `DeepseekV4IntegrationTests.aShortToolPromptEmitsOneDSMLToolCall` passes on
      the real weights
- [ ] `chatModeToolRoundReusesThePromptCache` and
      `thinkingModeToolRoundReusesThePromptCache` both get past
      `#require(roundOne.toolCalls.first)`, which is the agentic measurement that
      every tool-round card waits for

## The absent token is exactly one vocabulary entry

The published `tokenizer.json` holds NO `invoke` token. It writes the word as
two tokens, and the tag ends with one more:

| piece | id |
| --- | ---: |
| `</` | 1718 |
| `｜DSML｜` | 128825 |
| `inv` | 40148 |
| `oke` | 5406 |
| `>\n` | 1018 |

The correct tag is the five identifiers `[1718, 128825, 40148, 5406, 1018]`. The
answer holds the four identifiers `[1718, 128825, 40148, 1018]`. The loss is one
whole token at a true token boundary, and both sequences are legal outputs of a
sampler. The correct text is 182 bytes and the observed text is 179.

## Our text path is proven clean, thus the defect is upstream of it

`DeepSeekV4TokenizerIntegrationTests.aToolCallStreamsBackWholeAcrossItsNewlines`
is weight-free and GREEN. It tokenizes the correct DSML answer with the
production tokenization, streams the identifiers one at a time through
`NaiveStreamingDetokenizer`, and gets the answer back byte for byte. The test
covers the newlines, which is where `NaiveStreamingDetokenizer` starts a new
segment.

Each other stage is clear as well:

- `NaiveStreamingDetokenizer`. `common` is by construction a prefix of
  `newSegment`, thus a divergence RE-EMITS a tail and never deletes one. The
  incomplete-character path returns `nil` without advancing `segment`.
  `startNewSegment()` is the one place with a deletion hazard, and it runs only
  after a newline, which `oke` does not follow.
- `StopStringFilter` is off: `generation_config.json` states no stop string, and
  `stopStrings` and `extraEOSTokens` are both empty. The filter is lossless in
  any case, because a partial match stays in its buffer.
- `ToolCallProcessor` gives the parser a CONTIGUOUS slice of its own buffer, and
  `rawText` carries no sanitizing. A slice cannot hold a gap, thus the buffer
  itself lacked the token.
- `DSMLToolCallParser` holds no regex. It walks literal ranges, and
  `invokeClose` is the literal `</｜DSML｜invoke>`. The parser is correct, and it
  must NOT be loosened: the published `parse_tool_calls` accepts the one syntax
  and raises on every other.
- Speculative decoding is not in play. The load filter of `DeepSeekV4Model` drops
  every `mtp.` tensor, thus the multi-token-prediction head never loads.

Thus the identifier 5406 never entered the stream, and the cause is in the
generation of the identifiers.

## The two candidates

1. The 4-bit quantization of `mlx-community/DeepSeek-V4-Flash-4bit`. Greedy
   decoding is deterministic, thus a wrong argmax at one step repeats on every
   run.
2. A number in the DeepSeek-V4 attention port. A single skipped step is what an
   off-by-one in the pooled-chunk path or in the cache offset looks like.

## How to tell the two apart

Run the same conversation through the Python reference (`ml-explore/mlx-lm`) on
the same checkpoint, greedy, and read the generated identifiers.

- The reference also omits 5406: the checkpoint is the cause, and the port is
  correct.
- The reference writes 5406: the port is the cause.

Capture the raw identifiers on the Swift side as well. `TokenIterator` gives
them directly, as `greedyFirstTokensMatchThePythonFixture` already does.

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process.
#deepseek-v4