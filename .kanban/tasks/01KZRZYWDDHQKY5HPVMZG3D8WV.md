---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: 'DeepSeek-V4 tokenizer integration tests: marker round trip and eos token id'
---
## What

Write the two DeepSeek-V4 tokenizer tests that a unit test cannot do. Both tests
need a real tokenizer download. This repository keeps that class of test in
`IntegrationTesting/IntegrationTestingTests/`, not in `Tests/MLXLMTests/`.

These two items come from the card `^gbsaqc2` (Port DeepseekV4ChatEncoder core
rendering). That card made the encoder and proved its output against DeepSeek's
own Python. It could not make these two tests, because the unit suite does not
download a tokenizer.

Make a new file `IntegrationTesting/IntegrationTestingTests/DeepseekV4TokenizerIntegrationTests.swift`.
Follow the shape of the tests that are already in that directory.

Model: `mlx-community/DeepSeek-V4-Flash-4bit`, or the tokenizer of
`deepseek-ai/DeepSeek-V4-Flash`.

## Test 1: the markers make one token each, and the detokenizer does not split them

Push the token ids below through `NaiveStreamingDetokenizer`, one id at a time,
and assert that the text that comes out holds each marker complete. A marker
must never come out in pieces, and `skip_special_tokens` must not remove the
turn markers.

The ids come from the `added_tokens` array of the published `tokenizer.json`:

| Token | Id | `special` |
|---|---|---|
| `<｜begin▁of▁sentence｜>` | 0 | true |
| `<｜end▁of▁sentence｜>` | 1 | true |
| `<｜User｜>` | 128803 | false |
| `<｜Assistant｜>` | 128804 | false |
| `<think>` | 128821 | false |
| `</think>` | 128822 | false |
| `｜DSML｜` | 128825 | false |
| `<｜latest_reminder｜>` | 128828 | false |

The test must assert:

- [ ] Each id above decodes to its marker string, complete, with no split.
- [ ] A stream that mixes markers and ordinary text gives the markers complete
      and the text in the correct order.
- [ ] The turn markers stay in the output. They are not special, thus a
      "skip specials" path must not remove them.

## Test 2: the end-of-sentence token id is 1

The test must assert:

- [ ] `Tokenizer.eosTokenId` is `1` for the DeepSeek-V4 tokenizer.
- [ ] Token id `1` decodes to `<｜end▁of▁sentence｜>`.

## A fact this work established — the card `^gbsaqc2` named the wrong file

`tokenizer_config.json` holds **no** `eos_token_id` key. It holds `eos_token`,
an `AddedToken` whose `content` is `<｜end▁of▁sentence｜>`. `generation_config.json`
holds `eos_token_id: 1`. `tokenizer.json` gives the string `<｜end▁of▁sentence｜>`
the id 1.

`Tokenizer.eosTokenId` in this repository is `tokenId(of: eosToken)`. It reads
the string from `tokenizer_config.json` and looks the string up in the
vocabulary, thus it gives 1. The value 1 is correct. Do not write a test that
reads an `eos_token_id` key out of `tokenizer_config.json`, because that key is
not there.

## Provenance

- `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Encoder under test: `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`, type
  `DeepSeekV4ChatEncoder`. Its file header holds the same token id table.

## Workflow

- Use `/tdd`.
- Run the new tests from the `IntegrationTesting` Xcode project, as the other
  tests in that directory do.
#deepseek-v4