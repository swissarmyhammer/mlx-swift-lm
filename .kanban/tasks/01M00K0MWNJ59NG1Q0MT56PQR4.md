---
assignees:
- claude-code
position_column: todo
position_ordinal: 9a80
title: swift-transformers splits every newline on its own, thus every prompt with a blank line gets the wrong token identifiers
---
Found on 2026-08-14 while card `^2dvj1g6` looked for the reason DeepSeek-V4
writes no DSML tool call. It BLOCKS that card.

## The measurement

`DeepSeekV4TokenizerIntegrationTests.theToolPromptTokenizesToThePublishedIdentifiers`
renders one tool prompt through the production path
(`DeepSeekV4EncodingTokenizer.applyChatTemplate`) and compares the identifiers
with the published `tokenizer.json` of `deepseek-ai/DeepSeek-V4-Flash`:

```
Swift wrote 353 identifiers and the reference holds 328.
The first difference is at index 15:
Swift     [... 2910 "Ġgiven", 16 ".", 201 "Ċ", 201 "Ċ", 372 "##", 27193 "ĠTools", 201 "Ċ"]
reference [... 2910 "Ġgiven", 339 ".ĊĊ", 372 "##", 27193 "ĠTools", 271 "ĊĊ", 3476 "You"]
```

The Swift tokenizer writes one `Ċ` for each newline. The published tokenizer
merges a newline run into one token, `ĊĊ` (271), and it merges a newline run
onto the punctuation in front of it, `.ĊĊ` (339).

## The defect

`swift-transformers` 1.3.3,
`Sources/Tokenizers/String+PreTokenization.swift`. `SplitPreTokenizer` sends a
`Regex` pattern through `String.split(by:options:includeSeparators:)`, which
loops on `String.range(of:options:.regularExpression)`. That Foundation search
cannot match a `\r` or a `\n` inside a character class.

Measured on the published pre-tokenizer pattern with the text
`"given.\n\n## Tools\n\nYou have"`:

```
String.range(of:options:.regularExpression):
  ["given", ".", "\n", "\n", "##", " Tools", "\n", "\nYou", " have"]

NSRegularExpression, SAME pattern:
  ["given", ".\n\n", "##", " Tools", "\n\n", "You", " have"]
```

The second line is the reference answer. Sharper still:
` ?[\p{P}\p{S}]+[\r\n]*` against `".\n\n## Tools"` matches `"."` alone under
Foundation, and `\s*[\r\n]+` against `"\n\n## Tools"` matches nothing at all.

`swift-transformers` already holds a correct path: `splitMatches(in:with:)` and
`split(by captureRegex: NSRegularExpression)` both use `NSRegularExpression`.
`SplitPreTokenizer` is the one place that does not.

## The reach

This is not a DeepSeek-V4 defect. Any checkpoint whose `tokenizer.json` holds a
`Split` pre-tokenizer with a pattern that crosses a newline gets the wrong
identifiers. DeepSeek-V4 shows it first because its `## Tools` section is 24
lines with 10 blank lines in it, thus that section carries nearly all of the 25
extra identifiers and the model reads it in a shape it never saw in training.

## The choice a person must make

The package is `https://github.com/huggingface/swift-transformers` 1.3.3, and it
arrives through `https://github.com/huggingface/swift-huggingface` 0.9.0. The
root `Package.swift` of this repository does not declare it, and 1.3.3 is the
newest tag, thus a version bump corrects nothing.

- [ ] Choose the correction: send it upstream and move the pin, fork or vendor
      `swift-transformers` here, or give `DeepSeekV4EncodingTokenizer` its own
      pre-tokenization path
- [ ] Make the correction
- [ ] `theToolPromptTokenizesToThePublishedIdentifiers` passes
- [ ] Run `aShortToolPromptEmitsOneDSMLToolCall` again, and record what the
      model writes on a correct token sequence

## Memory

The DeepSeek-V4 checkpoint holds 141 GiB. Run ONE real-weights test for each
process, or the machine runs out of memory. #deepseek-v4
