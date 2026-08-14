---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m00ma3x241d26gph0w40ve0a
  text: |
    ### The user chose correction 3

    > Give `DeepSeekV4EncodingTokenizer` its own pre-tokenization path that does
    > not use the broken splitter.

    Thus the correction stays inside this repository. Do NOT fork the package, do
    NOT vendor it, and do NOT wait on an upstream change.

    The other two are closed:

    1. Send the correction upstream, then move the pin — NOT chosen.
    2. Fork or vendor `swift-transformers` here — NOT chosen.

    ### What correction 3 must respect

    - The correction belongs to the DeepSeek-V4 path. This branch is under a strict
      rule that the difference against `main` holds DeepSeek-V4 and MiniMax work
      alone, thus a change to a shared file must be one that only the DeepSeek-V4
      path reads.
    - `theToolPromptTokenizesToThePublishedIdentifiers` is the acceptance test. It
      needs no weights, and it is red now.
    - `NSRegularExpression` with the SAME pattern already gives the reference
      answer. The correction is the search, not the pattern.
  timestamp: 2026-08-14T17:14:58.338878+00:00
- actor: claude-code
  id: 01m00nrrns54hd2mze3t13gt73
  text: |
    Research for correction 3. The measurements below come from the published
    `tokenizer.json` of `deepseek-ai/DeepSeek-V4-Flash` @ 60d8d707, read from the
    Hub cache on this machine, and from a Python model of the correction that runs
    beside the published `tokenizers` 0.23.1 library.

    ## What the published `tokenizer.json` states

    The `pre_tokenizer` is a `Sequence` of four steps:

    1. `Split` `\p{N}{1,3}`, behavior `Isolated`
    2. `Split` `[一-龥぀-ゟ゠-ヿ]+`, behavior `Isolated`
    3. `Split` `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, behavior `Isolated`
    4. `ByteLevel`, `add_prefix_space` false, `use_regex` FALSE

    Step 4 does no splitting: it only byte-encodes each piece. Thus step 3 is the
    one that groups a run of newlines, and it is the one Foundation breaks.

    The `normalizer` is an empty `Sequence`, thus it is identity. The
    `post_processor` is `ByteLevel`, and `ByteLevelPostProcessor.postProcess`
    returns its tokens unchanged, thus it adds no identifier. The model is `BPE`
    with 128000 vocabulary entries and 127741 merges, `byte_fallback` false,
    `ignore_merges` absent.

    ## The merges are reachable through the vocabulary alone

    `DeepSeekV4EncodingTokenizer` holds `any Tokenizer`, thus it can look an
    identifier up by token text and it cannot read the merge list. It does not need
    the merge list:

    - Each merge result is in the vocabulary (0 missing of 127741).
    - The identifier of each merge result GROWS with the merge index, measured
      strictly increasing over all 127741 merges.
    - The 259 vocabulary entries that no merge makes hold the identifiers 0 to 258:
      three markers and the 256 byte tokens.

    Thus "merge the neighbouring pair whose joined text has the LOWEST vocabulary
    identifier" gives the same answer as "merge the pair with the lowest merge
    rank". Measured over 76880 distinct pre-tokens taken from 200 Swift files of
    this repository and from `tokenizer.json` itself: 0 disagreements.

    ## The Python model of the correction

    Marker split, then the three splits above through a correct regular-expression
    engine, then byte-level encode, then the lowest-identifier merge. Compared with
    `tokenizers.Tokenizer.from_file(...).encode(text, add_special_tokens=False)`:

    | case | tokens | equal |
    | --- | ---: | --- |
    | the failing tool prompt | 328 | yes |
    | markers with newlines beside them | 10 | yes |
    | eight newlines in a row | 3 | yes |
    | a fenced JSON block | 18 | yes |
    | German, Japanese, digits, tabs, CRLF | 27 | yes |
    | runs of punctuation and newlines | 4 | yes |
    | leading and trailing whitespace | 3 | yes |
    | `Libraries/MLXLMCommon/Tokenizer.swift` | 4305 | yes |
    | `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift` | 9511 | yes |
    | emoji beside a marker and a newline | 11 | yes |

    The same model reproduces the card's reference split word for word:

    ```
    "given.\n\n## Tools\n\nYou have"
      -> ['given', '.\n\n', '##', ' Tools', '\n\n', 'You', ' have']
    ```

    ## Why the marker split comes first

    `PreTrainedTokenizer.tokenize` of `swift-transformers` removes each added token
    BEFORE it pre-tokenizes. That order matters: `<｜User｜>` ends with `>`, which
    is `\p{P}`, thus a following newline would otherwise join the marker's last
    character in one piece ` ?[\p{P}\p{S}]+[\r\n]*`. The correction keeps the same
    order and splits on the markers that `DeepSeekV4ChatEncoder.SpecialToken`
    writes.

    ## What the correction cannot use

    `swift-transformers` keeps `SplitPreTokenizer`, `PreTokenizerFactory` and
    `PreTrainedTokenizer.model` internal, thus this repository cannot reach the
    pre-tokenizer or the merge table of the loaded tokenizer. The correction reads
    the loaded tokenizer only through `convertTokenToId(_:)`, which the
    `MLXLMCommon.Tokenizer` protocol publishes.
  timestamp: 2026-08-14T17:40:26.937537+00:00
- actor: claude-code
  id: 01m00rxags3vsskd0ws2rjyy2p
  text: |
    The real-weights run, and what it rules out.

    ## The command and the answer

    ```
    xcodebuild test-without-building ... \
      "-only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/aShortToolPromptEmitsOneDSMLToolCall()"

    Tool round text: <<<

    <functioncall>
    {"name": "get_stock_level", "arguments": {"bay_id": "bay 7"}}>>>
    Tool round calls: []
    ✘ Test aShortToolPromptEmitsOneDSMLToolCall() failed after 46.773 seconds
    ```

    That answer is BYTE IDENTICAL to the answer of 2026-08-14 before this
    correction, although the prompt now carries 328 identifiers where it carried
    353. The corrected identifiers moved the greedy trajectory not at all.

    ## The run did read the corrected identifiers

    Three facts tie the run to the correction:

    - `aShortToolPromptEmitsOneDSMLToolCall` opens a `ChatSession` with
      `additionalContext: ["thinking": false]`, `tools: [stockToolSpec]` and
      `stockToolUserPrompt`. That is the same conversation, in the same mode, that
      `theToolPromptTokenizesToThePublishedIdentifiers` renders, and that test now
      passes with 328 identifiers.
    - `ChatSession` reaches `DeepSeekV4EncodingTokenizer.applyChatTemplate`, which
      `DeepSeekV4EncoderWiringTests` pins.
    - The tokenizer of `mlx-community/DeepSeek-V4-Flash-4bit` and the tokenizer of
      `deepseek-ai/DeepSeek-V4-Flash` agree on every part this work reads:
      `pre_tokenizer`, `normalizer`, `post_processor`, `decoder`, `added_tokens`
      and the whole 128000-entry vocabulary are equal, and the two files give the
      failing prompt the same 328 identifiers. The two `merges` arrays differ only
      in their serialized form.

    ## What this rules out

    The tokenizer is dead as a cause, the same way the render, the parser, the tool
    placement, the prompt length and the generation mode are already dead:

    | cause | state | evidence |
    | --- | --- | --- |
    | the render | dead | byte equal to `encoding_dsv4.py`, 1454 bytes |
    | the parser | dead | a faithful reading of `parse_tool_calls` |
    | the tool placement | dead | agrees with `test_encoding_dsv4.py` cases 1 and 3 |
    | the prompt length | dead | 3626-token recall test answered `4172` |
    | the generation mode | dead | thinking mode fails too |
    | the token identifiers | dead NOW | 328 of 328 equal to the published answer |

    What is left is the weights and the layers that read them: the 4-bit
    quantization of this checkpoint, or a number in the DeepSeek-V4 attention port.
    Card `^2dvj1g6` holds that question. Its last work item — "Correct what the
    answer names, and make one tool round complete" — stays open, and this card no
    longer blocks it.

    ## One note for the next agent

    The model reads `bay_id` where the `## Tools` section states `bay`, and it has
    done so in all four runs (2026-08-13 chat, 2026-08-14 chat, 2026-08-14
    thinking, 2026-08-14 chat with correct identifiers). `bay_id` is in NO part of
    the prompt. The model writes a name it invents and a syntax it invents, and it
    gets the tool NAME right each time. That shape says the model reads the user
    turn and does not read the `## Tools` section, and distance is not the reason.
  timestamp: 2026-08-14T18:35:22.009879+00:00
- actor: claude-code
  id: 01m00rxsvy2jws07d92dqwj3dd
  text: |
    ### implement — changed
    - evidence: 4 files —
      `Libraries/MLXLMCommon/DeepSeekV4Tokenization.swift` (new: the byte-level
      spelling, the three published `Split` patterns through `NSRegularExpression`,
      the byte-pair merge by lowest vocabulary identifier, and
      `DeepSeekV4Tokenization` which puts the marker step in front of them),
      `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift` (`SpecialToken.allMarkers`
      and `QuickInstructionTask: CaseIterable`),
      `Libraries/MLXLMCommon/Tokenizer.swift` (`DeepSeekV4EncodingTokenizer` reads
      through `DeepSeekV4Tokenization` in `encode(text:addSpecialTokens:)` and in
      `applyChatTemplate`), and
      `Tests/MLXLMTests/DeepSeekV4TokenizationTests.swift` (new: 10 weight-free
      tests).
      The version pin did not move, no package was forked or vendored, and no other
      model reads any changed line.
      `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test`: 1028 tests in
      108 suites, all passed, zero failures and zero new warnings.
      `xcodebuild build-for-testing` for `IntegrationTesting`: TEST BUILD SUCCEEDED,
      no new warning.
      `IntegrationTestingTests/DeepSeekV4TokenizerIntegrationTests`: 4 of 4 passed,
      `theToolPromptTokenizesToThePublishedIdentifiers()` included. It was red at
      353 identifiers and it is now green at 328, each equal to the fixture.
      The new splitter tests failed to compile before the correction landed and pass
      after it; the sharpest one holds the case of the card:
      `"given.\n\n## Tools\n\nYou have"` breaks into
      `["given", ".\n\n", "##", " Tools", "\n\n", "You", " have"]`.
      ONE real-weights run of
      `DeepseekV4IntegrationTests/aShortToolPromptEmitsOneDSMLToolCall()` FAILED:
      on the corrected 328 identifiers the model wrote `<functioncall>` and
      `{"name": "get_stock_level", "arguments": {"bay_id": "bay 7"}}`, byte
      identical to the answer it wrote on the wrong identifiers.
    - next: review. Every work item of this card is done. The card no longer blocks
      `^2dvj1g6`: the tokenizer is dead as a cause of the missing DSML tool call,
      and what is left there is the 4-bit quantization or a number in the
      DeepSeek-V4 attention port.
  timestamp: 2026-08-14T18:35:37.726828+00:00
position_column: doing
position_ordinal: '8280'
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

- [x] Choose the correction: send it upstream and move the pin, fork or vendor
      `swift-transformers` here, or give `DeepSeekV4EncodingTokenizer` its own
      pre-tokenization path — the user chose the third
- [x] Make the correction — `Libraries/MLXLMCommon/DeepSeekV4Tokenization.swift`
- [x] `theToolPromptTokenizesToThePublishedIdentifiers` passes — 328 identifiers,
      each equal to the fixture
- [x] Run `aShortToolPromptEmitsOneDSMLToolCall` again, and record what the
      model writes on a correct token sequence — it writes `<functioncall>`
      with plain JSON, thus the tokenizer was not the last cause

## Memory

The DeepSeek-V4 checkpoint holds 141 GiB. Run ONE real-weights test for each
process, or the machine runs out of memory. #deepseek-v4
