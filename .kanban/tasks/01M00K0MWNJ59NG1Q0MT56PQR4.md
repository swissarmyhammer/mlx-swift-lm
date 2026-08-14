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
- actor: claude-code
  id: 01m00t4hdwgt0b9gjg71m52cay
  text: |
    Two review findings on `Libraries/MLXLMCommon/DeepSeekV4Tokenization.swift`
    are corrected. Both are requirements, and each cause is removed from the WHOLE
    file and not from the named line alone.

    ## Finding 1 — `swift/naming-clarity`

    The rule states "Clarity over brevity", "Omit needless words. Every word must
    carry salient information at the use site", "Name by role, not type" and
    "Compensate for weak type information". I read every local name of the file
    against those four lines and corrected each cryptic one:

    | was | is now | why |
    | --- | --- | --- |
    | `wide` | `textAsNSString` | the named line. `text` is the role, `AsNSString` is the salient difference from the `text` parameter |
    | `start` | `nextPartStart` | a bare `Int` offset. The noun says which start |
    | `moved` | `nextMovedCodePoint` | an adjective stood for a `UInt32` code point |
    | `spelled` | `spelledText` | an adjective stood for a `String` |
    | `next` | `nextPieces` | an adjective stood for a `[String]` |
    | `lowest` | `lowestIdentifierSoFar` | lowest WHAT was not said |
    | `alternatives` | `escapedMarkers` | says what the parts are, not that they are branches |
    | `marker` | `markerIdentifier` | it holds an `Int`, not a marker text. This is the weak-type rule word for word |
    | `result` (2 places) | `allIdentifiers` | `result` carries no salient information |
    | `identifiers` (inner, 2 places) | `partIdentifiers`, `pieceIdentifiers` | each says which identifiers |

    `gap`, `parts`, `piece`, `index`, `identifier`, `isByteLevel` and `expression`
    each name a role already, thus they stay.

    `textAsNSString` does not fight "Name by role, not type": the role is the text,
    and the representation is the salient difference from the `text` parameter that
    stands beside it.

    ## Finding 2 — `completeness/public-output-contract`

    A marker that the vocabulary does not hold went to `segmentIdentifiers` and
    became ordinary text. The type contract states that each marker becomes exactly
    ONE identifier, thus that answer was wrong and it looked correct. The
    byte-level path now reports the failure, exactly as the finding states.

    Measured before the correction, on a vocabulary that holds every byte character
    of `<｜User｜>` and no marker:

    ```
    identifiers(of: marker + "a") -> [700, 708, 709, 710, 704, 705, 706, 707,
                                      708, 709, 710, 711, 65]
    ```

    13 identifiers where the contract states 2. After the correction the whole text
    goes to the wrapped tokenizer, which is the documented answer of this type for
    a tokenizer that is not the published one.

    I read the whole file for other places that degrade in silence. There is one
    more, and it must STAY:

    - `identifiers(of:)` gives the whole text to the wrapped tokenizer when
      `byteLevelIdentifiers` answers `nil`. That is the documented contract of the
      type, the same validator forbids turning it into a hard failure (check 4, "A
      previously-succeeding path now fails hard"), and the correction that the
      finding prescribes (`isByteLevel = false; return`) REACHES that fallback. It
      is the reporting path, not a degradation.

    These are not degradations, thus they stay unchanged:

    - `readParts` skips a zero-length match. No text is lost: the gap logic still
      gives every character to the reader.
    - `DeepSeekV4BytePairMerge.identifiers` breaks the merge loop when no pair is
      left, then answers `nil` for a part the vocabulary does not hold.
    - `compiledExpression` and `scalar(of:)` stop with `preconditionFailure`. Both
      are programmer error, which `swift/error-handling` permits.

    The type doc comment, the `identifiers(of:)` comment and the
    `byteLevelIdentifiers` `- Returns:` line now all state the marker failure, thus
    the contract is written down.

    ## Memory

    NO real-weights test ran for this work. Both findings are provable with
    weight-free tests alone, and the 141 GiB checkpoint stayed on disk.
  timestamp: 2026-08-14T18:56:47.036413+00:00
- actor: claude-code
  id: 01m00t4z40v6wft8tq53ynzvhs
  text: |
    ### implement — changed
    - evidence: 2 files —
      `Libraries/MLXLMCommon/DeepSeekV4Tokenization.swift` (finding 1: 11 cryptic
      local names corrected over the whole file, `wide` -> `textAsNSString`
      included; finding 2: a marker that the vocabulary does not hold now stops the
      byte-level path, and the type doc comment states that contract), and
      `Tests/MLXLMTests/DeepSeekV4TokenizationTests.swift` (a new weight-free test
      `aMarkerOutsideTheVocabularyStopsTheByteLevelPath`, with the
      `vocabularyWithoutTheMarker` fixture that holds every byte character of the
      marker and no marker).
      The new test was RED first, and for the correct reason: it answered
      `[700, 708, 709, 710, 704, 705, 706, 707, 708, 709, 710, 711, 65]` — 13
      identifiers where the contract states 2, because the marker went through the
      ordinary-text path. It is green after the correction.
      `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test`: 1029 tests
      (814 + 69 + 139 + 7) in 108 suites, all passed. That is 1028 before plus the
      new one. Zero failures. The only warnings are the three package build-graph
      warnings that stand on `main`.
      `xcodebuild build-for-testing` for `IntegrationTesting`: TEST BUILD
      SUCCEEDED, no new warning.
      ACCEPTANCE:
      `IntegrationTestingTests/DeepSeekV4TokenizerIntegrationTests/theToolPromptTokenizesToThePublishedIdentifiers()`
      passed, thus the 328 identifiers each stay equal to the fixture. The whole
      `DeepSeekV4TokenizerIntegrationTests` suite passed, 4 of 4.
      `swift-format` with `.swift-format`: both files need no change.
      The only production caller is `DeepSeekV4EncodingTokenizer` of
      `Libraries/MLXLMCommon/Tokenizer.swift`, which wraps the checkpoint
      tokenizer. That tokenizer holds every marker, which the 328-identifier
      acceptance test measures, thus the new failure path never runs on the real
      checkpoint and the answer for real work does not move.
      NO real-weights test ran. The 141 GiB checkpoint stayed on disk.
    - next: review.
  timestamp: 2026-08-14T18:57:01.056681+00:00
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
