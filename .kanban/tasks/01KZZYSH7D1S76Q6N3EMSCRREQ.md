---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzzyye0gb3qa5x3dqbt8bd4v
  text: |
    ### The cause is measured, not guessed

    Run of 2026-08-14, `aLongConversationMeasuresPromptCacheReuseAcrossTurns`:

    ```
    conversation follow-up extends round 1 = true
    conversation follow-up common prefix with round 1 = 3626 of 3626
    conversation round 1 divergent tail = <<<>>>
    conversation follow-up divergent tail = <<<</think>4172<｜end▁of▁sentence｜><｜User｜>Now say whether that bay is sealed. Reply in one>>>
    ```

    Read the last line. Round 1's render holds 3,626 tokens and the follow-up
    render writes `</think>` at position 3,626. The LEDGER writes round 1's first
    GENERATED token there, which is `4172`.

    `ExtendCachedPrefixRule` compares the new render against the ledger, and the
    ledger is `round 1's render + the tokens round 1 generated`. The two thus part
    by ONE token: the encoder adds `</think>` when it renders an assistant turn as
    history, and the live prompt of round 1 does not end with it.

    The fall-back is `RewindToCommonPrefixRule`, and it needs `cache.isTrimmable`.
    `DeepSeekV4Model.newCache` now gives a `RotatingKVCache(maxSize: 128)` to every
    layer with no compressor, and `RotatingKVCache.isTrimmable` is
    `offset < maxSize`. At 3,626 tokens that is false. Thus the policy falls to
    `.rebuild` and feeds all 3,646 tokens.

    ### Two independent corrections

    1. **The seam.** Make the live prompt of a chat-mode turn end with the same
       `</think>` the history render writes, or make the ledger hold it. Then
       `ExtendCachedPrefixRule` fires and NO trim is needed.
    2. **The trim.** Give the DeepSeek-V4 cache a rewind that works past the
       window, thus a broken seam is still rescued.

    Correction 1 is the cheaper one and it removes the need for correction 2 in
    this shape. Prove it with a unit test on `DeepSeekV4ChatEncoder` first: no
    weights are needed to compare the live render with the history render.
  timestamp: 2026-08-14T11:01:35.376992+00:00
- actor: claude-code
  id: 01m0375m94sp743bca4kxvh3pk
  text: |
    ### Research before the no-weights proof

    Read: `DeepSeekV4.newCache`, `DeepSeekV4Cache`, `DeepSeekV4ChunkCache`,
    `RotatingKVCache`, `PromptCacheReusePolicy`, `ChatSession` (the ledger), and
    `DeepSeekV4ChatEncoder`.

    **The trimmable half, read from the source.**

    - `DeepSeekV4SyntheticCheckpoint` has `compress_ratios: []` and no
      `sliding_window`, thus `slidingWindow` takes its default of 128 and every
      layer gets `RotatingKVCache(maxSize: 128, keep: 0)`. This is the shape the
      card names.
    - `RotatingKVCache.isTrimmable(after:)` is `offset + positions < maxCacheSize`.
    - `DeepSeekV4Cache.isTrimmable(after:)` is the window AND
      `attentionChunks.holdsEveryRewindableRow` AND the indexer branch.
      `holdsEveryRewindableRow` is `carryStart == 0`, and
      `DeepSeekV4ChunkCache.nextCarryStart` drops the raw rows as soon as
      `endPosition > rewindableTokenCount`, which is the same 128.

    Thus both cache kinds stop rewinding at the same position, by design. The
    header of `DeepSeekV4Cache.swift` states that design in words: the pool keeps
    every raw row "while `RotatingKVCache` still answers `isTrimmable`", and past
    that point it "holds the minimal carry and answers `isTrimmable` false".

    **The ledger, read from `ChatSession`.**

    `Conversation.record` appends only the generated tokens the cache really
    processed, thus after a turn `cachedTokens.count == processedTokenCount` and
    `mainCacheIsAligned` holds. The stop token is not fed back, thus the ledger
    ends one token before the render's end-of-sentence marker. A render that
    retokenizes the assistant text to the same identifiers the model sampled IS
    therefore a prefix extension of the ledger.

    **What the chat-mode log does NOT show.**

    The recorded run printed `follow-up extends round 1 = true`,
    `3626 of 3626`, and a follow-up divergent tail that starts with `</think>`.
    Read against the encoder, that tail can only start with `</think>` if the
    assistant TEXT starts with `</think>` -- in chat mode the encoder writes no
    `</think>` of its own in an assistant body. The earlier comment's claim that
    the ledger holds `4172` at position 3626 is an inference, not a printed
    number. The reuse DECISION was never printed. Thus the seam half is still
    unproven, which is exactly the open card item that needs the real weights.
  timestamp: 2026-08-15T17:23:03.076776+00:00
- actor: claude-code
  id: 01m037x6smm768dv6c08krx4sk
  text: |
    ### Discovery: a SECOND prefix break, in thinking mode, that needs no weights

    Read from `DeepSeekV4ChatEncoder.tail(of:isAtOrAfterLastUser:thinkingMode:dropsReasoning:)`:

    ```swift
    let opensReasoning =
        thinkingMode == .thinking && (!dropsReasoning || isAtOrAfterLastUser)
    return SpecialToken.assistant
        + (opensReasoning ? SpecialToken.thinkStart : SpecialToken.thinkEnd)
    ```

    In `thinking` mode with NO tool in the conversation, `dropsReasoning` is true.
    Thus:

    - Round 1 renders its own user turn as the LAST user turn, thus
      `isAtOrAfterLastUser` is true and the prompt ends `<｜Assistant｜><think>`.
    - Round 2 renders that same user turn as history, thus
      `isAtOrAfterLastUser` is false and the render writes
      `<｜Assistant｜></think>` at the SAME position.

    One token of the already-cached region is rewritten, thus round 1's render is
    NOT a prefix of round 2's render, and no reuse rule can fire. This is a
    different break from the one the card measured -- the card measured `chat`
    mode, where both renders write `</think>` and the render prefix IS good.

    This is a real second cause and it needs no weights to see. It is NOT part of
    this card's work list, thus no test of it landed here. It also may be by
    design: `removingEarlierReasoning` transcribes `encoding_dsv4.py`, and that
    reference deliberately rewrites the reasoning region of an earlier turn. A
    person must decide whether the prompt cache or the reference wins.
  timestamp: 2026-08-15T17:35:55.700435+00:00
- actor: claude-code
  id: 01m0385pr3ytbyexehq5hg9t0t
  text: |
    ### implement — stuck

    **Done: the no-weights proof (card item 1).**

    New file `Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift`, two tests
    on `DeepSeekV4SyntheticCheckpoint`.

    RED first. The first version asserted the reuse the card wants:

    ```
    ✘ Test aPromptCachePastTheSlidingWindowStillRewinds() recorded an issue at
      DeepSeekV4PromptCacheRewindTests.swift:73:9:
      Expectation failed: canTrimPromptCache(caches)
    ✘ Test run with 827 tests in 69 suites failed after 79.120 seconds with 1 issue.
    ```

    The two expectations before it PASSED: an empty cache rewinds, and every layer
    reached offset 200. Thus the failure is the rewind alone, not the fixture.

    GREEN after the tests record the measured fact:

    | Position | `canTrimPromptCache` |
    | --- | --- |
    | 0 tokens | true |
    | 127 tokens (window - 1) | true |
    | 200 tokens | false |

    The card's trimmable hypothesis is CONFIRMED, and the boundary is exactly the
    sliding window.

    **Not done: card item 3, "correct what the proof names". This is the blocker.**

    The proof names the rotating window, and that blocker cannot be corrected
    where it stands. `RotatingKVCache.isTrimmable` answers `offset < maxCacheSize`
    because past the window the ring HAS wrapped and the keys a rewind needs are
    overwritten; `DeepSeekV4ChunkCache` drops its raw rows at the same window on
    purpose, to hold its memory down; and the same limit belongs to every
    sliding-window model of this repository, thus it is a shared contract rather
    than a DeepSeek-V4 defect. No change to the gate brings the keys back.

    The three corrections that remain are recorded in the description
    (A keep the keys, B close the seam with a `PromptCacheReuseRule`, C accept no
    reuse). Each changes a contract this card does not record, and B needs the
    reuse decision of a real two-round run first. A person must choose.

    **Left alone on purpose.** The two fact-(b) assertions of
    `aLongConversationMeasuresPromptCacheReuseAcrossTurns` still say the follow-up
    skips nothing. Nothing in this change moves that number, thus they stay as
    they are.

    - evidence: 1 file added, `Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift`.
      Five bundles, 0 failures: MLXLMTests 475 XCTest + 828 Swift Testing,
      MLXGuidedGenerationTests 70, MLXFoundationModelsTests 141, CXGrammarTests 7,
      MLXHuggingFaceMacrosTests 5 XCTest. 480 XCTest + 1046 Swift Testing = 1526,
      against a baseline of 1524. `xcodebuild build-for-testing` on the package and
      on `IntegrationTesting` both report TEST BUILD SUCCEEDED. `swift-format lint`
      is clean.
    - next: a person picks correction A, B or C, then runs the two real-weights
      items with the command in the description.
  timestamp: 2026-08-15T17:40:34.179775+00:00
- actor: claude-code
  id: 01m05n54jpcs1s538bz7bky8wh
  text: |
    ### Correction B is still needed: `^z1j7f67` does NOT cover it

    Read `Libraries/MLXFoundationModels/ExecutorPromptCache.swift` and the two new
    `package` functions of `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`.

    `^z1j7f67` holds the ledger to the rendered PROMPT alone, and it pays for that
    with a REWIND after each generation:

    ```swift
    func committed() -> ExecutorPromptCacheEntry? {
        guard rewindPromptCache(caches, to: promptTokens.count) else { return nil }
        return ExecutorPromptCacheEntry(caches: caches, tokens: promptTokens)
    }
    ```

    `rewindPromptCache` answers false when a cache does not land on the position it
    asks for, and its own documentation states the outcome: "A cache past its
    sliding window cannot rewind, thus a sliding-window model takes the nil
    answer."

    DeepSeek-V4 IS that model. This card already measured `canTrimPromptCache` false
    past the 128-token window with no weights. Thus, on that design, each
    DeepSeek-V4 turn ends with `committed() == nil` and the next turn starts cold.
    The seam closes and the reuse still does not appear.

    Two more facts separate the two paths:

    1. `ExecutorPromptCache` serves `MLXLanguageModel.Executor`. The measurement of
       this card runs through `MLXLMCommon.ChatSession`, which keeps its own ledger
       (`Conversation.cachedTokens`) of the prompt PLUS the tokens the turn
       generated.
    2. `reusablePromptPrefix` states in words that "It consults no protocol rule",
       thus a `PromptCacheReuseRule` never reaches the executor path at all.

    Correction B is therefore a second answer to a second question, not a second
    mechanism for the same one. A sliding-window cache that cannot rewind can reuse
    only when the tokens it already holds ARE a prefix of the next render. That is
    what a DSML rule has to establish.

    ### The seam still has to be measured first

    The remaining unknown is one question: are the tokens the model sampled in
    round 1 the same tokens the encoder writes when it renders that same turn as
    history? Fact (a) is measured -- the two RENDERS nest perfectly, 3626 of 3626 --
    thus the render is not the cause. The next step is the real two-round run with
    the reuse decision printed.
  timestamp: 2026-08-16T16:05:55.926461+00:00
- actor: claude-code
  id: 01m05p3c1k8ey6s88z21jwegxq
  text: |
    ### The measurement, and what it changed

    **Card item 2, the reuse decision of a real two-round run.** A temporary print
    went beside `promptCachePolicy.decide` in `ChatSession.swift`, the ONE test ran
    in its own process, and the print was then reverted:

    ```
    DSV4 SEAM: decision = appendSuffix(suffixStart: 3509, ...);
        prompt = 3525; ledger = 3509; processed = 3509;
        aligned = true; trimmable = false; seam = 3509
    DSV4 SEAM: ledger tail = []
    ```

    The seam stands at 3509 of a 3509-token ledger, thus the ledger is a WHOLE
    prefix of the new render and nothing parts. `ExtendCachedPrefixRule` fires.

    **Card item 3, correct what the proof names.** No new mechanism landed, and the
    card's own three corrections are each unnecessary. The proof named the rotating
    window; the measurement says the window is not what this card pays for, because
    `trimmable = false` stands beside a decision that uses no rewind. Two commits
    that landed after the 2026-08-14 measurement closed the seam: `3c301af` gave
    DeepSeek-V4 its own pre-tokenization path, and `851e224` installed that path at
    load. Before `851e224`, `LLMModelFactory._load` never called
    `LLMModel.promptTokenizer(wrapping:)`, thus the new path reached no loaded
    model. Both files of this card now record the measured behavior.

    **Card item 4, invert the two fact-(b) assertions.** Done, and each one now
    reads the way `expectAgenticRound` already reads: `skippedTokenCount > 0` and
    `prefillSeconds < control * 0.8`.

    ### What did not need to be built

    Correction B -- a DSML `PromptCacheReuseRule` -- was checked first, as asked.
    `^z1j7f67`'s `ExecutorPromptCache` does NOT cover it: that design rewinds to the
    prompt length after each generation, and the rewind does not land past a sliding
    window, thus `committed()` answers nil and each turn starts cold. But the
    measurement then made correction B unnecessary on its own: a rule can only help
    where the prefix breaks, and the prefix does not break.

    The executor gap is real and it belongs to a different code path. Card
    `^2nztex1`, "The executor prompt cache gives a sliding-window model no reuse",
    holds it.

    ### The thinking-mode break stays open

    The break of the 2026-08-15 comment -- `<｜Assistant｜><think>` live against
    `<｜Assistant｜></think>` as history -- is a turn whose prefix DOES break, and
    past the window there is no rewind to rescue it. It was not on this card's work
    list, thus no change of this card touches it.
  timestamp: 2026-08-16T16:22:26.611050+00:00
- actor: claude-code
  id: 01m05p3mvcegh9e532mvq5ee7x
  text: |
    ### implement — changed

    - evidence: 2 files —
      `IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift`,
      `Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift`.
      Five package bundles, 0 failures and 0 skipped: MLXLMTests 475 XCTest + 846
      Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 155,
      CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 XCTest = 480 XCTest + 1078
      Swift Testing. `xcodebuild build-for-testing` on the package and on
      `IntegrationTesting` each report TEST BUILD SUCCEEDED, with no warning but
      the AppIntents metadata line. `swift-format lint --strict` is clean on both
      changed files.
      The real-weights test passes with the inverted assertions, one test in one
      process: round 1 renders 3506 and feeds 3506 in 11.92 s; the follow-up
      renders 3525, feeds 16, skips 3509 and takes 1.99 s; the cold control feeds
      3525 in 12.72 s.
    - next: `/review`. Card `^2nztex1` holds the executor gap this work found.
  timestamp: 2026-08-16T16:22:35.628015+00:00
position_column: doing
position_ordinal: '8480'
title: DeepSeek-V4 gets a good prefix and still reprocesses the whole prompt
---
Measured against `mlx-community/DeepSeek-V4-Flash-4bit` with the real weights,
by
`DeepseekV4AgenticPromptCacheAssessmentTests.aLongConversationMeasuresPromptCacheReuseAcrossTurns`.

## The numbers of 2026-08-14, which opened this card

| Pass | Rendered | Fed | Skipped | Prefill |
| --- | --- | --- | --- | --- |
| round 1 | 3626 | 3626 | 0 | 12.93 s |
| follow-up | 3646 | 3646 | 0 | 13.52 s |
| cold control | 3646 | 3646 | 0 | 14.18 s |

The follow-up render IS a true prefix extension of round 1's render. Thus
DeepSeek-V4 clears fact (a), which Qwen-3.6 fails (`^2ajc82t`, abandoned), and
it still saved no work.

## The numbers of 2026-08-16, from the same test

| Pass | Rendered | Fed | Skipped | Prefill |
| --- | --- | --- | --- | --- |
| round 1 | 3506 | 3506 | 0 | 11.93 s |
| follow-up | 3525 | **16** | **3509** | **2.00 s** |
| cold control | 3525 | 3525 | 0 | 12.77 s |

The reuse works. The follow-up round feeds 16 tokens of its 3525, and it is 6.4
times faster than the cold control.

## The reuse decision, printed

A temporary print beside `promptCachePolicy.decide` in `ChatSession.swift` gave
the four numbers this card asked for. The print was removed after the run:

```
DSV4 SEAM: decision = appendSuffix(suffixStart: 3509, ...);
    prompt = 3525; ledger = 3509; processed = 3509;
    aligned = true; trimmable = false; seam = 3509
DSV4 SEAM: ledger tail = []
```

- `ledger = 3509` is round 1's render of 3506 tokens PLUS the 3 generated
  tokens the cache really processed.
- `seam = 3509` says the two part at NO position inside the ledger, thus the
  ledger is a WHOLE prefix of the new render and the ledger tail is empty.
- `ExtendCachedPrefixRule` therefore fires, and it feeds the 16 new tokens.
- `trimmable = false` says the `RotatingKVCache` past its 128-token window
  still does not rewind. No rewind takes part in this decision.

## What closed the seam

Two commits, and the second one is what made the first take effect:

1. `3c301af`, `fix(mlx-lm): give DeepSeek-V4 its own pre-tokenization path`, at
   13:36 on 2026-08-14.
2. `851e224`, `fix(mlx-llm): install the model's prompt tokenizer at load`, at
   10:47 on 2026-08-16. `LLMModelFactory._load` never called
   `LLMModel.promptTokenizer(wrapping:)` before it, thus the new path reached
   no loaded model.

The measurement that opened this card was taken at 06:01 on 2026-08-14, thus it
precedes both.

`SplitPreTokenizer` of `swift-transformers` 1.3.3 sends its pattern through
`String.range(of:options:.regularExpression)`, and that search cannot match
`\r` or `\n` inside a character class. Each newline thus became its own piece.
The prompt of this test holds 120 report rows, one on each line, which is
exactly the 120 tokens each render lost. The tokens the model samples and the
tokens the encoder writes when it renders that same turn as history now agree,
thus the ledger is a whole prefix again.

## Corrections A, B and C are each unnecessary

This card recorded three corrections a person had to choose between. The
measurement answers all three:

- **A. Keep the keys.** Nothing needs the rewind that correction A restores.
  The prefix holds, thus `RewindToCommonPrefixRule` is never reached.
- **B. Close the seam with a `PromptCacheReuseRule`.** The seam is already
  closed. A DSML rule would be a second mechanism for a question that has no
  open half.
- **C. Accept no reuse.** The number refutes it.

Correction B was also checked against `^z1j7f67`, which landed
`ExecutorPromptCache`. That design holds its ledger to the rendered prompt
alone and rewinds to the prompt length after each generation, thus it cannot
serve a sliding-window model either -- the rewind does not land. It is a
different code path (`MLXLanguageModel.Executor`, not `ChatSession`), and card
`^2nztex1` holds that gap.

## What still has no rewind, on purpose

`Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift` measures, with NO
weights, that `canTrimPromptCache` is true at 0 tokens, true at 127 tokens and
false at 200 tokens. That measurement stands. A turn whose prefix BREAKS still
pays the whole prefill, because the fall-back rewind is gone past the window.
The thinking-mode break of the comment thread is one such turn, and it is not
part of this card.

## The work

- [x] Prove the cause. A unit test with the synthetic checkpoint answers the
      trimmable half with NO weights: build `newCache`, feed 200 tokens, and
      read `canTrimPromptCache`. It answers false at the sliding window
- [x] Print the reuse decision of a real two-round run, thus the seam half is
      measured and not guessed. The decision is
      `appendSuffix(suffixStart: 3509)` and the seam stands at 3509 of a
      3509-token ledger
- [x] Correct what the proof names. The proof names the rotating window, and
      the measurement says the window is not what this card pays for: commits
      `3c301af` and `851e224` closed the seam, thus the reuse needs no rewind.
      No new mechanism landed, and the record of both files now states the
      measured behavior
- [x] Invert the two fact-(b) assertions of
      `aLongConversationMeasuresPromptCacheReuseAcrossTurns` and record the new
      numbers

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process, or
the machine runs out of memory. #deepseek-v4 #performance
