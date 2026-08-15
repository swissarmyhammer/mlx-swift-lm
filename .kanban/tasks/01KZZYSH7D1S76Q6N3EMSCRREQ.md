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
position_column: doing
position_ordinal: '8480'
title: DeepSeek-V4 gets a good prefix and still reprocesses the whole prompt
---
Measured on 2026-08-14 against `mlx-community/DeepSeek-V4-Flash-4bit` with the
real weights, by
`DeepseekV4AgenticPromptCacheAssessmentTests.aLongConversationMeasuresPromptCacheReuseAcrossTurns`.

## The numbers

| Pass | Rendered | Fed | Skipped | Prefill |
| --- | --- | --- | --- | --- |
| round 1 | 3626 | 3626 | 0 | 12.93 s |
| follow-up | 3646 | 3646 | 0 | 13.52 s |
| cold control | 3646 | 3646 | 0 | 14.18 s |

The follow-up render IS a true prefix extension of round 1's render. Thus
DeepSeek-V4 clears fact (a), which Qwen-3.6 fails (`^2ajc82t`, abandoned), and
it still saves no work.

## Why this matters

An agent pays the whole prefill again at every turn. On a 3.6k-token
transcript that is 13 s for each turn, and the cost grows with the transcript.
Requirement 3 of the agentic goal rides on this number.

## The likely cause, not yet proven

`ExtendCachedPrefixRule` needs `turn.promptTokens.starts(with:
cache.cachedTokens)`, and `cachedTokens` holds the round-1 prompt PLUS the
tokens round 1 generated. The render of the assistant turn need not tokenize
into the same ids the model sampled, thus a seam of a few tokens can break the
prefix. The fall-back is `RewindToCommonPrefixRule`, which needs
`cache.isTrimmable`, and:

- Before the sparse path, all 43 layers were `KVCacheSimple`, thus all rewound.
- `DeepSeekV4Model.newCache` now gives `RotatingKVCache(maxSize: 128)` to a
  layer with no compressor and `DeepSeekV4Cache` to the rest.
  `RotatingKVCache.isTrimmable` is `offset < maxSize`, thus a cache past 128
  tokens does NOT rewind.

Thus the rewind that used to rescue a broken seam is no longer available.

## What the no-weights proof showed (2026-08-15)

`Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift` builds `newCache`
from `DeepSeekV4SyntheticCheckpoint` and reads `canTrimPromptCache`:

| Position | `canTrimPromptCache` |
| --- | --- |
| 0 tokens | true |
| 127 tokens (window - 1) | true |
| 200 tokens | **false** |

The trimmable half of the card is thus CONFIRMED, and the boundary is exactly
the sliding window. The seam half stays unproven: the recorded run printed the
two RENDERS and never printed the reuse decision or the ledger.

## The blocker on "correct what the proof names"

The proof names the rotating window, and that blocker cannot be corrected
where it stands:

1. `RotatingKVCache` answers `offset < maxCacheSize` because past the window
   the ring HAS wrapped and the keys a rewind needs are overwritten. A rewind
   there gives a window that is short at its old end, and `temporalOrder`
   reads the stale slots as the oldest rows. The keys are gone; no gate change
   brings them back.
2. `DeepSeekV4ChunkCache` drops its raw rows at the same window ON PURPOSE, to
   hold its memory down. Keeping every row costs `tokens * hiddenSize` for
   each branch of each layer, which is several GiB on a 3.6k-token
   transcript.
3. The same limit belongs to EVERY sliding-window model of this repository --
   Gemma 3/4, GPT-OSS, Exaone4, Mistral3 -- because they all take
   `RotatingKVCache`. It is a shared contract, not a DeepSeek-V4 defect.

A person must choose between three corrections, and none of them is recorded
on this card:

- **A. Keep the keys.** Give the window layers a cache that holds every key
  and masks to the window. It restores the rewind and it costs memory.
- **B. Close the seam.** Give DeepSeek-V4 a `PromptCacheReuseRule`, as
  `.gptOSS` and `.atem` already do in `ToolCallFormat.promptCacheReuseRules`,
  so `ExtendCachedPrefixRule` fires and no rewind is needed. This needs the
  reuse decision of a real run FIRST, which is the unchecked item below.
- **C. Accept no reuse** for DeepSeek-V4 and record it.

## What a person must run to finish the two real-weights items

The checkpoint holds 141 GiB. Run ONE test for each process.

```sh
xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
    -scheme IntegrationTesting -destination 'platform=macOS' \
    -only-testing:IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests/aLongConversationMeasuresPromptCacheReuseAcrossTurns
```

Add a print of `decision` beside `promptCachePolicy.decide` in
`ChatSession.swift` first, and print `cachedTokenIds.count`,
`kvCache.processedTokenCount` and the first position at which
`promptTokenIds` and `cachedTokenIds` part. Those four numbers name the seam.

## The work

- [x] Prove the cause. A unit test with the synthetic checkpoint can answer the
      trimmable half with NO weights: build `newCache`, feed 200 tokens, and
      read `canTrimPromptCache`
- [ ] Print the reuse decision of a real two-round run, thus the seam half is
      measured and not guessed
- [ ] Correct what the proof names
- [ ] Invert the two fact-(b) assertions of
      `aLongConversationMeasuresPromptCacheReuseAcrossTurns` and record the new
      numbers

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process, or
the machine runs out of memory. #deepseek-v4 #performance