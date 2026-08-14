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
position_column: todo
position_ordinal: '9780'
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

## The work

- [ ] Prove the cause. A unit test with the synthetic checkpoint can answer the
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