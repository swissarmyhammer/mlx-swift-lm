---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: The executor prompt cache gives a sliding-window model no reuse
---
`Libraries/MLXFoundationModels/ExecutorPromptCache.swift` landed with card
`^z1j7f67`. It holds its ledger to the rendered PROMPT of the last turn alone,
and it pays for that with a REWIND after each generation:

```swift
func committed() -> ExecutorPromptCacheEntry? {
    guard rewindPromptCache(caches, to: promptTokens.count) else { return nil }
    return ExecutorPromptCacheEntry(caches: caches, tokens: promptTokens)
}
```

`rewindPromptCache` answers false when a cache does not land on the position it
asks for. Its own documentation states the outcome: "A cache past its sliding
window cannot rewind, thus a sliding-window model takes the nil answer."

## Why this is a defect

Every sliding-window model of this repository takes that nil answer -- DeepSeek-V4,
Gemma 3/4, GPT-OSS, Exaone4 and Mistral3 all take a `RotatingKVCache`. Each turn
of such a model therefore ends with `committed() == nil`, and the next turn of the
same session starts cold. The executor gives them NO prompt-cache reuse at all.

Card `^mscrreq` measured what the OTHER path does for the same model on the same
day. `MLXLMCommon.ChatSession` keeps a ledger of the render PLUS the tokens the
turn generated, thus `ExtendCachedPrefixRule` fires with NO rewind:

```
DSV4 SEAM: decision = appendSuffix(suffixStart: 3509, ...);
    prompt = 3525; ledger = 3509; processed = 3509;
    aligned = true; trimmable = false; seam = 3509
```

Read `trimmable = false` beside `seam = 3509`. The cache does not rewind and it
does not have to, because the tokens it already holds ARE a prefix of the next
render. That measurement is the evidence that a rewind is not the only way to
carry a cache from one turn to the next.

## The work

- [ ] Prove the gap with the synthetic checkpoint and NO weights: build
      `DeepSeekV4Model.newCache`, feed more tokens than the sliding window, and
      read `rewindPromptCache(caches, to:)`
- [ ] Choose how the executor carries a cache a sliding-window model cannot
      rewind. The `ChatSession` answer -- a ledger of the render plus the
      committed generated tokens -- needs no rewind
- [ ] Measure the correction with the real weights on
      `mlx-community/DeepSeek-V4-Flash-4bit`, through the
      `MLXFoundationModels` executor and not through `ChatSession`

## Memory

The DeepSeek-V4 checkpoint holds 141 GiB. Run ONE real-weights test for each
process, or the machine runs out of memory.
#deepseek-v4 #performance