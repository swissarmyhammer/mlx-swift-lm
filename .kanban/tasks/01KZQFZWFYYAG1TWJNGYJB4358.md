---
assignees:
- claude-code
position_column: todo
position_ordinal: '9080'
title: Fix masked-position fill value in quantized KV cache attention
---
## What

`Libraries/MLXLMCommon/KVCache.swift:2038`, `:2042` and `:2051` fill a masked
position with `MLXArray(Float.leastNormalMagnitude)`.

`Float.leastNormalMagnitude` is **+1.175e-38**. It is the smallest positive
normal number. It is NOT a large negative number.

A masked position must go to a large negative number before the softmax, thus
`exp()` of it gives almost zero. A masked position that holds +1.175e-38 gives
`exp(0) = 1` after the softmax, which is the same weight as a score of 0. A
masked key thus keeps real weight in the answer.

The correct value is `-Float.greatestFiniteMagnitude`, or the `finfo(dtype).min`
that mlx itself uses in `fast.cpp`.

## Where this bites

Only the quantized key/value cache path, `quantizedScaledDotProductAttention`.
The path that is not quantized calls into mlx and is correct.

A person who turns on a quantized KV cache thus gets answers that read tokens
after the current one. The model still gives finite numbers, and the tests still
pass, thus nothing shows the error but the quality of the text.

## Where it came from

`git log -S` puts it in commit `37a8073`. It is older than the DeepSeek-V4 work.
A review of task ^ag7ant0 found it while it read the shared attention path.

## Acceptance Criteria

- [ ] Each of the three sites uses a large negative fill value.
- [ ] A test shows that a masked position gets almost no weight after the
      softmax. The test must fail against the value that is there now.
- [ ] Each model that reads a quantized KV cache still gives the same answer for
      a prompt with no mask.

## Tests

- [ ] A test that puts one masked position beside one open position and asserts
      the masked one holds less than 1e-6 of the weight.
- [ ] `swift test --filter KVCacheTests`
- [ ] Full `swift test` stays green.

#deepseek-v4
