---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzrtg6qacbbtas9g2wg1pjep
  text: |-
    Research: the path that is not quantized in `KVCache.swift` never fills a scalar by itself. `createCausalMask` gives a bool mask and `MLXFast.scaledDotProductAttention` fills the masked positions inside mlx with `finfo(dtype).min`. To agree with that path, the quantized path must use the same value.

    mlx-swift already gives that value: `MLXArray.maskFill(for: DType)` in `MLXArray+maskFill.swift`. It returns `-finfo(dtype).max` built in the score dtype, thus it never overflows to `-infinity` in float16 or bfloat16. No file in this repository used it yet.

    `-Float.infinity` is NOT the correct fill. A query row whose positions are all masked (a padded row) would then hold only `-infinity`, and the softmax of such a row gives NaN. The finite minimum gives even weights instead. This is why mlx uses the finite minimum.

    All three sites in `KVCache.swift` (`.causal`, `.array`, `.arrays`) held the same defect and take the same correction. A single `let maskedFill = MLXArray.maskFill(for: scores.dtype)` before the switch now serves all three. No other site in the file holds the defect -- `git ls-files '*.swift' | xargs grep -n leastNormalMagnitude` gives zero hits in this file now.

    Found outside this file: `Libraries/MLXLLM/Models/Gemma3nText.swift` builds its sliding-window mask with the same wrong value. It is a different file, thus a new task ^9spp7b9 holds it. Line 834 of that same file uses `leastNormalMagnitude` as an epsilon, which is correct there.
  timestamp: 2026-08-11T16:29:13.834157+00:00
- actor: claude-code
  id: 01kzrth0sxp12fs6m92cycjktr
  text: |-
    ### implement — changed
    - files: `Libraries/MLXLMCommon/KVCache.swift`, `Tests/MLXLMTests/KVCacheTests.swift`
    - fix: one `let maskedFill = MLXArray.maskFill(for: scores.dtype)` before the mask switch now feeds all three sites, in place of `MLXArray(Float.leastNormalMagnitude)`.
    - test fixture: two key positions. The open key gives a score of 2. The masked key gives a score of 40, thus it wins the weight unless the mask stops it. Both value rows hold two elements only, thus the eight-bit affine round trip is exact and the first element of the answer is the weight the masked position kept.
    - before the fix: masked weight 0.11920288 for `.causal`, `.array` and `.arrays`. That is `1 / (e^2 + 1)`, the weight a fill of about zero gives.
    - after the fix: masked weight 0.0 for all three.
    - control: with no mask the same key wins 1.0 of the weight, thus the fixture measures the mask and not a small key.
    - mutation `MLXArray(Float.leastNormalMagnitude)`: `quantizedAttentionGivesAMaskedPositionNoWeight` (both cases) and `quantizedAttentionGivesAMaskedPositionNoWeightWithAListMask` fail, all at 0.11920288.
    - mutation `MLXArray(Float(0), dtype: scores.dtype)`: the same three named tests fail, all at 0.11920288. The test catches a zero fill.
    - mutation `MLXArray(-Float.infinity, dtype: scores.dtype)`: `quantizedAttentionKeepsAFullyMaskedRowFinite` fails on NaN. This is why the fill is the finite minimum and not `-infinity`.
    - `swift test`: 822 tests in 98 suites, zero failures. The only warning is the SwiftPM build-system line `missing creator for mutated node ... mlx-swift_Cmlx.bundle`, which comes from the dependency bundle and was there before this change.
    - note: the deprecated `.arrays` mask mode cannot be built without a deprecation warning, thus its test carries `@available(*, deprecated)`. The build is warning-free.
    - next: `/review`
  timestamp: 2026-08-11T16:29:40.541750+00:00
position_column: doing
position_ordinal: '80'
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

- [x] Each of the three sites uses a large negative fill value.
- [x] A test shows that a masked position gets almost no weight after the
      softmax. The test must fail against the value that is there now.
- [x] Each model that reads a quantized KV cache still gives the same answer for
      a prompt with no mask.

## Tests

- [x] A test that puts one masked position beside one open position and asserts
      the masked one holds less than 1e-6 of the weight.
- [x] `swift test --filter KVCacheTests`
- [x] Full `swift test` stays green.

#deepseek-v4
