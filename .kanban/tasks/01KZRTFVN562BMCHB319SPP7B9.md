---
assignees:
- claude-code
position_column: todo
position_ordinal: '9580'
title: Fix masked-position fill value in Gemma3n sliding-window mask
---
## What

`Libraries/MLXLLM/Models/Gemma3nText.swift` builds the sliding-window mask with

    let minDtype = MLXArray(Float.leastNormalMagnitude, dtype: maskArray.dtype)
    ...
    let updatedMask = MLX.where(slidingWindowMask, minDtype, maskArray)

`Float.leastNormalMagnitude` is **+1.175e-38**, the smallest positive normal
number. It is not a large negative number. A position that gets this value
keeps almost the same weight after the softmax as a score of zero, thus the
sliding window does not hold the far tokens back.

This is the same defect that task ^yjb4358 corrected in
`Libraries/MLXLMCommon/KVCache.swift`. Found while that task read the file.

## How to correct it

Use `MLXArray.maskFill(for: maskArray.dtype)` from mlx-swift, the same call
`quantizedScaledDotProductAttention` now uses. It gives the most negative
finite number of the dtype, which is what mlx itself fills a masked position
with.

Note: line 834 of the same file uses `Float.leastNormalMagnitude` as an
epsilon for a magnitude. That use is correct. Do not change it.

## Acceptance Criteria

- [ ] The sliding-window mask fills a held-back position with a large negative
      number.
- [ ] A test shows a held-back position keeps almost no weight. The test must
      fail against the value that is there now.
- [ ] Full `swift test` stays green.
#gemma3n