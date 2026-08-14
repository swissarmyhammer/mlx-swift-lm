---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyvnrbjwy58jy34679z21wb
  text: |-
    ### Closed — the official upstream corrected this

    The catch-up to `ml-explore/mlx-swift-lm` on 2026-08-13 brought two commits that answer this card:

    - `631325f` **Fix Python finfo.min port in quantized attention masking (#369)** — this is the very defect the card names. The card says the mask fills a held-back position with `Float.leastNormalMagnitude`, which is +1.175e-38 and not a large negative number.
    - `b207f60` **Fix Gemma 3n Boolean attention masks (#479)**

    **Verified in the code, not assumed.** `Libraries/MLXLLM/Models/Gemma3nText.swift` now holds one use of `Float.leastNormalMagnitude`, at line 813:

        let epsilonTensor = MLXArray(Float.leastNormalMagnitude, dtype: h0.dtype)

    That is the epsilon for a magnitude, and this card says in its own words that the use is correct and must not change. The mask path no longer holds it. The defect is gone.

    `swift test` is green: 1068 tests, 0 failures.

    No work of ours was needed. Upstream owns `Gemma3nText.swift` now, thus a later reader must read that file, not this card.
  timestamp: 2026-08-14T00:45:10.898169+00:00
position_column: done
position_ordinal: f680
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