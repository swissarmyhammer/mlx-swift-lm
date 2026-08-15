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
- actor: claude-code
  id: 01kzrvdz0q9tjxgwgswfczp7pz
  text: |-
    ### review — clean

    - scope: `review sha HEAD~1..HEAD` (commit 2a6dd58), branch foundationmodels-fixes
    - engine counts: 139 findings, 151 confirmed, 12 refuted, 8 validators attempted, 0 failed, 0 skipped files
    - in-scope findings: 0

    **Lines this commit changed** in `Libraries/MLXLMCommon/KVCache.swift`: 2030-2035 (comment and the hoisted `let maskedFill = MLXArray.maskFill(for: scores.dtype)`), 2043, 2047, 2056. No engine finding falls on these lines. The two findings nearest the change are `KVCache.swift:2038` and `KVCache.swift:2042`, both "Magic numbers should be replaced by named constants" against `scores.dim(-2)` and `axis: -2`. Both lines are unchanged by this commit.

    **Out of scope — 139 findings, all on lines this commit did not touch.** Reason for each: the line is unchanged by commit 2a6dd58. No action taken.

    - Missing doc comments on public/open declarations — 44 findings: lines 11, 12, 108, 112, 114, 116, 119, 184, 187, 191, 200, 209, 212, 214, 218, 220, 222, 242, 312, 371, 383, 520, 536, 777, 833, 834, 835, 837, 1123, 1128, 1218, 1305, 1314, 1423, 1438, 1987.
    - Magic numbers should be replaced by named constants — 89 findings: lines 397 through 2153, including 1999, 2000, 2012, 2013, 2014, 2017, 2018, 2019, 2038, 2042, 2092, 2093, 2151, 2152, 2153.
    - `KVCacheError` is internal but is thrown from public functions — line 1574.
    - `var.instance message` is assignOnlyProperty — line 1575.
    - Cognitive complexity 17 in `unflattenArrays` — line 1809.
    - Cognitive complexity 26 in `unflattenMetadata` — line 1849.
    - Scheme switch statement should be a dictionary — line 2090.

    **Engine limits.** The engine read one file only: `Libraries/MLXLMCommon/KVCache.swift`. All 139 findings are against that file. The 159 added lines of `Tests/MLXLMTests/KVCacheTests.swift` got no validator, because the engine excludes test files. The `.kanban/*.md` and `.kanban/*.jsonl` files in the commit got no validator, because no validator globs those types. `attempted: 8` counts validators, not files.

    **Independent checks — all three pass.**

    1. `MLXArray.maskFill(for:)` truly gives `finfo(dtype).min`. Definition at `.build/checkouts/mlx-swift/Source/MLX/MLXArray+maskFill.swift:8`, body `return -dtype.greatestFiniteMagnitudeArray`. In `DType.swift`, `FInfo.min` is itself defined as `-max`, thus the two agree by construction. The value is built in the target dtype, thus float16 and bfloat16 get their own narrower minimum, not the float32 value. The checkout is clean and the code is upstream (mlx-swift commit bb0399d). The premise of the fix holds.
    2. All three mask modes changed, and the non-bool branches did not. `.causal` at 2043, `.array` bool at 2047, `.arrays` bool at 2056 all read the hoisted `maskedFill`. The `else` branches at 2049 and 2058 keep `scores = scores + maskArray` unchanged. `KVCache.swift` shows 13 changed lines only; no other cache behaviour changed.
    3. Mutation proof. `MLXArray(Float.leastNormalMagnitude)` put back at lines 2043, 2047, 2056 makes three tests fail at exactly 0.11920288: `quantizedAttentionGivesAMaskedPositionNoWeight(.causal)`, `quantizedAttentionGivesAMaskedPositionNoWeight(.boolArray)`, and `quantizedAttentionGivesAMaskedPositionNoWeightWithAListMask()`. After `git checkout`, `git status --porcelain -- '*.swift'` gives no output, and `swift test --filter KVCacheTests` passes 39 tests plus 1.

    - next: none. The change is confined to the masked-fill defect. Task moves to done.
  timestamp: 2026-08-11T16:45:28.983707+00:00
- actor: claude-code
  id: 01kzrvf13116f30xhp0627jhbg
  text: |
    ### finish iteration 1 — clean
    - implement: changed — KVCache.swift (13 lines) and KVCacheTests.swift (+159); all three mask sites now share `MLXArray.maskFill(for: scores.dtype)`
    - test: green — swift test full suite, 822 tests in 98 suites, 0 failures
    - commit: 2a6dd58
    - review: clean — 0 in-scope findings, card moved to done
    - measurement: a masked position held **0.11920288** of the softmax weight before, and **0.0** after. The control with no mask gives 1.0, thus the fixture measures the mask and not a small key.
    - note: `-Float.infinity` is WRONG here and the test proves it. That mutant kills `quantizedAttentionKeepsAFullyMaskedRowFinite` with NaN. mlx uses the finite dtype minimum for the same reason.
    - note: the review confirmed the premise from the vendored source. `MLXArray+maskFill.swift:8` gives `-dtype.greatestFiniteMagnitudeArray`, and `FInfo.min` is `-max`, thus the two agree. The array takes the target dtype, thus float16 and bfloat16 each get their own narrower minimum.
    - note: 139 engine findings all landed on untouched lines of this shared file. Each one is recorded as out of scope. This matches the 108 out-of-scope findings the same file gave on `^ag7ant0`.
    - follow-on: `Libraries/MLXLLM/Models/Gemma3nText.swift` holds the same defect in its sliding-window mask. Filed as `^9spp7b9`. Line 834 of that file uses the same constant as a real epsilon and is correct.
  timestamp: 2026-08-11T16:46:03.873222+00:00
position_column: done
position_ordinal: e780
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
