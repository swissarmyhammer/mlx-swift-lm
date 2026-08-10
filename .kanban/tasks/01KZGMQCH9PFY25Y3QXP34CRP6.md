---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp2mzv07wsh28tgbgmv28wh
  text: |
    ### Research

    **Port source.** `osaurus-ai/vmlx-swift-lm` @ `b166896353b9c95d773de993990c20a0b5ba6905` holds `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` (452 lines). `scouzi1966/mlx-swift-lm` @ `e1852869ce61ded0d23b76df3757e9b75c77c1f5` holds a later 937-line copy of the same file with the same `Copyright (c) 2026 Osaurus AI` header. `CONTRIBUTING.md` gives the DeepSeek-V4 copyright to Osaurus AI, thus the header block names the osaurus-ai path and SHA.

    **Numeric reference.** `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py`:
    - `hc_split_sinkhorn` lines 171-214
    - `DeepseekV4RoPE.__init__` lines 102-134 (YaRN inverse frequency)
    - `DeepseekV4RoPE.__call__` lines 145-164 (partial RoPE, forward and inverse)
    - `_score_func` lines 307-313 (sqrtsoftplus)
    - `_swiglu_limited` lines 367-371 (clamped SwiGLU)

    **The two Swift copies disagree on YaRN, and the Python decides it.** osaurus gives `low` to `betaSlow` and `high` to `betaFast`. scouzi reverses that and says osaurus is wrong. The Python (`low = floor(correction_dim(beta_fast))`, `high = ceil(correction_dim(beta_slow))`) agrees with scouzi, and so does this repository's own `YarnRoPE` in `Libraries/MLXLMCommon/RoPEUtils.swift`. The port follows the Python.

    **A second divergence, on the degenerate range.** Both Swift copies write `rangeWidth = max(high - low, 0.001)`. The Python writes `if low == high { high += 0.001 }` and divides by `high - low`. The two differ exactly where the `high = min(..., dim - 1)` clamp (Bug 10) bites: after the clamp `high` can fall below `low`, and only the Python rule shows the clamp in the output. Measured on `dim=8, base=10, origMaxPos=1e9, factor=4`: unclamped `high` is 33, the clamp makes it 7 against `low` 26, and the Python rule gives `invFreq / 4` while a missing clamp gives `invFreq`. That is a 4x separation, thus the clamp test is not vacuous. The port follows the Python.

    **The card and the reference disagree on the SwiGLU clamp.** The card says "clamp gate and up to +/-10". Every reference -- the Python `_swiglu_limited`, the osaurus Swift, and the scouzi Swift -- clamps `up` on both sides and `gate` on the high side only (`minimum(gate, limit)`). The card also says each function must match the Python reference output. The port follows the reference and the test fixture records the asymmetry: at `gate = -50` the output is `9.64e-20`, at `gate = -10` it is `4.54e-3`, thus a symmetric clamp fails the test.

    **Access level.** `DeepseekV3.swift` keeps its helpers `private` and publishes only the configuration and the model class. The new file follows that pattern with `internal`, and the tests reach it through `@testable import MLXLLM`.

    **No reuse is available.** `YarnRoPE` in `MLXLMCommon` computes the same correction range, but it keeps `_freqs` private, it holds periods and not inverse frequencies, and `MLXFast.RoPE` has no inverse direction. `DeepseekV3.swift` holds `yarnFindCorrectionDim` and `yarnFindCorrectionRange` as `private`, thus another file cannot call them, and the card forbids a change to an existing file.
  timestamp: 2026-08-10T14:53:55.936105+00:00
- actor: claude-code
  id: 01kzp3nk89b67zmxg39ej59tx9
  text: |
    ### Mutation proof

    Every mutation was applied to the implementation one at a time (or in disjoint batches), the suite was run, and the mutation was reverted. Each one failed, and each one failed in the test that owns it.

    | Mutation | Test that failed | Evidence |
    |---|---|---|
    | Sinkhorn row and column axes transposed | `hcSplitSinkhornMatchesPythonReference` | 6 of the 9 `comb` entries over tolerance. `hcSplitSinkhornCombIsDoublyStochastic` still passed, which is why the exact-matrix comparison is the one that catches a transposed axis. |
    | `postGain` 2 to 1 | `hcSplitSinkhornMatchesPythonReference` | all 3 `post` entries over tolerance |
    | SwiGLU gate clamped on both sides | `clampedSwiGLUMatchesPythonReference` | 1 entry over tolerance, the `gate = -50` one |
    | `reduceRoutedExpertsFP32` sums first and casts after | `reduceRoutedExpertsFP32BeatsABFloat16Accumulation` | `fp32Error < bfloat16Error` and `fp32Sum == 1031` both failed |
    | Partial RoPE turns the leading dimensions | `applyPartialRoPEMatchesPythonReference` and `applyPartialRoPELeavesTheNoPositionDimensionsAlone` | 8 entries over tolerance; the leading 448 dimensions moved |
    | YaRN `betaFast` and `betaSlow` reversed (the osaurus reading) | `yarnInvFreqMatchesPythonReference` | 31 of the 32 entries over tolerance |
    | YaRN `high = min(..., dim - 1)` clamp dropped | `yarnInvFreqClampsTheCorrectionRangeToTheTable` | got `[1, 0.562, 0.316, 0.178]`, expected `[0.25, 0.141, 0.079, 0.044]` -- the 4x separation the fixture predicted, and no other test moved |
    | Inverse RoPE sine sign flip dropped | `inversePartialRoPEUndoesTheForwardRotation` | 192 entries over tolerance |

    After the reverts: `swift test --filter DeepseekV4MathHelpersTests` gives 12 of 12 pass, and `git diff --quiet -- Libraries Tests` is clean, thus no file that already existed was changed. The two new files are the only entries `git status` reports.

    ### Notes for the next agent

    - **`git checkout` cannot revert these files.** Both are untracked, thus each mutation had to be reverted by hand. Read the table above before you try `git checkout --` on them.
    - **The `hc_split_sinkhorn` fixture is asymmetric on purpose.** `max|M - transpose(M)|` is 0.0233. A symmetric fixture would let a transposed axis pass.
    - **A doubly-stochastic assertion alone does not catch a transposed axis.** Measured: with the axes swapped, the row and column sums are still 1 to within 1e-4.
    - **`swiftlint no_magic_numbers` reads a literal inside a `static let` initializer expression.** `private static let fullTurn: Float = 2 * .pi` reported; `.pi + .pi` does not, and doubling in binary floating point is exact, thus the number is the same.
    - **`hcSplitSinkhorn` takes `iters` and `eps` as required arguments**, not as defaults. `DeepseekV4Configuration.Default` already holds 20 and 1e-6, and a second copy here would be a second source of truth.
    - **`import MLXRandom` does not resolve** in this package. `MLXRandom.seed` and `MLXRandom.normal` arrive through `import MLX`.
  timestamp: 2026-08-10T15:11:44.393530+00:00
- actor: claude-code
  id: 01kzp3px82wqn90fmr8kgbdex0
  text: |
    ### implement — changed
    - evidence: 2 new files — `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift`, `Tests/MLXLMTests/DeepseekV4MathHelpersTests.swift`. `swift build` clean. `swift test --filter DeepseekV4MathHelpersTests` 12 of 12 pass. Full `swift test` 739 tests in 92 suites, 0 failures, 0 new warnings (the `missing creator` note and the `ParoQuantTests` Sendable warning both reproduce with the two new files moved aside). 8 mutations applied and reverted, each caught by the test that owns it. `swift-format lint`, `swiftlint no_magic_numbers`, `missing_docs`, `cyclomatic_complexity` and `function_body_length` all clean on the new files. `git diff --quiet -- Libraries Tests` clean.
    - next: `/review`
  timestamp: 2026-08-10T15:12:27.394438+00:00
depends_on:
- 01KZGMPECN4FA7T3BFX6F6QMF7
position_column: doing
position_ordinal: '80'
title: Port DeepseekV4MathHelpers with numeric parity tests
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` — the pure-math kernels DSV4 needs, none of which exist in this repo today. **Highest-value early task**: every function here is testable with synthetic inputs and no model weights, so it de-risks the whole port.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` (1445 lines). Port only the functions listed below; leave the DSpark and mxtq helpers out.

Functions to implement (all NEW vs DeepseekV3):

1. `hcSplitSinkhorn` — 20-iteration alternating row/column normalization of the mHC `comb` mixing matrix. Iteration count comes from config `hc_sinkhorn_iters`; epsilon from `hc_eps`. The gap tracker flags **axis alignment as Bug 1** — the row/col order is easy to transpose and produces plausible-but-wrong output, so assert on a hand-computed 3x3 case.
2. `applyPartialRoPE` — RoPE applied to only the trailing `qk_rope_head_dim = 64` dims of a 512-dim head, leaving the leading 448 dims untouched.
3. Inverse partial RoPE — rotates the trailing 64 dims *backward* via `conj(freqs_cis)`. Used on the attention **output** to strip positional information before the residual add-back. Must round-trip: `inverse(apply(x)) ≈ x`.
4. `yarnInvFreq` — YaRN inverse-frequency table with DSV4 parameters `rope_factor=16`, `original_seq_len=65536`, `beta_fast=32`, `beta_slow=1`. The reference notes a `high = min(..., dim-1)` clamp as **Bug 10**; include and test it.
5. `sqrtSoftplus` — `sqrt(softplus(x))`, the DSV4 router scoring function replacing sigmoid. Must be numerically stable for large negative and large positive `x`.
6. Clamped SwiGLU (`_DSV4SwiGLU`) — clamp gate and up to ±10 **before** silu (`swiglu_limit`, flagged **Bug 2**). Test that inputs beyond ±10 saturate rather than passing through.
7. `reduceRoutedExpertsFP32` — accumulate routed-expert outputs in float32 regardless of activation dtype, then cast back.

Cross-check numerics against Python. Reference implementations: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py` (SHA `f4b69bb`, 40424 bytes), and `ml-explore/mlx-lm` PR 1189 (`machiabeli:feat/deepseek-v4`). Generate expected values from one of those and check them in as fixtures — do not hand-wave tolerances.

Note: `MLXFast` is already available to this target (`Libraries/MLXLMCommon/AttentionUtils.swift:46` already calls `MLXFast.scaledDotProductAttention`), so no `Package.swift` change is needed.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` (MIT; header attributes Osaurus AI).
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`; `ml-explore/mlx-lm` PR 1189.
- Bug numbering (1, 2, 10) is from `osaurus-ai/vmlx-swift-lm` `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` exists exposing all seven functions above.
- [x] Every function matches Python reference output within a documented tolerance (state the tolerance in the test, not in prose).
- [x] Sinkhorn axis alignment is asserted against a hand-computed matrix, not just a round-trip.
- [x] Inverse partial RoPE round-trips to within tolerance.
- [x] SwiGLU clamp saturates at ±10. (`up` on both sides; `gate` on the high side only. Every reference -- the Python `_swiglu_limited`, the osaurus Swift, and the scouzi Swift -- clamps `gate` with `minimum(gate, limit)`. The fixture records the asymmetry, thus a symmetric clamp fails.)
- [x] No changes to any existing file. (`git diff --quiet -- Libraries Tests` is clean; two new untracked files.)

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4MathHelpersTests.swift`.
- [x] Test: `hcSplitSinkhorn` on a 3x3 fixture — asserts rows AND columns both sum to 1 within tolerance, and asserts the exact expected matrix (catches a transposed axis).
- [x] Test: `applyPartialRoPE` leaves dims 0..<448 bit-identical and changes dims 448..<512.
- [x] Test: inverse RoPE round-trip `inverse(apply(x)) ≈ x`.
- [x] Test: `yarnInvFreq` matches a checked-in Python-generated fixture; separately assert the `high = min(..., dim-1)` clamp with a `dim` that would otherwise overflow.
- [x] Test: `sqrtSoftplus` finite and non-NaN at x = -100, 0, 100; matches fixture values.
- [x] Test: clamped SwiGLU saturates for inputs ±50; matches fixture.
- [x] Test: `reduceRoutedExpertsFP32` on bfloat16 inputs is closer to an fp64-computed sum than a naive bfloat16 accumulation.
- [x] Run: `swift test --filter DeepseekV4MathHelpersTests` — all pass. (12 of 12.)

## Workflow
- Use `/tdd` — generate the Python fixtures and write the failing assertions first, then port each function to green.
#deepseek-v4