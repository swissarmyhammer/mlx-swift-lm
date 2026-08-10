---
assignees:
- claude-code
depends_on:
- 01KZGMPECN4FA7T3BFX6F6QMF7
position_column: todo
position_ordinal: '8280'
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

- [ ] `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` exists exposing all seven functions above.
- [ ] Every function matches Python reference output within a documented tolerance (state the tolerance in the test, not in prose).
- [ ] Sinkhorn axis alignment is asserted against a hand-computed matrix, not just a round-trip.
- [ ] Inverse partial RoPE round-trips to within tolerance.
- [ ] SwiGLU clamp saturates at ±10.
- [ ] No changes to any existing file.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4MathHelpersTests.swift`.
- [ ] Test: `hcSplitSinkhorn` on a 3x3 fixture — asserts rows AND columns both sum to 1 within tolerance, and asserts the exact expected matrix (catches a transposed axis).
- [ ] Test: `applyPartialRoPE` leaves dims 0..<448 bit-identical and changes dims 448..<512.
- [ ] Test: inverse RoPE round-trip `inverse(apply(x)) ≈ x`.
- [ ] Test: `yarnInvFreq` matches a checked-in Python-generated fixture; separately assert the `high = min(..., dim-1)` clamp with a `dim` that would otherwise overflow.
- [ ] Test: `sqrtSoftplus` finite and non-NaN at x = -100, 0, 100; matches fixture values.
- [ ] Test: clamped SwiGLU saturates for inputs ±50; matches fixture.
- [ ] Test: `reduceRoutedExpertsFP32` on bfloat16 inputs is closer to an fp64-computed sum than a naive bfloat16 accumulation.
- [ ] Run: `swift test --filter DeepseekV4MathHelpersTests` — all pass.

## Workflow
- Use `/tdd` — generate the Python fixtures and write the failing assertions first, then port each function to green.
#deepseek-v4