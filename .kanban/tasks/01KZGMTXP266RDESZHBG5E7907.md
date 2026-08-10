---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp8t1z0sh2w4tnm7y3t6ddd
  text: |
    ## Note carried from the math-helpers task `p34crp6` (review, 2026-08-10)

    Two points this card must know before it wires in `clampedSwiGLU` and `reduceRoutedExpertsFP32`.

    ### 1. The SwiGLU clamp is asymmetric. Do not write "±10" into a test.

    The user decided this on `p34crp6` on 2026-08-10. The reference wins:

    ```swift
    gate = minimum(gate, limit)                // the high side only
    up   = clip(up, min: -limit, max: limit)   // both sides
    ```

    The Python `_swiglu_limited`, the osaurus Swift copy and the scouzi Swift copy all agree. Reason the user gave: `silu()` falls to almost zero at a large negative gate, thus the low side cannot run away.

    **This card carries the wording the user corrected.** "What" item 3 says "gate/up clamped to ±10", the acceptance criterion says "Expert SwiGLU clamps at ±10", and the test line says "expert output saturates for inputs beyond ±50 (clamp is active)". Read as written, a test that asserts a `gate` of -50 saturates will fail against the correct code. Correct these three lines to the asymmetric rule before you write the test: an `up` below -10 or above +10 saturates, a `gate` above +10 saturates, and a `gate` below -10 does not saturate. At `gate = -50` the reference output is `9.643749e-20`, not zero.

    ### 2. `clampedSwiGLU` computes in float32 and casts back to the input dtype.

    `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` casts `gate` and `up` to float32, clamps and applies silu there, then returns the input dtype. The Python computes in the input dtype throughout. This is not an error -- the float32 path is more accurate, and it is the same pattern item 7 of `p34crp6` asks for in `reduceRoutedExpertsFP32`.

    It matters here because every test on `p34crp6` runs float32, where the round trip changes nothing. This card is the first place bfloat16 activations reach the function. **When you build the MoE parity fixtures, state whether the expected numbers come from a Python run in bfloat16 or in float32.** A bfloat16 fixture generated from the Python will differ from this Swift by about one bfloat16 unit in the last place. Choose the tolerance with that in mind rather than discovering it as a mystery failure.
  timestamp: 2026-08-10T16:41:33.408327+00:00
- actor: claude-code
  id: 01kzp8w24mz5g6b9pq56m4z2cj
  text: |
    ## Correction from task ^p34crp6 — read before you start

    ### 1. This card has the wrong SwiGLU clamp rule in three places

    This card says "gate/up clamped to plus or minus 10". That is wrong, and the
    user decided against it on 2026-08-10 while ^p34crp6 was in review.

    The correct rule, which `DeepseekV4Math.clampedSwiGLU` already has:

    ```swift
    gate = minimum(gate, limit)                 // the high side only
    up   = clip(up, min: -limit, max: limit)    // both sides
    ```

    `silu()` falls to almost zero at a large negative gate, thus the low side of the
    gate cannot run away. Only `up` multiplies, thus only `up` needs both sides.
    This agrees with the Python `_swiglu_limited` and with both Swift reference
    copies.

    The planned test on this card, "expert output saturates for inputs beyond plus
    or minus 50", FAILS against correct code as written. A gate of -50 gives
    9.64e-20 and a gate of -10 gives 4.54e-3; the two are not the same, and that
    difference is correct. Correct this card's wording before you write the test.

    ### 2. Say which dtype your parity fixtures use

    `clampedSwiGLU` casts to float32, does the arithmetic, and casts back to the
    dtype of `gate`. The Python computes in the input dtype. The two agree in
    float32, thus every ^p34crp6 test passes.

    This card is the first to send bfloat16 activations into that function. A
    fixture from a Python run in bfloat16 differs from one in float32 by about one
    unit in the last place. Write in the test which one you used.

    ### 3. Reuse, do not copy

    `DeepseekV4Math` already holds `clampedSwiGLU`, `sqrtSoftplus`, and
    `reduceRoutedExpertsFP32`, and all three have parity tests. Call them.
  timestamp: 2026-08-10T16:42:39.124357+00:00
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
- 01KZGMRCBJ4SPK7C26PWKV5J6F
position_column: todo
position_ordinal: '8680'
title: 'Port DeepseekV4 MoE: sqrtsoftplus gate, hash routing, clamped-SwiGLU experts'
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4MoE.swift` with `DeepseekV4MoEGate`, `DeepseekV4MoE`, and `DeepseekV4MLP`.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4MoEGate` at line 997, `DeepseekV4MoE` at line 1235, `DeepseekV4MLP` at line 1332.

Real config shape: 256 routed experts, 6 experts per token, 1 shared expert.

Features, all NEW vs DeepseekV3:

1. **`sqrt(softplus(x))` routing scores** replacing DSV3's sigmoid. Critical subtlety: the routing bias is applied **for the top-k selection**, but the weights actually gathered must be the **UNBIASED** scores. Getting this backwards is silent and degrades quality without crashing.
2. **Hash routing on layers 0-2** (`num_hash_layers=3`): load the `tid2eid` int64 hash table from the checkpoint and route by token id, bypassing top-k entirely. Use `isHashLayer(_:)` from the config task.
3. **Clamped SwiGLU experts** — gate/up clamped to ±10 before silu via the math-helpers function (`swiglu_limit`, Bug 2).
4. **FP32 routed-expert reduction** — accumulate expert outputs in float32 via `reduceRoutedExpertsFP32` before casting back.
5. Shared expert added to the routed result.

Build the routed experts on this repo's existing `SwitchGLU` / `SwitchLinear` / `QuantizedSwitchLinear` in `Libraries/MLXLMCommon/SwitchLayers.swift` — do **not** copy the reference's 1025-line variant; ours has diverged and its fused Metal kernels are deliberately out of scope (see task `wkv5j6f`). Use `gatherSort`/`scatterUnsort` (`SwitchLayers.swift:30`, `:51`) as the other MoE models here do.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~997-1360 (MIT; header attributes Osaurus AI).
- Routing/bias and hash-routing details: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`; `ml-explore/mlx-lm` PR 1189.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4MoE.swift` exists with all three types.
- [ ] Gate selects top-6 of 256 using **biased** scores but returns **unbiased** weights — asserted by a test with a deliberately lopsided bias.
- [ ] Hash-routing layers (0-2) bypass top-k and route deterministically by token id; a repeated token id always hits the same expert.
- [ ] Non-hash layers (3+) use the sqrtsoftplus top-k path.
- [ ] Expert SwiGLU clamps at ±10.
- [ ] Routed reduction happens in float32.
- [ ] `Libraries/MLXLMCommon/SwitchLayers.swift` unmodified by this task.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4MoETests.swift`, synthetic config (e.g. 8 experts, top-2) and random weights.
- [ ] Test: with a bias vector that strongly favors expert 0, the *selected indices* shift toward expert 0 while the *returned weights* equal the unbiased scores at those indices. This is the bias-vs-weight guard.
- [ ] Test: hash layer determinism — the same token id routes to the same expert across two calls, and does not depend on the hidden-state values.
- [ ] Test: layer 3 uses top-k (assert exactly `numExpertsPerTok` nonzero weights per token).
- [ ] Test: expert output saturates for inputs beyond ±50 (clamp is active).
- [ ] Test: full `DeepseekV4MoE` forward shape `[1, 4, hidden]` to same, `allFinite`, and shared-expert contribution is nonzero.
- [ ] Run: `swift test --filter DeepseekV4MoETests` — all pass.

## Workflow
- Use `/tdd` — the biased-selection/unbiased-weight test is the highest-value assertion; write it first.
#deepseek-v4