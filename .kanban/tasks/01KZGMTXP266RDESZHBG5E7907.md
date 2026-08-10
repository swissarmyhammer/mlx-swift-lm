---
assignees:
- claude-code
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