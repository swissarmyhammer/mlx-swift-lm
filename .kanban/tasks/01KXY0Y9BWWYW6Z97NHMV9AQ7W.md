---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'MiniMax-M3: MoE block + swigluoai activation building blocks'
---
## What

First slice of MiniMax-M3 support (model_type `minimax_m3_vl`, arch `MiniMaxM3SparseForConditionalGeneration`, reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py`). Create `Libraries/MLXVLM/Models/MiniMaxM3.swift` containing the two self-contained compute blocks the decoder will build on, with unit tests — no decoder/model yet:

1. **`MiniMaxM3SwiGLUOAI` activation** — clipped SwiGLU: `x_glu = clip(glu, max: limit)`, `x_linear = clip(linear, min: -limit, max: limit)`, output `x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)` with config-driven `alpha` (1.702), `limit` (7.0), `beta` (1.0). Config key `hidden_act: "swigluoai"`.
2. **`MiniMaxM3SparseMoeBlock`** — extends the existing M2 pattern (`Libraries/MLXLLM/Models/MiniMax.swift`'s `MiniMaxSparseMoeBlock`: sigmoid scoring + `e_score_correction_bias` routing bias + `SwitchGLU` from `Libraries/MLXLMCommon/SwitchLayers.swift`) with M3's additions: **1 shared expert** (a plain MLP with its own intermediate size, always active, summed with routed output) and **`routed_scaling_factor` (2.0)** applied to normalized routing weights. 128 local experts, top-4 per token, expert intermediate size 3072.

Follow the repo's model-file conventions (`@ModuleInfo`/`@ParameterInfo` keys matching the checkpoint's weight names — mirror mlx-vlm's parameter naming for `block_sparse_moe`, `shared_mlp`/shared-expert keys, and gate; check the reference for exact key strings).

## Acceptance Criteria

- [ ] `MiniMaxM3SwiGLUOAI` numerically matches the reference formula (hand-computed expectations for known inputs, including clipping at the limit boundary)
- [ ] `MiniMaxM3SparseMoeBlock` output shape is `(B, L, hiddenSize)` for a tiny config, shared-expert contribution verified (zeroing shared-expert weights changes output; routing-only path matches M2-style math)
- [ ] `routed_scaling_factor` and `e_score_correction_bias` are applied exactly as mlx-vlm's `language.py` does (document the formula in a doc comment)
- [ ] `swift build` clean — no warnings

## Tests

- [ ] New: `Tests/MLXLMTests/MiniMaxM3Tests.swift` — activation formula tests (interior + clipped regions), MoE block shape/routing tests with a tiny config (e.g. 8 experts, top-2, hidden 16)
- [ ] The new suite runs GPU forward passes under SwiftPM: it MUST invoke the GPU/metallib symlink test bootstrap from its `init()`, exactly like the other GPU-touching suites in `Tests/MLXLMTests` (see `Tests/MLXLMTests/TestBootstrap.swift` and the root fix from kanban 23ff1zx) — without it, `swift test` cannot load the Metal library and the suite fails spuriously. Applies to the follow-up M3 tasks extending this suite too.
- [ ] Run: `swift test --filter MLXLMTests.MiniMaxM3` → passes; `swift test --filter MLXLMTests` → no regressions

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax