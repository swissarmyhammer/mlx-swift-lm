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

### Folded from ^s2ajtyr (chain reconciliation 2026-07-22)

References: upstream mlx-lm PRs #1398 (`minimax_m3_vl.py`) and #1401 (`minimax_m3`); locally `GPTOSS.swift:81-94` (swigluoai — a two-argument product with clamps on both the gate and linear inputs and `(x + 1)` on the linear term; honor `swiglu_alpha`/`swiglu_limit` from config rather than hardcoding).

- **Fused, pre-stacked expert weights are the primary checkpoint layout.** The mlx-community checkpoint stores experts as `...block_sparse_moe.switch_mlp.gate_up_proj.{weight,scales,biases}` and `...down_proj.{...}`: gate+up fused into one tensor, already stacked across experts, already quantized. There is NO per-expert `experts.N.w1/w2/w3` in this checkpoint. The repo's `FusedGateUpSwitchGLU` (`Libraries/MLXLMCommon/SwitchLayers.swift:136`, key `gate_up_proj`) matches the layout but its public init takes a single-argument activation and computes `activation(gate) * up` (`SwitchLayers.swift:200-206`) — insufficient for swigluoai's two-arg product. GPT-OSS solves the same problem with its own private `SwiGLUSwitchGLU` rather than `SwitchGLU`. Extend `FusedGateUpSwitchGLU` with a two-arg activation seam or use a private module like GPT-OSS's, and **verify the gate/up packing order (interleaved vs concatenated halves) against upstream `minimax_m3_vl.py` (PR #1398)** before writing the split.
- Routing semantics detail: router is `Linear(hidden, num_local_experts /* 128 */, bias: false)`; scores computed in **float32** with sigmoid (`scoring_func: "sigmoid"`). `e_score_correction_bias` (confirmed checkpoint weight name, same as M2) is used for top-k *selection only* — routing weights are gathered from the *original unbiased* sigmoid scores, normalized to sum 1 (+epsilon), then scaled by `routed_scaling_factor` (2.0).
- **Shared expert weight keys: verify against the checkpoint's `model.safetensors.index.json` FIRST** — index sampling did not surface any "shared" keys, so confirm existence and exact naming (if the checkpoint truly has none, drop the module and record why). A wrong key name silently leaves dead parameters at load.
- Test ideas folded: routing test with crafted router weight + bias proving selection uses biased scores while weighting uses unbiased normalized scores (style: `LFM2MoeRoutingTests.swift`); fused gate_up split-order pin test (known input, hand-computed expected output through a tiny fused expert); swigluoai clamp test matching GPT-OSS behavior for identical inputs.

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

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax #minimax-m3