---
comments:
- actor: claude-code
  id: 01ky6083gfksxqb0qpz0q1grx0
  text: 'Closed as duplicate in chain reconciliation (user decision 2026-07-22): superseded by ^mv9aq7w; unique content folded there.'
  timestamp: 2026-07-22T22:48:23.823460+00:00
depends_on:
- 01KXX9B4XP0F51FFS6G0ZXGT4W
position_column: done
position_ordinal: cd80
title: 'MiniMax M3: hybrid dense/MoE MLP with swigluoai, routing bias, shared expert'
---
#minimax-m3

## What
Add the MLP/MoE layers to `Libraries/MLXLLM/Models/MiniMaxM3.swift`. M3 is hybrid: layers 0-2 use a dense MLP, layers 3-59 use 128-expert MoE with a shared expert. References: upstream mlx-lm PRs #1398/#1401, and locally `MiniMax.swift` (M2 MoE block) + `GPTOSS.swift` (swigluoai).

- **SwigluOAI activation** (`hidden_act: "swigluoai"`, alpha 1.702, limit 7.0): a TWO-argument product — clamps on both gate and linear inputs and `(x + 1)` on the linear term; see `GPTOSS.swift:81-94`. Honor swiglu_alpha and swiglu_limit from config rather than hardcoding.
- **Fused, pre-stacked expert weights — this is the primary layout.** The mlx-community checkpoint stores experts as `...block_sparse_moe.switch_mlp.gate_up_proj.{weight,scales,biases}` and `...down_proj.{...}`: gate+up fused into one tensor, already stacked across experts, already quantized. There is NO per-expert `experts.N.w1/w2/w3` in this checkpoint. The repo's `FusedGateUpSwitchGLU` (`Libraries/MLXLMCommon/SwitchLayers.swift:136`, key `gate_up_proj`) matches the layout but its public init takes a single-argument activation and computes `activation(gate) * up` (`SwitchLayers.swift:200-206`) — insufficient for swigluoai. GPT-OSS solves the same problem with its own private `SwiGLUSwitchGLU` rather than `SwitchGLU`. Implement a fused expert module with the two-arg swigluoai product (extend `FusedGateUpSwitchGLU` with a two-arg activation seam or a private module like GPT-OSS's), and **verify the gate/up packing order (interleaved vs concatenated halves) against upstream `minimax_m3_vl.py` (PR #1398)** before writing the split.
- **`MiniMaxM3SparseMoeBlock`**:
  - Router `Linear(hidden, num_local_experts /* 128 */, bias: false)`, scores computed in float32 with sigmoid (`scoring_func: "sigmoid"`).
  - Routing bias for top-k *selection only*, weights gathered from the *original* unbiased sigmoid scores — the checkpoint weight name is confirmed: `e_score_correction_bias` (same as M2's handling in `MiniMax.swift`).
  - Top-4 selection (`num_experts_per_tok`), gathered scores normalized to sum 1 (+epsilon), then scaled by `routed_scaling_factor` (2.0).
  - **Shared expert** (`n_shared_experts: 1`, `shared_intermediate_size` 3072): a dense swigluoai MLP added to the routed output. **Verify the actual shared-expert weight key names against the checkpoint's `model.safetensors.index.json` FIRST** — index sampling did not surface any "shared" keys, so confirm existence and exact naming (and if the checkpoint truly has none, drop the module and record why). A wrong key name silently leaves dead parameters at load.
- **Dense MLP** for dense layers: gate/up/down (or fused, matching the checkpoint layout) with width `dense_intermediate_size` (12288), swigluoai activation.
- Layer schedule helper: dense vs MoE per layer index from `moe_layer_freq` (`[0,0,0,1,...]` — 0-2 dense, 3-59 MoE), usable by the assembly task.

## Acceptance Criteria
- [ ] Routing test with crafted router weight + bias proves: selection uses biased scores, weighting uses unbiased normalized scores, output scaled by routed_scaling_factor
- [ ] Fused gate_up split order verified against upstream reference and pinned by a test (known input, hand-computed expected output through a tiny fused expert)
- [ ] SwigluOAI clamps as configured (limit 7.0) and matches GPT-OSS behavior for identical inputs
- [ ] Shared-expert key names verified against the real index (or its absence documented); module weight keys match
- [ ] Dense layers use dense_intermediate_size, MoE experts use intermediate_size; output shape `(B, L, hidden)` for both block kinds from a tiny config

## Tests
- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: routing semantics (see `LFM2MoeRoutingTests.swift` for style), fused-split pin test, shared-expert test, swigluoai clamp test, dense-vs-moe schedule test
- [ ] Run `swift test --filter MiniMaxM3`; expect pass

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax