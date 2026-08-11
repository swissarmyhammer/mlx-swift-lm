---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: Decide whether the DeepSeek-V4 shared expert reads the SwiGLU clamp
---
## What

The two references disagree about the shared expert of a DeepSeek-V4 MoE
layer, and this repository must follow one of them.

- `scouzi1966/mlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift:1284-1287`
  and `osaurus-ai/vmlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift:558-561`
  build `DeepseekV4MLP` with `swigluLimit: config.swigluLimit`, thus the shared
  expert reads the clamp.
- `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`
  `mlx_lm/models/deepseek_v4.py:397-401` builds the same layer with
  `swiglu_limit=0.0`, thus the shared expert does NOT read the clamp.
  `_swiglu_limited` returns `silu(gate) * up` with no clamp when the limit is
  zero.

Task `^g5e7907` ported the Swift form, because the card of that task names the
Swift file as the reference to port. `Libraries/MLXLLM/Models/DeepseekV4MoE.swift`
states the divergence in its header.

The difference is a real numeric difference. The clamp changes the output of
the shared expert whenever a gate or an up value leaves the range, and the
shared expert reads every token of every layer.

## How to decide

Read the shared expert of the DeepSeek-V4 checkpoint against a run of the
model. One of these gives the answer:

1. The `modeling_deepseek_v4.py` of the checkpoint itself on Hugging Face, if
   the repository publishes it. That file is the training-time definition and
   it decides.
2. A run of DeepSeek-V4-Flash with each of the two forms, comparing the output
   of one layer against the reference implementation the checkpoint ships.

## Acceptance Criteria

- [ ] The decision is written on this task, with the source that decides it.
- [ ] `DeepseekV4MoE.swift` agrees with the decision.
- [ ] The header of that file states the decision instead of the divergence.
- [ ] A test pins the choice, so a later edit cannot flip it without a red run.
#deepseek-v4