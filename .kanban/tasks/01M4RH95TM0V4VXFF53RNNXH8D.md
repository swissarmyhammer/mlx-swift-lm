---
assignees:
- claude-code
position_column: todo
position_ordinal: '8280'
title: 'DeepSeek-V4 speed: compile the small hot functions of the decode step'
---
## What

The DeepSeek-V4 port has no compiled path. `Libraries/MLXLLM/Models/DeepSeekV4.swift:47-50` records this as a known difference from the Python reference, which uses `compile(shapeless:)`.

Functions to compile:
- The gate selection: `DeepSeekV4MoE.swift:309-346`. Also move the float32 cast of the gate weight at line 312 to load time.
- `clampedSwiGLU`: `DeepSeekV4MathHelpers.swift:341-350`.
- The hyper-connection `expand`: `DeepSeekV4HyperConnection.swift:286-298`.
- Replace the manual query-head norm at `DeepSeekV4Attention.swift:673-674` with `MLXFast.rmsNorm`.

Python equivalents in Blaizzy/mlx-vlm `deepseek_v4/language.py`: `_expert_select`, `_limited_swiglu`, `_hc_expand_op`.

## Warning

Give the position offset to a compiled function as an array, not as an `Int`. In mlx-lm PR #1189, an `Int` offset made a new compile-cache entry for each token, and decode fell from 29 to 17.7 tok/s after 6K tokens. `Libraries/MLXLMCommon/CompiledTrace.swift` already exists.

## Acceptance Criteria

- [ ] The listed functions run through `compile`. No compiled function takes a value that changes on each step as a Swift scalar.
- [ ] Parity tests show the same results as the current path.
- [ ] The existing DeepSeek-V4 unit tests pass.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after, at 128 and at 16k context.
#deepseek-v4 #performance