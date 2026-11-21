---
assignees:
- claude-code
position_column: todo
position_ordinal: '8480'
title: 'Inkling 2: load the oQ4e (oMLX layout) target weights'
---
`mlx-community/Inkling-2-Mini-oQ4e-mtp` stores its target weights in the oMLX layout (`model.embed`, `model.layers.N.attn.in_proj` fused projections, `attn.k_conv`/`k_static`, `attn_conv`/`mlp_conv`, `loop_down`/`loop_gain_forget`, `mlp.router_weight`/`router_bias`, `experts.w13`/`w2` with per-layer mixed precision), not the Hugging Face layout `Inkling2.prepareCheckpoint` maps. The port loads only `mlx-community/Inkling-2-Mini-4bit`; ^04yn8q4 drafts with the oQ4e repo's `mtp.safetensors` for that 4bit target.

Map the oMLX names and quantization (see jundot/omlx omlx/patches/inkling2/loader.py, oq_layout.py, model.py) onto the port's modules so the oQ4e checkpoint loads as the target too, then run `Inkling2MTPIntegrationTests` with it.

#inkling