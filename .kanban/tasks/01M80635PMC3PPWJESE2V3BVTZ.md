---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m81yjbcmk5a2hvwfmgkrkes8
  text: |-
    Done: the oQ4e (oMLX layout) checkpoint loads as the target. Commits: 3e138e5f, 0743a400 (review fixes), 08e8229e (import fix).

    Mapping (Libraries/MLXVLM/Models/Inkling2OMLXLayout.swift, applied in Inkling2.prepareCheckpoint; reference jundot/omlx omlx/patches/inkling2 loader.py/oq_layout.py/model.py):
    - sliding `attn.in_proj` rows q|k|v|r|gate|sink → q/k/v/r/gate/sink_proj; full `attn.q_proj` q|r|gate|sink; full `attn.k_proj`/`v_proj` → k/v_proj + k/v_sconv.weight_proj (weight, scales, biases split by rows from the text config);
    - `loop_down` [4,2,D,r] → q/k/v/mlp_message_gate.down_proj [r,2D] (bf16, exact), `loop_gain_forget` [4,r,2D] → message_gate_proj / forget_gate_proj [D,r]; per-layer `loop_message.query` → stacked `message_input.query`;
    - conv [C,K] → [C,K,1]; one-row queries → vectors; renames of norms, q/r scales, rel_proj, k/v_static, router weight/bias, experts w13/w2 (interleaved gate/up, same as the port), embed/embed_norm/norm/final_input, vision backbone (ln_1/ln_2/ls_1/ls_2/c_fc/c_proj/ln_pre/proj_in/proj_out);
    - per-layer quantization of config.json (8-bit embed, lm_head, attention projections) renamed onto the split projections' paths; expert formats from tensor shapes (layers 0–19 nvfp4 with global_scale, 20–23 affine 4-bit g64). Each key maps on its own, so the on-demand vision load maps too.
    Verified offline against mlx-community/Inkling-2-Mini-4bit (same source): nvfp4 experts, routers, convs, norms, queries, message gates, vision tensors bit-identical after mapping; requantized projection splits correlate 0.992–0.996 with the 4bit dequantized ones.
    Tests: Inkling2OMLXLayoutTests (tiny model written in the oMLX layout → every parameter back; the oMLX checkpoint loads and the forward pass equals the original; vision names; quantization paths).

    Review findings (3e138e5f), all resolved in 0743a400, re-review clean: Inkling2OQ4eTargetLoader duplicated Inkling2MiniCheckpoint (now takes its configuration); greedy-equality check duplicated (expectMatchesPlain); empty-collection literals (3); magic numbers (5); data-driven fused table; k/v case duplication (fusedParts, modulePaths); modulePaths re-listed fused parts.

    Numbers (Inkling2MTPIntegrationTests.oQ4eTargetBesideThe4BitTarget, both targets in one process, same protocol, median of 3 after warm-up; GPU shared with a concurrent SWE-bench, two runs shown as run1 / run2):
    - Memory: parameters 14.45 GiB oQ4e vs 15.50 GiB 4bit (files 15.10 vs 16.12 GiB).
    - Plain tok/s short: oQ4e 59.0 / 64.6 vs 4bit 60.3 / 66.2; long (5077 tok): 57.4 / 61.4 vs 59.9 / 62.6.
    - MTP block 3 short: oQ4e 71.8 / 81.1 vs 79.2 (run2) ; long: oQ4e 90.9 / 114.8 vs 89.6 / 97.6 (oQ4e acceptance 210/342 vs 204/354).
    - MTP block 4 long: oQ4e 91.9 / 93.9 vs 86.1 / 92.6.
    - Greedy MTP == plain on oQ4e: exact at the short prompt (blocks 3 and 4) and below the window; past the window within the near tie (^2r778tc: at long block 4, token 77 is a three-way tie, MTP's token 0.52 below the best). With the exact ring-order verify (before the ^2r778tc revert) oQ4e matched plain exactly at long blocks 3 and 4 (exact long-context MTP 103.4 tok/s block 3 vs 4bit 90.1 in that run).

    Tests on final code: all 5 bundles green; Inkling2MTPIntegrationTests 6/6 (incl. the oQ4e test), Inkling2CheckpointIntegrationTests 4/4, Inkling2ToolLoopIntegrationTests 1/1.
  timestamp: 2026-11-22T15:39:02.932328+00:00
position_column: done
position_ordinal: ffffd480
title: 'Inkling 2: load the oQ4e (oMLX layout) target weights'
---
`mlx-community/Inkling-2-Mini-oQ4e-mtp` stores its target weights in the oMLX layout (`model.embed`, `model.layers.N.attn.in_proj` fused projections, `attn.k_conv`/`k_static`, `attn_conv`/`mlp_conv`, `loop_down`/`loop_gain_forget`, `mlp.router_weight`/`router_bias`, `experts.w13`/`w2` with per-layer mixed precision), not the Hugging Face layout `Inkling2.prepareCheckpoint` maps. The port loads only `mlx-community/Inkling-2-Mini-4bit`; ^04yn8q4 drafts with the oQ4e repo's `mtp.safetensors` for that 4bit target.

Map the oMLX names and quantization (see jundot/omlx omlx/patches/inkling2/loader.py, oq_layout.py, model.py) onto the port's modules so the oQ4e checkpoint loads as the target too, then run `Inkling2MTPIntegrationTests` with it.

#inkling