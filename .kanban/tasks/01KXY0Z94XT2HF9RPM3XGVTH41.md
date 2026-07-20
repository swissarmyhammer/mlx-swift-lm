---
assignees:
- claude-code
depends_on:
- 01KXY0Y9BWWYW6Z97NHMV9AQ7W
position_column: todo
position_ordinal: 8a80
title: 'MiniMax-M3: config + dense-attention language model + weight sanitization'
---
## What

The M3 language model core in `Libraries/MLXVLM/Models/MiniMaxM3.swift` (same file as ^mv9aq7w's blocks), **dense attention only** — MSA sparse attention is the dependent follow-up ^(msa task). Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py` and `config.py`; structural template in this repo: `Libraries/MLXVLM/Models/Qwen3VL.swift` (text config nesting) and `Libraries/MLXLLM/Models/MiniMax.swift` (M2 attention/decoder shape).

1. **`MiniMaxM3TextConfiguration`** decoded from `text_config`: hidden 6144, 60 layers, 64 attention heads, 4 KV heads, head_dim 128, vocab 200,064, rms_norm_eps 1e-6 (**Gemma-mode norm** — `(1 + weight)` scaling), rope_theta 5,000,000, **rotary_dim 64 / partial_rotary_factor 0.5** (use `initializeRope` in `Libraries/MLXLMCommon/RoPEUtils.swift`, which already handles partial rotary), per-head **QK norm**, MoE fields (128 experts, top-4, shared expert, `first_k_dense_replace`-style: layers 0–2 dense MLP with intermediate 12,288, MoE from layer 3), sparse-attention fields (decode them now — block 128, topk 16, index_heads 4, index_dim 128 — even though this task runs dense), MTP fields (7 modules). Top-level `MiniMaxM3Configuration` holds `text_config` + `vision_config` (vision decoded but unused here).
2. **`MiniMaxM3Attention`** — dense: partial RoPE on q/k, per-head Gemma-mode RMSNorm QK norm, GQA 64/4, `attentionWithCacheUpdate` with standard `KVCacheSimple`-compatible caching.
3. **Decoder layer / model / lm_head** (check `tie_word_embeddings`), layers 0–2 dense MLP, 3–59 MoE.
4. **`sanitize(weights:)`** — drop vision-tower weights (`vision_tower.*`/`multi_modal_projector.*` — verify exact prefixes against the real checkpoint index) and drop the 7 MTP modules' weights (mirroring mlx-vlm; MTP task ^(mtp) will revisit), remap expert weights to `SwitchGLU`'s stacked layout exactly as M2's `sanitize` does, and keep `e_score_correction_bias` + gate weights un-quantized/FP32 per the reference's cast predicate.

Do NOT register with the VLM factory yet (that's ^(registration task)) — this task's deliverable is a model type that compiles and passes tiny-config forward passes.

## Acceptance Criteria

- [ ] Tiny-config (e.g. 2 dense + 2 MoE layers, hidden 32) forward pass produces `(B, L, vocab)` logits, runs incrementally with a KV cache, and is deterministic under a fixed `MLXRandom.seed`
- [ ] Partial rotary verified: only the first 64 of 128 head dims are rotated (test against a hand-rolled reference on one head)
- [ ] `sanitize` on a synthetic weight dict containing vision/MTP/expert keys drops vision + MTP keys and produces exactly the model's expected parameter set (`loadWeights`-style verification, mirroring `Tests/MLXLMTests/LoadWeightsTests.swift` patterns)
- [ ] `newCache(parameters:)` returns 60 `KVCacheSimple` entries (dense pass)

## Tests

- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: config decode from a literal JSON snippet of the real config.json, tiny-model forward/cache tests, partial-rotary unit test, sanitize test
- [ ] Run: `swift test --filter MLXLMTests.MiniMaxM3` → passes; `swift test --filter MLXLMTests` → no regressions

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax