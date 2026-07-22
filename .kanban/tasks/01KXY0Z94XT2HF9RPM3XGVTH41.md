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

### Folded from ^agbfebr (chain reconciliation 2026-07-22)

- **Flat-config decoding — the config must decode BOTH shapes**: the VL-nested layout above (`model_type: "minimax_m3_vl"`, arch `MiniMaxM3SparseForConditionalGeneration`, text keys nested under `text_config`) AND a flat text config (`model_type: "minimax_m3"`, produced by upstream mlx-lm PR #1401-style conversions of text-only repos). Follow `Gemma3Text.swift`'s custom `init(from:)` fallback-to-`text_config` pattern (see its `CodingKeys.textConfig`). Add a flat-vs-nested equivalence test (a synthetic flat `minimax_m3` JSON decodes to the same field values as the nested fixture).
- Download `config.json` from https://huggingface.co/mlx-community/MiniMax-M3-4bit and check it in as a test fixture (follow the repo's fixture conventions in `Tests/MLXLMTests`; embedding the JSON in the test file is acceptable).
- Additional verified config values to assert beyond the list above: intermediate_size 3072 (per-expert width), dense_intermediate_size 12288, n_shared_experts 1, shared_intermediate_size 3072, scoring_func "sigmoid", use_routing_bias true, routed_scaling_factor 2.0, hidden_act "swigluoai", swiglu_alpha 1.702, swiglu_limit 7.0, use_qk_norm true, qk_norm_type "per_head", use_gemma_norm true, max_position_embeddings 1048576, tie_word_embeddings false. The MoE layer schedule appears as `moe_layer_freq = [0,0,0,1,1,...]` (layers 0–2 dense / 3–59 MoE) — decode the schedule key as it actually appears in the fixture.
- Sparse-attention block — **verified real shape**: nested under `text_config.sparse_attention_config` as `{sparse_index_dim: 128, sparse_num_index_heads: 4, sparse_topk_blocks: 16, sparse_block_size: 128}`. Do NOT invent flat keys like `use_sparse_attention`/`sparse_init_block`; they are not in the config.
- Unknown keys (vision_config, mtp, quantization) must not fail decoding; provide sensible defaults so a tiny synthetic config (few layers, few experts) constructs without a full config file — needed by later unit tests.

### Folded from ^0zxgt4w (chain reconciliation 2026-07-22)

References: upstream mlx-lm PRs #1398 (`minimax_m3_vl.py`) and #1401 (`minimax_m3`).

- **Per-head QK norm layout**: RMSNorm over `headDim` applied after reshaping to `(B, L, heads, headDim)` — unlike M2's flat RMSNorm over `heads * headDim` before the reshape. Verify the norm weight shape against the checkpoint index / upstream PR #1401 before committing to the layout; test with a non-uniform norm weight.
- **Gemma-mode RMSNorm**: `(1 + weight)` scaling with float32 normalization — reuse the repo's existing Gemma norm (see `Gemma.swift` / `Gemma3Text.swift`) if accessible from this module; otherwise a local private copy.
- Partial RoPE: keep the `applyRotaryPosition`/`ropeOffset` cache pattern from `MiniMax.swift`; scale headDim^-0.5, no biases. Test idea: partial-RoPE tail-passthrough test — only the first 64 of 128 head dims rotate, the unrotated tail passes through (see `RoPEApplicationTests.swift` for conventions).
- `attention_output_gate` is false for M3 — no gate module.

### Folded from ^weryyak (chain reconciliation 2026-07-22)

- **CRITICAL — keep the `language_model.` prefix as a module path; do NOT re-key it away.** The VL checkpoint's weights AND its per-module quantization overrides are keyed `language_model.model.layers.N...` (e.g. `language_model.model.layers.N.block_sparse_moe.gate` is 8-bit while the default is 4-bit gs64 affine). `Load.swift:64-71` resolves quantization by module path (`perLayerQuantization.quantization(layer: path)`) and the factory passes the quant dict through unmodified — re-keying weights to `model.*` makes the 8-bit gate lookup miss, fall back to 4-bit, and fail shape-check at load (and it would only surface on a big-memory machine in the integration task). Follow the `Qwen35.swift` precedent: `@ModuleInfo(key: "language_model")` wrapper (`Qwen35.swift:642, 667-672`) so module paths match both checkpoint and quant-dict keys. Flat `minimax_m3` conversions (upstream PR #1401 style) have the inverse layout — support them by sanitize-re-keying flat weights INTO the `language_model.`-prefixed layout.
- `sanitize` specifics: fused pre-stacked expert weights (`...block_sparse_moe.switch_mlp.gate_up_proj` / `down_proj` — see ^mv9aq7w) load directly; keep M2's per-expert `w1/w2/w3` stacking loop only as a fallback for unconverted per-expert checkpoints. Sparse-attention index-head weights are confirmed present in the checkpoint as `index_q_proj` / `index_k_proj` — this chain implements real MSA in ^8dbc476, so the dense-only stage must handle them deliberately (strip until ^8dbc476 constructs the indexer, mirroring the vision/MTP un-drop pattern). Keep M2's fp8 `weight_scale_inv` block-dequant path for bf16/fp8 originals. Vision/MTP weights appear absent from the mlx-community 4-bit checkpoint — confirm names against the safetensors index.
- Test ideas folded: per-module quantization unit test — with a synthetic quant dict containing `language_model.model.layers.N.block_sparse_moe.gate: 8-bit`, the module-path lookup resolves the gate to 8 bits (pins the no-re-keying invariant); sanitize unit tests with synthetic weight dicts (flat→prefixed re-keying, index_q/k_proj handling, vision/mtp keys dropped, per-expert fallback stacking); incremental decode with KV cache matches full-context forward on the same token sequence (per-step logits close within dtype tolerance).

## Acceptance Criteria

- [ ] Tiny-config (e.g. 2 dense + 2 MoE layers, hidden 32) forward pass produces `(B, L, vocab)` logits, runs incrementally with a KV cache, and is deterministic under a fixed `MLXRandom.seed`
- [ ] Partial rotary verified: only the first 64 of 128 head dims are rotated (test against a hand-rolled reference on one head)
- [ ] `sanitize` on a synthetic weight dict containing vision/MTP/expert keys drops vision + MTP keys and produces exactly the model's expected parameter set (`loadWeights`-style verification, mirroring `Tests/MLXLMTests/LoadWeightsTests.swift` patterns)
- [ ] `newCache(parameters:)` returns 60 `KVCacheSimple` entries (dense pass)

## Tests

- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: config decode from a literal JSON snippet of the real config.json, tiny-model forward/cache tests, partial-rotary unit test, sanitize test
- [ ] Run: `swift test --filter MLXLMTests.MiniMaxM3` → passes; `swift test --filter MLXLMTests` → no regressions

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax #minimax-m3