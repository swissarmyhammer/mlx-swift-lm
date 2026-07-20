---
depends_on:
- 01KXX9BTB2P9NK0TW78S2AJTYR
position_column: todo
position_ordinal: '8580'
title: 'MiniMax M3: model assembly, weight sanitize, factory registration'
---
#minimax-m3

## What
Assemble the full model in `Libraries/MLXLLM/Models/MiniMaxM3.swift`, implement weight sanitization, and register the model types — after this task, `swift test` proves a tiny M3 end to end.

- `MiniMaxM3DecoderLayer` (attention + dense-or-MoE MLP per schedule, Gemma-norm input/post-attention layer norms), `MiniMaxM3ModelInner` (embed_tokens, 60 layers, final norm), `MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider` with untied `lm_head` (`tie_word_embeddings: false`), plus a `LoRAModel` extension — mirror `MiniMax.swift`'s structure.
- **CRITICAL — keep the `language_model.` prefix as a module path; do NOT re-key it away.** The VL checkpoint's weights AND its per-module quantization overrides are keyed `language_model.model.layers.N...` (e.g. `language_model.model.layers.N.block_sparse_moe.gate` is 8-bit while the default is 4-bit gs64 affine). `Load.swift:64-71` resolves quantization by module path (`perLayerQuantization.quantization(layer: path)`) and `LLMModelFactory.swift:736-738` passes the quant dict through unmodified — re-keying weights to `model.*` makes the 8-bit gate lookup miss, fall back to 4-bit, and fail shape-check at load (and it would only surface on a 214 GB machine in the integration task). Follow the `Qwen35.swift` precedent: `@ModuleInfo(key: "language_model")` wrapper (`Qwen35.swift:642, 667-672`) so module paths match both checkpoint and quant-dict keys. The VL layout is the PRIMARY layout; flat `minimax_m3` conversions (upstream PR #1401 style) have the inverse layout — support them by sanitize-re-keying flat weights INTO the `language_model.`-prefixed layout.
- `sanitize(weights:)`:
  - Fused pre-stacked expert weights (`switch_mlp.gate_up_proj` / `down_proj` — see the MoE task) load directly; keep M2's per-expert `w1/w2/w3` stacking loop only as a fallback for unconverted per-expert checkpoints.
  - Strip sparse-attention index-head weights — confirmed present in the checkpoint as `index_q_proj` / `index_k_proj` (unused in the dense fallback).
  - Strip vision weights (`vision_tower.*`, `multi_modal_projector.*`) and MTP module weights — these appear absent from the mlx-community 4-bit checkpoint, so stripping is harmless robustness for other conversions; confirm names against the safetensors index.
  - Keep M2's fp8 `weight_scale_inv` block dequant path for bf16/fp8 originals.
- Register in `LLMModelFactory.swift`: both `"minimax_m3"` and `"minimax_m3_vl"` -> `create(MiniMaxM3Configuration.self, MiniMaxM3Model.init)` — precedent: `"gemma3"`/`"gemma3_text"` both map to Gemma3Text (`LLMModelFactory.swift:35-36`).
- Do not modify the existing `"minimax"` (M2) registration or `MiniMax.swift`.

## Acceptance Criteria
- [ ] Tiny-model forward pass: scaled-down config (e.g. 4 layers = 2 dense + 2 MoE, 8 experts, hidden 64, vocab 128) produces logits `(B, L, vocab)`
- [ ] Incremental decode with KV cache matches full-context forward on the same token sequence (per-step logits close within dtype tolerance)
- [ ] Per-module quantization unit test: with a synthetic quant dict containing `language_model.model.layers.N.block_sparse_moe.gate: 8-bit`, the module-path lookup resolves the gate to 8 bits (pins the no-re-keying invariant)
- [ ] `sanitize` unit tests with synthetic weight dicts: flat->prefixed re-keying, index_q/k_proj dropped, vision/mtp keys dropped, per-expert fallback stacking
- [ ] Both model type strings resolve in the factory (see `LLMRegistryTests.swift` conventions)
- [ ] `swift build && swift test` green; existing MiniMax (M2) tests untouched and passing

## Tests
- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: tiny forward + KV-cache-consistency + quant-path + sanitize + registration tests
- [ ] Run `swift test --filter MiniMaxM3` and the full `swift test`; expect pass

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax