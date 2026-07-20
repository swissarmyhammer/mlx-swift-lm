---
position_column: todo
position_ordinal: '8180'
title: 'MiniMax M3: MiniMaxM3Configuration decodes flat and VL-nested configs'
---
#minimax-m3

## What
Start `Libraries/MLXLLM/Models/MiniMaxM3.swift` with the configuration layer for MiniMax M3 text-backbone support. M3 is a 428B-total / ~23B-active hybrid dense+MoE model, natively multimodal; we implement the text backbone the same way `Gemma3Text.swift` / `Mistral3Text.swift` do for their VL families.

- Download `config.json` from https://huggingface.co/mlx-community/MiniMax-M3-4bit and check it in as a test fixture (follow the repo's existing fixture conventions in `Tests/MLXLMTests`; embedding the JSON in the test file is acceptable).
- `MiniMaxM3Configuration: Codable, Sendable` must decode BOTH shapes: flat text config (`model_type: "minimax_m3"`, produced by upstream mlx-lm PR #1401-style conversions) and VL-nested (`model_type: "minimax_m3_vl"`, `architectures: ["MiniMaxM3SparseForConditionalGeneration"]`, text keys nested under `text_config`). Follow `Gemma3Text.swift`'s custom `init(from:)` fallback-to-`text_config` pattern (see its `CodingKeys.textConfig`).
- Expected values (from the mlx-community config): hidden_size 6144, num_hidden_layers 60, num_attention_heads 64, num_key_value_heads 4, head_dim 128, intermediate_size 3072 (per-expert width), dense_intermediate_size 12288, num_local_experts 128, num_experts_per_tok 4, n_shared_experts 1, shared_intermediate_size 3072, scoring_func "sigmoid", use_routing_bias true, routed_scaling_factor 2.0, MoE layer schedule = layers 0-2 dense / 3-59 MoE (decode the schedule key as it actually appears in the fixture — moe_layer_freq or mlp layer types), hidden_act "swigluoai", swiglu_alpha 1.702, swiglu_limit 7.0, use_qk_norm true, qk_norm_type "per_head", use_gemma_norm true, rms_norm_eps 1e-6, rope_theta 5000000, rotary_dim 64 (partial_rotary_factor 0.5), vocab_size 200064, max_position_embeddings 1048576, tie_word_embeddings false. Decode the sparse-attention block (use_sparse_attention, sparse_topk_blocks 16, sparse_block_size 128, sparse_init_block, sparse_local_block, sparse_attention_freq) — later tasks use it only to document the dense fallback. Tolerate and ignore vision/MTP/projector keys.
- Provide sensible defaults so a tiny synthetic config (few layers, few experts) constructs without a full config file — needed by later unit tests.

## Acceptance Criteria
- [ ] Real mlx-community fixture config decodes with all values above asserted
- [ ] A synthetic flat `minimax_m3` JSON decodes to the same field values
- [ ] Unknown keys (vision_config, mtp, quantization) do not fail decoding
- [ ] Public/internal API documented; no changes to existing MiniMax (M2) code

## Tests
- [ ] `Tests/MLXLMTests/MiniMaxM3Tests.swift` (config section): fixture decode assertions, flat-vs-nested equivalence, tiny-config defaults
- [ ] Run `swift test --filter MiniMaxM3`; expect pass

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.