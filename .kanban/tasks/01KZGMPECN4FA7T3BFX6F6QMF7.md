---
assignees:
- claude-code
depends_on:
- 01KZGMN4FQKCVPAYJDJJHK0APK
position_column: todo
position_ordinal: '8180'
title: Port DeepseekV4Configuration (deepseek_v4 config decoding)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` — the `Codable` config struct for `model_type == "deepseek_v4"`. Foundation for every other DSV4 task; no model code yet.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (437 lines). Cross-check key names against the real checkpoint: `https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/raw/main/config.json`.

Verified target values from that config.json: `hidden_size=4096`, 43 layers, 64 attention heads, `num_key_value_heads=1`, `head_dim=512`, `max_position_embeddings=1048576`, 256 routed experts, 6 experts per token, 1 shared expert, q/o LoRA rank 1024.

Config keys the reference decodes (from its `CodingKeys`): `hc_sinkhorn_iters`, `hc_eps`, `rope_theta`, `compress_rope_theta`, `rope_scaling`, `sliding_window`, `compress_ratios`, `index_n_heads`, `index_head_dim`, `index_topk`, `use_attn_sink`, `dspark_block_size`, `dspark_noise_token_id`, `dspark_target_layer_ids`, `dspark_markov_rank`, plus nested quant plans (`routed_expert`, `default_bits`, `routed_layer_bits`, `routed_expert_bits`, `mxtq_bits`, `routed_expert_bit_plan`, `routed_experts`, `bit_plan`).

Scope decisions for this task:
- **Include** the `dspark_*` keys as decoded-but-unused stored properties so unknown-key handling never trips; DSpark behavior itself is out of scope (see the out-of-scope task).
- **Skip** the `mxtq_bits` / JANGTQ nested plans entirely — that is a separate quant format we do not support (out of scope).
- Provide the reference's derived helpers: `isHashLayer(_:)`, `hasCompressor(_:)`, `ropeTheta(forLayer:)`. Per-layer theta selects `rope_theta` when `compress_ratio == 0` and `compress_rope_theta` otherwise.
- Follow this repo's existing config idiom — see `Libraries/MLXLLM/Models/DeepseekV3.swift` for the sibling `DeepseekV3Configuration`. Do **not** modify `DeepseekV3.swift`; it still serves DSV3/DSV3.2/Kimi/GLM-5.1/Nemotron bundles.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (MIT).
- Upstream copyright per file header: Osaurus AI, `SPDX-License-Identifier: MIT`. Apply the header decided in task `jhk0apk`.
- Gap tracker cross-reference: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` exists with `public struct DeepseekV4Configuration: Codable, Sendable`.
- [ ] Decoding the real `config.json` from `mlx-community/DeepSeek-V4-Flash-4bit` succeeds and yields the verified values above.
- [ ] `isHashLayer`, `hasCompressor`, and `ropeTheta(forLayer:)` return correct values for layers 0, 2, 3, and 42.
- [ ] Absent/optional keys fall back to documented defaults rather than throwing.
- [ ] `DeepseekV3.swift` is unmodified (`git diff --stat` shows no change to it).

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` with a checked-in fixture copy of the model's `config.json` (weights not needed).
- [ ] Test: decodes the fixture; asserts hiddenSize 4096, 43 layers, 64 heads, kvHeads 1, headDim 512, maxPositionEmbeddings 1048576, 256 routed experts, 6 per token, 1 shared.
- [ ] Test: `isHashLayer(0..2) == true`, `isHashLayer(3) == false`.
- [ ] Test: `ropeTheta(forLayer:)` picks `compress_rope_theta` where `compress_ratios[i] != 0` and `rope_theta` where it is 0.
- [ ] Test: a minimal JSON with only required keys decodes without throwing.
- [ ] Run: `swift test --filter DeepseekV4ConfigurationTests` — all pass.

## Workflow
- Use `/tdd` — write the failing decode tests against the real fixture first, then implement the struct.
#deepseek-v4