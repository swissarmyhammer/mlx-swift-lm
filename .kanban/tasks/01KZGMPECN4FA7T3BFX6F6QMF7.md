---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kznzq1w1d3cc37ytbggywyn9
  text: |-
    Research notes for the next agent.

    Sources read (real paths, real SHAs — both public, MIT):
    - `osaurus-ai/vmlx-swift-lm` @ `b166896353b9c95d773de993990c20a0b5ba6905` — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift`, 393 lines. This is the attribution source that `CONTRIBUTING.md` names. Last commit for that path: "fix(dsv4): preserve jangtq-k routed bit plan", 2026-05-12.
    - `scouzi1966/mlx-swift-lm` @ `e1852869ce61ded0d23b76df3757e9b75c77c1f5` — same path, 437 lines. It is a transcription of the file above with the four `dspark_*` keys, an `activationQATEnabled` flag, and no `RuntimeMoETopKOverride`. Both files give the same `Copyright © 2026 Osaurus AI` line.

    Discoveries:
    - The `dspark_*` keys exist ONLY in the scouzi copy, not in the osaurus file. The new file thus names both sources in its header — osaurus in the block that `CONTRIBUTING.md` fixes, scouzi in a note below it. Both repositories are already in `THIRD-PARTY-NOTICES.md`, thus no notice entry was added.
    - `RuntimeMoETopKOverride` (used by the osaurus `init`) does not exist in this repository. The port reads `num_experts_per_tok` straight from the file.
    - Real `config.json` shape: `compress_ratios` holds 44 entries against 43 layers (the extra entry is the MTP layer). Layers 0 and 1 hold 0, layer 2 holds 4, layer 3 holds 128, layer 42 holds 4, layer 43 holds 0. `use_attn_sink` and every `dspark_*` key are absent, thus the tests examine the defaults for them.
    - The real file holds both `quantization` and `quantization_config`, each with 644 keys. Neither is decoded — the JANGTQ/MXTQ plans stay out of scope per this card.

    Not done, and why:
    - `activationQATEnabled` from the scouzi copy is not a `config.json` key (that copy hardcodes it to `true`). It belongs with the attention/indexer numerics, not with config decoding, thus it is left for the attention task.

    Deviation to note for review: the card names the helper `hasCompressor(_:)`. The implementation uses `hasCompressor(layer:)`, which is the reference's own signature, and which the Swift naming rule requires — a bare `Int` argument needs a noun that says what it is. `isHashLayer(_:)` and `ropeTheta(forLayer:)` match the card exactly.
  timestamp: 2026-08-10T14:02:37.825867+00:00
- actor: claude-code
  id: 01kznzqvyxccdr7s0etqxmdx7w
  text: |-
    ### implement — changed
    - evidence: 4 files — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (new), `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` (new, 9 tests), `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` (new fixture, a copy of the real `config.json`), `Package.swift` (the fixture added to the `MLXLMTests` resources).
    - TDD: the test came first and failed with "cannot find type 'DeepseekV4Configuration' in scope". After the struct landed, `swift test --filter DeepseekV4ConfigurationTests` gave 9 tests, 0 failures.
    - `swift build`: complete, no compiler warning. Full `swift test`: exit 0, 0 failures. The 2 skips are the pre-existing, documented skips in `CompiledDecodeCorrectnessTests`.
    - `swift-format lint --configuration .swift-format` is clean on the three changed Swift files, and a format run makes no further change.
    - `DeepseekV3.swift` is unmodified.
    - Note: a full-tree `swift-format` run reformatted 50 unrelated files (the installed swift-format is 603.0.0 and disagrees with the version that formatted the repository). Those files were reverted. Format only the files you change.
    - next: `/review`
  timestamp: 2026-08-10T14:03:04.541638+00:00
depends_on:
- 01KZGMN4FQKCVPAYJDJJHK0APK
position_column: doing
position_ordinal: '80'
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

- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` exists with `public struct DeepseekV4Configuration: Codable, Sendable`.
- [x] Decoding the real `config.json` from `mlx-community/DeepSeek-V4-Flash-4bit` succeeds and yields the verified values above.
- [x] `isHashLayer`, `hasCompressor`, and `ropeTheta(forLayer:)` return correct values for layers 0, 2, 3, and 42.
- [x] Absent/optional keys fall back to documented defaults rather than throwing.
- [x] `DeepseekV3.swift` is unmodified (`git diff --stat` shows no change to it).

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` with a checked-in fixture copy of the model's `config.json` (weights not needed).
- [x] Test: decodes the fixture; asserts hiddenSize 4096, 43 layers, 64 heads, kvHeads 1, headDim 512, maxPositionEmbeddings 1048576, 256 routed experts, 6 per token, 1 shared.
- [x] Test: `isHashLayer(0..2) == true`, `isHashLayer(3) == false`.
- [x] Test: `ropeTheta(forLayer:)` picks `compress_rope_theta` where `compress_ratios[i] != 0` and `rope_theta` where it is 0.
- [x] Test: a minimal JSON with only required keys decodes without throwing.
- [x] Run: `swift test --filter DeepseekV4ConfigurationTests` — all pass.

## Workflow
- Use `/tdd` — write the failing decode tests against the real fixture first, then implement the struct.
#deepseek-v4