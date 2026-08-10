---
assignees:
- claude-code
depends_on:
- 01KZGMZ34SVP6FRPF89R92PJCR
position_column: todo
position_ordinal: 8d80
title: Port DeepseekV4 Compressor (pooled-KV long-context attention)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift` — the pooled-KV half of DSV4's compressed sparse attention, and the piece that unlocks its 1M-token context. Follows the Indexer task (`r92pjcr`); split out because the combined reference file is 1164 lines.

Port the `Compressor` portion of `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`.

Config drivers, already decoded by task `6f6qmf7`: `sliding_window=128` and `compress_ratios[i]` in `{4, 128, 0}`. A layer with `compress_ratio == 0` uses plain attention; nonzero values engage the compressor at that pooling ratio. Note that `compress_ratio` also selects the RoPE theta per layer (already handled in the attention task `ag7ant0`) — make sure the two agree on which layers are compressed.

This task removes the remaining `compressor.*` drop from `DeepseekV4Model.sanitize`.

`osaurus-ai/vmlx-swift-lm` also ships `Tests/MLXLMTests/DeepseekV4CacheDiskRoundTripTests.swift`, suggesting the compressed cache needs to survive a serialize/deserialize round trip — worth covering if this repo persists caches.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`, `Compressor` portion (MIT; header attributes Osaurus AI).
- Test-shape hint: `osaurus-ai/vmlx-swift-lm` — `Tests/MLXLMTests/DeepseekV4CacheDiskRoundTripTests.swift`.
- Family reference: `ml-explore/mlx-lm` `mlx_lm/models/deepseek_v32.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift` exists with the `Compressor` module, wired to the Indexer from task `r92pjcr`.
- [ ] The `compressor.*` drop is removed from `DeepseekV4Model.sanitize`; no drop filters for DSV4 remain.
- [ ] Layers with `compress_ratio == 0` take the plain-attention path and are bit-identical to the pre-task behavior.
- [ ] The set of layers the compressor treats as compressed matches the set for which `ropeTheta(forLayer:)` returns `compress_rope_theta`.
- [ ] A prompt longer than `sliding_window=128` produces coherent output.
- [ ] Short-prompt output unchanged.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4CompressorTests.swift`.
- [ ] Test: pooled-KV output shape for `compress_ratio` 4 and 128; `allFinite`.
- [ ] Test: `compress_ratio == 0` path is bit-identical to plain attention on the same inputs.
- [ ] Test: compressed-vs-RoPE layer-set agreement — assert the two predicates return the same set over all 43 layers of the real config.
- [ ] Test: `sanitize` retains `compressor.*` and `indexer.*` keys (inverse of the assertion added in `pwr8r3h`).
- [ ] Test: a sequence of length 512 (past the 128 window) runs and produces `allFinite` logits.
- [ ] Regression: `swift test --filter 'DeepseekV4ModelTests|DeepseekV4IndexerTests'` still green.
- [ ] Run: `swift test --filter DeepseekV4CompressorTests` — all pass.

## Workflow
- Use `/tdd` — the `compress_ratio == 0` bit-identity test is the regression guard; write it before touching the attention path.
#deepseek-v4