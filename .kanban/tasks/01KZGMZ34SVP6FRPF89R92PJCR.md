---
assignees:
- claude-code
depends_on:
- 01KZGMY5D5PRZ67GNK9E7B24WS
position_column: todo
position_ordinal: 8b80
title: Port DeepseekV4 Indexer (causal top-k selection for sparse attention)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Indexer.swift` — the top-k key-selection half of DSV4's compressed sparse attention. The `Compressor` half is a separate follow-on task; this one is split out because the reference file is 1164 lines, well past the 500-line guideline, and the indexer's causal top-k is independently testable.

**Deliberately deferred until basic support works.** Everything before this ships a working DSV4 for short prompts; `sanitize` in task `pwr8r3h` drops `compressor.*` and `indexer.*` keys precisely so the model loads without either half. This task removes the `indexer.*` part of that filter only; the `compressor.*` filter stays until the Compressor task.

Port the `Indexer` portion of `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`.

Config drivers, already decoded by task `6f6qmf7`: `index_n_heads`, `index_head_dim`, `index_topk`.

`osaurus-ai/vmlx-swift-lm` ships a dedicated `Tests/MLXLMTests/DeepseekV4IndexerCausalTopKTests.swift`, which confirms this is a real seam and tells you what to assert.

Second reference worth reading: upstream `mlx-lm`'s `mlx_lm/models/deepseek_v32.py` is the same sparse-attention family. We have no `deepseek_v32` port to build on.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`, `Indexer` portion (MIT; header attributes Osaurus AI).
- Test-shape hint: `osaurus-ai/vmlx-swift-lm` — `Tests/MLXLMTests/DeepseekV4IndexerCausalTopKTests.swift`.
- Family reference: `ml-explore/mlx-lm` `mlx_lm/models/deepseek_v32.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4Indexer.swift` exists with the `Indexer` module.
- [ ] Top-k selection is strictly causal: for every query position, every selected key index is less than or equal to the query index.
- [ ] Exactly `index_topk` keys are selected per query when enough history exists, and fewer (not garbage) when it does not.
- [ ] The `indexer.*` drop is removed from `DeepseekV4Model.sanitize` and those weights load; the `compressor.*` drop remains, with a comment naming the Compressor task.
- [ ] Short-prompt output unchanged from before this task.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4IndexerTests.swift`.
- [ ] Test: randomized causal-mask property test over many shapes — no selected index ever exceeds the query index. This is the off-by-one guard.
- [ ] Test: selection count equals `index_topk` for a long-enough sequence, and degrades gracefully at positions 0 and 1.
- [ ] Test: `sanitize` no longer drops `indexer.*` but still drops `compressor.*`.
- [ ] Regression: `swift test --filter DeepseekV4ModelTests` still green — short prompts unaffected.
- [ ] Run: `swift test --filter DeepseekV4IndexerTests` — all pass.

## Workflow
- Use `/tdd` — write the causal property test first; it is cheap and catches the classic off-by-one.
#deepseek-v4