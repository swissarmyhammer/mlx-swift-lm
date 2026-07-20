---
assignees:
- claude-code
depends_on:
- 01KXY0Z94XT2HF9RPM3XGVTH41
- 01KXY0ZVCCPBKZ1ANETWZ8Y8QQ
position_column: todo
position_ordinal: 8c80
title: 'MiniMax-M3: MSA sparse attention — indexer, block top-k selection, sparse KV cache'
---
## What

Replace ^xgvth41's dense-only attention with real MiniMax Sparse Attention (MSA) on layers 3–59, mirroring mlx-vlm's `language.py`:

1. **`MiniMaxM3Indexer`** (inside `Libraries/MLXVLM/Models/MiniMaxM3.swift`): learnable `index_q_proj`/`index_k_proj` (index_heads 4, index_dim 128) with their own per-head RMSNorms and RoPE, scoring key blocks (block size 128) per query, top-k 16 block selection (`max` score type per the config; support `lse` if the reference does), causal masking with init/local block handling.
2. **`MiniMaxM3KVCache`**: a custom cache type (conforming to `MLXLMCommon.KVCache` — see `Libraries/MLXLMCommon/KVCache.swift`) that stores regular K/V plus `index_keys` and an `index_offset`, mirroring the reference's sparse-aware cache. Lives model-local in MiniMaxM3.swift unless a shared home is clearly better. `newCache(parameters:)` returns this type for sparse layers, `KVCacheSimple` for layers 0–2.
3. **Dense fallback** exactly like the reference: when total blocks ≤ topk (sequence ≤ 2048 tokens) or during short prefill, run dense attention — the results are numerically identical there, which is also the equivalence test.

Known interaction to document (not fix): `PromptCache.isChunkable`/`isHybridMambaAttention` (MLXFoundationModels) recognize neither this cache type — M3 sessions simply won't participate in prompt-cache reuse; `MLXLanguageModel.supportsPromptCacheReuse` correctly reports `false`. State this in a doc comment on `MiniMaxM3KVCache`.

## Acceptance Criteria

- [ ] For sequences ≤ 2048 tokens, sparse and dense paths produce identical logits on a tiny config (max-abs-diff ≤ 1e-5) — the exactness window is the regression anchor
- [ ] For sequences > topk×block on a tiny config (shrink block/topk in the test config, e.g. block 4 / topk 2), the indexer selects the expected blocks for a hand-constructed input, and generation runs incrementally through `MiniMaxM3KVCache` without shape errors
- [ ] Real-weights coherence test from ^wz8y8qq still passes with sparse attention active
- [ ] `isTrimmable == false` on the sparse cache and a doc comment records the PromptCache non-participation

## Tests

- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: sparse==dense equivalence test, indexer block-selection test with shrunk block/topk, incremental decode test through the custom cache
- [ ] Run: `swift test --filter MLXLMTests` → green; re-run the ^wz8y8qq integration coherence case → passes

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax