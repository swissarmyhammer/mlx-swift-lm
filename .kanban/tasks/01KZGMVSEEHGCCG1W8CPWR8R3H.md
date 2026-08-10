---
assignees:
- claude-code
depends_on:
- 01KZGMSFQN4AS74HYXAAG7ANT0
- 01KZGMT5M19CHE8SK3FSBWYQ83
- 01KZGMTXP266RDESZHBG5E7907
position_column: todo
position_ordinal: '8780'
title: Assemble DeepseekV4 decoder layer, model, and weight sanitize
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4.swift` — the decoder layer and top-level model that compose the attention, mHC, and MoE pieces built in the preceding tasks.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4DecoderLayer` at line 1569, `DeepseekV4ModelInner` at line 1746, `DeepseekV4Model` at line 1845. Skip everything DSpark (`DeepseekV4DSparkMarkovHead:757`, `DeepseekV4DSparkConfidenceHead:777`, `DeepseekV4DSparkProposal:792`, `DeepseekV4DSparkStage:800`, `DeepseekV4DSparkGenerator:2355`) — out of scope.

Pieces:

1. `DeepseekV4DecoderLayer` — mHC pre → attention → mHC post → mHC pre → MoE (or dense MLP for the dense prefix layers) → mHC post.
2. `DeepseekV4ModelInner` — embedding, 43 layers, `_hc_head_reduce`, final norm.
3. `DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel` — matching the reference's conformances. `lm_head` must handle tied embeddings (this repo has hit tied-embedding bugs before: see commits `fed151f` / `4805454` for GLM4 — apply the same optional-`lm_head` pattern).
4. **`sanitize(weights:metadata:)`** — the load-side filtering the gap tracker calls out:
   - Drop `mtp.0.*` keys (multi-token-prediction head; we do not use it here).
   - Drop `compressor.*` and `indexer.*` keys until sparse attention lands (task `<sparse-attn>` — remove this filter when it does).
   - Stack per-expert routed weights into `switch_mlp.*` for `SwitchGLU`, mirroring how `DeepseekV3.swift` and the other MoE models here do it.
   - Load the `tid2eid` int64 hash table for the hash-routing layers.

Also port the env-gated numeric tracer from the reference (its `VMLX_DSV4_NUMERIC_TRACE`, reference lines ~60-84) under a repo-appropriate env var name — it is how you will bisect a numeric mismatch against Python without a debugger, and it costs nothing when off.

Do not modify `DeepseekV3.swift`.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~1569-2354 plus the tracer at ~60-84 (MIT; header attributes Osaurus AI).
- Sanitize/load gap list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4.swift` exists with `DeepseekV4DecoderLayer`, `DeepseekV4ModelInner`, and `DeepseekV4Model` conforming to `LLMModel`, `KVCacheDimensionProvider`, and `LoRAModel`.
- [ ] `sanitize` drops `mtp.0.*`, `compressor.*`, and `indexer.*` keys, and stacks routed experts into `switch_mlp.*`.
- [ ] Tied-embedding checkpoints load without a missing-`lm_head` error; untied ones use the checkpoint's `lm_head`.
- [ ] A synthetic-weight forward produces logits of shape `[B, L, vocabSize]`, `allFinite`.
- [ ] The `compressor.*`/`indexer.*` filter carries a code comment naming the sparse-attention task that removes it.
- [ ] No DSpark types present.
- [ ] `DeepseekV3.swift` unmodified.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4ModelTests.swift`, tiny synthetic config (e.g. 4 layers, hidden 64, 8 experts, small vocab) and random weights — no download.
- [ ] Test: prefill forward `[1, 8]` token ids → logits `[1, 8, vocab]`, `allFinite`.
- [ ] Test: decode forward with cache `[1, 1]` → `[1, 1, vocab]`; after N steps cache offset == N.
- [ ] Test: `sanitize` on a synthetic weight dict containing `mtp.0.foo`, `compressor.bar`, `indexer.baz`, and per-expert keys — asserts the first three are gone and the experts are stacked.
- [ ] Test: tied-embedding path — config with `tie_word_embeddings=true` and no `lm_head` weight loads and runs.
- [ ] Run: `swift test --filter DeepseekV4ModelTests` — all pass.

## Workflow
- Use `/tdd` — the `sanitize` test is pure dictionary manipulation and should be written first; it needs no model at all.
#deepseek-v4