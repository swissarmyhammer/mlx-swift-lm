---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kyakq4gxa30kdw424z16dd5b
  text: |-
    Implementation complete, tests green. Summary:

    **Research phase**: Fetched the real mlx-vlm `minimax_m3_vl/language.py` (2282 lines) and `config.py` from GitHub directly (network access available) to get the exact MSA algorithm rather than guessing — this surfaced the real formulas for `_build_sparse_causal_mask_compiled`/`select_blocks`/`build_block_mask`, the `MiniMaxM3Indexer`/`MiniMaxM3KVCache` Python classes, and the real `sparse_attention_config` defaults (`sparse_init_block`: 0, `sparse_local_block`: 1, `sparse_score_type`: "max", `sparse_index_dim` defaults to `head_dim` when absent, `sparse_num_index_heads`: 4).

    **Design decisions**:
    - **Block scoring/selection**: ported the *generic* (mask-free) branch of `select_blocks` exactly — intra-block max-or-lse aggregation (config `scoreType`), then always-max aggregation across index heads, forced `initBlocks`/`localBlocks` via huge sentinel scores (1e30/1e29), `argPartition`-based top-k, ascending sort with `-1` sentinel for unused slots. Deliberately did NOT port the two Metal-kernel fast paths (`_minimax_m3_sparse_prefill_one_pass_kernel`, `_select_sparse_block_indices_compiled`) or the "valid"/padding-mask plumbing — this Swift port has no batched left-padding, and a plain boolean mask fed to `MLXFast.scaledDotProductAttention` is mathematically identical to gather-based sparse attention (masking zeroes exactly the same keys a gather would exclude), just without the extra performance optimization. Documented this trade-off in `MiniMaxM3Indexer`'s doc comment.
    - **Dense-fallback threshold**: exactly `totalLen > topkBlocks * blockSize` (else `nil` from the indexer, causer falls back to the existing dense causal mask) — this reduces to bit-identical dense attention when true, since every causal block is selected when there are ≤ topk causal blocks available.
    - **Sparse-layer eligibility**: reused the already-verified `moeLayerFreq`/`isMoELayer` schedule (layers 3-59) rather than decode the unverified `sparse_attention_freq`/`use_sparse_attention`/`sparse_disable_index_value` config fields (no real values ever verified for those in this chain — documented in `MiniMaxM3SparseAttentionConfiguration`'s doc comment).
    - **Cache design**: `MiniMaxM3KVCache: KVCache` (not a `KVCacheSimple` subclass, to control `innerState()` fully) composing a private `KVCacheSimple` for K/V plus an independent `indexKeys`/`indexOffset` buffer with its own step-256 growth logic, mirroring the reference's `MiniMaxM3KVCache.update_index_and_fetch`. `isTrimmable == false` (task-mandated; the reference supports trim, this Swift port's first landing deliberately doesn't). Doc comment records the `PromptCache.isChunkable`/`isHybridMambaAttention` non-recognition and resulting `supportsPromptCacheReuse == false`, per the task's "document, don't fix" instruction.
    - **Indexer as non-Module**: `MiniMaxM3Indexer` is a plain struct (mirrors the Python reference, which is also a plain class, not an `nn.Module`) that borrows references to `index_q_proj`/`index_k_proj`/`index_q_norm`/`index_k_norm` — those submodules live directly on `MiniMaxM3Attention` (checkpoint key `self_attn.index_q_proj`, not `self_attn.indexer.index_q_proj`) so checkpoint loading isn't broken.
    - **sanitize**: stopped stripping `self_attn.index_*` weights in `_filterUnusedWeights` (they're loaded now, not dropped).

    **Tests** (`Tests/MLXLMTests/MiniMaxM3Tests.swift`): added 4 new tests — sparse-vs-dense equivalence (copies weights between a dense-only and a sparse-capable `MiniMaxM3Attention`, asserts max-abs-diff ≤ 1e-5 on an 8-token sequence, well under the 2048-token default threshold); indexer block-selection with a hand-constructed input (shrunk block=4/topk=2, `rotaryDim: 2` with vectors placed in the non-rotating tail dims so the expected-block math is exactly hand-computable, not approximate); incremental decode through `MiniMaxM3KVCache` (shrunk block=2/topk=2) verifying shapes across a prefill + 5 single-token decode steps; `MiniMaxM3KVCache.isTrimmable == false`. Also fixed 3 pre-existing tests whose assertions were invalidated by this task's intended behavior change (`newCacheReturnsOneSimpleCachePerLayer`/`newCacheReturns60EntriesForRealSchedule` now expect `MiniMaxM3KVCache` for MoE layers instead of `KVCacheSimple`; the sanitize test no longer injects wrong-shaped garbage `self_attn.index_*` keys since those are real, correctly-shaped parameters now).

    **Dead end**: initially tried disabling RoPE entirely (`rotaryDim: 0`) for the hand-constructed indexer test to make the block-selection math trivial — `MLXFast.RoPE` fatal-errors on `dims: 0` ("dims must be positive but got 0"). Fixed by using `rotaryDim: 2` with indexDim: 4 and placing the hand-crafted vectors' only nonzero component in dims 2/3 (beyond rotaryDim), which pass through RoPE completely unchanged regardless of position — exact, not approximate.

    **Test results**: `swift build --build-tests` clean, zero warnings from touched files. `swift test --filter MLXLMTests.MiniMaxM3` — 40/40 pass (36 existing + 4 new). `swift test --filter MLXLMTests` — 340/340 pass across 24 suites, no regressions.

    **Real-weights coherence re-test (^wz8y8qq)**: could not run — same bandwidth constraint documented on prior tasks in this chain. Verified concretely this session: the `mlx-community/MiniMax-M3-4bit` checkpoint is NOT cached locally (only a stale `.locks` entry, no snapshot/blobs — confirmed via `~/.cache/huggingface/hub`), the real `model.safetensors.index.json` reports `total_size: 241,480,683,520` bytes (~225 GiB, at the high end of the 120-214GB range prior tasks estimated), and a fresh throughput measurement (100MB ranged GET from the actual HF-hosted shard) measured ~6.18 MB/s sustained — implying ~10.9 hours minimum for a full download, not feasible within this session. Physical memory (512GB) does clear the integration test's 220GB gate, so the test itself would run if the checkpoint were available.

    Leaving in `doing` per workflow (orchestrator's /review step moves to review).
  timestamp: 2026-07-24T17:45:37.053641+00:00
- actor: claude-code
  id: 01kyan80pyafvtkectrzsbvmmc
  text: |-
    Adversarial double-check (via really-done) returned REVISE with two substantive findings plus a disclosure note; both actionable findings fixed:

    1. **Untested `sparse_score_type == "lse"` branch** (the least-safe branch, since it's the one where the NaN guard actually matters — `logSumExp` on an all-`-inf` masked block can produce NaN via `-inf - (-inf)`, whereas plain `max` never does). Added `indexerSelectsExpectedBlocksWithLogSumExpScoreType` to `Tests/MLXLMTests/MiniMaxM3Tests.swift`: reuses the existing hand-constructed block-selection scenario with `scoreType: "lse"`, asserting the same `[1, 2]` winner at the last query position (block 1's logsumexp over 4 equal high-similarity tokens still beats block 0's), AND asserts correctness at the first query position (position 0), where blocks 1/2 are entirely non-causal/masked — exactly the NaN-guard's target case — confirming no NaN leaks into the selection (block 0 is the only real selection, second slot is the `-1` sentinel).
    2. **Minor duplicated-literal**: `MiniMaxM3KVCache.indexStep` was a hardcoded `256` literal duplicating `KVCacheSimple`'s default `step`. Changed to a computed property `private var indexStep: Int { kvCache.step }` reading the wrapped cache's actual step value directly, eliminating the duplication (can't drift out of sync since there's only one source of truth now).
    3. **Disclosure note** (not a code fix): the reviewer flagged that my "verification" summary didn't explicitly call out that the ^wz8y8qq real-weights integration re-test was skipped — it *was* documented in my prior comment (checkpoint absence + measured ~6.2MB/s throughput + ~10.9h estimate) and in the task's Acceptance Criteria/Tests checkboxes, but noting it here again for full visibility: that item remains unresolved due to the same bandwidth constraint documented on prior tasks in this chain, not silently skipped.

    Also confirmed by the reviewer (no changes needed): the block-scoring math, KV-cache offset bookkeeping (including `innerState()`'s `Evaluatable` conformance), the four-way `resolveMask`/cache-type dispatch (dense-only / sparse+`MiniMaxM3KVCache` / sparse+`RotatingKVCache` / sparse+nil), the dense-fallback threshold's ordering relative to the cache update, and the hand-computed expected block selection in the original "max" test were all independently re-derived against the real mlx-vlm reference and found correct.

    Re-verified after fixes: `swift build --build-tests` clean (zero warnings from touched files), `swift test --filter MLXLMTests.MiniMaxM3` — 41/41 pass (was 40; +1 new lse test), `swift test --filter MLXLMTests` — 341/341 pass across 24 suites (no regressions).

    Task is green and ready for `/review`. Leaving in `doing` per workflow.
  timestamp: 2026-07-24T18:12:18.782296+00:00
- actor: claude-code
  id: 01kyaqbe4yxcp2bn9wr7dpskcd
  text: |-
    Fixed all 20 confirmed review findings from the 2026-07-24 13:18 pass, all in Libraries/MLXVLM/Models/MiniMaxM3.swift:

    - Marked 5 classes `final`: MiniMaxM3SparseMoeBlock, MiniMaxM3Attention, MiniMaxM3MLP, MiniMaxM3DecoderLayer, MiniMaxM3ModelInner. Verified none are subclassed anywhere in the repo (grepped for `: MiniMaxM3<Name>` — only property-type declarations matched, no subclass declarations) before adding `final`. (MiniMaxM3KVCache, MiniMaxM3LanguageModel, MiniMaxM3Model were already `final`.)
    - Added doc comments to all 9 flagged MiniMaxM3KVCache public members: `offset`, `maxSize`, `update(keys:values:)`, `innerState()`, `state`, `metaState`, `trim(_:)`, `makeMask(n:windowSize:returnArray:)`, `copy()`.
    - Replaced both `indexKeys!` force-unwraps with `guard let indexKeys else { fatalError(...) }` (in `updateIndexAndFetch` and the `state` setter's `case 3` branch). Confirmed via the surrounding control flow that the guard's else-branch is unreachable in both cases (mirrors the review's suggested fix).
    - Extracted named constants: `MiniMaxM3KVCache.expectedStateComponentsWithoutIndex`/`expectedStateComponentsWithIndex` (2/3) used in both the switch case labels and the `fatalError` message; `MiniMaxM3Model.languageModelPrefix` ("language_model.") used in `_filterUnusedWeights`'s ternary and lm_head-drop key, and in `_remapExpertWeights`'s layer-prefix construction; a local `gateWeightType` ("w1") constant in `_remapExpertWeights` used at all three "w1" call sites (existence guards + `collectExpertWeights` call) so the checkpoint-probe and collection paths can't drift.

    Verification: `swift build` — clean, exit 0, no new warnings. `swift test --filter MLXLMTests` — 341/341 passed, exit 0.

    Note: a fresh `review working` pass (2026-07-24 13:38, after these fixes) surfaced one new, unrelated finding — `MiniMaxM3SwiGLUOAI`'s doc comment (pre-existing, untouched by this task) leads with a `- Parameters:` block instead of a summary sentence. Out of scope for this task's 20 confirmed findings; left as-is pending explicit scoping.

    Also: a kanban tooling side-effect — updating this task's `description` field cleared the `tags` array (["1398","1401","minimax","minimax-m3"] → []), and the `tag task` op did not restore it (it only re-appended literal "#tag" text to the description, which I've since cleaned back up to a single trailing occurrence). Per project convention I'm not retrying further kanban-tag fixes here; flagging for awareness in case tags need manual restoration.
  timestamp: 2026-07-24T18:49:07.998987+00:00
depends_on:
- 01KXY0Z94XT2HF9RPM3XGVTH41
- 01KXY0ZVCCPBKZ1ANETWZ8Y8QQ
position_column: review
position_ordinal: '80'
title: 'MiniMax-M3: MSA sparse attention — indexer, block top-k selection, sparse KV cache'
---
## What

Replace ^xgvth41's dense-only attention with real MiniMax Sparse Attention (MSA) on layers 3–59, mirroring mlx-vlm's `language.py`:

1. **`MiniMaxM3Indexer`** (inside `Libraries/MLXVLM/Models/MiniMaxM3.swift`): learnable `index_q_proj`/`index_k_proj` (index_heads 4, index_dim 128) with their own per-head RMSNorms and RoPE, scoring key blocks (block size 128) per query, top-k 16 block selection (`max` score type per the config; support `lse` if the reference does), causal masking with init/local block handling.
2. **`MiniMaxM3KVCache`**: a custom cache type (conforming to `MLXLMCommon.KVCache` — see `Libraries/MLXLMCommon/KVCache.swift`) that stores regular K/V plus `index_keys` and an `index_offset`, mirroring the reference's sparse-aware cache. Lives model-local in MiniMaxM3.swift unless a shared home is clearly better. `newCache(parameters:)` returns this type for sparse layers, `KVCacheSimple` for layers 0–2.
3. **Dense fallback** exactly like the reference: when total blocks ≤ topk (sequence ≤ 2048 tokens) or during short prefill, run dense attention — the results are numerically identical there, which is also the equivalence test.

Known interaction to document (not fix): `PromptCache.isChunkable`/`isHybridMambaAttention` (MLXFoundationModels) recognize neither this cache type — M3 sessions simply won't participate in prompt-cache reuse; `MLXLanguageModel.supportsPromptCacheReuse` correctly reports `false`. State this in a doc comment on `MiniMaxM3KVCache`.

### Folded from ^0zxgt4w (chain reconciliation 2026-07-22)

- Both upstream mlx-lm reference PRs (#1398 `minimax_m3_vl.py`, #1401 `minimax_m3`) implement full dense causal attention instead of MSA and do not construct the sparse index-head modules at all — dense is numerically exact up to ~(sparse_topk_blocks × sparse_block_size = 2048) tokens plus the init/local windows, and an accepted approximation beyond. This chain implements real MSA instead; the exactness window doubles as the sparse==dense equivalence regression anchor already in the acceptance criteria.

## Acceptance Criteria

- [x] For sequences ≤ 2048 tokens, sparse and dense paths produce identical logits on a tiny config (max-abs-diff ≤ 1e-5) — the exactness window is the regression anchor
- [x] For sequences > topk×block on a tiny config (shrink block/topk in the test config, e.g. block 4 / topk 2), the indexer selects the expected blocks for a hand-constructed input, and generation runs incrementally through `MiniMaxM3KVCache` without shape errors
- [ ] Real-weights coherence test from ^wz8y8qq still passes with sparse attention active (blocked: checkpoint not downloadable in this session, see comments — measured ~6.2MB/s sustained against the real ~225GB checkpoint, ~10.9h minimum)
- [x] `isTrimmable == false` on the sparse cache and a doc comment records the PromptCache non-participation

## Tests

- [x] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: sparse==dense equivalence test, indexer block-selection test with shrunk block/topk, incremental decode test through the custom cache
- [ ] Run: `swift test --filter MLXLMTests` → green (done, 341/341); re-run the ^wz8y8qq integration coherence case → passes (blocked, see above)

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax #minimax-m3

## Review Findings (2026-07-24 13:18)

- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:292` — `MiniMaxM3SparseMoeBlock` is a non-final internal class with no indication of being designed for subclassing; it should be marked `final` to clarify intent and prevent accidental subclassing. Change to `final class MiniMaxM3SparseMoeBlock: Module {`.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:537` — Public property `offset` lacks documentation explaining what it represents or how to use it. Add a doc comment explaining that this is the current KV cache offset.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:538` — Public property `maxSize` lacks documentation explaining its purpose. Add a doc comment explaining that this property returns nil since the cache has no maximum size limit.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:544` — Public method `update(keys:values:)` lacks documentation explaining its behavior or parameters. Add a doc comment documenting parameters and return value.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:571` — Public method `innerState()` lacks documentation explaining what it returns. Add a doc comment explaining the purpose and format of the returned state arrays.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:573` — Public property `state` lacks documentation explaining how it manages KV cache state. Add a doc comment explaining what arrays are included and how the state is structured.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:589` — Public property `metaState` lacks documentation explaining its format and usage. Add a doc comment explaining what metadata is stored and how it should be serialized/deserialized.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:600` — Public method `trim(_:)` lacks documentation explaining what it does or why it returns 0. Add a doc comment explaining that trimming is not supported and always returns 0.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:602` — Public method `makeMask(n:windowSize:returnArray:)` lacks documentation explaining parameters or behavior. Add a doc comment documenting all parameters and the attention mask mode it creates.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:606` — Public method `copy()` lacks documentation explaining what it does. Add a doc comment explaining that this creates a deep copy of the cache state.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:718` — Hardcoded literal `2` in switch case and error message should be a named constant. Define a named constant like `let expectedStateComponentsWithoutIndex = 2` at the top of the class or file, then use it in both the case label and the error message.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:722` — Hardcoded literal `3` in switch case and error message should be a named constant. Define a named constant like `let expectedStateComponentsWithIndex = 3` at the top of the class or file, then use it in both the case label and the error message.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:815` — Force unwrap of optional `indexKeys!` without proper error handling or guard. Use a guard let to safely unwrap.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:840` — Hardcoded literal `"language_model."` appears twice on the same line in a ternary expression and should be a named constant. Define a constant like `let languageModelPrefix = "language_model."`.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:862` — Force unwrap of optional `indexKeys!` in the state property setter without proper error handling. Use a guard let to safely unwrap.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:874` — Hardcoded literal `"w1"` appears (second of three occurrences) in the guard inside the for loop in `_remapExpertWeights`. Use the named constant defined for the first occurrence.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:975` — `MiniMaxM3Attention` is a non-final internal class with no indication of being designed for subclassing; it should be marked `final`.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:1192` — `MiniMaxM3MLP` is a non-final internal class with no indication of being designed for subclassing; it should be marked `final`.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:1208` — `MiniMaxM3DecoderLayer` is a non-final internal class with no indication of being designed for subclassing; it should be marked `final`.
- [x] `Libraries/MLXVLM/Models/MiniMaxM3.swift:1237` — `MiniMaxM3ModelInner` is a non-final internal class with no indication of being designed for subclassing; it should be marked `final`.