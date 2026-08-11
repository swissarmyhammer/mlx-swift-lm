---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzrne25s9zq1d7776hzv0aq9
  text: |
    ## Research, and the one decision the card did not name

    ### The module names are the checkpoint's names, not the reference's

    The file the card names -- `scouzi1966/mlx-swift-lm`
    `Libraries/MLXLLM/Models/DeepseekV4.swift`, and the attribution source
    `osaurus-ai/vmlx-swift-lm` at `b166896` -- keys a decoder layer `self_attn`,
    `mlp`, `input_layernorm` and `post_attention_layernorm`. Those are the
    DeepSeek-V3 names, and that file's `sanitize` maps the DeepSeek-V4 checkpoint
    onto them.

    This repository cannot take those names. `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`
    holds the published `quantization` block, and that block names
    `model.layers.N.attn.wq_a`, `model.layers.N.ffn.switch_mlp.gate_proj` and so
    on. `quantize(model:filter:)` hands its filter the FLATTENED MODULE PATH
    (`Libraries/MLXLMCommon/Load.swift`), thus a tree under the DeepSeek-V3 names
    resolves NO per-layer entry of that plan: every routed expert would take the
    affine default at group size 64 rather than mxfp4 at group size 32.
    `DeepseekV4QuantizationPlanTests` already pins that contract from the plan's
    side, and its `ProbeModel` is a stand-in tree with exactly these names.

    The names this file uses are therefore the ones the Python reference gives --
    `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
    `mlx_lm/models/deepseek_v4.py`, `DeepseekV4Block` and `DeepseekV4Model`:
    `attn`, `ffn`, `attn_norm`, `ffn_norm`, `hc_attn`, `hc_ffn`, `hc_head`. Those
    are the names the quantization block states.

    `theModulePathsAreTheCheckpointKeyPaths` pins every one of them.

    ### Two more corrections to the card text

    1. **There is no dense prefix.** The card's "What" item 1 says "MoE (or dense
       MLP for the dense prefix layers)". `DeepseekV4Block.__init__` builds a
       `DeepseekV4MoE` for EVERY layer, and DeepSeek-V4 `config.json` carries no
       `first_k_dense_replace` key. I corrected the card.
    2. **The sparse-attention tasks have real ids.** The card wrote
       `<sparse-attn>`. They are `^tty95f4` (Compressor) and `^r92pjcr` (Indexer).
       The code comment names both.

    ### What the file leaves out, and why

    - **`newCache` is the one `KVCacheDimensionProvider` gives.** The reference
      allocates a `RotatingKVCache` for a layer of compress ratio 0 and a
      compressing cache for every other layer. A rotating window is correct only
      beside the compressor that carries the global context, and this file drops
      the compressor, thus a window here would silently lose context past
      `sliding_window`. A plain cache keeps every key until sparse attention lands.
    - **No `compile(shapeless:)` decode path and no stage profiler.** Both are
      performance work of their own and neither changes a number.
    - **The tracer reads `MLX_DSV4_NUMERIC_TRACE`.** Every environment variable in
      this repository starts with `MLX_` (`MLX_RUN_VLM_INTEGRATION`,
      `MLX_MINIMAX_M3_CHECKPOINT`, `MLX_RUN_COLD_FETCH`).

    ### `DeepseekV4Configuration` gained one key

    `tie_word_embeddings`. The published `config.json` carries it (`false` on
    DeepSeek-V4-Flash) and the configuration did not decode it, so the tied path
    had nothing to read. The default is `false`, and
    `aCheckpointThatNamesNoTieBuildsItsOwnLanguageModelHead` pins that default.
  timestamp: 2026-08-11T15:00:40.761037+00:00
- actor: claude-code
  id: 01kzrnf6h70td660hjzw2fcpzk
  text: |
    ## The port landed, and every mutation died

    ### What the files hold

    `Libraries/MLXLLM/Models/DeepseekV4.swift`, 528 lines:

    - `DeepseekV4NumericTrace` -- the env-gated tracer, on
      `MLX_DSV4_NUMERIC_TRACE=1`. Proved live: a prefill run writes 29 lines to
      standard error with the variable set and 0 lines without it. Each line names
      the stage, the shape, the mean, the root mean square, the largest absolute
      value and the four leading values, which is what a bisect against the Python
      needs.
    - `DeepseekV4DecoderLayer` -- the two halves. `wrapped(_:connection:norm:half:block:)`
      holds the collapse, the norm, the block and the expand ONE time, and each
      half calls it. Writing the two halves out twice would have been a
      `duplication-parsed` finding and a place for the two to drift.
    - `DeepseekV4ModelInner` -- the embedding gains the copy axis, the layers run,
      `hc_head` takes the copy axis away, the final norm reads what is left. The
      attention mask reads the embedding, which is one copy of the stream, exactly
      as the Python reads `h[:, :, 0, :]`.
    - `DeepseekV4Model` -- `kvHeads`, an optional `lm_head`, `loraLayers`, and the
      load filter.

    `Tests/MLXLMTests/DeepseekV4ModelTests.swift`, 626 lines, 20 tests.

    ### Why these tests, and not parity fixtures

    Every piece this file assembles already carries parity tests against a NumPy
    transcription of the Python reference. What no parity test can see is the
    WIRING: the order of the two halves, the stream each half reads, the reduction
    at the top, the module path of each tensor, and the key map.

    `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` states the order of
    `DeepseekV4Block.__call__` out of the pieces themselves -- collapse, norm,
    block, expand, twice, attention first -- and holds the layer to it value by
    value. A production file that swapped the two halves, dropped the expand, or
    lost the residual stops agreeing with it. The mutation table below shows all
    three.

    `theDecoderLayerHandsTheTokenIdentifiersToTheHashRoutingGate` runs ONE residual
    stream through a hash layer with two different runs of token identifiers and
    demands the answers differ. A hash layer names its experts from the identifiers
    alone, thus this dies the moment the identifiers stop reaching the gate.

    `loadedModel(tieWordEmbeddings:)` loads through
    `update(parameters:verify: [.all])`, which is the verification
    `MLXLMCommon.loadWeights` applies, thus the module tree is held to accepting a
    complete parameter set rather than to a looser check.

    ### The mutations

    Thirteen mutations of the production file, each run against
    `swift test --filter DeepseekV4ModelTests`. Every one died, and no test
    survived its own mutation:

    | Mutation | The test that died |
    |---|---|
    | M1 the mixture half runs before the attention half | `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` |
    | M2 the manifold expand is dropped; the block output is broadcast back | `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` |
    | M3 the manifold expand drops its residual term | `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` |
    | M4 the final head reduce is skipped; the norm reads the copies | `theStackReducesTheParallelCopiesBeforeTheFinalNorm` |
    | M5 the final head reduce is skipped; the first copy stands for the stream | `theStackReducesTheParallelCopiesBeforeTheFinalNorm` |
    | M6 the load filter maps `hc_head_base` onto `model.hc_head.bias` | `sanitizeGivesEveryCheckpointKeyItsModulePath` |
    | M7 the load filter keeps the multi-token-prediction head | `sanitizeDropsTheMultiTokenPredictionHead` |
    | M8 the load filter keeps the compressor and the indexer | `sanitizeDropsTheCompressorAndTheIndexerUntilSparseAttentionLands` |
    | M9 the load filter keeps a layer beyond the configured depth | `sanitizeDropsALayerTheConfigurationDoesNotDeclare` |
    | M10 the expert stack runs in reverse expert order | `sanitizeStacksThePerExpertWeightsIntoTheSwitchLayer` |
    | M11 the language-model head is built even for a tied checkpoint | `aTiedCheckpointDeclaresNoLanguageModelHead`, `aTiedCheckpointLoadsWithoutItsOwnLanguageModelHead`, `aTiedCheckpointProjectsThroughItsEmbeddingTable` |
    | M12 the mixture gate reads zeros instead of the token identifiers | `theDecoderLayerHandsTheTokenIdentifiersToTheHashRoutingGate`, `theDecoderLayerRunsTheAttentionHalfBeforeTheMixtureHalf` |
    | M13 the residual stream carries one copy instead of `hc_mult` | CRASHED -- see below |

    M13 is a death of a different shape. The stream then carries one copy where the
    hyper-connection expects `hc_mult`, and
    `ManifoldStream.checkShape(of:copyCount:width:)` in
    `DeepseekV4HyperConnection.swift` stops the process on its precondition. The
    suite cannot report green, which is what a death means, but no named test
    records it.

    The harness is `mutate.py` in the scratchpad of this session. It applies one
    change, runs the suite, and puts the file back; it asserts the SHA-256 of the
    restored file equals the SHA-256 of the file before the run, and that assertion
    held. `git status` shows the production file untracked-and-new with no other
    change.

    ### Tests, format and lint

    - `swift test --filter DeepseekV4ModelTests`: 20 of 20 pass.
    - `swift test --filter DeepseekV4`: 86 of 86 pass in 6 suites.
    - `swift test` (whole suite): 445 + 0 + 80 + 282 + 7 pass, 0 failures.
    - The only `warning:` line of the build is `missing creator for mutated node:
      ... mlx-swift_Cmlx.bundle/Contents/MacOS`. It is the SwiftPM build-graph
      warning about the vendored MLX bundle that task `^g5e7907` already measured
      as present with and without a source change. It is not a compiler warning.
    - `swiftlint` with the shipped `no_magic_numbers` options
      (`allowed_numbers: [0, 1, -1, 100]`): 0 violations over the three files.
    - `swiftlint` with the shipped `cyclomatic_complexity` (15) and
      `function_body_length` (250): 0 violations.
    - `swiftlint` with the shipped `missing_docs` options: 0 violations.
    - `swift-format format --in-place` on the new files changed nothing on a second
      run, and `swift-format lint --strict` exits 0. I did not run `swift-format`
      over the tree.

    ### One defect found, and filed rather than fixed here

    `DeepseekV4MoEGate` declares BOTH `tid2eid` and `bias` on EVERY layer. The
    Python declares `tid2eid` on a hash layer alone and `e_score_correction_bias`
    on every other layer alone. Under the `.allModelKeysSet` verification that
    `loadWeights` applies, the published checkpoint therefore cannot load: layers 0
    to 2 carry no `bias`, and layers 3 to 42 carry no `tid2eid`. The tests of this
    card cannot see it, because they build the checkpoint FROM the module tree.

    The fix belongs in `DeepseekV4MoE.swift`, whose doc comment records the
    opposite decision on purpose, thus it is not this card's to make. Filed as
    `^3zest44`.
  timestamp: 2026-08-11T15:01:17.991538+00:00
- actor: claude-code
  id: 01kzrnff8sw1trr1ngrsrxb6r9
  text: |
    ### implement — changed
    - evidence: 3 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV4.swift (528 lines, new), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4ModelTests.swift (626 lines, new, 20 tests), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV4Configuration.swift (+6, `tie_word_embeddings`). `swift test --filter DeepseekV4ModelTests` 20/20; `swift test --filter DeepseekV4` 86/86; `swift test` 445+0+80+282+7, 0 failures, 0 compiler warnings. 13 of 13 mutations killed, 0 survivors. swiftlint `no_magic_numbers`, `cyclomatic_complexity`, `function_body_length` and `missing_docs` each report 0 over the three files. `DeepseekV3.swift` untouched. New defect filed as `^3zest44`.
    - next: /review
  timestamp: 2026-08-11T15:01:26.937207+00:00
depends_on:
- 01KZGMSFQN4AS74HYXAAG7ANT0
- 01KZGMT5M19CHE8SK3FSBWYQ83
- 01KZGMTXP266RDESZHBG5E7907
position_column: doing
position_ordinal: '80'
title: Assemble DeepseekV4 decoder layer, model, and weight sanitize
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4.swift` — the decoder layer and top-level model that compose the attention, mHC, and MoE pieces built in the preceding tasks.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4DecoderLayer` at line 1569, `DeepseekV4ModelInner` at line 1746, `DeepseekV4Model` at line 1845. Skip everything DSpark (`DeepseekV4DSparkMarkovHead:757`, `DeepseekV4DSparkConfidenceHead:777`, `DeepseekV4DSparkProposal:792`, `DeepseekV4DSparkStage:800`, `DeepseekV4DSparkGenerator:2355`) — out of scope.

Pieces:

1. `DeepseekV4DecoderLayer` — mHC pre → attention → mHC post → mHC pre → MoE → mHC post. **Card correction (2026-08-11):** the card said "or dense MLP for the dense prefix layers". DeepSeek-V4 has NO dense prefix. `DeepseekV4Block.__init__` of the Python reference builds a `DeepseekV4MoE` for every layer, and DeepSeek-V4 `config.json` carries no `first_k_dense_replace` key. Every layer holds a mixture of experts.
2. `DeepseekV4ModelInner` — embedding, 43 layers, `hc_head` reduce, final norm.
3. `DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel` — matching the reference's own conformances. `lm_head` handles tied embeddings through the optional-`lm_head` pattern of GLM4 (commits `fed151f` / `4805454`).
4. **`sanitize`** — the load-side filtering the gap tracker calls out:
   - Drop `mtp.0.*` keys (multi-token-prediction head; we do not use it here).
   - Drop `compressor.*` and `indexer.*` keys until sparse attention lands (tasks `^tty95f4` and `^r92pjcr` — remove this filter with them).
   - Stack per-expert routed weights into `switch_mlp.*` for the switch layer.
   - Load the `tid2eid` int64 hash table for the hash-routing layers.

Also port the env-gated numeric tracer from the reference (its `VMLX_DSV4_NUMERIC_TRACE`, reference lines ~60-84) under a repo-appropriate env var name — it is how you will bisect a numeric mismatch against Python without a debugger, and it costs nothing when off. **It reads `MLX_DSV4_NUMERIC_TRACE`**, because every environment variable of this repository starts with `MLX_`.

Do not modify `DeepseekV3.swift`.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~1569-2354 plus the tracer at ~60-84 (MIT; header attributes Osaurus AI).
- Sanitize/load gap list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Module names: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py`, plus the `quantization` block of `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4.swift` exists with `DeepseekV4DecoderLayer`, `DeepseekV4ModelInner`, and `DeepseekV4Model` conforming to `LLMModel`, `KVCacheDimensionProvider`, and `LoRAModel`.
- [x] `sanitize` drops `mtp.0.*`, `compressor.*`, and `indexer.*` keys, and stacks routed experts into `switch_mlp.*`.
- [x] Tied-embedding checkpoints load without a missing-`lm_head` error; untied ones use the checkpoint's `lm_head`.
- [x] A synthetic-weight forward produces logits of shape `[B, L, vocabSize]`, `allFinite`.
- [x] The `compressor.*`/`indexer.*` filter carries a code comment naming the sparse-attention tasks that remove it.
- [x] No DSpark types present.
- [x] `DeepseekV3.swift` unmodified.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4ModelTests.swift`, tiny synthetic config (4 layers, hidden 16, 8 experts, vocab 12) and random weights — no download.
- [x] Test: prefill forward `[1, 8]` token ids → logits `[1, 8, vocab]`, `allFinite`.
- [x] Test: decode forward with cache `[1, 1]` → `[1, 1, vocab]`; after N steps cache offset == N.
- [x] Test: `sanitize` on a synthetic weight dict containing `mtp.0.foo`, `compressor.bar`, `indexer.baz`, and per-expert keys — asserts the first three are gone and the experts are stacked.
- [x] Test: tied-embedding path — config with `tie_word_embeddings=true` and no `lm_head` weight loads and runs.
- [x] Run: `swift test --filter DeepseekV4ModelTests` — 20 of 20 pass.

## Workflow
- Use `/tdd` — the `sanitize` test is pure dictionary manipulation and should be written first; it needs no model at all.
#deepseek-v4