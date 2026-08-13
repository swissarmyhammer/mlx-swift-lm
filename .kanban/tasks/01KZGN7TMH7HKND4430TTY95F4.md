---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzpb3e58265x2m297j3q3yvz
  text: |-
    ### Layer rule for the compressor -- read this before you build the module

    From task `^wkv5j6f`, measured on `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`:

    - `compress_ratios` gives 0 to layer 0 and to layer 1. These two layers hold no compressor. Layer 2 is the first layer that holds one.
    - The published `quantization` block names compressor keys for the 41 layers 2 to 42 only -- 82 keys, `wgate` and `wkv` on each.
    - The compressor submodule must be **absent** on layers 0 and 1, not present and unused. A compressor built on all 43 layers fails the weight load, because the checkpoint holds no `model.layers.0.attn.compressor.*` and no `model.layers.1.attn.compressor.*` arrays. Make the submodule optional, as `AfMoE` and `BailingMoe` make `shared_experts` optional.
    - `hasCompressor(layer:)` in `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` already holds this rule. Use it, and do not write the rule again.
    - `compress_ratios` holds **44** entries while `num_hidden_layers` is **43**, and entry 43 is 0. `hasCompressor(layer:)` guards the bound, thus it is correct today. Any code that reads `compressRatios.count` as a layer count gets 44. Read `numHiddenLayers` for a layer count.

    `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift` pins the layer rule in `fixtureNamesCompressorKeysFromLayerTwoUp`.
  timestamp: 2026-08-10T17:21:37.960809+00:00
- actor: claude-code
  id: 01kzxwjk758mx512qv9xymdxsx
  text: |-
    ### Research, before any edit

    **The reference fetched.** `scouzi1966/mlx-swift-lm` @ `main`,
    `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`, 1353 lines. The
    `Compressor` part is lines 860 to 1226. The first 850 lines are
    `DeepseekV4Cache` -- a stateful pooled cache with partial-window buffers --
    which no card names.

    **The published checkpoint, measured.** `model.safetensors.index.json` of
    `mlx-community/DeepSeek-V4-Flash-4bit`:

    - `attn.compressor.*` on the 41 layers 2 to 42. `attn.indexer.compressor.*` on
      the 21 even layers 2 to 42. This confirms the layer rule of the card.
    - Layer 2 (ratio 4): `wkv.weight` and `wgate.weight` answer **1024**, `ape` is
      `[4, 1024]`, `norm.weight` is `[512]`.
    - Layer 3 (ratio 128): the two projections answer **512**, `ape` is
      `[128, 512]`, `norm.weight` is `[512]`.
    - Layer 2 indexer compressor: the two projections answer **256**, `ape` is
      `[4, 256]`, `norm.weight` is `[128]`.

    Thus the reference rule `outDim = headDim * (ratio == 4 ? 2 : 1)` is the rule
    the checkpoint states. A ratio-4 layer pools with overlap, thus each pooled
    chunk reads the chunk before it as well, and the projections answer twice the
    pooled width. A ratio-128 layer does not.

    The `quantization` block names 124 compressor keys: 82 for `attn.compressor`
    and 42 for `attn.indexer.compressor`. Both sets are affine, 4 bits, group 64.

    **No `mtp.*` key in the published checkpoint**, and the layer indices stop at
    42. Thus the two remaining tests of `isLoaded` fire on nothing there.

    **The prompt of the parity fixture is 18 tokens.** Below `sliding_window`
    (128), and the acceptance criterion "short-prompt output unchanged" plus the
    order to run `greedyFirstTokensMatchThePythonFixture()` both demand that this
    task change no attention number.

    **Scope, stated plainly.** The forward path of `DeepSeekV4Attention` cannot
    read pooled rows without the `DeepseekV4Cache` above: a decode step carries one
    token, thus a stateless compressor answers no pooled row and the global context
    would be lost. Sparse attention is therefore its own work, and this card lands
    the module, the weights and the wiring, the same way task `^r92pjcr` landed the
    indexer ahead of its caller.
  timestamp: 2026-08-13T15:41:41.477615+00:00
- actor: claude-code
  id: 01kzxxg8tfmtbke2080e7swvna
  text: |-
    ### What landed, and the deviations from the reference

    `Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift` holds
    ``DeepSeekV4Compressor``: `wkv`, `wgate`, `ape` and `norm`. It pools each whole
    run of `compress_ratio` tokens into one chunk with a softmax gate, norms the
    chunk, and turns it by the rotary position that chunk starts at.

    ``DeepSeekV4Attention`` holds an **optional** compressor, built only where
    `hasCompressor(layer:)` says so. ``DeepSeekV4Indexer`` holds a compressor of its
    own, which pools to `index_head_dim`. `DeepSeekV4RoPE` gained
    `cosSin(positions:)`, because the pooled chunks stand a whole compress ratio
    apart rather than in a run; `cosSin(offset:length:)` now calls it.

    Five deviations from the reference, each recorded in the file header:

    1. No pooled cache. The reference keeps its chunks in a `DeepseekV4Cache`; this
       is its `v4Cache == nil` path.
    2. Any batch size. The reference demands a batch of one, because its pool
       windows belong to one request.
    3. **A defect of the reference, of the same shape as the one the indexer port
       found.** The comment of the reference says the pooled chunk takes the rotary
       position of the "chunk centers", and the line under that comment reads
       `arange(pooled_count) * ratio + pool_base`, which is the FIRST raw position
       of each chunk. This port follows the line, not the comment. Whoever reads
       that comment and "corrects" the line moves every chunk half a chunk on.
    4. No activation quantization-aware round trip; the published `config.json`
       names no such key.
    5. The rotary tables come from the layer rope rather than from a `rope.invFreq`
       property, which would put `attn.rope.invFreq` into the weight-load check.

    ### Mutation proofs

    Each mutation was applied on its own, `swift test --filter
    DeepSeekV4CompressorTests` was run, and the mutation was taken back:

    | Mutation | Tests that caught it |
    |---|---|
    | the rope theta reads the checkpoint instead of the layer | `aCompressRatioZeroLayerIsBitIdenticalToPlainAttention` |
    | `hasCompressor` answers true for every layer | that test, plus `attentionHoldsACompressorOnlyOnACompressorLayer` and `theCompressedLayersAreTheCompressRopeThetaLayers` |
    | the overlap hands the NEXT chunk back instead of the one before | `aPooledChunkReadsNoTokenPastItsOwnEnd` |
    | the chunk position drops the block offset | `theOffsetTurnsOnlyTheRotaryHalfOfEachPooledChunk` |
    | an overlapping layer does not widen its projections | `theProjectionsTakeTheirWidthFromTheCompressRatio` and `theCompressorInsideAnIndexerPoolsToTheIndexHeadWidth` |
    | an incomplete run at the end becomes a chunk of its own | `aBlockShorterThanOneChunkPoolsNothing`, which stopped on the reshape |

    The sanitize test was red before the drop went away: 5 issues, one for each
    compressor key.

    ### The real weights

    `xcodebuild test-without-building` on the two named tests, against
    `mlx-community/DeepSeek-V4-Flash-4bit`:

    - `loadsTheRealCheckpointEndToEnd()` passed after 21.860 s. `loadWeights`
      verifies with `[.all]`, thus all 496 compressor tensors -- 41 attention
      compressors and 21 indexer compressors -- found a module, and no module went
      unfilled.
    - `greedyFirstTokensMatchThePythonFixture()` passed after 19.393 s, thus the
      short-prompt numbers did not move.

    The >12k-token endurance test was NOT run, as ordered.
  timestamp: 2026-08-13T15:57:53.871267+00:00
- actor: claude-code
  id: 01kzxxghn7q55vh7rtj5t6329p
  text: |-
    ### implement — changed

    - evidence: 8 files. New `Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift`
      and `Tests/MLXLMTests/DeepSeekV4CompressorTests.swift`; changed
      `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`,
      `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`,
      `Libraries/MLXLLM/Models/DeepSeekV4.swift`,
      `Tests/MLXLMTests/DeepSeekV4ModelTests.swift`,
      `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift`,
      `docs/deepseek-v4-support.md`.
      `swift test --filter DeepSeekV4CompressorTests` 12/12 pass;
      `swift test --filter DeepSeekV4` 130 tests in 15 suites pass;
      `swift test` 869 tests in 105 suites pass, 2 skipped, 0 failures, 0 warnings;
      `swift format lint --strict` clean on every touched file;
      `xcodebuild build-for-testing` on IntegrationTesting succeeded, and
      `loadsTheRealCheckpointEndToEnd()` plus
      `greedyFirstTokensMatchThePythonFixture()` pass in 41.253 s against the real
      weights. Six mutation runs, each caught -- see the comment above.
    - next: `/review`. New work opened as `^ab1eq0r` for the pooled cache and the
      sparse attention path.
  timestamp: 2026-08-13T15:58:02.919615+00:00
depends_on:
- 01KZGMZ34SVP6FRPF89R92PJCR
position_column: doing
position_ordinal: '80'
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
- Family reference: `ml-explore/mlx-lm` `mlx_lm/models/deepseek_v4.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift` exists with the `Compressor` module, wired to the Indexer from task `r92pjcr`. (The repository spells the file `DeepSeekV4...`, with a capital S.)
- [x] The `compressor.*` drop is removed from `DeepseekV4Model.sanitize`; no drop filter for a sparse-attention key remains. See "The two filters that stay" below.
- [x] Layers with `compress_ratio == 0` take the plain-attention path and are bit-identical to the pre-task behavior.
- [x] The set of layers the compressor treats as compressed matches the set for which `ropeTheta(forLayer:)` returns `compress_rope_theta`.
- [x] A prompt longer than `sliding_window=128` produces coherent output.
- [x] Short-prompt output unchanged.

## Tests

- [x] New `Tests/MLXLMTests/DeepSeekV4CompressorTests.swift`.
- [x] Test: pooled-KV output shape for `compress_ratio` 4 and 128; `allFinite`.
- [x] Test: `compress_ratio == 0` path is bit-identical to plain attention on the same inputs.
- [x] Test: compressed-vs-RoPE layer-set agreement — assert the two predicates return the same set over all 43 layers of the real config.
- [x] Test: `sanitize` retains `compressor.*` and `indexer.*` keys (inverse of the assertion added in `pwr8r3h`).
- [x] Test: a sequence of length 512 (past the 128 window) runs and produces `allFinite` logits.
- [x] Regression: `swift test --filter 'DeepSeekV4ModelTests|DeepSeekV4IndexerTests'` still green.
- [x] Run: `swift test --filter DeepSeekV4CompressorTests` — all pass.

## Workflow
- [x] Use `/tdd` — the `compress_ratio == 0` bit-identity test is the regression guard; write it before touching the attention path.

## The two filters that stay

`DeepSeekV4Model.sanitize` keeps two tests after this task, and neither one
names a sparse-attention key:

1. `mtp.` — the multi-token-prediction head. An earlier task decided to drop
   it, and the header of `Libraries/MLXLLM/Models/DeepSeekV4.swift` records
   that decision: this repository carries no DSpark type to load it into.
   Measured: the published `mlx-community/DeepSeek-V4-Flash-4bit` checkpoint
   holds **zero** `mtp.*` keys, thus the test fires on nothing there. A
   DeepSeek-V4 checkpoint that ships an MTP head would fail the load without
   it, thus removing it would be a regression with no gain.
2. `layer < layerCount` — a key that names a layer the configuration does not
   declare. Measured: the published checkpoint stops at layer 42 and declares
   43 layers, thus this test also fires on nothing there.

## What this task does not do

The attention path reads no pooled chunk yet. A decode step carries one token,
thus a stateless compressor pools nothing and the global context would go away
at the first decode step. The pooled cache and the sparse path that reads it
are task `^ab1eq0r`.
#deepseek-v4