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
- actor: claude-code
  id: 01kzxyvcssnwqc61cs9xy4jqgd
  text: |
    ### review — clean
    - evidence: `review file` on each of the four Swift source files of commit `d01df6c`. `DeepSeekV4Compressor.swift` 0 findings (run two times), `DeepSeekV4Indexer.swift` 0 findings, `DeepSeekV4Attention.swift` 4 findings (lines 248-251, blame `54e4770f`), `DeepSeekV4.swift` 2 findings (lines 449 and 551, blame `2624f899`). All 6 sit on lines this commit does not touch, thus the standing rule of `^ag7ant0` puts them out of scope. `Tests/MLXLMTests/DeepSeekV4CompressorTests.swift` gives "Nothing in scope to review" with `attempted: 0`; the engine reads no test file.
    - next: none for this card. Task `^ab1eq0r` carries the pooled cache and the sparse read.
  timestamp: 2026-08-13T16:21:26.969385+00:00
- actor: claude-code
  id: 01kzxywyz8t35sfm0cj3269arg
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 8 files. New `DeepSeekV4Compressor.swift` (256 lines) and `DeepSeekV4CompressorTests.swift` (12 tests). The `.compressor.` drop is gone from `sanitize`. Six mutation runs, each caught by a named test.
    - test: green — `swift test`, 869 tests in 105 suites, 0 failures, 2 skips that were there before, 0 warnings. Real weights: `loadsTheRealCheckpointEndToEnd()` and `greedyFirstTokensMatchThePythonFixture()` pass in 41.3 s, thus all 496 compressor tensors found a module under `[.all]` verification.
    - commit: d01df6c feat(mlx-lm): add DeepSeek-V4 Compressor for pooled-KV long-context attention
    - review: clean — `review file` ran on each of the four changed Swift source files, because the range mode under-reports. The Compressor and the Indexer answer 0 findings; the Attention answers 4 and the model answers 2, and `git blame` puts all six on earlier commits (54e4770f, 2624f899) that this commit did not touch, thus they are out of scope under the rule of `^ag7ant0`. The card moved to `done`.
  timestamp: 2026-08-13T16:22:18.344696+00:00
depends_on:
- 01KZGMZ34SVP6FRPF89R92PJCR
position_column: done
position_ordinal: f380
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

## Review Findings (2026-08-13 11:18)

Scope: commit `d01df6c`, range `HEAD~1..HEAD`. The engine ran `review file` on
each of the four Swift source files that the commit changed. The engine gives
no finding that this commit must correct.

| File | Findings |
| --- | --- |
| `Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift` | 0 |
| `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` | 0 |
| `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift` | 4, all out of scope |
| `Libraries/MLXLLM/Models/DeepSeekV4.swift` | 2, all out of scope |

The engine ran `DeepSeekV4Compressor.swift` two times and gave 0 findings both
times. On the other two files the engine made candidates and refuted some of
them (5 refuted and 3 refuted). The engine thus reads the files of this
directory correctly, and the two clean results are true results.

`Tests/MLXLMTests/DeepSeekV4CompressorTests.swift` gives "Nothing in scope to
review" with `attempted: 0`. The engine reads no test file. This result says
nothing about the quality of the tests.

### Out of scope — this commit does not touch these lines

The standing rule of task `^ag7ant0` says to record a finding on a line that
this commit does not touch, and not to correct it here. `git blame` gives
commit `54e4770f` for the four attention lines and commit `2624f899` for the
two model lines. Commit `d01df6c` changes neither group: its hunks in
`DeepSeekV4Attention.swift` cover the new lines 34-46, 175-194, 258-280 and
341-350, and its hunks in `DeepSeekV4.swift` cover the new lines 26-38, 59-68,
431-436 and 483-488.

The six findings, word for word:

- `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:248` — Public property `wqB` (query low-rank projection matrix) lacks documentation; its role in the attention mechanism is not self-evident. Add a doc comment explaining the purpose of this matrix, e.g., `/// The second projection matrix in the low-rank query pathway.`.
- `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:249` — Public property `wkv` (key/value projection matrix) lacks documentation; its specific role is not apparent without architectural knowledge. Add a doc comment explaining this is the projection for the latent key/value head, e.g., `/// The projection matrix for the latent key/value head.`.
- `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:250` — Public property `woA` (output low-rank projection matrix) lacks documentation; its role in the grouped output projection is unclear. Add a doc comment explaining this is the first part of the grouped low-rank output projection, e.g., `/// The first projection matrix in the grouped low-rank output pathway.`.
- `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:251` — Public property `woB` (output projection matrix) lacks documentation; its role in the output projection pathway is unclear. Add a doc comment explaining this is the second part of the output projection, e.g., `/// The second projection matrix that returns the output to residual width.`.
- `Libraries/MLXLLM/Models/DeepSeekV4.swift:449` — The doc comment contains two sentences before elaboration, but the rule requires a single-sentence summary on the first line. Merge into a single sentence: `/// The tensor names in a quantized projection; high-precision projections carry only the weight.`.
- `Libraries/MLXLLM/Models/DeepSeekV4.swift:551` — The function has deeply nested conditions and loops (4 levels deep), making control flow difficult to follow. The guard statement at line 562 requires tracking context through four nested levels: function → layer loop → projection loop → tensor loop. This depth increases the mental overhead for understanding and modifying the logic. Extract the innermost logic into a separate helper function (e.g., `processExpertTensor` or `stackProjectionTensors`). This reduces nesting depth and allows readers to reason about each level independently. For example, move the guard check and the inner for-loop that processes perExpert into a helper to reduce the nesting from 4 to 2 levels.

These six items belong to the files that hold them, not to this task. A later
task that changes those lines must correct them.

### The four rules of a pooled-KV port

The review looked at the four places a pooled-KV port goes wrong. Each one is
correct.

1. **The chunk position is the first raw position of the chunk.** Point 3 of
   the header of `DeepSeekV4Compressor.swift` records the defect of the
   reference: the comment of the reference says "chunk centers", and the line
   under it computes the first raw position. The header states that this port
   follows the line. `callAsFunction` computes
   `MLXArray(0 ..< chunkCount) * chunkWidth + offset`, which is that first raw
   position. The doc of the `offset` parameter and the doc of the `rope`
   parameter both state the same rule. The port is self-consistent, and the
   header records the choice.
2. **The overlap rule is right.** `poolsWithOverlap` is true only when the
   ratio equals `DeepSeekV4Configuration.indexerCompressRatio`, which is 4. An
   overlapping layer sets `projectionWidth` to `2 * headDim`, thus `wkv` and
   `wgate` answer twice the pooled width and `ape` takes the shape
   `(4, 2 * headDim)`. A ratio-128 layer keeps `projectionWidth` at `headDim`
   and takes `ape` of shape `(128, headDim)`. These are the shapes the header
   records for the published checkpoint.
3. **The two layer sets stay apart.** `hasCompressor(layer:)` is
   `compressRatio > 0`, which gives the 41 layers 2 to 42.
   `hasIndexer(layer:)` is `compressRatio == 4`, which gives the 21 even
   layers. `DeepSeekV4Attention` reads `hasIndexer` for the indexer and
   `hasCompressor` for the compressor, thus it does not confuse the two sets.
   The `init` of the compressor holds a `precondition(hasCompressor(layer:))`,
   which stops a build on a plain layer.
4. **The `compress_ratio == 0` path is bit-identical by construction.**
   `ropeTheta(forLayer:)` calls `hasCompressor(layer:)` itself, thus the set
   of compressed layers and the set of layers that take `compressRopeTheta`
   cannot disagree. Layers 0 and 1 have a ratio of 0, thus they build no
   compressor and take the plain rope theta.

### The two judgement calls

1. **The two filters that stay in `sanitize` are correct.** The acceptance
   criterion asks that no drop filter for a **sparse-attention** key remains.
   The commit removes `compressorSegment` and its test. The two tests that
   stay are `mtp.`, which is a decision of an earlier task, and
   `layer < layerCount`, which is a bounds guard. Neither one names a
   sparse-attention key, thus the criterion is met as it is written. Keep
   them.
2. **The deferral of the pooled read is correct.** The attention path reads no
   pooled chunk, thus the greedy parity fixture matches token for token. This
   is the expected result, not a gap. The compressor modules are built and
   their tensors load, which is what lets `sanitize` drop the filter — the
   deliverable of this task. Task `^ab1eq0r` exists on the board, names the
   pooled cache and the sparse read, and lists the tests for both. The header
   of `DeepSeekV4Compressor.swift` records the same deferral in point 1 and in
   its closing paragraph.
