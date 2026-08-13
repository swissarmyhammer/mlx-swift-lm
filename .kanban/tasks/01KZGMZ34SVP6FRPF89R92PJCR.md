---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzpb3n0fqspjaq9ctst5dq5s
  text: |-
    ### Layer rule for the indexer -- this card does not state it yet

    From task `^wkv5j6f`, measured on `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`:

    - The published `quantization` block names indexer keys for the **21 even layers 2 to 42** only -- 84 keys: `wq_b`, `weights_proj`, `compressor.wgate` and `compressor.wkv` on each. These are the layers whose compress ratio is 4. The odd layers 3 to 41 hold a ratio of 128 and hold no indexer, and layers 0 and 1 hold a ratio of 0 and hold neither an indexer nor a compressor.
    - This card names `index_n_heads`, `index_head_dim` and `index_topk` only, and states nothing about which layers hold indexer weights. An indexer built on all 43 layers, or on all 41 compressor layers, fails the weight load.
    - `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` holds `hasCompressor(layer:)` and **no** `hasIndexer(layer:)` beside it. Add one -- `compressRatios[layer] == 4`, with the same bound guard -- and build the indexer submodule only where it gives true.
    - `compress_ratios` holds **44** entries while `num_hidden_layers` is **43**, and entry 43 is 0. The bound guard makes `hasCompressor(layer:)` correct today. Any code that reads `compressRatios.count` as a layer count gets 44. Read `numHiddenLayers` for a layer count.

    `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift` pins the layer rule in `fixtureNamesIndexerKeysOnTheEvenLayersFromTwoUp`.
  timestamp: 2026-08-10T17:21:44.975894+00:00
- actor: claude-code
  id: 01kzxqy3bkdmb1dh39vnkzf5q0
  text: |
    ### Research before the code

    Read the two references. Both downloads were good.

    - `scouzi1966/mlx-swift-lm` @ main, `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`, lines 1228 to 1353: the `Indexer` class. It holds `wq_b`, `weights_proj` and a `compressor` submodule. The score path is: project the low-rank Q, apply the partial rotary position, matmul against the pooled keys, clamp at 0, scale, weight each head by `weights_proj(x) * nHeads^-0.5`, sum over the heads, mask, then `argPartition`.
    - The block-causal rule is in their `DeepseekV4MathHelpers.swift`, `compressedVisibility`: chunk `k` covers raw positions `[k*ratio, (k+1)*ratio)` and a query at absolute position `q` sees it when `(k+1) * ratio <= q + 1`.
    - `ml-explore/mlx-lm` `mlx_lm/models/deepseek_v32.py` `Indexer` gives the same score path with a different key source. It also gives the short-circuit `if k.shape[2] <= self.index_topk: return None`.
    - `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, `mlx_lm/models/deepseek_v4.py` line 634: its `Indexer` holds the parameters and no forward pass ("the actual topk gather path is not yet used in the forward pass"). It gives the parameter names only.

    ### Measurements on the published checkpoint

    Read `model.safetensors.index.json` and `config.json` of `mlx-community/DeepSeek-V4-Flash-4bit`:

    - 294 keys hold `.indexer.`, which is 14 keys on each of 21 layers. The layers are 2, 4, 6 ... 42, which agrees with the card comment.
    - The 14 keys of one layer: `indexer.wq_b.{weight,scales,biases}`, `indexer.weights_proj.{weight,scales,biases}`, `indexer.compressor.wkv.{weight,scales,biases}`, `indexer.compressor.wgate.{weight,scales,biases}`, `indexer.compressor.ape`, `indexer.compressor.norm.weight`.
    - Shapes on layer 2: `indexer.wq_b.weight` is `[8192, 128]` u32, thus `Linear(q_lora_rank=1024, index_n_heads * index_head_dim = 8192)`. `indexer.weights_proj.weight` is `[64, 512]` u32, thus `Linear(hidden_size=4096, index_n_heads=64)`.
    - `index_n_heads` 64, `index_head_dim` 128, `index_topk` 512. The `config.json` gives no activation-QAT key, thus the QAT round trip of the scouzi file has no input here and this port leaves it out.

    ### The one point the card leaves open, and how this task reads it

    `attn.indexer.compressor.*` is an `indexer.*` key AND a `compressor.*` key. The load filter tests substrings, thus the `.compressor.` test that the card keeps also keeps the indexer's own pooled-key compressor out. This task therefore lands the two Linear layers `indexer.wq_b` and `indexer.weights_proj`, and the compressor task `^tty95f4` lands `indexer.compressor.*` beside `attn.compressor.*`. That reading agrees with `^tty95f4`, which says "the `Compressor` module, wired to the Indexer from task `r92pjcr`" and "no drop filters for DSV4 remain".

    `MLXLMCommon.loadWeights` verifies with `[.all]`, which is `.noUnusedKeys` AND `.allModelKeysSet`. A weight with no module, or a module with no weight, both stop the load. That is why the two halves must land together.

    ### File name

    The card writes `DeepseekV4Indexer.swift`. Every DeepSeek-V4 file of this repository writes `DeepSeekV4`, with a capital S, and the card names the existing files the same wrong way (`DeepseekV4Configuration.swift`, `DeepseekV4Model.sanitize`). This task follows the repository, thus `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` and `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift`. macOS holds one file for the two spellings.
  timestamp: 2026-08-13T14:20:35.571696+00:00
- actor: claude-code
  id: 01kzxrwg5snbyy105xw6jcdfq1
  text: |
    ### The implementation landed

    Files:

    - `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` -- new. `DeepSeekV4Indexer` holds `wq_b` and `weights_proj`, and its `callAsFunction` answers a Boolean selection mask of shape `(batch, 1, tokens, chunks)`.
    - `Libraries/MLXLLM/Models/DeepSeekV4Configuration.swift` -- new `hasIndexer(layer:)` and a new `compressRatio(ofLayer:)` that `hasCompressor(layer:)` now reads too, thus the bound guard has one place. New `indexerCompressRatio` constant, which is 4.
    - `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift` -- new `@ModuleInfo(key: "indexer") var indexer: DeepSeekV4Indexer?`, built only where `hasIndexer(layer:)` gives true.
    - `Libraries/MLXLLM/Models/DeepSeekV4.swift` -- `sparseAttentionSegments` became `compressorSegment`. The `.indexer.` test is gone and the `.compressor.` test stays, with a comment that names task `^tty95f4`.
    - `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift` -- new, 8 tests.
    - `Tests/MLXLMTests/DeepSeekV4ModelTests.swift` -- `sanitizeDropsTheCompressorAndTheIndexerUntilSparseAttentionLands` became `sanitizeDropsTheCompressorUntilTheCompressorTaskLands`, because the old assertion `!sanitized.keys.contains { $0.contains("indexer") }` is not true now.

    ### Four points where this port and the reference do not agree

    The header of the new file states each one.

    1. **A block of one token takes the causal mask too.** `DeepseekV4Math.causalMaskedIndexerScores` of the reference opens with `guard S > 1, compressedLen > 0 else { return scores }`. A decode step carries one token, thus EVERY decode step of the reference ranks unmasked scores and may pick a chunk of its own future. This port masks each block, of any length.
    2. **The answer is a Boolean mask, not `argPartition` output.** `argPartition` fills its whole budget, thus a query with fewer visible chunks than the budget gets chunk indices it may not read. The mask AND-s the visibility back in, which is the "fewer, not garbage" line of the card.
    3. **The pooled keys arrive as an argument.** The reference holds a private compressor. Task `^tty95f4` lands that compressor here.
    4. **No activation-QAT round trip.** The published `config.json` names no such key.

    ### The proof that the tests are not vacuous

    Two mutation runs, each reverted after the measurement:

    - `(k + 1) * chunkWidth` became `k * chunkWidth`, which is the classic off-by-one: `everyPickedChunkStandsWhollyBehindItsQuery` failed with 348 issues.
    - The final `.&& visible` went away: `everyPickedChunkStandsWhollyBehindItsQuery` failed with 348 issues, `theCountIsTheTopKBudgetOrTheVisibleCount` failed with 180 issues, and `theFirstPositionsPickNothingUntilAWholeChunkStandsBehindThem` failed with 5.

    `git diff` after each revert shows no difference.

    ### The lint findings, and the correction

    `swiftlint no_magic_numbers` gave 5 findings in the new file, each of them an axis number: the array literal `[0, 2, 1, 3]` (2 findings), `.any(axis: -2)`, and two `swappedAxes(-1, -2)` calls. The correction removes the cause from the whole file: the transpose became `swappedAxes(Self.headAxis, Self.batchMajorHeadAxis)`, and three named constants -- `batchMajorHeadAxis`, `matrixRowAxis` and `rankedChunkAxis` -- name each axis. The file lints clean now, and `swift format` gives it back unchanged.
  timestamp: 2026-08-13T14:37:11.737219+00:00
- actor: claude-code
  id: 01kzxrxen611tm2p18vevf8hpr
  text: |
    ### implement — changed

    - evidence: 6 files. New: `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift`. Changed: `Libraries/MLXLLM/Models/DeepSeekV4.swift`, `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`, `Libraries/MLXLLM/Models/DeepSeekV4Configuration.swift`, `Tests/MLXLMTests/DeepSeekV4ModelTests.swift`.
    - `swift test --filter DeepSeekV4IndexerTests`: 8 tests in 1 suite passed.
    - `swift test --filter DeepSeekV4`: 119 tests in 14 suites passed.
    - `swift test`: 489 + 80 + 282 + 7 tests passed, 0 failures, 0 compiler warnings.
    - `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`: TEST BUILD SUCCEEDED.
    - Real weights, `xcodebuild test-without-building` over `loadsTheRealCheckpointEndToEnd()` and `greedyFirstTokensMatchThePythonFixture()`: 2 tests in 1 suite passed in 41.2 s. `loadWeights` verifies with `[.all]`, thus the load itself proves that the 42 new `attn.indexer.wq_b.*` and `attn.indexer.weights_proj.*` tensors each found a module and that no module parameter stayed empty. The greedy test proves the short-prompt output did not change. The endurance test past 12k tokens did not run, because the user skipped it.
    - `swift format --configuration .swift-format`: each of the 6 files comes back unchanged.
    - `swiftlint` with the `no_magic_numbers`, `cyclomatic_complexity`, `function_body_length` and `missing_docs` rules of the review set: 0 findings in the 6 files.
    - next: `/review`.
  timestamp: 2026-08-13T14:37:42.950809+00:00
depends_on:
- 01KZGMY5D5PRZ67GNK9E7B24WS
position_column: doing
position_ordinal: '80'
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

- [x] `Libraries/MLXLLM/Models/DeepseekV4Indexer.swift` exists with the `Indexer` module. Landed as `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, which is the spelling every other DeepSeek-V4 file of this repository uses.
- [x] Top-k selection is strictly causal: for every query position, every selected key index is less than or equal to the query index.
- [x] Exactly `index_topk` keys are selected per query when enough history exists, and fewer (not garbage) when it does not.
- [x] The `indexer.*` drop is removed from `DeepseekV4Model.sanitize` and those weights load; the `compressor.*` drop remains, with a comment naming the Compressor task. Note that `attn.indexer.compressor.*` holds the `.compressor.` name too, thus task `^tty95f4` lands that set beside `attn.compressor.*`.
- [x] Short-prompt output unchanged from before this task.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4IndexerTests.swift`. Landed as `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift`.
- [x] Test: randomized causal-mask property test over many shapes — no selected index ever exceeds the query index. This is the off-by-one guard.
- [x] Test: selection count equals `index_topk` for a long-enough sequence, and degrades gracefully at positions 0 and 1.
- [x] Test: `sanitize` no longer drops `indexer.*` but still drops `compressor.*`.
- [x] Regression: `swift test --filter DeepseekV4ModelTests` still green — short prompts unaffected.
- [x] Run: `swift test --filter DeepseekV4IndexerTests` — all pass.

## Workflow
- Use `/tdd` — write the causal property test first; it is cheap and catches the classic off-by-one.
#deepseek-v4