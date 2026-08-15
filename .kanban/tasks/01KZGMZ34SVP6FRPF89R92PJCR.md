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
- actor: claude-code
  id: 01kzxtgf8f6trqtjftm9gyz7v2
  text: |
    ### review — findings
    - evidence: 7 confirmed findings — DeepSeekV4Indexer.swift:56, :82, :83, :141, :215 (in scope); DeepSeekV4.swift:156, :581 (out of scope, earlier commits a75a043a and 2624f899)
    - scope: commit 0a98e37, range HEAD~1..HEAD
    - defect: `review sha HEAD~1..HEAD` gave 0 findings with `attempted: 8`, but a file sweep of the same commit gave 7. The range sweep did not examine the Swift source. `review file` of the new test file gave `attempted: 0`, thus the 399 new test lines got no review.
    - next: correct the 5 findings of DeepSeekV4Indexer.swift, then run the review again. Leave the 2 findings of DeepSeekV4.swift for a later task.
  timestamp: 2026-08-13T15:05:34.735565+00:00
- actor: claude-code
  id: 01kzxthm5b0sgks4d10m7z8vv9
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 6 files. New `DeepSeekV4Indexer.swift` and `DeepSeekV4IndexerTests.swift` (8 tests); new `hasIndexer(layer:)`; an optional indexer submodule on the 21 even layers 2 to 42; the `.indexer.` drop is gone from `sanitize`.
    - test: green — `swift test`, 0 failures, 2 skips that were there before, 0 new warnings. Real weights: the load test and the greedy-parity test pass in 41.2 s.
    - commit: 0a98e37 feat(mlx-lm): add DeepSeek-V4 Indexer for sparse attention chunk selection
    - review: findings — 5 in scope in `DeepSeekV4Indexer.swift` (:56, :82, :83, :141, :215), 2 out of scope in `DeepSeekV4.swift` (:156, :581).
    - ENGINE DEFECT found this round: `review sha HEAD~1..HEAD` answered 0 findings with `attempted: 8`, but `review file` on a file inside that same range answered 5 confirmed findings. The `attempted` count gives the number of validators, not the number of files, thus a count that is not zero does not show that the engine looked. Also `review file` on the new test file answered "Nothing in scope" with `attempted: 0`, thus the 399 new test lines got no review.
  timestamp: 2026-08-13T15:06:12.523379+00:00
- actor: claude-code
  id: 01kzxv1mn2b77mm3zc63wtwtfb
  text: |-
    ### The five findings of `DeepSeekV4Indexer.swift` are corrected

    One file changed: `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`.

    A finding gives one example of a cause, thus each correction went over the whole file first:

    - **`final`** — `class DeepSeekV4Indexer: Module` became `final class DeepSeekV4Indexer: Module`. A sweep of the file gives one class only, thus no other class needs the keyword. `git ls-files '*.swift' | xargs grep -n DeepSeekV4Indexer` gives 9 hits outside the file, and each one builds the type, names it or tests it. Nothing subclasses it, thus `final` breaks no caller.
    - **Doc comments** — `wqB` and `weightsProj` each take a doc comment that states what the projection reads, what it answers and where the score path uses it. A sweep of every member of the file shows these two were the only ones without one: the seven other stored properties, the five static constants, the initializer and the three functions each already had one.
    - **Doc voice** — the rule is "Match voice to kind: imperative verb phrase for effectful methods, noun phrase for values/types." A sweep of the three functions of the file:
      - `callAsFunction`: "Picks the pooled chunks each query of one block reads." became "The pooled chunks each query of one block reads."
      - `chunkScores`: "Scores each pooled chunk against each query of one block." became "The score of each pooled chunk against each query of one block."
      - `chunkVisibility`: "The block-causal visibility of each pooled chunk." — already a noun phrase, thus unchanged.
      - `init`: "Builds the selector of one layer." — unchanged. An initializer has an effect, thus the rule asks for the verb phrase, and every other DeepSeek-V4 file of this repository writes its initializer the same way (`DeepSeekV4.swift:330`, `DeepSeekV4Attention.swift:86`, `:113`, `:280`, `DeepSeekV4HyperConnection.swift:216`, `:350`).

    No code path changed. The change is the `final` keyword and doc text only, thus a real-weights run is not needed.

    ### The two findings of `DeepSeekV4.swift` stay open here

    The standing rule of task `^ag7ant0` makes them a record only, and this task does not correct them. Task `^dhv1ave` now holds them, so that they are not lost.

    ### A note on swiftlint

    This repository holds no `.swiftlint.yml`, thus a bare `swiftlint` run takes the default rule set, not the rule set of the review. That default set gives 3 findings on the file — one `function_parameter_count` and two `identifier_name` for the parameter `x`. The same 3 come from the file as `HEAD` holds it, word for word, thus this change adds none. The name `x` for the block input is the prevailing name of every model file of this repository.
  timestamp: 2026-08-13T15:14:57.314443+00:00
- actor: claude-code
  id: 01kzxv1zekqrevvrbw818hb8qm
  text: |-
    ### The review engine under-reports in range mode — read this before the next review

    The description now carries this under "Engine sweep defect". It is repeated here because it matters for every future review of this repository, not for this task alone.

    - `review sha HEAD~1..HEAD` on commit `0a98e37` answered `findings: 0` with `attempted: 8`, `skipped: 0` and an empty `skipped_files`. A `review file` of `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, a Swift file inside that same range, answered 5 confirmed findings. Thus the range mode did not examine the Swift source of the commit, and it gave no sign of that.
    - `attempted` counts validators, not files. The same count 8 comes back for a sweep of one file. A count that is not zero is therefore no proof that any file got read.
    - `review file Tests/MLXLMTests/DeepSeekV4IndexerTests.swift` answered "Nothing in scope to review" with `attempted: 0`. The 399 new lines of test code got no review at all, and that includes the randomized causal property test.

    **What rests on mutation proofs only, not on an engine verdict:**

    - the causal off-by-one of `chunkVisibility`,
    - the `argPartition(-masked, kth: topK - 1)` path and the `.&& visible` that follows it,
    - the 44-entry `compress_ratios` against the 43 layers of `num_hidden_layers`.

    The mutation proofs sit in the implement comment of 2026-08-13: `(k + 1) * chunkWidth` to `k * chunkWidth` made `everyPickedChunkStandsWhollyBehindItsQuery` fail with 348 issues, and the removal of `.&& visible` made three tests fail with 348, 180 and 5 issues. A later reviewer must not read the clean range sweep as coverage of these three points.
  timestamp: 2026-08-13T15:15:08.371433+00:00
- actor: claude-code
  id: 01kzxv279rvcz0a7pf97mh5b02
  text: |-
    ### implement — changed
    - evidence: 1 file — `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`. The 5 in-scope findings are closed: `:56` `final class`, `:82` `wqB` doc comment, `:83` `weightsProj` doc comment, `:141` `callAsFunction` noun phrase, `:215` `chunkScores` noun phrase. Each cause was then swept over the whole file — one class, thirteen other members with a doc comment already, three functions with the voice rule.
    - `swift test`: 489 + 80 + 282 + 7 tests pass, 0 failures, 0 compiler warnings. Duration 128.6 s. `DeepSeekV4IndexerTests` passed in 2.914 s.
    - `swift format --configuration .swift-format`: the file comes back unchanged.
    - `swiftlint`: 3 findings of the default rule set, and the file as `HEAD` holds it gives the same 3, thus this change adds none. This repository holds no `.swiftlint.yml`.
    - The 2 findings of `DeepSeekV4.swift` stay open, out of scope under the standing rule of `^ag7ant0`. New task `^dhv1ave` holds them.
    - next: `/review`.
  timestamp: 2026-08-13T15:15:16.408159+00:00
- actor: claude-code
  id: 01kzxv4hmdv9nhcy1pqrsnww8t
  text: |-
    ### commit — changed
    - evidence: 4c18b75 fix(mlx-lm): close review findings on DeepSeekV4Indexer (^r92pjcr)
    - next: none — commit is local only, not pushed
  timestamp: 2026-08-13T15:16:32.525190+00:00
- actor: claude-code
  id: 01kzxvryyx74fv5eb39j5zaq23
  text: |
    ### review — clean
    - evidence: `review file Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` gave 0 findings, 4 candidates refuted, 8 validators attempted, 0 skipped. `review sha HEAD~1..HEAD` gave 0 findings, 1 candidate refuted. Scope: commit 4c18b75, one source file, +17/-3.
    - coverage proof: the same `review file` command gave 5 confirmed findings at 09:59 and gives 0 now, thus the sweep does read this file. The clean answer is a result, not the silence of the range-mode defect.
    - the five in-scope findings are closed, each one read from the file and not from a checkbox: line 56 `final class`; `wqB` (line 88) and `weightsProj` (line 97) each with a doc comment; `callAsFunction` (line 155), `chunkVisibility` (line 209) and `chunkScores` (line 229) each with a noun phrase. The initializer (line 125) keeps its verb phrase, because an initializer has an effect. That agrees with the rule.
    - the two findings of `DeepSeekV4.swift` (:156, :581) stay unchecked. They sit on lines of earlier commits, thus the standing rule of `^ag7ant0` releases this task from them, and task `^dhv1ave` holds them. They are not a block against column `done`.
    - the real-weights suite and the endurance test did not run. This commit changed the `final` keyword and doc text only, thus no code path moved.
    - next: none. The task is in column `done`.
  timestamp: 2026-08-13T15:27:41.533488+00:00
- actor: claude-code
  id: 01kzxvt768hx64dc7ry01qxjdh
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 file, `DeepSeekV4Indexer.swift` (+17/-3). The 5 in-scope findings are closed and each cause was swept over the whole file.
    - test: green — the implement step ran `swift test` after the change: 489+80+282+7 tests pass, 0 failures, 0 warnings, 128.6 s. This iteration changed only the `final` keyword and doc text, thus no separate test run followed.
    - commit: 4c18b75 fix(mlx-lm): close review findings on DeepSeekV4Indexer (^r92pjcr)
    - review: clean — `review file` on `DeepSeekV4Indexer.swift` answers 0 findings now, where the same command answered 5 before the correction. That difference shows the command reads the file, thus the clean answer is a result and not the silence of the range-mode defect. The card moved to `done`. The 2 findings in `DeepSeekV4.swift` stay open and out of scope under the rule of `^ag7ant0`; task `^dhv1ave` holds them.
  timestamp: 2026-08-13T15:28:22.728858+00:00
depends_on:
- 01KZGMY5D5PRZ67GNK9E7B24WS
position_column: done
position_ordinal: f280
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

## Review Findings (2026-08-13 09:59)

Scope: commit `0a98e37`, range `HEAD~1..HEAD`.

The command `review sha HEAD~1..HEAD` gave 0 findings. That result is not correct. A file sweep of the same commit gave 7 confirmed findings. Thus this pass records the file-sweep results. Read "Engine sweep defect" below.

### In scope — `DeepSeekV4Indexer.swift`, a new file of this commit

- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:56` — Classes not designed for subclassing should be marked `final` to signal that they are not extension points. This class models a specific layer selector with no indication that subclassing is intended. Change `class DeepSeekV4Indexer: Module {` to `final class DeepSeekV4Indexer: Module {` to mark it as not designed for subclassing.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:82` — Public property `wqB` lacks documentation comment explaining its purpose and role in the indexer. Add a documentation comment above line 82 explaining what `wqB` represents and its role in the indexing pipeline.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:83` — Public property `weightsProj` lacks documentation comment explaining its purpose and role in the indexer. Add a documentation comment above line 83 explaining what `weightsProj` represents and its role in the scoring pipeline.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:141` — Documentation for a pure function (one that computes and returns values with no side effects) should use a noun phrase describing the result, not imperative voice describing the action. Per the rule: 'Match voice to kind: imperative verb phrase for effectful methods, noun phrase for values/types.'. Change to a noun phrase describing the result: '/// The pooled chunks each query of one block reads.' or '/// A selection mask indicating which pooled chunks each query reads.'.
- [x] `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:215` — Documentation for a pure function should use noun phrase, not imperative voice. The function `chunkScores` computes scores without side effects, so its documentation should describe what it returns, not what action it performs. Change to a noun phrase describing the result: '/// The scores of each pooled chunk against each query of one block.' or '/// Chunk scores for each query in this block.'.

#### The correction of the five, 2026-08-13

A finding gives one example of a cause, thus each correction went over the whole file:

- `final`: `DeepSeekV4Indexer` is now `final class`. It is the only class of the file. Nothing subclasses it — `git ls-files '*.swift' | xargs grep -n DeepSeekV4Indexer` gives 9 hits, and each one is a build, a type or a test.
- Doc comments: `wqB` and `weightsProj` each take one. They were the only two members of the file without a doc comment. The other seven stored properties, the five static constants, the initializer and the three functions each already had one.
- Doc voice: `callAsFunction` and `chunkScores` each take a noun phrase now. `chunkVisibility` already had one, thus the three functions of the file agree. The initializer keeps "Builds the selector of one layer.", because an initializer has an effect and because every other DeepSeek-V4 file of this repository writes its initializer the same way.

### Out of scope — lines that commit `0a98e37` did not touch

Standing rule from task `^ag7ant0`: the engine sweeps a full file, thus a finding on a line of an earlier commit is a record only. Do not correct these two here. `git blame` gives commit `a75a043a` for line 156 and commit `2624f899` for lines 579 to 583. Commit `0a98e37` touched only the file header, `compressorSegment` near line 434, and the filter test near line 495.

- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:156` — Class is not designed for subclassing and should be marked `final` to prevent accidental subclassing and enable compiler optimizations. Mark the class as final: `final class DeepSeekV4DecoderLayer: Module {`.
- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:581` — The for-key loop sits at nesting depth 4 (inside three for-loops and a guard statement), exceeding the maximum recommended depth of 3, which impairs readability and makes control flow difficult to follow. Extract the guard and inner for-key loop (lines 579–583) into a separate helper function, e.g. `removePerExpertWeights(from:perExpert:)`, to reduce stackRoutedExperts nesting to 3 levels.

### Engine sweep defect

Report these two items to the person who keeps the review engine. They are not work for this task.

- `review sha HEAD~1..HEAD` gave `findings: 0`, `attempted: 8`, `skipped: 0`, and an empty `skipped_files`. A `review file` of `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, a file of that same commit, gave 5 confirmed findings. Thus the range sweep did not examine the Swift source of the commit, and it gave no sign that it did not. The count `attempted` is also 8 for a sweep of one file, thus `attempted` does not count files and gives no proof of coverage.
- `review file Tests/MLXLMTests/DeepSeekV4IndexerTests.swift` gave "Nothing in scope to review" with `attempted: 0`. Thus no agent examined the 399 new lines of test code. The randomized causal property test, which is the guard against the off-by-one of the causal rule, got no review.

#### What the defect means for the correctness of this task

The range mode under-reports, thus no engine verdict stands behind the three points below. Each one rests on the mutation proofs of the implement step only, which the comment thread records:

- The causal off-by-one of `chunkVisibility`. The mutation `(k + 1) * chunkWidth` to `k * chunkWidth` made `everyPickedChunkStandsWhollyBehindItsQuery` fail with 348 issues.
- The `argPartition(-masked, kth: topK - 1)` path and the `.&& visible` that follows it. The mutation that removed the final `.&& visible` made three tests fail, with 348, 180 and 5 issues.
- The 44-entry `compress_ratios` against the 43 layers of `num_hidden_layers`. The bound guard of `compressRatio(ofLayer:)` covers it, and `fixtureNamesIndexerKeysOnTheEvenLayersFromTwoUp` pins the layer rule.

A later review of these three points must not read the clean range sweep as coverage.

### Files of this commit that the engine swept clean

- `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift` — 0 findings, 7 candidates refuted.
- `Libraries/MLXLLM/Models/DeepSeekV4Configuration.swift` — 0 findings, 1 candidate refuted.

## Review Findings (2026-08-13 10:21)

Scope: commit `4c18b75`, range `HEAD~1..HEAD`. That commit changed one source file, `Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift`, by 17 added lines and 3 removed lines.

No new finding. The board keeps this section for the record of the pass.

### The engine gave a clean answer, and the answer has a proof of coverage

- `review file Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift` — 0 findings, 4 candidates refuted, 8 validators attempted, 0 skipped.
- `review sha HEAD~1..HEAD` — 0 findings, 1 candidate refuted, 8 validators attempted, 0 skipped.

The same `review file` command gave 5 confirmed findings on the pass of 09:59 and gives 0 now. Thus the file sweep does look at this file, and the clean answer is a result and not the silence of the range-mode defect above.

### The five findings of 09:59 are closed, each one read from the file

- Line 56 reads `final class DeepSeekV4Indexer: Module {`.
- `wqB` (line 88) and `weightsProj` (line 97) each hold a doc comment that states what the projection reads, what it answers, and where the score path uses it.
- `callAsFunction` (line 155) reads "The pooled chunks each query of one block reads." `chunkVisibility` (line 209) reads "The block-causal visibility of each pooled chunk." `chunkScores` (line 229) reads "The score of each pooled chunk against each query of one block." Thus each of the three functions holds a noun phrase.
- The initializer (line 125) keeps "Builds the selector of one layer." The rule asks for a verb phrase for a method that has an effect, and an initializer has an effect. Thus this line agrees with the rule and is not a miss.

### The two open lines of `DeepSeekV4.swift` do not hold this task back

The two unchecked items of the section "Out of scope" above stay unchecked on purpose. `git blame` gives an earlier commit for each line, thus the standing rule of task `^ag7ant0` releases this task from them. Task `^dhv1ave` holds the two findings word for word in column `todo`. They are not a block against column `done` for this task.