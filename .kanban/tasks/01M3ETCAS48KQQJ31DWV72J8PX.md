---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3ew17rw3962nq56tvvh9rks
  text: |-
    ### Qwen3.8 baseline BEFORE any code change (HEAD a777c60)

    Build: `xcodebuild build-for-testing ... IntegrationTesting` -> TEST BUILD SUCCEEDED.
    All three suites pass on the current code. Numbers come from the unified log (subsystem `com.apple.FoundationModels-MLX`).

    **Qwen35SessionPromptCacheTests** — 3 of 3 pass (21 s).

    | test | turn | rendered | reused (cached) | fed |
    |---|---|---|---|---|
    | in memory | 1 | 73 | 0 | 73 |
    | in memory | 2 | 126 | 101 | 25 |
    | warm from disk | 1 | 73 | 0 | 73 |
    | warm from disk | 2 | 126 | 101 | 25 |
    | restored file (rope state test) | 2 | 126 | 101 | 25 |

    Spilled file: ledger 101 tokens, render 73 tokens, 64 offsets all equal 101, ropeDeltas true. Warm text "teal" equals uncached text "teal". Uncached turn 2 cached 0.

    **Qwen35AgenticPromptCacheAssessmentTests** — 3 of 3 pass (491 s). Qwen3.8-27B:

    | run | round | rendered | reused (cached) | fed |
    |---|---|---|---|---|
    | memory (CarriesAcross) | 1 | 27434 | 0 | 27434 |
    | memory (CarriesAcross) | 2 | 27616 | 27581 | 35 |
    | memory (CarriesAcross) | 3 | 27701 | 27666 | 35 |
    | memory (CarriesAcross) | 4 | 27787 | 27751 | 36 |
    | memory (CarriesAcross) | 5 | 27874 | 27838 | 36 |
    | cold control | - | 27874 | 0 | 27874 |
    | disk run | 1 | 27434 | 0 | 27434 |
    | disk run | 2 | 27616 | 27581 | 35 |
    | disk run | 3 | 27701 | 27666 | 35 |
    | disk run | 4 | 27787 | 27751 | 36 |
    | disk run | 5 | 27874 | 27838 | 36 |
    | disk run (memory reference) | 1..5 | same as memory rows | same | same |

    Cached round 5 and cold control share 32 of the first 32 generated tokens.

    **PromptCacheSpoolCostAssessmentTests** — 5 of 5 pass (152 s). qwen3.8-27b: restoredToken=15 originalToken=15 at context 4096 and 32768; maxAbsLogitDifferences over 8 decode steps = [0.0 x 8].
  timestamp: 2026-09-26T12:46:23.516371+00:00
- actor: claude-code
  id: 01m3ew77pg9arm5w3je7j4537c
  text: |-
    ### Audit (step 4), before the code change (HEAD a777c60)

    **A. Writes of `offset` on an `ArraysCache` / `MambaCache` outside `advance`** (searched `offset +=`, `offset -=`, `offset =`, `.offset =` in `Libraries/`)
    - `Libraries/MLXLLM/Models/FalconH1.swift:526-527` — `cache.advance(y.dim(1))` then `cache.offset += y.dim(1)`. Upstream has the same two lines. With the new `advance` this is a DOUBLE COUNT. Fix: delete line 527. Guard: `HybridRecurrentCacheOffsetTests` fixture `falconH1` (offset after prefill and each decode step) plus the new model-free double-count test.
    - `Libraries/MLXLMCommon/KVCache.swift:1661` — `MambaCache.advancePosition(by:)` does `advance` then `offset +=`. Deleted by this task.
    - `Libraries/MLXLMCommon/KVCache.swift:1469` — `ArraysCache.copyContents(to:)` copies `offset`. Copy, not a move. No double count.
    - `Libraries/MLXLMCommon/KVCache.swift:1674-1678` — `MambaCache.saveSpeculativeCheckpoint(advancedBy:)` stores `offset + tokenCount` (the fork already stores the position after the checkpoint token; upstream stores `offset`). The callers (`Libraries/MLXLLM/Models/Qwen35.swift:344-349`, `Libraries/MLXVLM/Models/Qwen35.swift:683-704`) save BEFORE they call the advance, thus `offset + tokenCount` is the correct position with the new `advance`. No double count. Existing guard: `Qwen35RecurrentCachePositionTests.aSpeculativeCheckpointRestoresThePositionOfItsToken` and the VLM variant.
    - `Libraries/MLXLMCommon/KVCache.swift:1689` — `restoreSpeculativeCheckpoint()` sets `offset = checkpoint.offset`. Assignment, not a move. No double count.
    - `Libraries/MLXLMCommon/KVCache.swift:2744` — prompt-cache load `apply(_:to:layer:)` sets the recorded offset on an `ArraysCache` (its saved values do not hold the offset). Assignment. No double count.
    - `Libraries/MLXLMCommon/KVCache.swift:1735-1740` — `CacheList.offset` is the max of the children; the setter is a `preconditionFailure`. With FalconH1 line 527 deleted, each child moves once.
    - `ArraysCache.filter(batchIndices:)` (`KVCache.swift:1479`) and `extend(other:)` (`KVCache.swift:1488`) do not touch `offset`. They have no caller in `Libraries/`.
    - Result: 1 double count (FalconH1:527). No other.

    **B. Reads of a cache offset in a hybrid model where the cache can be an `ArraysCache` / `MambaCache`**
    - Every RoPE / mask reader in the hybrid models reads the ATTENTION cache, never the recurrent one: `FalconH1.swift:311` (attention sub-cache `cache?[1]`), `FalconH1.swift:695` (`cache[0]?[1]`), `BaichuanM1.swift:118,133` (`CacheList[1]`), `GraniteMoeHybrid.swift:249,479`, `Jamba.swift:466` (`attnIdx`), `LFM2.swift:160,349`, `LFM2MoE.swift:161,410`, `NemotronH.swift:664`, `Qwen3Next.swift:93,596,658,722` (`faIdx`), `Qwen35.swift` (LLM) `559,814,936,1010,1210` (`faIdx`), `LFM2VL.swift:346,526`, VLM `Qwen35.swift:399,406,945,1084,1285` (`faIdx`).
    - `createSSMMask` (`KVCache.swift:445`) reads `leftPadding`/`lengths` through `makeMask`, not `offset`.
    - Generic readers of every cache: `ExecutorPromptCache.swift:1153-1154` (`caches.first?.offset`, `caches.map(\.offset)`), `ExecutorPromptCacheFile.swift:150`, `PromptCacheReusePolicy.swift:447` (`caches.first?.offset`), `KVCachePlan.swift:372-384` (attention leaves only). The first three NEED the recurrent offset to equal the ledger (they refuse the cache otherwise); none assumes 0.
    - `Tests/MLXLMTests/KVCacheConfigurationTests.swift:384,391` (test `modelCacheOwnsHybridProgressAcrossPrefillAndDecode`) asserts `recurrent.offset == 0` after its helper `HybridProgressModel` (line 513) calls `advance`. This is a 0-assuming reader. Fix: the assertion becomes 3 and 4 (the processed tokens), the new rule.
    - Result: no library reader assumes 0. One test reader assumes 0 (above), fixed in this task.

    **C. Every other caller of `advance(_:)`** (`ArraysCache`)
    - Libraries: only the model call sites (FalconH1:526; all others call `advancePosition(by:)` now and go back to `advance`). No batch code (`BatchKVCache` does not exist in this fork), no merge/extend caller.
    - Tests: `KVCacheTests.swift:731` (upstream test that asserts NO offset move; changes), `KVCacheTests.swift:814` (checks lengths/leftPadding only; still correct), `KVCacheConfigurationTests.swift:513` (see B). `PromptCacheSaveInputTests.swift:124` is a local helper named `advance`, not `ArraysCache.advance`. `ChatSessionTests.swift:1053` is unrelated.
    - `advancePosition(by:)` callers to change to `advance`: `ExecutorPromptCacheTests.swift:538,772`, `PromptCacheTemplateRestoreTests.swift:1505`, the 15 library sites listed in the task (LLM FalconH1 excluded) plus `BaichuanM1.swift:147`.

    **D. Model types whose `newCache` makes a `MambaCache` / `ArraysCache`** (searched `MambaCache(` / `ArraysCache(` in `Libraries/MLXLLM`, `Libraries/MLXVLM`, then the type registries)
    - LLM: `nemotron_h` (NemotronH.swift:748), `jamba` (Jamba.swift:493,526), `mamba2` (Mamba2.swift:279), `granitemoehybrid` (GraniteMoeHybrid.swift:531), `lfm2` (LFM2.swift:406), `lfm2_moe` (LFM2MoE.swift:500), `baichuan_m1` (BaichuanM1.swift:308, in a `CacheList`), `falcon_h1` (FalconH1.swift:780,844, in a `CacheList`), `qwen3_next` (Qwen3Next.swift:776,784), `qwen3_5` (`Qwen35Model`, Qwen35.swift:1248 -> 1132), `qwen3_5_moe` (`Qwen35MoEModel` subclasses `Qwen35Model`), `qwen3_5_text` (`Qwen35TextModel`, Qwen35.swift:1132).
    - VLM: `lfm2_vl` / `lfm2-vl` (LFM2VL.swift:1118), `qwen3_5` (VLM `Qwen35`, Qwen35.swift:1038), `qwen3_5_moe` (VLM `Qwen35MoE` subclasses `Qwen35`).
    - Not in `HybridRecurrentCacheOffsetTests` yet: `qwen3_next`, LLM `qwen3_5`, LLM `qwen3_5_moe`, `qwen3_5_text`, VLM `qwen3_5`, VLM `qwen3_5_moe`. All six are added.
    - No `ArraysCache` subclass other than `MambaCache` exists in `Libraries/`. `nemotron_labs_diffusion` and the Qwen 3.5 MTP drafter do not make a recurrent cache.
  timestamp: 2026-09-26T12:49:40.048788+00:00
- actor: claude-code
  id: 01m3ex4cckxwv3521y58ey6xc1
  text: |-
    ### TDD record
    - RED (before the library change, tests only): MLXLMTests 1241 Swift Testing tests, 5 issues, exactly the new rule: `testArraysCacheAdvanceMovesOffsetAndSequenceMetadata`, `testMambaCacheAdvanceMovesOffsetByTheTokenCount`, `testLayerStyleAdvanceCountsEachTokenOnceOverAPrefillAndDecodeSteps`, and `modelCacheOwnsHybridProgressAcrossPrefillAndDecode` (2 lines, its old assertion `recurrent.offset == 0` assumed 0).
    - GREEN: after `advance` moves `offset`, `advancePosition(by:)` is deleted and each call site uses `advance`: MLXLMTests 696 XCTest 0 failures, 1241 Swift Testing passed. `HybridRecurrentCacheOffsetTests` runs 16 fixtures (10 old + qwen3Next, qwen35Text, qwen35, qwen35MoE, qwen35VL, qwen35MoEVL) for each test, plus the new test `restoredPrefillExtendsExactlyTheLedger`.

    ### Discoveries
    - VLM `Qwen35` has no `callAsFunction(_:cache:)` (fatalError in the default) and a precondition refuses warm caches without `qwen35.ropeDeltas` state. The fixture helper therefore calls `callAsFunction(LMInput.Text, cache:, state:)` and threads the returned state through a `HybridRun` (caches + state). The disk helper saves and restores that state too, as the executor does.
    - A tiny Qwen 3.5 MoE with `num_experts_per_tok = 1` stops the process with `Assertion failed: (out.size() != in.size()), function eval_gpu, file reduce.cpp` (the top-k score normalization sums over one expert). The fixture uses 4 experts and 2 per token.
    - `qwen35TextConfigJSON` gained `expertsPerToken`, `hiddenLayers`, `fullAttentionInterval` parameters (defaults keep the old JSON). `qwen35VLMConfigJSON(textConfigJSON:)` and `qwen35WrappedTextConfigJSON(modelType:textConfigJSON:)` became internal so the offset suite reuses them.
    - All `advance` lines of the 12 upstream model files are textually equal to `upstream/main`; FalconH1 differs only by the deleted manual `offset +=`, BaichuanM1 has an `advance(L)` that upstream lacks (task step 2).
  timestamp: 2026-09-26T13:05:35.123129+00:00
- actor: claude-code
  id: 01m3ex4j6pq5a82gn75t0pm4tv
  text: |-
    ### implement — changed
    - evidence: 20 files — Libraries/MLXLMCommon/KVCache.swift, Libraries/MLXLLM/Models/{BaichuanM1,FalconH1,GraniteMoeHybrid,Jamba,LFM2,LFM2MoE,Mamba2,NemotronH,Qwen35,Qwen3Next}.swift, Libraries/MLXVLM/Models/{LFM2VL,Qwen35}.swift, Tests/MLXLMTests/{HybridRecurrentCacheOffsetTests,KVCacheTests,KVCacheConfigurationTests,PromptCacheTemplateRestoreTests,Qwen35MTPTests}.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, CLAUDE.md
    - next: /test (five bundles, MLXLMTests 3 times, IntegrationTesting compile, three Qwen3.8 suites)
  timestamp: 2026-09-26T13:05:41.078838+00:00
- actor: claude-code
  id: 01m3ey9geq4byhffdja2xg86at
  text: |-
    ### Qwen3.8 before/after reuse table (before = a777c60, after = working tree of this change)

    All three suites pass after the change. No reuse number is lower than the baseline; each number is equal.

    **Qwen35SessionPromptCacheTests** (3 of 3 pass, 16 s)
    | test | turn | rendered before/after | reused before/after | fed before/after |
    |---|---|---|---|---|
    | in memory | 1 | 73 / 73 | 0 / 0 | 73 / 73 |
    | in memory | 2 | 126 / 126 | 101 / 101 | 25 / 25 |
    | warm from disk | 1 | 73 / 73 | 0 / 0 | 73 / 73 |
    | warm from disk | 2 | 126 / 126 | 101 / 101 | 25 / 25 |
    | restored file (rope state) | 2 | 126 / 126 | 101 / 101 | 25 / 25 |
    After: spilled file ledger 101, render 73, every offset 101, ropeDeltas true; warm "teal" = uncached "teal"; uncached cached 0.

    **Qwen35AgenticPromptCacheAssessmentTests** (3 of 3 pass, 509 s; the prefill-time check passed on the first run)
    | run | round | rendered before/after | reused before/after | fed before/after |
    |---|---|---|---|---|
    | memory | 1 | 27434 / 27434 | 0 / 0 | 27434 / 27434 |
    | memory | 2 | 27616 / 27616 | 27581 / 27581 | 35 / 35 |
    | memory | 3 | 27701 / 27701 | 27666 / 27666 | 35 / 35 |
    | memory | 4 | 27787 / 27787 | 27751 / 27751 | 36 / 36 |
    | memory | 5 | 27874 / 27874 | 27838 / 27838 | 36 / 36 |
    | disk | 1 | 27434 / 27434 | 0 / 0 | 27434 / 27434 |
    | disk | 2 | 27616 / 27616 | 27581 / 27581 | 35 / 35 |
    | disk | 3 | 27701 / 27701 | 27666 / 27666 | 35 / 35 |
    | disk | 4 | 27787 / 27787 | 27751 / 27751 | 36 / 36 |
    | disk | 5 | 27874 / 27874 | 27838 / 27838 | 36 / 36 |
    After: cached round 5 and cold control share 32 of 32 tokens; warm prefill of rounds 2-5 0.29-0.37 s against 67 s cold.

    **PromptCacheSpoolCostAssessmentTests** (5 of 5 pass, 124 s): qwen3.8-27b restoredToken=15 originalToken=15 at 4096 and 32768 (before and after); maxAbsLogitDifferences over 8 decode steps [0.0 x 8] (before and after).

    Note: part of the unified log for the agentic run was evicted; the "after" numbers come from the xcodebuild output of the same run.
  timestamp: 2026-09-26T13:25:51.703638+00:00
- actor: claude-code
  id: 01m3ey9k1nbjv5y0cb9r67demv
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` only the known warning `missing creator for mutated node`; `xcrun xctest` MLXLMTests x3: 696 XCTest 0 failures + 1241 Swift Testing passed (each run); MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 329 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed; 0 skipped. IntegrationTesting build-for-testing TEST BUILD SUCCEEDED; Qwen35SessionPromptCacheTests 3/3, Qwen35AgenticPromptCacheAssessmentTests 3/3, PromptCacheSpoolCostAssessmentTests 5/5.
    - next: /commit
  timestamp: 2026-09-26T13:25:54.357983+00:00
- actor: claude-code
  id: 01m3eyb6eate86kk28emasbbcj
  text: |-
    ### commit — changed
    - evidence: 355bcab fix(kv-cache): make ArraysCache.advance(_:) move offset and remove MambaCache.advancePosition(by:)
    - next: /review v72j8px HEAD~1..HEAD
  timestamp: 2026-09-26T13:26:46.986508+00:00
- actor: claude-code
  id: 01m3eyjj2ksd8ghkvfr3xsr69h
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 3 findings (3 confirmed, 1 refuted): Tests/MLXLMTests/KVCacheTests.swift:729, :771, :781 `code-hygiene/idioms-swift` swiftTestingTestCaseNames
    - next: /implement — drop the `test` prefix of the three new Swift Testing functions
  timestamp: 2026-09-26T13:30:48.275801+00:00
- actor: claude-code
  id: 01m3eyjm8grg57ncegncgq2b90
  text: |-
    ### finish iteration 1 — findings
    - implement: changed (20 files)
    - test: green (MLXLMTests x3 696 XCTest + 1241 Swift Testing; 70; 329; 7; 5; 0 skipped; IntegrationTesting build OK; 3 Qwen3.8 suites pass, reuse equal to baseline)
    - commit: 355bcab
    - review: findings — 3 × swiftTestingTestCaseNames at KVCacheTests.swift:729, :771, :781
  timestamp: 2026-09-26T13:30:50.512838+00:00
- actor: claude-code
  id: 01m3eyr8fjdrw81c9hgmbmjx5q
  text: |-
    ### implement — changed
    - evidence: Tests/MLXLMTests/KVCacheTests.swift — the three new Swift Testing functions lose the `test` prefix (`arraysCacheAdvanceMovesOffsetAndSequenceMetadata`, `mambaCacheAdvanceMovesOffsetByTheTokenCount`, `layerStyleAdvanceCountsEachTokenOnceOverAPrefillAndDecodeSteps`), as `swiftformat --rules swiftTestingTestCaseNames` writes them. The other `test`-prefixed functions of the file existed before this task; the review skill drops findings that ask to refactor existing tests, and the review reported none on them.
    - next: /test
  timestamp: 2026-09-26T13:33:55.058352+00:00
- actor: claude-code
  id: 01m3eztg1zbkznq1arxk58hcrv
  text: |-
    ### test — green
    - evidence: build only the known warning; MLXLMTests x3: 696 XCTest 0 failures + 1241 Swift Testing passed each run; MLXGuidedGenerationTests 70; MLXFoundationModelsTests 329; CXGrammarTests 7; MLXHuggingFaceMacrosTests 5; 0 skipped. IntegrationTesting TEST BUILD SUCCEEDED; Qwen35SessionPromptCacheTests 3/3 (turn 2 cached 101 of 126, restored 101, warm = uncached "teal"); Qwen35AgenticPromptCacheAssessmentTests 3/3 (rounds 2-5 fed 35/35/36/36, cached 27581/27666/27751/27838 in memory and on disk); PromptCacheSpoolCostAssessmentTests 5/5 (restoredToken=originalToken=15, logit differences 0.0 x 8). All equal to the baseline.
    - next: /commit
  timestamp: 2026-09-26T13:52:36.927309+00:00
- actor: claude-code
  id: 01m3ezx7zqwwgd99dm7kgxjvpz
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (256b187) — 0 findings, 0 refuted; every prior finding item is checked
    - next: move to done
  timestamp: 2026-09-26T13:54:06.967839+00:00
- actor: claude-code
  id: 01m3ezxab1nyz8fpb6fpnjtpyr
  text: |-
    ### finish iteration 2 — clean
    - implement: changed (KVCacheTests.swift: three new Swift Testing names without the `test` prefix)
    - test: green (MLXLMTests x3 696 + 1241; 70; 329; 7; 5; 0 skipped; IntegrationTesting build OK; 3 Qwen3.8 suites pass, reuse equal to baseline)
    - commit: 256b187
    - review: clean (0 findings)
  timestamp: 2026-09-26T13:54:09.377291+00:00
position_column: done
position_ordinal: ffb880
title: Make ArraysCache.advance(_:) move offset, remove MambaCache.advancePosition(by:), and prove that every MambaCache model caches (Qwen3.8 first)
---
#prompt-cache

## Why the current state is not a fix

`ArraysCache.advance(_:)` (`Libraries/MLXLMCommon/KVCache.swift:1533`) moves `lengths` and `leftPadding` but NOT `offset`. `MambaCache.advancePosition(by:)` (`KVCache.swift:1659`) calls `advance` and then moves `offset`. Thus there are two public methods for one operation, and the one with the natural name gives a wrong cache for this fork:

- This fork's prompt cache keeps a cache only when EVERY cache offset equals the token ledger length (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`, `mainCacheIsAligned`). A recurrent cache whose offset stays at 0 is refused, in memory and on disk.
- Upstream (`ml-explore/mlx-swift-lm`, `upstream/main`) has no such check, thus upstream never needed a recurrent offset. Every upstream hybrid model calls `advance(_:)`: FalconH1, GraniteMoeHybrid, Jamba, LFM2, LFM2MoE, Mamba2, NemotronH, Qwen35 (3 sites), Qwen3Next (2 sites), LFM2VL, VLM Qwen35. The fork changed all of them to `advancePosition(by:)` in tasks ^qr0p806, ^qfennz0 and ^znyes82.
- Consequence: every upstream merge that adds a new hybrid model, or a new `advance` call site, silently brings the defect back. The doc comment added in ^qfennz0 does not stop that. 10 models had the defect before it was found.

## What

Make the natural method correct, so that one method has one meaning and upstream code is correct as it is:

1. In `Libraries/MLXLMCommon/KVCache.swift`, make `ArraysCache.advance(_:)` also do `offset += N`. Update its doc comment: the offset is the number of tokens the cache holds, which the prompt cache compares with the ledger.
2. Delete `MambaCache.advancePosition(by:)`. Change every call site back to `advance(_:)`: the models in `Libraries/MLXLLM/Models` and `Libraries/MLXVLM/Models` listed above, and `BaichuanM1.swift:147` (`convCache`). This returns those lines to the upstream text and shrinks the fork diff.
3. `Libraries/MLXLLM/Models/FalconH1.swift:526-527` calls `advance` AND `cache.offset += y.dim(1)` (same in upstream). With step 1 this counts twice. Delete the manual `offset +=` line.
4. Audit before the change, and write the result on this task:
   - Every write of `offset` on an `ArraysCache` / `MambaCache` outside `advance` (search `offset +=`, `offset =` in `Libraries/`), to find other double counts. Include the speculative checkpoint code (`saveSpeculativeCheckpoint(advancedBy:)` and its restore) and `CacheList`.
   - Every read of a cache offset in a hybrid model where the cache can be an `ArraysCache` / `MambaCache` (for example `cache?.first?.offset`, `cache[0].offset`, mask or RoPE position from a cache). A reader that assumed 0 must be found now.
   - Every other caller of `advance(_:)`, including batch code (`BatchKVCache`, merge/extend of `ArraysCache`) and tests.
   - The complete list of model types in `Libraries/MLXLLM/Models` and `Libraries/MLXVLM/Models` whose `newCache` makes a `MambaCache` or `ArraysCache` (directly or inside a `CacheList`). Search, do not rely on the list above.
5. Tests:
   - `Tests/MLXLMTests/KVCacheTests.swift:726` `testArraysCacheAdvanceUpdatesSequenceMetadataOnly` (an upstream test) asserts that `advance` does NOT move `offset`. Change it to assert the new rule and rename it (for example `testArraysCacheAdvanceMovesOffsetAndSequenceMetadata`). State in the commit message that this intentionally diverges from upstream, and why.
   - Test helpers that call `advance` or `advancePosition` (`Tests/MLXLMTests/KVCacheConfigurationTests.swift:513`, `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift:1505`, `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift:538` and `:772`, `Tests/MLXLMTests/KVCacheTests.swift:814`): update to `advance`, and check that none now double-counts.
   - Add one unit test: a `MambaCache` after `advance(n)` has `offset == n`, and a model-free double-count guard: a layer-style sequence of `advance` calls over a prefill and 3 decode steps gives offset = total tokens.
6. Add a short note to the upstream-merge section of `CLAUDE.md` (or create one next to "Before you commit"): in this fork, `ArraysCache.advance(_:)` moves `offset`; after an upstream merge, search the merged models for `advance(` followed by a manual `offset +=` and delete the manual line, and run `HybridRecurrentCacheOffsetTests` and the Qwen3.8 suites below.

## Every MambaCache model must actually cache (the user's requirement)

A correct offset is not the goal; warm reuse is. Prove, for every model type that the audit in step 4 lists, that turn 2 reuses the cache of turn 1 and gives the same output as a cold run.

### Qwen3.8 — the gate that matters most

`mlx-community/Qwen3.8-27B-mxfp4` (`qwen3_5`, MambaCache + KVCacheSimple, in the local Hugging Face cache) is the primary proof. BEFORE the change, run these suites and record the numbers on this task as the baseline. AFTER the change, run them again. The task is not done unless each passes and no reuse number is lower than the baseline:
- `IntegrationTestingTests/Qwen35SessionPromptCacheTests` — warm from memory, warm from disk (`aTurnRestoredFromDiskHoldsTheRopeStateAndGivesTheUncachedText`): turn 2 reuses at least the whole render of turn 1 (baseline 101 of 126 tokens), the file holds `qwen35.ropeDeltas`, all cache offsets equal the ledger, and the warm text equals the uncached text.
- `IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests` — the in-memory test and `hybridModelRestoresEachToolRoundFromDisk`: each round after the first feeds only the new tail (baseline 35 to 36 tokens), and the disk run gives the same reasoning, text and tool calls as the memory run. The prefill-time check of `hybridModelCarriesThePromptCacheAcrossToolRounds` measures time and can fail under external load; if it fails, run it again and record both runs.
- `IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests` for Qwen3.8 — restored logits equal the original over 8 decode steps.
Write a table on this task: for each round/turn, rendered tokens, reused tokens and fed tokens, before and after.

### Every other MambaCache / ArraysCache model

`Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift` already checks, for 10 tiny models, the offsets, that `reconcilePromptCache` extends the ledger, a warm continuation equal to a cold prefill, and a disk restore. It must stay green with no change to its assertions. Add to its fixture list every model type from the audit that is not in it yet, including the VLM Qwen35 (`Libraries/MLXVLM/Models/Qwen35.swift`), which is the model class that Qwen3.8 actually loads through (MLXVLM is tried first for `qwen3_5` with a `vision_config`). For each model, the test must assert that the reuse plan reuses exactly the turn-1 ledger length (not only that it is greater than 0).

## Acceptance Criteria
- [x] `advancePosition(by:)` does not exist in `Libraries/` or `Tests/`.
- [x] `ArraysCache.advance(_:)` moves `offset` by `N`, and its doc comment says so.
- [x] FalconH1 moves each recurrent offset exactly once per token (no manual `offset +=` after `advance`).
- [x] The audit (step 4) is written on this task with file:line for each finding, and each double count or 0-assuming reader it finds is fixed and tested.
- [x] Every model type from the audit is in `HybridRecurrentCacheOffsetTests`, which passes with the original assertions plus the exact-reuse assertion.
- [x] The three Qwen3.8 suites pass after the change, and the before/after reuse table on this task shows no lower number.
- [x] The model call sites listed above are textually equal to `upstream/main` for the `advance` line (check with `git diff upstream/main -- <file>`).
- [x] `CLAUDE.md` has the merge note.

## Tests
- [x] `swift build --build-tests`: only the known warning `missing creator for mutated node`.
- [x] `xcrun xctest .build/out/Products/Debug/<Bundle>.xctest` for all five bundles: 0 failures, 0 skipped (baseline MLXLMTests 696 XCTest + 1238 Swift Testing, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 329, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5, plus new tests). Run MLXLMTests 3 times.
- [x] `xcodebuild build-for-testing -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` compiles, and the three Qwen3.8 suites above pass with `-only-testing:`.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.
- Run the Qwen3.8 baseline BEFORE the first code change.
- Optional follow-up for the user to decide: offer the same change upstream (it is safe there too, and it prevents the same trap for any upstream prompt-cache work).

## Review Findings (2026-09-26 08:26)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 19 file(s) reviewed, 3 not reviewed.

> 3 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M3ETCAS48KQQJ31DWV72J8PX.jsonl` — no validator matches this file
> - `.kanban/tasks/01M3ETCAS48KQQJ31DWV72J8PX.md` — no validator matches this file
> - `CLAUDE.md` — no validator matches this file

- [x] `Tests/MLXLMTests/KVCacheTests.swift:729` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXLMTests/KVCacheTests.swift:771` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXLMTests/KVCacheTests.swift:781` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.