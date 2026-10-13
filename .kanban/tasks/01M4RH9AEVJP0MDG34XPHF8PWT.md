---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rzce08r3nj65ygamxwe756
  text: |-
    ### Research 2026-10-12 (session scratchpad dl/, Python sources)
    - Put `+inf` in the freqs table for the 224 no-position pairs. Then one `fast.rope` turns the full 512-wide head with no slice or concatenation (`mlx_vlm .../language.py:308-339`).
    - Inverse rotation: use the negated table, kept after the first build.
    - Pooled rows: use `freq_scale = ratio` and `offset / ratio` (`mlx_vlm .../language.py:610-616, 646-649`).
  timestamp: 2026-10-12T21:13:02.216881+00:00
- actor: claude-code
  id: 01m4vpfb6w4ktak77g8t752fqt
  text: |-
    ### Research 2026-10-13 (implement)
    - MLX kernel (`backend/metal/kernels/rope.metal`, freqs variant): `inv_freq = 1.0 / freqs[i]`, `theta = scale * (pos + offset) * inv_freq`, `metal::fast::cos/sin`, rotation in float32, one round to the dtype of x. The op path rounds cos/sin to the dtype of x and rounds each product. Thus the two paths cannot be bit-exact for bfloat16; the parity tests measure the gap.
    - A pair with period `+inf` gets `inv_freq = 0`, `theta = 0`, thus it does not move. A negated table gives the inverse rotation.
    - `fast.rope` needs `ndim >= 3` and turns along axis -2. Queries/keys/output are `(B, H, L, D)`; pooled rows `(B, C, D)` get a head axis first, as in mlx-vlm.
    - `CompiledTrace` sends `innerState()` (which includes arrays whose names start with `_`) as trace inputs. Thus tables stored as `_`-named `MLXArray` values on the rope module are trace inputs, not captured constants. The position goes into the traces as a scalar int32 `MLXArray` argument; no Swift Int is captured.
    - Plan: the layer rope keeps one period table for each (head width, position stride, direction) that the layer reads, built one time in `init` as leaf arrays. Pooled rows use the table divided by the compress ratio and the offset divided by the ratio (mlx-vlm `freq_scale`). The op path (`cosSin` + `applyPartialRoPE`) moves to the test target as the parity oracle, because no production caller stays.
  timestamp: 2026-10-13T22:35:03.772867+00:00
- actor: claude-code
  id: 01m4vrsr6e2s0ta1dgbrsyc1qv
  text: |-
    ### Implementation landed (not committed)
    - `DeepSeekV4Math.rotaryPeriods(inverseFrequency:headDim:positionStride:inverse:)` builds the period table: `+inf` for the (headDim - ropeDim)/2 pairs with no position, `1 / invFreq / stride` for the rotary pairs, negated for the inverse rotation. `DeepSeekV4Math.applyRoPE(_:periods:offset:)` is one `MLXFast.RoPE` call (traditional, full head width, array offset).
    - `DeepSeekV4RoPE` builds every table its layer reads one time in `init` (from `yarnInvFreq`, thus `rope_theta` / `compress_rope_theta` and YaRN go into the table): forward at `head_dim`, backward at `head_dim`, forward at `index_head_dim` (indexer layers), and pooled tables at the compress ratio (`freq_scale`) for each compressor. The tables are `_`-named arrays of the module, thus compiled traces read them as state inputs. `rotatedChunks(_:firstPosition:)` uses offset / ratio.
    - Attention queries, keys, output (inverse), indexer queries and compressor chunks each turn with one `MLXFast.RoPE` call. `rotaryTables`, `cosSin`, `applyPartialRoPE`, `applyInversePartialRoPE` and the three `ropeDim` properties are gone (no caller left). The decode traces take one scalar int32 position array (`AttentionInputSlot.position`, `LayerTailSlot.position`) instead of the cos/sin tables.
    - The old op path lives on as the test oracle `OpPathRoPE` in `Tests/MLXLMTests/DeepSeekV4RoPEParityTests.swift`.
    ### Parity (unit)
    - The no-position dimensions keep their bits on every case (bit-exact).
    - Rotary dimensions are not bit-exact: the kernel reads `metal::fast::cos/sin` and rounds one time; the op path rounds cos, sin and each product. Measured largest gap, float32: 2.4e-7 at offsets 0 and 9; 2.0e-5 at offset 4093 and 1.6e-4 at offset 32765 on the `rope_theta` layer; 4.8e-6 or less on `compress_rope_theta` + YaRN layers, inverse rotation and pooled rows (ratio 4 and 128, widths 512 and 128). bfloat16: 0.015625 (one bfloat16 step at magnitude 2-4) on every case. Tolerances: 3.2e-4 (float32), 0.03125 (bfloat16).
    - New model test: three consecutive decode steps through the compiled traces give the bits of the op path, thus the position is not a trace constant.
    ### What did not work / notes
    - A `TableKind` struct key in a dictionary would make the tables `.other` items, which `innerState()` skips; then the traces would capture them as constants. String keys keep them in the module state.
  timestamp: 2026-10-13T23:15:41.902364+00:00
- actor: claude-code
  id: 01m4vrt48axqan8ntzdby76b2x
  text: |-
    ### Measurements 2026-10-13 (real checkpoint, GPU idle before each run, one GPU job at a time)
    - Correction to the research note: the pooled rows go to `MLXFast.RoPE` as `(B, C, D)` with no extra head axis; the kernel turns axis -2 of a 3-axis array, and the pooled-row parity tests pass on that shape.
    - Unit tests: `swift build --build-tests` clean (0 warnings), `xcrun xctest MLXLMTests.xctest`: 716 XCTest 0 failures, 1389 Swift Testing passed.
    - Integration: `greedyFirstTokensMatchThePythonFixture()` (xcodebuild, alone) passed after 23.5 s.
    - Bench `step` (A direct sync / C generate, ms): HEAD 128: 38.9-39.5 / 37.8-38.5; 4k: 40.3-40.4 / 43.6-45.9; 16k: 41.8 / 44.3-47.0. After 128: 36.7-37.4 / 36.9-37.4; 4k: 38.2-38.8 / 40.4-41.8; 16k: 39.9-40.5 / 43.2-45.3. Files: scratchpad/dsv4bench/step-head-phf.txt, step-after-phf.txt.
    - Bench full `generate` ms/token, HEAD -> after: 32: 37.7 -> 34.8; 2k: 39.0 -> 36.8; 8k: 39.9 -> 37.3; 32k: 41.7 -> 39.7. Prefill tok/s HEAD -> after: 2k 412.8 -> 416.7; 8k 387.1 -> 391.3; 32k 283.3 -> 285.8 (not worse). Files: full-head-phf.txt, full-after-phf.txt.
    - Cache checks: HARD reuse PASS both modes; tie-aware PASS both modes (thinking=false now warm == cold text YES; thinking=true first difference token #3, gap 0.0).
    - Generated text changed at ctx 8192 and 32768. Tie-aware headtie (new build, teacher forced): 8192 first differing token #5 head " very" 34.0 vs new " long" 34.5, gap 0.5; 32768 first differing token #14 head " massive" 37.0 vs new " text" 37.0, gap 0.0. Both are within the cut-spread margin 4.0, prefix argmax mismatches 0. The 8192 pair " very"/" long" is the same near tie ^9xmds28 recorded in the other direction. File: headtie-phf.txt.
  timestamp: 2026-10-13T23:15:54.250220+00:00
- actor: claude-code
  id: 01m4vrtgzphkxcs8rfde8yc5hr
  text: |-
    ### implement — changed
    - evidence: 10 files — Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, DeepSeekV4Attention.swift, DeepSeekV4Indexer.swift, DeepSeekV4Compressor.swift, DeepSeekV4.swift; Tests/MLXLMTests/DeepSeekV4RoPEParityTests.swift (new), DeepSeekV4MathHelpersTests.swift, DeepSeekV4AttentionTests.swift, DeepSeekV4IndexerTests.swift, DeepSeekV4ModelTests.swift. `swift build --build-tests` 0 warnings; `xcrun xctest MLXLMTests.xctest` 716 XCTest + 1389 Swift Testing, 0 failures; greedy parity integration test passed. generate ms/token HEAD -> after: 32 37.7 -> 34.8, 2k 39.0 -> 36.8, 8k 39.9 -> 37.3, 32k 41.7 -> 39.7; direct step 128 38.9 -> 36.7, 4k 40.4 -> 38.2, 16k 41.8 -> 39.9-40.5. Prefill not worse. Text changes at 8k/32k are near ties (gap 0.5 and 0.0 <= 4.0).
    - next: /review (task left in doing, not committed)
  timestamp: 2026-10-13T23:16:07.286170+00:00
- actor: claude-code
  id: 01m4vs3j9eayrzqfac22x4k7ka
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0 (changed files touched; only known warning "missing creator for mutated node"). xcrun xctest on all 5 bundles, exit 0 each: MLXLMTests 1389 tests passed (716 XCTest, 0 failures), MLXFoundationModelsTests 367 passed, MLXGuidedGenerationTests 70 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 5 passed (0 failures). 0 skipped. DeepSeekV4RoPEParityTests suite passed. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED, 0 warnings. Real-weights integration tests not run, as ordered.
    - next: review
  timestamp: 2026-10-13T23:21:03.534879+00:00
- actor: claude-code
  id: 01m4vscrpw9jv0zw7xp8sd9rhg
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (16a80b0): 2 findings (2 confirmed, 2 refuted, 14 attempted, 0 failed). Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:176 (completeness/invariant-propagation), Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift:293 (duplication/duplication).
    - acceptance criteria: each one has its evidence in a comment. One MLXFast.RoPE call for each site: "Implementation landed". Parity: the tolerances (3.2e-4 float32, 0.03125 bfloat16) and the measured gaps they come from are in "Implementation landed"; the rotary dimensions are not bit-exact because the fused kernel rounds one time. Unit tests: "test — green". Decode measurement before and after, GPU idle: "Measurements 2026-10-13".
    - next: implement the two findings in the 2026-10-13 18:21 Review Findings section. The task stays in review.
  timestamp: 2026-10-13T23:26:05.020544+00:00
- actor: claude-code
  id: 01m4vsdc6frejt7gkf7f2n4nz5
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — DeepSeekV4MathHelpers.swift, DeepSeekV4Attention.swift, DeepSeekV4Indexer.swift, DeepSeekV4Compressor.swift, DeepSeekV4.swift, DeepSeekV4RoPEParityTests.swift (new), and 4 test files
    - test: green — all 5 bundles exit 0, 0 failures; IntegrationTesting build-for-testing succeeded
    - commit: 16a80b0
    - review: findings — Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:176, Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift:293
  timestamp: 2026-10-13T23:26:24.975288+00:00
- actor: claude-code
  id: 01m4vstq7sw70kj66fnpd41805
  text: |-
    ### Review findings 2026-10-13 18:21 — work done (not committed)
    - Finding 1 (invariant-propagation): `TableKind` is now `Hashable`, and `tableKinds(configuration:layer:)` returns `Set<TableKind>` (the head widths are also a `Set<Int>`). Thus each kind is in the list one time by construction, and `key` names all three values, thus `Dictionary(uniqueKeysWithValues:)` cannot trap. When `index_head_dim == head_dim`, the attention and the indexer read the same table. The search of the DeepSeek-V4 files found no other dictionary or table built from these kinds.
    - RED before the fix: the new test `DeepSeekV4IndexerTests.anIndexerAsWideAsTheAttentionHeadsBuildsAndRunsAForward()` (layer 2, `index_head_dim` 8 == `head_dim` 8, 16 tokens, cache nil) stopped the process: `Fatal error: Duplicate values for key: 'width8-stride1-forward'`. Note: the forward kinds also had the duplicate key, not only the pooled kinds. GREEN after the fix: passed in 0.047 s.
    - The test file has a new shared helper `loadRepeatableWeights(into:seed:)`; `loadedIndexer(seed:)` calls it. `configuration(indexHeadWidth:)` takes the indexer width (default `indexHeadDim`), and `head_dim` is the named constant `attentionHeadDim`.
    - Finding 2 (duplication): new test support file `Tests/MLXLMTests/DeepSeekV4FlashYarn.swift` (`enum DeepSeekV4FlashYarn`) holds the Flash constants (`ropeDim`, `compressRopeTheta`, `factor`, `originalMaxPositionEmbeddings`, `betaFast`, `betaSlow`) and `inverseFrequency() -> MLXArray`. `DeepSeekV4RoPEParityTests.yarnInverseFrequency()` and `DeepSeekV4MathHelpersTests.deepseekV4InverseFrequency()` are deleted, with their copies of the constants. The third copy of the call, `yarnInvFreqMatchesPythonReference()` (literal 160_000, 65536, 16, 32, 1), now calls the shared helper too. `.asArray(Float.self)` stays at the MathHelpers call site.
    - Production behaviour on Flash does not change: widths 512 and 128 are different, thus the layer builds the same tables as before. No real-weights run.
  timestamp: 2026-10-13T23:33:42.265325+00:00
- actor: claude-code
  id: 01m4vsttbb1vr07m8995brtgxr
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLLM/Models/DeepSeekV4Attention.swift; Tests/MLXLMTests/DeepSeekV4FlashYarn.swift (new), DeepSeekV4IndexerTests.swift, DeepSeekV4MathHelpersTests.swift, DeepSeekV4RoPEParityTests.swift. `swift build --build-tests` exit 0 (only the known toolchain warning "missing creator for mutated node"); `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` exit 0: 716 XCTest 0 failures, 1390 Swift Testing passed in 111 suites (one new test). Both 2026-10-13 18:21 findings checked.
    - next: /test, then /review (task left in doing, not committed)
  timestamp: 2026-10-13T23:33:45.451987+00:00
- actor: claude-code
  id: 01m4vt3dpe0ecedng3rb2hrq5a
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched) — no new warning; only the known "missing creator for mutated node" (mlx-swift_Cmlx.bundle). xcrun xctest on all 5 bundles, exit 0. MLXLMTests: 716 XCTest, 0 failures; 1390 Swift Testing in 111 suites, passed. CXGrammarTests 7 ST. MLXFoundationModelsTests 367 ST. MLXGuidedGenerationTests 70 ST. MLXHuggingFaceMacrosTests 5 XCTest. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED, no new warning. 0 skipped.
    - next: review
  timestamp: 2026-10-13T23:38:27.406871+00:00
- actor: claude-code
  id: 01m4vt6tcy4q43acdxsjpekny9
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (7536f55): 1 finding, 1 confirmed, 0 refuted — Tests/MLXLMTests/DeepSeekV4IndexerTests.swift:140 (reuse/reuse).
    - prior items: both are fixed in the code. `tableKinds` gives a `Set<TableKind>`, and `key` holds width, stride and direction, thus `Dictionary(uniqueKeysWithValues:)` cannot get a duplicate key. `DeepSeekV4FlashYarn.inverseFrequency()` is the only `yarnInvFreq` call with the Flash YaRN constants in Tests/MLXLMTests.
    - next: move `loadRepeatableWeights(into:seed:)` to a shared test file and make the Cache and Compressor suites call it. The task stays in review.
  timestamp: 2026-10-13T23:40:18.718259+00:00
- actor: claude-code
  id: 01m4vt7c528a1tzn1t76zvw21d
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — DeepSeekV4Attention.swift (unique table kinds), DeepSeekV4FlashYarn.swift (new), DeepSeekV4IndexerTests.swift, DeepSeekV4MathHelpersTests.swift, DeepSeekV4RoPEParityTests.swift
    - test: green — MLXLMTests 716 XCTest + 1390 Swift Testing, 0 failures; IntegrationTesting build-for-testing succeeded
    - commit: 7536f55
    - review: findings — Tests/MLXLMTests/DeepSeekV4IndexerTests.swift:140 (prior 2 items verified fixed)
  timestamp: 2026-10-13T23:40:36.898227+00:00
- actor: claude-code
  id: 01m4vtf9ny5pwsz4k4b3h8bvvv
  text: |-
    ### Research 2026-10-13 (finding 18:38, reuse)
    - Five DeepSeek-V4 suites hold a copy of the weight filler: Indexer (`loadRepeatableWeights`), Cache (`checkpoint(for:seed:)`), Compressor (`checkpoint(for:seed:)`), Model (`loadedModel`), GateCheckpointLoad (`publishedCheckpoint`). AttentionTests, MoETests, HyperConnectionTests, CompiledDecodeTests and EncoderWiringTests fill by hand or save weights; they have no filler copy.
    - Common: seed of parameter n = start seed + n (n from 1), `MLXRandom.uniform(low:high:shape,key: MLXRandom.key(seed))`, float32, range -0.5 ... 0.5 from the constants of each suite.
    - Differences that change values: (1) key order. Indexer reads `flattened()` order; the others sort the full paths. `NestedDictionary.flattened()` sorts each level on its own and gives array items by index, thus `layers.10` comes after `layers.2`, and `a.b` / `a_c` can change places. The two orders are different. (2) Model puts a hash table (not random) in each `tid2eid` key, but the seed still moves on for that key. (3) Gate skips the routing tensors, and the seed does not move on for a skipped key. (4) Gate and one Compressor test use the weights without a load into the same module.
    - The file `DeepSeekV4SyntheticCheckpoint.swift` already exists and holds a `config.json` (no forward pass). Thus the filler goes into a new file `DeepSeekV4RepeatableWeights.swift`, with a `KeyOrder` enum, an `including` filter and a `replacing` closure.
  timestamp: 2026-10-13T23:44:56.510620+00:00
- actor: claude-code
  id: 01m4vtqkxnwc907t7cbtrpxa77
  text: |-
    ### Finding 18:38 (reuse) — work done (not committed)
    - New test support file `Tests/MLXLMTests/DeepSeekV4RepeatableWeights.swift` (`enum DeepSeekV4RepeatableWeights`). `checkpoint(for:seed:low:high:order:including:replacing:)` builds the weights; `load(into:seed:low:high:order:replacing:)` fills the module and loads it through `update(parameters:verify:[.all])`, then `eval`. The seed of parameter n is the start seed + n (n from 1), the value is `MLXRandom.uniform(low:high:shape,key:)` in float32, as in every copy before.
    - Parameters for the differences that change values: `order` (`KeyOrder.flattened` for the Indexer, `.sorted` for the others), `low` / `high` (each suite passes its own range constants, thus the weight range stays tied to the fixture range of the suite), `including` (Gate: skip the routing tensors, and a skipped key takes no seed), `replacing` (Model: a hash table in each `tid2eid` key, and the key still takes its seed).
    - Removed copies: `DeepSeekV4IndexerTests.loadRepeatableWeights`, `DeepSeekV4CacheTests.checkpoint(for:seed:)`, `DeepSeekV4CompressorTests.checkpoint(for:seed:)`, the loop in `DeepSeekV4ModelTests.loadedModel`, the loop in `DeepSeekV4GateCheckpointLoadTests.publishedCheckpoint`. All 9 call sites call the shared helper.
    - Not put into `DeepSeekV4SyntheticCheckpoint.swift`: that file already exists and holds only a `config.json` for suites that run no forward pass.
    - `git diff -U0 -- Tests` shows no change to a seed, a range, a tolerance or an expected value.
    - Tests: `swift build --build-tests` exit 0 (only the known warning "missing creator for mutated node"). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` exit 0: 716 XCTest, 0 failures; 1390 Swift Testing in 111 suites passed. The counts did not change.
  timestamp: 2026-10-13T23:49:29.141273+00:00
- actor: claude-code
  id: 01m4vtqpgkkn777ts0vcgfxg80
  text: |-
    ### implement — changed
    - evidence: 6 files — Tests/MLXLMTests/DeepSeekV4RepeatableWeights.swift (new), DeepSeekV4IndexerTests.swift, DeepSeekV4CacheTests.swift, DeepSeekV4CompressorTests.swift, DeepSeekV4ModelTests.swift, DeepSeekV4GateCheckpointLoadTests.swift. `swift build --build-tests` exit 0; `xcrun xctest MLXLMTests.xctest` exit 0: 716 XCTest 0 failures, 1390 Swift Testing passed (111 suites). The 2026-10-13 18:38 finding is checked.
    - next: /test, then /review (task left in doing, not committed)
  timestamp: 2026-10-13T23:49:31.795627+00:00
- actor: claude-code
  id: 01m4vv04kn7e4cc93d64yh7fz6
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched, rebuilt; only the known warning "missing creator for mutated node" for mlx-swift_Cmlx.bundle). xcrun xctest on all 5 bundles, all exit 0. MLXLMTests: 716 XCTest passed, 0 failures; 1390 Swift Testing tests in 111 suites passed. CXGrammarTests 7 passed. MLXFoundationModelsTests 367 passed. MLXGuidedGenerationTests 70 passed. MLXHuggingFaceMacrosTests 5 XCTest passed. 0 skipped, 0 new warnings.
    - next: review.
  timestamp: 2026-10-13T23:54:08.373376+00:00
- actor: claude-code
  id: 01m4vv2wgk9z781p6x1pkej3rt
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (3c6162a). 0 findings, 0 confirmed, 0 refuted. 6 files reviewed. 2 files not reviewed (.kanban task files; no validator matches them). All 3 prior findings are checked. A check of Tests/MLXLMTests/DeepSeekV4*Tests.swift found no other seeded fill-and-load copy. The Cache, Compressor, Indexer, Model and GateCheckpointLoad suites call DeepSeekV4RepeatableWeights.load or DeepSeekV4RepeatableWeights.checkpoint.
    - next: none. The task is in done.
  timestamp: 2026-10-13T23:55:38.387099+00:00
- actor: claude-code
  id: 01m4vv3c9mad9vagd7mgyhpw73
  text: |-
    ### finish iteration 3 — clean
    - implement: changed — DeepSeekV4RepeatableWeights.swift (new) and 5 DeepSeek-V4 test suites use it
    - test: green — MLXLMTests 716 XCTest + 1390 Swift Testing, 0 failures, same counts
    - commit: 3c6162a
    - review: clean — review sha HEAD~1..HEAD, 0 findings, 3/3 prior items checked
  timestamp: 2026-10-13T23:55:54.548918+00:00
position_column: done
position_ordinal: ffc180
title: 'DeepSeek-V4 speed: one MLXFast.rope call with a freqs table'
---
## What

Our port calculates cos/sin tables in each layer, then slices, stacks and concatenates the rotary and non-rotary parts of each head.

Our code:
- `DeepSeekV4Attention.swift:193-209, 411-438, 669-689`
- `DeepSeekV4MathHelpers.swift:192-230`
- `DeepSeekV4Indexer.swift:149`
- `DeepSeekV4Compressor.swift:200`

Blaizzy/mlx-vlm `deepseek_v4/language.py:258-339` gives an infinite period to the pairs that take no position. Then one `rope` call with a `freqs` table turns the full head, with no slice and no concatenation. The YaRN scaling (`rope_scaling` factor 16) and `compress_rope_theta` go into the table.

## Evidence

In mlx-lm PR 1192, the Apple (angeloskath) RoPE changes took decode on an M3 Ultra from 30.2 to 33.6 tok/s.

## Acceptance Criteria

- [x] Attention, the indexer and the compressor each turn positions with one `MLXFast.RoPE` call.
- [x] Parity tests show the same results as the current path, for both `rope_theta` and `compress_rope_theta` layers, and with YaRN scaling.
- [x] The existing DeepSeek-V4 unit tests pass.
- [x] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after.
#deepseek-v4 #performance

## Review Findings (2026-10-13 18:21)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4RH9AEVJP0MDG34XPHF8PWT.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4RH9AEVJP0MDG34XPHF8PWT.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:176` `completeness/invariant-propagation` — The period-table key is built from head width, stride and direction only. When a layer holds both an attention compressor and an indexer, and the two head widths are equal, `tableKinds` returns two kinds with the same key. `Dictionary(uniqueKeysWithValues:)` then traps at layer init. The fix that added the indexer width did not carry the uniqueness rule to the keys it builds. Deduplicate the kinds before building the dictionary, for example with `Dictionary(_:uniquingKeysWith:)` or by collecting the kinds into a `Set` first. Add a test with `indexHeadDim == headDim` on a compressor-and-indexer layer so the case is locked in.
- [x] `Tests/MLXLMTests/DeepSeekV4MathHelpersTests.swift:293` `duplication/duplication` — deepseekV4InverseFrequency() repeats the same yarnInvFreq call that DeepSeekV4RoPEParityTests.yarnInverseFrequency() makes. Both pass dim 64, base 160000, original max positions 65536, factor 16, beta_fast 32 and beta_slow 1. If one copy changes and the other does not, the two test suites check different tables. Keep one shared helper for the DeepSeek-V4-Flash YaRN inverse-frequency table, in one test support file. Have both suites call it and delete the second copy. Keep the return-type conversion at the call site.

## Review Findings (2026-10-13 18:38)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4RH9AEVJP0MDG34XPHF8PWT.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4RH9AEVJP0MDG34XPHF8PWT.md` — no validator matches this file

- [x] `Tests/MLXLMTests/DeepSeekV4IndexerTests.swift:140` `reuse/reuse` — The new private `loadRepeatableWeights(into:seed:)` repeats the fill-and-load logic of the `checkpoint` helpers in other DeepSeek-V4 test suites. Each suite keeps its own copy, so one fix to the weight filler must be made in several places. Move `loadRepeatableWeights` to a shared test file, such as `DeepSeekV4SyntheticCheckpoint.swift`, and have the Cache and Compressor suites call it. If their `checkpoint` helpers differ in a way that matters, parameterize the shared helper for that difference instead of keeping a parallel copy.
