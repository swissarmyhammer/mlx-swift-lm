---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzpb3s51ctfzk9nm87a53yvn
  text: |-
    ### Two facts about `compress_ratios`, from task `^wkv5j6f`

    This card holds the rule for the RoPE theta. Two related facts, measured on `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`:

    - `compress_ratios` gives 0 to layer 0 and to layer 1. These two layers hold no compressor, thus `ropeTheta(forLayer:)` gives them the plain `rope_theta`. The published `quantization` block names compressor keys for the 41 layers 2 to 42 only, and indexer keys for the 21 even layers 2 to 42 only.
    - `compress_ratios` holds **44** entries while `num_hidden_layers` is **43**, and entry 43 is 0. `hasCompressor(layer:)` guards the bound, thus `ropeTheta(forLayer:)` is correct today. Any code that reads `compressRatios.count` as a layer count gets 44. Read `numHiddenLayers` for a layer count.

    The per-layer-theta test on this card must thus pick its compressed layer from 2 to 42, and its plain layer from 0 or 1.
  timestamp: 2026-08-10T17:21:49.217879+00:00
- actor: claude-code
  id: 01kzqd9hbbtqctwqtysky79kcy
  text: |
    ### Research before the port

    Sources read, all three at the SHAs the card names:

    - `osaurus-ai/vmlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift` @ `b166896353b9c95d773de993990c20a0b5ba6905` -- 1182 lines, `DeepseekV4RoPE` at line 45, `DeepseekV4Attention` at line 87. This is the attribution source of the new file.
    - `scouzi1966/mlx-swift-lm` same path @ `e1852869ce61ded0d23b76df3757e9b75c77c1f5` -- 2067 lines, the same two types plus a compressor, an indexer and a DSpark path.
    - `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py` -- `DeepseekV4RoPE` line 84, `V4Attention` line 470.

    **Adjudications where the references disagree. The Python decides each one.**

    1. **Compress ratio of a layer.** Both Swift copies invent a pattern when `compress_ratios` is empty: layer 0 and the last layer get 0, and the layers between alternate 4 and 128. The Python has no such pattern -- `ratios[layer_idx] if layer_idx < len(ratios) else 0`. The Python wins. This port reads `hasCompressor(layer:)` and `ropeTheta(forLayer:)` from `DeepseekV4Configuration`, which already give the Python answer, and no pattern is added.
    2. **`use_attn_sink`.** The Python passes `sinks=` to SDPA with no condition; both Swift copies read `config.useAttnSink`. The card requires the gate, and the default of that key is true, thus the two agree on the real checkpoint. This port keeps the gate.
    3. **Bias on the linear layers.** The Python reads `attention_bias`, which defaults to false and which `DeepSeek-V4-Flash-4bit-config.json` does not give. `DeepseekV4Configuration` has no such key. Every projection thus takes `bias: false`, as both Swift copies do.

    **Facts measured, not assumed.**

    - `MLXFast.scaledDotProductAttention` in mlx-swift already takes `sinks: MLXArray?`. mlx `fast.cpp` prepends the sink column after the mask and before the softmax, then slices column 0 off; the sink logit does not take the attention scale. This is the card's "prepend before softmax, drop after" word for word.
    - `QuantizedKVCache.update(keys:values:)` in `Libraries/MLXLMCommon/KVCache.swift` is a `fatalError` -- it demands `updateQuantized`. Thus a hand-rolled `cache.update` path crashes on a quantized KV cache, and `attentionWithCacheUpdate` is the only correct cache entry. That helper takes no sinks today.
    - `DeepSeek-V4-Flash-4bit-config.json` gives `rope_scaling = {type: yarn, factor: 16, original_max_position_embeddings: 65536, beta_fast: 32, beta_slow: 1}` and gives no `use_attn_sink`, thus the sink is on.
    - The prevailing pattern for a derived rotary table on a `Module` in this repository is a plain stored `MLXArray`: `YarnRoPE` in `Libraries/MLXLMCommon/RoPEUtils.swift` holds `private let _freqs: MLXArray`. The port follows it.

    **Scope.** The compressor and the indexer are their own cards (`^tty95f4`, `^r92pjcr`) and are not in this file.
  timestamp: 2026-08-11T03:19:09.419733+00:00
- actor: claude-code
  id: 01kzqefstvm8jpk7gnk4w3361f
  text: |
    ### The port landed. Two shared files also changed, and here is why.

    **New files**

    - `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` -- `DeepseekV4RoPE` and `DeepseekV4Attention`.
    - `Tests/MLXLMTests/DeepseekV4AttentionTests.swift` -- 11 tests.

    **Changed files**

    - `Libraries/MLXLMCommon/AttentionUtils.swift` -- `attentionWithCacheUpdate` takes `sinks: MLXArray? = nil` and hands it to both branches.
    - `Libraries/MLXLMCommon/KVCache.swift` -- `quantizedScaledDotProductAttention` takes the same argument, and a new private `softmaxWithSinks` prepends the sink column, softmaxes, and drops column 0.
    - `Tests/MLXLMTests/KVCacheTests.swift` -- one test for the quantized sink path.

    The card asks the block to read `attentionWithCacheUpdate` and allows a custom path "if a custom path is unavoidable because of the sink column". A custom path is worse than unavoidable here: `QuantizedKVCache.update(keys:values:)` is a `fatalError`, thus a hand-rolled `cache.update` crashes the moment a reader turns a quantized KV cache on. Adding the argument to the shared helper keeps the cache handling in one place and keeps the quantized branch correct. Both new arguments default to `nil`, thus no other model changes behavior.

    **How the numbers were made.** A NumPy transcription of `DeepseekV4RoPE.__init__`, `DeepseekV4RoPE.__call__`, `V4Attention._grouped_output_projection` and `V4Attention.__call__`, plus the sink path of `mlx/fast.cpp`, run in float64. The weights and the inputs are not random -- both sides fill an array with `(((i + seed) * 37) % 17 - 8) / 8`, which lands on multiples of an eighth and is exact in float32 and float64 alike. Six fixtures: the plain inverse-frequency table, the YaRN table, prefill with the sink, prefill without it, prefill on a compressed layer, one decode step at offset 2, and the grouped projection.

    **Mutation proof. Every mutation was reverted and `git status` is clean.**

    | Mutation | Result |
    |---|---|
    | A. `wo_a` weight reshaped `(rank, groups, features)` in place of `(groups, rank, features)` | the einsum refuses the shape and the suite dies |
    | B. inverse rotation on the output changed to a forward one | 4 tests fail |
    | C. `sinks:` forced to `nil` | 4 tests fail |
    | D. `ropeTheta(forLayer:)` changed to the plain `ropeTheta` | 2 tests fail |
    | E. the return of `cache.update` dropped, the block's own tensor fed to SDPA -- the mlx-lm 1662 shape | `attentionReadsTheArraysTheCacheReturned` fails, and nothing else |
    | F. `scale` changed from `1 / sqrt(headDim)` to `1 / headDim` | 4 tests fail |
    | G. the output transpose changed to the identity `[0, 1, 2, 3]` -- a transposed axis with valid shapes | 3 tests fail on exact values |
    | H. the sink column dropped from `softmaxWithSinks` | `quantizedAttentionReadsTheLearnedSink` fails |

    G answers the warning from `^p34crp6`: a transposed axis that keeps every shape valid is caught here, because the tests read exact values rather than structure.

    **Two findings on the Swift references, both against the Python.**

    1. The inverse-frequency table must not be a module parameter. The Python hides it and says why: "This is derived from config, not a checkpoint parameter". Both Swift copies hold it as a plain `invFreq`, which puts `attn.rope.invFreq` into `parameters()` and thus into any `update(parameters:verify: [.all])`, although no checkpoint gives that key. This port names it `_inverseFrequency`; mlx-swift's `filterValidParameters` drops a name that starts with an underscore, the same way `YarnRoPE._freqs` does in this repository. `parameterKeysMatchTheCheckpoint` locks the key set down.
    2. The quantized grouped projection of the osaurus file calls `quantizedMatmul` with no `mode:`, thus it reads the affine mode for every checkpoint. The Python passes `mode=self.wo_a.mode`. This port passes the mode of the layer.

    **Scope statement, so that nothing looks silently skipped.** The dispatching note asks for the checkpoint paths `model.layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}` **and** `model.layers.N.attn.compressor.{wgate,wkv}`. The first set is delivered and tested. The second set needs the `DeepseekV4Compressor` module, which is card `^tty95f4`, and the indexer is `^r92pjcr`. Neither is in the Acceptance Criteria of this card, and this card's description names them as separate work. `ropeTheta(forLayer:)` already gives a compressed layer its own theta with no compressor module present, which is what a layer 2 to 42 needs today.
  timestamp: 2026-08-11T03:40:03.291390+00:00
- actor: claude-code
  id: 01kzqeg0x6hdpyx5z0vc4qpsvx
  text: |
    ### implement — changed
    - evidence: 5 files — new `Libraries/MLXLLM/Models/DeepseekV4Attention.swift`, new `Tests/MLXLMTests/DeepseekV4AttentionTests.swift`, changed `Libraries/MLXLMCommon/AttentionUtils.swift`, `Libraries/MLXLMCommon/KVCache.swift`, `Tests/MLXLMTests/KVCacheTests.swift`. `swift build` clean, no warning in any touched file. `swift test --filter DeepseekV4AttentionTests` 11 of 11 pass. Full `swift test` 764 tests pass across 94 suites, zero failures; `Gemma4ChunkedPrefillTests` passed, thus the known flake did not appear. Eight mutations A through H each turned the suite red, every one reverted, `git status` shows only the five files above. `Libraries/MLXLLM/Models/DeepseekV3.swift` untouched.
    - next: `/review`
  timestamp: 2026-08-11T03:40:10.534445+00:00
- actor: claude-code
  id: 01kzqemme7a80nj04xxbzffn52
  text: |-
    ### commit — changed
    - evidence: 54e4770 feat(mlx-lm): add DeepSeek-V4 attention port
    - evidence: dfc930a chore(kanban): record DeepSeek-V4 attention port result
    - next: none
  timestamp: 2026-08-11T03:42:41.607706+00:00
- actor: claude-code
  id: 01kzqfyc6d7hn2pbstkc8a42qp
  text: |
    ### review — findings

    - evidence: 112 findings, 127 confirmed, 3 refuted, 9 files attempted, 0 failed. Scope `54e4770~1..54e4770`. 4 findings in the new file `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` at lines 57, 249, 392, 396. The other 108 findings are in `Libraries/MLXLMCommon/KVCache.swift`, in lines 11 to 2148. The commit changed only lines 1948 to 2062 of that 2173-line file, thus the findings outside that band name conditions that were present before this commit.
    - next: correct the findings, then run `/review` again.

    ### Independent checks of the claims on this card

    **1. The shared files do not change any other model.** `softmaxWithSinks` starts with `guard let sinks else { return softmax(scores, axis: -1) }`. That is the same call, the same axis, and no dtype or shape work. The nil path is thus identical to the line it replaced. Both new arguments sit last in their signature and hold a default. Swift matches an argument by its label, thus no call site can shift. 57 files call `attentionWithCacheUpdate` and 10 files call `quantizedScaledDotProductAttention`. Not one of them passes `sinks`. The full `swift test` run passed 764 tests in 94 suites with zero failures.

    **2. The numeric parity is genuine, not circular.** The reference Python was read from `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`. All seven fixtures were made again in float64 NumPy from that reference alone. The largest gap is 4.9e-11, which is the rounding of the tenth printed decimal. The grouped projection agrees to 0.0e+00 on all 48 values. No fixture value round-trips through float32, thus no value came from a Swift run. A counterfactual that gives the sink logit the attention scale moves the answer by 2.2e-1, which is 2200 times the test tolerance. The sink detail is correct: `quantizedScaledDotProductAttention` scales the queries at `queries * scale` before the scores, and the sink column enters with `sinks.asType(scores.dtype)` and no multiply. Column 0 leaves after the softmax. This matches `fast.cpp`.

    **3. Mutations E and G were run again.** G, the output transpose changed to the identity `[0, 1, 2, 3]`: 3 tests fail — `prefillMatchesThePythonReference`, `attentionSinkChangesTheOutput`, `compressedLayerReadsTheCompressRopeTheta`. This matches the record. E, the return of the cache update dropped: 2 tests fail — `attentionReadsTheArraysTheCacheReturned` **and** `decodeStepMatchesThePythonReference`. The record says only the first one fails. The guard is stronger than the record states, not weaker. Both mutations were reverted and `git status` shows no source file changed.

    **4a. `_inverseFrequency`.** `Module.parameterIsValid` is `!key.hasPrefix("_")`. The field is a plain `private let _inverseFrequency: MLXArray`, thus the mirror label is the key and the filter drops it. `parameterKeysMatchTheCheckpoint` asserts set equality on 8 keys, thus a leaked `rope.*` key fails the test. One correction: no test runs `update(parameters:verify: [.all])`. The only `update` call passes `verify: []`. The key-set assertion is equal evidence, but the `[.all]` claim is inferred and not run.

    **4b. `mode:` on the grouped projection.** The Python does pass `mode=self.wo_a.mode`. `quantizedMM` in mlx-swift defaults `mode` to `.affine`, thus an omission reads affine. The change is right. One correction to the motive: the `quantization` block of `DeepSeek-V4-Flash-4bit-config.json` holds 641 entries, 512 affine and 129 mxfp4, and every mxfp4 entry is an FFN expert tensor. The count of mxfp4 attention tensors is 0. Every `attn.*` entry reads `"mode": "affine"`. An omission would thus give the same numbers on this checkpoint. The file header states the motive too strongly.

    **5. Checkpoint key paths.** Layer 0 gives exactly `wq_a, wq_b, wkv, wo_a, wo_b`. The counts are 43 each for the 43 layers. Layer 2 adds `compressor.wgate`, `compressor.wkv`, and six `indexer.*` keys; `compressor.*` = 41 and `indexer.*` = 21. The five linear names match one for one, with no extra key in either direction. `attention_bias` is false, thus no `.bias` key is expected. Caveat: the `quantization` block lists quantized tensors only. `q_norm.weight`, `kv_norm.weight` and `attn_sink` are norms and a logit, thus they are never quantized and this file cannot confirm them. Three of the eight asserted keys need the real `model.safetensors.index.json`, which is not in this repository.

    **6. The scope statement is honest.** No Acceptance Criterion and no Test line on this card names a compressor or an indexer. `^tty95f4` and `^r92pjcr` exist, are in `todo`, and own that work. The file header states the boundary. This is not a silent narrowing.

    **7. Every acceptance box and every test box is met and is now checked.** The evidence, in order: the file holds both types; `groupedOutputProjectionReachesTheFullHiddenSize` runs prefill `[1, 8, 4096]` and decode `[1, 1, 4096]` and asserts `isFinite`; `Default.useAttnSink` is `true` in `DeepseekV4Configuration.swift` and `attentionSinkChangesTheOutput` asserts a gap above the tolerance; `rotationReachesOnlyTheTrailingHeadDimensions` asserts dims 0 to 447 do not move and dims 448 to 511 do move; the wide test reaches `hidden_size` 4096 from 8 groups; `attentionReadsTheArraysTheCacheReturned` reads a `GainKVCache` that scales its values, thus a block that fed its own tensor gives the unscaled answer; `git show --stat 54e4770 -- Libraries/MLXLLM/Models/DeepseekV3.swift` is empty; `swift test --filter DeepseekV4AttentionTests` passes 11 of 11.

    **One condition found outside this commit.** `Libraries/MLXLMCommon/KVCache.swift:2038, 2042, 2051` fill a masked position with `MLXArray(Float.leastNormalMagnitude)`. That value is +1.175e-38, the smallest positive normal, and not a large negative. mlx uses `finfo(dtype).min`, which is -3.4e38. A masked key thus gets a score near 0 and keeps real softmax weight in `quantizedScaledDotProductAttention`. `git log -S` puts this in commit `37a8073`, thus it is older than `54e4770` and it is not part of this port. It needs its own card.
  timestamp: 2026-08-11T04:05:29.421286+00:00
- actor: claude-code
  id: 01kzqg09fk13ry5w1rz3zv6tj3
  text: |
    ### finish iteration 1 — findings
    - implement: changed — DeepseekV4Attention.swift (new), DeepseekV4AttentionTests.swift (new, 11 tests), AttentionUtils.swift, KVCache.swift, KVCacheTests.swift
    - test: green — `swift build` clean; filter 11/11; full `swift test` 764 tests in 94 suites, 0 failures
    - commit: 54e4770 feat(mlx-lm): add DeepSeek-V4 attention port; dfc930a and ae60919 (kanban chores)
    - review: findings — 112 findings. 9 files attempted.
    - VERIFIED GOOD: the shared-file change is safe. `softmaxWithSinks` opens with a guard that gives the same call for a nil sink. 57 files call `attentionWithCacheUpdate` and 10 call `quantizedScaledDotProductAttention`; none passes `sinks`. Numeric parity is genuine: the reviewer rebuilt all 7 fixtures in float64 NumPy from the Python alone, and the largest gap is 4.9e-11. The grouped projection agrees to 0.0e+00 on all 48 values. The sink is not scaled, which is correct; a counterfactual that scales it moves the answer by 2.2e-1, which is 2200 times the tolerance. Mutation G gives 3 failures as recorded. Mutation E gives 2 failures, thus the guard is stronger than the record claims.
    - SCOPE OF THE FINDINGS: only 4 of the 112 sit in the new file. The other 108 sit in `KVCache.swift` lines 11 to 2148, while this commit touched lines 1948 to 2062 of that 2173-line file. That is a sweep of code the commit did not write.
    - SEPARATE BUG FOUND, now task ^yjb4358: `KVCache.swift:2038`, `:2042` and `:2051` fill a masked position with `Float.leastNormalMagnitude`, which is +1.175e-38 and not a large negative number. A masked key thus keeps real weight after the softmax in the quantized path. `git log -S` puts it in commit `37a8073`, which is older than this port.
    - next: implement — fix the 4 findings in the new file. Do not sweep `KVCache.swift`.
  timestamp: 2026-08-11T04:06:32.179906+00:00
- actor: claude-code
  id: 01kzqgq559rxf6bv15q7tnjp0s
  text: |
    ### Three corrections to the record on this card

    **1. Mutation E fails 2 tests, not 1.** The earlier comment says that dropping the return of `cache.update` and feeding the block's own tensor to SDPA -- the `ml-explore/mlx-lm` 1662 shape -- fails `attentionReadsTheArraysTheCacheReturned` and nothing else. It fails **two** tests: `attentionReadsTheArraysTheCacheReturned` **and** `decodeStepMatchesThePythonReference`. The guard is stronger than the record claimed.

    **2. The motive for `mode:` on `quantizedMatmul` was too strong.** The old file header said that an omission "reads the affine mode for every checkpoint", which reads as a defect on the real checkpoint. The measured fact: the `quantization` block of `DeepSeek-V4-Flash-4bit-config.json` holds 641 entries, 512 affine and 129 mxfp4, and every mxfp4 entry is an FFN expert tensor. The count of mxfp4 `attn.*` tensors is **0**, thus an omission gives the same numbers on this checkpoint. The change is still right for two reasons: the Python passes `mode=self.wo_a.mode`, and an attention tensor in another mode would give the wrong numbers. The file header now states that reason.

    **3. A known limit of `parameterKeysMatchTheCheckpoint`.** The test asserts set equality on 8 parameter keys. Five of them -- `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` -- are confirmed against the `quantization` block of the config in `Tests/MLXLMTests/Resources/`. The other three -- `q_norm.weight`, `kv_norm.weight` and `attn_sink` -- **cannot** be confirmed from that block, because the block lists quantized tensors only, and a norm weight and a logit are never quantized. To confirm those three needs the real `model.safetensors.index.json`, which is not in this repository. The test still guards the key set of the module against drift; it is not full confirmation against the checkpoint.
  timestamp: 2026-08-11T04:19:01.417405+00:00
- actor: claude-code
  id: 01kzqgqmg8580f2gs1jx1rwng4
  text: |
    ### Review round 1 -- what the fix was, and what it did not touch

    **The 4 findings in the new file are closed.**

    - Line 57, `dim` is assignOnlyProperty. The stored property is gone. `DeepseekV4RoPE.init` still takes `dim:` and hands it to `DeepseekV4Math.yarnInvFreq`; only the stored copy was dead. A grep of `Libraries` and `Tests` found no reader.
    - Lines 249, 392 and 396, magic numbers. Each of the three axis permutations was a bare `[0, 2, 1, 3]`-style literal. Two named tables now carry the axis positions: `BatchMajorAxis` for a `(batch, tokens, heads, width)` tensor, and `GroupMajorAxis` for the `(groups, batch, tokens, width)` layout that one batched quantized matrix multiply reads. `headMajor`, `groupMajor` and `groupMinor` read those names. `swiftlint` was run with the rule the validator runs -- `only_rules: [no_magic_numbers]`, `allowed_numbers: [0, 1, -1, 100]` -- and it reported the same 6 violations on those 3 lines before the change and reports 0 after it. A `static let` inside the two tables is exempt from the rule, thus the names do not move the problem.

    **The 108 findings in `Libraries/MLXLMCommon/KVCache.swift` are marked out of scope on the card, one by one, each with the reason.** They are not silently skipped. They name lines 11 to 2148 of a 2173-line file while commit `54e4770` touched lines 1948 to 2062 of it. `^yjb4358` holds the one true bug the review found in that file; it is not fixed here.

    **Mutation G was run again after the change**, to prove the named-constant refactor did not soften the tests. The output transpose was set to the identity `[0, 1, 2, 3]` -- a permutation that keeps every shape valid. Three tests failed on exact values: `prefillMatchesThePythonReference`, `attentionSinkChangesTheOutput` and `compressedLayerReadsTheCompressRopeTheta`. That is the same 3 the record names. The mutation was reverted; `git status` shows one changed source file.

    **Verification.** `swift build` clean. `swift test --filter DeepseekV4AttentionTests` 11 of 11 pass. Full `swift test` 764 tests in 94 suites, zero failures. `swift format` was run on the one touched file only, with the project `.swift-format`; no tree-wide format was run.
  timestamp: 2026-08-11T04:19:17.128498+00:00
- actor: claude-code
  id: 01kzqgqskkvyj9qba0r06ksz1b
  text: |
    ### implement — changed
    - evidence: 1 file — `Libraries/MLXLLM/Models/DeepseekV4Attention.swift`. 4 findings closed (lines 57, 249, 392, 396); 108 `KVCache.swift` findings marked out of scope on the card with the reason. `swift build` clean. `swift test --filter DeepseekV4AttentionTests` 11 of 11 pass. Full `swift test` 764 tests in 94 suites, 0 failures. `swiftlint no_magic_numbers` 6 violations before, 0 after. Mutation G (output transpose to the identity) fails the same 3 tests, reverted, tree clean.
    - next: `/review`
  timestamp: 2026-08-11T04:19:22.355049+00:00
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
position_column: doing
position_ordinal: '80'
title: 'Port DeepseekV4 attention: partial RoPE, attn_sink, inverse-RoPE output, grouped low-rank O'
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` holding `DeepseekV4RoPE` and `DeepseekV4Attention`. This is the first half of the model forward; the mHC, MoE, and decoder-layer pieces are separate tasks.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4RoPE` at line 91, `DeepseekV4Attention` at line 133 (the file is 2509 lines total; take only these two types). Split into its own file rather than one 2500-line file, matching how this repo keeps model files navigable.

Shape facts from the real config: `head_dim=512`, 64 query heads, `num_key_value_heads=1` (one latent KV head broadcast to all 64 Q heads via GQA), RoPE applied to only the trailing `qk_rope_head_dim=64` dims, q/o LoRA rank 1024.

Features to implement, all NEW vs DeepseekV3:

1. **Per-layer RoPE theta** — `DeepseekV4RoPE` selects `rope_theta=10000` when `compress_ratio == 0` and `compress_rope_theta=160000` (with YaRN scaling) otherwise. Use `ropeTheta(forLayer:)` and `yarnInvFreq` from the config/math-helper tasks.
2. **Learned per-head `attn_sink`** — prepend a learned sink logit column before softmax, drop it after. `use_attn_sink` defaults **ON**. Flagged Bug 3 in the gap tracker.
3. **Inverse RoPE on the attention output** — rotate the trailing 64 dims backward via `conj(freqs_cis)` to strip positional info before the residual add-back. Uses the inverse partial RoPE from the math-helpers task.
4. **Grouped low-rank O projection** — `o_groups=8`: reshape O to `(B, L, 8, 4096)`, then `wo_a` einsum (`bsgd,grd -> bsgr`) followed by `wo_b(flatten(...))`.

**Critical — Metal buffer leak.** `ml-explore/mlx-lm` issue 1662: a model that discards the value returned by the cache's `updateAndFetch` leaks one Metal buffer per layer per generated token, and `deepseek_v4` deterministically crashes at ~11.5k generated tokens. Use this repo's existing `attentionWithCacheUpdate` helper (`Libraries/MLXLMCommon/AttentionUtils.swift:37`), which already handles the cache update correctly, rather than hand-rolling the cache interaction. If a custom path is unavoidable because of the sink column, the returned keys/values MUST be the ones fed to SDPA.

Note: `MLXFast` is already reachable from these targets (`AttentionUtils.swift:46`), so no `Package.swift` change is needed. Do not modify `DeepseekV3.swift`.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~91-756 (MIT; header attributes Osaurus AI).
- Bug numbering and feature list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`.
- Leak landmine: `ml-explore/mlx-lm` issue 1662.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` exists with `DeepseekV4RoPE` and `DeepseekV4Attention`.
- [x] Forward pass on synthetic weights returns the correct shape for both a prefill (L>1) and a decode (L=1) step.
- [x] `attn_sink` is on by default; a test with `use_attn_sink=false` produces different output, proving the sink actually participates.
- [x] Output dims 0..<448 are unaffected by position while dims 448..<512 are — i.e. the inverse RoPE demonstrably ran.
- [x] Grouped low-rank O projection produces `hidden_size=4096` output from `o_groups=8` grouping.
- [x] The value returned by the cache update is the value passed to SDPA (guards the mlx-lm 1662 leak).
- [x] `DeepseekV3.swift` unmodified.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4AttentionTests.swift` using small synthetic config + random weights (no download).
- [x] Test: prefill shape `[1, 8, 4096]` in to `[1, 8, 4096]` out; decode shape `[1, 1, 4096]` in to `[1, 1, 4096]` out; output `allFinite`.
- [x] Test: `use_attn_sink` true vs false yields materially different output (assert not-almost-equal).
- [x] Test: per-layer theta — a layer with `compress_ratio == 0` and one with nonzero ratio produce different RoPE output for the same input.
- [x] Test: cache growth — after N decode steps the cache offset is exactly N, and the arrays SDPA consumed are the arrays the cache returned (regression test for the mlx-lm 1662 leak).
- [x] Run: `swift test --filter DeepseekV4AttentionTests` — all pass.

## Workflow
- Use `/tdd` — write the shape, sink-participation, and cache-identity tests first, then port.
#deepseek-v4

## Review Findings (2026-08-10 22:44)

### Fixed — `Libraries/MLXLLM/Models/DeepseekV4Attention.swift`

- [x] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift:57` — var.instance `dim` is assignOnlyProperty. Fixed: the stored `dim` is gone from `DeepseekV4RoPE`. The init parameter still builds the inverse-frequency table. No reader of the property was present in the library or in the tests.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift:249` — Magic numbers should be replaced by named constants. Fixed: `headMajor` reads the named axis positions of the new `BatchMajorAxis`.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift:392` — Magic numbers should be replaced by named constants. Fixed: `groupMajor` reads the named axis positions of `BatchMajorAxis`.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift:396` — Magic numbers should be replaced by named constants. Fixed: `groupMinor` reads the named axis positions of the new `GroupMajorAxis`.

`swiftlint` with the `no_magic_numbers` rule and `allowed_numbers [0, 1, -1, 100]` gave 6 violations on the 3 lines above before the change, and gives 0 after it.

### Out of scope for this card — `Libraries/MLXLMCommon/KVCache.swift` (108 findings)

Each of the 108 findings below is out of scope for this card. None of them is silently skipped. The reason: they name lines 11 to 2148 of a 2173-line file, while commit `54e4770` touched lines 1948 to 2062 of it. They name conditions that were present before this card, thus a fix here would turn a model-port card into a rewrite of the shared cache code.

One true bug that the review found in that file has its own card, `^yjb4358`: a masked position takes `Float.leastNormalMagnitude`, which is +1.175e-38 and not a large negative number, thus a masked key keeps softmax weight in the quantized path. It is not fixed here.

- [x] `Libraries/MLXLMCommon/KVCache.swift:11` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:12` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:108` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:112` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:114` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:116` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:119` — Public function `withPreparedCache` lacks documentation. This is a generic public utility that manages cache lifecycle preparation and needs a doc comment explaining its purpose and usage. Add a doc comment above line 119 explaining what the function does, its parameters, and return value. Example: `/// Prepare a cache for batched sequence generation and run a body, then finalize.`. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:119` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:242` — Public function `createCausalMask` lacks documentation. This is a foundational utility for attention masking and needs explanation of its purpose and parameters. Add a doc comment above line 242 explaining the causal mask creation, what each parameter does (especially `offset`, `windowSize`, and `lengths`), and what the returned MLXArray represents. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:242` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:312` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:371` — Public function `createSSMMask` lacks documentation. This is a model-specific mask creation function that needs explanation. Add a doc comment above line 371 explaining what SSM stands for (Selective State Space Models), the purpose of this mask, and the meaning of each parameter. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:371` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:397` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:405` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:406` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:408` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:419` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:420` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:427` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:441` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:451` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:456` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:480` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:481` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:485` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:560` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:565` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:573` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:587` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:597` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:598` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:606` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:607` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:608` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:613` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:623` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:624` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:633` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:663` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:674` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:684` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:699` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:703` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:704` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:705` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:814` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:840` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:893` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:922` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:923` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:924` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:946` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1023` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1039` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1042` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1043` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1044` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1045` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1057` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1062` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1063` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1131` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1134` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1142` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1145` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1146` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1148` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1159` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1160` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1167` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1199` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1378` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1388` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1389` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1409` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1545` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1552` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1555` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1575` — var.instance `message` is assignOnlyProperty. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1712` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1718` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1755` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1809` — Function has cognitive complexity of 17, exceeding the gate threshold of 15, with 4-level nesting depth and nested loops (2 deep). Multiple for loops with conditionals and nested if statements make control flow difficult to follow. Refactor to reduce nesting: extract inner loop logic into a helper function, or split the array-building phase from the dictionary-building phase into separate functions. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1815` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1849` — Function has cognitive complexity of 26, significantly exceeding the gate threshold of 15, with 4-level nesting depth, nested loops (2 deep), and an if/else-if chain with 2 boolean operators per condition. The combination of multiple branch types, loops within conditionals, and complex boolean logic makes this function hard to understand and maintain. Break into separate functions: one to parse cache info (with the first if branch and its while loops), one for user metadata, and one for cache classes. Or use a data structure to eliminate some of the nested loops. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1857` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1859` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1870` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1874` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1921` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1987` — public declarations should be documented. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1999` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2000` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2012` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2013` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2014` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2017` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2018` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2019` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2033` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2037` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2086` — A switch statement over known scheme string constants should be expressed as a data table (dictionary). Each arm differs only in the returned constant tuple, making this a table written as control flow rather than data. Replace the switch with a dictionary: `let schemes: [String: (bits: Int, groupSize: Int)] = ["affine4": (4, 64), "affine8": (8, 64)]; return schemes[scheme]`. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2087` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2088` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2146` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2147` — Magic numbers should be replaced by named constants. Out of scope, see above.
- [x] `Libraries/MLXLMCommon/KVCache.swift:2148` — Magic numbers should be replaced by named constants. Out of scope, see above.