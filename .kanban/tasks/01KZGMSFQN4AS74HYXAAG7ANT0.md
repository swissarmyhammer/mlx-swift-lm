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

- [ ] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` exists with `DeepseekV4RoPE` and `DeepseekV4Attention`.
- [ ] Forward pass on synthetic weights returns the correct shape for both a prefill (L>1) and a decode (L=1) step.
- [ ] `attn_sink` is on by default; a test with `use_attn_sink=false` produces different output, proving the sink actually participates.
- [ ] Output dims 0..<448 are unaffected by position while dims 448..<512 are — i.e. the inverse RoPE demonstrably ran.
- [ ] Grouped low-rank O projection produces `hidden_size=4096` output from `o_groups=8` grouping.
- [ ] The value returned by the cache update is the value passed to SDPA (guards the mlx-lm 1662 leak).
- [ ] `DeepseekV3.swift` unmodified.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4AttentionTests.swift` using small synthetic config + random weights (no download).
- [ ] Test: prefill shape `[1, 8, 4096]` in to `[1, 8, 4096]` out; decode shape `[1, 1, 4096]` in to `[1, 1, 4096]` out; output `allFinite`.
- [ ] Test: `use_attn_sink` true vs false yields materially different output (assert not-almost-equal).
- [ ] Test: per-layer theta — a layer with `compress_ratio == 0` and one with nonzero ratio produce different RoPE output for the same input.
- [ ] Test: cache growth — after N decode steps the cache offset is exactly N, and the arrays SDPA consumed are the arrays the cache returned (regression test for the mlx-lm 1662 leak).
- [ ] Run: `swift test --filter DeepseekV4AttentionTests` — all pass.

## Workflow
- Use `/tdd` — write the shape, sink-participation, and cache-identity tests first, then port.
#deepseek-v4