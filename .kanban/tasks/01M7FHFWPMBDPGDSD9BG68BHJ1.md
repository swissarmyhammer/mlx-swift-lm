---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7fqy7e5dfxhpg9vcaevwt73
  text: |-
    Research (implement):
    - oQ4e and oQ8e store `softmax_offset` (49) and `expert_bias` (47) in BF16; activations are BF16 (embedding/q_norm/scales BF16).
    - `MLXArray.asType` returns `self` when the dtype already matches (mlx-swift MLXArray.swift: `guard type != self.dtype else { return self }`), and MLX SDPA's own `astype(sinks, final_type)` is a no-op too. So `sinks.asType(queries.dtype)` adds NO kernel on these checkpoints; the card's "49 per token" for the sinks does not hold. It only costs a kernel when the sinks dtype differs from the activations (e.g. f32 sinks beside bf16 weights).
    - `expertBias.asType(.float32)` is a real bf16->f32 kernel per MoE layer: 47 per token.
    - `weightedExpertSum(routed.asType(.float32), weights)`: `weightedExpertSum` is a shapeless `compile`, and `weights` is float32, so passing `routed` as bf16 makes the promotion happen inside the compiled multiply (fused, same values: bf16->f32 is exact). That drops a standalone cast kernel per MoE layer: 47 per token.
    - Cannot be dropped without changing the reference math: `gate(x).asType(.float32)` before sigmoid, `.asType(x.dtype)` after the combine, `model(...).asType(.float32)` before lm_head (one per token). The first two go to the compile card ^pg2hsb8.
    Plan: sanitize stores expert_bias in float32 and the sinks in the dtype of the layer's q_norm weight; forward drops the bias cast and the routed cast; the sinks `asType(queries.dtype)` stays as a zero-cost guard (returns self when dtypes match).
  timestamp: 2026-11-15T13:56:52.293351+00:00
- actor: claude-code
  id: 01m7frf031ejshkw3ppd635ybe
  text: |-
    Implementation landed (Flint.swift, FlintTests.swift):
    - `FlintModel.sanitize` now calls a new private `storeInForwardDtypes`: `expert_bias` -> float32, `softmax_offset` -> the dtype of the layer's `q_norm.weight`.
    - `FlintSparseMoeBlock`: `scores + expertBias` (no explicit cast; a non-f32 bias still promotes to f32 in the add), and `weightedExpertSum(routed, weights)` (the f32 weights promote routed inside the compiled multiply).
    - `FlintAttentionLayer`: the `sinks.asType(queries.dtype)` stays as a guard. It returns `self` once the sinks load in the query dtype, and it only casts when parameters change dtype after load.
    - Two new tests: testSanitizeStoresTheExpertBiasInFloat32, testSanitizeStoresTheSinksInTheDtypeOfTheQueryNorm (both RED before the change, GREEN after).
    Measured decode-step graph (flintcli new `dot` mode, oQ4e, exportToDot of one decode step): 3221 -> 3127 primitives. AsType 191 -> 97, so 94 cast kernels per token are gone: 47 for the expert bias and 47 for routed->f32. `CompiledBroadcastBroadcastMultiply` became `CompiledAsTypeBroadcastBroadcastMultiply` (the same 47 kernels with the cast fused in). There were no sinks casts before the change (bf16 == bf16).
    Correctness:
    - flintcheck (oQ4e): MATCH x4 (256 greedy ids each), TOOL CALL get_weather {"city": "Paris"}. golden_base.json untouched; output at scratchpad/golden_g68bhj1.json.
    - oQ8e greedy golden vs golden8_fused.json: MATCH x4.
    - Float32 logits (MLX_ENABLE_TF32=0 FLINT_F32=1, parity_ids, 666 positions) vs fused32_*.safetensors: max|d| 0.0 on oQ4e and oQ8e.
    - oQ8e `chat 200 hello`: loads and answers sensibly (129 tok/s gen).
    flintbench (oQ4e, median of 3, two runs) vs the prior numbers after ^j4pqtwr (7.13/7.53/7.88 ms, 6202/6188/5763 tok/s):
    - ctx 2048: decode 6.87 / 6.88 ms/tok, prefill 6554 / 6505 tok/s
    - ctx 8192: decode 7.24 / 7.26 ms/tok, prefill 6384 / 6396 tok/s
    - ctx 16384: decode 7.61 / 7.59 ms/tok, prefill 5948 / 5913 tok/s
    That is about 0.26-0.29 ms/tok (~3.7%) faster decode.
  timestamp: 2026-11-15T14:06:01.825659+00:00
- actor: claude-code
  id: 01m7frp4mpr4hpzt3crr7nfxph
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLLM/Models/Flint.swift and Tests/MLXLMTests/FlintTests.swift (2 new tests). Per-token decode graph on oQ4e: 3221 -> 3127 primitives. AsType went 191 -> 97 (47 expert-bias casts and 47 routed->f32 casts removed); there were 0 sinks casts before the change. Tests: MLXLMTests 1092 XCTest (4 LFM2 env skips that were already there, 0 failures) + 2331 Swift Testing (3 known issues that were already there); Guided 76; FM 494; CXGrammar 7; Macros 11, all pass. swift-format lint -s is clean on both files. flintcheck: MATCH x4 + TOOL CALL get_weather Paris. oQ8e golden: MATCH x4. Float32 logits on oQ4e and oQ8e: max|d| 0.0. oQ8e chat works. flintbench decode 6.87/7.24/7.61 ms/tok (second run 6.88/7.26/7.59), down from 7.13/7.53/7.88; prefill 6554/6384/5948 tok/s.
    - next: review
  timestamp: 2026-11-15T14:09:55.862958+00:00
- actor: claude-code
  id: 01m7fs4zjs5apqz82yry1d09hp
  text: |-
    ### test — green
    - swift build --build-tests: complete, no compiler warnings or errors in the sources (one SwiftPM build-system line, "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS", comes from the swiftbuild product layout, not from the code)
    - MLXLMTests: XCTest 1092 run, 0 failures, 4 skipped (LFM2 env baseline); Swift Testing 2331 tests / 221 suites passed, 3 known issues (baseline)
    - MLXGuidedGenerationTests: 76 tests / 17 suites passed
    - MLXFoundationModelsTests: 494 tests / 66 suites passed
    - CXGrammarTests: 7 tests / 5 suites passed
    - MLXHuggingFaceMacrosTests: 11 XCTest tests, 0 failures
    - swift-format lint -s on Libraries/MLXLLM/Models/Flint.swift and Tests/MLXLMTests/FlintTests.swift: clean
    - flintcheck.sh: MATCH x4 (all 256 greedy ids identical on parity_ids, long_ids, prompt_87, prompt_706) and TOOL CALL: get_weather Paris
    - next: review
  timestamp: 2026-11-15T14:18:02.201834+00:00
- actor: claude-code
  id: 01m7fs5r0e2wjavqgphmcr3yed
  text: |-
    ### commit — changed
    - evidence: commit with subject "perf(flint): cast the expert bias and the sinks once at load, not on every forward (^g68bhj1)" on branch stable; files: Libraries/MLXLLM/Models/Flint.swift, Tests/MLXLMTests/FlintTests.swift, .kanban/ (the board change ships in the same commit, so the sha is read from git log by that subject)
    - next: review
  timestamp: 2026-11-15T14:18:27.214828+00:00
position_column: doing
position_ordinal: '80'
title: 'Flint: stop casting the attention sinks and the expert bias on every forward'
---
## Why
Two casts run on every forward pass and repeat the same work each time:
- `FlintAttention` passes `sinks.asType(queries.dtype)`, one kernel per layer (49 per token).
- `FlintSparseMoeBlock` computes `expertBias.asType(.float32)`, one per MoE layer (47 per token).

Decode is kernel-count bound (~7.7 ms/token against a ~2.4 ms bandwidth floor on oQ4e), so ~96 dispatches per token matter.

## What
- Cast once, when weights load or on first use, keyed on dtype. For example, store the bias in float32 in `sanitize`, and cache the sinks in the activation dtype.
- Look for other per-step casts in the decode path at the same time (`routed.asType(.float32)`, `out.asType(x.dtype)`). Fold them into the compiled combine of the decode-compile card if they can't be dropped.
- Keep the float32 routing math the reference uses.

## Acceptance
- Greedy token ids are identical on oQ4e and oQ8e.
- Float32 logit parity is unchanged.
- ms/token is better than a same-session baseline (flintcli `bench`).
- All five test bundles are green. #flint-perf