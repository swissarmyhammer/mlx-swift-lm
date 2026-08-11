---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp8t1z0sh2w4tnm7y3t6ddd
  text: |
    ## Note carried from the math-helpers task `p34crp6` (review, 2026-08-10)

    Two points this card must know before it wires in `clampedSwiGLU` and `reduceRoutedExpertsFP32`.

    ### 1. The SwiGLU clamp is asymmetric. Do not write "±10" into a test.

    The user decided this on `p34crp6` on 2026-08-10. The reference wins:

    ```swift
    gate = minimum(gate, limit)                // the high side only
    up   = clip(up, min: -limit, max: limit)   // both sides
    ```

    The Python `_swiglu_limited`, the osaurus Swift copy and the scouzi Swift copy all agree. Reason the user gave: `silu()` falls to almost zero at a large negative gate, thus the low side cannot run away.

    **This card carries the wording the user corrected.** "What" item 3 says "gate/up clamped to ±10", the acceptance criterion says "Expert SwiGLU clamps at ±10", and the test line says "expert output saturates for inputs beyond ±50 (clamp is active)". Read as written, a test that asserts a `gate` of -50 saturates will fail against the correct code. Correct these three lines to the asymmetric rule before you write the test: an `up` below -10 or above +10 saturates, a `gate` above +10 saturates, and a `gate` below -10 does not saturate. At `gate = -50` the reference output is `9.643749e-20`, not zero.

    ### 2. `clampedSwiGLU` computes in float32 and casts back to the input dtype.

    `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` casts `gate` and `up` to float32, clamps and applies silu there, then returns the input dtype. The Python computes in the input dtype throughout. This is not an error -- the float32 path is more accurate, and it is the same pattern item 7 of `p34crp6` asks for in `reduceRoutedExpertsFP32`.

    It matters here because every test on `p34crp6` runs float32, where the round trip changes nothing. This card is the first place bfloat16 activations reach the function. **When you build the MoE parity fixtures, state whether the expected numbers come from a Python run in bfloat16 or in float32.** A bfloat16 fixture generated from the Python will differ from this Swift by about one bfloat16 unit in the last place. Choose the tolerance with that in mind rather than discovering it as a mystery failure.
  timestamp: 2026-08-10T16:41:33.408327+00:00
- actor: claude-code
  id: 01kzp8w24mz5g6b9pq56m4z2cj
  text: |
    ## Correction from task ^p34crp6 — read before you start

    ### 1. This card has the wrong SwiGLU clamp rule in three places

    This card says "gate/up clamped to plus or minus 10". That is wrong, and the
    user decided against it on 2026-08-10 while ^p34crp6 was in review.

    The correct rule, which `DeepseekV4Math.clampedSwiGLU` already has:

    ```swift
    gate = minimum(gate, limit)                 // the high side only
    up   = clip(up, min: -limit, max: limit)    // both sides
    ```

    `silu()` falls to almost zero at a large negative gate, thus the low side of the
    gate cannot run away. Only `up` multiplies, thus only `up` needs both sides.
    This agrees with the Python `_swiglu_limited` and with both Swift reference
    copies.

    The planned test on this card, "expert output saturates for inputs beyond plus
    or minus 50", FAILS against correct code as written. A gate of -50 gives
    9.64e-20 and a gate of -10 gives 4.54e-3; the two are not the same, and that
    difference is correct. Correct this card's wording before you write the test.

    ### 2. Say which dtype your parity fixtures use

    `clampedSwiGLU` casts to float32, does the arithmetic, and casts back to the
    dtype of `gate`. The Python computes in the input dtype. The two agree in
    float32, thus every ^p34crp6 test passes.

    This card is the first to send bfloat16 activations into that function. A
    fixture from a Python run in bfloat16 differs from one in float32 by about one
    unit in the last place. Write in the test which one you used.

    ### 3. Reuse, do not copy

    `DeepseekV4Math` already holds `clampedSwiGLU`, `sqrtSoftplus`, and
    `reduceRoutedExpertsFP32`, and all three have parity tests. Call them.
  timestamp: 2026-08-10T16:42:39.124357+00:00
- actor: claude-code
  id: 01kzr888xk3txsgfskqvntm3m6
  text: |
    ## Research, and the card text correction

    ### The card now states the asymmetric clamp

    I corrected the three lines the two comments above name -- "What" item 3, the
    acceptance criterion, and the test line. The card now says: `up` stays inside
    `-limit` through `+limit`, `gate` stays below `+limit` only, and a large
    negative gate is NOT clamped.

    The NumPy transcription of the Python reference gives, at a limit of 10:

    | gate | up | output |
    |---|---|---|
    | +50 | 1 | 9.999546021313 |
    | +10 | 1 | 9.999546021313 |
    | -50 | 1 | -9.643749239820e-21 |
    | -10 | 1 | -4.539786870243e-04 |
    | 1 | +50 | 7.310585786300 |
    | 1 | +10 | 7.310585786300 |
    | 1 | -50 | -7.310585786300 |
    | 1 | -10 | -7.310585786300 |

    Note one correction to the comment above: `silu(-50)` is `-9.643749e-21`, not
    `9.643749e-20`. The mantissa in that comment is right and the exponent is one
    place out.

    ### What the reference does, read from three sources

    - `scouzi1966` `DeepseekV4.swift:997-1357` (the file the card names) and
      `osaurus-ai/vmlx-swift-lm` `DeepseekV4.swift:435-594` (the attribution
      source). Both keep the fused Metal kernels and the compiled selector that
      this repository leaves out.
    - The Python, `mlx_lm/models/deepseek_v4.py:307-414`. `MoEGate.__call__` shows
      the bias-versus-weight split in as many words: `biased = scores + bias`,
      `inds = argpartition(-biased)`, `weights = take_along_axis(orig, inds)`.

    ### Checkpoint key paths, from task ^cbj4spk

    The quantization plan of the real checkpoint names
    `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj` (mxfp4, group size 32),
    `model.layers.N.ffn.shared_experts.{gate,up,down}_proj` (affine, group size
    64), and `model.layers.N.ffn.gate` is absent from the block and stays in high
    precision. The Swift module tree must give those flattened paths, thus the
    routed experts need a submodule keyed `switch_mlp`.

    ### Why this file cannot call `SwitchGLU`

    `SwitchGLU` takes a ONE-argument activation and computes `activation(gate) *
    up`. The DeepSeek-V4 clamp is asymmetric and touches `up` as well, thus it
    cannot be written as a function of `gate` alone. `FusedGateUpSwitchGLU` does
    take a two-argument activation, but it reads one fused `gate_up_proj` tensor,
    and this checkpoint ships `gate_proj` and `up_proj` apart. The card forbids
    changing `SwitchLayers.swift`, thus this file builds a small `switch_mlp` of
    three `SwitchLinear` layers and calls `gatherSort`/`scatterUnsort`, which is
    the reuse the card asks for.

    ### One divergence between the references, and a follow-up task

    The two Swift references give `swiglu_limit` to the SHARED expert. The Python
    gives `swiglu_limit=0.0` there, which turns the clamp off. This card names the
    Swift file as the reference to port, thus this port follows the Swift and
    passes the limit. I filed a task to adjudicate the point against the real
    checkpoint.
  timestamp: 2026-08-11T11:10:19.571474+00:00
- actor: claude-code
  id: 01kzr90vvn1cn4bjme1hz7rxwg
  text: |
    ## The port landed, and every mutation died

    ### What the file holds

    `Libraries/MLXLLM/Models/DeepseekV4MoE.swift`, 352 lines, four types:

    - `DeepseekV4MLP` -- the dense block the shared expert reads. Calls
      `DeepseekV4Math.clampedSwiGLU`.
    - `DeepseekV4SwitchGLU` -- the routed experts, keyed `switch_mlp`. Three
      `SwitchLinear` layers, `gatherSort`/`scatterUnsort`, and the same
      `clampedSwiGLU`. `SwitchLayers.swift` is unchanged: `git diff --stat` on it
      is empty.
    - `DeepseekV4MoEGate` -- the router. `selectedExperts(scores:inputIds:)` is
      the only function the bias reaches, and `routedWeights(gatheredFrom:at:)`
      only ever sees the unbiased scores. The split makes the bug of the card
      hard to write by accident.
    - `DeepseekV4MoE` -- the layer. Weighs the routed outputs, calls
      `DeepseekV4Math.reduceRoutedExpertsFP32`, adds the shared expert, and casts
      back to the dtype of the input one time.

    The gate takes `inputIds` as an argument. The reference threads them through a
    mutable `currentInputIds` property so that its layer can conform to
    `UnaryLayer`. Every DeepSeek-V4 layer holds a mixture of experts, thus the
    decoder layer can hold this type itself; the argument holds no state and
    cannot be forgotten between calls.

    ### The tests, and the mutation proof

    `Tests/MLXLMTests/DeepseekV4MoETests.swift`, 14 tests. Every expected number
    comes from a NumPy transcription of the Python reference, run in float64. The
    transcription is not in this repository, thus no fixture came out of the Swift.

    `swift test --filter DeepseekV4MoETests`: 14 of 14 pass.
    `swift test --filter DeepseekV4`: 50 of 50 pass in 4 suites.
    `swift test`: 409 + 80 + 282 + 7 pass, 0 fail, 0 warnings.

    Nine mutations of the production file, each run against the whole suite. Every
    one died, and no test survived its own mutation:

    | Mutation | The test that died |
    |---|---|
    | The gate gathers the BIASED score | `gateWeightsAreTheUnbiasedScoresOfTheBiasedSelection`, `gateNormalizesAndScalesTheSelectedWeights` |
    | Hash routing always runs | `topKLayerSelectsTheHighestScoresRatherThanTheHashTable`, plus the two gate tests and `theRoutedReductionRunsInFloat32` |
    | Hash routing never runs | `hashLayerRoutesEachTokenToItsTableRow`, `hashLayerWeighsItsExpertsByTheirOwnScores` |
    | The routed experts take a limit of 0 | `theRoutedExpertsReadTheClampOfTheCheckpoint` |
    | The shared expert takes a limit of 0 | `theClampHoldsTheHighSideOfTheGate`, `theClampHoldsBothSidesOfTheUpProjection` |
    | The shared expert clamps the gate on BOTH sides | `theClampLeavesTheLowSideOfTheGateAlone` |
    | The routed reduction runs in the activation dtype | `theRoutedReductionRunsInFloat32` |
    | The sorted path never unsorts | `theSortedRoutingPathAgreesWithTheUnsortedPath` |
    | The shared expert never joins the result | `theSharedExpertAddsToTheRoutedResult` |

    The harness is `mutate.py` in the scratchpad of this session. It copies the
    file, changes one line, runs the suite, and puts the file back. The restored
    file is byte for byte the file before the run.

    ### Two notes for the next agent

    1. **The float32 reduction needed a fixture that survives the final cast.** The
       layer casts back to the dtype of the input, thus a float32 sum and a
       bfloat16 sum of the same numbers usually round to the same answer and the
       test proves nothing. The fixture makes the two routed outputs almost
       cancel: in float32 the answer is -0.001716613770, and in bfloat16 each
       product rounds to 0.609375 first and the answer is exactly zero. That gap
       survives the cast.
    2. **`swift build --build-tests` broke once on a stale code signature**, on
       `MLXFoundationModelsTests.xctest`, which held a `.cstemp` file from an
       earlier run. `rm -rf` on that bundle and a rebuild cleared it. It is not
       related to this change.
  timestamp: 2026-08-11T11:23:45.397100+00:00
- actor: claude-code
  id: 01kzr91yavsgnqpcvgtr0c8m45
  text: |
    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV4MoE.swift (352 lines, new), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4MoETests.swift (599 lines, new). `swift test --filter DeepseekV4MoETests` 14/14; `swift test --filter DeepseekV4` 50/50; `swift test` 409+80+282+7, 0 failures, 0 warnings. 9 of 9 mutations killed, 0 survivors. `SwitchLayers.swift` untouched. `swift-format` on the two new files changed nothing.
    - next: /review
  timestamp: 2026-08-11T11:24:20.699489+00:00
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
- 01KZGMRCBJ4SPK7C26PWKV5J6F
position_column: doing
position_ordinal: '80'
title: 'Port DeepseekV4 MoE: sqrtsoftplus gate, hash routing, clamped-SwiGLU experts'
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4MoE.swift` with `DeepseekV4MoEGate`, `DeepseekV4MoE`, and `DeepseekV4MLP`.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4MoEGate` at line 997, `DeepseekV4MoE` at line 1235, `DeepseekV4MLP` at line 1332.

Real config shape: 256 routed experts, 6 experts per token, 1 shared expert.

Features, all NEW vs DeepseekV3:

1. **`sqrt(softplus(x))` routing scores** replacing DSV3's sigmoid. Critical subtlety: the routing bias is applied **for the top-k selection**, but the weights actually gathered must be the **UNBIASED** scores. Getting this backwards is silent and degrades quality without crashing.
2. **Hash routing on layers 0-2** (`num_hash_layers=3`): load the `tid2eid` int64 hash table from the checkpoint and route by token id, bypassing top-k entirely. Use `isHashLayer(_:)` from the config task.
3. **Clamped SwiGLU experts.** The clamp is ASYMMETRIC. `up` stays inside `-swiglu_limit` through `+swiglu_limit`; `gate` stays below `+swiglu_limit` only, and its low side is not clamped, because `silu` falls to almost zero at a large negative gate. Read the clamp through the math helper `DeepseekV4Math.clampedSwiGLU` (`swiglu_limit`, Bug 2). The user decided the asymmetry on 2026-08-10 while task `^p34crp6` was in review; the earlier "plus or minus 10" wording of this card was wrong and is corrected here.
4. **FP32 routed-expert reduction** — accumulate expert outputs in float32 via `reduceRoutedExpertsFP32` before casting back.
5. Shared expert added to the routed result.

Build the routed experts on this repo's existing `SwitchGLU` / `SwitchLinear` / `QuantizedSwitchLinear` in `Libraries/MLXLMCommon/SwitchLayers.swift` — do **not** copy the reference's 1025-line variant; ours has diverged and its fused Metal kernels are deliberately out of scope (see task `wkv5j6f`). Use `gatherSort`/`scatterUnsort` (`SwitchLayers.swift:30`, `:51`) as the other MoE models here do.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~997-1360 (MIT; header attributes Osaurus AI).
- Routing/bias and hash-routing details: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`; `ml-explore/mlx-lm` PR 1189.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4MoE.swift` exists with all three types. It holds a fourth, `DeepseekV4SwitchGLU`, because the checkpoint names the routed experts `switch_mlp` and that path needs a module of its own.
- [x] Gate selects top-6 of 256 using **biased** scores but returns **unbiased** weights — asserted by a test with a deliberately lopsided bias.
- [x] Hash-routing layers (0-2) bypass top-k and route deterministically by token id; a repeated token id always hits the same expert.
- [x] Non-hash layers (3+) use the sqrtsoftplus top-k path.
- [x] Expert SwiGLU reads the asymmetric clamp: `up` inside -10 through +10, and `gate` below +10 with no low-side clamp.
- [x] Routed reduction happens in float32.
- [x] `Libraries/MLXLMCommon/SwitchLayers.swift` unmodified by this task.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4MoETests.swift`, synthetic config (8 experts, top-2) and repeatable random weights.
- [x] Test: with a bias vector that strongly favors expert 0, the *selected indices* shift toward expert 0 while the *returned weights* equal the unbiased scores at those indices. This is the bias-vs-weight guard.
- [x] Test: hash layer determinism — the same token id routes to the same expert across two calls, and does not depend on the hidden-state values.
- [x] Test: layer 3 uses top-k (assert exactly `numExpertsPerTok` nonzero weights per token).
- [x] Test: the clamp is active and asymmetric. An `up` of +50 gives the same expert output as an `up` of +10, an `up` of -50 gives the same output as an `up` of -10, and a `gate` of +50 gives the same output as a `gate` of +10. A `gate` of -50 does NOT give the same output as a `gate` of -10, because `silu(-50)` and `silu(-10)` are different numbers.
- [x] Test: full `DeepseekV4MoE` forward shape `[1, 4, hidden]` to same, every value finite, and shared-expert contribution is nonzero.
- [x] Run: `swift test --filter DeepseekV4MoETests` — 14 of 14 pass.

## Workflow
- Use `/tdd` — the biased-selection/unbiased-weight test is the highest-value assertion; write it first.
#deepseek-v4