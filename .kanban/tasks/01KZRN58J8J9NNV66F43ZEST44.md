---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzrqf3sqv5yvganm1ecepz1j
  text: |-
    Research. `MLXNN.Module` supports `@ParameterInfo(key:) var x: MLXArray?`: `unwrapProperty` handles `ParameterInfo<MLXArray?>`, and `update(parameters:verify:)` pairs an unset optional parameter with `(.none, .none)`, which it passes over even under `.allModelKeysSet`. `GLM4MOELite.swift` already declares `@ParameterInfo(key: "biases") var biases: MLXArray?`, thus the idiom is in the repository. The guarded-`Optional` pattern of `GLM4MOE.swift` and `GLM4LmHeadTiedLoadTests.swift` therefore carries over from a module to a raw parameter with no change.

    Reversal of a recorded decision. The doc comment of `tokenToExpert` said a layer that does not route through the hash table still holds the parameter "because the reference does and because a checkpoint that carries the tensor then loads". That reason does not survive the load requirement: `.all` carries `.allModelKeysSet`, which throws `keyNotFound` for a module parameter no weight fills, thus a placeholder table makes EVERY published checkpoint fail on the top-k layers, and the bias makes it fail on the hash layers. The reference named there is the Swift copy in osaurus-ai/vmlx-swift-lm, which builds a load path of its own. The reversal is written into the file header as detail 4 and into both parameter doc comments.

    `routedWeights(gatheredFrom:at:)` needed no change. It reads the scores and the selected indices and touches neither routing parameter, thus it already reads only what the layer holds.

    Dead end to avoid. `git checkout -- Libraries/MLXLLM/Models/DeepseekV4MoE.swift` during the mutation run threw the whole (uncommitted) fix away, because `HEAD` still holds the broken file. Copy the file to the scratchpad and copy it back instead.
  timestamp: 2026-08-11T15:36:12.343689+00:00
- actor: claude-code
  id: 01kzrqfkbdj21w9fjbs6hjdn5m
  text: |-
    ### implement — changed
    - files: Libraries/MLXLLM/Models/DeepseekV4MoE.swift; Tests/MLXLMTests/DeepseekV4GateCheckpointLoadTests.swift (new); Tests/MLXLMTests/DeepseekV4MoETests.swift; Tests/MLXLMTests/DeepseekV4ModelTests.swift
    - change: `bias` and `tid2eid` are now `MLXArray?`. A hash layer builds the table alone, every later layer builds the bias alone. `isHashLayer` became a computed property (`tokenToExpert != nil`), thus the gate holds one answer and not two. `selectedExperts` reads the parameter the layer has. `unusedHashTableShape` is gone. The file header carries the reversal as detail 4.
    - TDD: the new test was written first and failed with `.keyNotFound(path: ["model","layers","0","ffn","gate","bias"], modules: [... DeepseekV4MoEGate])` — the production failure word for word.
    - mutation 1, the gate declares both parameters again: `aCheckpointThatSplitsTheRoutingTensorsByLayerLoads` (keyNotFound on `...gate.bias`), `eachGateDeclaresTheRoutingTensorOfItsOwnKind`, `theLoadedHashLayerRoutesEachTokenToItsTableRow`, `theLoadedTopKLayerSelectsTheHighestBiasedScores`, and 6 named tests of DeepseekV4MoETests died.
    - mutation 2, the gate declares neither parameter: the same four tests died with `.incompatibleItems(path: [... "tid2eid"], item: "none")`, and `theModulePathsAreTheCheckpointKeyPaths` died as well.
    - tests: `swift test` — 449 + 80 + 282 + 7 + 0 tests over every target, 0 failures. The one `warning:` of the run is the SwiftPM build-graph note `missing creator for mutated node: (... mlx-swift_Cmlx.bundle/Contents/MacOS)`, which no source file of this change produces.
    - format: `swift-format format --in-place` and `swift-format lint` on the four changed files alone; lint exits 0.
    - next: `/review`
  timestamp: 2026-08-11T15:36:28.269854+00:00
position_column: doing
position_ordinal: '80'
title: DeepseekV4MoEGate declares tid2eid and bias on every layer, thus no real checkpoint loads
---
## What

`DeepseekV4MoEGate` (`Libraries/MLXLLM/Models/DeepseekV4MoE.swift`) declared BOTH
`@ParameterInfo(key: "tid2eid") var tokenToExpert` and
`@ParameterInfo(key: "bias") var bias` on EVERY layer. The DeepSeek-V4 Python
reference declares one or the other, never both:

```python
# Thump604/mlx-lm @ deepseek-v4-support-fixes, mlx_lm/models/deepseek_v4.py,
# MoEGate.__init__
if self.hash:
    self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
else:
    self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)
```

`MLXLMCommon.loadWeights` calls `model.update(parameters:verify: [.all])`, and
`.allModelKeysSet` throws `UpdateError.keyNotFound` for a module parameter no
weight fills. Thus, against the published checkpoint:

- layers 0 to 2 (the hash layers) hold `ffn.gate.tid2eid` and no
  `ffn.gate.bias`, so `bias` throws;
- layers 3 to 42 hold `ffn.gate.bias` and no `ffn.gate.tid2eid`, so `tid2eid`
  throws.

Either one stopped the load. No DeepSeek-V4 checkpoint could load.

The doc comment of `tokenToExpert` recorded the opposite decision -- "A layer
that does not route through the hash table still holds this parameter, at the
placeholder shape below, because the reference does and because a checkpoint
that carries the tensor then loads." The reference it named is the Swift copy
in `osaurus-ai/vmlx-swift-lm`, which builds its own load path. The Python is
the file the checkpoint was written for, and it disagrees. The declarations and
the doc comment are corrected together, and the file header records the
reversal.

## Where it came from

Task `^pwr8r3h` found this while it assembled the model. That card's tests use
a synthetic checkpoint built FROM the module tree, thus every parameter the
tree declares gets a value and the gap is invisible there.

## Acceptance Criteria

- [x] `DeepseekV4MoEGate` declares `tid2eid` on a hash layer alone, and `bias`
      on every other layer alone.
- [x] `selectedExperts(scores:inputIds:)` and `routedWeights(gatheredFrom:at:)`
      read the one the layer holds. (`routedWeights` reads the scores and the
      selected indices alone, thus it needed no change.)
- [x] A weight dictionary that holds `tid2eid` for layers 0 to 2 and `bias` for
      layers 3 and up loads into a `DeepseekV4Model` through
      `update(parameters:verify: [.all])` without an error.
- [x] `Tests/MLXLMTests/DeepseekV4MoETests.swift` still passes, corrected where
      it fills a parameter the layer no longer declares.

## Tests

- [x] New test: `Tests/MLXLMTests/DeepseekV4GateCheckpointLoadTests.swift`
      builds a checkpoint by rule -- the hash table on the hash layers alone
      and the bias on the later layers alone -- and loads it through
      `verify: [.all]`. Before the fix it threw `keyNotFound`.
- [x] Run: `swift test --filter DeepseekV4` -- 90 tests, all pass. Full
      `swift test` -- 0 failures.
#deepseek-v4