---
assignees:
- claude-code
position_column: todo
position_ordinal: '9580'
title: DeepseekV4MoEGate declares tid2eid and bias on every layer, thus no real checkpoint loads
---
## What

`DeepseekV4MoEGate` (`Libraries/MLXLLM/Models/DeepseekV4MoE.swift`) declares BOTH
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

Either one stops the load. No DeepSeek-V4 checkpoint can load today.

The doc comment of `tokenToExpert` records the opposite decision -- "A layer
that does not route through the hash table still holds this parameter, at the
placeholder shape below, because the reference does and because a checkpoint
that carries the tensor then loads." The reference it names is the Swift copy
in `osaurus-ai/vmlx-swift-lm`, which builds its own load path. The Python is
the file the checkpoint was written for, and it disagrees. Correct the
declarations and the doc comment together.

## Where it came from

Task `^pwr8r3h` found this while it assembled the model. That card's tests use
a synthetic checkpoint built FROM the module tree, thus every parameter the
tree declares gets a value and the gap is invisible there.

## Acceptance Criteria

- [ ] `DeepseekV4MoEGate` declares `tid2eid` on a hash layer alone, and `bias`
      on every other layer alone.
- [ ] `selectedExperts(scores:inputIds:)` and `routedWeights(gatheredFrom:at:)`
      read the one the layer holds.
- [ ] A weight dictionary that holds `tid2eid` for layers 0 to 2 and `bias` for
      layers 3 and up loads into a `DeepseekV4Model` through
      `update(parameters:verify: [.all])` without an error.
- [ ] `Tests/MLXLMTests/DeepseekV4MoETests.swift` still passes, corrected where
      it fills a parameter the layer no longer declares.

## Tests

- [ ] New test: build a model, drop `bias` from every hash layer and `tid2eid`
      from every other layer, and load through `verify: [.all]`. This test
      fails today with `keyNotFound`.
- [ ] Run: `swift test --filter DeepseekV4` -- all pass.
#deepseek-v4