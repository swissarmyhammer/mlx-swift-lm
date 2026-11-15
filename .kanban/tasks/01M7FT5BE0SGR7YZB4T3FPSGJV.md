---
assignees:
- claude-code
position_column: todo
position_ordinal: '8680'
title: 'GDN decode kernel test: build the fixture layers from a seeded random state'
---
## Why
- `GDNDecodeKernelLayerTests.anOpPathThatSkipsAStateMovesTheOutputPastTheTolerance` is a positive control: zeroing the recurrent state must move the output past the tolerance.
- `GDNDecodeKernelFixture.layer` builds each layer with `kind.make(headCase)`, which draws the projection weights from the **global** MLX random state. Only the norm weight, the states and the token use seeded keys.
- The weights therefore depend on how many global draws the other suites of `MLXLMTests` made first. During ^pg2hsb8, one MLXLMTests run with an unseeded module build in a new suite made the Qwen3-Next / bfloat16 case fail: `Fixture.worstGap(noRecurrentState.output, full.output) > Fixture.tolerance(dtype)`. The 7 earlier logs pass.

## What
- Build each fixture layer inside `withRandomState(MLXRandom.RandomState(seed: ...))`, with a named seed in `Fixture.Seed`, so that the weights are the same in every run of the bundle.
- Check that the positive control holds for every layer case and dtype with the seeded weights.

## Acceptance
- The fixture layers no longer read the global random state.
- MLXLMTests is green. #test-flake