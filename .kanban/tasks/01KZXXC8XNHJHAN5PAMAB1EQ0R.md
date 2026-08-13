---
assignees:
- claude-code
position_column: todo
position_ordinal: '9880'
title: 'Read the pooled chunks in DeepSeek-V4 attention: the sparse path and its pooled cache'
---
## What

``DeepSeekV4Attention`` holds a ``DeepSeekV4Compressor`` on each of the 41
layers 2 to 42, and a ``DeepSeekV4Indexer`` on each of the 21 even layers 2 to
42. Every tensor of both loads from the published checkpoint. The forward path
reads neither. This task makes the attention path read them, which is what
turns the 1M-token context from a loaded weight into a working one.

Task `^r92pjcr` landed the indexer, and task `^tty95f4` landed the compressor.
Both stop at the module boundary on purpose: the forward path cannot read
pooled chunks without a cache that keeps them.

## Why a cache is needed

A decode step carries one token. A stateless compressor pools nothing out of
one token, thus the global context would go away at the first decode step. The
reference holds the pooled chunks, and the tokens of an incomplete chunk, in a
`DeepseekV4Cache` that lives across calls -- the first 850 lines of
`scouzi1966/mlx-swift-lm` @ `main`,
`Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`. Two pool branches live in
that cache: one for the compressor of the attention and one for the compressor
inside the indexer.

## What the attention path must do

Read `attend` of `scouzi1966/mlx-swift-lm` @ `main`,
`Libraries/MLXLLM/Models/DeepseekV4.swift`. On a layer whose compress ratio is
more than 0:

- Pool the block through the compressor and keep the chunks in the cache.
- On a ratio-4 layer, ask the indexer which chunks each query reads. The
  selection ``DeepSeekV4Indexer`` answers is already a Boolean mask that holds
  the causal rule, thus the mask work of the reference is smaller here.
- Join the local sliding window of 128 keys with the chosen pooled chunks, and
  build the two-part mask `[window visibility | chunk visibility]`.
- Give a layer whose compress ratio is 0 the plain path it has today.

The local half also needs the rotating cache the header of
`Libraries/MLXLLM/Models/DeepSeekV4.swift` records: a window is only correct
beside a global path that carries the rest of the context.

## Acceptance Criteria

- [ ] A layer whose compress ratio is 0 answers exactly what it answers today.
- [ ] A prompt shorter than `sliding_window` answers exactly what it answers
      today, thus `greedyFirstTokensMatchThePythonFixture()` still passes.
- [ ] A prompt longer than `sliding_window` reads the pooled chunks, and a
      decode step after it reads them too.
- [ ] The pooled state survives a serialize and deserialize round trip, as
      `osaurus-ai/vmlx-swift-lm`
      `Tests/MLXLMTests/DeepseekV4CacheDiskRoundTripTests.swift` asks.
- [ ] `docs/deepseek-v4-support.md` records the closed gap.

## Tests

- [ ] A block fed in one call and the same block fed in several calls give the
      same pooled chunks.
- [ ] A decode step reads the chunks the prefill pooled.
- [ ] A prompt past 12k tokens keeps its answer coherent.
#deepseek-v4