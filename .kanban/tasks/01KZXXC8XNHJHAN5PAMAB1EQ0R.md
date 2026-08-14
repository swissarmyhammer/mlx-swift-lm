---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyzgnwm38k9t7gxmx48dpfg
  text: |
    ### Research — the design the attention path needs

    Folded with `^3x0krt4`: the same work.

    **What the port has today**

    - `DeepSeekV4ModelInner` builds one mask with `createAttentionMask(h:cache:)` and NO `windowSize:`, thus every layer attends densely.
    - `newCache` comes from `KVCacheDimensionProvider`: 43 `KVCacheSimple` layers.
    - `DeepSeekV4Attention` builds the compressor and the indexer and calls neither.

    **Infrastructure this repository already gives**

    - `RotatingKVCache(maxSize:keep:)`. Its `makeMask(n:windowSize:returnArray:)` gives the windowed causal mask a local window needs, and its `logicalView(tail:)` reads the ring without a write.
    - `RotatingKVCache.isTrimmable` is `offset < maxSize`. A fresh cache is thus trimmable, which is what `everyCacheLayerRewinds` reads. After 128 tokens it is not, thus `RewindToCommonPrefixRule` cannot rescue a broken prefix — `ExtendCachedPrefixRule` is the rule requirement 3 rides on, and that rule needs no trim.
    - Prefill runs in balanced chunks of at most 512 tokens (`PrefillParameters.defaultStepSize`). A chunk boundary is thus NOT a multiple of the compress ratio, which is why the pooled state needs a carry buffer.

    **The pooled cache, and how it differs from the reference**

    `scouzi1966/mlx-swift-lm` keeps the incomplete tail as PROJECTED `kv`/`gate` rows and holds a hand-written overlap accumulator (`accumulateOverlapWindows`) for the ratio-4 layers. This port keeps the incomplete tail as RAW hidden rows and re-pools them, because:

    - `DeepSeekV4Compressor` is already tested as a stateless pooler with 12 tests and six mutation proofs. A raw carry feeds that very code and needs no second pooling path.
    - An overlapping layer keeps one MORE whole chunk of raw rows and drops the first pooled row of each call. Chunk `c` then reads chunk `c - 1`'s leading half from the real tokens rather than from the `-inf` padding the first row takes.
    - The two branches of a ratio-4 layer (the attention compressor and the compressor inside the indexer) pool the SAME raw rows at the same ratio, thus one carry serves both and only the two pooled tensors differ.

    Cost: at most `2 * ratio` rows are re-projected for each call. On a ratio-128 layer that is 256 rows through two 4096x512 linears for each decode step, which is under 1 percent of a 0.60 s step.

    **The selection is a mask, not a gather**

    `DeepSeekV4Indexer` answers a Boolean mask that already holds the causal rule. The reference gathers the top-k rows instead. A mask gives the same numbers, because a masked chunk takes no softmax weight, and it keeps the one selection path for prefill and decode.
  timestamp: 2026-08-14T01:52:18.836301+00:00
position_column: done
position_ordinal: fe80
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