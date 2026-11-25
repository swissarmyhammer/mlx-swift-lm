---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m886e2w0zqetqmt0fm6z0rws
  text: |-
    Implemented. `canLoseRewind` now (1) recurses into `CacheList` children, (2) reads a window through `cache.maxSize != nil` in place of `is RotatingKVCache`, so a composite that forwards its window (DeepSeekV4Cache, RotatingStagedKVCache) answers too, and (3) treats every `ArraysCache` (recurrent and convolution caches, incl. Inkling 2's SpeculativeConvolutionCache and LFM2's RewindableConvolutionCache) as able to lose its rewind, as KVCacheTree's `.recurrent -> .copy` already does. (3) matters: without it a global Inkling 2 layer, CacheList(KVCacheSimple, SpeculativeConvolutionCache), saved `.trim`, and a restore after more tokens returned nil because the convolution child cannot trim back. `saved(_:)` now copies a CacheList with `copyForCheckpoint()` on its rotating children (new `checkpointCopy(of:)`), so a nested ring does not pin prefill rows past its window either.

    Dead end: a first version used `!cache.isTrimmable(after: Int.max)` as a general probe. It broke 2 ChatSessionQwenTextMTPTests, because `SpeculativeStagingCache` (test fake) records every isTrimmable(after:) width it refuses. Replaced with the side-effect-free predicate above; the doc comment says why.

    Behaviour change to note: LFM2 top-level RewindableConvolutionCache layers now read as able to lose their rewind, so LFM2 sessions take checkpoints (copy) where before they relied on a trim that fails past the rollback. All LFM2 tests pass.

    Tests: new Tests/MLXLMTests/PromptCacheCheckpointCompositeTests.swift (7): a list with a sliding layer / a convolution / a variance-normalized cache can lose its rewind; a list of attention caches alone never does; the caches of the tiny Inkling 2 target can lose their rewind; an Inkling 2 checkpoint restores after the sliding window wraps; a checkpoint of a list keeps only the window of its sliding layer (640 vs 256 bytes before the fix). RED observed for 5 of the first 6 tests and the byte-count test. Full 5-bundle run green (Inkling 2 and prompt-cache suites included).

    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/PromptCacheCheckpoint.swift, Tests/MLXLMTests/PromptCacheCheckpointCompositeTests.swift (new)
    - next: review
  timestamp: 2026-11-25T01:51:58.336804+00:00
position_column: done
position_ordinal: fffff780
title: PromptCacheCheckpoint.canLoseRewind does not look inside CacheList
---
Found while auditing cache access for ^rpafc4q.

`PromptCacheCheckpoint.canLoseRewind` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift) tests `!cache.isTrimmable || cache is RotatingKVCache || cache is VarianceNormalizedKVCache` on the top-level cache only. Inkling 2 makes each layer a `CacheList(RotatingKVCache or KVCacheSimple, SpeculativeConvolutionCache)`. A fresh `CacheList` reports `isTrimmable == true` (CacheList.isTrimmable(after: 0) is true for a fresh ring and for the convolution cache), and the type check never sees the `RotatingKVCache` inside it. So `PromptCacheCheckpoint.applies(to:)` (the form without `adding:`, used by `PromptCacheCheckpoint.applies(to: model)` and MLXFoundationModels/ThinkingRender.swift) reads a sliding-window Inkling 2 cache as one that never loses its rewind.

- [x] Write a failing test: `PromptCacheCheckpoint.applies(to:)` on the caches of `Inkling2TinyModels.target(...)` should be `true`.
- [x] Decide whether `canLoseRewind` should recurse into `CacheList` children (KVCacheTree.leaves), and fix.
- [x] Check the Inkling 2 prompt-cache tests still pass.

#cluster