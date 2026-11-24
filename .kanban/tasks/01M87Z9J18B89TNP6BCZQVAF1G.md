---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: PromptCacheCheckpoint.canLoseRewind does not look inside CacheList
---
Found while auditing cache access for ^rpafc4q.

`PromptCacheCheckpoint.canLoseRewind` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift) tests `!cache.isTrimmable || cache is RotatingKVCache || cache is VarianceNormalizedKVCache` on the top-level cache only. Inkling 2 makes each layer a `CacheList(RotatingKVCache or KVCacheSimple, SpeculativeConvolutionCache)`. A fresh `CacheList` reports `isTrimmable == true` (CacheList.isTrimmable(after: 0) is true for a fresh ring and for the convolution cache), and the type check never sees the `RotatingKVCache` inside it. So `PromptCacheCheckpoint.applies(to:)` (the form without `adding:`, used by `PromptCacheCheckpoint.applies(to: model)` and MLXFoundationModels/ThinkingRender.swift) reads a sliding-window Inkling 2 cache as one that never loses its rewind.

- [ ] Write a failing test: `PromptCacheCheckpoint.applies(to:)` on the caches of `Inkling2TinyModels.target(...)` should be `true`.
- [ ] Decide whether `canLoseRewind` should recurse into `CacheList` children (KVCacheTree.leaves), and fix.
- [ ] Check the Inkling 2 prompt-cache tests still pass.

#cluster