---
assignees:
- claude-code
position_column: todo
position_ordinal: '8e80'
title: Check the top-level offset of a CacheList (BaichuanM1, FalconH1) against the prompt cache ledger
---
## What

Found during ^qfennz0. `CacheList` (`Libraries/MLXLMCommon/KVCache.swift`, `public class CacheList: BaseKVCache`) does not override `offset`. Its sub-caches move their own offsets, but the `offset` of the `CacheList` itself seems to stay 0.

`reconcilePromptCache` (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift:448`) sets `mainCacheIsAligned` only when every top-level cache offset equals the ledger length. BaichuanM1 (`newCache` gives `CacheList(MambaCache(), kvCache)` for each layer) and FalconH1 (`CacheList(MambaCache(), attentionCache)`) can thus never extend the ledger, as Qwen3-Next could not before ^qr0p806.

`CachedForwardSmokeTests.assertCacheAdvancesOnce` checks only `list[1].offset` of a `CacheList`, thus no test sees this.

## Acceptance Criteria

- [ ] A test with a tiny BaichuanM1 and a tiny FalconH1 shows the top-level offset of each cache after a prefill and after decode steps.
- [ ] If the offset is not the prompt length, `CacheList` (or the models) moves it, or the task records why the offset must stay.
- [ ] A warm continuation through `reconcilePromptCache` and through `savePromptCache`/`loadPromptCacheSnapshot` works for both models.
- [ ] All five unit bundles pass.

#prompt-cache