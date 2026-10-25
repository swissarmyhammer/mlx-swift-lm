---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzjcpchtf2cj3f2knxdmr
  text: 'Change after the PR 581 port (commit 843854ec): do not add a new type `MTPDrafterCacheState`. Extend the existing `MTPDrafterContinuation` (Libraries/MLXLMCommon/MTPDrafterModel.swift) with `byteCount`, a drafter identity, `isAligned(withMainTokenCount:drafterID:)`, and `arrays()` / `metadata()` / `init(arrays:metadata:drafter:)`. The deep copy for fork already exists in `RealizedCache.copy()`.'
  timestamp: 2026-10-25T13:46:11.222975+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8e80'
title: 'MTP: make the drafter state a portable cache value (copy, byte count, serialize, cold start)'
---
## What
Two separate prompt caches must carry the MTP drafter state (drafter KV cache + pending hidden row) next to the main cache:
- ChatSession: `RealizedCache` (Libraries/MLXLMCommon/ChatSession.swift ~365-411), `PromptCacheCheckpoint` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift), and `PromptCacheSnapshot` (Libraries/MLXLMCommon/KVCache.swift ~2340-2400).
- FoundationModels: `ExecutorPromptCacheEntry` (Libraries/MLXFoundationModels/ExecutorPromptCache.swift ~33-107), its checkpoint, and its spill file (Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift).

Give both caches one shared value type, so that the drafter rules are written one time. The fields `processedTokenCount` and `pendingHidden` and the cold start come from the model-side continuation task; this task wraps them. Rules:
- The drafter caches must NOT go into the `caches: [KVCache]` array (the template restore needs the model layer count, KVCache.swift ~2496, and the offset checks compare every cache with the token ledger, ExecutorPromptCache.swift ~1281-1303, ExecutorPromptCacheFile.swift ~150-154).
- The pending hidden row must NOT go into `LMOutput.State` (it sets `carriesModelState`, which turns off `RewindToCommonPrefixRule`, PromptCacheReusePolicy.swift ~76-78, ~325).
- A drafter state cannot be trimmed by itself (entry j needs target hidden j, and no cache keeps old hidden rows). It is aligned (same call, or a checkpoint at the same position) or it is dropped for the cold start.

- [ ] Add `MTPDrafterCacheState` (new file Libraries/MLXLMCommon/MTPDrafterCacheState.swift) that wraps `MTPDrafterState` with: `copy()` (deep enough for fork), `byteCount`, a drafter identity (model ID / config hash), and `isAligned(withMainTokenCount:drafterID:)` (true only when drafter == main − 1, `pendingHidden` is present, and the identity matches).
- [ ] Add `arrays() -> [String: MLXArray]` and `metadata() -> [String: String]` with reserved key prefixes (for example `__mlx_lm_mtp_drafter_`), that also record the cache kind (simple or rotating, with window), and a throwing `init(arrays:metadata:drafter:)` that checks the format version, the drafter identity, the cache kind, and the offsets.

## Acceptance Criteria
- [ ] A round trip through `arrays()` / `init(arrays:...)` gives a drafter state that continues a stream with the same greedy tokens as the original.
- [ ] `copy()` gives two states that continue independently.
- [ ] `init(arrays:...)` throws for a wrong format version, a wrong drafter identity, or wrong offsets; `isAligned` is false for a different drafter.

## Tests
- [ ] Tests/MLXLMTests/MTPDrafterCacheStateTests.swift on the shared fixture: round trip, copy independence, alignment checks, rejects.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache