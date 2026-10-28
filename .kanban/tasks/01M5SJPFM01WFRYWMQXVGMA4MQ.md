---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzjcpchtf2cj3f2knxdmr
  text: 'Change after the PR 581 port (commit 843854ec): do not add a new type `MTPDrafterCacheState`. Extend the existing `MTPDrafterContinuation` (Libraries/MLXLMCommon/MTPDrafterModel.swift) with `byteCount`, a drafter identity, `isAligned(withMainTokenCount:drafterID:)`, and `arrays()` / `metadata()` / `init(arrays:metadata:drafter:)`. The deep copy for fork already exists in `RealizedCache.copy()`.'
  timestamp: 2026-10-25T13:46:11.222975+00:00
- actor: claude-code
  id: 01m5zwdhs8qk7fmj2548cjbbm4
  text: |-
    Picked up. Research:
    - Per the earlier comment, extend `MTPDrafterContinuation` (MTPDrafterModel.swift) instead of adding `MTPDrafterCacheState`. Its fields: `state: MTPDrafterState`, `targetBoundaryHidden` (= the spec's pendingHidden, never nil in this type), `targetProcessedTokenCount` (= processedTokenCount). Made only by `MTPSpeculativeTokenIterator.finalizeDrafterContinuation`, which leaves `nextPosition == targetProcessedTokenCount - 1`, `proposalAppended == 0`, seeds nil (removeLastEntries drops them).
    - The deep copy lives inline in `ChatSession.RealizedCache.copy()` (copies each drafter KVCache). Moving it to `MTPDrafterContinuation.copy()` gives ^achvxy1 one call and keeps the rule in one place.
    - No drafter identity exists. The continuation must carry the identity of the drafter that made it, so the drafter must expose one: add `drafterID` to `ResumableMTPDrafterModel` (default = type name; Qwen LLM/VLM drafters = type name + hash of the sorted-key JSON of their configuration). MLXLMCommon builds on Linux (Libraries/MLXLMCommon/Linux), so no CryptoKit: FNV-1a 64.
    - Serialization: reuse the prompt-cache layout and the template restore in KVCache.swift (`preparePromptCacheSave`, `PromptCacheFileContents`, `PromptCacheTemplateRestore`): the template restore already checks layer count, class (simple vs rotating) and fixed configuration (rotating window/keep) against the drafter's fresh caches, plus the offset record. Plan: give `PromptCacheFileContents` an in-memory init and add a package `restorePromptCacheSnapshot(arrays:metadata:into:)`; the continuation prefixes every key with `__mlx_lm_mtp_drafter_` so ^achvxy1 / the FM spill file can merge it beside the main cache keys.
    - Byte count: `residentByteCount` of the drafter caches (the measure `PromptCacheCheckpoint.byteCount` uses) + nbytes of the boundary hidden and the seeds.
    - The iterator's resume check is `targetProcessedTokenCount == cachePosition` and it adopts `mtpContinuation.state` regardless of drafter; switch it to `isAligned(withMainTokenCount:drafterID:)` so a continuation from another drafter cold-starts instead of feeding foreign caches to this drafter.
  timestamp: 2026-10-27T23:51:35.976091+00:00
- actor: claude-code
  id: 01m5zx8e0xs6qnbxhhqwasffwp
  text: |-
    Implementation landed (TDD; RED was compile failure of the new suite, then mutations checked each behavior test can fail: shallow copy() fails the copy test, removing both ID checks fails the cold-start test, removing the iterator-init ID filter crashes the layout test with "Qwen MTP cache/layer count mismatch").
    - New Libraries/MLXLMCommon/MTPDrafterContinuation.swift: `MTPDrafterID` (type name + FNV-1a-64 of the sorted-key JSON of the configuration; no CryptoKit because MLXLMCommon builds on Linux), the `MTPDrafterContinuation` struct moved out of MTPDrafterModel.swift with a new `drafterID`, `copy()` (new KVCache objects per drafter layer, evaluated), `byteCount` (drafter cache `residentByteCount` + boundary row + seeds, same measure as `PromptCacheCheckpoint.byteCount`), `isAligned(withMainTokenCount:drafterID:)`.
    - New Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift: `reservedKeyPrefix = "__mlx_lm_mtp_drafter_"`, `arrays()`/`metadata()` (drafter caches in the prompt cache layout under `__mlx_lm_mtp_drafter_cache.`, so class name and rotating window/keep are recorded), `init(arrays:metadata:drafter:)` checks version, identity, boundary row shape, restores into `drafter.makeState(nil).cache` through the existing template restore (layer count, class, fixed configuration, offset record), then positions vs cache offsets.
    - KVCache.swift: `PromptCacheFileContents.init(arrays:metadata:)` split out of `init(url:)`; new package `restorePromptCacheSnapshot(arrays:metadata:into:)`; `loadPromptCacheSnapshot(url:into:)` now shares the private restore body.
    - `ResumableMTPDrafterModel.drafterID` requirement (default = type name); Qwen LLM and VLM drafters store `MTPDrafterID(drafterType:configuration:)`.
    - Decode path: the iterator only adopts a continuation whose `drafterID` matches its drafter (else `makeState`), and both resume checks (iterator prime + `MTPDrafterPrefillFeed`) use `isAligned(withMainTokenCount:drafterID:)` instead of `targetProcessedTokenCount == cachePosition`.
    - `ChatSession.RealizedCache.copy()` now calls `continuation.copy()`.
    Deviations to note: `arrays()`/`metadata()` are `throws` (spec signature was non-throwing) because `preparePromptCacheSave` throws for a cache class without a serialized name; test file is MTPDrafterContinuationTests.swift (named after the type the earlier comment chose).
    Left for the next tasks: storing/counting the continuation in `PromptCacheCheckpoint` (^achvxy1), restoring it on rewind/edit (^ahnb0j9), and the FoundationModels `ExecutorPromptCacheEntry`/spill-file wiring. A reader that merges the continuation into a prompt cache file must strip keys with `reservedKeyPrefix` before `loadPromptCacheSnapshot`, since `PromptCacheFileContents` does not know them.
  timestamp: 2026-10-28T00:06:16.861135+00:00
- actor: claude-code
  id: 01m5zx8jywtpc0jwv34zbbhzre
  text: |-
    ### implement — changed
    - evidence: 11 files — new Libraries/MLXLMCommon/MTPDrafterContinuation.swift, Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift, Tests/MLXLMTests/MTPDrafterContinuationTests.swift; modified Libraries/MLXLMCommon/{MTPDrafterModel,MTPSpeculativeTokenIterator,MTPDrafterPrefillFeed,ChatSession,KVCache}.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Tests/MLXLMTests/Qwen35MTPTests.swift. `swift build --build-tests` no new warnings; `xcrun xctest MLXLMTests.xctest` 895 XCTest + 1872 Swift Testing (1860 + 12 new), 0 failures, only the known benchmarkVerification() skip; MLXFoundationModelsTests 376 passed; Release `qwen38FourBitMTPSystemPromptCache()` TEST SUCCEEDED with the 2 known issues.
    - next: /review. Left in doing.
  timestamp: 2026-10-28T00:06:21.916499+00:00
- actor: claude-code
  id: 01m5zxh5j0g484aevs2cpgbe3y
  text: |-
    ### test — green
    - evidence: swift build --build-tests (touched changed files; only warning is the known "missing creator ... mlx-swift_Cmlx.bundle"); xcrun xctest MLXLMTests.xctest — 895 XCTest, 0 failures; Swift Testing — 1872 tests in 166 suites passed (only skip is the known env-gated benchmarkVerification()); MLXFoundationModelsTests — 376 passed. No code changed.
    - next: review/commit
  timestamp: 2026-10-28T00:11:03.104436+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
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

(Per the comment of 2026-10-25: the shared value type is the existing `MTPDrafterContinuation`, not a new `MTPDrafterCacheState`.)

- [x] Add `MTPDrafterCacheState` (new file Libraries/MLXLMCommon/MTPDrafterCacheState.swift) that wraps `MTPDrafterState` with: `copy()` (deep enough for fork), `byteCount`, a drafter identity (model ID / config hash), and `isAligned(withMainTokenCount:drafterID:)` (true only when drafter == main − 1, `pendingHidden` is present, and the identity matches). — done on `MTPDrafterContinuation` in Libraries/MLXLMCommon/MTPDrafterContinuation.swift; identity is `MTPDrafterID` / `ResumableMTPDrafterModel.drafterID`.
- [x] Add `arrays() -> [String: MLXArray]` and `metadata() -> [String: String]` with reserved key prefixes (for example `__mlx_lm_mtp_drafter_`), that also record the cache kind (simple or rotating, with window), and a throwing `init(arrays:metadata:drafter:)` that checks the format version, the drafter identity, the cache kind, and the offsets. — Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift; `arrays()`/`metadata()` are `throws` (a drafter cache class without a serialized name cannot be saved).

## Acceptance Criteria
- [x] A round trip through `arrays()` / `init(arrays:...)` gives a drafter state that continues a stream with the same greedy tokens as the original.
- [x] `copy()` gives two states that continue independently.
- [x] `init(arrays:...)` throws for a wrong format version, a wrong drafter identity, or wrong offsets; `isAligned` is false for a different drafter.

## Tests
- [x] Tests/MLXLMTests/MTPDrafterCacheStateTests.swift on the shared fixture: round trip, copy independence, alignment checks, rejects. — named Tests/MLXLMTests/MTPDrafterContinuationTests.swift after the type.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache