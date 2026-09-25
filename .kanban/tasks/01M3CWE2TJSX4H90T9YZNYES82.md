---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3cyjpzahrrjwhkxzaps61kj
  text: |-
    ### Research and proof of the defect

    - The defect is real for both models. `CacheList` did not override `offset`, thus the stored `BaseKVCache.offset` of the list stayed 0.
    - Red run (before the fix), tiny BaichuanM1 and FalconH1 added to `HybridRecurrentModelFixture` in `Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift`: "falconH1 after the prefill, layer 0 (CacheList): offset 0", "baichuanM1 after the restored decode, layer 0 (CacheList): offset 0". 97 issues, all in `HybridRecurrentCacheOffsetTests`, only for these two models.
    - A second defect in BaichuanM1: the conv `MambaCache` child (leaf [i, 0]) never moved its offset ("baichuanM1 after the warm continuation, leaf [0, 0] (MambaCache): offset 0"; the attention leaf had 23). FalconH1 already moved its `MambaCache` offset (`advance` plus `offset +=`).
    - A third gap: the prompt cache offset record on disk applies only to top-level caches. A restored `MambaCache` inside a `CacheList` thus kept offset 0.
    - Without the fix, `reconcilePromptCache` did not give `.extend`; the warm test fed the whole next prompt into the live caches (attention leaf offset 23 = 10 + 13).

    ### Fix

    - `CacheList.offset` is now computed: the largest offset of the children (0 for an empty list). The setter is a `preconditionFailure`, because the offset comes from the children. This keeps the semantics correct for each user of `CacheList`, not only these two models.
    - `PromptCacheOffsetRecord.apply` goes into each child of a `CacheList`, so a restored `MambaCache` child gets the recorded offset, and each other child is checked against it.
    - BaichuanM1 calls `convCache.advancePosition(by: L)`, thus each child of its list agrees.
    - Test note: the BaichuanM1 fixture uses `sliding_window` 64. The model computes one mask from `cache.first` (the `CacheList`) with no window, thus a window shorter than the prompt makes a cold prefill differ from a warm decode for reasons outside this task.
  timestamp: 2026-09-25T18:52:24.426889+00:00
- actor: claude-code
  id: 01m3cyk1gmrmc1dkqth0f0r20v
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/KVCache.swift (computed `CacheList.offset`, offset record applied into `CacheList` children), Libraries/MLXLLM/Models/BaichuanM1.swift (`convCache.advancePosition(by: L)`), Tests/MLXLMTests/HybridRecurrentCacheOffsetTests.swift (baichuanM1 and falconH1 fixtures, leaf offset checks, new warm continuation test). MLXLMTests after the fix: 695 XCTest 0 failures, 1216 Swift Testing passed.
    - new task for a discovery outside the scope: BaichuanM1 sliding-window mask (task 01M3CYJWDRH16AEGTFJZQYT5KS).
    - next: /test
  timestamp: 2026-09-25T18:52:35.220919+00:00
- actor: claude-code
  id: 01m3d30d10dzy8a4gamqftgtw0
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. `xcrun xctest` for each bundle: MLXLMTests 695 XCTest 0 failures + 1216 Swift Testing passed (runs 1, 3 and 4); MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 312 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 XCTest 0 failures. 0 skipped.
    - note: MLXLMTests run 2 had 1 issue in a wall-clock benchmark outside this change: `TextToolCallRecoveryBenchmark` "Incomplete candidate processing scales near-linearly" (`large < small * 3.5`, large 45925916, small * 3.5 30555875). Other processes held the load average at 20 to 24 during the runs. Runs 3 and 4 passed that test. `ChatSessionTests.testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns` (^k8hm41j) passed in each run.
    - next: /commit
  timestamp: 2026-09-25T20:09:47.296692+00:00
- actor: claude-code
  id: 01m3d38qpj3k9w2nkpkm32j2z4
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (2d459aa) — 0 findings, 0 confirmed, 1 refuted, 7 attempted, 0 failed. 3 source files reviewed; 4 kanban files had no validator.
    - next: done
  timestamp: 2026-09-25T20:14:20.370335+00:00
- actor: claude-code
  id: 01m3d38xxh4fq8a3m6fvqt6tmc
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — KVCache.swift (computed `CacheList.offset`, offset record applied into `CacheList` children), BaichuanM1.swift (conv `MambaCache` moves its offset), HybridRecurrentCacheOffsetTests.swift (BaichuanM1 and FalconH1 fixtures, leaf offsets, warm continuation test). Defect proved real for both models before the fix.
    - test: green — all five bundles pass; MLXLMTests passed in runs 1, 3 and 4 (695 XCTest + 1216 Swift Testing); run 2 had one wall-clock benchmark issue under load average 20 to 24 (TextToolCallRecoveryBenchmark), outside this change.
    - commit: 2d459aa fix(prompt-cache): read the offset of a CacheList from its children
    - review: clean — 0 findings
  timestamp: 2026-09-25T20:14:26.737842+00:00
position_column: done
position_ordinal: ffac80
title: Check the top-level offset of a CacheList (BaichuanM1, FalconH1) against the prompt cache ledger
---
## What

Found during ^qfennz0. `CacheList` (`Libraries/MLXLMCommon/KVCache.swift`, `public class CacheList: BaseKVCache`) does not override `offset`. Its sub-caches move their own offsets, but the `offset` of the `CacheList` itself seems to stay 0.

`reconcilePromptCache` (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift:448`) sets `mainCacheIsAligned` only when every top-level cache offset equals the ledger length. BaichuanM1 (`newCache` gives `CacheList(MambaCache(), kvCache)` for each layer) and FalconH1 (`CacheList(MambaCache(), attentionCache)`) can thus never extend the ledger, as Qwen3-Next could not before ^qr0p806.

`CachedForwardSmokeTests.assertCacheAdvancesOnce` checks only `list[1].offset` of a `CacheList`, thus no test sees this.

## Acceptance Criteria

- [x] A test with a tiny BaichuanM1 and a tiny FalconH1 shows the top-level offset of each cache after a prefill and after decode steps.
- [x] If the offset is not the prompt length, `CacheList` (or the models) moves it, or the task records why the offset must stay.
- [x] A warm continuation through `reconcilePromptCache` and through `savePromptCache`/`loadPromptCacheSnapshot` works for both models.
- [x] All five unit bundles pass.

#prompt-cache