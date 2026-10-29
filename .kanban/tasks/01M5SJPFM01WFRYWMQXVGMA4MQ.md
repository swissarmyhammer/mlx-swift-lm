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
- actor: claude-code
  id: 01m5zxxbaffy64ksk7d1hr0mnp
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (1d4b0d24): 1 finding (1 confirmed, 4 refuted, 21 attempted) — Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:661 reuse/reuse. Requested checks: (1) copy() independence holds — each drafter KVCache is a new object, the shared arrays are immutable MLX values, and MLX does not donate a buffer another array still holds; (2) byteCount follows the residentByteCount contract the main-cache checkpoints use (a copied KVCacheSimple counts its offset rows, not the source's step-rounded buffer; boundary/seed rows are counted by both a copy and its source while both live) — same accounting as main caches, no finding; (3) config-hash identity accepted: spill files are already keyed by modelID, and in-memory continuations stay with their session's drafter, so a weights-only mismatch needs the same modelID and config; (4) init(arrays:metadata:drafter:) throws on format version and drafter identity before touching arrays, then checks layers/class via the prompt cache restore and the positions against cache offsets.
    - next: implementer resolves the open item; re-review.
  timestamp: 2026-10-28T00:17:42.223370+00:00
- actor: claude-code
  id: 01m5zxy1vmq83c03f4xzq2jzp6
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 11 files (continuation copy, byteCount, MTPDrafterID, serialization)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1872 Swift Testing; FoundationModels 376
    - commit: 1d4b0d24
    - review: findings — Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:661
  timestamp: 2026-10-28T00:18:05.300081+00:00
- actor: claude-code
  id: 01m5zymbxgdd5bhnqjxavnfc32
  text: |-
    Picked back up from review to resolve the open finding plus the user's extra robustness item.
    - Finding (reuse, MTPSpeculativeTokenIterator.swift `primeDrafterState`): added `MTPDrafterContinuation.alignedBoundary(withMainTokenCount:drafterID:) -> MLXArray?` (the boundary row when `isAligned`, else nil). `MTPDrafterPrefillFeed.init` and `primeDrafterState` both call it; the closure is gone. The iterator init's identity-only filter (adopt the continuation's drafter state only when `drafterID` matches) is a different rule and stays.
    - Extra item: no drafter stated its hidden width, so added `ResumableMTPDrafterModel.hiddenSize: Int?` (default `nil`, same default-extension pattern as `drafterID`; the test fakes keep nil). Qwen LLM and VLM drafters return `configuration.hiddenSize` (the seed hidden row is the predictor output, also `hiddenSize` wide). `init(arrays:metadata:drafter:)` now runs one `checkHiddenRow` (rank 3, dim 1 == 1, last dim == hiddenSize when stated) on the boundary row and on the seed hidden row, throwing `KVCacheError`. `boundaryHiddenRank` renamed `hiddenRowRank` since it now covers both rows.
    - TDD: RED 1 was the compile failure of the new `alignedBoundary` test; RED 2 (one full bundle run) showed the new `boundaryHiddenWidth` and `seedHiddenWidth` tamper cases failing with "an error was expected but none was thrown". GREEN after the check.
    - Note for the next agent: `xcrun xctest -XCTest <Suite>` does not filter Swift Testing tests (it ran 0), and `swift test --filter` dies on the metallib (see CLAUDE.md), so a single Swift Testing test cannot be run alone here; one full MLXLMTests run is about 3.5 minutes.
  timestamp: 2026-10-28T00:30:16.496085+00:00
- actor: claude-code
  id: 01m5zymfm1r4etqs8cqs896ybh
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/{MTPDrafterContinuation,MTPDrafterContinuationSerialization,MTPDrafterModel,MTPDrafterPrefillFeed,MTPSpeculativeTokenIterator}.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Tests/MLXLMTests/MTPDrafterContinuationTests.swift. `swift build --build-tests`: only the known "missing creator ... mlx-swift_Cmlx.bundle" warning. `xcrun xctest MLXLMTests.xctest`: 895 XCTest 0 failures; Swift Testing 1873 tests in 166 suites passed (1872 + 1 new; the tamper test gained 2 argument cases), only the known benchmarkVerification() skip. MLXFoundationModelsTests: 376 passed. Finding checked; extra width-validation item recorded as a checked subtask. Not committed.
    - next: /review. Left in doing.
  timestamp: 2026-10-28T00:30:20.289245+00:00
- actor: claude-code
  id: 01m5zywz4066bptny9eeqtdap3
  text: |-
    ### test — green
    - evidence: touched changed .swift files; swift build --build-tests (Build complete, no new warnings; only the known "missing creator ... mlx-swift_Cmlx.bundle"); xcrun xctest MLXLMTests: 895 XCTest, 0 failures + 1873 Swift Testing passed; MLXFoundationModelsTests: 376 passed, 0 failed. Nothing committed.
    - next: review/commit
  timestamp: 2026-10-28T00:34:58.304184+00:00
- actor: claude-code
  id: 01m5zz0ty643f47ca6e79esega
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (9a3db931) — 2 findings, 2 confirmed, 0 refuted — Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift:196, Tests/MLXLMTests/MTPDrafterContinuationTests.swift:383. Prior item MTPSpeculativeTokenIterator.swift:661 reuse is fixed: one `MTPDrafterContinuation.alignedBoundary(withMainTokenCount:drafterID:)` used by both the iterator and MTPDrafterPrefillFeed.init, no leftover duplicate.
    - next: implement the 2026-10-27 19:35 findings (batch-dimension agreement check plus a seed-batch tamper test; named constants in the test), then re-review.
  timestamp: 2026-10-28T00:37:05.094461+00:00
- actor: claude-code
  id: 01m5zz18bgx84ffseyd7jyva7x
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — 8 files (shared alignedBoundary; hiddenSize check on load)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1873 Swift Testing; FoundationModels 376
    - commit: 9a3db931
    - review: findings — Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift:196, Tests/MLXLMTests/MTPDrafterContinuationTests.swift:383
  timestamp: 2026-10-28T00:37:18.832482+00:00
- actor: claude-code
  id: 01m5zzs02p96xk37vzfexw7pzs
  text: |-
    Picked back up from review for the two 2026-10-27 19:35 findings.
    - Finding 1 (invariant-propagation, batch dimension): looked at the whole load path for shape invariants between the boundary row, the seed proposal and the drafter caches. Four were unchecked: (a) seed hidden row batch vs boundary batch, (b) seed token batch vs boundary batch, (c) seed token shape, which must be `[B, 1]` (`sampleMTPSeed` -> `normalizedMTPColumn` always makes rank 2, one column; `draftBlock` returns the seed as the proposal and the iterator flattens it into draft tokens, so a wider seed token would push extra drafts), (d) batch of every drafter cache array (KVCacheSimple keys/values `[B, H, T, D]`) vs boundary batch. The prompt cache template restore checks layer count, class, fixed configuration and offsets, but no batch.
    - Fix in MTPDrafterContinuationSerialization.swift: one `checkShape(_:named:expected: [Int?])` (nil = any size) replaces `checkHiddenRow` and the `hiddenRowRank` constant (renamed constant `batchAxis = 0`). Boundary row `[nil, 1, hiddenSize]`; new `seedProposal(in:batchSize:hiddenSize:)` checks the half-present case, seed token `[B, 1]` and seed hidden `[B, 1, hiddenSize]`; new `checkCacheBatch(_:batchSize:)` runs after the restore over `cache.flatMap(\.state)`. `drafterState` now takes the checked seed and only does positions.
    - TDD: RED was one full bundle run — the new tamper cases `seedHiddenBatch`, `seedTokenBatch`, `seedTokenCount`, `cacheBatch` failed with "an error was expected but none was thrown"; the new control test `restoreKeepsAWellFormedSeedProposal` (a `[B,1]` token + `[B,1,H]` row restores) passed, so the new checks are not over-strict. GREEN after the fix. The `seedHiddenWidth` case now writes a `[B, 1]` seed token (it used `[B]`) so it fails only on width.
    - Finding 2 (magic numbers): reproduced with `swiftlint lint --config` (only_rules no_magic_numbers, allowed [0,1,-1,100]); the only hit was `row.dim(2) + 1`. Named `batchAxis`, `hiddenAxis`, `excessSeedTokenCount` in the tamper enum and replaced every axis literal (`dim(0)`, `dim(2)`) in the file; the helper `widenedRow` is gone (seed tampers share `writeSeed(into:shapes:)`). swiftlint now reports 0 for both files.
    - Not covered, noted for later: a drafter cache's KV-head count / head dim vs the drafter configuration is not checked on load (the drafter protocol does not expose them; the main prompt cache loader does not check them either).
  timestamp: 2026-10-28T00:50:16.790986+00:00
- actor: claude-code
  id: 01m5zzs30a6v62g987yzj46px5
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift, Tests/MLXLMTests/MTPDrafterContinuationTests.swift. `swift build --build-tests`: only the known "missing creator ... mlx-swift_Cmlx.bundle" warning. `xcrun xctest MLXLMTests.xctest`: 895 XCTest, 0 failures; Swift Testing 1874 tests in 166 suites passed (1873 + 1 new control test; the tamper test gained 4 argument cases), only the known benchmarkVerification() skip. MLXFoundationModelsTests: 376 passed. swiftlint no_magic_numbers: 0 on both files. Both 2026-10-27 19:35 findings checked. Not committed.
    - next: /review. Left in doing.
  timestamp: 2026-10-28T00:50:19.786587+00:00
- actor: claude-code
  id: 01m60016tm02zqhb3dxcz76vcp
  text: |-
    ### test — green
    - evidence: swift build --build-tests (only known "missing creator ... mlx-swift_Cmlx.bundle" warning); xcrun xctest MLXLMTests — 895 XCTest, 0 failures, 1874 Swift Testing passed; MLXFoundationModelsTests — 376 passed; only skip is the env-gated benchmarkVerification() (known pre-existing)
    - next: review/commit (nothing committed by this step)
  timestamp: 2026-10-28T00:54:45.844869+00:00
- actor: claude-code
  id: 01m6004rnwhzfmsgf8haem2vcp
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (bea60e67) — 1 finding (confirmed 1, refuted 0): Tests/MLXLMTests/MTPDrafterContinuationTests.swift:375 reuse/reuse. Prior 19:35 items verified fixed in the diff: batch checks for seed token, seed hidden row and drafter caches against the boundary row (seedProposal/checkShape/checkCacheBatch) with seedHiddenBatch/seedTokenBatch/seedTokenCount/cacheBatch tamper cases; magic numbers replaced by batchAxis/hiddenAxis/excessSeedTokenCount. Drafter KV head count / head size validation is not in scope: this is a diff op, and that check concerns unchanged loader code that the engine did not flag.
    - next: delete the test-local `batchAxis` and use `MTPDrafterContinuation.batchAxis`, then re-review.
  timestamp: 2026-10-28T00:56:42.428917+00:00
- actor: claude-code
  id: 01m6005aa4xm3papsdhthb6jdd
  text: |-
    ### finish iteration 3 — findings
    - implement: changed — 2 files (batch/seed shape checks on load; named axis constants)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1874 Swift Testing; FoundationModels 376
    - commit: bea60e67
    - review: findings — Tests/MLXLMTests/MTPDrafterContinuationTests.swift:375
  timestamp: 2026-10-28T00:57:00.484514+00:00
- actor: claude-code
  id: 01m600f6agb67skwtcrzw1c0cf
  text: |-
    Picked back up from review for the 2026-10-27 19:55 finding (reuse/reuse, test-local `batchAxis`).
    - Deleted the tamper enum's `private static let batchAxis = 0`; all three uses (`boundaryHiddenWidth` shape, `cacheBatch` concatenation, `writeSeed`) now read `MTPDrafterContinuation.batchAxis` directly (internal, visible through `@testable import MLXLMCommon`). `hiddenAxis` and `excessSeedTokenCount` kept: the library has no counterpart.
    - Same cause elsewhere in the file: the `cacheKind` tamper hard-coded the prompt-cache class names "KVCache" and "RotatingKVCache", which the library defines as `KVCacheSimple.serializedClassName` and `RotatingKVCache.serializedClassName`. It now reads both through `try #require(...)` (they are `String?`).
    - Checked and left: `"0"` (formatVersion tamper) is a deliberately unknown version, not a restatement of `serializationFormatVersion`; `"another drafter"` duplicates the suite's own `foreignDrafterID` text, not a library constant; the `1` middle dimension of hidden rows / seed tokens has no named library constant (the loader writes `[nil, 1, hiddenSize]` / `[batchSize, 1]` inline).
    - TDD note: test-only refactor, no production change, so no RED step; the tamper test still runs all 12 cases and passes.
  timestamp: 2026-10-28T01:02:24.080274+00:00
- actor: claude-code
  id: 01m600fanjc8nchwg5qz1s6p9d
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/MTPDrafterContinuationTests.swift. `swift build --build-tests`: only the known "missing creator ... mlx-swift_Cmlx.bundle" warning. `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 895 XCTest, 0 failures; Swift Testing 1874 tests in 166 suites passed (tamper test 12 cases), only the known benchmarkVerification() skip. 2026-10-27 19:55 finding checked. Not committed.
    - next: /review. Left in doing.
  timestamp: 2026-10-28T01:02:28.530295+00:00
- actor: claude-code
  id: 01m600gvdzkg2p8yd95z798vw0
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (a2004e42) — 0 findings, 0 confirmed, 0 refuted (7 attempted, 0 failed); 1 file reviewed. The prior item (Tests/MLXLMTests/MTPDrafterContinuationTests.swift:375 reuse/reuse, batchAxis) is checked, and the commit replaces it with the library constant. Every prior Review Findings item is checked.
    - next: task moved to done.
  timestamp: 2026-10-28T01:03:18.463529+00:00
- actor: claude-code
  id: 01m600h5d723n4t4efzpstwygm
  text: |-
    ### finish iteration 4 — clean
    - implement: changed — 1 file (tamper tests read batchAxis and cache class names from the library)
    - test: green — implementer's run: MLXLMTests 895 XCTest 0 failures + 1874 Swift Testing (test-only change)
    - commit: a2004e42
    - review: clean — 0 findings, prior item verified; task done
  timestamp: 2026-10-28T01:03:28.679547+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffea80
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
- [x] (Added 2026-10-27 by the user, from the review: caching must be robust.) `init(arrays:metadata:drafter:)` checks the width of the boundary and seed hidden rows against the drafter, so a damaged file with a valid identity throws a typed error (`KVCacheError`) on load instead of failing at the first drafter call. — `ResumableMTPDrafterModel.hiddenSize` (default `nil`; the Qwen LLM/VLM drafters give `configuration.hiddenSize`); tested by the `boundaryHiddenWidth` and `seedHiddenWidth` tamper cases.

## Acceptance Criteria
- [x] A round trip through `arrays()` / `init(arrays:...)` gives a drafter state that continues a stream with the same greedy tokens as the original.
- [x] `copy()` gives two states that continue independently.
- [x] `init(arrays:...)` throws for a wrong format version, a wrong drafter identity, or wrong offsets; `isAligned` is false for a different drafter.

## Tests
- [x] Tests/MLXLMTests/MTPDrafterCacheStateTests.swift on the shared fixture: round trip, copy independence, alignment checks, rejects. — named Tests/MLXLMTests/MTPDrafterContinuationTests.swift after the type.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache

## Review Findings (2026-10-27 19:11)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 11 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWGEWP4506CZYXCPV3C64M.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWGEWP4506CZYXCPV3C64M.md` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift:661` `reuse/reuse` — The `alignedBoundary` closure repeats the continuation-alignment predicate that `MTPDrafterPrefillFeed.init` already applies to the same `(continuation, cachePosition, drafter)` triple. Two call sites now independently decide when a continuation's boundary can be reused. Any later change to the alignment rule has to be made in both places, and they can drift apart. Move the aligned-boundary selection onto `MTPDrafterContinuation`, for example `func alignedBoundary(withMainTokenCount:drafterID:) -> MLXArray?`, and call it from both `MTPDrafterPrefillFeed.init` and this site. The feed then has one source for the rule.

## Review Findings (2026-10-27 19:35)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 8 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWGWNFDVP77YAW9ACHVXY1.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWGWNFDVP77YAW9ACHVXY1.md` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/MTPDrafterContinuationSerialization.swift:196` `completeness/invariant-propagation` — The new seed-row check validates the seed hidden row's rank, dim(1) and width against `hiddenSize`, but not its batch dimension. The seed token and the boundary row are never compared to it either. A file whose seed hidden row has a different batch size from the seed token or the boundary row still loads, and the mismatch surfaces at the first drafter call. That is the failure this change is meant to move to load time. In `checkHiddenRow` or in `drafterState`, require the batch dimension of the boundary row, the seed hidden row and the seed token to agree. Add one tamper case that changes the seed batch and asserts `KVCacheError` at load.
- [x] `Tests/MLXLMTests/MTPDrafterContinuationTests.swift:383` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.

## Review Findings (2026-10-27 19:55)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 2 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.md` — no validator matches this file

- [x] `Tests/MLXLMTests/MTPDrafterContinuationTests.swift:375` `reuse/reuse` — The tamper enum declares its own `private static let batchAxis = 0`, which repeats the axis constant the library already defines for the same purpose. The two copies can drift apart, and a test that uses its own axis could pass while the restore check reads a different axis. Delete the private `batchAxis` at line 375 and refer to `MTPDrafterContinuation.batchAxis` in its place. Keep `hiddenAxis` as is, since the library has no matching constant for it.
