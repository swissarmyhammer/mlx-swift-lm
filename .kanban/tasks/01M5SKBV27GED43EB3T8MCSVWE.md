---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m61sn8ks10ty21bp375m8qd1
  text: |-
    Picked up. Research findings:
    - `rewindSpeculativePromptCache` now lives in Libraries/MLXLMCommon/KVCache.swift; its private `rewindSpeculativeEntries` guards `numTokens == 1` and calls `MambaCache.restoreSpeculativeCheckpoint()`.
    - `MambaCache` (ArraysCache.swift) already has `selectSpeculativeSlot(rollingBack:)` for any r in 1..<T when it holds kernel snapshot slots, but the GDN op path (`recurrentBody` in MLXLLM Qwen35 and the VLM twin) still splits at `checkpointAfter` and keeps ONE checkpoint (`GatedDeltaSpeculativeStates.checkpoint`). The shared Qwen35MTPFixture is float32 with hidden 32, so it runs the op path: a rewind by r > 1 needs the op path to keep every slot too.
    - The verify index travels as `checkpointAfter: Int?` through Qwen35 LLM/VLM forward, HybridDecodeLayer, CompiledDecodeSegments (`CompiledVerifyShape(tokens:checkpointAfter:)`), and Qwen3Next (precondition nil).
    - `Qwen35MTPDraftModel.maximumBlockSize = 2`, so the iterator only reaches r > 1 with a test drafter; ^hcfzyem raises it later.
    - After a partial-accept rewind the slots are released, so `finalizeGeneration` cannot rewind committed lookahead of a partially accepted round (unreachable at blockSize 2).
    Plan: the op path keeps the state after each token as `GatedDeltaDecodeKernel.Snapshots` (per-token `gatedDeltaUpdate`, bit-identical to today's 1+1 split at T = 2); the verify request becomes a Bool (no index); the single-checkpoint kind is removed as dead; the rewind selects slot r and trims attention by r; `maximumNativeTargetCacheRewind` = `GatedDeltaDecodeKernel.mostTokens - 1` (5).
  timestamp: 2026-10-28T17:41:51.865201+00:00
- actor: claude-code
  id: 01m61t8831pyxv7vkmqfafhzfb
  text: |-
    Implementation landed (library compiles, 0 warnings):
    - RED first: KVCacheContinuationTests `rewindingAHybridCacheByEachDraftCountSelectsItsSlotAndTrimsTheAttention` / `aHybridRewindItHoldsNoSlotForLeavesTheCacheAsItWas`; new Tests/MLXLMTests/Qwen35VerifyRewindTests.swift (direct T=6 verify rewound by r=1..5 vs a verify of the kept tokens, compiled and general; scripted-drafter iterator test blockSize 2..6 equals plain decode, covers every rejected-draft count, and checks the final cache against a one-shot prefill). RED observed: rewind returned 0 for r>1 (old restore picked slot after token 1), iterator threw unsupportedSpeculativeCache for blockSize>2.
    - Finding: "attention exact" only holds when the kept-token reference runs the same verify path; against a plain general-path run the attention KV differs by bf16 rounding of upstream GDN outputs (op path vs kernel). The direct test therefore compares to a verify of the kept tokens on the same path.
    - `rewindSpeculativePromptCache` takes 0 < r < slot count: trims attention by r, `selectSpeculativeSlot(rollingBack: r)` on each MambaCache.
    - Verify asks for all slots via new public `mtpVerifyKeepsSnapshotsKey: Key<Bool>` (replaces `mtpCacheCheckpointIndexKey: Key<Int>`); plumbing `checkpointAfter: Int?` -> `keepsSnapshots: Bool` in MLXLLM/MLXVLM Qwen35, Qwen3Next, HybridDecodeLayer, CompiledDecodeSegments (`CompiledVerifyShape(tokens:)`).
    - Op path keeps per-token slots (`GatedDeltaDecodeKernel.Snapshots.gatedDeltaUpdate`, one-token `gatedDeltaUpdate` per token; at T=2 the same two updates as the old split, so r=1 values are unchanged). The single-checkpoint kind (`GatedDeltaSpeculativeStates`, MambaCache checkpoint/restore API) is removed as dead.
    - `maximumNativeTargetCacheRewind` = `GatedDeltaDecodeKernel.mostRewoundTokens` (mostTokens - decodeTokens = 5) in all three places.
    Next: migrate old tests to the new API, full unit bundles, Release integration benchmarks.
  timestamp: 2026-10-28T17:52:13.921552+00:00
- actor: claude-code
  id: 01m61wb4nfhwaqgectg2s4m1cd
  text: |-
    ### implement — changed
    - evidence: 17 files. Libraries: MLXLMCommon/{KVCache, ArraysCache, GatedDeltaDecodeKernel, HybridDecodeLayer, CompiledDecodeSegments, MTPDrafterModel, MTPSpeculativeTokenIterator}.swift, MLXLLM/Models/{Qwen35, Qwen3Next}.swift, MLXVLM/Models/Qwen35.swift. Tests/MLXLMTests: KVCacheContinuationTests, Qwen35VerifyRewindTests (new), Qwen35MTPTests, HybridDecodeLayerTests, Qwen35GDNDecodeKernelTests, Qwen35RecurrentCachePositionTests, Qwen35VLMCompiledDecodeTests, Qwen3NextCompiledDecodeTests.
    - Unit (xcrun xctest, Debug, `swift build --build-tests` 0 errors / 0 warnings): MLXLMTests 897 XCTest 0 failures + 1917 Swift Testing passed; CXGrammarTests 7 passed; MLXFoundationModelsTests 376 passed; MLXGuidedGenerationTests 70 passed; MLXHuggingFaceMacrosTests 5 XCTest 0 failures.
    - Release qwen38FourBitMTPGeneration: PASS, 0 warnings; plain 59.6/59.6/59.5 tok/s; MTP math 92.7 (1.56x, acc 93.9%), code 83.1 (1.39x, acc 73.6%), prose 80.6 (1.35x, acc 69.0%) tok/s, all `matches plain: true` (within 3% of ~91/81/79). The test prints no model load time.
    - Release qwen38FourBitMTPSystemPromptCache: PASS, no known issues; edit turn with MTP prefilled 123 tok (cached 2169), follow-up turns prefilled 25–38 tok; MTP decode 86–91 tok/s vs plain ~56. Three near-tie text divergences (gap 0.125–0.375) were reported, and the test passed with them.
    - Public API change: `mtpCacheCheckpointIndexKey: Key<Int>` -> `mtpVerifyKeepsSnapshotsKey: Key<Bool>`; MLXVLM open `checkpointAfter:` params -> `keepsSnapshots:`.
    - Discovered: ^eenn005 (finalizeGeneration cannot rewind committed lookahead after a partially accepted round; reachable only at blockSize > 2).
    - next: /review
  timestamp: 2026-10-28T18:28:45.871095+00:00
- actor: claude-code
  id: 01m61wkrch6gzaf5c6m925swj2
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (987f3fb8) — 1 finding (1 confirmed, 2 refuted): Tests/MLXLMTests/KVCacheContinuationTests.swift:281 (reuse/reuse)
    - next: implement the 2026-10-28 13:29 Review Findings item, then re-review
  timestamp: 2026-10-28T18:33:28.209587+00:00
- actor: claude-code
  id: 01m61wm93g19ypvzx538bqajkk
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 17 files
    - test: green (implementer run) — 5 bundles, MLXLMTests 897 + 1917; Release MTP 92.7/83.1/80.6 = plain; cache guard pass, edit turn 123 tok
    - commit: 987f3fb8
    - review: findings — Tests/MLXLMTests/KVCacheContinuationTests.swift:281
  timestamp: 2026-10-28T18:33:45.328181+00:00
- actor: claude-code
  id: 01m61x44z7m590h9bzfb5nhs7c
  text: |-
    Review finding (reuse, KVCacheContinuationTests.swift:281) addressed:
    - New shared helper `ZeroFilledKVCache.make(tokens:heads:headDimension:)` in Tests/MLXLMTestSupport/ZeroFilledKVCache.swift (heads and headDimension default to 1): a KVCacheSimple with one `update(keys:values:)` of zero entries.
    - `hybridCacheAfterAVerify` (KVCacheContinuationTests) and `testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically` (Qwen35MTPTests) both call it; the recurrent half stays per-file (SnapshotSlotFixture / inline snapshots).
    - Other copies of the same zero-filled KVCacheSimple setup, routed through the helper: TranscriptKVCacheTests.attention(_:) (keeps 2 heads x 64 wide, named; 64 is needed for the quantization group), CacheConfigurationTests.cacheStatusReportsAllocatedMemory (heads/headDimension named and reused in the byte formula), ChatSessionQwenTextMTPTests raw-prompt-cache test, KVCacheConfigurationTests.modelCacheCopyAndTrimPreserveOneTimeline and its `populatedSimpleCache(headDimension:)` (was a `state =` assignment; now update-based, all its tests still pass), MiniMaxM3Tests dense cache in the prompt-cache round-trip test.
    - Left alone on purpose (not the same setup): caches fed distinct keys and values (ones vs zeros, MLXRandom, per-token fills), RotatingKVCache/QuantizedKVCache setups, and model forwards that update an existing cache (RecurrentAdvanceFixture.feed, HybridProgressModel).
  timestamp: 2026-10-28T18:42:25.383478+00:00
- actor: claude-code
  id: 01m61x48vq6skv9tbp7tk0nnh0
  text: |-
    ### implement — changed
    - evidence: 8 files, test-only — Tests/MLXLMTestSupport/ZeroFilledKVCache.swift (new), Tests/MLXLMTests/{KVCacheContinuationTests, Qwen35MTPTests, CacheConfigurationTests, ChatSessionQwenTextMTPTests, KVCacheConfigurationTests, MiniMaxM3Tests, ScriptedLM/TranscriptKVCacheTests}.swift. `swift build --build-tests`: Build complete, 0 compiler warnings/errors (one SwiftPM build-system note: "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS", not a compiler warning). xcrun xctest Debug: MLXLMTests 897 XCTest 0 failures + Swift Testing 1917 tests in 169 suites passed (1 pre-existing skipped: benchmarkVerification()); CXGrammarTests 7 passed; MLXFoundationModelsTests 376 passed; MLXGuidedGenerationTests 70 passed; MLXHuggingFaceMacrosTests 5 XCTest 0 failures. Review finding 2026-10-28 13:29 flipped to [x].
    - next: /review
  timestamp: 2026-10-28T18:42:29.367320+00:00
- actor: claude-code
  id: 01m61x7qm8vhx9dds7zqvxv7hw
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (db906f04), counts: findings 2, confirmed 2, refuted 1, attempted 7, failed 0. Prior finding (2026-10-28 13:29, `Tests/MLXLMTests/KVCacheContinuationTests.swift:281` reuse/reuse) is resolved: `Tests/MLXLMTestSupport/ZeroFilledKVCache.swift` adds `ZeroFilledKVCache.make(tokens:heads:headDimension:)`, used by `hybridCacheAfterAVerify` and the Qwen35MTPTests rewind test; item is checked.
    - dropped under the review skill's written rule (drop findings that ask to refactor tests that already existed): both confirmed findings land on `Tests/MLXLMTests/MiniMaxM3Tests.swift:898` (`duplication/duplication` and `reuse/reuse`). They ask to generalize the helper into `ZeroFilledKVCache.fill(_ cache: KVCache, tokens:heads:headDimension:)` and replace the `MiniMaxM3KVCache` sparse-cache zero fill with it. That sparse-cache `update(keys:values:)` fill existed before this commit (HEAD~1 MiniMaxM3Tests.swift, `sparseCache.update(`) in a pre-existing test; this commit only rewrote its literals to the shared `tokens`/`heads`/`headDimension` constants.
    - next: none; task moved to done. If the drop is not wanted, reopen and apply the `fill(_:tokens:heads:headDimension:)` generalization.
  timestamp: 2026-10-28T18:44:22.792637+00:00
- actor: claude-code
  id: 01m61x8c50x0hnw04nvd0gabbc
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 8 test files (shared ZeroFilledKVCache helper)
    - test: green (implementer run) — 5 bundles, MLXLMTests 897 + 1917
    - commit: db906f04
    - review: clean — 2 MiniMaxM3Tests.swift:898 findings dropped by the reviewer under the review skill's pre-existing-test rule (the sparse-cache fill predates this commit; only its literals changed). Task in done.
  timestamp: 2026-10-28T18:44:43.808110+00:00
depends_on:
- 01M5QWJGM1679S6CWHFCST7WAN
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffef80
title: 'MTP: rewind the hybrid main cache by more than one token after a verify'
---
## What
Three places allow a rewind of one token only:
- `rewindSpeculativePromptCache` has `guard numTokens == 1` (Libraries/MLXLMCommon/KVCache.swift ~3797), and `finalizeGeneration` calls it.
- The verify pass sets `verifyState[mtpCacheCheckpointIndexKey] = nativeHybridRewind ? 1 : nil` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift ~527).
- `maximumNativeTargetCacheRewind { 1 }` in MLXLLM Qwen35.swift (two places, ~1332 and the `Qwen35Model` extension ~1399) and MLXVLM Qwen35.swift (~1989). The iterator compares it with `blockSize − 1` (~182-186, ~437-441, ~740-744).

With the snapshot slots of the T-token GDN kernel, a rewind by r selects slot r, and the attention caches trim by r.

- [x] Change `rewindSpeculativePromptCache` to take r ≤ number of slots: select GDN slot r and trim the attention caches by r. Keep the r = 1 result identical to today.
- [x] Remove the fixed index 1 at the verify: the verify asks the GDN layers for all slots; after the accept decision the iterator rewinds by `numDraft − accepted`.
- [x] Raise `maximumNativeTargetCacheRewind` to 5 in the three places.

## Acceptance Criteria
- [x] On the shared fixture with blockSize 2, the output and the caches are the same as before this task.
- [x] A direct test: run a T-token verify, rewind by r = 1..5, and compare the main caches with a run of the kept tokens only (GDN within tolerance, attention exact).

## Tests
- [x] Tests/MLXLMTests/KVCacheTests.swift and Tests/MLXLMTests/Qwen35MTPTests.swift: the rewind tests above. (Placed in the split topic files: Tests/MLXLMTests/KVCacheContinuationTests.swift and the new Tests/MLXLMTests/Qwen35VerifyRewindTests.swift, because Qwen35MTPTests.swift is already 104 KB.)
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache

## Review Findings (2026-10-28 13:29)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 18 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5SKBV27GED43EB3T8MCSVWE.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SKBV27GED43EB3T8MCSVWE.md` — no validator matches this file
> - `.kanban/tasks/01M61T8ZW58TVX5JPXREENN005.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61T8ZW58TVX5JPXREENN005.md` — no validator matches this file

- [x] `Tests/MLXLMTests/KVCacheContinuationTests.swift:281` `reuse/reuse` — The new `hybridCacheAfterAVerify` helper rebuilds the attention half of a hybrid cache by hand: a `KVCacheSimple`, a zero key tensor, and an `update(keys:values:)` call. The same three-step setup already exists inline in `testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically`. Two copies of this setup will drift apart, and a fix to one will not reach the other. Move the attention-cache setup into one shared test helper, for example a `KVCacheSimple` factory that takes a token count, and call it from both `hybridCacheAfterAVerify` and the Qwen35MTPTests rewind test. Keep the recurrent-cache half in the per-file helper, since it depends on `SnapshotSlotFixture`.
