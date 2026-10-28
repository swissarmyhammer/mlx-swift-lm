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
depends_on:
- 01M5QWJGM1679S6CWHFCST7WAN
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8280'
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