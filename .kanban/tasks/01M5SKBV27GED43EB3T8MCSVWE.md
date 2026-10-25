---
assignees:
- claude-code
depends_on:
- 01M5QWJGM1679S6CWHFCST7WAN
- 01M5SK9FZ2HTB7XJDAPCBRDDM5
position_column: todo
position_ordinal: '9580'
title: 'MTP: rewind the hybrid main cache by more than one token after a verify'
---
## What
Three places allow a rewind of one token only:
- `rewindSpeculativePromptCache` has `guard numTokens == 1` (Libraries/MLXLMCommon/KVCache.swift ~3797), and `finalizeGeneration` calls it.
- The verify pass sets `verifyState[mtpCacheCheckpointIndexKey] = nativeHybridRewind ? 1 : nil` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift ~527).
- `maximumNativeTargetCacheRewind { 1 }` in MLXLLM Qwen35.swift (two places, ~1332 and the `Qwen35Model` extension ~1399) and MLXVLM Qwen35.swift (~1989). The iterator compares it with `blockSize − 1` (~182-186, ~437-441, ~740-744).

With the snapshot slots of the T-token GDN kernel, a rewind by r selects slot r, and the attention caches trim by r.

- [ ] Change `rewindSpeculativePromptCache` to take r ≤ number of slots: select GDN slot r and trim the attention caches by r. Keep the r = 1 result identical to today.
- [ ] Remove the fixed index 1 at the verify: the verify asks the GDN layers for all slots; after the accept decision the iterator rewinds by `numDraft − accepted`.
- [ ] Raise `maximumNativeTargetCacheRewind` to 5 in the three places.

## Acceptance Criteria
- [ ] On the shared fixture with blockSize 2, the output and the caches are the same as before this task.
- [ ] A direct test: run a T-token verify, rewind by r = 1..5, and compare the main caches with a run of the kept tokens only (GDN within tolerance, attention exact).

## Tests
- [ ] Tests/MLXLMTests/KVCacheTests.swift and Tests/MLXLMTests/Qwen35MTPTests.swift: the rewind tests above.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache