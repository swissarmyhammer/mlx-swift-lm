---
assignees:
- claude-code
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
position_column: todo
position_ordinal: '8980'
title: Keep the offset of each cache through a prompt cache file (recurrent caches and front-trimmed chunked caches)
---
#prompt-cache

(Plan item F4.)

## What

`ArraysCache`/`MambaCache` lose their `offset` through `savePromptCache`: their `metaState` (`Libraries/MLXLMCommon/KVCache.swift:1531-1554`) has no offset. The executor then refuses the restored cache, because every cache offset must equal the ledger length (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift:448`). Qwen3.5 / Qwen3-Next thus always come back cold. The `ChunkedKVCache` offset after a front trim is also not verified.

- In `savePromptCache` (`KVCache.swift:1957`), record the `offset` of each top-level cache in a NEW RESERVED metadata namespace (for example the prefix `__mlx_lm_offset_`), in the same way as the model-state prefix `__mlx_lm_state_` (`KVCache.swift:2178-2185`). Do NOT put it in `metaState` (the format that `testCacheSerialization` compares) and NOT in the user namespace.
- `savePromptCache` refuses user metadata that uses the new prefix (as it does for the state prefix, `KVCache.swift:2077`).
- `loadPromptCacheSnapshot(url:)`, `loadPromptCache(url:)` and `loadPromptCacheSnapshot(url:into:)` (task ^jvag56k) remove the record from the user `metadata` they give back. A file without the record loads as before.
- Apply the record after the restore. `KVCache.offset` has only a getter in the protocol, and `DeepSeekV4Cache.offset` / `MiniMaxM3KVCache.offset` are computed from inner caches. Thus: SET the offset (through `BaseKVCache`) only for `ArraysCache`/`MambaCache` and for `ChunkedKVCache`; for every other type, CHECK that the restored offset equals the recorded one and throw `KVCacheError` if it does not.

## Acceptance Criteria
- [ ] `MambaCache`, `ArraysCache`, and `ChunkedKVCache` after a front trim come back with the offset they had before the save, through both load functions.
- [ ] For every other type, a file whose record disagrees with the restored offset throws `KVCacheError`.
- [ ] The record does not appear in the user `metadata` of any load function.
- [ ] `savePromptCache` with user metadata that uses the reserved prefix throws.
- [ ] A file written before this change (no record) loads as before.
- [ ] All existing tests in `Tests/MLXLMTests/KVCacheTests.swift` pass without changes (they assert `snapshot.metadata == ["source": "test"]` and that no reserved key appears).

## Tests
- [ ] Extend `Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift` (from ^jvag56k) with the offset cases, the disagreement case, the reserved-prefix refusal, and an old-format file.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` — the new tests pass, and no test in `KVCacheTests.swift` changes result. (On 2026-09-24, `stable` already has 36 XCTest failures and 101 Swift Testing issues in DeepSeekV4*, MiniMaxM3, Qwen35MTPMetal and the chunked SSM test. Do not add to that list.)

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.