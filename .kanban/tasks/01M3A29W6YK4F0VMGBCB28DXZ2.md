---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3a98y6ggqeb9e22q7carrxh
  text: |-
    Research done.
    - `ArraysCache.restoreFromMetaState` does not set `offset`, thus a restored `MambaCache`/`ArraysCache` has offset 0.
    - `ChunkedKVCache` inherits the `KVCacheSimple` state setter, which sets `offset = keys.dim(2)`. After `maybeTrimFront()` the true offset is `startPosition + keys.dim(2)`, thus the restore loses `startPosition` tokens.
    - The model-state record lives in the user-metadata part (`1.__mlx_lm_state_*`) and `loadPromptCacheState` removes it. The offset record uses the same part with the new prefix `__mlx_lm_offset_`, which does not start with `__mlx_lm_state_`, thus the state-count check does not see it.
    - Plan: a namespace `PromptCacheOffsetRecord` in `KVCache.swift` writes the record in `savePromptCache`, reads and removes it in `PromptCacheFileContents`, and applies it in the three load functions after the restore.
  timestamp: 2026-09-24T18:01:35.184288+00:00
- actor: claude-code
  id: 01m3a9s0b1cfjrdj80y4eckjmf
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLMCommon/KVCache.swift (new private enum `PromptCacheOffsetRecord`; `savePromptCache` writes `1.__mlx_lm_offset_<layer>` and refuses the prefix in user metadata; `PromptCacheFileContents` reads and removes the record; the three load functions apply it after the restore: SET for ArraysCache/MambaCache/ChunkedKVCache, CHECK and throw `KVCacheError` for other classes), Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift (7 new tests, new `OffsetCacheKind`).
    - TDD: red run had the new tests fail (offset 0 or 16 after load, no throw on disagreement or malformed record); green run: MLXLMTests 694 XCTest with 36 failures (baseline) and 1202 Swift Testing tests with 101 issues (baseline). All new tests pass, KVCacheTests pass without changes.
    - Discovery: every existing class (KVCacheSimple, Rotating, Quantized, TurboQuant, VarianceNormalized, CacheList, DeepSeekV4Cache, MiniMaxM3KVCache) already restores its exact offset, thus the CHECK does not refuse any file that the old code wrote correctly.
    - next: /test
  timestamp: 2026-09-24T18:10:21.665890+00:00
- actor: claude-code
  id: 01m3a9v1d5bzb0qxgw1d2cshq1
  text: |-
    ### test — green (baseline only)
    - evidence: `swift build --build-tests` — one known warning `missing creator for mutated node`, no new warning. MLXLMTests: 694 XCTest, 36 failures (baseline: GlmOcr/Qwen25VL/Qwen35/Qwen3VL ContinuationTests, NanbeigeTests); 1202 Swift Testing tests, 101 issues (baseline: DeepSeekV4Attention/HyperConnection/MoE, MiniMaxM3, Qwen35MTP, SSM). MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 244 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 XCTest passed. No skipped test.
    - next: /commit
  timestamp: 2026-09-24T18:11:28.293971+00:00
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
position_column: doing
position_ordinal: '80'
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