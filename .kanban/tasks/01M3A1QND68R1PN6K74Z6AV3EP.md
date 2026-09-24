---
assignees:
- claude-code
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
- 01M3A29W6YK4F0VMGBCB28DXZ2
- 01M3A1QAFD56F5TEPPADDJHENH
position_column: todo
position_ordinal: '8280'
title: Write an executor prompt cache entry to one file and read it back
---
#prompt-cache

## What

The disk spool stores one `ExecutorPromptCacheEntry` (`Libraries/MLXFoundationModels/ExecutorPromptCache.swift:33`) in one safetensors file. The entry already carries `state: LMOutput.State?` (`:58`), thus the file carries caches, model state, and the two token ledgers.

Two-step save in `Libraries/MLXLMCommon/KVCache.swift` (needed by the spool, task ^w0s77dt, to avoid a race):
- `public struct PromptCacheSaveInput: @unchecked Sendable` — the flat array dictionary (NEW `MLXArray` handles, not the caches' own objects), the flat metadata and the class names. It holds no reference to a cache.
- `public func preparePromptCacheSave(cache:metadata:state:) throws -> PromptCacheSaveInput` — reads `state`/`metaState` of each cache synchronously. The caller runs it while it owns the caches.
- `public func writePromptCache(_ input: PromptCacheSaveInput, url: URL) throws` — only calls `save(arrays:metadata:url:)`.
- `savePromptCache(url:cache:metadata:state:)` becomes `writePromptCache(preparePromptCacheSave(...), url:)`, with the same behavior.

Create `Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift` with `enum ExecutorPromptCacheFile`:
- `static func prepare(_ entry: ExecutorPromptCacheEntry, key: ExecutorPromptCacheKey) throws -> PromptCacheSaveInput`. User metadata: `format` = `"executor-prompt-cache-1"`, `modelID`, `sessionID`, `tokens` and `renderTokens` (each as base64 of little-endian Int32; a comment says a token id never exceeds Int32), `byteCount` (from ^ddjhenh).
- `static func write(_ input: PromptCacheSaveInput, to url: URL) throws` — write to a temporary file first, then rename to `url`, thus a crash never leaves a half file under the real name. The temporary name MUST end in `.safetensors`, because `save(arrays:metadata:url:)` selects the format from `url.pathExtension` and throws `unknownExtension` for anything else (`.build/checkouts/mlx-swift/Source/MLX/IO.swift`). Use `<name>.partial.safetensors`.
- `static func read(from url: URL, key: ExecutorPromptCacheKey, templates: [KVCache]) throws -> ExecutorPromptCacheEntry` — `loadPromptCacheSnapshot(url:into:)` (^jvag56k, with the offsets of ^b28dxz2). Throw when `format`, `modelID` or `sessionID` differ from `key`, when a ledger does not decode, or when any cache offset differs from `tokens.count`.
- `static func fileName(for key: ExecutorPromptCacheKey, generation: UInt64) -> String` — SHA-256 hex of `modelID + "\u{0}" + sessionID`, then `-<generation>.safetensors`. Model ids contain `/`, thus key strings never go into a path directly.

## Acceptance Criteria
- [ ] prepare → write → read gives an entry with equal `tokens`, `renderTokens`, cache offsets, `state` arrays and `LMOutput.State`.
- [ ] After `prepare` returns, an `update(keys:values:)` on the entry's caches does not change the bytes that `write` puts in the file.
- [ ] `savePromptCache` output is byte-identical to before the change for the same input, and all existing tests in `Tests/MLXLMTests/KVCacheTests.swift` pass unchanged.
- [ ] A read with the wrong key throws.
- [ ] A truncated file, a file with another `format`, and a missing file each throw, and none stops the process.
- [ ] No `*.partial.safetensors` file remains after a successful write.

## Tests
- [ ] New `Tests/MLXFoundationModelsTests/ExecutorPromptCacheFileTests.swift`: the cases above with `KVCacheSimple`, `RotatingKVCache` and `MambaCache` layers, and one entry with an `LMOutput.State`. Temporary directory.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` and `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` — the new tests pass and no existing test changes result.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.