---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3aayh9xanjeexdwm0xsrrdj
  text: |-
    Research and implementation notes:
    - A full `RotatingKVCache` gives its own `keys`/`values` objects as `state`, and a later single-token `update` writes into those objects with a subscript set. Thus `preparePromptCacheSave` takes a new handle of each array (`$0[.ellipsis]`, the pattern of `ArraysCache.copy()`). The new test "an update after prepare does not change the bytes that write puts in the file" uses a full ring to prove it.
    - The mlx safetensors load is lazy and does not check the file size. A file cut in the array data loads, and the eval then reads past the end of the file, which stops the process. `ExecutorPromptCacheFile.read` thus reads the header first (`SafetensorsLength.requireWholeFile`) and throws `truncated` before any array is made.
    - The byte identity of `savePromptCache` is held by a golden SHA-256 digest (`PromptCacheSaveInputTests.savedFileDigest`), which the OLD code wrote for a fixed input before the change.
    - `ArraysCache` has no `update(keys:values:)`; a test advances a `MambaCache` by new slot arrays.
    - Baseline before the change: MLXLMTests 694 XCTest with 36 failures; Swift Testing 1203 tests with 101 baseline issues.
  timestamp: 2026-09-24T18:30:51.453365+00:00
- actor: claude-code
  id: 01m3aaykaj8x8prmpv30z3maqr
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/KVCache.swift, Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift (new), Tests/MLXLMTests/PromptCacheSaveInputTests.swift (new), Tests/MLXFoundationModelsTests/ExecutorPromptCacheFileTests.swift (new). MLXFoundationModelsTests: 258 tests pass.
    - next: /test
  timestamp: 2026-09-24T18:30:53.522205+00:00
- actor: claude-code
  id: 01m3abpt4tyveyp9kpq9w9mmyb
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` has only the known `missing creator for mutated node` warning. MLXLMTests: 694 XCTest with 36 failures (baseline), Swift Testing 1207 tests with 101 issues (baseline 101); PromptCacheSaveInputTests passes. MLXFoundationModelsTests 258 pass, MLXGuidedGenerationTests 70 pass, CXGrammarTests 7 pass, MLXHuggingFaceMacrosTests 5 pass.
    - Discovery: the safetensors writer places the arrays in the order of its own hash map, thus two writes of one input can differ in byte order. The first test run failed on raw byte compares for that reason. The tests now compare a content digest (every metadata entry and every array byte, in name order). The golden digest `1ba2a6a1...c424` was written by the OLD code (HEAD `KVCache.swift`) in two separate runs, with the same value.
    - next: /commit
  timestamp: 2026-09-24T18:44:06.938119+00:00
depends_on:
- 01M3A1PX65M12926Y4BJVAG56K
- 01M3A29W6YK4F0VMGBCB28DXZ2
- 01M3A1QAFD56F5TEPPADDJHENH
position_column: doing
position_ordinal: '80'
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
- [x] prepare → write → read gives an entry with equal `tokens`, `renderTokens`, cache offsets, `state` arrays and `LMOutput.State`.
- [x] After `prepare` returns, an `update(keys:values:)` on the entry's caches does not change the bytes that `write` puts in the file.
- [x] `savePromptCache` output is byte-identical to before the change for the same input, and all existing tests in `Tests/MLXLMTests/KVCacheTests.swift` pass unchanged.
- [x] A read with the wrong key throws.
- [x] A truncated file, a file with another `format`, and a missing file each throw, and none stops the process.
- [x] No `*.partial.safetensors` file remains after a successful write.

## Tests
- [x] New `Tests/MLXFoundationModelsTests/ExecutorPromptCacheFileTests.swift`: the cases above with `KVCacheSimple`, `RotatingKVCache` and `MambaCache` layers, and one entry with an `LMOutput.State`. Temporary directory.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` and `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` — the new tests pass and no existing test changes result.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.