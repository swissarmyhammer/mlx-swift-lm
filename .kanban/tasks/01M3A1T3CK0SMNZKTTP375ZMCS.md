---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3a2xseyqh32mfxqzspbwmcv
  text: |-
    Research (implement, iteration 1):
    - `BaseKVCache.innerState()` is `public`, not `open`. `BaseKVCache` must declare its own `open var residentByteCount` so that a subclass override uses dynamic dispatch. Classes that conform to `KVCache` directly (`DeepSeekV4Cache`, `MiniMaxM3KVCache`, `RotatingStagedKVCache`) get the protocol-extension default.
    - Rule per type:
      - `KVCacheSimple`, `ChunkedKVCache`: default. `innerState()` gives the full step-padded `keys`/`values` buffers.
      - `RotatingKVCache`: default. `innerState()` gives the full ring buffers.
      - `QuantizedKVCache`: default. `innerState()` gives packed values + scales + biases for keys and values.
      - `ArraysCache`/`MambaCache`: default. The present slots only.
      - `CacheList`: override. The sum of `residentByteCount` of the children, so that a child with an override (for example `TurboQuantKVCache`) is counted correctly.
      - `VarianceNormalizedKVCache`: override. The existing `compactStorageByteCount` (slabs + pending tiles) plus the raw tail. This does not make the tile views that `state` makes.
      - `TurboQuantKVCache`: override. The sum of `nbytes` over raw keys/values, affine key triplet, packed arrays, norms and calibration scale. The shared codecs are not counted, because all layers share them. `memoryBytes` already counts the same set; it now returns `residentByteCount`.
      - `DeepSeekV4Cache`: default. `innerState()` gives the window + both chunk branches (chunks + carry).
      - `MiniMaxM3KVCache`: default. `innerState()` gives the wrapped keys/values + the raw `indexKeys` buffer.
    - `LMOutput.State.contents` is private, thus the extension goes in `LanguageModel.swift`.
    - `Cmlx` is not a product of mlx-swift. The test tries `import Cmlx` first (a transitive clang module); `MLXArray.ctx` is public for read.
  timestamp: 2026-09-24T16:10:38.430887+00:00
- actor: claude-code
  id: 01m3a3d47a1zpef907q0jaw3t0
  text: |-
    Implementation landed (TDD: the new suite failed to compile first, then passed).
    - `KVCache.residentByteCount` is a protocol requirement. The protocol-extension default and `BaseKVCache` (`open var`) both give the sum of `nbytes` over `innerState()` (new internal helper `[MLXArray].totalByteCount`).
    - Overrides: `CacheList` (sum of the children), `VarianceNormalizedKVCache` (`compactStorageByteCount` + raw tail), `TurboQuantKVCache` (all ten own arrays). `TurboQuantKVCache.memoryBytes` now returns `residentByteCount`; the file-private `DType.bytesPerElement` had no other caller, thus it is deleted. No `innerState()` changed.
    - `LMOutput.State.residentByteCount` in `LanguageModel.swift`.
    - `import Cmlx` compiles in the MLXLMTests target, thus the no-evaluation test uses `_mlx_array_is_available` as the card says.
    - Discovery: `any KVCache` is not class-bound, thus a test must use `var cache: any KVCache` to set `state`.
    - MLXLMTests: KVCacheByteCountTests 14/14 pass. Bundle totals equal the baseline: XCTest 36 failures, Swift Testing 101 issues. The build warning `maskedFill` at KVCache.swift (never used) was there before this change.

    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/KVCache.swift, Libraries/MLXLMCommon/LanguageModel.swift, Libraries/MLXLMCommon/TurboQuantKVCache.swift, Libraries/MLXLMCommon/VarianceNormalizedKVCache.swift, Tests/MLXLMTests/KVCacheByteCountTests.swift
    - next: /test
  timestamp: 2026-09-24T16:19:00.970659+00:00
- actor: claude-code
  id: 01m3a3mwzz4e08z1drhqs89hgn
  text: |-
    ### test — red (baseline failures only; no new failure, no new warning)
    - evidence: `swift build --build-tests` — no new warning (only the SwiftPM `missing creator for mutated node` line and the `maskedFill` never-used warning that were there before). `xcrun xctest` MLXLMTests: XCTest 694 run, 36 failures; Swift Testing 1184 tests, 101 issues — equal to the 2026-09-24 baseline; KVCacheByteCountTests 14/14 pass. The XCTest failures are the continuation tolerance tests (Qwen35/Qwen3VL/Qwen25VL/GlmOcr/Nanbeige `...ContinuationTests`, for example "0.002961278 is greater than 0.001"); none reads `residentByteCount`. MLXGuidedGenerationTests 70 pass, MLXFoundationModelsTests 228 pass, CXGrammarTests 7 pass, MLXHuggingFaceMacrosTests 5 pass. No skipped test.
    - next: /commit (the orchestrator permits only baseline failures)
  timestamp: 2026-09-24T16:23:15.711889+00:00
- actor: claude-code
  id: 01m3a3wxw4nwvnkdr0y7xnhsp7
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 1 finding (1 confirmed, 0 refuted): Tests/MLXLMTests/KVCacheByteCountTests.swift:134 `reuse/reuse`
    - next: /implement works the unchecked finding

    ### finish iteration 1 — findings
    - implement: changed (5 files)
    - test: baseline failures only (MLXLMTests XCTest 36 failures / Swift Testing 101 issues, same as baseline; new suite 14/14 pass; other 4 bundles pass)
    - commit: e5d93ff
    - review: findings — `Tests/MLXLMTests/KVCacheByteCountTests.swift:134` `reuse/reuse` — The test utility function `byteCount(of:)` reimplements the same logic as the newly added `totalByteCount` extension on `Array<MLXArray>` in KVCache.swift.
  timestamp: 2026-09-24T16:27:38.756570+00:00
- actor: claude-code
  id: 01m3a45vn1f7w3bxx0yy9pgdv6
  text: |-
    ### finish iteration 2 — clean
    - implement: changed (1 file — Tests/MLXLMTests/KVCacheByteCountTests.swift: removed `byteCount(of:)`, the four calls now use `totalByteCount`; the reuse finding is checked)
    - test: baseline failures only (MLXLMTests XCTest 694 run / 36 failures, Swift Testing 1184 tests / 101 issues, same as baseline; KVCacheByteCountTests pass; no new warning)
    - commit: e187c28
    - review: clean — review sha HEAD~1..HEAD, 0 findings; every prior finding is checked; task moved to done
  timestamp: 2026-09-24T16:32:31.393821+00:00
position_column: done
position_ordinal: ff9480
title: Measure the resident bytes of a KV cache for each cache type, with no evaluation
---
#prompt-cache

(Plan item F1.)

## What

The byte-limited memory tier (^ddjhenh) needs the real size of an entry. `KVCache.state` undercounts: `KVCacheSimple.state` is a slice of a buffer that grows in steps of 256 (`Libraries/MLXLMCommon/KVCache.swift:444-494`).

- Add `var residentByteCount: Int { get }` as a REQUIREMENT of the `KVCache` protocol (`KVCache.swift:71` area), with a default implementation in a protocol extension: the sum of `nbytes` over `innerState()`. It must be a requirement, not an extension-only member: an extension-only member uses static dispatch, and the store holds `[KVCache]`, thus a type-specific version would never be called.
- `MLXArray.nbytes` reads shape and dtype only, thus it evaluates nothing.
- Override where `innerState()` does not give the full buffers. Known gap: `TurboQuantKVCache` (`Libraries/MLXLMCommon/TurboQuantKVCache.swift:633`) does not override `innerState()`, thus the default gives 0. Implement its count from its own arrays (packed arrays, norms, raw prefill keys, calibration scale). Do NOT change `innerState()` of any type: `eval` and `CompiledTrace` use it.
- Check and give one rule for each type: `KVCacheSimple` (full step-padded buffer), `RotatingKVCache`, `QuantizedKVCache` (packed values + scales + biases), `ChunkedKVCache`, `ArraysCache`/`MambaCache` (present slots), `CacheList` (sum of children), `VarianceNormalizedKVCache`, `TurboQuantKVCache`, `DeepSeekV4Cache` (`Libraries/MLXLLM/Models/DeepSeekV4Cache.swift:433`: window + both chunk branches), `MiniMaxM3KVCache`.
- Add `public extension LMOutput.State { var residentByteCount: Int }` over its array values (`Libraries/MLXLMCommon/LanguageModel.swift:242`). Count only array values; do not throw.

## Acceptance Criteria
- [x] For each type above, `residentByteCount` read through `any KVCache` equals the sum of the sizes of the buffers the cache holds, including the step padding of `KVCacheSimple`.
- [x] `TurboQuantKVCache` with data gives a value greater than 0.
- [x] An empty cache gives 0.
- [x] No evaluation: build a cache from a lazy operation (for example `MLXArray.ones(...) * 2` not evaluated), read `residentByteCount`, then assert that each array is still not available with `_mlx_array_is_available` from the mlx-c header. If `Cmlx` cannot be imported from the test target, the implementer names the replacement check on this task before the tests are written.

## Tests
- [x] New `Tests/MLXLMTests/KVCacheByteCountTests.swift` (Swift Testing): one case for each type through `any KVCache`, one for `LMOutput.State`, and the no-evaluation case.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` — the new tests pass. (On 2026-09-24, `stable` already has 36 XCTest failures and 101 Swift Testing issues in DeepSeekV4*, MiniMaxM3, Qwen35MTPMetal and the chunked SSM test. Do not add to that list.)

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.

## Review Findings (2026-09-24 11:23)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 22 not reviewed (the `.kanban/tasks/*` files: no validator matches them).

- [x] `Tests/MLXLMTests/KVCacheByteCountTests.swift:134` `reuse/reuse` — The test utility function `byteCount(of:)` reimplements the same logic as the newly added `totalByteCount` extension on `Array<MLXArray>` in KVCache.swift. Both compute the sum of `nbytes` across an array of MLXArray values using identical reduce logic, so the test should reuse the shared extension instead. Remove the `byteCount(of arrays: [MLXArray]) -> Int` static function (lines 134-136) and update all test calls from `Self.byteCount(of: array)` to `array.totalByteCount` to reuse the shared extension property.