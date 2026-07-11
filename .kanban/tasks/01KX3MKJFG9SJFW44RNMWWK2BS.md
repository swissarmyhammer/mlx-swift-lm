---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx4fd8ywn73sa3mnqdak5fna
  text: 'Correction from user review: dependency task ^23ff1zx was revised from "pin CPU device" to a ROOT FIX (symlink mlx.metallib beside the xctest binary so GPU eval works under plain swift test). Where this task''s description says "(CPU-device test)", read "on the default device via the metallib bootstrap from ^23ff1zx" — no CPU pinning.'
  timestamp: 2026-07-09T22:19:11.196634+00:00
- actor: claude-code
  id: 01kx67rwkv8z3sb6ynxn5x1s7w
  text: |-
    Implemented and verified. Summary:

    **Scope-boundary decision**: `assemble(chunks:layerCount:)` was added as a `nonisolated static func` on the `PromptCache` actor in `Libraries/MLXFoundationModels/PromptCache.swift` (end of the "MARK: - Chunk store" section, after `chunkCount(modelID:)`), NOT wired into `resolve()`/`Executor`. This matches wx4w28j/mej3zgh's precedent: the acceptance criteria test `assemble()` directly (not via `resolve()`), and wiring the chunk store into cache reconstruction is explicitly called out in the existing `chunkStore` doc comment as "a later Assembly task" — this task IS that Assembly step, but wiring it into production `resolve()` reads as a further, still-later task. Confirmed via grep that `assemble` has zero call sites outside its own tests.

    **Implementation**: For each of `layerCount` layers, concatenates every chunk's key/value slices (in the caller-supplied chain order) along axis 2 (token/sequence dimension, matching `sliceChunks`'/`update()`'s own axis convention), wraps each concatenated result in `ownedCopy(of:)`, and installs `[keys, values]` via a fresh `KVCacheSimple`'s `state` setter (which derives `offset` from `keys.dim(2)`, landing exactly on the matched token count).

    **MLX buffer-aliasing investigation (the hard-won wx4w28j lesson, applied here)**: Read the actual vendored MLX C++ source. Two findings:
    1. `concatenated(_:axis:)` with 2+ inputs is genuinely safe: `Concatenate::eval_cpu`/`eval_gpu` (`mlx/backend/cpu/primitives.cpp`, `mlx/backend/metal/slicing.cpp`) unconditionally `malloc`s a fresh output buffer and copies every input into it — no aliasing.
    2. BUT `mlx::core::concatenate` (`mlx/ops.cpp`) special-cases a SINGLE input array: `if (arrays.size() == 1) { return arrays[0]; }` — it returns the exact same array object, never allocating anything or invoking the `Concatenate` primitive at all. Since a layer with exactly one matched chunk hits this path, a naive `concatenated([chunk.keys], axis: 2)` without a following copy step would hand back the chunk store's own owned tensor BY REFERENCE — violating "every resolve gets its own assembled copy."
       Fix: wrapped every layer's concatenated result (regardless of chunk count) in `ownedCopy(of:)` — the same `asData(access: .copy)` round-trip `sliceChunks` already uses for the identical reason. Widened `ownedCopy(of:)` from `private` to internal visibility in `PromptCacheChunks.swift` (doc comment updated to explain the new caller) so `assemble` could reuse it rather than reimplementing. Added `import MLX` to `PromptCache.swift` for `concatenated`.

    **Tests**: New file `Tests/MLXFoundationModelsTests/PromptCacheAssembleTests.swift` (10 tests, following `PromptCacheChunkTests`'s real-tensor-content + `MetalLibraryTestBootstrap` pattern, default/GPU device, no CPU pinning) — content equality (single- and multi-layer), zero-chunks-yields-empty, offset correctness, post-assembly `update()` advancing offset/preserving prefix, sequential distinct-instances, a genuinely concurrent distinct-instances test (`withTaskGroup` + `SendableBox`, per `PromptCacheConcurrencyTests`'s pattern), two aliasing regression tests for the single-chunk case (ObjectIdentifier must differ from the store's own tensor; mutating the store's tensor in place after assembly must not change the assembled cache), and a defensive-guard regression test for mismatched `layers.count`.

    **Adversarial review round**: Spawned `double-check`; verdict REVISE with one real finding (unguarded `layers[layerIndex]` index — a caller-supplied `layerCount` not matching a chunk's actual `layers.count` trapped with `Fatal error: Index out of range` instead of degrading like `sliceChunks` does) and one coverage gap (distinct-instances test was sequential, not literally concurrent, per the AC wording). Both fixed: added a `chunks.allSatisfy { $0.layers.count == layerCount }` guard (verified RED via the crash, then GREEN after the fix) and the TaskGroup-based concurrent test above. Third finding (new test file vs. the task text's literal "extend PromptCacheChunkTests.swift") was judged a non-functional deviation — mirrors how `PromptCacheChunkStoreTests.swift` already splits related-but-distinct chunk-store concerns into their own file; no action taken.

    **Verification** (all fresh, this session):
    - `swift build`: clean (only pre-existing unrelated warnings).
    - `swift test --filter 'PromptCache'`: 88 tests / 17 suites, all pass.
    - Full unfiltered safe-pattern suite (`xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` then `xcrun xctest` on the built `MLXFoundationModelsTests.xctest`, no filter): 195 tests / 38 suites, all pass, zero failures.

    Task left in `doing` per instructions; no commit made.
  timestamp: 2026-07-10T14:44:12.027241+00:00
- actor: claude-code
  id: 01kx69cr9ygvvxxq5x4xag80an
  text: |-
    Follow-up: strengthened the two weak `assemble()` single-chunk-aliasing regression tests per the verifier's finding that they never actually caught the bug (both passed even with `ownedCopy(of:)` temporarily removed).

    **Root cause of the weakness**: `MLXArray.concatenated(...)` always constructs a brand-new Swift `MLXArray` wrapper object regardless of whether the underlying MLX C++ buffer is aliased -- so `ObjectIdentifier` comparison is always true and proves nothing. Separately, `liveKeys[...] = zeros` (subscript assignment) builds a NEW array via a functional/scatter-style update and never mutates the original buffer in place, so the "mutate source, check assembled" test never touched the shared buffer either.

    **Investigation**: Explored all three hinted approaches.
    - Direct raw-pointer access via `Cmlx`'s `mlx_array_data_uint8` was NOT feasible: `Cmlx` is a `Target` in the vendored `mlx-swift` package but is NOT exposed as a `products:` entry in that package's `Package.swift` (only `MLX`, `MLXRandom`, `MLXNN`, etc. are) -- SwiftPM only allows depending on declared products across package boundaries, so our test target cannot `import Cmlx` directly.
    - Found a fully public path instead: `MLXArray.asData(access: .noCopy)` (`Source/MLX/MLXArray+Bytes.swift` in the vendored package) wraps `mlx_array_data_uint8(ctx)`'s raw pointer directly into a `Data(bytesNoCopy:count:deallocator:.none)` with NO copy -- this reaches past the Swift wrapper to the actual underlying C++ buffer address, entirely through already-public API (the same `asData` family `ownedCopy(of:)` itself already uses for `.copy`).
    - Memory-accounting (`MLX.Memory.activeMemory`, the `wx4w28j`-precedent approach) would also have worked but the raw-pointer approach is more precise/deterministic (no threshold tuning, no flakiness), so I used it instead.

    **Changes** (`Tests/MLXFoundationModelsTests/PromptCacheAssembleTests.swift`):
    - Added `rawBufferAddress(of:)` helper: compares actual underlying buffer addresses via `asData(access: .noCopy).data.withUnsafeBytes { baseAddress }`.
    - Added `mutateFirstElementInPlace(of:)` helper: writes directly through that same no-copy pointer, bypassing MLX's functional update path entirely, to genuinely mutate the source's backing buffer in place.
    - Replaced `singleChunkAssemblyProducesIndependentBuffers` (ObjectIdentifier-based) with `singleChunkAssemblyAllocatesIndependentBuffer` (raw-buffer-address-based).
    - Replaced `mutatingSourceChunkAfterAssemblyLeavesAssembledCacheUnchanged` (subscript-assignment-based) with `mutatingSourceChunkBufferInPlaceAfterAssemblyLeavesAssembledCacheUnchanged` (in-place-write-based).

    **RED/GREEN evidence** (this session, fresh):
    - GREEN (current fixed code): `swift build` clean; `swift test --filter 'PromptCache'` -- 88/88 tests pass, including both new tests.
    - RED: temporarily edited `PromptCache.assemble` to drop `ownedCopy(of:)` from both `keys`/`values` (reproducing the exact naive pre-fix bug: `let keys = concatenated(...)` / `let values = concatenated(...)` with no wrapping copy). Rebuilt, reran `swift test --filter 'PromptCache'`: **4 issues, both new tests failed** --
      - `singleChunkAssemblyAllocatesIndependentBuffer`: `Expectation failed: rawBufferAddress(of: assembledCache.state[0]) != rawBufferAddress(of: chunks[0].layers[0].keys)` and the same for `state[1]`/`values`.
      - `mutatingSourceChunkBufferInPlaceAfterAssemblyLeavesAssembledCacheUnchanged`: `Expectation failed: assembledKeysAfter == assembledKeysBefore` and the same for values.
      - All 86 other tests still passed (only the two strengthened tests caught the regression, exactly as intended).
    - GREEN again: reverted the temporary edit (restored `ownedCopy(of: concatenated(...))` for both keys/values), rebuilt, reran: 88/88 pass again.
    - Full unfiltered safe-pattern suite (`xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` then `xcrun xctest` on the built `.xctest`, no filter): 195 tests / 38 suites, all pass, zero failures.

    Task left in `doing`; no commit made, per instructions.
  timestamp: 2026-07-10T15:12:31.550173+00:00
depends_on:
- 01KX3MJKYYNXQ8CK197WX4W28J
position_column: done
position_ordinal: '9880'
title: 'Assembly: build a fresh private KVCacheSimple stack from matched chunks'
---
## What
Nonisolated static in Libraries/MLXFoundationModels/PromptCache.swift: `assemble(chunks: [StoredChunk], layerCount: Int) -> [KVCache]` — for each layer, `concatenated([...], axis: 2)` the chunks' K and V slices and install via a fresh `KVCacheSimple`'s `state` setter (offset derives from keys.dim(2) — verified: the setter does `self.offset = self.keys!.dim(2)`). The result is a PRIVATE cache: chunks are immutable, every resolve gets its own assembled copy, so there is no checkout, no steal, no double-checkout hazard by construction. Zero matched chunks ⇒ return `model.newCache(parameters:)` and feed everything (existing fallback path).

Assembly must be byte-exact: an assembled prefix must equal the source cache's own state slices element-for-element (this is what preserves the PromptCacheEquivalenceTests guarantee downstream).

## Acceptance Criteria
- [ ] Assembled per-layer keys/values are element-equal to the original cache's state sliced to the matched token count (CPU-device test)
- [ ] Assembled cache offset == matched token count; feeding continues from there via the normal update() path (simulate one update() and verify offset advances and returned slices include the assembled prefix)
- [ ] Two concurrent resolves for the same prefix return distinct cache instances backed by the same stored chunks (instance identity differs; content equal)

## Tests
- [ ] Extend Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift: round-trip slice→store→assemble equality; offset correctness; post-assembly update() behavior; distinct-instances test
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.