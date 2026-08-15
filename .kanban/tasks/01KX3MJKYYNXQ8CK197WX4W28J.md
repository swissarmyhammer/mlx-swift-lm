---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx4fd7ev29m7vbpwkvtj4y7g
  text: 'Correction from user review: dependency task ^23ff1zx was revised from "pin CPU device" to a ROOT FIX (symlink mlx.metallib beside the xctest binary so GPU eval works under plain swift test). Where this task''s description says "(CPU-device test)", read "on the default device via the metallib bootstrap from ^23ff1zx" — no CPU pinning.'
  timestamp: 2026-07-09T22:19:09.659037+00:00
- actor: claude-code
  id: 01kx6161rd3wxtvhpn03dfp1m2
  text: |-
    Implementation complete, TDD throughout, adversarial review completed (2 rounds).

    ## Files
    - `Libraries/MLXFoundationModels/PromptCacheChunks.swift` (new): `PromptCache.StoredChunk`, `PromptCache.rootChunkKey`, `PromptCache.chunkKey(parentKey:tokens:)`, `PromptCache.sliceChunks(tokens:cache:chunkSize:)`.
    - `Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift` (new): 14 tests, `@Suite("PromptCache chunk slicing", .serialized)`.

    ## Investigation highlights
    - `KVCacheSimple.state` getter returns `[keys, values]` directly (the SAME object, not a copy) when `offset == keys.dim(2)`; otherwise a sliced view. `sliceChunks` requires the former (mirrors the task's "verified" shape).
    - `MLXArray.nbytes`/`itemSize` (in mlx-swift's `MLXArray.swift`) are the real byte-count APIs; `nbytes` reports only *logical* size (shape × itemSize), NOT real retained buffer size -- this matters for the ownership investigation below.
    - **Ownership mechanism (the crux)**: read mlx-swift's `Source/Cmlx/mlx/mlx/backend/common/slicing.cpp` -- MLX's `Slice` primitive ALWAYS evaluates via `shared_buffer_slice`/`copy_shared_buffer`, an unconditional zero-copy view sharing the source's entire underlying allocator buffer, regardless of contiguity. Calling `.eval()` on a slice does NOT copy data -- `array::detach()` (`array.cpp`) only clears the compute-graph `inputs_`, orthogonal to the buffer the array's data still points into. So a naive "slice, then eval" chunk keeps the ENTIRE source cache's buffer alive for as long as the chunk lives, invisible to both content-equality checks and to `.nbytes`. Fix: round-trip through `array.asData(access: .copy)` (docs guarantee an unconditional independent copy) → `MLXArray(data:)`, which builds a fresh `mlx_array` via `mlx_array_new_data` with no shared buffer.
    - **RED evidence for the crux**: temporarily reverted `ownedCopy` to a naive `array.eval(); return array`, ran `chunkCopiesReleaseSourceBufferMemory` (slices one 8-token chunk from a 200,000-token cache, drops the source, measures `MLX.Memory.activeMemory`) -- active memory grew ~12.8MB, matching the full source cache's footprint almost exactly (confirming the entire buffer stayed retained). Restored the real `asData(access:.copy)` fix -- same test passed with memory growth far below the threshold. This is the actual RED/GREEN pair for the task's own "CRITICAL -- owned copies" requirement.

    ## Adversarial review findings (fixed)
    First `double-check` round found 3 issues (fixed with new RED→GREEN regression tests each):
    1. **HIGH**: `layer as? KVCacheSimple` does not exclude `ChunkedKVCache` (a subclass overriding `update`/`trim`/`copy`/`metaState` but NOT `state`/`isTrimmable`) -- once `maybeTrimFront()` trims physically while `offset` keeps the full logical count, the inherited `state` getter's `offset == keys.dim(2)` branch breaks, risking a wrong-extent slice. Fixed with `type(of: layer) == KVCacheSimple.self` (exact dynamic-type check) before the cast. Test `chunkedKVCacheLayerReturnsNil`.
    2. **LOW**: empty `cache` array silently produced zero-layer chunks instead of `nil`. Fixed with `!cache.isEmpty` guard. Test `emptyCacheArrayReturnsNil`.
    3. **LOW**: the memory-based ownership test wasn't `.serialized` despite reading process-global `MLX.Memory.activeMemory`. Added `.serialized` to the suite.

    Second `double-check` round confirmed all 3 fixes correct and sound, but flagged that `.serialized` on a single suite (per `ModelCacheEvictionTests.swift`'s own documented caveat) doesn't prevent *other* top-level suites in the target from running concurrently with this one's memory measurement. Verified empirically (grepped every `MLXArray`/`.zeros` use across the whole `MLXFoundationModelsTests` target): every other suite only builds trivial single-element or small-token-count arrays, orders of magnitude below this test's ~6MB safety margin -- so the residual cross-suite race is real in principle but negligible in practice given the current test suite composition. Fully closing it would mean nesting `PromptCacheChunkTests` under a shared serialized parent alongside unrelated suites in other files, which is out of this task's scope (would be modifying test infrastructure owned by other areas/tasks). Corrected the misleading "matches the precedent" comment to accurately describe what was and wasn't done, and documented this residual risk in the suite's doc comment. Proceeding per really-done's bounded-loop / logged-justification allowance (one re-check already spent).

    ## Verification (all fresh, this session)
    - `swift build`: clean.
    - `swift test --filter 'PromptCacheChunk'`: 14/14 green, serial (no interleaving).
    - `swift test --filter 'PromptCache'`: 65/65 green across 15 suites (task's own literal acceptance criterion).
    - `swift test --filter 'MLXFoundationModelsTests'` (full target): 172/172 green across 36 suites.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package ... ` + unfiltered `xcrun xctest <bundle>.xctest` (never `-XCTest` filtered, never piped through `tail`): 172/172 green.

    Leaving in `doing` for review per orchestrator convention.
  timestamp: 2026-07-10T12:49:03.245995+00:00
depends_on:
- 01KX3MJ2B3ZEVW5MAKT23FF1ZX
position_column: done
position_ordinal: '9680'
title: 'KV chunk types and slicing: cut a verified KVCacheSimple stack into token-range chunks'
---
## What
In Libraries/MLXFoundationModels/PromptCache.swift (or a new PromptCacheChunks.swift alongside it), define the chunk model and the slicing function:
- `StoredChunk`: token ids for the chunk, per-layer `[(keys: MLXArray, values: MLXArray)]` slices, `parentKey`, its own `chunkKey`, `byteSize`, `lastUsed` recency stamp.
- Chunk keying: hash chain — `chunkKey = hash(parentKey, chunkTokens)` (Hasher over parent key + token ids; root parent = a fixed seed). NOTE: Hasher is per-process seeded — chunk keys must never be persisted or compared across processes.
- `sliceChunks(tokens: [Int], cache: [KVCache], chunkSize: Int) -> [StoredChunk]?` (nonisolated static, pure): every layer must be a `KVCacheSimple` with `state == [keys, values]` and offset == tokens.count, else return nil (not chunkable — mirrors today's isTrimmable degradation for Rotating/Chunked caches). Slice each layer's state arrays at `[.ellipsis, a..<b, 0...]` into FULL chunks only; the partial tail (< chunkSize tokens) is not stored.
- CRITICAL — owned copies: MLX slices are lazy views sharing the source buffer. Stored chunk tensors MUST be materialized as eval'd, contiguous, OWNED copies at slice time (e.g. explicit copy + `eval()`), otherwise (a) a stored chunk retains the entire source cache's buffers so eviction frees nothing, (b) `byteSize` undercounts real retained memory and the byte-budget task becomes fiction, (c) unevaluated graph nodes keep sources alive.
- `byteSize` from array dims × dtype size (implementer: verify MLXArray nbytes/itemSize API) — must equal the OWNED copy's real footprint.

## Acceptance Criteria
- [ ] Slicing a cache of 5×chunkSize+7 tokens yields exactly 5 chunks, tail dropped
- [ ] Chunk keys form a chain: same prefix tokens ⇒ identical keys across different conversations; divergence at chunk k ⇒ keys differ from k onward
- [ ] A stack containing one RotatingKVCache layer returns nil
- [ ] Chunk tensors are element-equal to the corresponding slices of the source cache (CPU-device test)
- [ ] Chunk tensors are OWNED: mutating/deallocating the source cache after slicing leaves chunk contents intact, and chunks are already evaluated (no deferred graph)

## Tests
- [ ] Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift (new): boundary math, key-chain identity/divergence, non-simple-layer nil, element equality vs source slices, ownership test (overwrite source K/V in place after slicing → chunk contents unchanged)
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.