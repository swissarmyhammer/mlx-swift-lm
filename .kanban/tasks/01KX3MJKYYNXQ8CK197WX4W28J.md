---
assignees:
- claude-code
depends_on:
- 01KX3MJ2B3ZEVW5MAKT23FF1ZX
position_column: todo
position_ordinal: '8980'
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