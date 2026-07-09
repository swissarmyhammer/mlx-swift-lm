---
assignees:
- claude-code
depends_on:
- 01KX3MJKYYNXQ8CK197WX4W28J
position_column: todo
position_ordinal: 8b80
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