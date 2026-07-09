---
assignees:
- claude-code
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
position_column: todo
position_ordinal: 8d80
title: Configurable chunk size (default 64) with change-invalidates-store semantics
---
## What
In Libraries/MLXFoundationModels/PromptCache.swift: `static let defaultChunkSize = 64`; actor state `private var chunkSize = PromptCache.defaultChunkSize`; `func setChunkSize(_ size: Int)` clamped to ≥ 1. Changing the size makes existing chunk keys meaningless (keys are chains over fixed-width windows), so setChunkSize with a DIFFERENT value must `evictAll()` — document this. Public passthrough in Libraries/MLXFoundationModels/MLXLanguageModel.swift beside evictAll(): `public static func setPromptCacheChunkSize(_ size: Int) async`, doc: trade-off (smaller = finer fork-point granularity; larger = coarser sharing; tail past the last chunk boundary always re-prefills, so worst-case extra prefill ≈ chunkSize-1 tokens).

## Acceptance Criteria
- [ ] Store/resolve round-trips at sizes 1, 64, 256 (parameterized): each case uses > 2×chunkSize tokens and asserts ACTUAL reuse — tokensToFeed strictly smaller than the full prompt on the second resolve (a zero-chunks-stored vacuous pass must be impossible)
- [ ] setChunkSize to a new value drops all stored chunks (next resolve rebuilds); setting the SAME value does not evict
- [ ] Clamp: 0/negative → 1

## Tests
- [ ] Extend Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift: parameterized sizes with the >2×chunkSize + reuse assertion, change-evicts, same-value-no-evict, clamp (arguments: [0, -3])
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.