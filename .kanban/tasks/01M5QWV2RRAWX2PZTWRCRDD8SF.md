---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzxkzg6vv47mbarf1fz2c
  text: 'Change after the PR 581 port: a cache that cannot stage a verify block now throws `MTPInitializationError.unsupportedSpeculativeCache` before prefill, and ChatSession falls back with `SpeculativeDecodingFallbackReason.unsupportedCache`. Use this path for unsupported rotating/quantized combinations.'
  timestamp: 2026-10-25T13:46:22.719346+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SJPFM01WFRYWMQXVGMA4MQ
position_column: todo
position_ordinal: 8b80
title: 'MTP: define drafter behaviour with maxKVSize (rotating) and quantized main KV caches'
---
## What
The Qwen MTP drafter cache is a plain `KVCacheSimple` (`Qwen35MTPPredictor.newCache`, Libraries/MLXLLM/Models/Qwen35MTP.swift ~43-45, ~113-115). It never rotates or quantizes. When the main full-attention caches use `maxKVSize` (rotating) or KV quantization (`kvBits`, `KVCachePlan`), the rule "drafter = main − 1" (by `processedTokenCount`) and the drafter memory use are not defined.

- [ ] Rotating main cache: make the drafter cache rotate with the same window (`RotatingKVCache` from `makeAttentionKVCache(parameters:)`). The alignment checks compare logical token counts, not stored rows. `commitDrafterState` trims rejected entries (`trimPromptCache`, Qwen35MTP.swift ~231) and finalize trims `lookahead + 1`; after the ring wraps, `RotatingKVCache` can trim only a limited count (KVCache.swift ~1041-1054). Keep the keep-size large enough for `maximumBlockSize` entries, or change to passthrough with a reason when a trim is not possible.
- [ ] Record the cache kind and window in `MTPDrafterCacheState.arrays()` / `metadata()`, and reject a restore into a different kind.
- [ ] Quantized main cache: keep the drafter cache bf16 (it is one layer; about 2 × 4 heads × 256 × 2 B = 4 KB for each token, which is small). Record this rule in a doc comment in Qwen35MTP.swift.
- [ ] If a combination is not supported, `MTPSpeculativeTokenIterator` must change to passthrough with a clear `passthroughReason`, and must not trap or produce wrong tokens.

## Acceptance Criteria
- [ ] With `maxKVSize` smaller than the generated length, MTP greedy output equals plain greedy output with the same `maxKVSize`, the drafter cache stays bounded, and a reject after the ring wraps does not trap.
- [ ] With `kvBits = 8`, MTP greedy output equals plain greedy output with the same KV setting, or the stream changes to passthrough with a reason. A test asserts which of the two happens.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared fixture: rotating-cache equality, a reject after wrap, bounded drafter size, quantized-KV behaviour, and a restore of a rotating drafter state into a simple cache (rejected).
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache