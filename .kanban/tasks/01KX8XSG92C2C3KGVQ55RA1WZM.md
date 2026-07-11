---
assignees:
- claude-code
position_column: todo
position_ordinal: a280
title: Store the partial tail chunk and slice at assembly — short conversations currently get ZERO caching
---
## What
Real-model integration runs (2026-07-11, Qwen2.5-3B-4bit) prove the chunk-only PromptCache fails the core goal for short transcripts: PromptCacheReuseTests' second round fed 58/58 tokens (first round was 38 + ~8 generated = 46 stored tokens → ZERO full 64-token chunks sliced → nothing to reuse), and both PromptCachePrewarmTests forks report cachedTokenCount == 0 (parent instructions transcript < 64 tokens → prewarm stores nothing). Unit suites all pass because they craft token counts above the chunk floor — the floor itself is the bug: any conversation prefix below chunkSize gets no caching at all, and every conversation permanently re-prefills its sub-chunk tail (up to 63 tokens) every turn.

Fix in Libraries/MLXFoundationModels/PromptCache.swift (+ PromptCacheChunks.swift): in addition to full chunks, store the VARIABLE-LENGTH partial tail as a final chunk (keyed like any chunk: hash(parentKey, tailTokens), token-verified on match). At lookup, after walking full chunks, match the stored tail by longest common prefix with the remaining new tokens; at assembly, SLICE the tail's tensors to the matched prefix length — chunks stay immutable (assembly already copies, so copy-time slicing costs nothing extra). Keep the cap: never cover the entire prompt (≥1 token must remain to feed). Dedup note: a growing conversation replaces its tail every turn (old tails become garbage quickly) — make tails cheap to evict (they already carry byteSize/LRU) and dedup identical tails as today.

This is SGLang's radix-node semantics (variable-length nodes) applied to our hash-chain design.

## Acceptance Criteria
- [ ] Unit: storing a 40-token conversation at chunkSize 64 yields a reusable prefix — next resolve feeds only the new suffix (cap-at-count-1 respected)
- [ ] Unit: divergence INSIDE the tail assembles only the common prefix of the tail (slice-at-assembly), byte-exact vs source
- [ ] Integration: PromptCacheReuseTests second round promptTokenCount/cachedTokenCount proves suffix-only prefill (currently failing — this is the regression test)
- [ ] Integration: both failing PromptCachePrewarmTests pass (fork's first round cachedTokenCount > 0)
- [ ] Byte accounting includes tails; eviction covers them

## Tests
- [ ] New unit tests in Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift (tail store/lookup/slice/dedup/eviction)
- [ ] `swift test --filter 'PromptCache'` green
- [ ] `xcodebuild test-without-building ... -only-testing:IntegrationTestingTests/PromptCacheReuseTests -only-testing:IntegrationTestingTests/PromptCachePrewarmTests` → all pass (commands in scratchpad log integration-run1.log; build needs -skipPackagePluginValidation, run needs -parallel-testing-enabled NO)

## Workflow
- Use `/tdd` — the failing integration tests above are the RED state; write the unit-level failing tests first, then implement.