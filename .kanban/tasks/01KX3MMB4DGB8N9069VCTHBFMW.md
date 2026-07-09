---
assignees:
- claude-code
depends_on:
- 01KX3MKJFG9SJFW44RNMWWK2BS
position_column: todo
position_ordinal: 8c80
title: Rewire resolve()/store() to the chunk path; delete slot machinery and obsolete suites
---
## What
The production cutover, in Libraries/MLXFoundationModels/PromptCache.swift — signatures of `resolve(modelID:newTokens:model:parameters:)` and `store(modelID:tokens:cache:)` stay EXACTLY as-is (SendableBox plumbing included) so MLXLanguageModel.swift's executor wiring (resolvePromptCache/storePromptCache/commitPromptCache and both reconcile* functions) is untouched:
- `resolve`: lookupLongestPrefix → assemble → (cache, tokensToFeed = suffix past matched chunks). No slot removal — nothing is checked out.
- `store`: sliceChunks (nil ⇒ drop silently, same as today's unchunkable degradation) → insert.
- DELETE dead production code: `Slot`, `selectSlot`, `decide`, `Decision`, `applyDecision`, `regenerateLastToken`, `setMaxSlotsPerModel`, `maxSlotsPerModel`/`defaultMaxSlotsPerModel`. KEEP `reconcileGeneratedTokens`/`reconcileCacheAdvance` AND `trimAndVerify` (commitPromptCache's `.trimCacheByOne` branch in MLXLanguageModel.swift still calls trimAndVerify — it stays, with its existing tests).
- DELETE the test suites that are the spec of deleted code, so the target compiles green: PromptCacheTests.swift's decide/selectSlot suites + both property suites (KEEP the reconciliation suites and trim-and-verify suite), PromptCacheResolveTests.swift, PromptCacheSlotPoolTests.swift, PromptCacheMultiSessionTests.swift, PromptCacheConcurrencyTests.swift. Their chunk-semantics replacements are the follow-up suite tasks — write a MINIMAL green resolve/store chunk test in this task (extension turn feeds only suffix; divergence feeds from last shared chunk boundary) so the cutover is not merged untested.
- Rewrite the PromptCache doc header: llama.cpp slot-pool comparison → chunked shared store (SGLang RadixAttention-style, assemble-on-hit, contiguous-copy because MLX SDPA needs contiguous K/V).
- KNOWN WINDOW (acceptable on-branch, document in the header): between this task and the byte-budget task the store has NO capacity bound.

## Acceptance Criteria
- [ ] `git grep 'selectSlot\|applyDecision\|regenerateLastToken\|maxSlotsPerModel\|SlotLimit' Libraries/ Tests/` returns nothing
- [ ] MLXLanguageModel.swift diff for this task is limited to removing setPromptCacheSlotLimit (if not already removed by the disposition/byte-budget sequencing — coordinate; exactly one task deletes it)
- [ ] Second resolve of the same continuation still reuses (no checkout): tokensToFeed counts prove the prefix was served from chunks for BOTH calls
- [ ] Full unit target compiles and passes: `swift test --filter MLXFoundationModelsTests` — zero failures (pre-existing metallib end-crash excepted)

## Tests
- [ ] Minimal new chunk-semantics tests in this task (see What); full behavioral suites are the dependent tasks
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.