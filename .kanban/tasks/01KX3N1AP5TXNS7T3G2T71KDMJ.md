---
assignees:
- claude-code
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
position_column: todo
position_ordinal: '9280'
title: Rewrite resolve/eviction-scope/concurrency suites to chunk semantics
---
## What
Full replacement of the unit suites deleted at cutover (split out of the cutover task to keep it reviewable):
- Tests/MLXFoundationModelsTests/PromptCacheResolveTests.swift (new content): extension turn feeds only the post-chunk-boundary suffix; divergence mid-chunk feeds from the last shared chunk boundary; identical prompt feeds exactly the capped remainder (≥1 token, never the whole prompt); unchunkable cache (RotatingKVCache layer) stores nothing and the next resolve rebuilds; assembled-cache offset correctness after a simulated turn.
- Eviction-scope suite (evictAll drops all models; remove is per-model; identical tokens never cross model IDs) rewritten over chunk storage — same behavioral guarantees as the deleted PromptCacheSlotPoolTests eviction-scope suite.
- Concurrency suite: concurrent resolves for the same prefix return DISTINCT assembled instances (identity) with equal content; sustained parallel resolve/store churn on one modelID never corrupts store bookkeeping (adapt the CheckoutTracker test — double-checkout is now impossible by construction, assert distinct instances instead).
- Reuse PromptCacheTestSupport.swift fixtures (probe model, resolveOnce, CPU-device pattern); update helpers that reference deleted slot semantics.

## Acceptance Criteria
- [ ] Every behavioral guarantee from the deleted suites has a chunk-semantics equivalent (reuse, divergence, identical-prompt, unchunkable degradation, evictAll/remove/isolation, concurrency safety)
- [ ] Assertions are deterministic (token counts / element equality / instance identity — no timing dependence)

## Tests
- [ ] The suites above; `swift test --filter 'PromptCache'` green; `swift test --filter MLXFoundationModelsTests` zero failures

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.