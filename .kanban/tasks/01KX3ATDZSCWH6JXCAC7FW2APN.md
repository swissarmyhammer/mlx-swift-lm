---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3c8x1s5fp4wxyze5zf5hez
  text: 'Done and green. PromptCacheMultiSessionTests.swift: (1) round-robin interleaved sessions each reuse their OWN KV instance every turn, feeding only the new turn''s tokens; (2) concurrent sessions (task-group parallel, one per maxSlotsPerModel slot) never share a KVCache instance across sessions and each achieves reuse on every continuation turn; (3) oversubscribed round-robin (cap+2 sessions) degrades to full-prompt rebuilds, never a wrong-prefix reuse (feed-suffix reconstruction invariant asserted every turn). Sessions hold their stored cache instances live to keep identity comparisons sound. Verified in the 47-test run. Ready for /review.'
  timestamp: 2026-07-09T12:05:07.769133+00:00
position_column: doing
position_ordinal: '8480'
title: Add multi-session interleaved-conversation test proving per-session KV reuse in parallel
---
The multi-agent/multi-session scenario end-to-end at the actor level: N concurrent "sessions" (distinct system-prompt token prefixes) interleave multi-turn rounds against ONE PromptCache and one modelID — the exact shape of a multi-agent environment sharing a model.

Write a unit test in Tests/MLXFoundationModelsTests:

1. 3-4 sessions, each with a unique token prefix. Each session runs several turns; each turn does resolve() (expect: reuse of that session's OWN slot — same instance identity as that session's last store, tokensToFeed == just the new turn's tokens) then store() with the grown token sequence and advanced offset.
2. Interleave turns round-robin across sessions (session1 turn1, session2 turn1, ..., session1 turn2, ...) — proving slot selection keeps sessions from stealing each other's KV state while alternating.
3. Concurrent variant: run the sessions as parallel Tasks; assert no double-checkout (instance identities disjoint while outstanding) AND that each session's turns after the first predominantly reuse rather than rebuild.
4. Oversubscribed variant: 6 sessions (> maxSlotsPerModel=4) interleaved — must stay correct (rebuild when a session's slot was evicted, never a wrong-prefix reuse: verify tokensToFeed always reconstructs the full prompt when appended to the reused prefix).

Follow PromptCacheConcurrencyTests.swift patterns (probe model, SendableBox per call, CheckoutTracker-style actor if needed).