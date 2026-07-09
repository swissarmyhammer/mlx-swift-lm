---
assignees:
- claude-code
depends_on:
- 01KX3N1AP5TXNS7T3G2T71KDMJ
- 01KX3MN39J77KZTKPEM2SDT6DJ
position_column: todo
position_ordinal: 8f80
title: Multi-session and fork behavioral suites under chunk semantics
---
## What
Rewrite Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift for the chunk-only world — the assertions get STRONGER because nothing steals:
- Interleaved round-robin sessions: every session's continuation turn feeds only tokens past its last chunk boundary — for ALL sessions simultaneously, every round (previously only ≤4 could, and forks stole).
- Fork fan-out: one parent conversation, then N (≥6) forked prompts sharing the parent transcript prefix plus distinct tails — parent and EVERY fork resolve with the shared prefix served from chunks (tokensToFeed ≈ tail only), interleaved and in parallel Tasks. This is the FoundationModelsRouter fork scenario at the unit level.
- Concurrent isolation: parallel sessions get distinct assembled cache instances (identity) with equal prefix content; a mutation via one session's update() never appears in another's assembled cache (immutability of stored chunks).
- Oversubscription under a small byte budget: correctness degrades to larger tokensToFeed, never a wrong-prefix assembly (suffix-reconstruction invariant from the existing suite carries over).

## Acceptance Criteria
- [ ] All four scenarios above have passing tests with deterministic assertions (tokensToFeed counts / element equality — no timing dependence)
- [ ] Fork test explicitly named/documented as the FoundationModelsRouter scenario

## Tests
- [ ] Rewritten Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift
- [ ] `swift test --filter 'PromptCache'` green; `swift test --filter MLXFoundationModelsTests` zero failures

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.