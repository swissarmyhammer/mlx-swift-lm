---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'Strengthen cache-reuse integration assertions: magnitude bounds, tool/reasoning-round coverage, gated multimodal run'
---
## What
Third adversarial verification round (2026-07-12, agent ad3d19b114f62f6d5) confirmed the cache works but found the integration assertions prove less than we claim. Four gaps:

1. MAGNITUDE: PromptCachePrewarmTests (`forkedSessionFirstRoundReusesPrewarmedParentPrefix`, `prewarmingSameTranscriptTwiceStaysHealthy`) assert only `cachedTokenCount > 0` — one cached token out of 58 would pass. PromptCacheReuseTests' growth-exceeding bound is better but still loose (≈half-prefix reuse could pass). Fix: capture the shared-prefix/first-round token count and assert `cachedTokenCount >= sharedPrefixTokens - slack` where slack = the known partial-tail remainder (or one chunkSize as a conservative bound). Apply the same tightening to PromptCacheForkReuseTests.

2. TOOL-CALLING / REASONING ROUNDS: zero prompt-cache integration coverage exercises tool-calling or reasoning (`<think>`) rounds (grep across all PromptCache integration files: no hits). Those executor paths commit via the token-ID `commitPromptCache` overload and the two-phase think-then-call flow — add at least one real-model test each asserting round-2 `cachedTokenCount > 0` (magnitude-bounded per item 1) after a tool round and after a reasoning round (reasoning needs a reasoning-capable model, e.g. the Qwen3 id already used by ToolCallingReasoningTests).

3. MULTIMODAL BOUNDARY: PromptCacheMultimodalBoundaryTests is gated on `MLX_RUN_VLM_INTEGRATION=1` and has NEVER been executed — the passing "multimodal skip" prewarm test only proves prewarm doesn't throw, not that a multimodal round leaves neighboring text-turn caches undisturbed. Run the gated suite once with a VLM and record the result; if it cannot run in this environment, document it as explicitly untested in the suite header (not silently skipped).

4. STALE CAVEAT: PromptCachePrewarmTests.swift's header still says "SANDBOX CAVEAT: unverified by execution ... not run to a passing result" — it has now passed repeatedly with a real model (2026-07-11 20:19 log; re-run at c624207). Delete/update the caveat.

CONTEXT — what IS now verified at integration level (real model, HEAD c624207, all TEST EXECUTE SUCCEEDED): equivalence 3/3 (byte-identical cached-vs-fresh across a 4-turn conversation, edited-turn divergence, and a forked transcript across model reload), fork reuse, guided round-trip, reuse round 2, prewarm fork rounds; unit 96 tests/14 suites.

## Acceptance Criteria
- [ ] Prewarm/fork/reuse integration tests assert a magnitude lower bound tied to the measured shared prefix, not just > 0
- [ ] New real-model integration tests: tool-calling round 2 reuse and reasoning round 2 reuse, magnitude-bounded
- [ ] Multimodal boundary suite either has a recorded passing run (MLX_RUN_VLM_INTEGRATION=1) or a documented explicitly-untested status
- [ ] Stale sandbox caveat removed from PromptCachePrewarmTests.swift
- [ ] All PromptCache integration suites pass at the then-current HEAD (single-worker xcodebuild invocation as documented in ^5ra1wzm)

## Tests
- [ ] The strengthened/new integration tests themselves; `swift test --filter 'PromptCache'` stays green

## Workflow
- Use `/tdd` — tighten one assertion at a time and watch it fail against a deliberately weakened bound before trusting it.