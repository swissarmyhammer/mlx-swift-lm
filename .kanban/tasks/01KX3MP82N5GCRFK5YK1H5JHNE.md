---
assignees:
- claude-code
depends_on:
- 01KX3MNKS2VPNP2G73HAP4MG4Y
- 01KX3MMR8RVJZV7HHX7BBDA7XG
position_column: todo
position_ordinal: '9080'
title: 'Integration: fork reuse and byte-identical equivalence end-to-end with a real model'
---
## What
In IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/ (real model, runs via xcodebuild on-device — not swift test):
- New PromptCacheForkReuseTests.swift: run a parent conversation turn; build TWO transcripts extending it differently (the fork shape FoundationModelsRouter's makeFork produces — transcript-seeded LanguageModelSession); respond on both. Assert BOTH report cachedTokenCount > 0 / reduced prompt-fed counts, AND a subsequent parent turn is also still cached (the no-steal guarantee, which was false under the slot design).
- Extend PromptCacheEquivalenceTests.swift: a forked-transcript response must be byte-identical to the same response after MLXLanguageModel.evictAll() (fresh rebuild) — extends the existing greedy-equivalence pattern in that file.
- Audit PromptCacheReuseTests + UpdateUsageEmissionTests expectations: cachedTokenCount is now CHUNK-ALIGNED (≤ true prefix, short by up to chunkSize-1 + the always-fed last token) — update assertions from exact counts to bounds where needed.

## Acceptance Criteria
- [ ] Fork test: parent + both forks all show prefix reuse in usage events; parent reuse survives the forks
- [ ] Equivalence: forked cached output == evicted fresh output, byte-identical (greedy sampling)
- [ ] Existing PromptCache* integration suites pass with chunk-aligned expectations

## Tests
- [ ] New PromptCacheForkReuseTests.swift + extended PromptCacheEquivalenceTests.swift as above
- [ ] `xcodebuild test -scheme <IntegrationTesting scheme> -only-testing:...PromptCache...` green (document exact invocation in the test file header, mirroring existing suites)

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.