---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3b465gvsvzmtt27ad95eat
  text: 'Implementation landed: Tests/MLXFoundationModelsTests/PromptCacheResolveTests.swift (7 actor-level resolve() tests covering reuseSuffix>0, reuseSuffix==0 n_past-- regeneration, trimTo with/without suffix, non-trimmable rebuild fallbacks, failed-trim-verification fallback, checkout-removes-slot). Shared fixtures extracted to PromptCacheTestSupport.swift; PromptCacheConcurrencyTests now uses the shared PromptCacheProbeModel instead of its private copy. BLOCKER for test run: a concurrent session has Libraries/MLXFoundationModels/MLXLanguageModel.swift mid-edit (preparedInputMappingImageFailures refactor, call site at ~line 1290 missing new transcriptEntries arg) so the package temporarily doesn''t compile — waiting for their edit to land before running the suite. Test files themselves are complete.'
  timestamp: 2026-07-09T11:45:04.688811+00:00
- actor: claude-code
  id: 01kx3c8k2xtrrbsmpsfypqw8sn
  text: 'Done and green. PromptCacheResolveTests.swift now has 8 tests: all applyDecision branches through resolve(), including the double-check reviewer''s requested case (identical prompt + trimmable-but-mispositioned multi-layer cache → regenerateLastToken''s trimAndVerify failure → rebuild; also the only multi-layer exercise of trimAndVerify''s allSatisfy). Identity assertions compare against live instances to rule out ObjectIdentifier address-recycling false positives. Verified: swift test --filter ''PromptCache'' → 47 tests / 12 suites pass; PromptCache.swift 100% line+region+function coverage (was 79.4%); double-check adversarial review returned one finding, fixed. Ready for /review.'
  timestamp: 2026-07-09T12:04:57.565791+00:00
position_column: doing
position_ordinal: '8180'
title: Add actor-level resolve() tests covering regenerateLastToken and every applyDecision branch
---
Libraries/MLXFoundationModels/PromptCache.swift — uncovered lines 220-227 (regenerateLastToken), 242-246 (.reuseSuffix count==0), 255 (successful .trimTo reuse with suffix), 258-264 (.trimTo thenSuffix==0, shrunk prompt), 267 (applyDecision .rebuild for a non-trimmable diverging cache).

Coverage: 79.4% (108/136 lines) — these branches are the gap.

Write unit tests in Tests/MLXFoundationModelsTests (new file, e.g. PromptCacheResolveTests.swift) that drive the actor end-to-end via store() + resolve() with KVCacheSimple instances whose .offset is set to match the stored token count (so trimAndVerify succeeds):

1. Identical prompt resent (reuseSuffix count==0): store slot with offset==tokens.count, resolve with the SAME tokens → returns the SAME cache instance (ObjectIdentifier), cache.offset trimmed to count-1, tokensToFeed == [last token].
2. regenerateLastToken fallback: same scenario but a non-trimmable cache → rebuild (fresh instance, all tokens fed).
3. Successful trim-reuse: store slot offset==tokens.count, resolve with tokens diverging after k → same instance, offset==k, tokensToFeed == new suffix.
4. Shrunk prompt (thenSuffix==0): resolve with a strict prefix of the stored tokens → same instance, offset==prefix.count-1, tokensToFeed == [prefix.last].
5. applyDecision .rebuild: non-trimmable cache (test conformer whose isTrimmable/trim is unsupported — see canTrimPromptCache in MLXLMCommon) diverging after k>0 → fresh cache, all tokens fed.

Follow the existing style of PromptCacheConcurrencyTests.swift (ConcurrencyProbeModel stand-in, SendableBox one-per-call, #if FoundationModelsIntegration gates).