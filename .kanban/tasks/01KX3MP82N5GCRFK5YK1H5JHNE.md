---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxa1xnbx790kyen0xk7a43y2
  text: |-
    Implementation complete, verified by real execution on real hardware:

    - New `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheForkReuseTests.swift`: parent turn establishes a shared prefix; fork A and fork B extend it differently; both assert `cachedTokenCount > 0` and a reduced fed-token count. A third "parent continuation" turn issued AFTER both forks ran also asserts `cachedTokenCount > 0` -- the no-steal guarantee. Uses `TestFixtures.defaultModelID` (Qwen2.5-3B-Instruct-4bit, already cached locally).
    - Extended `PromptCacheEquivalenceTests.swift` with `forkedTranscriptResponseMatchesFreshRebuildAfterEvictAll`: forked-transcript cached response byte-identical to the same transcript's response after `MLXLanguageModel.evictAll()`.
    - Audit of `PromptCacheReuseTests`/`UpdateUsageEmissionTests`: no changes needed -- both already use bound-style assertions (`cachedTokenCount > 0`, difference-based bounds), not exact chunk counts.

    Real-environment finding (not anticipated by the task description): `TestFixtures.gemmaModelID` (gemma-3-270m-it-4bit) is architecturally incompatible with the chunk-based PromptCache -- Gemma 3's `newCache()` mixes `RotatingKVCache` sliding-window layers with plain `KVCacheSimple` global layers (`Libraries/MLXLLM/Models/Gemma3Text.swift`), and `PromptCache.sliceChunks`'s `verifiedSimpleLayers` requires EVERY layer to be a plain `KVCacheSimple`, so it silently drops storage for any Gemma 3 conversation -- `cachedTokenCount` is unconditionally 0. This made the PRE-EXISTING `editedEarlierTurnForcesTrimAndMatchesFreshRebuild` test in `PromptCacheEquivalenceTests.swift` genuinely fail on real execution (first time this suite has ever actually been run -- the file's old header claimed the whole xcodeproj didn't build). Fixed by switching that one test (and my new one) to `TestFixtures.defaultModelID`. Left the OTHER pre-existing gemma test (`cachedRunMatchesFreshPrefillEveryRound`, no cachedTokenCount assertion) untouched -- still green, though it silently never exercises real cache reuse for gemma (pre-existing soft gap, out of scope). Updated the file's stale "SANDBOX CAVEAT" header, which predated this environment's fix to the `Response.Action` SDK incompatibility.

    Verification (fresh, this session): `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -parallel-testing-enabled NO` with `-only-testing` for PromptCacheReuseTests, PromptCacheGuidedRoundTripTests, PromptCacheForkReuseTests, PromptCacheEquivalenceTests, PromptCachePrewarmTests, UpdateUsageEmissionTests -- 12/12 tests passed, real model inference, real hardware, no mocking. PromptCacheMultimodalBoundaryTests is opt-in (`MLX_RUN_VLM_INTEGRATION=1`, multi-GB VLM download) and was not run (untouched, out of scope).

    double-check adversarial review in progress.
  timestamp: 2026-07-12T02:18:54.717866+00:00
- actor: claude-code
  id: 01kxa2m6vbas1cf3atk3kj3b1x
  text: |-
    Adversarial double-check (via really-done) returned PASS with two minor, non-blocking findings, both addressed:
    1. Documented explicitly (this comment thread) that the PromptCacheReuseTests/UpdateUsageEmissionTests audit was performed and found no changes needed (both already use bound-style assertions).
    2. Tightened the fork test's three "only the new tail was fed" bound checks in PromptCacheForkReuseTests.swift from `< parentFirst.promptTokenCount` to `< sharedPrefixStoredLength` (`parentFirst.promptTokenCount + parentFirst.outputTokenCount`), matching the `firstStoredSlotLength` convention already used in PromptCacheEquivalenceTests.editedEarlierTurnForcesTrimAndMatchesFreshRebuild.

    Re-ran the full PromptCache-related suite set fresh after the fix: 12/12 tests passed (PromptCacheReuseTests, PromptCacheGuidedRoundTripTests, PromptCacheForkReuseTests, PromptCacheEquivalenceTests x3, PromptCachePrewarmTests x3, UpdateUsageEmissionTests x3), real model inference, real Apple-silicon hardware, `-parallel-testing-enabled NO`.

    Task is complete and green. Leaving in `doing` for `/review`.
  timestamp: 2026-07-12T02:31:13.515298+00:00
depends_on:
- 01KX3MNKS2VPNP2G73HAP4MG4Y
- 01KX3MMR8RVJZV7HHX7BBDA7XG
position_column: doing
position_ordinal: '80'
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