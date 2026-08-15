---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxcbc7j0cx73a63f0j68asx4
  text: |-
    Implementation complete, all real-model runs green. Summary per item:

    1. MAGNITUDE (PromptCacheReuseTests.swift, PromptCacheForkReuseTests.swift, PromptCachePrewarmTests.swift): replaced every `cachedTokenCount > 0` (and the looser derived `promptTokenCount - cachedTokenCount < X` checks) with `cachedTokenCount >= sharedPrefixTokens - slack`, where sharedPrefixTokens = round1.promptTokenCount + round1.outputTokenCount (the real stored-slot yardstick). Slack values are real-run-calibrated, not guessed: `-1` for Reuse/ForkReuse (empirically measured EXACT match, `-1` covers the documented `.trimCacheByOne` edge case only). For Prewarm, added a `measuredPromptTokenCount` helper that renders the parent transcript through the exact same TranscriptConverter+UserInput pipeline `populatePromptCacheChunks` uses, for a precise (not estimated) shared-prefix count; slack=4, calibrated against a real measured gap of exactly 2 tokens (the chat template's trailing generation-prompt tokens that a fork's own continuation never reproduces at that position — confirmed via a temporary slack=0 probe run that intentionally failed to observe the true gap before finalizing the value).

    2. TOOL-CALLING / REASONING (new file PromptCacheToolReasoningReuseTests.swift): two new tests. Tool-calling test hit a real, pre-existing, unrelated bug along the way — both `TestFixtures.defaultModelID` and `Qwen3-1.7B-4bit` drove a free-form `@Guide(description:)` string tool-arg field into a degenerate token-repetition loop (confirmed this is not test-specific: `FoundationModelsToolCallingTests.toolAwarePromptRoutesWeatherQueryToGetWeather`, untouched, currently fails the same way). Worked around by constraining the schema to `@Guide(.anyOf(["Tokyo","Paris","New York"]))` — out of scope to fix the underlying bug, but worth a follow-up task. Reasoning test required a different yardstick: `TranscriptConverter` never replays `.reasoning` entries into chat history, so round 2 cannot possibly reuse the `<think>` chain — padded the prompt past one `chunkSize` (64) and bound reuse to `floor(promptTokenCount/chunkSize)*chunkSize`, the number of tokens guaranteed to sit in a chunk boundary entirely within the shared, verbatim-replayed prompt.

    3. MULTIMODAL (PromptCacheMultimodalBoundaryTests.swift): attempted for real with `MLX_RUN_VLM_INTEGRATION=1` (via an edited .xctestrun since xcodebuild doesn't forward shell env to the xctest process automatically). Confirmed real download progress past 2.7GB of the Qwen3-VL-4B-Instruct-4bit weights, but the suite's own `.timeLimit(.minutes(10))` fired before the 3-round test finished (download+load+3 real generation rounds too slow for that budget on this environment's network throughput). Replaced the stale "SANDBOX CAVEAT" header with an "EXPLICITLY UNTESTED" header documenting this real, specific attempt and result rather than the old, now-inaccurate excuse.

    4. STALE CAVEAT: removed from PromptCachePrewarmTests.swift, replaced with a real verification note.

    Verification: full combined run `xcodebuild test` across all 7 PromptCache*Tests suites (single-worker, `-parallel-testing-enabled NO`) — 12 tests, 7 suites, all passed (multimodal suite correctly reports SKIPPED, not silently omitted). `swift test --filter 'PromptCache'` — 96 tests, 14 suites, all passed (unaffected, no production code touched). Adversarial double-check review dispatched; awaiting result before final handoff.
  timestamp: 2026-07-12T23:42:38.144079+00:00
- actor: claude-code
  id: 01kxcbq7bhqzm6tdds1gvfp229
  text: |-
    Adversarial double-check: PASS, no blocking findings. Verified slack values are non-vacuous against the documented measured gaps, verified the reasoning test's floor-to-chunk-boundary formula against the real PromptCache.swift/PromptCacheChunks.swift/TranscriptConverter.swift source (not taken on faith), confirmed no naming-convention violations, confirmed the new file's local duplication (WeatherArgs/weatherTool/ToolCallAndUsage) matches this directory's established per-file-self-contained convention, confirmed the multimodal header's "EXPLICITLY UNTESTED" text is honest, and independently re-ran `xcodebuild build-for-testing` for PromptCacheToolReasoningReuseTests (TEST BUILD SUCCEEDED). One minor non-blocking note: the tool-calling test doesn't force greedy sampling, consistent with existing directory convention (ToolCallingReasoningTests), and a missing tool call fails loudly via #require rather than passing silently.

    Task is green and ready for /review. Leaving in doing per /implement convention (review step moves the column).
  timestamp: 2026-07-12T23:48:38.385844+00:00
position_column: done
position_ordinal: b380
title: 'Strengthen cache-reuse integration assertions: magnitude bounds, tool/reasoning-round coverage, gated multimodal run'
---
## What
Third adversarial verification round (2026-07-12, agent ad3d19b114f62f6d5) confirmed the cache works but found the integration assertions prove less than we claim. Four gaps:

1. MAGNITUDE: PromptCachePrewarmTests (`forkedSessionFirstRoundReusesPrewarmedParentPrefix`, `prewarmingSameTranscriptTwiceStaysHealthy`) assert only `cachedTokenCount > 0` -- one cached token out of 58 would pass. PromptCacheReuseTests' growth-exceeding bound is better but still loose (≈half-prefix reuse could pass). Fix: capture the shared-prefix/first-round token count and assert `cachedTokenCount >= sharedPrefixTokens - slack` where slack = the known partial-tail remainder (or one chunkSize as a conservative bound). Apply the same tightening to PromptCacheForkReuseTests.

2. TOOL-CALLING / REASONING ROUNDS: zero prompt-cache integration coverage exercises tool-calling or reasoning (`<think>`) rounds (grep across all PromptCache integration files: no hits). Those executor paths commit via the token-ID `commitPromptCache` overload and the two-phase think-then-call flow -- add at least one real-model test each asserting round-2 `cachedTokenCount > 0` (magnitude-bounded per item 1) after a tool round and after a reasoning round (reasoning needs a reasoning-capable model, e.g. the Qwen3 id already used by ToolCallingReasoningTests).

3. MULTIMODAL BOUNDARY: PromptCacheMultimodalBoundaryTests is gated on `MLX_RUN_VLM_INTEGRATION=1` and has NEVER been executed -- the passing "multimodal skip" prewarm test only proves prewarm doesn't throw, not that a multimodal round leaves neighboring text-turn caches undisturbed. Run the gated suite once with a VLM and record the result; if it cannot run in this environment, document it as explicitly untested in the suite header (not silently skipped).

4. STALE CAVEAT: PromptCachePrewarmTests.swift's header still says "SANDBOX CAVEAT: unverified by execution ... not run to a passing result" -- it has now passed repeatedly with a real model (2026-07-11 20:19 log; re-run at c624207). Delete/update the caveat.

CONTEXT -- what IS now verified at integration level (real model, HEAD c624207, all TEST EXECUTE SUCCEEDED): equivalence 3/3 (byte-identical cached-vs-fresh across a 4-turn conversation, edited-turn divergence, and a forked transcript across model reload), fork reuse, guided round-trip, reuse round 2, prewarm fork rounds; unit 96 tests/14 suites.

## Acceptance Criteria
- [x] Prewarm/fork/reuse integration tests assert a magnitude lower bound tied to the measured shared prefix, not just > 0
- [x] New real-model integration tests: tool-calling round 2 reuse and reasoning round 2 reuse, magnitude-bounded
- [x] Multimodal boundary suite either has a recorded passing run (MLX_RUN_VLM_INTEGRATION=1) or a documented explicitly-untested status -- attempted for real (download confirmed past 2.7GB), infeasible within the suite's own 10-minute timeLimit in this environment; documented honestly in the suite header, not silently skipped
- [x] Stale sandbox caveat removed from PromptCachePrewarmTests.swift
- [x] All PromptCache integration suites pass at the then-current HEAD (single-worker xcodebuild invocation as documented in ^5ra1wzm) -- 12 tests/7 suites green (multimodal correctly SKIPPED)

## Tests
- [x] The strengthened/new integration tests themselves; `swift test --filter 'PromptCache'` stays green -- 96 tests/14 suites green

## Workflow
- Use `/tdd` -- tighten one assertion at a time and watch it fail against a deliberately weakened bound before trusting it.

Context you need:
- This task lives in IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/ — the relevant files are PromptCacheReuseTests.swift, PromptCachePrewarmTests.swift, PromptCacheForkReuseTests.swift, PromptCacheEquivalenceTests.swift, PromptCacheMultimodalBoundaryTests.swift, and likely ToolCallingReasoningTests.swift (for the reasoning-capable model ID reference).
- This is a real-model integration-test target — this session confirmed the xcodeproj target builds and tests actually run for real via `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -parallel-testing-enabled NO` against real downloaded models on real Apple-silicon hardware. Prefer running real tests over speculative changes wherever feasible.
- IMPORTANT lesson from a prior failed attempt on a DIFFERENT task in this session: do NOT wander through open-ended exploration (e.g. `find /` searches, broad greps across the whole repo) — that caused a prior implementer to hang for 13 hours with zero progress. Go directly to the specific files named above with targeted `grep -n` and `Read`. If something is taking an unusually long time or a command seems to hang, do not retry the same broad approach — narrow your search immediately.
- Item 3 (multimodal, MLX_RUN_VLM_INTEGRATION=1) requires downloading a VLM model — this may not be feasible in this environment (disk space, model availability, time). If you cannot run it, follow the acceptance criterion's own fallback: "if it cannot run in this environment, document it as explicitly untested in the suite header (not silently skipped)" — don't block on this, don't spend excessive time trying to make it work if it's clearly infeasible (e.g. no VLM model already cached locally).
- Governing conventions for this whole session: uppercase ID acronyms (modelID, entryID, tokenID -- never modelId), `Set<String> = []` literal idiom over `Set<String>()`.
- Do NOT do any bonus refactoring beyond this task's scope. Stay within the IntegrationTesting test files (and maybe a doc-comment update) named above.

When done, report back: what you changed per each of the 4 numbered items, what you ran, pass/fail results, and honestly document if item 3 (multimodal) was infeasible to run and why.