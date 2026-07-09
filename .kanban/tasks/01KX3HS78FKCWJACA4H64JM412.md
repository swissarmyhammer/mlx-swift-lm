---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3jhngsxc29bhahzykgf8ed
  text: 'Done and green via strict TDD. RED: 3 new tests failed with the setter stubbed inert (verified). GREEN: store() consults actor-isolated maxSlotsPerModel; MLXLanguageModel.setPromptCacheSlotLimit(_:) public static added beside evictAll(). Double-check verdict REVISE with one minor finding (clamp test claimed negative coverage but only tested 0) — fixed by parameterizing over [0, -3]. Final: 50 PromptCache tests / 12 suites pass; full target 114 pass / 0 fail. Rationale documented in code: no session-count signal exists to derive the cap from (FoundationModels hides sessions), so capacity is declared; the router should push maxConcurrentForks into this API (see follow-up task on session-identity caching). Ready for /review.'
  timestamp: 2026-07-09T13:54:46.425201+00:00
- actor: claude-code
  id: 01kx3psfvhgkpzy6sw0gndgpbt
  text: |-
    Completed the missing acceptance-criteria coverage from the prior session: a test proving MLXLanguageModel.setPromptCacheSlotLimit(_:) (the public static passthrough) actually reaches the process-global PromptCache instance, not just the actor-level setMaxSlotsPerModel already covered by PromptCacheSlotPoolTests.

    Added Tests/MLXFoundationModelsTests/PromptCacheSlotLimitPassthroughTests.swift. Since MLXLanguageModel.promptCache is `private static let` (unreachable even via @testable import), the test drives the round-trip through MLXLanguageModel's own internal (non-private) test seams `resolvePromptCache`/`storePromptCache` -- the exact functions Executor.respond() calls in production -- instead of a fresh local PromptCache(). It: calls MLXLanguageModel.setPromptCacheSlotLimit(1), stores two slots for a UUID-unique modelID via storePromptCache, then resolves both via resolvePromptCache to prove the older slot was evicted (cache identity mismatch + full-prompt rebuild) and the newer slot survived (identity match). Restores the default limit at the end of the function body (not a `defer`, since defer bodies can't await; #expect doesn't unwind on failure so the restore still runs).

    TDD: confirmed RED by temporarily removing the setPromptCacheSlotLimit(1) call -- both assertions failed with clear messages -- then restored and confirmed GREEN.

    Double-check (adversarial review) round 1 found a real isolation gap: the suite was a top-level @Suite, unprotected from MLXLanguageModel.evictAll() (key-agnostic, wipes every model's slots in the same process-global promptCache) called by TokenizerBiasCacheTests.swift and ModelCacheEvictionTests.swift in the same bundle -- Swift Testing parallelizes independent top-level suites by default, so this was a genuine race/pollution hazard. Fixed by nesting the suite under the existing `.serialized` parent `FoundationModelsCacheTests` (same pattern TokenizerBiasCaching already uses). Round 2 double-check: PASS.

    Verification (fresh, this session): `swift build` clean. `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` succeeded. All 4 mandated bundles via unfiltered `xcrun xctest <bundle>` (each wrapped in timeout, no -XCTest filtering): CXGrammarTests 7 tests/5 suites, MLXGuidedGenerationTests 62/13, MLXFoundationModelsTests 145/31 (includes the new test, confirmed running serialized immediately after the other FoundationModelsCacheTests-nested suites), MLXLMTests 245/19 -- all green, zero failures.

    Did not touch PromptCache.swift or MLXLanguageModel.swift's existing design/implementation -- only added the test file. Left in `doing` for review per scope; not committed (orchestrator handles that).
  timestamp: 2026-07-09T15:08:57.073546+00:00
- actor: claude-code
  id: 01kx3qsws1tejhhkxnevezsxj5
  text: 'Review round (2026-07-09 10:15): 4 findings, all confirmed pre-existing/untouched by this commit (duplicated reasoning error message, hardcoded tokenCount:1 repeated 3x) — deferred to task 9jtbtkd. Re-reviewing.'
  timestamp: 2026-07-09T15:26:38.881767+00:00
- actor: claude-code
  id: 01kx3s16tz7fwee2e7qv8rbt4k
  text: 'Review round 3 (2026-07-09 10:41): 4 findings, all the same recurring duplicated-reasoning-error-message issue already tracked in task 9jtbtkd (now with more specificity: extract a shared `throwCannotDisableReasoningError(_:)` helper covering both the message constant and the catch-block pattern). Confirmed pre-existing/untouched by this commit (doc-only change). Deferred, no code change needed here.'
  timestamp: 2026-07-09T15:48:07.135885+00:00
- actor: claude-code
  id: 01kx3sey320zw0y06qpq9bxqfa
  text: 'Review round 4 (2026-07-09 10:48): 2 findings, both confirmed pre-existing/untouched by this commit (Stream.gpu.synchronize() duplication in respond()''s catch blocks; recurring reasoning error-mapping duplication). Deferred to 9jtbtkd. Re-reviewing.'
  timestamp: 2026-07-09T15:55:36.930421+00:00
- actor: claude-code
  id: 01kx3te588vfqr9y2dstps1h66
  text: 'Review round 5 (2026-07-09 10:55): 9 findings — 6 rejected as an acronym-casing contradiction (would revert the deliberate XG→Xg rename established in q3ddgqy), 3 deferred to 9jtbtkd (respond() length re-litigation, runUnconstrained nesting, recurring reasoning-message duplication), all confirmed pre-existing/untouched by this commit (doc-only). No code change needed. Re-reviewing.'
  timestamp: 2026-07-09T16:12:40.072232+00:00
position_column: done
position_ordinal: '8780'
title: Make PromptCache slot cap configurable (public API, replaces hard-coded 4)
---
User finding: the hard-coded PromptCache.maxSlotsPerModel = 4 is useless as a fixed constant, and there is NO existing session-concurrency setting to derive it from (FoundationModels hides sessions from the executor entirely; SerialAccessContainer serializes generation to 1 per container, which is the wrong axis — the cap bounds KV retention across turns, not simultaneous generation).

Design: the constant becomes PromptCache.defaultMaxSlotsPerModel; the actor gains setMaxSlotsPerModel(_:) (clamped to >= 1) consulted by store()'s eviction; MLXLanguageModel gains a public static setPromptCacheSlotLimit(_:) async passthrough mirroring the existing process-global evictAll() pattern. TDD: behavior tests first (lower limit evicts at the new cap; raise limit retains more; clamp; public passthrough works through the process-global cache).