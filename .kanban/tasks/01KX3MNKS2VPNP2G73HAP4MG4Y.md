---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx8rvs0aja4vekacehym50wv
  text: |-
    Implementation complete, verification in progress (double-check agent running).

    Confirmed `Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift` did not exist (was one of the files deleted at cthbfmw cutover). Recreated it from scratch, matching t71kdmj's established conventions (PromptCacheTestSupport.swift fixtures: PromptCacheProbeModel, makeChunkableCache, resolveOnce, cacheIdentity; doc-comment style; MetalLibraryTestBootstrap.ensureColocatedMetallib GPU bootstrap).

    New suite `PromptCache multi-session and fork conversations (chunk semantics)` with 4 tests, one per required scenario:

    1. `allInterleavedSessionsReuseChunkAlignedPrefixEveryRoundSimultaneously` -- 10 sessions (more than the deleted slot pool's `defaultMaxSlotsPerModel == 4`), 5 rounds each, round-robin interleaved. Asserts an EXACT `tokensToFeed` count per turn per session (formula: `newTokens.count - (previousStoredTokenCount/chunkSize)*chunkSize`), proving reuse for every session every round with no stealing/eviction.
    2. `forkFanOutSharesParentPrefixFromChunksForEveryForkFoundationModelsRouterScenario` -- one chunk-aligned parent prefix, 7 forks + the parent's own continuation (8 total), each with a distinct tail, all resolved via `withTaskGroup` (genuine parallel Tasks, not sequential awaits). Asserts every single one's `tokensToFeed == its own tail exactly`, and that all 8 got distinct assembled cache instances. Explicitly named/documented as the FoundationModelsRouter scenario, cross-referencing `MLXLanguageModel.prewarm`'s doc comment.
    3. `concurrentSessionsGetIsolatedAssembledCachesMutationNeverLeaks` -- two sessions resolve the SAME shared prefix concurrently (`withTaskGroup`), confirms distinct identities + equal content (this part overlaps in spirit with `PromptCacheConcurrencyTests`/`PromptCacheAssembleTests` but is NOT redundant -- it's driven through the multi-session `resolve()`/`store()` surface, not `assemble()` directly), then goes further: mutates session A's assembled cache via `update()` and proves session B's already-resolved cache is untouched, AND a fresh third resolve against the same stored prefix still sees the original un-mutated content -- proving chunk-store immutability, which neither existing file tests.
    4. `oversubscribedSessionsUnderSmallByteBudgetDegradeNeverAssembleWrongPrefix` -- 8 sessions seed one chunk-aligned round each (byte cost measured, not hardcoded), then budget is clamped to ~2 lineages' worth for the remaining growth turns, forcing real cross-session eviction pressure. Asserts the suffix-reconstruction invariant every turn (fed tokens always == true trailing suffix of that session's real prompt, whatever the match length), plus the byte-budget invariant holds throughout.

    Verification so far:
    - `swift build` clean; `mcp__sah__diagnostics check working` reports 0 errors/0 warnings.
    - `swift test --filter PromptCacheMultiSessionTests` -- 4/4 new tests green.
    - `swift test --filter PromptCache` -- 79/79 across 13 suites green (no regressions in sibling suites).
    - `swift test --filter MLXFoundationModelsTests` -- 186/186 across 35 suites green.
    - `xcodebuild build-for-testing` (mlx-swift-lm-Package scheme) succeeded; unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (no -XCTest filter, not piped through tail) -- 186/186 tests, 35 suites, passed in ~4.7s.

    No production code changes were needed -- this was purely test-file work as scoped. Adversarial double-check agent dispatched to verify scenario coverage and non-redundancy before final hand-off; will report its verdict next.
  timestamp: 2026-07-11T14:21:21.290724+00:00
- actor: claude-code
  id: 01kx8tfz32wcs818cxw061m06s
  text: |-
    Work complete and verified green. Two rounds of adversarial double-check surfaced and all findings were fixed (not just logged) before this final handoff.

    ## File
    `Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift` recreated from scratch (confirmed via `git show ccb26e1^:...` that it existed pre-cutover and was deleted at `cthbfmw`; it did not exist on this branch before this task). New suite: `PromptCache multi-session and fork conversations (chunk semantics)`, 4 tests.

    ## Scenario -> test mapping

    1. **Interleaved round-robin sessions** -> `allInterleavedSessionsReuseChunkAlignedPrefixEveryRoundSimultaneously`. 10 sessions (more than the deleted slot pool's `defaultMaxSlotsPerModel == 4`), 5 rounds, round-robin interleaved. Every continuation turn asserts an EXACT `tokensToFeed` count via the formula `newTokens.count - (previousStoredTokenCount/chunkSize)*chunkSize`, proving every session reuses its own chunk-aligned prefix every round with no cross-session stealing/eviction (chunks are dedup'd/read-only, no artificial byte-budget clamp applied in this test).

    2. **Fork fan-out (FoundationModelsRouter scenario)** -> `forkFanOutSharesParentPrefixFromChunksForEveryForkFoundationModelsRouterScenario`. One chunk-aligned parent prefix, 7 forks + the parent's own continuation (8 total, each a distinct tail), all resolved via genuine parallel `Task`s in one `withTaskGroup` (not sequential awaits). Asserts every single one's `tokensToFeed == its own tail` exactly, and all 8 get distinct assembled instances. Explicitly named/documented as the FoundationModelsRouter scenario, cross-referenced against `MLXLanguageModel.prewarm`'s doc comment (`Libraries/MLXFoundationModels/MLXLanguageModel.swift`) which already names this production scenario.

    3. **Concurrent isolation** -> `concurrentSessionsGetIsolatedAssembledCachesRawBufferMutationNeverLeaks`. Two sessions resolve the SAME single-chunk shared prefix concurrently, proving distinct assembled instances with equal content AND genuinely independent underlying buffers (verified via raw C++ buffer address, not just Swift `ObjectIdentifier`/`.==`). A direct in-place raw-buffer write to session A's buffer (bypassing `KVCacheSimple.update()`'s functional replace, which can never exercise this hazard) never appears in session B's already-resolved cache nor in a later fresh resolve. Single-chunk prefix is deliberate: MLX's `concatenate` only special-cases exactly one input, so this is the one shape where a missing `ownedCopy` in `assemble()` would alias the store's own buffer. Non-redundant with `PromptCacheConcurrencyTests`/`PromptCacheAssembleTests` (identity/content-equality only, no mutation, and/or calls `assemble()` directly rather than through a simulated multi-session `resolve()`/`store()` pattern) -- confirmed by both double-check passes.

    4. **Oversubscription under a small byte budget** -> `oversubscribedSessionsUnderSmallByteBudgetDegradeNeverAssembleWrongPrefix`. 8 sessions each seed exactly one chunk-aligned lineage (byte cost measured empirically, not hardcoded); budget is then clamped to exactly that seeded total (zero headroom), so later turns' growth to a second chunk per session pushes true demand to ~2x budget -- genuine oversubscription emerging from real growth. Asserts the suffix-reconstruction invariant every turn (whatever is fed always reconstructs the session's true trailing suffix, never a wrong prefix) PLUS two non-tautological guarantees added after the first double-check round: `observedGenuinePartialReuse` (reuse actually happens somewhere) and `observedFullRebuildUnderPressure` (a real budget-driven full miss actually happens somewhere) -- both would fail if eviction/reuse regressed in either direction.

    ## Review history (both rounds' findings fixed, not just logged)
    - Round 1 REVISE: (a) oversubscription test's suffix check was tautological (resolve() always returns a suffix regardless of correctness) -- fixed by adding the two `observed*` flags above and retuning the budget to actually force cross-session eviction from real growth. (b) isolation test's 3-chunk prefix + `update()`-based mutation could never catch a real aliasing bug (concatenate only special-cases ONE input; `update()` always functionally replaces, never mutates in place) -- fixed by switching to a single-chunk prefix and raw-buffer read/write helpers mirroring `PromptCacheAssembleTests.swift`'s own technique.
    - Round 2 REVISE (narrower): a third assertion I'd added (`minimumChunkCountSeen < sessionCount * 2`) was itself vacuous (trivially true from turn 1 onward, before any real growth-driven pressure). Removed it; the two genuine `observed*` flags plus the pre-existing `totalStoredByteCount() <= budget` check already prove the real guarantee without it.

    ## Verification (all fresh, this session, post-fixes)
    - `swift build` / `swift build --build-tests`: clean.
    - `swift test --filter 'PromptCacheMultiSessionTests'`: 4/4 pass.
    - `swift test --filter 'PromptCache'`: 79/79 across 13 suites, no regressions.
    - `swift test --filter 'MLXFoundationModelsTests'`: 186/186 across 35 suites.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - Unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (no `-XCTest` filter, not piped through tail, wrapped in `timeout`): 186/186 tests, 35 suites, exit 0.

    Note: the diagnostics tool flags a false-positive `Cannot find 'MetalLibraryTestBootstrap' in scope` error on this file via its SourceKit backend -- this workspace's sourcekit-lsp is not installed (per the sah MCP server's own setup status), and the REAL compiler (swiftc via SwiftPM, confirmed via both `swift build --build-tests` and `xcodebuild build-for-testing`) compiles and runs this file with zero errors. Not a real issue.

    No production code changes were needed or made -- purely test-file work as scoped. Leaving in `doing` per scope discipline for the orchestrator to move/commit.
  timestamp: 2026-07-11T14:49:51.458400+00:00
- actor: claude-code
  id: 01kx8w1mrebwzy1sk8qck3tevv
  text: |-
    Fixed the review finding: extracted the duplicated `rawBufferAddress(of:)` / `mutateFirstElementInPlace(of:)` helpers (byte-identical in both files) out of `PromptCacheAssembleTests.swift` and `PromptCacheMultiSessionTests.swift` into the shared `PromptCacheTestSupport.swift`, as internal (non-private) functions alongside the file's existing `makeChunkableCache`/`resolveOnce`/`cacheIdentity` fixtures. Both call sites now reference the single shared implementation; left a short comment in `PromptCacheMultiSessionTests.swift` pointing to where the helpers now live. Merged both files' doc comments into one combined doc comment on the shared version covering both use cases (assemble()'s single-chunk buffer-ownership proof and the multi-session concurrent-isolation proof).

    Verification:
    - `swift build`: clean, exit 0.
    - `swift test --filter 'PromptCache'`: 79/79 passed.
    - Mandated safe pattern (`xcodebuild build-for-testing ... ` then unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`, no `-XCTest` filter, no `tail` piping on the xctest run, wrapped in `timeout`): 186 tests in 35 suites passed.
    - Re-ran by name: `concurrentSessionsGetIsolatedAssembledCachesRawBufferMutationNeverLeaks` (PromptCacheMultiSessionTests) and `singleChunkAssemblyAllocatesIndependentBuffer` + `mutatingSourceChunkBufferInPlaceAfterAssemblyLeavesAssembledCacheUnchanged` (PromptCacheAssembleTests) — all pass. Since the shared functions are byte-identical to what each file had locally (only relocated), their aliasing-detection capability (raw C++ buffer address comparison past the Swift wrapper) is unchanged.

    Second finding (headDim vs headDimension naming) remains REJECTED per the existing note — not touched.

    Task left in `review` per scope; not committing (orchestrator's job).
  timestamp: 2026-07-11T15:16:59.278514+00:00
depends_on:
- 01KX3N1AP5TXNS7T3G2T71KDMJ
- 01KX3MN39J77KZTKPEM2SDT6DJ
position_column: done
position_ordinal: a380
title: Multi-session and fork behavioral suites under chunk semantics
---
## What\nRewrite Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift for the chunk-only world — the assertions get STRONGER because nothing steals:\n- Interleaved round-robin sessions: every session's continuation turn feeds only tokens past its last chunk boundary — for ALL sessions simultaneously, every round (previously only ≤4 could, and forks stole).\n- Fork fan-out: one parent conversation, then N (≥6) forked prompts sharing the parent transcript prefix plus distinct tails — parent and EVERY fork resolve with the shared prefix served from chunks (tokensToFeed ≈ tail only), interleaved and in parallel Tasks. This is the FoundationModelsRouter fork scenario at the unit level.\n- Concurrent isolation: parallel sessions get distinct assembled cache instances (identity) with equal prefix content; a mutation via one session's update() never appears in another's assembled cache (immutability of stored chunks).\n- Oversubscription under a small byte budget: correctness degrades to larger tokensToFeed, never a wrong-prefix assembly (suffix-reconstruction invariant from the existing suite carries over).\n\n## Acceptance Criteria\n- [x] All four scenarios above have passing tests with deterministic assertions (tokensToFeed counts / element equality — no timing dependence)\n- [x] Fork test explicitly named/documented as the FoundationModelsRouter scenario\n\n## Tests\n- [x] Rewritten Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift\n- [x] `swift test --filter 'PromptCache'` green; `swift test --filter MLXFoundationModelsTests` zero failures\n\n## Workflow\n- Use `/tdd` — write failing tests first, then implement to make them pass.\n\n## Resolution notes (commit 43367e8)\nRecreated from scratch (file was deleted at the cthbfmw cutover). Went through 2 rounds of the implementer's own adversarial double-check (catching a tautological oversubscription assertion, then catching a vacuous assertion the first fix introduced). Independently re-verified: all 4 tests hand-traced as genuinely meaningful against the real production chunking/eviction arithmetic; the concurrent-isolation test's aliasing-detection claim additionally confirmed via a real fail-without-fix reproduction (stripped ownedCopy(), confirmed the test catches the resulting buffer aliasing).\n\n## Review Findings (2026-07-11 10:01)\n\n- [x] `PromptCacheMultiSessionTests.swift:280` — the raw-buffer-address helper (`rawBufferAddress`/`mutateFirstElementInPlace`-style) mirrors `PromptCacheAssembleTests.swift`'s existing helper, per the new file's own doc comment acknowledging the parallel. Extract to shared `PromptCacheTestSupport.swift` and have both files use the one shared version.\n- [ ] `PromptCacheMultiSessionTests.swift:346` — `headDim` should be `headDimension` per naming-clarity convention. REJECTED — `headDim` is an established parameter name used consistently across 5 OTHER pre-existing files in this same test family (`PromptCacheTestSupport.swift`'s shared `makeChunkableCache(tokenCount:headDim:valueOffset:)` fixture, `PromptCacheConcurrencyTests.swift`, `PromptCacheResolveTests.swift`, `PromptCacheChunkTests.swift`, `PromptCacheAssembleTests.swift`). Renaming only this new file's local variable would create inconsistency with the established, already-settled convention across the whole file family rather than fix debt. Not touching.