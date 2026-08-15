---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx6dxv7n46thz0g5ajzc9rgd
  text: |
    ## Production cutover complete: resolve()/store() rewired to the chunk path

    ### Investigation (done before touching anything)
    - Read the entire `PromptCache.swift` (slot mechanism: `Slot`, `entries`, `Decision`, `decide`, `selectSlot`, `regenerateLastToken`, `applyDecision`, `maxSlotsPerModel`/`setMaxSlotsPerModel` + the additive chunk-store surface from the 3 prior tasks: `chunkStore`, `insert`, `lookupLongestPrefix`, `chunkCount`, `assemble`, plus `trimAndVerify`/`reconcileGeneratedTokens`/`reconcileCacheAdvance`).
    - Read `PromptCacheChunks.swift` (`sliceChunks`, `StoredChunk`, `ChunkKey`, `chunkKey(parentKey:tokens:)`, `rootChunkKey`, `ownedCopy(of:)`).
    - Read `MLXLanguageModel.swift`'s entire `PromptCache` usage: `resolvePromptCache`/`storePromptCache`/`commitPromptCache` (both overloads)/`reconcileCacheAdvance`/`reconcileGeneratedTokens`/`trimAndVerify` call site (the `.trimCacheByOne` branch)/`setPromptCacheSlotLimit`/`evictAll`/`removePromptCache`/`remove`. Confirmed `resolve`/`store`'s call sites (`resolvePromptCache`, `storePromptCache`) only touch the return shape and named params — no internal-shape assumptions — so keeping the public signatures verbatim was sufficient to leave this file's generation-path code completely untouched.
    - Read all 6 test files slated for deletion, plus discovered a 7th not named in the task description: `PromptCacheSlotLimitPassthroughTests.swift` tests `MLXLanguageModel.setPromptCacheSlotLimit(_:)` directly — since that API is deleted (its whole point was calling the now-gone `setMaxSlotsPerModel`), this suite has no surviving subject and had to go too (the acceptance criterion's `SlotLimit` grep would otherwise still hit its `@Suite`/`@Test` name strings).
    - Checked `bbda7xg` (chunk-size task) and `2sdt6dj` (byte-budget task) descriptions to confirm neither had landed yet and to see the planned `defaultChunkSize` shape ahead of time.

    ### What was deleted (production)
    `Slot`, `entries`, `Decision`, `decide`, `selectSlot`, `regenerateLastToken`, `applyDecision`, `defaultMaxSlotsPerModel`, `maxSlotsPerModel`, `setMaxSlotsPerModel` — all gone from `PromptCache.swift`. `MLXLanguageModel.swift`'s diff is exactly one removal: `setPromptCacheSlotLimit(_:)` (nothing else had touched it yet, confirmed via grep before removing).

    ### What was kept
    `trimAndVerify` (still called by `commitPromptCache`'s `.trimCacheByOne` branch), `reconcileGeneratedTokens`, `CacheAdvanceReconciliation`/`reconcileCacheAdvance`, and the whole chunk-store surface (`chunkStore`, `insert`, `lookupLongestPrefix`, `chunkCount`, `assemble`, `sliceChunks` in `PromptCacheChunks.swift`) — all unchanged except `assemble`'s doc comment, which no longer says "wiring into resolve() is a later task" since this task *is* that wiring.

    ### Chunk-size decision
    No existing chunk-size constant/config existed yet (that's `bbda7xg`, still in `todo`). Added `static let defaultChunkSize = 64` directly on `PromptCache`, referenced literally by both `resolve()` and `store()`. Documented in its own doc comment that an actor-isolated `chunkSize` property + `setChunkSize(_:)` (evict-on-change) is `bbda7xg`'s job — this task deliberately does not build that, matching "don't invent your own capacity/config scheme" scope discipline.

    ### New resolve()/store() (signatures unchanged, `SendableBox` plumbing intact)
    - `resolve`: `lookupLongestPrefix(chunkSize: defaultChunkSize)` → if no chunks matched, `(model.newCache(parameters:), newTokens)` (full rebuild, same fallback shape as before) → else `assemble(chunks:layerCount: chunks.first!.layers.count)`, `tokensToFeed = newTokens.suffix(newTokens.count - chunks.count * chunkSize)`. No checkout: nothing is removed from `chunkStore` on read.
    - `store`: `sliceChunks(chunkSize: defaultChunkSize)` → `nil` drops silently (same degradation as the old non-trimmable-cache case) → else `insert(modelID:chunks:)`.
    - `evictAll`/`remove` simplified to `chunkStore`-only (the `entries` line dropped).
    - Rewrote the actor's doc header: llama.cpp slot-pool comparison → chunked shared store, SGLang RadixAttention-style, assemble-on-hit, contiguous-copy-because-MLX-SDPA-needs-contiguous-K/V, and documented the KNOWN WINDOW (no capacity bound until `2sdt6dj`).
    - Cleaned two stale doc references in `PromptCacheTestSupport.swift` (`applyDecision`/`decide` mentions) and reworded `lookupLongestPrefix`'s doc so `git grep 'regenerateLastToken'` also comes up empty (it was previously only a prose reference to the deleted symbol, not a real usage, but the acceptance criterion's grep is literal).

    ### Tests deleted
    `PromptCacheResolveTests.swift`, `PromptCacheSlotPoolTests.swift`, `PromptCacheMultiSessionTests.swift`, `PromptCacheConcurrencyTests.swift`, `PromptCacheSlotLimitPassthroughTests.swift` (the 5th, discovered during investigation — see above).

    ### PromptCacheTests.swift — surgical edit
    Deleted: `PromptCacheDecisionTests` (decide), `PromptCacheSlotSelectionTests` (selectSlot), `PromptCacheDecidePropertyTests`/`PromptCacheSelectSlotPropertyTests` (both property suites) + their now-orphaned shared helpers (`SplitMix64`, `propertyTestIterationCount`, `testVocabularySize`, `randomTokenSequence`, `residentPrefixAndFeedSuffix`).
    Kept: `PromptCacheReconciliationTests`, `PromptCacheAdvanceReconciliationTests`, `PromptCacheTrimAndVerifyTests` — unmodified.

    ### New minimal chunk-semantics test (this task, per TDD requirement)
    `Tests/MLXFoundationModelsTests/PromptCacheChunkCutoverTests.swift` — 3 tests, real tensor content (mirrors `PromptCacheChunkTests.makeCache`, not the offset-only `PromptCacheTestSupport` fixtures, since `store()` now requires a genuine `KVCacheSimple.state`):
    1. **Extension turn + no-checkout reuse** (the acceptance criterion's specific ask): stores 3 chunks (192 tokens), resolves with `+5` extension tokens twice in a row with NO intervening store. Both calls assert `tokensToFeed.count == 5` (not 197) and `cache[0].offset == 192` — proving the prefix was served from chunks on BOTH calls, i.e. nothing was checked out/consumed by the first `resolve()`.
    2. **Divergence**: shares 2 of 3 originally-stored chunks verbatim, diverges for a 70-token tail — asserts the walk stops at the last shared chunk boundary (matched=2, not 3) and feeds exactly the diverging tail.
    3. **Unchunkable degradation**: `store()` with a `RotatingKVCache` layer → `chunkCount == 0` → next `resolve()` rebuilds and feeds every token.

    ### Verification evidence
    - `swift build` (full package): clean, exit 0.
    - `git grep 'selectSlot\|applyDecision\|regenerateLastToken\|maxSlotsPerModel\|SlotLimit' Libraries/ Tests/`: empty (exit 1/no matches) — confirmed after also cleaning the 3 stale prose references caught by the first pass.
    - `swift test --filter 'PromptCache'`: **52 tests, 7 suites, all passed** (4.72s) — includes the new "PromptCache resolve/store chunk-path cutover" suite (3/3).
    - `swift test --filter MLXFoundationModelsTests`: **159 tests, 29 suites, all passed** (3.48s) — zero failures, matching the acceptance criterion's literal wording.
    - Full safe-pattern suite: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` → **TEST BUILD SUCCEEDED**; unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (wrapped in `timeout`, no `-XCTest` filter, not piped through `tail`) → **159 tests in 29 suites passed**, no crash — confirms `23ff1zx`'s metallib fix holds for a genuinely clean run, not a "pre-existing crash excepted" pass.
    - `swift test --filter MLXLMTests` / `swift test --filter MLXGuidedGenerationTests`: both hit the pre-existing "Failed to load the default metallib" abort under plain `swift test` — this is NOT a regression from this task: `MetalLibraryTestBootstrap` (the `23ff1zx` fix) lives only in `Tests/MLXFoundationModelsTests/TestBootstrap.swift`, scoped to that one target; neither MLXLMTests nor MLXGuidedGenerationTests has ever had that bootstrap, and `git diff --stat` against `Libraries/MLXLMCommon`, `Libraries/MLXGuidedGeneration`, `Libraries/MLXLLM`, `Tests/MLXLMTests`, `Tests/MLXGuidedGenerationTests` shows **zero changes** in any of those paths. The `PromptCache` string hits in those directories are the unrelated free functions `trimPromptCache`/`canTrimPromptCache`/`makePromptCache` in `MLXLMCommon/KVCache.swift`, not this actor. Both targets ran a full batch of their own tests successfully before hitting the (unrelated, environment-only) metallib abort, and the full-package `swift build` compiled everything cleanly — together this rules out an accidental API break from the deletions.

    Task left in `doing` per the orchestrator's instructions — not moved to review, nothing committed.
  timestamp: 2026-07-10T16:31:45.909645+00:00
- actor: claude-code
  id: 01kx6fcrmn1c5xvd9fd6rmz24h
  text: |-
    Doc-cleanup follow-up pass (post-cutover stale references), pure documentation-only -- no test logic, assertions, or production code touched.

    ### Investigation
    Grepped which suites actually consume `PromptCacheTestSupport.swift`'s fixtures (`makeSlotCache`/`makeNonTrimmableSlotCache`/`resolveOnce`/`storeWellFormedSlot`/`cacheIdentity`/`PromptCacheProbeModel`) across `Tests/MLXFoundationModelsTests`: only `PromptCacheChunkCutoverTests.swift` actually calls them (`PromptCacheChunkTests.swift` only *mentions* the file's name in a contrasting doc comment, doesn't use its fixtures; `PromptCacheTests.swift`/`PromptCacheAssembleTests.swift`/`PromptCacheChunkStoreTests.swift` have zero references).

    ### Fixes
    1. `Tests/MLXFoundationModelsTests/PromptCacheTestSupport.swift` -- file-header doc block updated: dropped the now-deleted `PromptCacheResolveTests`/`PromptCacheSlotPoolTests`/`PromptCacheMultiSessionTests`/`PromptCacheConcurrencyTests` suite list, replaced with the actual current consumer, `PromptCacheChunkCutoverTests`.
    2. `Tests/MLXFoundationModelsTests/PromptCacheChunkStoreTests.swift` -- the "ADDITIVE state alongside the existing slot-pool mechanism" comment (accurate when written during `mej3zgh`, before this cutover) rewritten to state the chunk store IS the production path now: `resolve()`/`store()` call straight into `lookupLongestPrefix`/`insert`, the slot pool (`entries`, `selectSlot`, `decide`) is gone, and cross-referenced `PromptCacheChunkCutoverTests` for the assemble/slice wiring these tests don't cover.
    3. `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheEquivalenceTests.swift` -- swept every stale reference to the deleted `decide`/`selectSlot`/`Decision` machinery (`.trimTo`/`.reuseSuffix`/`.rebuild` enum cases, real `KVCache.trim()` calls) in the file header, one `@Test` description string, and inline comments/failure messages around the edited-earlier-turn test -- reworded to describe the actual current mechanism (`PromptCache.lookupLongestPrefix`'s chunk walk stopping at the last shared chunk boundary and feeding the diverging suffix, with `cachedTokenCount` now measured against matched-chunk coverage rather than a `commonPrefixLength` decision case). Assertions and control flow in this file are byte-for-byte unchanged -- only comments and one `@Test(...)` display-string were edited.

    ### Verification
    - `swift build`: clean, exit 0.
    - `swift test --filter 'PromptCache'`: 52 tests, 7 suites, all passed.
    - Mandated safe pattern: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package ...` -> **TEST BUILD SUCCEEDED**; unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (wrapped in `timeout`, not `-XCTest` filtered, not piped through `tail`) -> **159 tests in 29 suites passed**.
    - `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting ...`: fails with the same pre-existing, documented 4 errors in `Support/FMTestHelpers.swift` (missing `tokenCount` argument, `Self` pattern vs `Response.Action`, `_ErrorCodeProtocol` has no `updateUsage`, no macro named `require` -- the known Xcode-beta/FoundationModels SDK mismatch). Confirmed via full untruncated log that zero errors reference `PromptCacheEquivalenceTests.swift` itself -- it only appears in a whole-module compile-job's file list alongside `FMTestHelpers.swift`, not as an error source. No new errors from this doc-only edit.

    Task left in `doing`; nothing moved or committed.
  timestamp: 2026-07-10T16:57:23.349382+00:00
- actor: claude-code
  id: 01kx6g9q5sc9xte5wpc06c69fz
  text: |-
    Review of commit ccb26e1 (2026-07-10 11:59) returned 17 findings, all confirmed pre-existing via `git diff HEAD~1..HEAD` (that commit's only change to MLXLanguageModel.swift is the setPromptCacheSlotLimit deletion — none of these lines were touched):

    - 13 findings suggesting `entryId`/`toolCallsEntryId`/`reasoningEntryId`/`responseEntryId` → `entryID`/etc. (reverting to uppercase), citing `modelID` as the "consistent" project convention. REJECTED — this is the same reversed-direction false positive seen earlier this session: the actual established convention (confirmed via task `12d8p71`, which deliberately renamed these exact identifiers FROM `entryID`-style TO `entryId`-style, verified against the file-wide `tokenId`/`stopTokenIds`/`Xg` pattern) is lowercase-interior-acronym. `modelID` is not a valid counter-precedent — it's itself already flagged as wrong casing and tracked for the same fix in task `sd05wkh` ("Rename modelID -> modelId across the codebase"). Applying this finding would revert already-verified, deliberate work and move the codebase further from its own established convention.
    - 4 findings: `resolvePromptCache`/`storePromptCache`/`removePromptCache`/`isDownloadingInCache` should be explicitly `private` rather than defaulting to `internal`. Legitimate, but confirmed pre-existing and out of scope for this task's PromptCache-cutover diff. Deferred to new tracking task ^4dvscnb.

    cthbfmw's own diff has zero legitimate open findings. Moving to done.
  timestamp: 2026-07-10T17:13:12.121884+00:00
depends_on:
- 01KX3MKJFG9SJFW44RNMWWK2BS
position_column: done
position_ordinal: 9a80
title: Rewire resolve()/store() to the chunk path; delete slot machinery and obsolete suites
---
## What
The production cutover, in Libraries/MLXFoundationModels/PromptCache.swift — signatures of `resolve(modelID:newTokens:model:parameters:)` and `store(modelID:tokens:cache:)` stay EXACTLY as-is (SendableBox plumbing included) so MLXLanguageModel.swift's executor wiring (resolvePromptCache/storePromptCache/commitPromptCache and both reconcile* functions) is untouched:
- `resolve`: lookupLongestPrefix → assemble → (cache, tokensToFeed = suffix past matched chunks). No slot removal — nothing is checked out.
- `store`: sliceChunks (nil ⇒ drop silently, same as today's unchunkable degradation) → insert.
- DELETE dead production code: `Slot`, `selectSlot`, `decide`, `Decision`, `applyDecision`, `regenerateLastToken`, `setMaxSlotsPerModel`, `maxSlotsPerModel`/`defaultMaxSlotsPerModel`. KEEP `reconcileGeneratedTokens`/`reconcileCacheAdvance` AND `trimAndVerify` (commitPromptCache's `.trimCacheByOne` branch in MLXLanguageModel.swift still calls trimAndVerify — it stays, with its existing tests).
- DELETE the test suites that are the spec of deleted code, so the target compiles green: PromptCacheTests.swift's decide/selectSlot suites + both property suites (KEEP the reconciliation suites and trim-and-verify suite), PromptCacheResolveTests.swift, PromptCacheSlotPoolTests.swift, PromptCacheMultiSessionTests.swift, PromptCacheConcurrencyTests.swift. Their chunk-semantics replacements are the follow-up suite tasks — write a MINIMAL green resolve/store chunk test in this task (extension turn feeds only suffix; divergence feeds from last shared chunk boundary) so the cutover is not merged untested.
- Rewrite the PromptCache doc header: llama.cpp slot-pool comparison → chunked shared store (SGLang RadixAttention-style, assemble-on-hit, contiguous-copy because MLX SDPA needs contiguous K/V).
- KNOWN WINDOW (acceptable on-branch, document in the header): between this task and the byte-budget task the store has NO capacity bound.

## Acceptance Criteria
- [ ] `git grep 'selectSlot\|applyDecision\|regenerateLastToken\|maxSlotsPerModel\|SlotLimit' Libraries/ Tests/` returns nothing
- [ ] MLXLanguageModel.swift diff for this task is limited to removing setPromptCacheSlotLimit (if not already removed by the disposition/byte-budget sequencing — coordinate; exactly one task deletes it)
- [ ] Second resolve of the same continuation still reuses (no checkout): tokensToFeed counts prove the prefix was served from chunks for BOTH calls
- [ ] Full unit target compiles and passes: `swift test --filter MLXFoundationModelsTests` — zero failures (pre-existing metallib end-crash excepted)

## Tests
- [ ] Minimal new chunk-semantics tests in this task (see What); full behavioral suites are the dependent tasks
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.