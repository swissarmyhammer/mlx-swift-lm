---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx7npdxfbmrymhftkx7dtwek
  text: |-
    Implemented via TDD.

    ## Implementation

    `Libraries/MLXFoundationModels/PromptCache.swift`:
    - Added `private var chunkSize = PromptCache.defaultChunkSize` (actor state, initialized from the existing `static let defaultChunkSize = 64`).
    - `resolve()` and `store()` now thread the actor's `chunkSize` property into `lookupLongestPrefix`/`sliceChunks`/the matched-token-count math, replacing the hardcoded `Self.defaultChunkSize` references from the cthbfmw cutover.
    - Added `func setChunkSize(_ size: Int)`: clamps to `max(1, size)`, then evicts (`evictAll()`) ONLY when the clamped value differs from the current `chunkSize` — comparing against the current (not the raw requested) value is what makes "same value ⇒ no evict" work correctly even when the caller passes an already-clamped duplicate.
    - `evictAll()` itself was unchanged (still just `chunkStore.removeAll()` — confirmed the old slot-pool `entries` field is fully gone post-cutover, nothing else to clear).
    - Updated the `defaultChunkSize` doc comment to drop the forward-reference to this task now that chunkSize is configurable.

    `Libraries/MLXFoundationModels/MLXLanguageModel.swift`:
    - Added `public static func setPromptCacheChunkSize(_ size: Int) async` immediately after `evictAll()`, passthrough to `promptCache.setChunkSize(size)`, documenting the fork-granularity/coarser-sharing trade-off and the `chunkSize - 1` worst-case extra-prefill bound for the tail past the last chunk boundary.

    ## Tests (TDD: watched RED before implementing)

    Extended `Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift` (confirmed compile failure first — `value of type 'PromptCache' has no member 'setChunkSize'` — before writing `setChunkSize`):

    1. **`configurableChunkSizeRoundTripsWithReuse`** — parameterized over `[1, 64, 256]`. Stores `2*chunkSize + 7` tokens (>2×chunkSize, guaranteeing ≥2 full chunks sliced), asserts `chunkCount(modelID:) > 0` (rules out a vacuous zero-chunks-stored pass), then resolves an extension of that exact prompt and asserts `tokensToFeed.count < extensionTokens.count` — i.e. ACTUAL reuse, not just "didn't crash." All 3 cases pass.
    2. **`setChunkSizeToNewValueEvictsStore`** — sets chunkSize 8, stores real chunks, confirms `chunkCount > 0`, calls `setChunkSize(16)`, asserts `chunkCount == 0`.
    3. **`setChunkSizeToSameValueDoesNotEvict`** — sets chunkSize 8, stores, records `countBefore`, calls `setChunkSize(8)` again (same value), asserts `chunkCount == countBefore` (unchanged — proves the no-op path is real, not just "eviction happened to leave count the same").
    4. **`setChunkSizeClampsNonPositiveToOne`** — parameterized over `[0, -3]`. Calls `setChunkSize(requested)`, then stores exactly 3 tokens and asserts `chunkCount == 3`, which is only possible if the effective chunkSize clamped to 1 (chunking every single token) rather than 0/negative (which would either produce 0 chunks or divide-by-zero in the chunk-count math).

    ## Verification evidence

    - `swift build`: clean (one pre-existing, unrelated deprecation warning on `LanguageModelCapabilities(capabilities:)` in `MLXLanguageModel.swift`, not touched by this change).
    - `swift test --filter 'PromptCache'`: **56 tests in 7 suites passed** (0 failures).
    - Full safe-pattern suite: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` → `** TEST BUILD SUCCEEDED **`, then unfiltered `timeout 900 xcrun xctest .../MLXFoundationModelsTests.xctest` → **163 tests in 29 suites passed** (0 failures, 0 crashes), confirming no regression to any existing `PromptCache` test (chunk assembly, chunk store dedup/chain-walk/collision-safety, resolve/store cutover, reconciliation, trim-and-verify all still green alongside the new configurable-chunk-size tests).

    Task left in `doing` per scope discipline for the orchestrator to move through review/commit.
  timestamp: 2026-07-11T04:06:45.935911+00:00
- actor: claude-code
  id: 01kx7prnsbn8fa1bhmfzrykya9
  text: Review of commit 54d7f24 (2026-07-10 23:17) returned 1 finding claiming `resolveOnce()` is an undefined helper, making the test unable to compile. REJECTED — factually incorrect. `resolveOnce(cache:modelID:newTokens:model:)` is a well-established shared helper defined in `Tests/MLXFoundationModelsTests/PromptCacheTestSupport.swift:88`, used across this entire test target since much earlier in this session's work (confirmed via `grep -n "func resolveOnce" Tests/MLXFoundationModelsTests/*.swift`). The test genuinely compiles and passes — confirmed independently twice already (build clean, `swift test --filter 'PromptCache'` 56/56, mandated safe pattern 163/163, both before and after this exact commit). Same class of stale/hallucinated review citation seen repeatedly this session. Moving to done.
  timestamp: 2026-07-11T04:25:28.107820+00:00
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
position_column: done
position_ordinal: 9d80
title: Configurable chunk size (default 64) with change-invalidates-store semantics
---
## What
In Libraries/MLXFoundationModels/PromptCache.swift: `static let defaultChunkSize = 64`; actor state `private var chunkSize = PromptCache.defaultChunkSize`; `func setChunkSize(_ size: Int)` clamped to ≥ 1. Changing the size makes existing chunk keys meaningless (keys are chains over fixed-width windows), so setChunkSize with a DIFFERENT value must `evictAll()` — document this. Public passthrough in Libraries/MLXFoundationModels/MLXLanguageModel.swift beside evictAll(): `public static func setPromptCacheChunkSize(_ size: Int) async`, doc: trade-off (smaller = finer fork-point granularity; larger = coarser sharing; tail past the last chunk boundary always re-prefills, so worst-case extra prefill ≈ chunkSize-1 tokens).

## Acceptance Criteria
- [ ] Store/resolve round-trips at sizes 1, 64, 256 (parameterized): each case uses > 2×chunkSize tokens and asserts ACTUAL reuse — tokensToFeed strictly smaller than the full prompt on the second resolve (a zero-chunks-stored vacuous pass must be impossible)
- [ ] setChunkSize to a new value drops all stored chunks (next resolve rebuilds); setting the SAME value does not evict
- [ ] Clamp: 0/negative → 1

## Tests
- [ ] Extend Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift: parameterized sizes with the >2×chunkSize + reuse assertion, change-evicts, same-value-no-evict, clamp (arguments: [0, -3])
- [ ] `swift test --filter 'PromptCache'` green

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.