---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx7xbst5zbk8nm3say2x75t0
  text: |-
    Implemented byte-budget LRU eviction via TDD. Summary of design decisions and test evidence:

    **Budget scope: GLOBAL across models, not per-model.** `PromptCache.totalStoredBytes`/`byteBudget` are single actor-level properties shared by every model's chunk store. Rationale (documented on `byteBudget`'s doc comment): every cached model competes for the SAME physical unified-memory pool in a shared process, so a per-model split would either need to be decided up front (starving an idle model while wasting headroom on one nobody's evicting from) or reduce back to a global figure anyway to reason about real memory pressure. A single shared budget lets whichever model is actively serving requests claim more of the pool, with `lastUsed`-driven LRU naturally reclaiming from cold models first.

    **Orphan-reclamation strategy: TRANSITIVE eviction (option a from the task description), not lazy-sweep.** `evictChunkAndDescendants(modelID:key:)` recursively removes every chunk whose `parentKey` equals the just-evicted chunk's key, immediately, in the same eviction pass — reclaiming a lineage's whole byte footprint in one shot. Chose transitive over lazy-sweep because a lazy sweep needs its own pass to find unreachable chunks (no cheaper than the recursive walk) while leaving `totalStoredBytes` overcounting reachable-only memory in the meantime, which would let `evictToBudget()` under-evict.

    **Default budget derivation:** `PromptCache.defaultByteBudget()` uses `MLX.GPU.maxRecommendedWorkingSetBytes()` (backed by Metal's `recommendedMaxWorkingSetSize`, confirmed present in the vendored mlx-swift package at `.build/checkouts/mlx-swift/Source/MLX/GPU+Metal.swift`) scaled by `unifiedMemoryBudgetFraction = 0.25`, leaving headroom for model weights, MLX's own GPU buffer cache (`Memory.cacheLimit`, which itself defaults to the FULL memory limit), and per-request assembled-prefix copies. Falls back to a fixed 2 GiB (`fallbackDefaultByteBudget`) when MLX can't report a working-set size, clamped up to a 16 MiB floor (`minimumByteBudget`) that applies ONLY to this derived/fallback default.

    **Important clamp-semantics correction made mid-implementation:** `setByteBudget(_:)` (and the public `MLXLanguageModel.setPromptCacheByteBudget(_:)`) clamp the caller's value only up to `1` (mirroring `setChunkSize`'s `max(1, size)` pattern) — NOT up to the 16 MiB `minimumByteBudget` floor. Initially I clamped `setByteBudget` to `minimumByteBudget` too, which broke my own tests (a caller-requested 300-byte budget was silently bumped to 16 MiB) — the 16 MiB floor is only appropriate for the unconfigured default, since an explicit caller request for a tiny budget (memory-constrained scenario, or a test asserting eviction under tight pressure) should be honored as requested.

    **LRU tracking:** scan-based (`globalLRUChunk()` linearly scans every model's `lastUsed` for the global minimum) rather than a separate ordered index, per the investigation guidance — store sizes in practice are bounded by distinct resident chunks, not request volume.

    **Public API:** `MLXLanguageModel.setPromptCacheByteBudget(_ bytes: Int) async` added beside `evictAll()`/`setPromptCacheChunkSize(_:)`, with the peak-memory model (budget + in-flight-request × assembled-prefix-size, competing with GPU cache and model weights) documented in its doc comment.

    **Verified `setPromptCacheSlotLimit` is already gone** (per `ccb26e1`/cthbfmw cutover) — `git grep -i 'slotLimit\|SlotsPerModel' Libraries/ Tests/` returns nothing.

    **Test evidence (TDD: watched all 3 fail before implementing, all pass after):**
    - New file `Tests/MLXFoundationModelsTests/PromptCacheByteBudgetTests.swift`, 3 tests:
      1. `insertPastBudgetNeverExceedsBudget` — byte-accounting invariant: asserts `totalStoredByteCount() <= budget` after every insert, inserting 5×100-byte lineages against a 300-byte budget.
      2. `byteBudgetEvictsLeastRecentlyTouchedLineageFirst` — LRU order through the real `resolve()`/`store()` surface: stores an "untouched" lineage first, then a "touched" one, touches the touched lineage via `resolve()`, tightens the budget to fit exactly one lineage, and asserts the untouched lineage was evicted (full rebuild) while the touched one survives (partial reuse).
      3. `evictingLineageHeadReclaimsWholeLineageBytesTransitively` — orphan reclamation: a 4-chunk lineage (400 bytes) under a 350-byte budget (exceeded by only one chunk's worth) must have its WHOLE 400 bytes reclaimed (`totalStoredByteCount() == 0`, `chunkCount == 0`), not just enough freed to satisfy the budget numerically — proving transitive reclamation, not partial/lazy unreachability.
    - `swift test --filter 'PromptCache'`: 66 tests in 11 suites, all passed.
    - `swift build`: clean, no new warnings (one pre-existing unrelated deprecation warning at line ~814 for `LanguageModelCapabilities.init(capabilities:)`, untouched by this change).
    - Full safe-pattern suite: `xcodebuild build-for-testing` succeeded, unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`: 173 tests in 33 suites, all passed, zero failures.

    Adversarial double-check review dispatched to verify correctness of the recursive eviction (no double-free/infinite-loop hazards) and test rigor before handoff. Leaving task in `doing` for `/review` per implement workflow.
  timestamp: 2026-07-11T06:20:46.277340+00:00
- actor: claude-code
  id: 01kx7xywgdr52vy6zaz3ve4smc
  text: |-
    Adversarial double-check review (REVISE, then fixed):

    **Finding:** `byteBudgetEvictsLeastRecentlyTouchedLineageFirst` in `PromptCacheByteBudgetTests.swift` stored the "untouched" lineage first and "touched" second, so `insert()`'s monotonic `nextRecency()` counter already made "touched" strictly more recent than "untouched" from store order ALONE — before the `resolve()` touch ever ran. The test would have passed identically even if `lookupLongestPrefix`'s `chunk.lastUsed = nextRecency()` bump were a no-op, meaning it didn't actually isolate/prove the resolve()-driven recency mechanism.

    **Fix:** Reversed store order (touched first, untouched second) so store order ALONE would make "touched" the eviction target; only the subsequent `resolve()` touch can push touched's recency above untouched's and make it survive.

    **Empirically verified the fix is not vacuous:** temporarily replaced `chunk.lastUsed = nextRecency()` in `lookupLongestPrefix` with a no-op, re-ran the single test — it failed exactly as expected (untouched survived, touched was evicted, the opposite of what the assertions require). Reverted the no-op and confirmed the test (and the full suite) passes again.

    **Final verification after fix:**
    - `swift test --filter 'PromptCache'`: 66/66 tests passed (11 suites).
    - Full safe-pattern suite: `xcodebuild build-for-testing` succeeded, unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`: 173/173 tests passed (33 suites), zero failures.

    All other findings from the double-check review checked out clean (no infinite-loop/double-free hazard in `evictChunkAndDescendants`/`evictToBudget`, byte accounting correct, doc comments consistent, `slotLimit`/`SlotsPerModel` grep empty). Task is green and ready for `/review`.
  timestamp: 2026-07-11T06:31:11.629688+00:00
- actor: claude-code
  id: 01kx80c7ejyjnr4sf5hhxz7r0z
  text: |-
    Added 2 new tests to `Tests/MLXFoundationModelsTests/PromptCacheByteBudgetTests.swift` closing the two coverage gaps an independent verification agent flagged, per the assigned task.

    ## New tests

    **4. `insertEvictsMultipleLineagesInOneCallWhenBudgetForcesIt`** (MARK: "4. Multi-lineage eviction within a single insert() call")
    Inserts 5 independent single-chunk lineages (100 bytes each = 500 total) in ONE `insert(modelID:chunks:)` call with budget=250. Asserts `finalTotal == 200` (3 lineages evicted, landing exactly at the point where 2 remain) and `chunkCount == 2` — proving `evictToBudget()`'s `while` loop re-scans and evicts MORE THAN ONE lineage per call, not just one.

    **5. `evictionPicksVictimAcrossModelsNotJustTheInsertingModel`** (MARK: "5. Cross-model LRU eviction")
    Stores a chunk for model B, then model A (via low-level `insert`), touches model A's chunk via a direct `lookupLongestPrefix` call (the same recency-bump mechanism `resolve()` uses, avoiding the GPU bootstrap since this only needs the chunk-store primitives), then inserts a second chunk for model A that pushes the GLOBAL total over budget. Asserts model B's untouched chunk is evicted (chunkCount(modelB) == 0) while both of model A's chunks survive (chunkCount(modelA) == 2) — proving `globalLRUChunk()` scans across ALL models, not just the model the triggering `insert()` call was for.

    Both follow the established low-level chunk-store-primitive style of tests 1/3 in this file (hand-crafted `StoredChunk` fixtures via the existing `makeStoredChunk` helper), needing no GPU bootstrap.

    ## RED/GREEN evidence

    **Test 4** — temporarily changed `evictToBudget()`'s `while` to `if` in `PromptCache.swift`: test 4 failed (`finalTotal == 400`, `remainingCount == 4`, only 1 of 5 lineages evicted) while tests 1/2/3/5 stayed green (none of them need more than one eviction per call). Reverted to `while`; reran — all 5 green.

    **Test 5** — temporarily restricted `globalLRUChunk()`'s scan to only the model passed into the triggering `insert()` call (threaded a `scopedToModelID` param through `evictToBudget`/`globalLRUChunk`, called from `insert()` with its own `modelID`): test 5 failed (`chunkCount(modelB) == 1` instead of 0, `chunkCount(modelA) == 1` instead of 2) while tests 1/2/3/4 stayed green (none of them span multiple models). Reverted `globalLRUChunk`/`evictToBudget` to their original unscoped signatures; reran — all 5 green, and `PromptCache.swift` is now byte-identical to its pre-fault-injection state.

    ## Verification
    - `swift build` — clean.
    - `swift test --filter 'PromptCache'` — **68/68 pass** (66 existing + 2 new), confirmed both before and after the RED/GREEN fault-injection round-trips.
    - Mandated safe pattern (`xcodebuild build-for-testing ...` + unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`, wrapped in `timeout`) — **175/175 pass, 0 failures**, stayed green.
      - Note: this xctest run does not appear to discover/execute ANY of the GPU-bootstrap-touching suites at all (not just PromptCache — `MetalLibraryBootstrapTests` is likewise entirely absent from its 33-suite/175-test list, confirmed via `nm` showing the symbols ARE present in the compiled binary). This matches the pre-existing "swiftpm-test-gpu-metallib-limit" memory note about this harness's known GPU-test limitations — not a regression introduced by this change. The `swift test --filter 'PromptCache'` run above is the one that actually exercises and confirms these tests.

    Task left in `doing` per scope discipline — no column move, no commit.
  timestamp: 2026-07-11T07:13:25.970383+00:00
- actor: claude-code
  id: 01kx81mgf0tjg5w5es56h2arzw
  text: |-
    Added a test consumer for `currentByteBudget()` per the 2026-07-11 review finding, in Tests/MLXFoundationModelsTests/PromptCacheByteBudgetTests.swift (new "MARK: - 6. currentByteBudget() round-trip" section):

    - `setByteBudgetIsObservableViaCurrentByteBudget()` — sets a budget of 12,345 via `setByteBudget(_:)` and asserts `currentByteBudget()` reflects it.
    - `setByteBudgetClampsNonPositiveToOneObservably(requested:)` — parameterized over `[0, -3]`, asserts `currentByteBudget() == 1` after each, proving the clamp-to-≥1 behavior directly through the getter (mirrors `PromptCacheChunkTests`'s `setChunkSizeClampsNonPositiveToOne` pattern, but here uses the direct getter since `currentByteBudget()` already exists as a test seam, unlike `chunkSize` which has no direct getter).

    Verification:
    - `swift build` clean (only pre-existing unrelated `.docc`/README resource warnings).
    - `swift test --filter 'PromptCache'`: 70 tests in 11 suites passed, 0 failures (previously 68 test instances; the 2 new ones bring it to 70 — the new plain test contributes 1, the clamp test's 2 parameterized cases contribute 2, net +3 test instances from the 2 new `@Test` functions since one existing count baseline was slightly off).
    - Mandated safe pattern: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package ...` succeeded (TEST BUILD SUCCEEDED), followed by unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (no `-XCTest` filter, not piped through `tail`): 177 tests in 33 suites passed, 0 failures.

    Left in `review` per scope; not committing (orchestrator's job). Both new-code checkboxes in the description are now checked off.
  timestamp: 2026-07-11T07:35:25.920510+00:00
depends_on:
- 01KX3MMB4DGB8N9069VCTHBFMW
position_column: done
position_ordinal: a080
title: Byte-budget LRU eviction; remove the superseded slot-limit API
---
## What\nIn Libraries/MLXFoundationModels/PromptCache.swift: track total stored bytes (StoredChunk.byteSize — REAL owned-copy footprint per the slicing task, not view metadata). On insert exceeding the budget, evict least-recently-used chunks until under budget. Evicting a chunk orphans its chain descendants (lookup already stops at the missing parent, so orphans are unreachable) — orphaned chunks' bytes MUST still be reclaimed: either evict descendants transitively with their parent, or treat unreachable chunks as immediately evictable; implementer picks one and documents it. Default budget: derive from the GPU/unified-memory limit if mlx-swift exposes one (check `MLX.GPU.memoryLimit` / recommendedMaxWorkingSetSize), else a documented fixed default (e.g. 2 GiB); clamp ≥ one chunk.\n\nPEAK-MEMORY MODEL (document in the API doc): the budget bounds the STORE only — every resolve additionally materializes one assembled prefix copy per in-flight request, so peak unified memory ≈ budget + (in-flight requests × assembled prefix size), and chunk residency competes with MLX's own GPU buffer cache (GPU cacheLimit). The default budget derivation must leave explicit headroom for assembly copies.\n\nPublic API in Libraries/MLXFoundationModels/MLXLanguageModel.swift: `public static func setPromptCacheByteBudget(_ bytes: Int) async` beside evictAll(). REMOVE the superseded slot-limit surface if it still exists at this point (`setPromptCacheSlotLimit` — coordinate with the cutover task; exactly one task deletes it).\n\n## Acceptance Criteria\n- [x] Inserting past the budget evicts the least-recently-used chunks first; recently-touched (resolved) chunks survive\n- [x] Total accounted bytes never exceeds the budget after insert returns\n- [x] Orphaning a lineage (evict its head under budget pressure) reclaims the whole lineage's bytes — accounted bytes converge to reachable chunks only\n- [x] `git grep -i 'slotLimit\\|SlotsPerModel' Libraries/ Tests/` returns nothing\n- [x] The peak-memory model and headroom rationale appear in setPromptCacheByteBudget's doc comment\n\n## Tests\n- [x] New/extended tests in Tests/MLXFoundationModelsTests: budget-driven eviction order (touch one lineage via resolve, insert until evict, assert the untouched lineage died first), byte-accounting invariant, orphaned-lineage byte reclamation (assert freed bytes, not just unreachability)\n- [x] `swift test --filter 'PromptCache'` green\n\n## Workflow\n- Use `/tdd` — write failing tests first, then implement to make them pass.\n\n## Resolution notes (commit ed42775)\nVerified across 3 independent rounds (implementer's own double-check, an independent peer session, and a dedicated orchestrator verification pass) including hand-traced/empirically-confirmed multi-lineage-eviction-in-one-call, cross-model LRU correctness, and multi-level orphan-chain reclamation. 2 additional tests added to close a coverage gap the independent pass found (multi-lineage eviction, cross-model LRU), each with real RED/GREEN evidence.\n\n## Review Findings (2026-07-11 02:15)\n\n**Deferred to new tracking task `6gv0ea0`** (confirmed pre-existing via `git diff HEAD~1..HEAD` — zero matches, this commit's diff only adds content, none of these were touched): 7 blank-line doc-comment formatting findings in `MLXLanguageModel.swift` (lines ~345, 375, 418, 474, 524, 541, 688 as of the review pass — will have shifted).\n\n**Fix directly** (new code from this commit):\n- [x] `Libraries/MLXFoundationModels/PromptCache.swift:295` — `currentByteBudget()` has no inbound callers/test consumer, unlike its siblings `totalStoredByteCount`/`chunkCount` which ARE exercised by tests. Add a test that verifies `setByteBudget`'s effect is observable via `currentByteBudget()` (e.g. set a new budget, assert `currentByteBudget()` reflects it, including the clamp-to-≥1 behavior) — this both gives it a real consumer and adds valuable coverage for the budget-setter/getter round-trip that wasn't otherwise directly tested.