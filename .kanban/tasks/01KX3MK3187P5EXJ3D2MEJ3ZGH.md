---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx63nfxc7c25v2k24p4gey21
  text: |
    ## Scope-boundary investigation and decision

    Read `PromptCache.swift` and `PromptCacheChunks.swift` in full, then grepped `MLXLanguageModel.swift` for actual production call sites. Confirmed the existing slot-based mechanism is very much alive and load-bearing today:
    - `Executor.resolvePromptCache` calls `promptCache.resolve(...)` (around the `respond()` path)
    - `Executor.commitPromptCache` calls `promptCache.store(modelID:tokens:cache:)`
    - `MLXLanguageModel.remove(modelID:)`/`evictAll()` call `promptCache.remove`/`promptCache.evictAll`
    - `MLXLanguageModel.setPromptCacheSlotLimit` calls `promptCache.setMaxSlotsPerModel`

    No task on the board yet wires the new chunk store into `resolve()`/`Executor` (that's the still-pending "Assembly: build a fresh private KVCacheSimple stack from matched chunks" task). Per the task's own scope-boundary guidance, I treated "replace the slot dictionary" as applying at the STORAGE level of this task's own new methods (a fresh `chunkStore: [String: [ChunkKey: StoredChunk]]`, not reusing/repurposing `entries: [String: [Slot]]`) — NOT as license to delete the existing, still-used `entries`/`resolve()`/`store()` slot mechanism. Added the chunk store as purely additive actor state and methods; `entries`, `resolve()`, `store()`, `Slot`, `Decision`, `selectSlot`, `regenerateLastToken`, etc. are all untouched.

    ## Implementation

    `Libraries/MLXFoundationModels/PromptCache.swift`:
    - New actor-private `chunkStore: [String: [ChunkKey: StoredChunk]]`, sharing the existing `recencyCounter`/`nextRecency()` with the slot pool (per the task spec: "keep `nextRecency()`").
    - `insert(modelID:chunks:)`: dedups by `chunk.chunkKey` — an existing key only gets `lastUsed` refreshed (tensors untouched, never replaced); a new key is stored fresh.
    - `lookupLongestPrefix(modelID:newTokens:chunkSize:) -> SendableBox<[StoredChunk]>`: walks the hash chain from `PromptCache.rootChunkKey` over chunk-aligned windows, capped at `maxChunkCount = (newTokens.count - 1) / chunkSize` so at least one token always remains to feed. On every key match, verifies `chunk.tokens == window` (collision safety) before accepting it — mismatch stops the walk exactly like a genuine miss. Touches (bumps `lastUsed` via `nextRecency()`) every matched chunk. Returned boxed in `SendableBox` (matching `resolve()`'s existing pattern) since `StoredChunk` carries non-`Sendable` `MLXArray` tensors and Swift 6 strict concurrency rejected a bare `[StoredChunk]` crossing actor isolation.
    - `chunkCount(modelID:)`: small test-observability accessor (dedup/eviction bookkeeping without leaking the actor's private storage shape) — not in the original spec text but necessary to make the "existing ONCE" dedup acceptance criterion observable from outside the actor.
    - `evictAll()`/`remove(modelID:)` extended to also clear `chunkStore` (per-model scoping preserved), alongside the existing `entries` clearing.

    `Libraries/MLXFoundationModels/PromptCacheChunks.swift`: added `typealias ChunkKey = Int` for the store's key type (matches the task's own `[ChunkKey: StoredChunk]` phrasing).

    ## Tests

    New `Tests/MLXFoundationModelsTests/PromptCacheChunkStoreTests.swift`, 13 tests: dedup (shared-prefix-stored-once, tensor-not-replaced, lastUsed-refreshed), chain walk (longest-prefix-until-missing, stops-at-divergence, touches-lastUsed), cap-at-count-1 (floor((count-1)/chunkSize), single-chunk-prompt-yields-nothing), forced-collision-is-miss (both a first-hop and a mid-chain forced collision), evictAll/remove scoping, and multi-model isolation.

    RED (before implementation): `swift build --build-tests` failed — `value of type 'PromptCache' has no member 'insert'/'chunkCount'/'lookupLongestPrefix'`.

    RED/GREEN specifically for the collision-safety check (the task's own highest-emphasis safety property): temporarily removed the `chunk.tokens == window` guard clause (leaving only the key lookup). Re-ran `swift test --filter PromptCacheChunkStoreTests`:
    - `"a key match whose stored tokens differ from the chunk-aligned window is treated as a miss..."` FAILED: `matched.isEmpty → false`, returned the forged chunk with tokens `[9999, 9998, 9997, 9996]` for a query that was actually `[0,1,2,3]` — i.e. without the check, a hash collision would silently hand back wrong-token KV state, exactly the failure mode the task warns about.
    - `"a forced collision partway down the chain stops the walk..."` FAILED: `matched.count == 1 → false (was 3)` — the walk didn't stop at the corrupted link.

    Restored the guard clause; both tests immediately GREEN again, confirmed via a fresh `swift test --filter 'PromptCache'` run (78/78 passed).

    ## Verification
    - `swift build` and `swift build --build-tests`: clean.
    - `swift test --filter 'PromptCache'`: 78/78 tests passed, 16 suites (includes the new "PromptCache chunk store" suite, 13 tests).
    - Full safe-pattern suite: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` → `** TEST BUILD SUCCEEDED **`, then `timeout 900 xcrun xctest .../MLXFoundationModelsTests.xctest` (unfiltered) → `Test run with 185 tests in 37 suites passed`, exit code 0. No regression to any existing PromptCache test.

    Left in `doing` per scope discipline — no column move, no commit.
  timestamp: 2026-07-10T13:32:26.412782+00:00
- actor: claude-code
  id: 01kx64n2pgka86w55rpz5j3vve
  text: |-
    Fixed the review finding: consolidated `insert(modelID:chunks:)`'s if/else branches in Libraries/MLXFoundationModels/PromptCache.swift into a single code path:

    ```swift
    func insert(modelID: String, chunks: [StoredChunk]) {
        var models = chunkStore[modelID] ?? [:]
        for chunk in chunks {
            // `entry` binds to the EXISTING stored chunk when its key is
            // already present (dedup: only `lastUsed` changes, tensors
            // untouched), or to the new `chunk` parameter when absent.
            var entry = models[chunk.chunkKey] ?? chunk
            entry.lastUsed = nextRecency()
            models[chunk.chunkKey] = entry
        }
        chunkStore[modelID] = models
    }
    ```

    This preserves the "don't replace tensors on dedup" semantics: `entry` binds to the existing stored chunk (only `lastUsed` mutates) when the key is present, and to the new `chunk` parameter otherwise.

    Verification:
    - `swift build` clean, no warnings introduced.
    - `swift test --filter 'PromptCache'` -- 78/78 passed.
    - Mandated safe pattern: `xcodebuild build-for-testing` (scheme mlx-swift-lm-Package) succeeded, then unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` -- 185/185 passed, same count as before.
    - Specifically confirmed the dedup test "dedup on an already-present key refreshes lastUsed but never replaces the stored tensors" passed, plus the "forced collision partway down the chain stops the walk at the last genuine match" test, verifying the consolidation didn't break tensor-identity/collision-safety guarantees.

    Task left in `review` per scope discipline; not committing (orchestrator handles that).
  timestamp: 2026-07-10T13:49:41.456151+00:00
- actor: claude-code
  id: 01kx65eaweh0t8n01qbgbbd656
  text: |-
    Fixed the review finding: consolidated lookupLongestPrefix's two verbatim `return SendableBox([])` early-exit guards into one. Restructured so `maxChunkCount` is computed via a ternary (`chunkSize > 0 ? (newTokens.count - 1) / chunkSize : 0`) before the guard — this avoids integer division-by-zero when chunkSize is invalid, so the invalid-chunkSize case and the zero-max-chunk-count case now share a single `guard maxChunkCount > 0, var models = chunkStore[modelID] else { return SendableBox([]) }`. Only one `SendableBox([])` construction remains in the function.

    Verified:
    - `swift build`: clean, exit 0.
    - `swift test --filter 'PromptCache'`: 78/78 passed.
    - Mandated safe pattern: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` succeeded, then unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`: 185/185 passed (same count as before the fix), no failures anywhere in output.

    Task left in `review` per scope discipline; not committing (orchestrator handles that).
  timestamp: 2026-07-10T14:03:29.038924+00:00
- actor: claude-code
  id: 01kx665dz82tqzqnczhck0smhy
  text: 'Final review pass (2026-07-10 09:04) initially returned 1 finding (LCP-computation duplication between decide()/selectSlot()) with a partial-failure warning (1/15 review sub-tasks failed). Confirmed via git diff that finding was pre-existing/untouched by this commit — folded into tracking task ^6q1z64f alongside the earlier rebuild-pattern debt. Re-ran the review for a complete pass (2026-07-10 09:12): zero findings, 0 failed. Moving to done.'
  timestamp: 2026-07-10T14:16:05.864734+00:00
depends_on:
- 01KX3MJKYYNXQ8CK197WX4W28J
position_column: done
position_ordinal: '9780'
title: 'Chunk store: per-model insert with dedup, longest-prefix lookup, LRU touch'
---
## What\nInside the PromptCache actor (Libraries/MLXFoundationModels/PromptCache.swift): replace the slot dictionary with a per-model chunk store `[String /*modelID*/: [ChunkKey: StoredChunk]]` plus the monotonic recency counter (keep `nextRecency()`).\n- `insert(modelID:chunks:)`: walk the chunk list; a key already present is deduplicated (refresh its `lastUsed`, do NOT replace tensors — dedup is the point); new keys are stored.\n- `lookupLongestPrefix(modelID:newTokens:chunkSize:) -> [StoredChunk]`: walk the hash chain from the root over chunk-aligned windows of `newTokens`, stopping at the first missing key. CRITICAL: cap the walk at `newTokens.count - 1` tokens so at least one token always remains to feed (this replaces the old regenerateLastToken / n_past-- machinery — generation always needs ≥1 token for fresh logits). Touch every matched chunk's `lastUsed`.\n- COLLISION SAFETY: a 64-bit hash collision would serve wrong KV state silently — the worst possible cache failure. On every key match, verify `chunk.tokens == the chunk-aligned window` (cheap array compare, SGLang-style); mismatch = treat as miss and stop the walk.\n- Keep `evictAll()` / `remove(modelID:)` semantics over the new storage.\n\n## Acceptance Criteria\n- [x] Storing two conversations sharing a 3-chunk prefix results in the shared chunks existing ONCE (dedup observable via store size / byte accounting)\n- [x] Lookup returns the longest chain prefix; diverging tokens stop the walk at the last shared chunk boundary\n- [x] Lookup for newTokens exactly equal to a fully-stored sequence returns at most floor((count-1)/chunkSize) chunks — never covers the whole prompt\n- [x] A key match whose stored tokens differ from the window (forced collision in test) is treated as a miss — wrong-token KV is never returned\n- [x] evictAll/remove drop chunks for the right scope (per-model isolation preserved)\n\n## Tests\n- [x] Extend Tests/MLXFoundationModelsTests/PromptCacheChunkTests.swift (or new PromptCacheChunkStoreTests.swift): dedup, chain walk, divergence, cap-at-count-1, forced-collision-is-miss (inject a StoredChunk under a colliding key with different tokens), evictAll/remove scoping, multi-model isolation\n- [x] `swift test --filter 'PromptCache'` green\n\n## Workflow\n- Use `/tdd` — write failing tests first, then implement to make them pass.\n\n## Resolution notes (commits d4e203d, 74d3ab9)\nImplemented as purely additive new actor state/methods alongside the existing, still-production-used slot-based cache mechanism (verified via grep that MLXLanguageModel.swift has zero references to the new API). Verified across two independent rounds, including a reproduced RED/GREEN on the collision-safety guard and algebraic tracing of the cap arithmetic across boundary cases. Round-2 review's insert() dedup-branch duplication finding fixed in 74d3ab9.\n\n## Review Findings (2026-07-10 08:50) — triage\n\n**Fix directly** (new code from this task's own d4e203d commit — confirmed via `git diff 00738c1..HEAD`):\n- [x] `lookupLongestPrefix`'s two `return SendableBox([])` early-exit guards (empty/invalid-chunkSize case, zero-max-chunk-count case) are verbatim duplicates. Consolidate into one shared early-exit path or a single combined guard.\n\n**Deferred to new tracking task `6q1z64f`** (confirmed pre-existing via `git diff 00738c1..HEAD` — zero matches, predates this task entirely, part of the original multi-slot `resolve()`/`applyDecision` mechanism from earlier in the session): the repeated `(model.newCache(parameters: parameters), newTokens)` rebuild-pattern at 4 sites, and the near-duplicate `(slot.cache, Array(newTokens.suffix(count)))`-shaped returns.