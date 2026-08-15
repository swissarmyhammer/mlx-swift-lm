---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kycr9admt3g81pe1n1y73z7c
  text: 'Created IntegrationTesting/IntegrationTestingTests/MiniMaxM3CacheIntegrationTests.swift with the two tests (saveAndRestoreCacheContinuesWithFactRecall, incrementalCacheMatchesFreshFullContextRebuild) per the task spec. Confirmed the Xcode project uses file-system-synchronized groups (no pbxproj edit needed for new files). One compile error fixed: #expect''s message param is Comment, which only auto-converts from string literals, not String expressions built via + concatenation -- switched to a plain triple-quoted string literal with interpolation. build-for-testing now succeeds. Starting the real xcodebuild test run against the already-cached checkpoint next.'
  timestamp: 2026-07-25T13:43:56.084148+00:00
- actor: claude-code
  id: 01kycsfrw9fzenxf1yqzx17j1b
  text: |-
    Real bug found and fixed: the first real-weights run genuinely CRASHED (not skipped) in saveAndRestoreCacheContinuesWithFactRecall with "MLXLMCommon/KVCache.swift:444: Fatal error: KVCacheSimple state must have exactly 2 arrays (keys, values)". incrementalCacheMatchesFreshFullContextRebuild genuinely PASSED (279.6s, no skip) in the same run.

    Root cause: Libraries/MLXLMCommon/KVCache.swift's cacheClassName(_:)/restoreCacheFromMetaState(...) (used by public savePromptCache/loadPromptCache) only recognize cache types defined inside MLXLMCommon itself (KVCacheSimple, RotatingKVCache, QuantizedKVCache, ChunkedKVCache, MambaCache, ArraysCache, CacheList). MiniMaxM3KVCache lives in MLXVLM (a downstream module MLXLMCommon can't import), so it silently fell into cacheClassName's `default: return "KVCache"` branch, got saved under the WRONG class name, and loadPromptCache then tried to reconstruct it as a plain KVCacheSimple -- which only accepts a 2-array state, and MiniMaxM3KVCache's state has 3 arrays (keys, values, indexKeys) once any index-key history exists (i.e. any time a sparse/MoE layer, 3..<60, has actually run). Crash was deterministic, not flaky.

    Fix: added `KVCacheSerializationRegistry` (public enum, NSLock-protected, nonisolated(unsafe) backing storage) to Libraries/MLXLMCommon/KVCache.swift -- a registration point for KVCache types defined outside MLXLMCommon. `cacheClassName`/`restoreCacheFromMetaState` consult it only in their existing `default:` fallback branches (zero behavior change for all built-in types). MiniMaxM3KVCache (Libraries/MLXVLM/Models/MiniMaxM3.swift) self-registers under the name "MiniMaxM3KVCache" via a `static let registerSerialization: Void = { ... }()` triggered from `init()`, guaranteeing registration before any instance could be saved.

    Added a fast synthetic regression test at the unit level (Tests/MLXLMTests/MiniMaxM3Tests.swift: miniMaxM3KVCacheSurvivesSaveAndRestoreRoundTrip) that reproduces the exact crash shape (heterogeneous [KVCacheSimple, MiniMaxM3KVCache] array, 3-array sparse state) without needing real weights -- this bug should never require a 5-minute real-weights run to catch again. Full swift test suite: 716/716 green (347+0+80+282+7), no regressions. Re-running the real-weights xcodebuild test now to confirm the fix holds end-to-end.
  timestamp: 2026-07-25T14:04:56.073360+00:00
- actor: claude-code
  id: 01kyct15bm7phtx3dkq3d45vv5
  text: |-
    Fix confirmed against real weights: re-ran the real-weights integration test after the KVCacheSerializationRegistry fix. Both tests genuinely passed in a single test-host process (no crash, no restart, no "Skipping" line):
    - saveAndRestoreCacheContinuesWithFactRecall() passed after 185.005 seconds
    - incrementalCacheMatchesFreshFullContextRebuild() passed after 246.613 seconds
    - ** TEST SUCCEEDED **

    This proves both real-weights claims: (1) MiniMaxM3KVCache's save/restore round-trip via savePromptCache/loadPromptCache genuinely carries forward correct model state (fact recall across a serialized/restored cache), and (2) the sparse-attention cache's incremental decode path is numerically identical to a from-scratch full-context rebuild.

    Summary of everything landed on this task:
    - New file: IntegrationTesting/IntegrationTestingTests/MiniMaxM3CacheIntegrationTests.swift (2 real-weights tests)
    - Fixed real bug in Libraries/MLXLMCommon/KVCache.swift: added KVCacheSerializationRegistry so downstream-module KVCache types (like MiniMaxM3KVCache in MLXVLM) can register themselves for savePromptCache/loadPromptCache support instead of silently misclassifying as KVCacheSimple
    - Libraries/MLXVLM/Models/MiniMaxM3.swift: MiniMaxM3KVCache self-registers via a static let triggered from init()
    - New regression test: Tests/MLXLMTests/MiniMaxM3Tests.swift miniMaxM3KVCacheSurvivesSaveAndRestoreRoundTrip (fast synthetic reproduction, no real weights needed)
    - Full swift test suite: 716/716 green, no regressions
    - xcodebuild build-for-testing: succeeds

    Task is ready for /review.
  timestamp: 2026-07-25T14:14:25.908518+00:00
position_column: done
position_ordinal: d880
title: 'MiniMax-M3: real-weights integration test proving MiniMaxM3KVCache correctness across turns'
---
## What

Add a real-weights integration test proving `MiniMaxM3KVCache` (the custom sparse-attention KV cache landed in kanban `^8dbc476`) actually works correctly across multi-turn generation, on the real `mlx-community/MiniMax-M3-4bit` checkpoint — now cached locally at ~225GB after the `^wz8y8qq` coherence test's successful real-weights run. `Tests/MLXLMTests/MiniMaxM3Tests.swift` already proves cache correctness on tiny synthetic configs; this task proves it on real weights, which is a materially different bar (real quantization, real sparse-attention block counts, real tokenizer).

**Explicitly out of scope / not what this tests**: MLXFoundationModels' cross-session `PromptCache` reuse system. `^8dbc476` documented that `MiniMaxM3KVCache` is deliberately NOT recognized by `PromptCache.isChunkable`/`isHybridMambaAttention`, so `MLXLanguageModel.supportsPromptCacheReuse` correctly reports `false` for M3 and it never participates in that system — nothing here should try to make it participate. This test operates one level lower, at the `MLXLMCommon.ChatSession`/`KVCache` level (the same level `Libraries/MLXVLM/Models` code and `MiniMaxM3CoherenceIntegrationTests.swift` already exercise).

Create `IntegrationTesting/IntegrationTestingTests/MiniMaxM3CacheIntegrationTests.swift` (same directory as `MiniMaxM3CoherenceIntegrationTests.swift` and `MiniMaxM3ImageQAIntegrationTests.swift` — both live directly under `IntegrationTesting/IntegrationTestingTests/`, not under `MLXFoundationModelsIntegration/`, since neither uses the FoundationModels executor path). Reuse the existing gating helpers already made `internal` (not `private`) in `MiniMaxM3CoherenceIntegrationTests.swift` specifically for this purpose: `minimaxM3RequiredMemoryBytes`, `resolveMiniMaxM3Configuration()`, `checkpointIsAvailable(_:)`. Mirror that file's `@Suite(.serialized, .timeLimit(.minutes(240)))` gating pattern (memory check → checkpoint availability check → catch-and-skip on load failure), and its doc-comment convention of stating the exact `xcodebuild` invocation to run this suite directly.

Two tests, both using `MLXLMCommon.ChatSession` with `GenerateParameters(maxTokens: <n>, temperature: 0)` (greedy, deterministic) so outputs can be compared byte-for-byte:

1. **`saveAndRestoreCacheContinuesWithFactRecall`** — cache save/restore round-trip via 100%-public API (do NOT use `ChatSession.withCache`/`.copy()` — that method is internal, not `public`, and `IntegrationTesting` does not `@testable import MLXLMCommon`; using the public save/restore path is also a materially better test since it exercises the safetensors serialization of `MiniMaxM3KVCache.state`/`.metaState`, including the sparse indexer's `index_keys` array, which nothing has exercised on real weights before):
   - Turn 1 on session A: prompt containing a specific fact to remember (e.g. "Remember this number: 8214. Just acknowledge.").
   - `try await sessionA.saveCache(to: <temp file URL>)`.
   - `let (restoredCache, _) = try loadPromptCache(url: <temp file URL>)`.
   - Seed a brand-new `ChatSession(container, cache: restoredCache, generateParameters: ...)` (session B).
   - Ask the fact-recall question on session B (e.g. "What number did I just ask you to remember?").
   - Assert the response contains "8214" — proving the serialized/restored `MiniMaxM3KVCache` genuinely carries forward correct state, not just a shape-compatible but semantically-empty cache.

2. **`incrementalCacheMatchesFreshFullContextRebuild`** — byte-identical equivalence between incremental and from-scratch generation for the same final transcript:
   - Path A (incremental): one `ChatSession`, `respond(to: turn1Prompt)` then `respond(to: turn2Prompt)` on the SAME session (turn 2 reuses turn 1's live, in-place `MiniMaxM3KVCache` via the session's normal incremental-decode path) — capture turn 2's output as `incrementalOutput`.
   - Path B (fresh): a NEW `ChatSession(container, history: [.user(content: turn1Prompt), .assistant(content: <path A's turn 1 output>)], generateParameters: ...)` (the `history:` initializer forces a full fresh prefill of the whole accumulated transcript with no carried-over cache state — see `ChatSession.swift`'s `.history` case in `streamMap`), then `respond(to: turn2Prompt)` — capture as `freshOutput`.
   - Assert `incrementalOutput == freshOutput` (exact string equality; temperature 0 removes sampling variance) — proving the sparse-attention cache's incremental decode path (offset tracking, dense-fallback/indexer interaction, `index_keys` growth) produces numerically identical results to a full from-scratch recomputation.

## Acceptance Criteria

- [x] `saveAndRestoreCacheContinuesWithFactRecall` passes against real weights: the fact-recall response from the cache-restored session contains the remembered fact
- [x] `incrementalCacheMatchesFreshFullContextRebuild` passes against real weights: incremental and fresh-rebuild turn-2 outputs are exactly equal
- [x] Both tests skip gracefully (do not fail) under the same conditions `MiniMaxM3CoherenceIntegrationTests` does: insufficient physical memory, checkpoint unavailable, or checkpoint load failure — mirror that file's skip messages/pattern exactly
- [x] Neither test references `PromptCache`, `supportsPromptCacheReuse`, or any MLXFoundationModels cross-session caching API — this is pure `MLXLMCommon.ChatSession`/`KVCache` scope
- [x] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` succeeds with the new file compiled in

## Tests

- [x] New file: `IntegrationTesting/IntegrationTestingTests/MiniMaxM3CacheIntegrationTests.swift` containing the two tests above
- [x] Run: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/MiniMaxM3CacheIntegrationTests` → both tests pass against the already-locally-cached checkpoint (no fresh download expected since `mlx-community/MiniMax-M3-4bit` is fully cached from `^wz8y8qq`'s verification run)
- [x] `swift test --filter MLXLMTests` → still green, no regressions (this task touches no `Libraries/` source, only adds an `IntegrationTesting` test file, but confirm nothing else drifted)

## Workflow

- Use `/tdd` — write failing tests first (they'll fail to compile/run without the real checkpoint path wired up), then implement to make them pass.

## Review Findings (2026-07-25 09:14)

Scope: `review working` (uncommitted changes vs HEAD). 26 findings, 21 refuted, 14 attempted. All 26 confirmed findings fall on line numbers that predate this task's actual additions — the `KVCacheSerializationRegistry` I added (fixing a real save/restore crash found during this task's real-weights testing) and its two call sites are NOT among them. Every finding is pre-existing debt (missing doc comments, magic-number/index literals, nesting depth) scattered across the rest of `Libraries/MLXLMCommon/KVCache.swift` (lines 357-1323, a 1600+ line file) and one unrelated line in `Libraries/MLXVLM/Models/MiniMaxM3.swift` (line 205, `MiniMaxM3Vision.applyActivation`, added by a prior task `^9a2aw98`, not touched here) — surfaced only because both files appear in this task's diff. Consistent with the disposition established 5 times earlier this session (`^mv9aq7w`/`^xgvth41`/`^wz8y8qq`/`^ayw1xee`/`^9a2aw98`) for this exact "incidental touch surfaces unrelated pre-existing debt in a large shared file" pattern — see individual findings in the raw engine markdown preserved on this task's audit trail via kanban comment.

- [ ] `Libraries/MLXLMCommon/KVCache.swift:357` through `:1323` (25 findings) — pre-existing missing doc comments, magic-number/index literals, and nesting-depth findings scattered across this 1600+ line file, none touching the `KVCacheSerializationRegistry` addition. (SKIPPED: pre-existing KVCache.swift debt unrelated to this task, consistent with established disposition on ^mv9aq7w/^xgvth41/^wz8y8qq/^ayw1xee/^9a2aw98)
- [ ] `Libraries/MLXVLM/Models/MiniMaxM3.swift:205` — pre-existing magic number 1.702 in `MiniMaxM3Vision.applyActivation` (added by task ^9a2aw98, not touched by this task). (SKIPPED: pre-existing debt unrelated to this task, consistent with established disposition) #minimax #minimax-m3