---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m616f6qstdymdkss3rzkdjf2
  text: |-
    Research (implement):
    - The portable state is `MTPDrafterContinuation` with `arrays()` / `metadata()` / `init(arrays:metadata:drafter:)` (MTPDrafterContinuationSerialization.swift), keys under `reservedKeyPrefix` `__mlx_lm_mtp_drafter_`. There is no `MTPDrafterCacheState`.
    - The restore needs the drafter (`any ResumableMTPDrafterModel`), which `loadPromptCacheSnapshot(url:)` does not have. So the snapshot cannot hold a live continuation after a file load: it holds the serialized arrays+metadata, and the session resolves them against its drafter at the first turn (a throw = cold start).
    - File layout: drafter keys go at the top level of the safetensors arrays/metadata. `unflattenMetadata` already ignores keys whose first component is not 0/1/2, but `unflattenArrays` throws "Corrupt prompt cache: invalid array key" on any non `i.j` key, so `PromptCacheFileContents` must split off the drafter keys first. A pre-change reader thus refuses a new file (safe failure, no silent drop).
    - ChatSession: a raw-cache session (snapshot init) keeps `RealizedCache.mtpDrafterContinuation` and hands it to `MTPSpeculativeTokenIterator`, which resumes via `alignedBoundary` or cold-starts. A cold start sets `cacheStartPosition` to the cached position, a resume keeps the old start: that is the observable proof of reuse in a test.
    - Each turn writes `RealizedCache` back without extra fields, so a pending saved continuation is dropped after the first turn automatically.
  timestamp: 2026-10-28T12:06:30.393552+00:00
- actor: claude-code
  id: 01m617gkep4ktse8yz7fthymv1
  text: |-
    Implementation landed (not committed):
    - New `SerializedMTPDrafterContinuation` (MTPDrafterContinuationSerialization.swift): `init(_ continuation:)`, `init(arrays:metadata:)` (keeps only reserved-prefix keys), `restored(by:)`, and an internal `removed(from:metadata:)` the file reader uses. It replaces the `MTPDrafterCacheState` the card first named, because `init(arrays:metadata:drafter:)` needs the drafter and a file load has none.
    - KVCache.swift: `PromptCacheSnapshot.drafterContinuation`, `savePromptCache`/`preparePromptCacheSave(... drafterContinuation:)` (refuses a continuation with an empty cache, like model state), and `PromptCacheFileContents` splits the drafter keys off before `unflattenArrays` (which throws on any non `i.j` key). Both load paths (plain and template restore) carry it. A pre-change reader thus refuses a new file instead of dropping the drafter silently.
    - ChatSession: `RealizedCache.savedDrafterContinuation`; the ModelContainer `cache:` init now forwards to a private designated init that takes it (the ModelContext `cache:` init forwards too, which removed its copied body); `promptCache:` inits pass the snapshot's continuation; the first turn restores it with `restoredDrafterContinuation` (resumable + compatible drafter, `try?` restore, nil = cold start) before the memory-policy check; `saveCache(to:)` writes the live continuation, or the still-unrestored saved one.
    - Discovery: on a raw-cache session `info.cachedPromptTokenCount` stays 0 (only the transcript path sets it), so the ChatSession tests read the cached position from `cacheStatus()` instead.
    - Proof of reuse: a cold start sets `cacheStartPosition` to the cached position; the restored session keeps the source's start (the prefilled-prefix position). With the restore line disabled the resume test failed on exactly those two checks (watched RED), then passed with it.
    - `swift package generate-documentation` is not available locally (no docc plugin), so the DocC link renames (`savePromptCache(url:cache:metadata:state:drafterContinuation:)`, `preparePromptCacheSave(...drafterContinuation:)`, `PromptCacheSnapshot/init(...drafterContinuation:)`) were checked by hand only.
  timestamp: 2026-10-28T12:24:44.758077+00:00
- actor: claude-code
  id: 01m617grgeecp83jkzyb4qkbx7
  text: |-
    ### implement — changed
    - evidence: 10 files — Libraries/MLXLMCommon/{KVCache.swift, ChatSession.swift, MTPDrafterContinuationSerialization.swift, ContinuationStateError.swift, KVCacheTypeRegistry.swift}, Tests/MLXLMTests/{MTPDrafterContinuationTests.swift, ChatSessionQwenTextMTPTests.swift, PromptCacheSaveInputTests.swift}, Tests/MLXLMTestSupport/TemporaryPromptCacheFile.swift (new), skills/mlx-swift-lm/references/kv-cache.md. swift build --build-tests: no compiler warnings (only the known "missing creator ... mlx-swift_Cmlx.bundle"). xctest: MLXLMTests 895 XCTest + 1907 Swift Testing pass (only skip: env-gated benchmarkVerification), MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures. Release xcodebuild (ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES) qwen38FourBitMTPSystemPromptCache(): TEST SUCCEEDED, no known issues; edit turn: cached 2169, prefilled 123 tok in 171.8 ms with MTP on (plain 186.6 ms), acceptance 97.1%, exact token match; follow-up MTP decode 86-90 tok/s vs plain ~55.
    - next: /review
  timestamp: 2026-10-28T12:24:49.934468+00:00
- actor: claude-code
  id: 01m618an5tq0bnkhd5qx7wvzyy
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (5b199438) — 3 findings (3 confirmed, 4 refuted): Libraries/MLXLMCommon/ChatSession.swift:2511, Libraries/MLXLMCommon/KVCache.swift:1, Tests/MLXLMTestSupport/TemporaryPromptCacheFile.swift:14
    - next: implement the 3 open Review Findings items, then re-review
  timestamp: 2026-10-28T12:38:58.490430+00:00
- actor: claude-code
  id: 01m618b56knn5ppyvj49bhrnm7
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 10 files
    - test: green — MLXLMTests 895 XCTest + 1907 Swift Testing, 0 failures
    - commit: 5b199438
    - review: findings — Libraries/MLXLMCommon/ChatSession.swift:2511, Libraries/MLXLMCommon/KVCache.swift:1, Tests/MLXLMTestSupport/TemporaryPromptCacheFile.swift:14
  timestamp: 2026-10-28T12:39:14.899443+00:00
- actor: claude-code
  id: 01m618x11eptf8h7qcx8s6w4et
  text: |-
    Review findings worked (not committed):
    - ChatSession.swift: the turn computes `mtpDrafterIsCompatible` once (`mtpDrafter?.isCompatible(with: model) == true`) and passes it to both `restoredDrafterContinuation(_:drafter:drafterIsCompatible:)` and `mtpFallbackReason(drafter:drafterIsCompatible:...)`. `isCompatible(with:)` is now called in one place only.
    - KVCache.swift split (pure move): `PromptCacheFile.swift` holds PromptCacheRestorable, PromptCacheSnapshot, save/prepare/write/load, PromptCacheFileContents, the offset record, the model-state keys and unflattenArrays/unflattenMetadata. `PromptCacheRestore.swift` holds validateBuiltInPromptCache, SavedPromptCacheLayer, PromptCacheTemplateRestore, restoreCacheFromMetaState/restoreBuiltInCache and every built-in validator. KVCacheError and makePromptCache/trim stay in KVCache.swift. Sizes: KVCache.swift 104209 B (was 178724, ~1.59x rendered => ~165 KB), PromptCacheFile.swift 37965 B, PromptCacheRestore.swift 36193 B.
    - Access changes the split forced (no behaviour change): `restoreCacheFromMetaState`, `PromptCacheTemplateRestore`, `SavedPromptCacheLayer` went private -> internal; `CacheList.savedChildren` fileprivate -> internal. `SavedCacheListChild` (private, field-for-field identical to `SavedPromptCacheLayer`) was deleted and `savedChildren` returns `[SavedPromptCacheLayer]`, so `prepareCacheList` passes each child through instead of copying it. A first split that left the template restore in the file layer needed seven built-in private symbols opened; moving the template restore beside the validators avoided that.
    - Temporary URL helper: `TemporaryPromptCacheFile.makeURL()` (MLXLMTestSupport) is the one helper. Deleted the private copies `PromptCacheSaveInputTests.temporaryURL`, `PromptCacheTemplateRestoreTests.temporaryURL` (also used by HybridRecurrentCacheOffsetTests and Qwen3NextPromptCacheRestoreTests), `KVCacheTypeRegistryTests.temporaryURL`, and `tempURL()` in KVCacheTests, and replaced the inline prompt-cache temp-file URLs in ChatSessionTests, TurboQuantTests, KVCacheConfigurationTests, Qwen35ContinuationTests, Qwen3VLContinuationTests, GlmOcrContinuationTests, MiniMaxM3Tests, RotatingKVCacheRewindTests and ScriptedLM/TranscriptKVCacheTests. Temp URLs for non-prompt-cache files (LoRA adapter weights, model directories, missing-weight paths, PNGs, spool directories) are a different file kind and were left alone.
    - skills/mlx-swift-lm/references/kv-cache.md names the two new files.
  timestamp: 2026-10-28T12:49:00.462405+00:00
- actor: claude-code
  id: 01m618x5c9vdnfhjv4er482r9e
  text: |-
    ### implement — changed
    - evidence: 22 files — Libraries/MLXLMCommon/{ChatSession.swift, KVCache.swift, PromptCacheFile.swift (new), PromptCacheRestore.swift (new)}, Tests/MLXLMTests/{ChatSessionTests, GlmOcrContinuationTests, HybridRecurrentCacheOffsetTests, KVCacheConfigurationTests, KVCacheTests, KVCacheTypeRegistryTests, MiniMaxM3Tests, PromptCacheSaveInputTests, PromptCacheTemplateRestoreTests, Qwen35ContinuationTests, Qwen3NextPromptCacheRestoreTests, Qwen3VLContinuationTests, RotatingKVCacheRewindTests, ScriptedLM/TranscriptKVCacheTests, TurboQuantTests}.swift, skills/mlx-swift-lm/references/kv-cache.md, plus the task card. swift build --build-tests: exit 0, no compiler warnings. xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest: 895 XCTest, 0 failures; 1907 Swift Testing in 167 suites passed (only skip: env-gated benchmarkVerification, as before). swift format lint on the changed files: no new warnings (only existing ones on untouched lines). All 3 review findings checked.
    - next: /review
  timestamp: 2026-10-28T12:49:04.905330+00:00
depends_on:
- 01M5QWHA34WP7DQMWQ4AHNB0J9
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8280'
title: 'MTP: save and load the drafter state with PromptCacheSnapshot'
---
## What
A session snapshot (`PromptCacheSnapshot`, `savePromptCache`, `loadPromptCacheSnapshot` in Libraries/MLXLMCommon/KVCache.swift) held only the main cache and the model state; `ChatSession.saveCache(to:)` saved `main` and `state` only. A session restored from it with MTP had no drafter to reuse.

The portable drafter state is `MTPDrafterContinuation` (MTPDrafterContinuation.swift, serialization in MTPDrafterContinuationSerialization.swift). Its restore needs the drafter, which a prompt cache load does not have, so the snapshot carries the serialized form `SerializedMTPDrafterContinuation` (in place of the `MTPDrafterCacheState` this card first named), and the session restores it with its drafter at the first turn.

- [x] Add an optional `drafterContinuation: SerializedMTPDrafterContinuation?` to `PromptCacheSnapshot`. Write it with the `arrays()` / `metadata()` of the continuation under `MTPDrafterContinuation.reservedKeyPrefix` (`savePromptCache` / `preparePromptCacheSave` take `drafterContinuation:`), and read it back in `loadPromptCacheSnapshot(url:)` and `loadPromptCacheSnapshot(url:into:)`; `SerializedMTPDrafterContinuation.restored(by:)` calls `init(arrays:metadata:drafter:)`. The main-cache layer parse ignores the drafter keys.
- [x] Make the ChatSession snapshot export (`saveCache(to:)`) and the `init(_:promptCache:speculativeDecoding:...)` initializers carry the drafter state. A session restored with an MTP config reuses the drafter without a new prefill.
- [x] Make sure that old snapshot files, which have no drafter keys, still load. Then the drafter uses the cold start. A mismatched or damaged drafter state also falls back to the cold start.

## Acceptance Criteria
- [x] Save a session snapshot, load it, and continue with MTP: the text is the same as with no save/load, and the drafter is reused (no drafter feed over the cached prefix).
- [x] A snapshot written before this change loads, and generation is correct.

## Tests
- [x] Tests/MLXLMTests: a `savePromptCache` / `loadPromptCacheSnapshot` round-trip unit test with the drafter keys, and a test with an old-format file.
- [x] Tests/MLXLMTests: a ChatSession snapshot round-trip test on the scripted Qwen35 + MTP fixture.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session

## Review Findings (2026-10-28 07:35)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 9 file(s) reviewed, 6 not reviewed.

> ⚠️ 1 file(s) not reviewed — the rendered prompt would exceed the agent's prompt cap:
> - `Libraries/MLXLMCommon/KVCache.swift` — 283604 rendered bytes, over the 262144-byte per-file cap; not reviewed by: duplication (split the file)

> 5 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWS53X7RKJKABDTBTFKTSY.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWS53X7RKJKABDTBTFKTSY.md` — no validator matches this file
> - `.kanban/tasks/01M5SKBCEWG4D25SZZNGG7BVM7.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SKBCEWG4D25SZZNGG7BVM7.md` — no validator matches this file
> - `skills/mlx-swift-lm/references/kv-cache.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/ChatSession.swift:2511` `reuse/reuse` — `restoredDrafterContinuation` repeats the `drafter.isCompatible(with: model)` check that `mtpFallbackReason` already performs for the same drafter and model in the same turn, so the compatibility rule now lives in two places. Pass the already-computed compatibility result into `restoredDrafterContinuation`, or call the shared check from one helper, so a future change to compatibility rules updates one place.
- [x] `Libraries/MLXLMCommon/KVCache.swift:1` `review-engine/prompt-cap` — This file exceeds the review prompt cap — 283604 rendered bytes against the 262144-byte per-file cap — so these validators could not review it: duplication. Split the file into smaller modules that fit the review prompt cap.
- [x] `Tests/MLXLMTestSupport/TemporaryPromptCacheFile.swift:14` `reuse/reuse` — `TemporaryPromptCacheFile.makeURL()` reimplements the temporary-file-URL helper that the test suite already has in several places. The new shared helper sits beside private copies that build the same thing, so there are now two canonical versions of one capability, and only the new one is shared. Use the existing `temporaryURL` helper in `PromptCacheSaveInputTests`, or move one shared helper into `MLXLMTestSupport` and have the existing private helpers call it. Do not keep a third copy beside them.
