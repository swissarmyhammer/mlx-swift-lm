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