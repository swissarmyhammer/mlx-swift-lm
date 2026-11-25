---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m884eqk1z0fjrd4frmy6s2s0
  text: 'Research: the model cache is `ModelCache` (Libraries/MLXFoundationModels/ModelCache.swift), keyed by `MLXLanguageModel.modelID` (a String). Paths: `load(modelID:)` (lookup+insert into `containers`/`loadingTasks`), `loadDrafter` (DrafterKey.modelID), `isDownloading`, `lastError`, `remove(modelID:)` (evict), `evictAll()`. The same `modelID` also keys `Executor.Configuration` (the framework caches executors by it), the executor''s prompt-cache key (`ExecutorPromptCacheKey.ModelID`, read from the executor''s modelID), tokenizer/bias/constraint caches, and `releasePromptCaches(of:)`. So a discriminator only in ModelCache''s dictionary would still let a clustered and a local load share an executor config and prompt-cache entries (a RemoteKVCache could be served to a local session). Plan: a public `ModelLoadIdentity` (`.local` default, `.clustered(planID:)`) on MLXLanguageModel, folded into `modelID` for clustered loads only, so local keys are unchanged and every per-model cache (container, executor config, prompt cache, availability) separates. Transport had no teardown: add `RemoteModelTransport.close()` (default no-op), `ClusterLanguageModel.close()` (idempotent, closes the session), and have `ModelCache.remove`/`evictAll` hand back the dropped containers so `evict()`/`evictAll()` close a cluster model''s transport. Fake transport: `InProcessRemoteModelWorker` gets a close count.'
  timestamp: 2026-11-25T01:17:22.401549+00:00
- actor: claude-code
  id: 01m884vdcam2ab7m7py21kjrh2
  text: |-
    Implemented. `ModelLoadIdentity` (Libraries/MLXFoundationModels/ModelLoadIdentity.swift: `.local`, `.clustered(planID:)`) is a new `MLXLanguageModel.init(... kvCache:, loadIdentity: = .local, ...)` parameter. `modelID` folds it in (`<checkpoint>#cluster=<planID>`; local unchanged), so the container, executor configuration, prompt cache entries, availability and tokenizer caches all separate. `ModelCache.remove(modelID:)` now returns the dropped container and `evictAll()` the dropped containers; `evict()`/`evictAll()` then call `ClusterLanguageModel.close()` inside `container.perform` (waits for an in-flight generation, so a running respond still finishes). Teardown hook: `RemoteModelTransport.close()` (default no-op extension, so ShardCheck's AnnouncingTransport is unchanged), `ClusterLanguageModel.close()` -> `RemoteModelSession.close()` (closes the transport one time; every later op throws the new `ClusterModelError.closed`). Fake: `InProcessRemoteModelWorker.closeCount`.

    Gotcha: one InProcessRemoteModelWorker serves one session (request sequence numbers), so each clustered container in a test needs its own worker. A first version of the "evicting X leaves Y" tests passed vacuously (the loader returned one fixed container / a load count that matched either way); now each load hands out a new container, and a mutation run (discriminator removed) fails 5 of the 7 new tests.

    Tests: FoundationModelsCacheTests/ClusteredModelCache (7, incl. an evictAll subprocess test) and ClusterLanguageModelTests close tests (2) pass; FoundationModelsCacheTests (31) and ClusterExecutorTests (6) pass.

    ### implement — changed
    - evidence: 8 files — Libraries/MLXFoundationModels/{ModelLoadIdentity.swift (new), MLXLanguageModel.swift, ModelCache.swift}, Libraries/MLXLMCommon/Remote/{RemoteModelTransport.swift, ClusterLanguageModel.swift}, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift, Tests/MLXLMTests/ClusterLanguageModelTests.swift, Tests/MLXFoundationModelsTests/ClusteredModelCacheTests.swift (new)
    - next: review
  timestamp: 2026-11-25T01:24:17.930860+00:00
- actor: claude-code
  id: 01m884wpa9m6qqn01wwh46dtej
  text: 'Follow-up in the same commit: tools/ShardCheck''s `AnnouncingTransport` already had a `close() throws` (sends `end` so the other ranks leave the loop). With the new non-throwing `RemoteModelTransport.close()` requirement it no longer witnessed the requirement and only shadowed it, so `ClusterLanguageModel.close()` would silently hit the no-op default. Renamed it to `endRun()` (ShardCheck.swift calls it after the run), so the name no longer suggests it is the protocol teardown.'
  timestamp: 2026-11-25T01:24:59.849682+00:00
- actor: claude-code
  id: 01m885y6jnrpf8zxkk0e5ch9se
  text: 'Regression found in the full MLXFoundationModelsTests run: MLXLanguageModelEvictionPoolTests (evict / evictAll leave no buffer in the pool) failed, because evict() held the removed container across releasePromptCaches, whose MLX.Memory.clearCache ran while the weights were still referenced. Fixed: teardown now runs first, on containers passed straight into `tearDown(_: [ModelContainer])` with no other reference, so they are freed before the prompt-cache release and pool clear. Folded into the task commit (now 6fc24c79). EvictionPool (2) and FoundationModelsCacheTests (31) pass.'
  timestamp: 2026-11-25T01:43:17.845291+00:00
position_column: done
position_ordinal: fffff580
title: 'Model cache key: discriminate clustered vs local loads'
---
The MLX model cache keys only on model identity, so a clustered load and a local load of the same model ID could collide. Add a load-identity discriminator to the cache key (e.g. `.local` vs `.clustered(planID)`), so they never share an entry. `evict()` must reach the clustered container, whose teardown kills its workers (via a teardown hook the transport provides).

- [x] Find the cache key and every lookup/insert/evict path.
- [x] Add the discriminator, defaulting to local, so existing callers are unchanged.
- [x] Tests: the same model ID loaded locally and clustered (with a fake transport) gives two entries; evicting one leaves the other; evicting the clustered one calls its teardown. #cluster