---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m87y86k3ypne6gcrsc823x7z
  text: |-
    Research notes (picked up, in doing):
    - LanguageModel.callAsFunction/nextTokenLogits/trim/copy are synchronous and non-throwing; only prepare/newCache throw. So the transport must be synchronous (blocking) and a failure in a non-throwing op needs a sticky failure state + task cancellation (generation loop checks Task.isCancelled each token).
    - MLXLanguageModel takes a `load` closure that returns a ModelContainer, so a container over ClusterLanguageModel slots in with no change to the executor.
    - Prompt-cache code reads caches only through the KVCache protocol (offset, maxTrimCount, isTrimmable(after:), trim, copy, residentByteCount) plus type switches in KVCacheTree (checkpoint strategy), PromptCacheCheckpoint.canLoseRewind, CacheLayerKind. A remote cache needs a `.remote` kind in KVCacheTree so KVCacheStorage.checkpoint() and PromptCacheCheckpoint pick trim or copy.
    - PromptCacheCheckpoint.prefill calls the full-sequence forward and drops the logits; for a remote model that would ship every position's logits. nextTokenLogits keeps cache writes and state by contract.
    - Disk spill/saveCache go through preparePromptCacheSave, which throws for a cache type with no serializedClassName.
    - MTP: drafters fall back with .incompatibleDrafter for an unknown target; MLXLanguageModel needs an explicit refusal.
    - Coordinator requirements from the worker session: request id + monotonic sequence number on every op, app-allocated UInt64 handles, fixed-size Int32 announce encoding (opcode, handle, count, stop flag, token payload <= 1024) with round-trip test, a capabilities/version op, typed values only.
    - Test model: Qwen35MTPFixture (tiny hybrid GDN + attention, seeded weights) with ScalarTokenizer.
  timestamp: 2026-11-24T23:28:56.931083+00:00
- actor: claude-code
  id: 01m87z8sfk7680svj4bsfgj89x
  text: |-
    ## Final op set (v1) — for the FoundationModelsCluster worker session

    Source: Libraries/MLXLMCommon/Remote/ (RemoteModelProtocol.swift is the DocC home of the table below).

    Envelope: `RemoteModelRequest { requestID: RemoteRequestID(UInt64), sequence: UInt64, op: RemoteModelOp }`. Sequence starts at 1 and grows by exactly one per request the app sends (a rank that sees a gap must fail). Request id: allocated by the app at `begin`; ops outside a begin/end pair carry the last id (0 before the first). Everything is Codable + Sendable + Hashable; no op carries a path, command or free text (only `.failed(String)` replies carry text, worker->app).

    `RemoteModelProtocol.version = 1`, `RemoteModelProtocol.maximumChunkTokenCount = 1024`.

    | op | reply |
    |---|---|
    | describe | capabilities(RemoteModelCapabilities{protocolVersion:Int32, modelFamilies:[String], supportsSharding:Bool, rankCount:Int, vocabularySize:Int, defaultPrefillStepSize:Int, maximumPrefillStepSize:Int?}) |
    | begin | done |
    | end | done (announce raises the stop flag) |
    | newCache(RemoteCacheID) | cache(RemoteCacheStatus) |
    | forward(tokens:[Int32] (<=1024), cache:RemoteCacheID, logits: RemoteLogitsScope .none/.lastPosition/.allPositions) | forwarded(RemoteCacheStatus, logits: RemoteLogits?) |
    | trim(RemoteCacheID, count:Int) | trimmed(Int, status: RemoteCacheStatus) |
    | copy(RemoteCacheID, into: RemoteCacheID) | cache(status of the copy) |
    | reset(RemoteCacheID) | cache(RemoteCacheStatus) |
    | release(RemoteCacheID) | done |
    | any | failed(String) |

    - Cache handles (`RemoteCacheID`, UInt64) are allocated by the app (newCache and the copy target). One handle = the caches of EVERY layer of one sequence (one round trip per trim/copy, not per layer). The worker should keep the per-handle `LMOutput.State` beside the caches and copy it on `copy` (the in-process test worker does).
    - Logits: `RemoteLogits { shape:[Int], elementType: float32|float16|bfloat16, bytes: Data }`, row-major, in the model's own dtype so app sampling is bit-identical. `RemoteLogits(MLXArray)` builds it worker-side; the cluster transport can fill `bytes` from shared memory.
    - Status: `RemoteCacheStatus { offset, maxSize?, evictedTokenCount, isTrimmable, maxTrimCount, rewindLimit: .always/.within(Int)/.never, canLoseRewind, residentByteCount }`. Build it worker-side with `RemoteCacheStatus(describing: caches)` (public). The app mirrors it, so offset/maxTrimCount/isTrimmable(after:) cost no round trip.
    - Announce: `RemoteOpAnnouncement.encode(request) -> [Int32]` / `decode([Int32])` / `stopsLoop(words)`. Fixed `wordCount = 11 + 1024`. Words: [0] opcode (1 describe, 2 begin, 3 end, 4 newCache, 5 forward, 6 trim, 7 copy, 8 reset, 9 release; 0 is never an op so a zero buffer never decodes), [1] flags (bit0 stop, bits1-2 logits scope), [2] count (forward tokens / trim count), [3,4] sequence lo/hi, [5,6] request id lo/hi, [7,8] handle lo/hi, [9,10] copy target lo/hi, [11...] token payload. Rank 0 writes, others write zeros, all-sum gives rank 0's words. Round-trip tests: Tests/MLXLMTests/RemoteModelProtocolTests.swift.
    - Transport: `protocol RemoteModelTransport: Sendable { func send(_ request: RemoteModelRequest) throws -> RemoteModelReply }` — synchronous/blocking, called from the generation thread, one request in flight at a time.
  timestamp: 2026-11-24T23:46:44.851622+00:00
- actor: claude-code
  id: 01m87z9ca2t4vz97yy5ca72xh6
  text: |-
    ## Cache-access audit (file:line on the working tree, and the decision)

    Routed through ops (work remotely, tested):
    - MLXLMCommon/PromptCacheReusePolicy.swift:464 rewindPromptCache, :572 reconcilePromptCache (:591 maxTrimCount, offsets) — read the mirrored RemoteCacheStatus; trim is the `trim` op. Test: aReconciledPromptCacheTrimsOnTheWorker.
    - MLXLMCommon/KVCache.swift trimPromptCache/canTrimPromptCache — protocol only; one handle so one `trim` op.
    - MLXLMCommon/PromptCacheCheckpoint.swift:170 canLoseRewind — NEW remote branch reads status.canLoseRewind (a remote cache with a sliding window is trimmable now but loses rewind later). :185 saved() then copies via `copy` op; :210 restored() copies again. Test: aPromptCheckpointCopiesARemoteCacheWithASlidingWindow (RED without the branch).
    - MLXLMCommon/PromptCacheCheckpoint.swift:263 prefill — CHANGED to nextTokenLogits (logits were discarded; the full forward would ship every position's logits). Same contract LanguageModel.prefill in TokenIterators already uses.
    - MLXLMCommon/KVCacheTree.swift:49 — NEW `.remote` leaf kind: checkpoint strategy copy when canLoseRewind else trim; supports only full precision; no TurboQuant protection. KVCachePlan.swift:434 checkpoint()/:475 restore() then work (test theCacheStorageOfAHybridRemoteCacheReturnsToItsCheckpoint, RED without it). KVCacheRound.swift (write-through like simple) and KVCacheRuntime.swift:287 (status like unsupported) take the new case.
    - MLXLMCommon/CacheConfiguration.swift CacheLayerKind(cache:) — default `.attention(maxSize:)`, fine.
    - MLXLMCommon/ChatSession.swift: checkpoints, trim (~2055), copy (~2098), fork copy (2792) — all via protocol/ops. Test: aChatSessionReusesItsPromptCacheAsTheLocalSessionDoes (3 turns, identical text and cachedPromptTokenCount).
    - MLXFoundationModels/ExecutorPromptCache.swift:150 newCache, :274/:421 caches.first?.offset; ExecutorPromptCacheEntry.swift:126 residentByteCount (= worker-reported bytes, so the memory budget weighs rank-0 cache memory), :168 copy() (session copy -> `copy` op). Tests: ClusterExecutorTests (plain, 2-turn reuse, allowed tool, required tool, schema).
    - MLXFoundationModels/MLXLanguageModel.swift:612 warmUp — one-token generate through the remote model; works unchanged.
    - MLXGuidedGeneration/GuidedTarget.swift:60 newCache, prefill (prepare -> .tokens -> model(...)), feedOneAtATime — ClusterLanguageModel.prepare leaves at most one step (<=1024) so the all-positions forward stays small. Tests: required tool + schema turns compare the guided decoder's sampled tokens.

    Refused with a clear error:
    - Speculative decoding (MTP and draft model): ClusterModelError.unsupported(.speculativeDecoding) via ClusterModelError.refuseSpeculativeDecoding at MTPSpeculation.swift:113, MTPSpeculativeTokenIterator.swift:449 (before drafter compatibility), TokenIterators.swift:788 (SpeculativeTokenIterator), ChatSession.swift:1864 (before the drafter loads), MLXLanguageModel.swift:193 (loadMTPSpeculation). Driven by new LMModelCapabilities.remoteForward. Tests for each path. MLXGuidedGeneration GuidedSpeculation is reached only with a speculation, which the executor refuses first.
    - Prompt-cache files: ChatSession.swift:2911 saveCache / savePromptCache and ExecutorPromptCacheStore.swift:761 spill -> ExecutorPromptCacheFile.swift:99 prepare -> preparePromptCacheSave throws "RemoteKVCache has no serializedClassName, so it cannot be saved". For the store this means an entry evicted for the memory budget is dropped (its handles release on the worker) instead of spilled. ExecutorPromptCacheSlot.swift:188 (disk restore templates) is unreachable for remote models since nothing is ever spilled.
    - VLM inputs: ClusterLanguageModel.prepare throws unsupported(.mediaInput) for image/video/audio.
    - KV-cache configuration / maxKVSize / kvBits: newCache throws unsupported(.cacheConfiguration) (the worker builds its own caches; app-side compression would need leaf rewrites).
    - Batched input: unsupported(.batchedInput) in prepare.

    No app-side change needed:
    - CompiledDecodeSegments / CompiledDecodeSchedule.swift:134,184 (`as? MambaCache`) run inside model code, i.e. on the worker. Turning compile off for sharded models is a worker decision.
    - ChatSession.swift:2034/2114 PreparedInputSplitting and :2406 BlockDiffusion casts: ClusterLanguageModel is neither.
    - RemoteKVCache.update/makeMask are preconditionFailure (no app-side layer ever calls them); state/metaState are empty.

    Unrelated observation (not changed): PromptCacheCheckpoint.canLoseRewind does not look inside CacheList, so a local Inkling 2 cache (CacheList(RotatingKVCache, SpeculativeConvolutionCache)) reads as never losing rewind in `applies(to:)` (no `adding:`). Filed as a separate task.
  timestamp: 2026-11-24T23:47:04.130737+00:00
- actor: claude-code
  id: 01m87zq9fh18zcns2x444fbfxh
  text: |-
    ### implement — changed
    - evidence: commit a7f2c9a4 on `cluster`. New: Libraries/MLXLMCommon/Remote/{RemoteModelProtocol,RemoteCacheStatus,RemoteOpAnnouncement,RemoteModelTransport,RemoteKVCache,ClusterLanguageModel,ClusterModelLoading}.swift; Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift; Tests/MLXLMTests/{RemoteModelProtocolTests,ClusterLanguageModelTests}.swift; Tests/MLXFoundationModelsTests/ClusterExecutorTests.swift. Changed: KVCacheTree/KVCacheRound/KVCacheRuntime (.remote leaf), PromptCacheCheckpoint (remote canLoseRewind; prefill via nextTokenLogits), LanguageModel (LMModelCapabilities.remoteForward), Load (WeightFileSelection.skipped), ModelFactory (load helper internal), speculation refusals in MTPSpeculation/MTPSpeculativeTokenIterator/TokenIterators/ChatSession/MLXLanguageModel.
    - tests: all 5 bundles green, run with xctest. MLXLMTests 2675 Swift Testing + 1260 XCTest, 0 failures (3 known issues, existing, not from this change). MLXFoundationModelsTests 611, MLXGuidedGenerationTests 95, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 (XCTest). New: RemoteModelProtocolTests 13 tests (parameterized), ClusterLanguageModelTests 23, ClusterExecutorTests 6. RED checked by removing the .remote kind and the remote canLoseRewind branch.
    - Identical tokens: greedy generate, long prompt (2500 tokens, prefill in chunks of at most 1024), 3-turn ChatSession with the same cachedPromptTokenCount, reconcile+trim on an attention-only model, executor plain/2-turn/allowed tool/required tool/schema (guided decoder sampled tokens compared).
    - Open questions for the worker side:
      1. Request end: LanguageModel has no hook at the end of a generation, so the app sends `end` for the open request at the next `prepare` and then `begin`. Ops outside prepare (prompt-cache checkpoint feeds, releases from deinit) carry the last request id. Is that fine for the per-token announce loop, or should the loop stay open between requests?
      2. Failure model: any transport error or `.failed` reply makes the session fail permanently (every later op throws without sending). A non-throwing forward that fails cancels the current Task, so the generation ends with stopReason .cancelled, and the next prepare/newCache throws ClusterModelError.transportFailed. Reconnecting is up to the cluster package (make a new ClusterLanguageModel).
      3. Model state: the app sends no LMOutput.State. The worker keeps it per handle and copies it on `copy`. The Qwen 3.5/Inkling text paths return none, so this matters only for future families that carry state.
      4. residentByteCount is what rank 0 reports, and it feeds the app's prompt-cache memory budget. Over budget, an entry is dropped (not spilled) and its handles are released.
      5. KV-cache configuration (kvBits, maxKVSize, KVCacheConfiguration) is refused for now. Supporting it would mean sending the configuration with newCache.
      6. Logits travel in the model's dtype (float32/float16/bfloat16 bytes). The cluster transport can fill `RemoteLogits.bytes` from shared memory.
      7. Should `InProcessRemoteModelWorker`'s op dispatch move into the library as a reusable worker-side executor? It is test support today.
    - next: /review
  timestamp: 2026-11-24T23:54:39.985301+00:00
position_column: done
position_ordinal: ffffef80
title: 'ClusterLanguageModel seam: remote forward and cache ops'
---
Let a `ModelContainer` hold a model whose forward pass and KV/recurrent cache operations run in a remote worker process (the rank-0 worker in FoundationModelsCluster). Everything around them stays in-process and unchanged: sampling, logit processors, grammar, stop strings, tool and reasoning parsing, and prompt-cache policy in `MLXLanguageModel`. Design: `cluster.md` in swissarmyhammer/FoundationModelsCluster (Architecture § "The app drives; workers replay").

- [x] Define a transport-agnostic protocol (e.g. `RemoteModelTransport`) with ops: begin/end request, forward(tokens, cache handle) → last-position logits, newCache, trim, copy, reset. No networking in this repo; FoundationModelsCluster implements the transport (SSH stdio + shared-memory logits).
- [x] `ClusterLanguageModel`: a `LanguageModel` whose `callAsFunction`/`prepare` and `newCache` go through the transport. Its caches are handles (a `KVCache` type the cache code can trim/copy/reset through ops).
- [x] Audit: every cache operation must go through the model. Check `PromptCacheReusePolicy`, `ChatSession`, `KVCacheTree`, prompt-cache snapshots and spooling, `MLXLanguageModel` warmup/guided/tool phases (MLXLanguageModel.swift ~647-655, 987, 1757, 2116). Anything that reads cache tensors directly must be disabled for clustered models or routed through an op.
- [x] v1 limits, each rejected with a clear error: VLM inputs (workers need embeddings), MTP/speculative decoding, prompt-cache snapshots to disk, CompiledDecodeSegments (compile off). Prefill chunked to ≤ 1024 tokens per step.
- [x] Tests: an in-process fake transport that wraps a real local model. A clustered container must produce identical tokens to the local one across generation, a tool-call turn, a guided-generation turn and prompt-cache reuse/trim.

Depends on nothing in this repo. FoundationModelsCluster's worker consumes the same op set, so agree on the op enum with that session (foundationmodelscluster-02). #cluster