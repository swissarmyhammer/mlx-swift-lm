---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8854j8ct46smdcw0jtqj61h
  text: |-
    Implemented.
    - PARK check: `RemoteRequestStream.admitPark(_:)` holds a park to the same rule as an op (the open request's id, or the last one's when none is open; no sequence number taken). Rank 0: new `RemoteModelExecutor.handlePark(requestID:)` -> `.done` or `.failed(requestMismatch text)`. Ranks > 0: `apply(announcement:)` reads the park's id with the new `RemoteOpAnnouncement.parkedRequestID(_:)` and throws `RemoteModelExecutorError.requestMismatch`. `read(_:low:high:)` now backs both `decode` and `parkedRequestID` (no second copy of the half-word join). ShardCheck's `AnnouncingTransport` calls `handlePark` before announcing its park, and now parks with the id of the last request (it used the next request's id, which a park of a new `begin` would have got wrong under the new rule); the follower doc notes the new throw.
    - describe: `RemoteModelCapabilities` gains `modelType: String?` and `quantization: BaseConfiguration.Quantization?` (Quantization is now `Hashable`, was `Equatable`). `RemoteModelExecutor.Configuration` gains the same fields; `init(modelDirectory:)` reads `model_type` and `quantization` from config.json through `CheckpointFields` (was `VocabularyFields`; per-layer quantization entries are ignored). `RemoteModelProtocol.version` = 2.
    - ClusterLanguageModel side: `loadClusterModelContext` now refuses a worker whose reported `model_type` differs from the checkpoint's (`ClusterModelError.modelTypeMismatch(expected:found:)`); a worker or checkpoint with no model_type passes. InProcessRemoteModelWorker takes an optional `modelType`.
    - DocC: RemoteModelProtocol gets a "Between turns" section (end only at the next prepare; rank 0's park after ~250 ms idle is what frees the other ranks between turns). `shardableModelFamilies` doc says qwen3_5_moe / prism_hadamard_qwen35 wait for the larger-models phase; tracked as new task ^ze3xffb.

    Tests (MLXLMTests): RemoteModelExecutorTests 36 pass (new: park of another request fails on a member / gets failed reply on rank 0 / park of open request is done on rank 0; configuration reads model type + quantization; describe test now checks both fields), RemoteModelProtocolTests 20 pass, ClusterLanguageModelTests 26 pass (new: cluster context refuses a worker of another model type). RED was compile-level (missing API); a mutation run (park check removed, describe fields dropped, model-type check removed) failed the 4 behaviour tests as expected.

    ### implement — changed
    - evidence: 11 files — Libraries/MLXLMCommon/{BaseConfiguration.swift, Remote/RemoteModelProtocol.swift, Remote/RemoteOpAnnouncement.swift, Remote/RemoteModelExecutor.swift, Remote/RemoteModelTransport.swift, Remote/ClusterModelLoading.swift}, tools/ShardCheck/AnnounceLoop.swift, Tests/MLXLMTests/{RemoteModelExecutorTests,RemoteModelProtocolTests,ClusterLanguageModelTests}.swift, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift
    - next: review
  timestamp: 2026-11-25T01:29:17.836747+00:00
position_column: done
position_ordinal: fffff680
title: 'Executor follow-ups: park request-id check, loaded model_type in describe'
---
Small follow-ups from 8aeeqed (RemoteModelExecutor, 38ff740d).
- [x] A PARK whose request id doesn't match the open request gets a `.failed` reply on rank 0 and an error on ranks > 0, so a desynced rank fails loudly.
- [x] `describe`'s capabilities also name the loaded `model_type` (and quantization), not only the shardable family list. Bump `RemoteModelProtocol.version` to 2, since nothing has shipped. Update ClusterLanguageModel and the tests.
- [x] Note for the worker session: ClusterLanguageModel sends `end` only at the next `prepare`, so the worker's PARK-after-~250 ms idle is what releases members between turns. Document that in RemoteModelProtocol's DocC.
- [x] Note: `prism_hadamard_qwen35` and `qwen3_5_moe` aren't shardable yet (rotated projections, MoE). That's for the larger-models phase. #cluster