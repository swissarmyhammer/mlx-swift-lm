---
assignees:
- claude-code
depends_on:
- 01M87WJZQS8TVVF0QD9RPAFC4Q
position_column: todo
position_ordinal: '8980'
title: Library worker-side executor for the remote model op set
---
The FoundationModelsCluster worker needs exactly what the test-support `InProcessRemoteModelWorker` (Tests/MLXLMTestSupport) does: take a `RemoteModelRequest`, run it against a loaded (and sharded) model, and return a `RemoteModelReply`. Move that dispatch into the library as a public, networking-free executor (e.g. `RemoteModelExecutor` in MLXLMCommon/Remote), so the worker only adds transport, JACCL and the allSum announce.

- [ ] Public `RemoteModelExecutor(model:configuration:)` with `handle(_ request: RemoteModelRequest) throws -> RemoteModelReply`. It covers every op (describe, begin/end, newCache, forward, trim, copy, reset, release), keeps per-handle cache state, and enforces the sequence and request-id rules.
- [ ] A hook so a rank > 0 can apply an op decoded from `RemoteOpAnnouncement` to its own shard the same way rank 0 does: the same executor, with replies discarded on non-zero ranks.
- [ ] `describe` reports the model families and sharding support from `TensorParallelShardable` conformance.
- [ ] Test support's `InProcessRemoteModelWorker` becomes a thin wrapper around it, so the existing 42 tests keep passing.
- [ ] Tests: a two-executor run over a ring group of 2 (ShardCheck-style, one Mac) produces the same tokens as local.

Depends on rpafc4q (done in a7f2c9a4). #cluster