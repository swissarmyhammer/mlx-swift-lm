---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m87zz739gfqy7dw2amjmyey8
  text: |-
    Picked up. Research:
    - InProcessRemoteModelWorker (test support) dispatches ops but enforces no sequence or request-id rule, and overwrites a held handle on newCache. The executor adds those rules; ClusterLanguageModel's RemoteModelSession already sends sequence 1,2,3... and ends an open request before begin, so the rules hold for every existing test.
    - Shardable families (by the factories ShardCheck loads: LLM first, then VLM): qwen3_5, qwen3_8 (Qwen35Model), qwen3_5_text, qwen3_8_text (Qwen35TextModel), inkling2, inkling2_loop (Inkling2). Not included: qwen3_5_moe (Qwen35MoE does not conform), prism_hadamard_qwen35 (subclass of Qwen35Model, but its substituted rotated projections are not stock projections, so its shard throws).
    - MLXDistributed.allSum runs on the CPU stream; the model stays on the default GPU stream.
    - Coordinator added: a PARK opcode in RemoteOpAnnouncement + isPark (rank 0 announces it after ~250 ms idle inside a request; members leave the allSum loop; no state change, next op keeps request id and sequence+1), and static RemoteModelExecutor capabilities for the planner before any load.
    Plan: RemoteModelExecutor (MLXLMCommon/Remote) with handle(_:), apply(announcement:), static capabilities; Configuration(vocabularySize:rankCount:) plus a config.json reader; RemoteOpAnnouncement.park/isPark + an allSum exchange helper; test fake wraps the executor; ShardCheck --announce mode + shard-check.sh announce run comparing tokens with a local generation.
  timestamp: 2026-11-24T23:58:59.689155+00:00
- actor: claude-code
  id: 01m881sehps6zqbsbt91abwnrg
  text: |-
    ### implement — changed
    - evidence: commits 38ff740d (feat(cluster): RemoteModelExecutor), b8be1963 (test(shard-check): announce loop). Files: Libraries/MLXLMCommon/Remote/{RemoteModelExecutor (new), RemoteOpAnnouncement, RemoteModelProtocol}.swift; Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift; Tests/MLXLMTests/{RemoteModelExecutorTests (new), RemoteModelProtocolTests}.swift; tools/ShardCheck/{AnnounceLoop (new), ShardCheck}.swift; scripts/shard-check.sh, scripts/shard_check_compare.py.
    - API:
      - `RemoteModelExecutor(model:configuration:)`; `Configuration(vocabularySize:rankCount:)` or `Configuration(modelDirectory:rankCount:)` (reads vocab_size or text_config.vocab_size from config.json).
      - `handle(_:) throws -> RemoteModelReply`: rule breaks (sequence gap expected/found, request mismatch, begin while open, stale begin id, end with none open, unknown handle, held handle on newCache/copy target, empty forward, >1024 tokens) give `.failed(text)` and change nothing; a request with the right sequence consumes it even if a request rule fails. Only a model error (newCache throwing) is thrown.
      - `apply(announcement:) throws -> RemoteModelReply?`: ranks > 0; nil for a park. Every rank must apply the same ops in the same order (forward included: the sharded layers all-sum).
      - `cacheCount`; static `shardableModelFamilies` (inkling2, inkling2_loop, qwen3_5, qwen3_5_text, qwen3_8, qwen3_8_text; a test creates each through LLM/VLM registry and checks TensorParallelShardable) and static `capabilities: RemoteWorkerCapabilities` (version, families, supportsSharding true). describe = the same families + supportsSharding && model is TensorParallelShardable, rankCount, vocab, prefill steps.
      - `RemoteOpAnnouncement.park(requestID:)`, `isPark(_:)` (opcode 10; decode throws .parkHasNoRequest; stop flag not set), `exchange(_:group:)` (allSum of the Int32 words on the CPU stream; rank 0 passes words, others nil).
    - Tests: all 5 bundles green via xctest: MLXLMTests 2715 Swift Testing (3 existing known issues) + 1260 XCTest; MLXFoundationModelsTests 611; MLXGuidedGenerationTests 95; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. 42 existing remote tests pass unchanged over the wrapper. New: 40 tests (RemoteModelExecutorTests 33 incl. 6 parameterized families, + 7 protocol tests). Mutation check: removing the sequence rule, the park branch, the conformance check or the held-handle check each failed its test.
    - 2-rank announce loop (scripts/shard-check.sh --announce, ring, one Mac, 9 greedy tokens, one park mid-request, rank 1: 15 ops, 1 park, 0 failed):
      - Qwen3.5-2B-mxfp4 (downloaded to HF cache, no local snapshot existed): default prompt PASS; story prompt PASS. 14.3-15.5 ms/token sharded vs 1.8 local; plain sharded GreedyRun also 14.3, so the announce adds no measurable cost.
      - Inkling-2-Mini-4bit: default prompt PASS, 36-40 ms/token vs 9.8 local (plain sharded 36.2). Story prompt diverges at token 7: an exact bf16 tie (tokens 12 and 17 both 22.08). The announce-loop tokens equal the plain sharded run and the plain full-forward local run; only the local TokenIterator path picks 17. Not a protocol bug.
    - Notes for the worker session: ShardCheck's AnnouncingTransport/AnnounceFollower (tools/ShardCheck/AnnounceLoop.swift) is the reference loop. ShardCheck closes with an explicit `end` (ClusterLanguageModel sends end only at the next prepare) and refuses later sends, so cache releases after close go nowhere. Park carries the request id but the executor does not check it.
    - next: /review
  timestamp: 2026-11-25T00:30:47.862674+00:00
depends_on:
- 01M87WJZQS8TVVF0QD9RPAFC4Q
position_column: done
position_ordinal: fffff080
title: Library worker-side executor for the remote model op set
---
The FoundationModelsCluster worker needs exactly what the test-support `InProcessRemoteModelWorker` (Tests/MLXLMTestSupport) does: take a `RemoteModelRequest`, run it against a loaded (and sharded) model, and return a `RemoteModelReply`. Move that dispatch into the library as a public, networking-free executor (e.g. `RemoteModelExecutor` in MLXLMCommon/Remote), so the worker only adds transport, JACCL and the allSum announce.

- [x] Public `RemoteModelExecutor(model:configuration:)` with `handle(_ request: RemoteModelRequest) throws -> RemoteModelReply`. It covers every op (describe, begin/end, newCache, forward, trim, copy, reset, release), keeps per-handle cache state, and enforces the sequence and request-id rules.
- [x] A hook so a rank > 0 can apply an op decoded from `RemoteOpAnnouncement` to its own shard the same way rank 0 does: the same executor, with replies discarded on non-zero ranks.
- [x] `describe` reports the model families and sharding support from `TensorParallelShardable` conformance.
- [x] Test support's `InProcessRemoteModelWorker` becomes a thin wrapper around it, so the existing 42 tests keep passing.
- [x] Tests: a two-executor run over a ring group of 2 (ShardCheck-style, one Mac) produces the same tokens as local.
- [x] (worker session) Reserve a PARK opcode in `RemoteOpAnnouncement` plus `isPark`; a park changes no executor state (same request id, sequence +1 on the next op). Test announce -> park -> resume.
- [x] (worker session) Static `RemoteModelExecutor.capabilities` for the planner before any load: protocol version, shardable model families, sharding support.

Depends on rpafc4q (done in a7f2c9a4). #cluster