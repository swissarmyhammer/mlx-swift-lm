---
comments:
- actor: claude-code
  id: 01m8fr0j643mfbx4xh8zr1nse0
  text: |-
    Research (implement):
    - Local KV8 path: model.newCache(parameters) makes KVCacheSimple; the KVCachePlan converts simple caches to QuantizedKVCache at fixed app-side points: TokenIterator after each step() (the prefill tail and each decode step) and applyAndValidate after prefill; MTPSpeculativeTokenIterator applyAndValidate after prefill, apply after each round. Prefill chunks inside model.prepare and PromptCacheCheckpoint.prefill (executor checkpoint feed) do NOT convert, so a local checkpoint stays full precision.
    - Local MTP with kvBits on Qwen 3.5 speculates (no passthrough), quantized main cache trims fine, drafter cache stays full precision (Qwen35MTPCacheKindTests.quantizedMainCacheMatchesPlainDecode).
    - Why the worker cannot just convert after every logits forward: PromptCacheCheckpoint.prefill asks nextTokenLogits (scope .lastPosition on the wire) for chunks whose logits nobody reads, so the worker would quantize at the checkpoint where local does not -> diverges from local in the FoundationModels executor path.
    - Design chosen: the app keeps the policy. newCache carries the affine quantization (bits, group size, compression start, compatibility policy) to every rank; each rank makes the caches with GenerateParameters(kvCache:) so a model that quantizes up front does so; a new op `quantize` converts at exactly the points where the local plan applies (KVCachePlan.apply on a RemoteKVCache leaf). The status carries `awaitsQuantization`, so once a cache is converted the app sends no more quantize ops (one extra round trip per conversion: once per first turn for quantizedKVStart 0).
    - Typed configs (MLXLanguageModel.kvCache, default compatibility .requireAtLeastOneLayer) validate app-side against the RemoteKVCache leaf, which today reports support only for full precision -> must report support for the strategy its worker applies.
    - maxKVSize / capacity, rewind, turboQuant and varianceNormalized stay refused with unsupported(.cacheConfiguration).
    - Fixture: Qwen35MTPFixture(headDimension: 64) is needed, the default head dim 8 is not quantizable (conversion silently skipped).
  timestamp: 2026-11-28T00:13:50.660855+00:00
- actor: claude-code
  id: 01m8frfmz8sc2x47eqn9zpz7yk
  text: |-
    RED: tests written first (Tests/MLXLMTests/ClusterQuantizedKVCacheTests.swift, additions to RemoteModelProtocolTests, RemoteModelExecutorTests, MLXFoundationModelsTests/ClusterExecutorTests); `swift build --build-tests` failed on the missing RemoteKVCacheQuantization / RemoteModelOp.quantize / newCache(_:quantization:) — the feature was absent, and the kvBits parity tests would have thrown unsupported(.cacheConfiguration) (the old test `a KV-cache configuration is refused` asserted exactly that; deleted, replaced by maxKVSize and TurboQuant refusals).
    Note: `xcrun xctest -XCTest <filter>` does not select Swift Testing suites; it ran 0 tests. Running the whole bundle is the only way.
    Implementation landed: protocol v5 (newCache carries the quantization in the count word + 3 payload words; opcode 12 quantize; RemoteCacheStatus.awaitsQuantization). KVCachePlan's fast apply calls RemoteKVCache.quantize(for:) on remote leaves; KVCacheTree reports a remote leaf as supporting the affine strategy it was made under (typed configs with requireAtLeastOneLayer validate). ClusterMTPTokenIterator now applies the plan after prepare and after each round, as MTPSpeculativeTokenIterator does.
  timestamp: 2026-11-28T00:22:05.032135+00:00
- actor: claude-code
  id: 01m8fs0085zqeac5jv0yrsp7w4
  text: |-
    GREEN. Full bundles, one process each (xcrun xctest .build/out/Products/Debug/<B>.xctest after swift build --build-tests, swift-format applied, build has no warnings):
    - MLXLMTests: Swift Testing 2843 tests / 281 suites passed (3 known issues, pre-existing withKnownIssue), XCTest 1261 executed 0 failures. New suite "A cluster model quantizes the KV caches of its workers" passed.
    - MLXFoundationModelsTests: 654 passed (incl. aSecondTurnUnderAnEightBitKVCache{WritesTheLocalText,ReusesThePromptCacheAsTheLocalTurnDoes}).
    - MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 XCTest: all passed.
    Parity in-process is exact token equality (stronger than the top-5 rule), because the worker converts after exactly the forwards the local KVCachePlan converts after.
    Real 2-Mac Qwen3.8-27B parity: NOT run. It needs the FoundationModelsCluster repo rebuilt against this protocol v5, and that repo will not compile unchanged: Sources/ClusterProtocol/ForwardTokenCount.swift switches exhaustively on RemoteModelOp (needs `.quantize`), IntegrationTests/Sources/fmcluster-parity/DecodeRun.swift matches `.newCache(let cache)` (now `.newCache(let cache, _)`), and Recording.swift `name` lacks `.speculate` already and will lack `.quantize`. Those edits belong to the Cluster repo.
    Also noticed (not changed, out of scope): distributed-inference.md still says the model refuses MTP speculative decoding, which ^rgzk56j made untrue.
  timestamp: 2026-11-28T00:31:00.869777+00:00
- actor: claude-code
  id: 01m8fs055q4380f4typm4cmgsa
  text: |-
    ### implement — changed
    - evidence: RemoteModelProtocol.version 4 -> 5. New Libraries/MLXLMCommon/Remote/RemoteKVCacheQuantization.swift; changed ClusterLanguageModel, ClusterMTPTokenIterator, RemoteCacheStatus, RemoteKVCache, RemoteModelExecutor, RemoteModelProtocol, RemoteModelTransport, RemoteOpAnnouncement, KVCacheRuntime, KVCacheTree, distributed-inference.md. Tests: new Tests/MLXLMTests/ClusterQuantizedKVCacheTests.swift (15 tests: config sent to every rank, quantized bytes per rank and vs local, one quantize op, maxKVSize/TurboQuant refused, kvBits 8/4 + late start + typed affine parity, ChatSession reuse, MTP parity/drafts/rank offsets/ChatSession MTP reuse); RemoteModelProtocolTests (newCache+quantization and quantize round trips, bad start, bad compatibility); RemoteModelExecutorTests (bad bits, awaitsQuantization x3, unknown handle); ClusterExecutorTests (2 eight-bit executor turn tests); test support kvCache params. All five bundles green.
    - next: /review. Real 2-Mac run not done (Cluster repo must adopt protocol v5 first).
  timestamp: 2026-11-28T00:31:05.911760+00:00
- actor: claude-code
  id: 01m8fsg9pn6xp8cjd52vk4kxjt
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 warnings, 0 errors); xcrun xctest per bundle: MLXLMTests 1261 XCTest + 2843 Swift Testing passed, 0 failures; MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 654 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed; 0 skipped; swift-format lint on all changed Swift files clean. MLXLMTests reports 3 known issues (withKnownIssue), pre-existing in files this change does not touch: AttentionKernelToleranceTests.swift:47 (aNearTieRejectsATokenOutsideTheTwoBest) and ToolTests.swift:2560-2561 (MiniCPM5 CDATA close tag).
    - next: review
  timestamp: 2026-11-28T00:39:54.837308+00:00
position_column: doing
position_ordinal: '80'
title: 'Clustered models: support a quantized KV cache (kvBits 8/4) on the workers'
---
## What
ClusterLanguageModel.newCache (Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift:102) throws unsupported(.cacheConfiguration) for any KV-cache configuration. Our best Qwen profile (q8kv8: Qwen3.8-27B-8bit, kvBits 8) failed on its first call in the cluster1q SWE-bench run (2026-11-27).

## Fix
Carry the KV-cache quantization (bits, group size) in the newCache op to each rank, so each worker makes a quantized cache for its heads, as the local path does (KV8 decode ^2h5wtct). Keep maxKVSize refused if it is not needed. Protocol version bump if the op changes.
Tests: the 2-Mac parity suite with kvBits 8 on Qwen3.8-27B matches the local greedy output under the same top-5 rule; a unit test that newCache with kvBits sends the config.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster