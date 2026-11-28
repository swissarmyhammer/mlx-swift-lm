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
- actor: claude-code
  id: 01m8fswq97zr9afw3bazxqw39p
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (4b825b21) — 3 findings (3 confirmed, 2 refuted): Libraries/MLXLMCommon/Remote/RemoteOpAnnouncement.swift:230, Tests/MLXLMTests/ClusterQuantizedKVCacheTests.swift:311, Tests/MLXLMTests/RemoteModelProtocolTests.swift:89
    - next: implement the 3 open items in ## Review Findings (2026-11-27 18:40), then re-review
  timestamp: 2026-11-28T00:46:41.959060+00:00
- actor: claude-code
  id: 01m8fsx5dsf435txee0jyj0kj8
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — protocol v5, RemoteKVCacheQuantization.swift + ClusterQuantizedKVCacheTests.swift (15 tests)
    - test: green — all 5 bundles (MLXLMTests 2843+1261, MLXFoundationModelsTests 654, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 4b825b21
    - review: findings — RemoteOpAnnouncement.swift:230, ClusterQuantizedKVCacheTests.swift:311, RemoteModelProtocolTests.swift:89
  timestamp: 2026-11-28T00:46:56.441934+00:00
- actor: claude-code
  id: 01m8ft5p5a3b9j4ym9x43v23bv
  text: 'Review findings pass: (1) RemoteOpAnnouncement: one fileprivate protocol PayloadWord (RawValue == Int) that RoundWord and QuantizationWord conform to, with one `index` property in a protocol extension; both payloadIndex overloads deleted. A generic `payloadIndex(_ word: some PayloadWord)` did NOT compile: implicit member syntax (`.token`) cannot infer the concrete type through `some`, so call sites now name the enum (`RoundWord.token.index`). (2) ClusterQuantizedKVCacheTests: both ChatSession tests call `expectTheLocalTurns(local:clustered:fixture:)` (forwards sourceLocation), prompts in `chatPrompts`. (3) RemoteModelProtocolTests: round-trip test of compressionStart Int(Int32.max) next to the rejection test, both via `newCacheRequest(compressionStart:)`. The round-trip test covers existing behavior, so it passes without a production change (boundary coverage, no RED possible).'
  timestamp: 2026-11-28T00:51:35.722741+00:00
- actor: claude-code
  id: 01m8ftfqn3ns5j3ege8358ftzm
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/Remote/RemoteOpAnnouncement.swift, Tests/MLXLMTests/ClusterQuantizedKVCacheTests.swift, Tests/MLXLMTests/RemoteModelProtocolTests.swift; 3/3 findings checked. swift-format applied; swift build --build-tests clean. xcrun xctest MLXLMTests (one process): Swift Testing 2844 tests / 281 suites passed (3 pre-existing known issues), XCTest 1261 executed 0 failures. MLXFoundationModelsTests: 654 passed.
    - next: /review
  timestamp: 2026-11-28T00:57:04.931708+00:00
- actor: claude-code
  id: 01m8fv08mfd6ppp9hyh07kgnek
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 source warnings); swift-format lint on the 3 changed files clean; xcrun xctest per bundle: MLXLMTests 2844 tests/281 suites passed (3 known issues, pre-existing, untouched files: AttentionKernelToleranceTests.aNearTieRejectsATokenOutsideTheTwoBest = intentional negative check; ToolTests MiniCPM5 CDATA close tag = existing withKnownIssue); MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 654 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed. 0 failures, 0 skipped.
    - next: review/commit
  timestamp: 2026-11-28T01:06:06.607357+00:00
- actor: claude-code
  id: 01m8fv2g4fhafss7yggr7hc0pw
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (72e6cd60): 0 findings (7 attempted, 0 failed); 3 prior findings all checked
    - next: task moved to done
  timestamp: 2026-11-28T01:07:19.823218+00:00
- actor: claude-code
  id: 01m8fv2tam6ze76e20abhe60z0
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 3 files, all 3 findings checked
    - test: green — all 5 bundles (MLXLMTests 2844, MLXFoundationModelsTests 654, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 72e6cd60
    - review: clean — 0 findings
  timestamp: 2026-11-28T01:07:30.260900+00:00
position_column: done
position_ordinal: ffffff8b80
title: 'Clustered models: support a quantized KV cache (kvBits 8/4) on the workers'
---
## What
ClusterLanguageModel.newCache (Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift:102) throws unsupported(.cacheConfiguration) for any KV-cache configuration. Our best Qwen profile (q8kv8: Qwen3.8-27B-8bit, kvBits 8) failed on its first call in the cluster1q SWE-bench run (2026-11-27).

## Fix
Carry the KV-cache quantization (bits, group size) in the newCache op to each rank, so each worker makes a quantized cache for its heads, as the local path does (KV8 decode ^2h5wtct). Keep maxKVSize refused if it is not needed. Protocol version bump if the op changes.
Tests: the 2-Mac parity suite with kvBits 8 on Qwen3.8-27B matches the local greedy output under the same top-5 rule; a unit test that newCache with kvBits sends the config.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster

## Review Findings (2026-11-27 18:40)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 18 file(s) reviewed, 5 not reviewed.

> 5 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8EPPTTXQ9EAHS4XK2RY6YZ7.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8EPPTTXQ9EAHS4XK2RY6YZ7.md` — no validator matches this file
> - `.kanban/tasks/01M8FKC3V6YDH4490VK6PQ8CEE.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8FKC3V6YDH4490VK6PQ8CEE.md` — no validator matches this file
> - `Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/Remote/RemoteOpAnnouncement.swift:230` `reuse/reuse` — The new `payloadIndex(_ word: QuantizationWord)` overload has the same body as the existing `payloadIndex(_ word: RoundWord)` (`headerWordCount + word.rawValue`). Two copies of one index computation will diverge when the header layout changes. A shared helper, or a generic over a payload-word protocol, would keep one implementation. Give `RoundWord` and `QuantizationWord` a shared protocol (for example `PayloadWord` with `var rawValue: Int`), and write one `payloadIndex(_ word: some PayloadWord) -> Int`. Or extract a single `payloadIndex(offset: Int)` that both call.
- [x] `Tests/MLXLMTests/ClusterQuantizedKVCacheTests.swift:311` `reuse/reuse` — The turn loop of `aChatSessionWithMTPAndKVBitsReusesItsPromptCacheAsTheLocalOneDoes` is a copy of the loop in `aChatSessionOfKVBitsReusesItsPromptCacheAsTheLocalSessionDoes`. Both build the same local and clustered `ChatTurn` arrays, then run the same two assertions. Any change to the turn protocol has to be made twice. Extract a helper, for example `private static func chatTurns(local: ChatSession, clustered: ChatSession, fixture: Qwen35MTPFixture) async throws -> (local: [ChatTurn], clustered: [ChatTurn])`, or a helper that checks both sessions and asserts. Call it from both tests.
- [x] `Tests/MLXLMTests/RemoteModelProtocolTests.swift:89` `completeness/inverse-operation-coverage` — The new compression-start tests only prove that Int32.max + 1 is rejected. Nothing proves that the largest accepted start, Int32.max itself, survives its announce words. The encode side's accept boundary is therefore unverified, and an off-by-one in the bound would pass these tests. The round-trip test at line 142 covers the matching boundary for draftLimit, so this is a one-sided boundary test. Add a round-trip test that encodes and decodes `.newCache` with `compressionStart: Int(Int32.max)` and asserts equality with the request, paired with the existing rejection test.
