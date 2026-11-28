---
comments:
- actor: claude-code
  id: 01m8fhx01gdrr1tvxvkk519tbp
  text: |-
    Research (implement): Refusal sites: refuseSpeculativeDecoding in MTPSpeculativeTokenIterator.validate, SpeculativeTokenIterator.init, MTPSpeculation.makeTokenIterator, ChatSession (respond path), MLXLanguageModel.loadMTPSpeculation. Both drafters (Qwen35MTPDraftModel, Inkling2DraftModel) are ResumableMTPDrafterModel and use the target's embedding + LM head, which TensorParallelShardable leaves whole on every rank, so rank 0's sharded target can host the drafter. Drafter state cannot be trimmed without old hidden rows (MTPDrafterContinuation docs) -> a long trim cold-starts the drafter, as locally. Qwen hybrid verify needs mtpVerifyKeepsSnapshotsKey on EVERY rank and rewindSpeculativePromptCache (not trimPromptCache) for rejects/lookahead. Cluster package (../FoundationModelsCluster Sources/fmcluster-worker/Serve.swift) does exchange+handle per op; it constructs RemoteModelExecutor(model:configuration:) and will need to pass the drafter on rank 0 and a draft-exchange closure (follow-up outside this repo).
    Design: protocol v4 adds RemoteModelCapabilities.drafter and op .speculate(cache, bonus token, rejected of previous round, max drafts, p-min). One app<->rank0 round trip per round: all ranks rewind the previous round's rejects, rank 0 commits+drafts, a second all-sum inside the op hands the drafts to every rank, all ranks verify block+1 tokens, reply carries drafts + all-position logits. Rank 0 keeps a per-handle drafter track fed by every forward, so prompt-cache copies/trims/turns keep it aligned with the target cache. App side: RemoteMTPDrafter placeholder vended by ClusterLanguageModel, ClusterMTPTokenIterator does verification (sample-and-match), telemetry and finalize (trim rejected + lookahead).
  timestamp: 2026-11-27T22:27:02.320282+00:00
- actor: claude-code
  id: 01m8fkd0gszdrx6c2ndrtwpzc0
  text: 'Implementation landed (TDD: new suites RED first, then GREEN). Protocol v4: RemoteDraftRound/.speculate, .drafted reply, RemoteDrafterCapabilities, RemoteDraftExchange (alone/group; shares RemoteOpAnnouncement.sum). Worker: RemoteDrafterTrack per handle on rank 0 (cold/aligned/primed/verifying phases), fed by every forward, copied by copy, settled/rewound by trim (long trim -> cold, as locally); verify keeps GDN snapshot slots on every rank; trim op falls back to rewindSpeculativePromptCache. App: RemoteMTPDrafter (ClusterLanguageModel.mtpDrafter), ClusterMTPTokenIterator (one op per round, rejects folded into the next round, finalize trims rejects+lookahead), MTPSpeculation.makeMTPIterator dispatch used by generateTokens/generate/makeTokenIterator, ChatSession + MLXLanguageModel.loadMTPSpeculation use the worker drafter; no worker drafter -> fallback .incompatibleDrafter. Removed ClusterModelFeature.speculativeDecoding + refuseSpeculativeDecoding; draft-model speculation now runs on clusters over generic ops (test on an attention-only target). Discovery: with a drafter, rank 0 emits hidden rows on prefill chunks; Qwen35 projected every row when emitting even for nextTokenLogits -> fixed to honor lastTokenOnly (bit-identical last row via quantizedVocabularyProjectionInput), test Qwen35EmittingProjectionTests. Follow-up for the cluster worker repo: ^6pq8cee.'
  timestamp: 2026-11-27T22:53:15.673510+00:00
- actor: claude-code
  id: 01m8fmzraw8hgg8bb9y317gdc5
  text: 'Reverted the Qwen35 "honor lastTokenOnly when emitting" change and its test: Qwen35PrefillLogitsTests.testTextWarmContinuationAndMTPState pins that an emitting nextTokenLogits returns full logits (a deliberate contract), so that prefill optimization is out of scope. Discovery: after this change the full MLXLMTests run crashed (SIGBUS, stack overflow) in DeepSeekV4AttentionTests.aPrefillBlockReadsTheFusedKernelOnTheGPU, and HEAD passed. Cause: that test updates with verify: [] and exportToDot walks leftover initializer params, which are lazy arrays off the GLOBAL MLXRandom key chain. Inkling2TinyModels constructed Inkling2/Inkling2DraftModel outside its task-local random state, so each tiny Inkling the new suites build extended that chain. Fix: the initializers now run under a throwaway RandomState(seed: initializerSeed), and the seeded weight draws are unchanged. Working tree verified after the orchestrator''s stash: stash popped, nothing left in git stash list, tree matched backup/rgzk56j-wip before the later revert and fix.'
  timestamp: 2026-11-27T23:20:58.460249+00:00
- actor: claude-code
  id: 01m8fmzxjnwz99p7nyx9gh182g
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests clean (no warnings in changed files). xcrun xctest: MLXLMTests 1261 XCTest + 2819 Swift Testing passed, 0 failures (3 known issues, same as HEAD); MLXFoundationModelsTests 652 passed; MLXGuidedGenerationTests 96; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. swift-format run. New: Libraries/MLXLMCommon/Remote/{ClusterMTPTokenIterator,RemoteDraftExchange,RemoteDrafterTrack,RemoteMTPDrafter}.swift, Tests/MLXLMTests/ClusterMTPDecodingTests.swift (15). Changed: the protocol, executor, announcement, ClusterLanguageModel, RemoteKVCache, RemoteCacheStatus, KVCachePlan (commitDroppedTokens), MTPSampledVerification (shared walk), MTPSpeculation/GenerateTokens/Evaluate (makeMTPIterator dispatch), ChatSession, MTPSpeculativeTokenIterator, TokenIterators, RemoteModelTransport (feature removed), MLXLanguageModel; tests: ClusterLanguageModelTests, RemoteModelExecutorTests, RemoteModelProtocolTests, ClusterExecutorTests, ExecutorPromptCacheMTPTests, InProcessRemoteModelWorker (drafter + follower ranks), Inkling2TinyModels.
    - next: /review. The real 2-Mac run needs ^6pq8cee (FoundationModelsCluster must pass the drafter on rank 0 and RemoteDraftExchange.group(group) on every rank).
  timestamp: 2026-11-27T23:21:03.829552+00:00
- actor: claude-code
  id: 01m8fnhkzj247khw3h9nsg1n9q
  text: |-
    ### test — green
    - evidence: swift build --build-tests (also with -Xswiftc -warnings-as-errors, exit 0, no source warnings); xcrun xctest per bundle, full MLXLMTests in one process. MLXLMTests: 1261 XCTest, 0 failures + 2819 Swift Testing in 280 suites passed (3 known issues, all pre-existing withKnownIssue in unmodified AttentionKernelToleranceTests.swift and ToolTests.swift). DeepSeekV4AttentionTests passed in the full run (no stack overflow). ClusterMTPDecodingTests suite passed. MLXFoundationModelsTests: 652 tests / 88 suites passed. MLXGuidedGenerationTests: 96 tests / 23 suites passed. CXGrammarTests: 7 tests / 5 suites passed. MLXHuggingFaceMacrosTests: 11 XCTest passed. 0 skipped, 0 failed. swift-format lint on all changed Swift files clean.
    - next: review/commit
  timestamp: 2026-11-27T23:30:43.826862+00:00
- actor: claude-code
  id: 01m8fpd325zvvbh1e68phw78wg
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (8e4fce9f); 7 findings (7 confirmed, 17 refuted) — Libraries/MLXLMCommon/KVCachePlan.swift:204, Libraries/MLXLMCommon/Remote/RemoteModelExecutor.swift:297, Libraries/MLXLMCommon/Remote/RemoteOpAnnouncement.swift:164, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift:62, Tests/MLXLMTestSupport/Inkling2TinyModels.swift:190, Tests/MLXLMTests/RemoteModelExecutorTests.swift:116, Tests/MLXLMTests/RemoteModelProtocolTests.swift:85
    - next: implement the 7 items in the 2026-11-27 17:31 Review Findings section; task stays in review
  timestamp: 2026-11-27T23:45:44.005450+00:00
- actor: claude-code
  id: 01m8fpdkfeys5bvant2890h4vc
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 new files, 24 changed
    - test: green — all 5 bundles (MLXLMTests 2819+1261, MLXFoundationModelsTests 652, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 8e4fce9f
    - review: findings — KVCachePlan.swift:204, RemoteModelExecutor.swift:297, RemoteOpAnnouncement.swift:164, InProcessRemoteModelWorker.swift:62, Inkling2TinyModels.swift:190, RemoteModelExecutorTests.swift:116, RemoteModelProtocolTests.swift:85
  timestamp: 2026-11-27T23:46:00.814053+00:00
- actor: claude-code
  id: 01m8fps70k5xykq6ahhfrqcard
  text: 'Review-findings pass (implement): KVCachePlan: trim, rewindSpeculative and rewindLastRound now all move the timeline back through commitDroppedTokens (rewindSpeculative previously kept lastRound; its only caller runs after a trim that already cleared it, so no behavior change). RemoteDraftRound.validate() (internal, throws invalidCount clamped) holds the chunk-fit and 0...Int32.max rejected-count checks; RemoteOpAnnouncement.write/round(of:) and RemoteModelExecutor.speculate call it. InProcessRemoteModelWorker: stored draftWords removed (the closures capture the local relay). Inkling2TinyModels: private seeded(_:_:) wraps withRandomState for all four draws (initializer seed for both initializers, target/drafter seeds unchanged). RemoteModelExecutorTests: makeExecutor(drafter:) replaces makeDraftingExecutor (drafting tests now also carry modelType/quantization). RemoteModelProtocolTests: round-trip at draftLimit = maximumChunkTokenCount - 1, plus three validate() tests (RED by compile failure first).'
  timestamp: 2026-11-27T23:52:21.267836+00:00
- actor: claude-code
  id: 01m8fq2nfq3rfbfe4885qtbr1w
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/KVCachePlan.swift, Libraries/MLXLMCommon/Remote/{RemoteModelProtocol,RemoteOpAnnouncement,RemoteModelExecutor}.swift, Tests/MLXLMTestSupport/{InProcessRemoteModelWorker,Inkling2TinyModels}.swift, Tests/MLXLMTests/{RemoteModelExecutorTests,RemoteModelProtocolTests}.swift. All 7 findings of 2026-11-27 17:31 checked. swift build --build-tests clean; swift-format run. xcrun xctest MLXLMTests (one process): 1261 XCTest 0 failures + 2823 Swift Testing in 280 suites passed (3 known issues, pre-existing; +4 new tests; DeepSeekV4AttentionTests passed). MLXFoundationModelsTests: 652 in 88 suites passed.
    - next: /review (task left in doing, not committed)
  timestamp: 2026-11-27T23:57:30.999887+00:00
- actor: claude-code
  id: 01m8fqjbda5bxc6je5xybf11fm
  text: |-
    ### test — green
    - evidence: swift build --build-tests complete; xcrun xctest: MLXLMTests 1261 XCTest + 2823 Swift Testing passed, 0 failures; MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 652 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed; swift-format lint on changed files clean; 0 skipped. Swift Testing reports 3 known issues (AttentionKernelToleranceTests, MiniCPM5ToolCallParserTests) in files this task did not touch, pre-existing withKnownIssue markers.
    - next: review
  timestamp: 2026-11-28T00:06:04.970561+00:00
- actor: claude-code
  id: 01m8fqn66mtzz0054ms2skqxev
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (91b6663f) — 0 findings, 0 confirmed, 0 refuted; 7 validator runs attempted, 0 failed; 8 files reviewed, 2 .kanban files not matched by any validator. All 7 prior findings from 2026-11-27 17:31 are checked.
    - next: none — task moved to done
  timestamp: 2026-11-28T00:07:37.941+00:00
- actor: claude-code
  id: 01m8fqndxnqqjeqr1vs4cncg3g
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 8 files, all 7 findings checked
    - test: green — all 5 bundles (MLXLMTests 2823+1261, MLXFoundationModelsTests 652, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 91b6663f
    - review: clean — 0 findings
  timestamp: 2026-11-28T00:07:45.845748+00:00
position_column: done
position_ordinal: ffffff8a80
title: 'Clustered models: support speculative decoding with the MTP drafter (Qwen MTP, Inkling DSpark)'
---
## What
ClusterModelFeature.speculativeDecoding is unsupported (Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift:112-119). In cluster1i (Inkling-2-Mini-4bit + oQ4e DSpark drafter) every one of 19 generation calls ran without MTP: 67 tok/s, against ~105 tok/s for the same model on one Mac with MTP (about 63% of drafts accepted). So on one task the cluster is slower than one Mac for Inkling, and Qwen loses its 1.7x MTP gain too.

## Fix
Run the drafter on rank 0 (it is small) and verify the drafted block with one clustered forward of block+1 tokens, with the cache trim on every rank on reject, as the local MTP path does. Expose the same generation.mtp.* telemetry.
Tests: parity suite with the drafter on Inkling and Qwen: greedy output equals the clustered no-MTP output; accepted-token counts reported.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster

## Review Findings (2026-11-27 17:31)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 29 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8EPPVEJZWEPNS7MXRGZK56J.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8EPPVEJZWEPNS7MXRGZK56J.md` — no validator matches this file
> - `.kanban/tasks/01M8FKC3V6YDH4490VK6PQ8CEE.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8FKC3V6YDH4490VK6PQ8CEE.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/KVCachePlan.swift:204` `reuse/reuse` — The new `commitDroppedTokens` repeats the tail of the existing `trim(_:)`: it subtracts the dropped count from `processedTokenCount`, resets `lastRound`, and checks the same bound on the count. The timeline bookkeeping is duplicated instead of shared, so a future change to the invariant has to be made in two places. Have `trim(_:)` call `commitDroppedTokens(trimmed)` after the cache trim, so the timeline update lives in one place. Keep the separate entry point for the worker path, since its comment explains why it differs from `trim`'s guard. If the two preconditions must keep different messages, move only the subtraction and `lastRound` reset into a shared private helper.
- [x] `Libraries/MLXLMCommon/Remote/RemoteModelExecutor.swift:297` `reuse/reuse` — `speculate(_:)` repeats the draft-limit and rejected-count checks that `RemoteOpAnnouncement` also performs for the same `RemoteDraftRound`. The check lives in two modules' worth of code with no shared owner. Move the guards into a `RemoteDraftRound` validation method and call it here, as described for the announcement file, so the rule is stated once.
- [x] `Libraries/MLXLMCommon/Remote/RemoteOpAnnouncement.swift:164` `reuse/reuse` — `round(of:)` re-implements the draft-limit and rejected-count validation that `write(_:into:)` and `speculate(_:)` already perform. The same two checks now exist in three places, so a future change to the limits must be made three times. Add one throwing `validate()` method on `RemoteDraftRound` that holds both checks. Call it from `write`, from `round(of:)` after the struct is built, and from `speculate`. Keep the error mapping the same at each call site.
- [x] `Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift:62` `code-hygiene/dead-code-swift` — var.instance `draftWords` is assignOnlyProperty.
- [x] `Tests/MLXLMTestSupport/Inkling2TinyModels.swift:190` `reuse/reuse` — The `drafter(json:)` initializer block repeats the same `initializerSeed` wrapper that `target(json:bfloat16:)` added at line 126. It is the second copy of one idiom added in this change. Use the same seeded helper proposed for line 126 here, so both initializers go through one function.
- [x] `Tests/MLXLMTests/RemoteModelExecutorTests.swift:116` `reuse/reuse` — `makeDraftingExecutor()` rebuilds a `RemoteModelExecutor` from `Qwen35MTPFixture` the same way the existing `makeExecutor(rankCount:dtype:)` does, but it is a parallel copy and not an extension of that helper. The one difference is the drafter. The copy also drops the `modelType` and `quantization` configuration that `makeExecutor` passes, so the two helpers already diverge. Any later change to how the executor is built must be made in two places. Give `makeExecutor` a `drafter: Bool = false` (or a drafter-returning) parameter that passes `fixture.drafter` through, and have the drafting tests call `makeExecutor(drafter: true)`. If the drafting tests genuinely need the bare configuration, keep the separate helper but build it from the shared fixture-and-configuration code rather than copying it.
- [x] `Tests/MLXLMTests/RemoteModelProtocolTests.swift:85` `completeness/inverse-operation-coverage` — The chunk-fit boundary of a speculate announce is tested only on the overflow side. The new test asserts that draftLimit == maximumChunkTokenCount has no announce, because the verify is draftLimit + 1 tokens. Nothing asserts that the largest round that fits (draftLimit == maximumChunkTokenCount - 1, verify == maximumChunkTokenCount) does encode and decode. An off-by-one in the fit check (`<` for `<=`, or a verify count that omits the +1) would pass this suite while rejecting or mis-sizing the largest valid round in production. Add a round-trip test (encode then decode equals the request) for a speculate op with draftLimit = RemoteModelProtocol.maximumChunkTokenCount - 1, which should produce a valid announce with the verify exactly filling one chunk. Keep the existing overflow test as the negative side.
