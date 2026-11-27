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
position_column: doing
position_ordinal: '80'
title: 'Clustered models: support speculative decoding with the MTP drafter (Qwen MTP, Inkling DSpark)'
---
## What
ClusterModelFeature.speculativeDecoding is unsupported (Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift:112-119). In cluster1i (Inkling-2-Mini-4bit + oQ4e DSpark drafter) every one of 19 generation calls ran without MTP: 67 tok/s, against ~105 tok/s for the same model on one Mac with MTP (about 63% of drafts accepted). So on one task the cluster is slower than one Mac for Inkling, and Qwen loses its 1.7x MTP gain too.

## Fix
Run the drafter on rank 0 (it is small) and verify the drafted block with one clustered forward of block+1 tokens, with the cache trim on every rank on reject, as the local MTP path does. Expose the same generation.mtp.* telemetry.
Tests: parity suite with the drafter on Inkling and Qwen: greedy output equals the clustered no-MTP output; accepted-token counts reported.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster