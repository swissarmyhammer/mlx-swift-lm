---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m88b1qn87qhtbz9qrj9qp4zp
  text: |-
    ## Review of origin/stable..cluster: findings and dispositions

    Review engine: `review sha origin/stable..cluster`, 55 files, 42 findings. All fixed:

    - Qwen35.swift:284 magic-numbers (`keyDim * 2`): named `Qwen35GatedDeltaNet.queryKeyParts`, used at every `keyDim * 2` / `2 * keyDim` in the file and in Qwen35TensorParallel.swift:56 (same finding).
    - Qwen35.swift:1226, :1521 missing-docs `kvHeads`: documented (share after shardLayers).
    - Load.swift:231 reuse + :234 duplication (`loadLazyModelCheckpoint`): deleted; `loadModelCheckpoint(urls:excludedNamespaces:evaluation:)` / `(urls:evaluation:isRead:)` take a `WeightEvaluation` and skip the per-item eval when deferred, so the lazy load keeps the concurrent fan-out and the one exclusion mapping. Progress reporting stays at-load only (a lazy tensor reads after the scope ends).
    - ClusterLanguageModel.swift:131, :199 magic-numbers (`ndim < 2`): `holdsOneSequence(_:)` with `batchRank`.
    - RemoteCacheStatus.swift:54 magic-numbers (`/ 2`): `RemoteRewindLimit.bisection`.
    - TensorParallel.swift:91 missing-docs `description`: documented.
    - InProcessRemoteModelWorker.swift:24 missing-docs: the `disable:next no_unchecked_sendable` line sat between the doc comment and the class, hiding the doc; now a `disable`/`enable` pair above the doc comment (the reason kept), as the disallowed-constructs rule allows.
    - ClusterLanguageModelTests.swift:494, RemoteModelExecutorTests.swift:616 magic-numbers: `layerCount`.
    - Inkling2TensorParallelTests.swift:167, :332 reuse: one `Inkling2ShardFixture` (textConfiguration(_:), tinyTextConfiguration(), tinyPlan(groupSize:)) used by every suite, incl. kvHeadsFollowKeptHeads.
    - LoadWeightsTests.swift:581 reuse: `testDeferredLoadInstallsTheCheckpointValues` folded into `testLoadWeightsReadsSidecarWeightsDeclaredByTheModel`, which runs `.atLoad` and `.deferred`.
    - PromptCacheCheckpointCompositeTests.swift:133 inverse-operation-coverage: new test "an Inkling 2 cache restored after the window wraps continues as a cold prefill does" (last logits of a continuation from the restored caches vs a cold prefill, atol 1e-4); offset test kept, both on a shared `restoredAfterWrap`.
    - TensorParallelShardingTests.swift:279 reuse: `Qwen35TensorParallelPlanTests.makeConfiguration(fullAttentionInterval:hiddenSize:intermediateSize:)` is the one head-count setup; Qwen35WholeShareTests calls it.
    - shard-check.sh:79, :83 duplication: `launch_rank <rank> <output>`.
    - shard_check_compare.py:361 duplication: `print_timings` and `verdict` shared by main and compare_tokens.
    - test_shard_check_compare.py:14 dead-code (`sys.dont_write_bytecode`): the comparator is now compiled from source and exec'd into a module registered in sys.modules, which writes no bytecode at all; the setting is gone. vulture clean; no `__pycache__` after a run.
    - ShardCheck.swift:61, :90, :122, :142 no_direct_standard_out_logs: console lines go through `Console.write` (FileHandle.standardOutput), the tool's one output channel.
    - ShardCheck.swift:136 duplication: `writeReport(...)` shared by run and runGeneration.
    - ShardCheck.swift:310 initialization: `GreedyRun.run(model:prompt:decodeSteps:forcedTokens:)`; the struct keeps memberwise fields.
    - ShardCheck.swift:412 dead-code (`DefaultPrompt.text`) and :421-430 magic-numbers (token tables): the sentence and its ids moved to `tools/ShardCheck/DefaultPrompts.json` (a `.copy` resource of the target, read through Bundle.module); `ShardCheckError.missingResource` when absent.
    - ShardCheck.swift:237, :368, :456, :464, :472, :492 magic-numbers: `positionalCount`, `millisecondsPerSecond`, `magicByte`/`version`, `headerLengthBytes`, `lowByteMask`, `attosecondsPerSecond`.

    Checked after the fixes with the same tools the rules run (swiftlint no_magic_numbers, swift-format AllPublicDeclarationsHaveDocumentation, vulture) over the branch's added lines: no output.

    ## Own pass (correctness, thread-safety, naming, dead code)

    - FIXED `TensorParallel.split(of:in:)` took `Dictionary.first` of the matching keys, so overlapping keys (`a`, `a.b`) picked an arbitrary split per process run. Longest key now wins; test "of keys that nest, the deepest that names the parameter decides its split" (RED: picked axis 5 of 7).
    - FIXED doc: `RemoteModelSession.perform` claimed every earlier failure throws `transportFailed`; it rethrows the stored failure (`workerFailed`, `closed`, ...).
    - FIXED doc: `loadClusterModelContext` now states that the returned model owns the transport and that on a throw (e.g. model_type mismatch after connect) the caller closes it.
    - Checked, no change: RemoteModelSession / AnnouncingTransport / InProcessRemoteModelWorker hold every stored var under their NSLock; ClusterLanguageModel and RemoteKVCache are not Sendable, as KVCache and LanguageModel are not; RemoteModelExecutor documents single-thread use. All-sums run on the CPU stream (MLXDistributed default), as MLXNN's sharded layers do.
    - Noted, not changed: Package.swift pins mlx-swift to the swissarmyhammer fork branch `jaccl-no-ipv4` (intended for clustering, per the cluster notes); a branch dependency stops version-based consumers, so it must return to ml-explore before a release.

    ## Pre-existing DocC link failures (^1qfv3es)

    verify-docs.sh failed in 4 targets, not only MLXLMCommon: 30 links in MLXLMCommon, MLXLLM, MLXVLM and MLXFoundationModels to internal/package symbols, private members or symbols of another module (DocC does not link across modules). Each now in code voice (`Type.member`).
  timestamp: 2026-11-25T03:12:36.520729+00:00
- actor: claude-code
  id: 01m88bdrmzwqtdjabvwkstd2j9
  text: |-
    Re-review of the fix commits (`review sha 2fc28da8..HEAD`): 1 finding, fixed.

    - ClusterLanguageModel.swift:198 completeness/invariant-propagation: `prepare` throws `unsupported(.batchedInput)` for a batch, but `forward` (behind the non-throwing `callAsFunction`/`nextTokenLogits`) aborted on a `precondition`. Now the batch goes through the failed-forward path: the failure is kept on the model, the task is cancelled, and zero logits of shape `[sequences, positions, vocabulary]` come back. Test "a batch given straight to the forward fails the model with the batch refusal" (RED: process abort at the precondition; GREEN: ClusterLanguageModelTests 27/27).

    Smoke: `scripts/shard-check.sh` on Qwen3.5-2B-mxfp4 PASS (teacher-forced, 9/9 argmax, max KL 0.00260, rank-0 load peak 0.59 GiB vs 0.94 GiB; default prompt read from DefaultPrompts.json). `python3 -I scripts/test_shard_check_compare.py`: 26 tests OK, no __pycache__. `scripts/verify-docs.sh`: all targets pass, with the new distributed-inference article.

    swift-format: local Homebrew is 604.0.0, CI pins 603.0.0; built 603.0.0 from its tag and formatted the whole repo with it. It changed only branch files (ModelFactory.swift, Qwen35MTPFixture+LanguageModel.swift, and the new edits).
  timestamp: 2026-11-25T03:19:10.751712+00:00
- actor: claude-code
  id: 01m88bwgxv3c4yzcp0x2cqnqcv
  text: |-
    ### implement — changed
    - evidence: commits 97d2b2f7 (DocC links, ^1qfv3es), 76f1892d (review fixes + tests), 8dae5929 (ShardCheck/scripts), aaf2f02d (distributed-inference DocC article + skill reference), 6b83d5a6 (batched forward). Tests via `swift build --build-tests` + `xcrun xctest`: MLXLMTests 1261 XCTest + 2743 Swift Testing (3 known issues, all pre-existing withKnownIssue on stable), MLXGuidedGenerationTests 95, MLXFoundationModelsTests 618, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11; 0 failures. The two parallel-load flakes named on the card did not recur in this full run. verify-docs.sh passes. shard-check Qwen3.5-2B-mxfp4 PASS. test_shard_check_compare.py 26 OK. Re-review of the fix commits clean after 6b83d5a6. pkdw8fc and e8smzkz were already done.
    - next: none; not pushed.
  timestamp: 2026-11-25T03:27:14.363722+00:00
depends_on: []
position_column: done
position_ordinal: fffffa80
title: Review + docs for tensor-parallel sharding (Qwen35, Inkling2)
---
Make the sharding work mergeable into `stable`.

- [ ] `/review` the cluster commits (de0b6969..HEAD) and fix the findings.
- [ ] `scripts/verify-docs.sh` passes. `TensorParallelShardable`, `TensorParallelShardingError` and the `TensorParallel` helpers are new public API in MLXLMCommon.
- [ ] DocC article or section, plus `skills/mlx-swift-lm/references/` (and the supported-models list) covering which families can shard, and how.
- [ ] `swift-format format --in-place --configuration .swift-format --recursive .` per CLAUDE.md, using CI's pinned version (see .github/workflows/pull_request.yml).
- [ ] All 5 test bundles green (MLXLMTests, MLXGuidedGenerationTests, MLXFoundationModelsTests, CXGrammarTests, MLXHuggingFaceMacrosTests), via `swift build --build-tests` + `xcrun xctest` per CLAUDE.md. Two known parallel-load flakes pass when run alone: "a chunked MTP prefill of a long prompt peaks lower…" and "a fork of a sliding-window model reuses the instructions past a wrapped window". Fix or stabilise them if they're real.
- [ ] Move pkdw8fc and e8smzkz to done.

Depends on e8smzkz. #cluster