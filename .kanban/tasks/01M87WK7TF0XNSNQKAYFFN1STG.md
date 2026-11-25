---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m886w12n5ka9v6n6y1x44br6
  text: |-
    Research: today's load is eager twice over — `loadModelCheckpoint` evaluates every tensor in its concurrent work items, then `materializeWeights` + `materializeModelForInference` evaluate the prepared model. So a lazy load needs (1) a lazy checkpoint read (`loadArraysAndMetadata` without eval) and (2) skipping materialize/prepare. MLXNN `shardInPlace`/`shardLinear` copy with `.contiguous()`, so an evaluated shard does not keep the full tensor's buffer alive. Qwen35/Inkling2 already build every layer's shards (`makeShards`, lazy, cheap) before mutating any: that stays the build-first validation; the per-layer loop then runs hooks around `layer.shard(plan, with:)`.

    Before numbers (ShardCheck, 2 ring ranks on localhost, `Memory.peakMemory` after load+shard, current full-load-then-shard path): Qwen3.5-2B-mxfp4 0.94 GiB per rank (= the unsharded reference's 0.94 GiB); Qwen3.8-27B-mxfp4 13.40 GiB per rank (= unsharded 13.40 GiB). Both PASS.
  timestamp: 2026-11-25T01:59:35.253935+00:00
- actor: claude-code
  id: 01m887q9vp3ay2ynw35x9ahhmh
  text: |-
    Implemented.
    - `TensorParallelShardable` now requires `shardLayers(group:hooks:)`; `shard(group:)` is a protocol extension (= `shardLayers` with `TensorParallelLayerHooks.none`). `TensorParallelLayerHooks` has `willShard`/`didShard` (index, layer module). `TensorParallel.shardLayers(_:hooks:makeShards:shard:)` builds every layer's shards first (the existing build-first validation: throws before any layer changes or hook runs), then runs willShard → shard → didShard per layer. Qwen35TextModel/Qwen35Model and Inkling2/Inkling2TextModel conform through it.
    - `WeightEvaluation` (`.atLoad` default, `.deferred`) on `loadWeights` and `ResolvedModelConfiguration.weightEvaluation`; LLM and VLM factories pass it. Deferred = `loadLazyModelCheckpoint` (loadArraysAndMetadata, no eval, reuses ConcurrentLoadState merge) and no materialize/prepare.
    - `loadShardedModelContext(from:using:group:willShardLayer:)` (Libraries/MLXLMCommon/TensorParallelLoading.swift): group of 1 = normal load; otherwise deferred load, tries factories until one gives a shardable model (the VLM factory registers first and its Qwen35 does not shard; a lazy load costs only headers), then on a global queue: per layer willShardLayer → eval(layer) → shard → eval(layer) → Memory.clearCache(), then materializeModelForInference. `willShardLayer` is the documented weight-broadcast hook (rank 0 all-sums the layer's tensors, others contribute zeros, ranks without files load with `.skipped`); no networking implemented.
    - ShardCheck uses it; report has `loadPeakBytes` (load time now includes the split; `shardSeconds` removed), each rank prints its load peak. shard_check_compare.py gates the sharded rank-0 load peak ≤ 75% of the reference's (2 new self-tests, 25 OK).

    Peak per rank (Memory.peakMemory after load, 2 ring ranks on localhost), before → after:
    - Qwen3.5-2B-mxfp4: 0.94 → 0.59 GiB (unsharded 0.94)
    - Qwen3.8-27B-mxfp4: 13.40 → 7.29 GiB (unsharded 13.40)
    - Inkling-2-Mini-oQ4e-mtp: unsharded 14.55 → 7.78 GiB per rank
    shard-check.sh: Qwen3.5-2B PASS; Qwen3.8-27B-mxfp4 PASS (max KL 0.00268), --announce PASS; Inkling-2-Mini-oQ4e-mtp PASS (max KL 0.01638). Load stays ~1 s warm.

    Tests: TDD RED = compile failure (no `weightEvaluation:`), then GREEN. New: LoadWeightsTests deferred load leaves preparation to the caller / installs checkpoint values; TensorParallelLayerShardingTests (validate all layers first; hook order). All 5 bundles green: MLXLMTests 1262 XCTest + 2740 Swift Testing (3 pre-existing known issues), MLXGuidedGenerationTests 95, MLXFoundationModelsTests 618, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11. No build warnings.
    DocC for MLXLMCommon: the new links resolve; the build still fails on links that were already broken → filed ^1qfv3es.

    ### implement — changed
    - evidence: Libraries/MLXLMCommon/{TensorParallel,TensorParallelLoading(new),Load,Downloader,ModelConversion,WeightResidency}.swift, Documentation.docc/porting.md, Libraries/MLXLLM/{LLMModelFactory,Models/Qwen35}.swift, Libraries/MLXVLM/{VLMModelFactory,Models/Inkling2}.swift, tools/ShardCheck/ShardCheck.swift, scripts/{shard-check.sh,shard_check_compare.py,test_shard_check_compare.py}, Tests/MLXLMTests/{LoadWeightsTests,TensorParallelLayerShardingTests(new)}.swift
    - next: review
  timestamp: 2026-11-25T02:14:28.982366+00:00
- actor: claude-code
  id: 01m887qnjd9zf4seawccr7aee7
  text: |-
    ### commit — changed
    - evidence: f525bcc9 feat(cluster): load a sharded model layer by layer (not pushed)
    - next: review
  timestamp: 2026-11-25T02:14:40.973388+00:00
depends_on:
- 01M87HW2SS24TQTQ9HFE8SMZKZ
position_column: doing
position_ordinal: '8180'
title: Chunked eval at load; shard after per-layer eval
---
A one-shot `eval` of a very large model (300 GB+) trips the GPU watchdog (mlx-lm#1572), and exo evaluates layer by layer to avoid a FAST_SYNCH deadlock. For clustered loads, the worker must evaluate weights in chunks, layer by layer, and shard each layer right after its weights are evaluated. Each rank then never holds more than one layer's unsharded weights at a time on top of its shard.

- [x] A load path (e.g. an option on the model factory, or a helper the worker calls) that loads lazily, then for each layer: eval → `shard` that layer → eval the shard → drop the full tensors.
- [x] Generalize `TensorParallelShardable` so sharding can run per layer, not only for the whole model. Qwen35 and Inkling2 conform.
- [x] Peak memory test: a sharded load's peak is well below the full model size, for Qwen3.5-2B on 2 ring ranks via ShardCheck.
- [x] Rank 0 can also broadcast weights layer by layer to members that don't have the files (allSum of the tensor with zeros elsewhere, like mlx_lm share). Leave the hook for FoundationModelsCluster.

Depends on the Inkling 2 sharding task (e8smzkz). #cluster