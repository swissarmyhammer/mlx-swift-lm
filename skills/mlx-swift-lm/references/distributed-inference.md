# Distributed Inference (Tensor Parallel Sharding and Cluster Models)

## Overview

A model too large for one Mac runs across several. Each Mac runs one worker process; the
workers join one `MLXDistributed.Group` (ring over TCP, or JACCL over Thunderbolt RDMA) and
each keeps its share of every sharded layer. The app process stays outside the group: it runs
a `ClusterLanguageModel` that sends each forward pass and cache op to the rank-0 worker.

This package holds the sharding, the remote op protocol, the app-side model and the
worker-side executor. It holds no networking: a cluster package supplies the
`RemoteModelTransport`, starts the workers, and runs the announce loop.

Core implementation:

- `Libraries/MLXLMCommon/TensorParallel.swift`: `TensorParallelShardable`, `TensorParallelLayerHooks`, `TensorParallelShardingError`, the `TensorParallel` helpers
- `Libraries/MLXLMCommon/TensorParallelLoading.swift`: `loadShardedModelContext(from:using:group:willShardLayer:)`
- `Libraries/MLXLLM/Models/Qwen35TensorParallel.swift`, `Libraries/MLXVLM/Models/Inkling2TensorParallel.swift`: the two conformances
- `Libraries/MLXLMCommon/Remote/`: `ClusterLanguageModel`, `loadClusterModelContext`, `RemoteModelProtocol`, `RemoteKVCache`, `RemoteModelExecutor`, `RemoteOpAnnouncement`
- `tools/ShardCheck/` and `scripts/shard-check.sh`: the logits and generation check

DocC article: `Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md`.

## Which Families Shard

| `model_type` | What splits |
|---|---|
| `qwen3_5`, `qwen3_5_text`, `qwen3_8`, `qwen3_8_text` (dense) | Full-attention and gated delta heads, the gated delta conv, the MLP width |
| `inkling2`, `inkling2_loop` (text stack) | Sliding and full attention heads, their short convolutions, each expert's inner width |

`RemoteModelExecutor.shardableModelFamilies` holds the same list. Refused:

- `qwen3_5_moe`, `qwen3_8_moe`: `TensorParallelShardingError.unsupported` (mixture-of-experts layers).
- `prism_hadamard_qwen35`: `.unsupported` (rotated projections are not stock `Linear`/`QuantizedLinear`).
- A LoRA-adapted projection: `.unsupported`, for the same reason.
- A group size that does not divide a head count, an MLP width, or whole quantization groups: `.indivisible(quantity:count:groupSize:)`.
- Any other model: `.notShardable(modelType:)`.

Every share is checked before any layer changes, so a refused model stays as it was.

## Shard a Model in Memory

```swift
let group = try MLXDistributed.initialize(backend: .ring, strict: true)
let context = try await loadModel(from: directory, using: tokenizerLoader)
try (context.model as? TensorParallelShardable)?.shard(group: group)
```

A group of one process leaves the model unchanged.

## Load a Share Layer by Layer (Preferred)

```swift
let context = try await loadShardedModelContext(
    from: directory, using: tokenizerLoader, group: group)
```

The load keeps the weights lazy (`ResolvedModelConfiguration.weightEvaluation = .deferred`),
then for each layer evaluates the full weights, splits, evaluates the shares and frees the full
weights. A process never holds more than one layer's full weights beside its shares.

`willShardLayer` runs on each layer before it evaluates. A process without the weight files
loads with `WeightFileSelection.skipped`, and the hook replaces the layer's parameters with
their all-sum across the group (rank 0 contributes the tensors, the others zeros), as
`mlx_lm.share` does.

## Drive a Cluster from the App

```swift
let context = try await loadClusterModelContext(
    from: directory, using: tokenizerLoader, transport: transport)
let session = ChatSession(ModelContainer(context: context))
```

- The load reads `config.json`, the tokenizer and the processor, and no weight file.
- `ClusterLanguageModel.connect(over:)` checks the protocol version; the load also checks the worker's `model_type`.
- `newCache(parameters:)` makes one `RemoteKVCache` naming every layer's cache on the worker; its `RemoteCacheStatus` is mirrored, so `offset` and rewind checks cost no round trip.
- `close()` tears the workers down; the FoundationModels adapter calls it on eviction, and `ModelLoadIdentity.clustered(planID:)` keeps a clustered load out of the cache entry of a local load.

Refused with `ClusterModelError.unsupported(_:)`: media input, speculative decoding (MTP and
draft model), a KV-cache configuration or `maxKVSize`, a batch of several sequences. A prompt
cache file cannot hold a remote cache.

A failed forward cannot throw: the model cancels its task, keeps the failure, and every later op
fails with it.

## The Remote Op Protocol

| Op | Reply |
|---|---|
| `describe` | `capabilities(RemoteModelCapabilities)` |
| `begin`, `end` | `done` |
| `newCache(id)`, `copy(id, into:)`, `reset(id)` | `cache(RemoteCacheStatus)` |
| `forward(tokens:cache:logits:)` | `forwarded(status, logits: RemoteLogits?)` |
| `trim(id, count:)` | `trimmed(count, status:)` |
| `release(id)` | `done` |

Rules: sequence numbers start at 1 and grow by one; the app allocates every cache handle; a
forward carries at most `RemoteModelProtocol.maximumChunkTokenCount` tokens; no op carries text
the worker would run.

## Worker Side

```swift
let executor = RemoteModelExecutor(
    model: context.model,
    configuration: try .init(modelDirectory: directory, rankCount: try group.size))

if try group.rank == 0 {
    _ = try RemoteOpAnnouncement.exchange(RemoteOpAnnouncement.encode(request), group: group)
    let reply = try executor.handle(request)          // send to the app
} else {
    while true {
        let words = try RemoteOpAnnouncement.exchange(nil, group: group)
        _ = try executor.apply(announcement: words)   // reply dropped
        if RemoteOpAnnouncement.stopsLoop(words) { break }
    }
}
```

Every rank must apply the same ops in the same order, or the all-sum of a sharded layer blocks.
Rank 0 announces a park (`RemoteOpAnnouncement.park(requestID:)`) between turns so the other
ranks can wait without polling. The executor is not thread-safe.

## Checking a Shard

```sh
scripts/shard-check.sh <model-directory>              # teacher-forced logits, KL + argmax gate
scripts/shard-check.sh --free-running <model-directory>
scripts/shard-check.sh --announce <model-directory>   # generation through ClusterLanguageModel
python3 -I scripts/test_shard_check_compare.py         # self-check of the comparator
```

## DON'T

- Don't shard with different groups or checkpoints on different processes: every process loads the same checkpoint and passes the same group.
- Don't run a forward on one rank only: the sharded layers sum across every rank.
- Don't pass `GenerateParameters(kvBits:)` or `maxKVSize` to a cluster model: the worker makes its caches.
