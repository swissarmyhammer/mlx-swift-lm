# Distributed inference

Split a model across the processes of a distributed group, and drive it from an app that
holds no weights.

## Overview

A model too large for one Mac runs on several. Each Mac runs one worker process, and the
workers join one `MLXDistributed.Group` (ring over TCP, or JACCL over Thunderbolt RDMA). Each
worker keeps its share of every sharded layer, and the layers sum their partial results across
the group inside the forward pass, so every worker computes the same logits. This is tensor
parallelism, the scheme of `model.shard(group)` in Python mlx-lm.

The app process stays out of the group. It loads the configuration and the tokenizer of the
checkpoint, and runs a ``ClusterLanguageModel`` that sends each forward pass and each cache op
to the rank-0 worker. Sampling, logit processors, grammar, stop strings, tool and reasoning
parsing and the prompt-cache policy stay in the app, so ``ModelContainer``, ``ChatSession``
and the FoundationModels adapter run on a cluster model unchanged.

This package holds no networking and starts no process. A cluster package supplies the
transport from the app to rank 0, starts the workers, and runs the announce loop that
``RemoteModelExecutor`` describes.

## Shard a model

A model that can split conforms to ``TensorParallelShardable``. Each process loads the same
checkpoint and calls ``TensorParallelShardable/shard(group:)`` with the same group:

```swift
let group = try MLXDistributed.initialize(backend: .ring, strict: true)
let context = try await loadModel(from: directory, using: tokenizerLoader)
try (context.model as? TensorParallelShardable)?.shard(group: group)
```

Each process keeps whole attention heads, and a slice of each MLP or expert width. The
projections that read the residual stream split their rows; the output and down projections
split their inputs and become sharded-to-all layers, which sum across the group. The embedding,
the LM head and the norms stay whole. A group of one process leaves the model unchanged.

Every share is checked before any layer changes. A model that cannot split evenly throws
``TensorParallelShardingError`` and stays as it was:

- ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)`` when the group size
  does not divide a count, such as the key and value heads, or when a process would keep part
  of a quantization group.
- ``TensorParallelShardingError/unsupported(feature:)`` for a part the sharding does not handle,
  such as a mixture-of-experts layer of Qwen 3.5, or a projection that is not a stock `Linear`
  or `QuantizedLinear` (a LoRA adapter, a rotated projection).
- ``TensorParallelShardingError/notShardable(modelType:)`` when the loaded model does not
  conform at all.

A conformance builds on the helpers of ``TensorParallel``: ``TensorParallel/share(of:named:across:)``
for each count, ``TensorParallel/SplitProjections`` to check every projection of a layer,
``TensorParallel/makeShardedToAll(_:at:group:)`` for the projections whose inputs split, and
``TensorParallel/shardAxes(of:_:group:)`` for the parameters that split in place.
``TensorParallel/shardLayers(_:hooks:makeShards:shard:)`` runs the two phases: it builds the
replacements of every layer first, then splits the layers one at a time.

### Which families shard

| `model_type` | Model | What splits |
|---|---|---|
| `qwen3_5`, `qwen3_5_text`, `qwen3_8`, `qwen3_8_text` | Qwen 3.5 and Qwen 3.8, dense | Full-attention and gated delta heads, the gated delta conv, the MLP width |
| `qwen3_5_moe`, `qwen3_8_moe` | Qwen 3.5 and Qwen 3.8, mixture of experts | The heads as for the dense model, the inner width of every expert and of the shared expert; the router stays whole |
| `prism_hadamard_qwen35` | Qwen 3.5 from a Hadamard-rotated pack | As for the dense model; a rotated projection whose inputs split keeps the signs of its own inputs |
| `inkling2`, `inkling2_loop` | Inkling 2 Small and Mini, text stack | Attention heads of the sliding and full layers, their short convolutions, the inner width of every expert |

``RemoteModelExecutor/shardableModelFamilies`` lists the same types for a planner. The text
stack loads through the LLM factory for Qwen 3.5, or through the VLM factory for a type only it
knows, such as Inkling 2. The vision tower of Inkling 2 stays whole, and a cluster model takes
no image input.

The experts of an MoE block sum their partial outputs once, routed and shared together, as the
experts of Inkling 2 do. The rotation of a ``HadamardQuantizedLinear`` is block diagonal over its
inputs, so a split of its inputs needs whole Hadamard blocks on each process, which
``TensorParallel/requireWholeInputs(of:at:across:)`` checks beside the quantization groups;
``ShardedToAllHadamardQuantizedLinear`` holds the share of one process.

Any other model type is refused: it does not conform to ``TensorParallelShardable``.

## Load a share layer by layer

``TensorParallelShardable/shard(group:)`` splits a model that is already in memory, so each
process first holds the whole model. ``loadShardedModelContext(from:using:group:willShardLayer:)``
instead keeps the weights lazy (``WeightEvaluation/deferred``) and splits the model one layer at
a time through ``TensorParallelShardable/shardLayers(group:hooks:)``: it evaluates the full
weights of one layer, splits the layer, evaluates the shares and frees the full weights. A
process thus never holds more than one layer's full weights beside its shares.

```swift
let context = try await loadShardedModelContext(
    from: directory, using: tokenizerLoader, group: group)
```

``TensorParallelLayerHooks`` carry the calls around the split of each layer. The
`willShardLayer` hook runs on each layer before its full weights evaluate. A process that lacks
the weight files loads with ``WeightFileSelection/skipped``, and the hook replaces the layer's
parameters with their all-sum across the group, rank 0 contributing the tensors it read and
every other process zeros, as `mlx_lm.share` does.

## Drive a cluster from the app

``loadClusterModelContext(from:using:transport:)`` reads the checkpoint with no weights and puts
a ``ClusterLanguageModel`` in the place of its model:

```swift
let context = try await loadClusterModelContext(
    from: directory, using: tokenizerLoader, transport: transport)
let container = ModelContainer(context: context)
let session = ChatSession(container)
```

The transport conforms to ``RemoteModelTransport``: ``RemoteModelTransport/send(_:)`` carries
one ``RemoteModelRequest`` to rank 0 and blocks until its ``RemoteModelReply`` arrives. The model
owns the transport, and ``ClusterLanguageModel/close()`` tears the workers down. The
FoundationModels adapter closes it when it evicts the model.

The model refuses, with a ``ClusterModelError``, what a worker cannot run: image, video and
audio input, MTP and draft-model speculative decoding, a KV-cache configuration other than an
affine quantization (`maxKVSize`, a rewind reserve, TurboQuant), a batch of several sequences,
and a worker of another protocol version or `model_type`. A prompt cache file cannot hold its
cache.

An affine KV-cache configuration (`kvBits` 8 or 4, or an affine
``GenerateParameters/kvCache``) quantizes the caches of every rank, each for the heads it
holds. The new cache carries the ``RemoteKVCacheQuantization`` to every rank, and the app sends
``RemoteModelOp/quantize(_:)`` where a local generation converts its caches, so the caches of
the workers convert after the forwards a local model converts after.

A forward that fails cannot throw, because ``LanguageModel/callAsFunction(_:cache:state:)`` does
not throw. The model then cancels the task that runs it, which ends the generation, and keeps
the failure: every later op fails, and the next throwing op reports it.

A transport that loses members of the cluster, for example a Mac that shut down, throws an
error that conforms to ``ClusterMemberLoss`` and whose ``ClusterMemberLoss/lostMember`` is not
nil. The model then fails with ``ClusterModelError/memberLost(_:)`` and not with
``ClusterModelError/transportFailed(_:)``. A host can tell the two apart: it drops the model and
loads a local one. The words of the member loss are for a log or a message, and a host does not
parse them for a single host: when the cluster cannot tell which member is gone, for example
after a hang, they list the `.local` name of every member, separated by commas. An error that
conforms to ``ClusterMemberLoss`` but whose ``ClusterMemberLoss/lostMember`` is nil lost no
member, and the model fails with ``ClusterModelError/transportFailed(_:)``, as for any other
error of the transport.

### The cache

``ClusterLanguageModel/newCache(parameters:)`` makes one ``RemoteKVCache``, which names the
caches of every layer on the worker by a ``RemoteCacheID`` the app allocates. Each op that
changes a cache answers with a ``RemoteCacheStatus``, which the remote cache mirrors, so a read
of its offset or of how far it can rewind costs no round trip. The prompt cache trims and copies
it as it does a local cache, and the cache releases its handle on the worker when it goes away.

## The remote op protocol

``RemoteModelProtocol`` names the op set and its rules. Each ``RemoteModelOp`` travels in a
``RemoteModelRequest``, whose sequence number grows by one for each request:

| Op | What the worker does |
|---|---|
| ``RemoteModelOp/describe`` | Reports its ``RemoteModelCapabilities``. |
| ``RemoteModelOp/begin``, ``RemoteModelOp/end`` | Opens and closes a generation request. |
| ``RemoteModelOp/newCache(_:)`` | Makes an empty cache under a handle the app chose. |
| ``RemoteModelOp/forward(tokens:cache:logits:)`` | Runs at most ``RemoteModelProtocol/maximumChunkTokenCount`` tokens into a cache, and returns the logits of a ``RemoteLogitsScope``. |
| ``RemoteModelOp/trim(_:count:)`` | Drops the newest positions of a cache. |
| ``RemoteModelOp/copy(_:into:)`` | Copies a cache under a second handle. |
| ``RemoteModelOp/reset(_:)``, ``RemoteModelOp/release(_:)`` | Empties a cache, or frees it. |

No op carries a path, a command or other text the worker would run. The logits come back as
``RemoteLogits``, the bytes of the array in the element type the model computed, so the app
samples the values a local model would give it.

Rank 0 announces each request to every rank as ``RemoteOpAnnouncement`` words: a fixed-size
`Int32` buffer that ``RemoteOpAnnouncement/exchange(_:group:)`` sums across the group, rank 0
writing the op and every other rank zeros. Between turns rank 0 announces a park
(``RemoteOpAnnouncement/park(requestID:)``), which lets the other ranks leave the all-sum loop
and wait without polling.

## Run the worker side

``RemoteModelExecutor`` runs each request on a loaded model. Each rank holds its own executor
over its own shard. Rank 0 announces each request, then calls ``RemoteModelExecutor/handle(_:)``
and sends the reply to the app; every other rank calls
``RemoteModelExecutor/apply(announcement:)`` with the words of the same exchange and drops the
reply:

```swift
let executor = RemoteModelExecutor(
    model: context.model,
    configuration: try .init(modelDirectory: directory, rankCount: try group.size))

if try group.rank == 0 {
    // For each request from the app:
    _ = try RemoteOpAnnouncement.exchange(RemoteOpAnnouncement.encode(request), group: group)
    let reply = try executor.handle(request)
} else {
    while true {
        let words = try RemoteOpAnnouncement.exchange(nil, group: group)
        _ = try executor.apply(announcement: words)
        if RemoteOpAnnouncement.stopsLoop(words) { break }
    }
}
```

Every rank must apply the same ops in the same order: a forward that runs on one rank and not on
another blocks both in the all-sum of a sharded layer. The executor checks the sequence and the
request of each op, and answers one that breaks a rule with ``RemoteModelReply/failed(_:)``
without changing anything. It is not thread-safe: call it from one thread, one request at a
time.

## Check a shard

`scripts/shard-check.sh <model-directory>` runs the `ShardCheck` tool unsharded as the
reference, then as two ring processes on one Mac, and compares the logits of each step. A step
passes when each run's argmax is in the other run's top-5 tokens, or within 2 bf16 ulps of the
other run's 5th-highest logit, as in vLLM's
[`check_logprobs_close`](https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py);
KL divergence is printed but not gated. It also checks that the layer-by-layer load of rank 0 peaks at most 75%
of the unsharded load. `--announce` instead generates through a ``ClusterLanguageModel`` whose
transport announces each op to the other rank, and compares the tokens with a local generation.

`scripts/shard-check.sh --tiny` needs no model directory and no download. For each model type of
``RemoteModelExecutor/shardableModelFamilies`` (or each one named after `--tiny`), `ShardCheck
tiny-checkpoint <family> <directory>` writes a tiny checkpoint on seeded random weights,
`config.json` and `model.safetensors` of 1 to 5 MB, with a configuration that splits across two
processes. The script then runs the teacher-forced check and the `--announce` check on it, with
prompt ids the tiny vocabulary holds, and prints one `PASS` or `FAIL` line for each family and
check. `ShardCheck tiny-families` lists the families. A family added to
``RemoteModelExecutor/shardableModelFamilies`` with no tiny checkpoint fails the check, so each
new family gets a logits check that runs on one Mac.

The 75% load-peak gate says nothing about a checkpoint of a few megabytes: both load peaks read
about 0.00 GiB. The tiny mode alone turns the gate off (the `--no-load-peak-gate` option of
`scripts/shard_check_compare.py`) and prints the peaks. A check of a real checkpoint keeps the
gate. No CI job runs the tiny mode: it needs a Mac GPU, and the Mac jobs of
`.github/workflows/pull_request.yml` run only in `ml-explore/mlx-swift-lm`, on its self-hosted
runner. Run it on a Mac before a change to sharding or to a shardable family lands.

## Topics

### Sharding

- ``TensorParallelShardable``
- ``TensorParallelLayerHooks``
- ``TensorParallelShardingError``
- ``TensorParallel``
- ``loadShardedModelContext(from:using:group:willShardLayer:)``
- ``WeightEvaluation``

### The app side

- ``ClusterLanguageModel``
- ``loadClusterModelContext(from:using:transport:)``
- ``RemoteModelTransport``
- ``RemoteKVCache``
- ``ClusterModelError``
- ``ClusterMemberLoss``
- ``ClusterModelFeature``

### The protocol

- ``RemoteModelProtocol``
- ``RemoteModelRequest``
- ``RemoteModelOp``
- ``RemoteModelReply``
- ``RemoteModelCapabilities``
- ``RemoteWorkerCapabilities``
- ``RemoteCacheStatus``
- ``RemoteLogits``
- ``RemoteOpAnnouncement``

### The worker side

- ``RemoteModelExecutor``
- ``RemoteModelExecutorError``
