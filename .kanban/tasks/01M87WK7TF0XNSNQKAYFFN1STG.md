---
assignees:
- claude-code
depends_on:
- 01M87HW2SS24TQTQ9HFE8SMZKZ
position_column: todo
position_ordinal: '8280'
title: Chunked eval at load; shard after per-layer eval
---
A one-shot `eval` of a very large model (300 GB+) trips the GPU watchdog (mlx-lm#1572), and exo evaluates layer by layer to avoid a FAST_SYNCH deadlock. For clustered loads, the worker must evaluate weights in chunks, layer by layer, and shard each layer right after its weights are evaluated. Each rank then never holds more than one layer's unsharded weights at a time on top of its shard.

- [ ] A load path (e.g. an option on the model factory, or a helper the worker calls) that loads lazily, then for each layer: eval → `shard` that layer → eval the shard → drop the full tensors.
- [ ] Generalize `TensorParallelShardable` so sharding can run per layer, not only for the whole model. Qwen35 and Inkling2 conform.
- [ ] Peak memory test: a sharded load's peak is well below the full model size, for Qwen3.5-2B on 2 ring ranks via ShardCheck.
- [ ] Rank 0 can also broadcast weights layer by layer to members that don't have the files (allSum of the tensor with zeros elsewhere, like mlx_lm share). Leave the hook for FoundationModelsCluster.

Depends on the Inkling 2 sharding task (e8smzkz). #cluster