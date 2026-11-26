---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'ShardCheck: write tiny random checkpoints so shard-check runs with no real checkpoint'
---
Found in ^ze3xffb. `scripts/shard-check.sh` (2 ring processes on localhost vs the unsharded logits) is the only sharded-vs-local logits check that runs the production `shard` path end to end, but it needs a model directory. For ^ze3xffb a throwaway test wrote tiny random `qwen3_5_moe` (4-bit affine, 8 experts) and `prism_hadamard_qwen35` (4-bit rotated, blocks 32/64/128) checkpoints (config.json + model.safetensors, ~2-5 MB) and both passed shard-check (teacher-forced and --announce); a per-shard signs mutation and an MoE missing-all-sum mutation each failed 9/9 steps. The writer is not committed.

- [ ] Add a way to write tiny random checkpoints of each shardable family (a ShardCheck subcommand or a script), with the configs that split across two processes.
- [ ] Run shard-check over each of them in CI or document the command, so a new shardable family gets a logits check with no download.
- [ ] Note that the 75% load-peak gate is not meaningful for megabyte-sized checkpoints (both read 0.00 GiB). #cluster