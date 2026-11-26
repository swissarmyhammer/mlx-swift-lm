---
assignees:
- claude-code
position_column: todo
position_ordinal: '8280'
title: 'Inkling 2 tiny target: the MoE branch does not reach the logits, so tiny shard-check cannot see an Inkling MoE sharding break'
---
Found in ^727pxfp. With `Inkling2TinyModels.target(json: Inkling2TinyModels.targetJSON)` (and the tiny `inkling2` / `inkling2_loop` checkpoints that `ShardCheck tiny-checkpoint` writes from it), the MoE output has no effect on the logits:

- In memory: zeroing every `mlp.experts.down_proj.global_scale` leaves the logits of an 8-token prompt bit-identical (with and without a cache).
- In the checkpoint: zeroing every `mlp.experts.down_proj.scales`, every `mlp.experts.gate_up_proj.scales`, or every `mlp.gate.weight` leaves the unsharded ShardCheck logits bit-identical; zeroing `self_attn.o_proj.weight` changes them by 2.4.
- Consequence: dropping the all-sum of `Inkling2SparseMoE.callAsFunction` (the path ShardCheck's sharded run takes: `prefill` -> `callAsFunction`, 54 calls with the group set over 9 steps) leaves the sharded logits bit-identical, so `scripts/shard-check.sh --tiny` cannot catch an Inkling MoE sharding break. Dropping the attention-output all-sum IS caught.

- [ ] Find why the MoE branch is invisible: a property of the tiny config (attention-residual blocks of 2 layers, `local_layer_ids`, softcaps, random weight scale 0.3), or a model bug where the MoE output is dropped. Check whether a real Inkling 2 checkpoint's logits depend on its experts.
- [ ] If it is the tiny config, change `Inkling2TinyModels.targetJSON` (or the weight scale) so the experts reach the logits, and show the tiny shard-check then fails when the `Inkling2SparseMoE` all-sum is dropped.
- [ ] If it is a model bug, fix it with a failing test first. #cluster