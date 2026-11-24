---
assignees:
- claude-code
depends_on:
- 01M87HW2SS24TQTQ9HFE8SMZKZ
position_column: todo
position_ordinal: '8480'
title: 'Long-run accuracy check: 2-Mac JACCL, 256+ decode steps'
---
On a 32-step Qwen3.8-27B run over JACCL, KL vs the single-Mac reference was < 0.002 at every step except step 31 (0.025), and the argmax matched at every step. Check whether KL drifts upward over longer generations, e.g. if GatedDelta recurrent state accumulates sharding round-off.

- [ ] ShardCheck: a `--decode-steps 256` (and 1024) run of Qwen3.8-27B-4bit and Inkling-2-Mini-4bit on m5u2 ↔ m5u1 with MLX_METAL_FAST_SYNCH=1. The cluster session has the setup in /tmp/fmtest on m5u1.
- [ ] Plot or print KL per step. If it trends up, find which layer type drives it (shard only attention vs only GatedDelta) and whether float32 accumulation in the allSum fixes it.
- [ ] Record the numbers on this task and in cluster.md.

Depends on e8smzkz (Inkling sharding). #cluster