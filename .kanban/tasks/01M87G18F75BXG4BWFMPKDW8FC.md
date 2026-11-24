---
comments:
- actor: wballard
  id: 01m87h2hw5zvs8negz5xmb4rjq
  text: |-
    Implementation notes:
    - allToSharded projections (q/k/v, in_proj_*, gate/up) are split in place with shardInPlace through their owner module, so they stay stock QuantizedLinear: the owner's update override drops the fusion, then prepare() re-fuses over the sharded rows (verified at runtime: every attn/GDN/MLP fusion prepared after shard on 2B). shardedToAll outputs (o_proj, out_proj, down_proj) are replaced with Quantized/ShardedToAllLinear (allSum on CPU stream).
    - Pitfall hit: Conv1d stores `groups`; sharding only its weight fails with "[conv] input channels must be divisible by the number of groups". Fix: build a new depthwise Conv1d with channels/N groups (TensorParallel.makeShardedDepthwiseConv).
    - q_proj needs no segments: the reshape (B,L,heads,-1).split(2) means rows are per head [q_h|gate_h], so a contiguous row split keeps whole heads.
    - GDN GQA maps value head h to key head h/(Hv/Hk) (kernel and op path), so contiguous per-segment head splits stay consistent.
    - Ring backend runs fine with 2 local processes (hostfile [["127.0.0.1:p"],["127.0.0.1:p+1"]]).
    - MoE throws unsupported (no MoE model to validate on).
  timestamp: 2026-11-24T19:38:40.389361+00:00
- actor: wballard
  id: 01m87hjmfte214h30nhd7kgcs1
  text: |-
    ### implement — changed
    - evidence: de0b6969, 36c72079. shard-check PASS on Qwen3.5-2B-mxfp4 (9/9 tokens, max abs diff 0.1875) and Qwen3.8-27B-4bit (9/9, 0.1875). Full suite: XCTest 1260/0 failures; swift-testing 2618 tests, 1 unexpected failure (ChatSessionRecurrentCheckpointTests sliding-window fork), which passes alone (21/21).
    - next: review
  timestamp: 2026-11-24T19:47:27.354646+00:00
- actor: claude-code
  id: 01m87wjpgdyxm3gwsk6at8ew2j
  text: |-
    ### implement — changed (needs review)
    - Commits: de0b6969 feat(qwen35): shard the Qwen 3.5 family; 36c72079 test(shard-check); b75c680d (JACCL from the environment); 965d6424 (KL gate). Pushed to origin/cluster.
    - Results:
      - One Mac, ring: Qwen3.5-2B and Qwen3.8-27B-4bit both 9/9 tokens.
      - Real 2-Mac JACCL (m5u2 rdma_en6 ↔ m5u1 rdma_en5): 27B 33/33 tokens, max KL 0.025 (at step 31), every other step < 0.002. 14.3 ms/token with MLX_METAL_FAST_SYNCH=1, vs 18.7 on one Mac and 58 without FAST_SYNCH.
    - Open: Qwen35 MoE throws `.unsupported`. verify-docs, DocC/skill pages and a swift-format pass are still to do (covered by the review task).
  timestamp: 2026-11-24T22:59:43.757928+00:00
position_column: doing
position_ordinal: '80'
title: Tensor-parallel sharding for Qwen35 + 2-process ShardCheck harness
---
Add a TensorParallelShardable protocol in MLXLMCommon (shard(group:) with typed divisibility errors). Conform Qwen35 (full attention fused q/gate, GatedDeltaNet segments + conv1d + per-head params, dense MLP, MoE if cheap), handling fused decode projections and compiled traces. Build a ShardCheck executable + scripts/shard-check.sh that runs 2 local ring processes and compares logits against the unsharded reference. Validate with Qwen3.5-2B-mxfp4 and Qwen3.8-27B-4bit.

- [ ] TensorParallelShardable protocol + error type
- [ ] Qwen35 conformance
- [ ] Unit tests (shard plan, divisibility error)
- [ ] ShardCheck executable target
- [ ] scripts/shard-check.sh
- [ ] Run on 2B and 27B
- [ ] Full unit suite green