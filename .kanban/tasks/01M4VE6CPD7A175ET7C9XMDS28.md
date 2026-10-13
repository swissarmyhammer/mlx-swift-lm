---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: 'DeepSeek-V4 speed: sparse attention gathers the top-k pooled chunks, not all chunks behind a mask'
---
## What

After task ^cm8fq2q, a decode step does no compressor work when no compress window is full. The decode step still grows with the context: 45.6 ms at 4k and 73.6 ms at 32k (scratch bench `step`, direct sync, GPU idle). mlx-vlm on the same machine is flat at about 31 ms.

On a ratio-4 layer with an indexer, `DeepSeekV4Attention.attentionOutput` concatenates the window keys with ALL pooled chunks (8192 chunks at 32k) and gives SDPA a joined Boolean mask of the same width (`chunkVisibility`, `joinedMask`, `keys`). mlx-vlm (`SparseCompressedAttention._attend`, `_sparse_pooled_attention` in `deepseek_v4/language.py`) gathers only the `index_topk` chunks the indexer selected when the pool holds more than `index_topk` chunks, and splits the softmax between the window and the gathered chunks.

This is a hypothesis from the code. It is not yet measured with a profile.

## Approach

- Profile one decode step at 32k (scratch bench `profile`) and confirm which primitives grow with the chunk count.
- Gather the top-k pooled chunks for each query when the pool holds more than `index_topk` chunks, as mlx-vlm does.
- Keep the greedy parity with the mlx-vlm fixture and the cut rule of `DeepSeekV4CacheTests`.

## Acceptance Criteria

- [ ] A profile at 32k names the primitives that grow with the chunk count.
- [ ] A decode step at 32k reads at most `index_topk` pooled chunks on a layer with an indexer.
- [ ] The 64-token greedy parity integration test passes.
- [ ] A decode-step measurement with the GPU otherwise idle shows the speed before and after, at 4k and at 32k. #deepseek-v4 #performance