---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'DeepSeek-V4: one attention call over too many scores gives wrong values (MLX 32-bit index overflow with sinks)'
---
## What

Found on 2026-10-13 during ^sebtwba.

`MLXFast.scaledDotProductAttention` with `sinks` and an array mask gives wrong values when `heads x queries x (keys + 1)` is more than 2^31. Head dim 512 has no fused kernel, thus MLX uses the op path. That path holds a 32-bit index. MLX 0.32.3, Python (mlx-vlm venv) and mlx-swift, same result.

Evidence (session scratchpad `scripts/sdpa_long_mask.py` and `dsv4bench sdpa`): 64 query heads, 1 key head, head dim 512, float32, 128-key window mask, the last query row against a float32 reference:
- n = 5792: relative error 1e-6 (64 x 5792 x 5793 is less than 2^31).
- n = 5793: 3e-6. n = 6000: 0.024. n = 8018: 0.061.
- Without sinks: 1e-6 for all n up to 8193.

On the real checkpoint, one block of 8018 tokens changes the layer-0 attention output of the last token by 3.3 % in float32 (blocks of 128 to 4096 tokens: 0.0 %). Both ports show it.

## Where this port can reach the limit

`DeepSeekV4Attention.attentionOutput` gives SDPA the window keys AND every pooled row (the mask selects the indexer top-k). The score count of one call is `64 x L x (windowKeys + pooledRows + 1)`.
- Unchunked prefill (`PrefillParameters.Chunking.unchunked`) or a direct `model(tokens)` call: every block of more than 5792 tokens.
- Prefill in 2048-token chunks: a ratio-4 layer holds `context / 4` pooled rows, thus a context of about 57k tokens or more.
- Prefill in 512-token chunks (the default step): about 259k tokens or more.

## Approach

1. Write a unit test that compares one attention call over a long block with the same block cut into query slices (small shapes are not enough; the test must cross 2^31 scores, or the split rule must be tested as a pure function).
2. Split the queries of one SDPA call so that each call holds less than 2^31 scores. Keep the decode path as it is.
3. Report the MLX issue upstream (ml-explore/mlx).

## Acceptance Criteria

- [ ] No SDPA call in DeepSeek-V4 attention holds 2^31 or more scores.
- [ ] A unit test proves the split.
- [ ] The real-weights greedy parity test and the scratch bench full mode still pass. #deepseek-v4