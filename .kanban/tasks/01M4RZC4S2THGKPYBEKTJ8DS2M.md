---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4vnpheydb536am8cqv16dkk
  text: 'Note from ^9xmds28 (not committed yet, in review): `DeepSeekV4Indexer.rankedChunks(_:queryResidual:pooledKeys:cos:sin:offset:)` now gives the top-k chunk INDICES, shape (batch, tokens, topK), for any block length; `callAsFunction` builds its membership mask from the same ranking (`topChunks(of:visible:)`). ^9xmds28 uses the indices only for decode (L = 1). The prefill path still builds the [B, L, topK, chunks] broadcast compare, so this card is unchanged. An alternative to the scatter: mlx-vlm gathers the top-k rows for L > 1 as well (`_gather_pooled`, `_sparse_pooled_attention` with a split softmax and `take_along_axis` of the pool mask), which removes the membership mask for prefill. Memory of that gather at a 2048-token prefill step: 2048 x 512 x 512 x 2 bytes = 1 GiB for each ratio-4 layer, against about 2 GiB of Booleans for the current compare at 8192 chunks (32k context). Measure before you choose.'
  timestamp: 2026-10-13T22:21:30.974251+00:00
- actor: claude-code
  id: 01m4vnpns3gsd1jbjfzjt8r48j
  text: 'Correction to the note above: the current broadcast compare at a 2048-token prefill step with 8192 chunks is 2048 x 512 x 8192 Booleans = 8 GiB for each ratio-4 layer, not 2 GiB. The gather alternative is 1 GiB (bf16 rows).'
  timestamp: 2026-10-13T22:21:35.395217+00:00
position_column: todo
position_ordinal: '8980'
title: 'DeepSeek-V4 prefill speed: indexer membership mask by scatter, not a broadcast compare'
---
## What

`Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:216-220` builds the membership mask of the selected chunks with a `[B, L, topK, chunks]` broadcast compare.

At prefill, L is large. In mlx-lm PR 1189 the same method needed about 34 GB for each layer at 128K context. A scatter (`put_along_axis`) gives the same mask with O(L * topK) writes. That change, with a fused scorer, took prefill at 64K context from 191 to 335 tok/s (PR 1189 comments, `c1189.md:1096-1124` in the session scratchpad `dl/`).

Also consider the mlx-vlm decode path, which gathers only the top-k pooled rows above 512 rows (`mlx_vlm .../language.py:426-473, 1011-1028`).

Baseline on 2026-10-12, M3 Ultra, GPU idle: prefill 397 tok/s at 2k, 375 at 8k, 276 at 32k.

## Acceptance Criteria

- [ ] The mask comes from a scatter, with a parity test against the broadcast compare.
- [ ] Peak memory and prefill speed at 32k context, before and after, on an idle GPU.
- [ ] Greedy output on the real checkpoint is the same as before.
#deepseek-v4 #performance