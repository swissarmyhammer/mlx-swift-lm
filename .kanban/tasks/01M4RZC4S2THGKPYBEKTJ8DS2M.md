---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'DeepSeek-V4 prefill speed: indexer membership mask by scatter, not a broadcast compare'
---
## What

`Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift:216-220` builds the membership mask of the selected chunks with a `[B, L, topK, chunks]` broadcast compare.

At prefill, L is large. In mlx-lm PR #1189 the same method needed about 34 GB for each layer at 128K context. A scatter (`put_along_axis`) gives the same mask with O(L * topK) writes. That change, with a fused scorer, took prefill at 64K context from 191 to 335 tok/s (PR #1189 comments, `c1189.md:1096-1124` in the session scratchpad `dl/`).

Also consider the mlx-vlm decode path, which gathers only the top-k pooled rows above 512 rows (`mlx_vlm .../language.py:426-473, 1011-1028`).

Baseline on 2026-10-12, M3 Ultra, GPU idle: prefill 397 tok/s at 2k, 375 at 8k, 276 at 32k.

## Acceptance Criteria

- [ ] The mask comes from a scatter, with a parity test against the broadcast compare.
- [ ] Peak memory and prefill speed at 32k context, before and after, on an idle GPU.
- [ ] Greedy output on the real checkpoint is the same as before.
#deepseek-v4 #performance