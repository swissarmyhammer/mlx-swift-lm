---
assignees:
- claude-code
position_column: todo
position_ordinal: '8380'
title: 'DeepSeek-V4 speed: one MLXFast.rope call with a freqs table'
---
## What

Our port calculates cos/sin tables in each layer, then slices, stacks and concatenates the rotary and non-rotary parts of each head.

Our code:
- `DeepSeekV4Attention.swift:193-209, 411-438, 669-689`
- `DeepSeekV4MathHelpers.swift:192-230`
- `DeepSeekV4Indexer.swift:149`
- `DeepSeekV4Compressor.swift:200`

Blaizzy/mlx-vlm `deepseek_v4/language.py:258-339` gives an infinite period to the pairs that take no position. Then one `rope` call with a `freqs` table turns the full head, with no slice and no concatenation. The YaRN scaling (`rope_scaling` factor 16) and `compress_rope_theta` go into the table.

## Evidence

In mlx-lm PR #1192, the Apple (angeloskath) RoPE changes took decode on an M3 Ultra from 30.2 to 33.6 tok/s.

## Acceptance Criteria

- [ ] Attention, the indexer and the compressor each turn positions with one `MLXFast.rope` call.
- [ ] Parity tests show the same results as the current path, for both `rope_theta` and `compress_rope_theta` layers, and with YaRN scaling.
- [ ] The existing DeepSeek-V4 unit tests pass.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after.
#deepseek-v4 #performance