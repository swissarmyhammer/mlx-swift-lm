---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: 'DeepSeek-V4 decode speed: remove the 258 scale Gather dispatches in each step'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`, `prof4/dot_decode_counts.txt`).

`scaleF32[preScaleIndex]`, `scaleF32[postScaleIndex]` and `scaleF32[combScaleIndex]` in `DeepSeekV4MathHelpers.swift` (lines 144, 149, 155) each make a `Gather` node. A decode step has 265 `Gather` nodes; mlx-vlm has 25. Make the three scale values once at load, or use a slice.

Expected gain: about 0.3-0.6 ms/token.

## Acceptance Criteria

- [ ] No `Gather` node for the scales in the decode graph.
- [ ] The compiled decode parity tests (same bits) pass.
- [ ] Decode ms/token at 32k, before and after, GPU idle. #deepseek-v4 #performance