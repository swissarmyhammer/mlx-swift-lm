---
assignees:
- claude-code
position_column: todo
position_ordinal: '8780'
title: 'DeepSeek-V4 correctness: check the orientation of the hyper-connection expand (comb or its transpose)'
---
## What

`Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift:296` (expand / `hc_post`) computes `comb @ residual`.

The references do not agree:
- `comb^T @ residual`, or the same as a sum over axis 2:
  - ml-explore/mlx-lm main `mlx_lm/models/deepseek_v41.py:256` (Apple)
  - Blaizzy/mlx-vlm `mlx_vlm/models/deepseek_v4/hyper_connection.py:425`
  - the official DeepSeek code
- `comb @ residual`, the same as ours:
  - mlx-lm PR #1195 `deepseek_v4.py:323`
  - Thump604/mlx-lm `deepseek-v4-support-fixes`

Blaizzy changed to the transpose in mlx-lm PR #1192 after reports of dropped tokens and repetition. See the PR #1192 comments, saved as `c1192.md:573` in the session scratchpad `dl/` folder.

Our greedy parity test (32 of 32 tokens, task ^e7b24ws) used a fixture from mlx-lm PR #1189 / Thump604, which has the same orientation as ours. Thus that test cannot find this fault.

## Approach

1. Read the official DeepSeek-V4 inference code (`hc_post`) and decide the correct orientation. Record the source and the line.
2. If ours is wrong: correct it, make a new parity fixture from a reference with the correct orientation, and run the real-weights checks: greedy text, chat and thinking, tool call, two-round recall.
3. If ours is correct: record why, with the source.

## Acceptance Criteria

- [ ] The orientation is decided against the official source, and the decision and the source are recorded on this task.
- [ ] A unit test pins the orientation with a non-symmetric `comb`.
- [ ] If the code changes: greedy output on the real checkpoint is checked against a reference that has the correct orientation.
#deepseek-v4