---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: 'DeepSeek-V4 decode performance: about 2.4 s/token blocks the >12k-token test'
---
## What

Measured 2026-08-12 on the M3 Ultra (512 GiB), `mlx-community/DeepSeek-V4-Flash-4bit`, card ^e7b24ws:

- 64 greedy tokens in ~177-210 s inside the parity test (load excluded) — about 2.4-3.3 s/token.
- The chat/thinking test took 232 s for two 48-token generations.
- At this speed the 12,400-token issue-1662 regression test (`longGenerationPastTwelveThousandTokensCompletes`) needs about 7 hours. The suite's per-test limit is 240 minutes, thus the test cannot complete. A run was killed after 83 minutes in progress.

For contrast, the Python reference (mlx-lm PR 1189) also decodes slowly on this checkpoint (~3.7 s/token for its 64-token fixture run), thus the cause is likely in the model's compute shape (284B-total MoE gathers), not only in the Swift port. Profile the Swift decode step, find the dominant cost (switch_mlp gather, hyper-connection, attention), and compare with the Python reference throughput.

## Acceptance Criteria

- [ ] A profile names the dominant cost of one decode step with numbers.
- [ ] A decision or a fix: either decode gets fast enough that 12,400 tokens fit inside the 240-minute test limit, or the card records why not and what the >12k test should do instead.

#deepseek-v4