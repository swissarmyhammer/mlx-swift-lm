---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'DeepSeek-V4: logits differ by up to 1.75 between one-block and chunked/stepwise computation of the same tokens'
---
## What

Found on 2026-10-13 during ^cm8fq2q with the tie-aware warm == cold check of the scratch bench (session scratchpad `dsv4bench`, `TieCheck.swift`), real checkpoint mlx-community/DeepSeek-V4-Flash-4bit, 8k-token context, chat mode, turn 2:
- The cold run (prefill in steps, then one token at a time) picked "ull".
- One teacher-forced forward over the same prompt plus the same answer prefix ranks "ulls" first at 41.75, and "ull" at 40.0, which is 1.75 below.
- bf16 rounding at this logit size is 0.25. A gap of 1.75 is not rounding noise.
- In thinking mode, the teacher-forced pass picks a different token from the cold run at 1 of 28 shared tokens.

At HEAD 9038b29, before ^cm8fq2q, the chat-mode warm and cold outputs already differed. Before commit e08347b (the hyper-connection transpose), both modes matched. Thus the cause is older than ^cm8fq2q.

A possible cause: the result depends on how a block is cut (prefill chunk boundaries against compress-window boundaries, the sliding window, or the indexer). `DeepSeekV4CacheTests` states the rule that the pooled chunks must not depend on the cut. This test covers the pooled chunks, not the full logits.

## Approach

1. On the real checkpoint, take one fixed sequence of about 8k tokens. Compute the last-position logits in four ways:
   - one block;
   - prefill in 2048-token chunks;
   - prefill in 512-token chunks;
   - prefill of N-k tokens followed by k single-token steps.
2. Report the largest absolute logit difference between the ways, and the token where it appears.
3. If a difference is larger than bf16 noise, bisect it by layer and by branch (sliding window, compressed chunks, indexer).
4. Do the same with the mlx-vlm reference (session scratchpad `py/venv-vlm`). This shows whether the reference has the same spread.

## Acceptance Criteria

- [ ] The four-way logit comparison is recorded, for our port and for mlx-vlm.
- [ ] If our spread is larger than the reference spread: the cause is found and fixed, with a unit test that cuts one block in different ways and compares the logits.
- [ ] If our spread is the same as the reference spread: the reason is recorded, and the warm == cold check uses that measured spread as its margin.
#deepseek-v4