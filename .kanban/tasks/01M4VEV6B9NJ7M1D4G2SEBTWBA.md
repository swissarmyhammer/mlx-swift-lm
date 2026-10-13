---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4vhtzqtskf36e4rj59mwvyh
  text: |-
    Research and measurements (2026-10-13). All files are in the session scratchpad.

    Tools added:
    - dsv4bench modes: `spreadids <out.json> <ctx>` (writes the chat prompt ids), `spread <ids.json> <k> <prefix>` (four ways, float32 logits of the last k positions to <prefix>-<way>.f32), `sdpa <n,...>` (pure MLX check, no model), `halves <ids.json> <layers>` (Debug build only, `@testable`, per layer-half bisect).
    - scripts: vlm_spread.py (same four ways on mlx-vlm), spread_report.py (pairwise report), vlm_spread_bisect.py and vlm_spread_halves.py (layer and branch bisect on mlx-vlm), vlm_layer0_window.py, sdpa_long_mask.py.
    - Same ids for both ports: spread/ids.json (8018 tokens) and spread/ids4k.json (4018 tokens). k = 16. Ways: block, chunk2048, chunk512, steps (block of N-16, then 16 one-token steps).

    Four-way spread, largest |diff| over the last 16 positions. "all" = full vocabulary; "head" = tokens within 10 of the top logit:
    - 8018 tokens, ours: all 10.78, head 4.25 (head without the one-block way: 4.0).
    - 8018 tokens, mlx-vlm: all 13.81, head 6.25 (without the one-block way: 5.375).
    - 4018 tokens, ours: all 17.19, head 2.0. mlx-vlm: all 16.09, head 4.625.
    - Our port is deterministic: a second run gave 0.0 difference for each way.
    Result: our spread is NOT larger than the reference spread. Criterion 3 applies.

    Cause 1 (one-block way, both ports): an MLX bug. SDPA with sinks and an array mask (head dim 512 has no fused kernel) gives wrong values when 64 x n x (n + 1) is more than 2^31. Pure-MLX check, float32, last row against a reference: n=5792 1e-6, n=5793 3e-6, n=6000 0.024, n=8018 0.061; without sinks 1e-6 for all n. Same in Python MLX 0.32.3 and in mlx-swift. Real checkpoint, layer 0, float32: blocks of 128..4096 tokens give the same last-token attention output (0.0), the 8018 block gives 3.3 %. The halves bisect shows the same picture in both ports: the one-block way differs from the other three ways by 4 % at the layer-0 attention and by 50 % at the layer-1 attention; the other three ways agree to about 1.5 % there.

    Cause 2 (the remaining spread, both ports): bf16 rounding that changes with the cut, made larger by discrete choices. Layers 0-1 (window only): about 1 % between the chunked and stepwise ways. The jump comes at layer 2, the first ratio-4 layer (attention branch 10 % between chunk2048 and steps in both ports). In mlx-vlm the indexer top-512 sets differ by 12 to 450 entries between ways, and the routed experts differ by 2 to 6 in many layers. The prompt is one paragraph repeated, thus many pooled rows are near copies and the indexer choice between them is a near tie.

    The card evidence (gap 1.75) came from TieCheck: it read the 8050-token cold prompt in ONE block, thus it hit cause 1.

    No commit bisect: the cause is not a commit of this port.

    New work: ^f24gprg. Our production prefill can reach the same MLX limit: unchunked prefill or a direct model call over more than 5792 tokens, or 2048-token chunks at about 57k context (ratio-4 layers hold context/4 pooled rows as keys), or 512-token chunks at about 259k.
  timestamp: 2026-10-13T21:14:02.362744+00:00
- actor: claude-code
  id: 01m4vhvjjfrhs1n5k9qa66b9t3
  text: |-
    ### implement — no-change
    - evidence: no production or test file changed (git status shows only .kanban files). Four-way spread (head = tokens within 10 of the top logit, last 16 positions): 8018 tokens ours 4.25 (4.0 without the one-block way) vs mlx-vlm 6.25 (5.375); 4018 tokens ours 2.0 vs mlx-vlm 4.625; full vocabulary 8018 ours 10.78 vs mlx-vlm 13.81. Ours is not larger. Cause of the one-block outlier: MLX SDPA with sinks overflows a 32-bit index at 64 x n x (n + 1) > 2^31 (n > 5792), same in Python MLX and mlx-swift. Scratch bench changed (not in the repository): TieCheck now reads the teacher-forced forward in 2048-token blocks and uses the margin cutSpreadMargin = 4.0. Bench run `<snapshot> 32 16 8192 1024`: chat turn 2 gap 0.75 ("ulls" 42.25 vs "ull" 41.5), thinking turn 2 gap 0.0, prefix argmax mismatches 0 in both, both verdicts PASS (tie within the cut spread), HARD reuse check PASS in both. Output in scratchpad dsv4bench/tie-sebtwba.txt.
    - next: review. New task ^f24gprg holds the production fix for the MLX limit.
  timestamp: 2026-10-13T21:14:21.647400+00:00
position_column: doing
position_ordinal: '80'
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

- [x] The four-way logit comparison is recorded, for our port and for mlx-vlm.
- [ ] If our spread is larger than the reference spread: the cause is found and fixed, with a unit test that cuts one block in different ways and compares the logits. (Does not apply: our spread is not larger. See the comments.)
- [x] If our spread is the same as the reference spread: the reason is recorded, and the warm == cold check uses that measured spread as its margin.
#deepseek-v4