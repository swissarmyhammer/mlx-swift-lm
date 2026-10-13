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
- actor: claude-code
  id: 01m4vhxwb133z1chkch03h8q9e
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (4388d9e): 0 engine findings, 0 files reviewed (4 .kanban files, no validator matches). Acceptance check: criterion 1 has its evidence (four ways, both ports, 8018 and 4018 tokens). Criterion 3 has its evidence (reason: MLX SDPA sinks 32-bit index limit plus bf16 rounding that changes with the cut; TieCheck margin cutSpreadMargin = 4.0, which is the measured 8018-token head spread without the one-block way; result: chat gap 0.75, thinking gap 0.0, both PASS). 1 finding: criterion 2 — for 4018 tokens the full-vocabulary spread is 17.19 for our port and 16.09 for mlx-vlm, and no comment tells why this case does not count.
    - next: add a comment that names the deciding metric and explains the 4018-token full-vocabulary case, then review again.
  timestamp: 2026-10-13T21:15:37.185170+00:00
- actor: claude-code
  id: 01m4vhydt0eha7rv9265r9vh2b
  text: |-
    ### finish iteration 1 — findings
    - implement: no-change — measurement only; scratch bench TieCheck changed (outside the repository); new task ^f24gprg
    - test: not run — no repository source changed
    - commit: 4388d9e (kanban only)
    - review: findings — acceptance criterion 2: 4018-token full-vocabulary spread ours 17.19 > mlx-vlm 16.09, no comment explains it
  timestamp: 2026-10-13T21:15:55.072635+00:00
- actor: claude-code
  id: 01m4vj5kfetsqeywzcftkshaec
  text: |-
    Iteration 2: data for the review finding (full-vocabulary spread at 4018 tokens, ours 17.19 > mlx-vlm 16.09). Script: scratchpad scripts/spread_where.py. Outputs: spread/where4k.txt, spread/where8k.txt, spread/where-mixed4k.txt.

    1. Where the full-vocabulary maximum sits (harbor prompt, 4018 tokens). For each pair, the token with the largest |diff| is in the far tail in BOTH ways of the pair:
    - Ours: the maximum 17.19 (chunk2048 vs steps) is token 121932 at position -4, rank 128764 / 105879 of 129280, logit minus top -46.5 / -29.9. Every other pair: rank 16178 or lower, logit minus top -19.7 or lower in the higher of the two ways.
    - mlx-vlm: the maximum 16.09 (block vs chunk2048) is the same token 121932 at position -4, rank 119259 / 128768, logit minus top -31.8 / -47.3.
    - A token 20 below the top has a probability of at most e^-20 (2e-9) at temperature 1. Greedy, top-k and top-p sampling never pick it.
    Banded spreads (largest over the 6 pairs):
    - Harbor 4018: within 10 of top: ours 2.0, mlx-vlm 4.625. Within 20: ours 12.19, mlx-vlm 11.91. Largest total-variation distance of the softmax: ours 0.198, mlx-vlm 0.562. Top-1 disagreements (6 pairs x 16 positions): ours 7, mlx-vlm 11.
    - Harbor 8018 (one-block way has the MLX bug in both ports): within 10: ours 4.25, mlx-vlm 6.25. Within 20: ours 6.63, mlx-vlm 6.41. TV: ours 0.585, mlx-vlm 0.689. Top-1 disagreements: ours 13, mlx-vlm 7.

    2. Second sequence: spread/ids-mixed4k.json, 4018 tokens, a chat prompt of a mixed-language preface (German, French, Japanese, Spanish, English) followed by the Swift source of DeepSeekV4Attention.swift (cut at 4018 tokens). Both ports read the same ids.
    - Full vocabulary: ours 3.125, mlx-vlm 3.1875.
    - Within 10 of top: ours 2.25, mlx-vlm 2.0.
    - Within 20 of top: ours 2.4375, mlx-vlm 2.75.
    - Largest TV distance: ours 0.180, mlx-vlm 0.152.
    - Top-1 disagreements: ours 0, mlx-vlm 0.
    - The "within 10" maximum: ours token 712 at position -11, 19.5 vs 17.25 (top 29.1); mlx-vlm token 14500 at position -12, 16.875 vs 18.5 (top 27.5). bf16 step at these values is 0.125, thus 2.25 vs 2.0 is 2 steps.

    3. Decision. The metrics that affect generation are "within 10 of top", TV distance and top-1 agreement; tokens more than 20 below the top cannot be sampled, thus the full-vocabulary maximum does not decide. On those metrics the ports trade places: on the harbor prompt ours is smaller (head 2.0 vs 4.625, TV 0.198 vs 0.562, top-1 7 vs 11); on the mixed prompt ours is larger (head 2.25 vs 2.0, TV 0.180 vs 0.152; top-1 0 vs 0). At 8018 tokens ours has more top-1 disagreements (13 vs 7). Thus ours IS larger on a metric that affects generation for one sequence. By the rule of the coordinator, criterion 2 applies. Outcome: stuck. No production code changed. A person must decide whether a difference of 0.25 in head (2 bf16 steps) and 0.03 in TV on one sequence, against an opposite and larger difference on the other, makes criterion 2 apply, or whether more sequences and positions are necessary.
  timestamp: 2026-10-13T21:19:50.254856+00:00
- actor: claude-code
  id: 01m4vj6geejnp76df29je4ndh0
  text: |-
    ### implement — stuck
    - evidence: no repository file changed except .kanban. Full-vocabulary maxima are far-tail tokens (harbor 4018: ours 17.19 at rank 128764/105879, logit-top -46.5/-29.9; mlx-vlm 16.09 at rank 119259/128768, -31.8/-47.3). Generation metrics, ours vs mlx-vlm: harbor 4018 head(10) 2.0 vs 4.625, TV 0.198 vs 0.562, top-1 disagreements 7 vs 11; mixed-language 4018 head(10) 2.25 vs 2.0, within-20 2.4375 vs 2.75, TV 0.180 vs 0.152, top-1 0 vs 0, full 3.125 vs 3.1875; harbor 8018 head 4.25 vs 6.25, TV 0.585 vs 0.689, top-1 13 vs 7. Ours is larger on head and TV for the mixed sequence, thus by the coordinator rule criterion 2 applies. No production code changed.
    - next: a person decides whether a 2-bf16-step head difference and 0.03 TV on one sequence (opposite and larger on the other) makes criterion 2 apply, or whether more sequences/positions are needed. Review finding checked: the iteration-2 comment holds the evidence.
  timestamp: 2026-10-13T21:20:19.918548+00:00
- actor: claude-code
  id: 01m4vjzxm0ajh8smbwtfmp1cyb
  text: |-
    ### Decision by the user — 2026-10-13
    The two ports are equivalent within bf16 noise on the measures that change generation (spread within 10 of the top, total-variation distance, top-1 changes): each port is smaller on one of the two 4018-token texts, and the largest difference against our port is 0.25 logit (2 bf16 steps) on the mixed-language text, with 0 top-1 changes in both ports. The far-tail full-vocabulary maximum (rank about 106k-129k of 129280, 30-47 below the top) does not affect generation. Thus criterion 2 does not apply. The original 1.75 gap is the MLX attention overflow above 5792 tokens, tracked in ^f24gprg. No production change.

    ### finish iteration 2 — stuck, then decided
    - implement: no-change — measurements on a second text and tail ranks; review finding checked
    - test: not run — no repository source changed
    - decision: the user closed criterion 2 as not applicable (above)
  timestamp: 2026-10-13T21:34:12.608759+00:00
- actor: claude-code
  id: 01m4vk15h1twpne1p2gr8cz0wy
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (724791f): 0 engine findings (findings 0, confirmed 0, refuted 0), 0 files reviewed (4 .kanban files, no validator matches). The prior finding (2026-10-13 16:15) is checked, and the comments hold its evidence: the tail ranks of the full-vocabulary maximum (token 121932, rank 105879-129280, 30-47 below the top, in both ports), the second-text measurements for both ports (mixed-language 4018 tokens: full 3.125 vs 3.1875, head 2.25 vs 2.0, TV 0.180 vs 0.152, top-1 0 vs 0), and the user decision. Acceptance check: criterion 1 has its evidence (research comment). Criterion 2 is marked NOT APPLICABLE by the comment "Decision by the user — 2026-10-13" (box text edited, original text kept). Criterion 3 has its evidence (MLX SDPA sinks 32-bit index limit plus bf16 rounding that changes with the cut; TieCheck margin cutSpreadMargin = 4.0; both verdicts PASS).
    - next: done. ^f24gprg holds the production fix for the MLX limit.
  timestamp: 2026-10-13T21:34:53.473332+00:00
position_column: done
position_ordinal: ffbf80
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
- [x] NOT APPLICABLE — closed by the comment "Decision by the user — 2026-10-13": the two ports are equivalent within bf16 noise on the generation measures, thus this criterion does not apply. Original text: If our spread is larger than the reference spread: the cause is found and fixed, with a unit test that cuts one block in different ways and compares the logits. (Open: see the iteration-2 comment. On the mixed-language sequence ours is larger on two generation metrics, by 2 bf16 steps in head and 0.03 in TV. A person must decide.)
- [x] If our spread is the same as the reference spread: the reason is recorded, and the warm == cold check uses that measured spread as its margin.
#deepseek-v4

## Review Findings (2026-10-13 16:15)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 0 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4VEV6B9NJ7M1D4G2SEBTWBA.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4VEV6B9NJ7M1D4G2SEBTWBA.md` — no validator matches this file
> - `.kanban/tasks/01M4VHTF0G6G964S3JSF24GPRG.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4VHTF0G6G964S3JSF24GPRG.md` — no validator matches this file

> Acceptance check against the task comments:

- [x] Acceptance criterion 2 ("our spread is not larger") — the research comment does not fully support this. For 4018 tokens, the full-vocabulary spread of our port is 17.19, and the mlx-vlm spread is 16.09. On this metric our spread is larger. The comment states "our spread is NOT larger" and gives no reason for this case. Add a comment that states which metric decides the criterion (for example, the head metric, tokens within 10 of the top logit), and why the 4018-token full-vocabulary result does not make criterion 2 apply. If you cannot give that reason, criterion 2 applies.