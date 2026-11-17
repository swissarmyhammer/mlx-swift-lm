---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jwznnsq3hzj3y26071x9jq
  text: |-
    Research + profiling (picked up at HEAD 6e90d12e). Temporary host timers in speculateRound / chainedMTPDrafts / the VLM verify call (ZZProf, scratchpad only, to be removed), Qwen3.8-27B-4bit, qids prompt, bs4 pMin 0.75, ms per round:

    | ctx | gate wait (commit fwd) | chain build+encode | chain waits | verify graph build | verify asyncEval (throttled encode) | verify readback | commit build+encode | round |
    |---|---|---|---|---|---|---|---|---|
    | 2k | 1.41 | 0.33 | 1.56+0.14 | 1.62-1.80 | 20.2 | 4.2 | 0.10+0.20 | 30.1-30.4 |
    | 32k | 1.45 | 0.23 | 1.21+0.07 | 1.14-1.42 | 21.8 | 3.0 | 0.09+0.21 | 29.6-29.8 |

    pMin 0 at 2k: verify build 2.15, round 32.2 with 3.57 tok/verify.

    Metal System Trace (xctrace, 2k, 1.48 s steady window, scratchpad/zz0.trace + zzgpu.py): GPU busy 88.3%, idle 173 ms = ~3.6 ms/round: about one 1.6-2.2 ms gap per round (the verify graph build after the last chain read) and ~3 gaps of 0.3-0.4 ms (step0 encode after the seed gate, commit encode after the readback, launch latency).

    Findings:
    - The drafter step graph build is already tiny: the whole compiled fc+layer call costs ~0.015 ms of host time per call; the encode (asyncEval) of a chain step is ~0.11 ms. Graph build of the drafter is not where the stall is.
    - The ~2 ms stall the card names is the TARGET verify graph build (1.6 ms, compiled segments, no sync inside) that runs after the last chain-step probability read, while the GPU idles.
    - asyncEval of the verify blocks ~20 ms: MLX throttles the host at MAX_ACTIVE_TASKS command buffers, so the host is in lockstep with the verify on the GPU.

    Experiment 1 (reverted, NO-GO): defer the chain's floor reads past the verify build (build a verify of every chained draft, then read the probabilities and cap acceptance). GPU idle dropped 173 -> 132 ms per window (busy 91.1%), but GPU work per token rose 8.27 -> 8.64 ms and ms/token did not improve (2k 9.35-9.51 vs 9.37). It also CHANGED acceptance (perVerify 3.25 vs 3.29, 79 vs 78 rounds): verify rows are not bit-stable across row counts (3-row vs 4-row compiled verify), so a capped 4-row verify is not the 3-row verify. Not allowed by the card (accepted tokens must stay identical).

    Experiment 2 (prototype): compiled drafter step in the VLM predictor (one trace for pre_fc norms+concat+fc+input norm+qkv, the KV write+M-RoPE+attention op path between, one trace for o_proj+residual+MLP+final norm). Isolated drafter step (qwencli draft, 2k): L1 1.30 -> 1.27 ms, L2 1.46 -> 1.39, L4 1.60 -> 1.48. In-round: no visible change in the gate wait or chain waits (30.2 ms round). Running qwenbench c0/c1 A/B now.
  timestamp: 2026-11-16T19:22:45.817971+00:00
- actor: claude-code
  id: 01m7jyqjc5x4kxpc9fajahxg2v
  text: |-
    ### Measured NO-GO: no library change; the working tree is back to HEAD 6e90d12e (only .kanban/ is modified)

    qwenbench (median of 3 each, same session, scratchpad/qwen_bench_nsh_c0.txt and qwen_bench_nsh_c1.txt):

    | metric | stable | ^ets0cjp | HEAD re-run (c0) | compiled drafter (c1) | c1 vs ^ets0cjp | c1 vs stable |
    |---|---|---|---|---|---|---|
    | mtp 2k ms/tok | 10.17 | 9.34 | 9.35 | 9.35 | +0.1% | -8.1% |
    | mtp 32k | 13.02 | 12.48 | 12.55 | 12.50 | +0.2% | -4.0% |
    | mtp 64k | 14.20 | 13.53 | 13.57 | 13.58 | +0.4% | -4.4% |
    | plain 2k/32k/64k | 18.26/20.60/23.44 | 17.04/19.77/22.67 | 17.09/19.87/22.74 | 17.10/19.84/22.74 | noise | unchanged |
    | tok/verify | 3.29/2.40/2.65 | same | same | same | | |

    The compiled drafter step against HEAD measured 0.0% / -0.4% / +0.1%, which is inside run-to-run noise (c0 vs ^ets0cjp moved +0.1 to +0.6% with no code change).

    Why there is no gain:
    1. **Compile the drafter step.** The drafter graph build was already about 0.015 ms per call. A chain step's encode costs about 0.11 ms. The GPU step is 1.27 ms compiled against 1.30 ms op path. In a round of about 30 ms, that is not measurable. The KV write, M-RoPE and attention cannot go inside a trace, because the cache grows every token. So the step would always be two traces around the attention, as the target layers already are.
    2. **Overlap the gate and readback with graph builds.** The stall the card names (about 2 ms per round) is the target verify graph build (1.6 ms) after the last chain-step floor read. Its shape depends on that read. Overlapping it means either deferring the floor reads (Experiment 1) or a speculative verify build with a cache rollback. Experiment 1 removed most of the 2 ms GPU gaps (GPU busy 88.3% -> 91.1% in a Metal System Trace), but ms/token did not improve, because GPU time per token rose 8.27 -> 8.64 ms. It also changed acceptance (3.25 vs 3.29 tok/verify), because verify rows are not bit-identical across verify row counts. Building step0 before the gate read saves only that build (about 0.05 ms). The commit after the verify readback needs the accepted count, so it cannot be built earlier without the same row-count problem.

    qwencheck and the five test bundles were not run, because the library is unchanged. Remaining ideas, outside this card: make the target verify graph build itself cheaper (1.6 ms of host time per round on compiled segments), or reduce MLX command-buffer launch gaps (about 3 gaps of 0.35 ms per round).
  timestamp: 2026-11-16T19:53:17.445649+00:00
- actor: claude-code
  id: 01m7jyqmk0807a2cfxwhegq0fn
  text: |-
    ### implement — no-change
    - evidence: measured NO-GO. qwenbench c1 (compiled drafter) vs HEAD c0: MTP 9.35/12.50/13.58 vs 9.35/12.55/13.57 ms/tok (noise). Deferred floor-read scheduling: no ms/tok gain, and acceptance changed (3.25 vs 3.29), so it is not allowed. All library edits were reverted; git status shows .kanban/ only.
    - next: the orchestrator closes the card as a measured NO-GO, or the user re-scopes it to the target verify graph build cost.
  timestamp: 2026-11-16T19:53:19.712838+00:00
- actor: claude-code
  id: 01m7jyryte3wk55t60633dr5zv
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (44f5ef94): 0 findings, 0 confirmed, 0 refuted; 4 .kanban/ files with no matching validator. `git diff 6e90d12e..HEAD` outside .kanban/ is empty, and the working tree is clean. Acceptance: qwencheck and the five bundles are marked not applicable (NO-GO, no code change). The qwenbench delta is on the card.
    - next: closed as a measured NO-GO, as ^117ymn8 was.
  timestamp: 2026-11-16T19:54:02.958735+00:00
- actor: claude-code
  id: 01m7jys9mcxj6nfyhnxfkpdyvh
  text: |-
    ### finish iteration 1 — clean
    - implement: no-change — measured NO-GO, library edits reverted
    - test: n/a — no code change
    - commit: 44f5ef94 (kanban evidence)
    - review: clean → done
  timestamp: 2026-11-16T19:54:14.028216+00:00
depends_on:
- 01M7J8449YT9F98KXC0117YMN8
position_column: done
position_ordinal: ffffb980
title: 'Qwen3.8 MTP round: compile the drafter step and remove CPU stalls'
---
## Why
The seed-gate `.item()` reads and the verify readback leave the GPU idle while the next graph builds. Comparing pMin 0.5 against pMin 0 puts this at about 2 ms per round at 2k. Flint commit 864b34d9 compiled the one-token decode trace (CompiledTrace per block).

## What
- Compile the drafter step (fc + 1 layer + lm_head) into one trace.
- Schedule the round so the gate decision and the verify readback overlap the next graph's build (asyncEval ordering) without changing which tokens are accepted.

Files: `QwenMTP.swift`, `Qwen35MTP.swift`, `MTPSpeculativeTokenIterator.swift`.

## Acceptance
- [x] `qwencheck.sh`: MTP greedy ids identical to the golden file, and the multi-turn cache-reuse check passes. ChatSession and FoundationModels drafter-state restore are unaffected. — not applicable: NO-GO, no code change (`git diff 6e90d12e..HEAD` outside `.kanban/` is empty)
- [x] All five bundles are green. `qwenbench.sh` delta on this card. #qwen-perf — five bundles: not applicable: NO-GO, no code change. qwenbench delta: on this card (comment "Measured NO-GO": compiled drafter c1 vs HEAD c0 MTP 9.35/12.50/13.58 vs 9.35/12.55/13.57 ms/tok at 2k/32k/64k, inside noise; tok/verify unchanged)