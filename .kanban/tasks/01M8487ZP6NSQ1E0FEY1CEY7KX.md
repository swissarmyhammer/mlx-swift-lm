---
assignees:
- claude-code
position_column: done
position_ordinal: ffffdd80
title: 'Inkling 2 perf harness: bench script + golden greedy ids (baseline on stable)'
---
## Why
Every inkling-perf card must report its speedup against the previous best and prove that the output is unchanged, as the qwen-perf batch did with qwenbench.sh and qwencheck.sh.

## What
The scratchpad harness is at /private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/f0ac1848-c8ee-41e7-ae15-69d439fe644b/scratchpad.
- `inklingcli` is a Release CLI that depends on the repo by path. It loads mlx-community/Inkling-2-Mini-4bit at revision 9856dc53 through VLMModelFactory.
  - `bench`: prefill tok/s and decode ms/tok through TokenIterator.
  - `golden`: greedy ids.
  - `profile`: CPU graph build vs GPU eval of one decode step.
  - `dispatch`: the cost of one dependent kernel launch.
- `inklingbench.sh <out> [ref]`: rebuilds, runs ctx 2k/8k/32k/64k with 128 decode tokens, takes the median of 3, and prints the delta vs ref.
- `inklingcheck.sh <out.json>`: compares id by id against the read-only `golden_base.json`. The prompts are a short chat, a code chat (256 ids each), and raw 4k and 32k prompts (128 ids each).
- The prompt corpus is `ids.json`: MLXLMCommon and MLXLLM Swift sources, 1.34M ids.

## Baseline (stable 24bd6047, M5 Ultra), bench_base.txt
| ctx | prefill tok/s | decode ms/tok | tok/s |
|---|---|---|---|
| 2k | 4725 | 12.61 | 79.3 |
| 8k | 4613 | 12.64 | 79.1 |
| 32k | 3585 | 13.92 | 71.9 |
| 64k | 2732 | 15.73 | 63.6 |
- Noise: a second run gives ±1% on prefill and ±2–3% on decode.
- golden_r2: all 4 MATCH, so the harness is deterministic.
- Profile at 2k: CPU build 3.6 ms and GPU eval 10.8 ms per decode step. TokenIterator overlaps the two.
- One dependent dispatch costs ~5.3 us. At ~48 layer visits × ~15–20 kernels per visit, launch overhead is ~4 ms per token.

## Acceptance
- [x] The scripts run end to end; the baseline and the golden are recorded.
- [x] Two runs on the same build give identical ids. #inkling-perf