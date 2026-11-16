---
assignees:
- claude-code
depends_on:
- 01M7J8449YT9F98KXC0117YMN8
position_column: todo
position_ordinal: '8980'
title: 'Qwen3.8 MTP round: compile the drafter step and remove CPU stalls'
---
## Why
The seed-gate `.item()` reads and the verify readback leave the GPU idle while the next graph builds. Comparing pMin 0.5 against pMin 0 puts this at about 2 ms per round at 2k. Flint commit 864b34d9 compiled the one-token decode trace (CompiledTrace per block).

## What
- Compile the drafter step (fc + 1 layer + lm_head) into one trace.
- Schedule the round so the gate decision and the verify readback overlap the next graph's build (asyncEval ordering) without changing which tokens are accepted.

Files: `QwenMTP.swift`, `Qwen35MTP.swift`, `MTPSpeculativeTokenIterator.swift`.

## Acceptance
- [ ] `qwencheck.sh`: MTP greedy ids identical to the golden file, and the multi-turn cache-reuse check passes. ChatSession and FoundationModels drafter-state restore are unaffected.
- [ ] All five bundles are green. `qwenbench.sh` delta on this card. #qwen-perf