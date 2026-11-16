---
assignees:
- claude-code
position_column: todo
position_ordinal: '8380'
title: 'Qwen3.8 perf harness: benchmark script + golden greedy ids + cache-reuse check (baseline on stable)'
---
## Why
Every qwen-perf card must report its speedup against the previous card and prove the output is unchanged, as the flint-perf batch did with flintbench.sh and flintcheck.sh.

## What
Build these in the scratchpad (/private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/49a08566-0602-419c-8d53-fb325aafef7c/scratchpad). Extend the existing `qwencli` harness, which loads through the production MLXVLM path with mlx-community/Qwen3.8-27B-4bit plus Qwen3.8-27B-MTP-4bit, built in Release.

1. **`qwenbench.sh`** takes the median of 3 runs and reports:
   - plain decode ms/tok at 2k, 32k and 64k;
   - MTP (blockSize 4, pMin 0.75) ms/tok and tokens per verify at 2k, 32k and 64k;
   - prefill tok/s at 2k, 16k and 64k.

   Use the same real Swift-code prompts as the profiler.
2. **`qwencheck.sh`:**
   - Writes the greedy token ids for 4 prompts in both plain and MTP mode to `qwen_golden_base.json`. **Never overwrite that file.** The 4 prompts are short, code, a 32k prompt and a tool-call chat.
   - Compares later runs exactly, id by id, against the baseline of the same mode.
   - Runs a multi-turn ChatSession check: a system prompt and then 3 turns. Follow-ups must have cachedPromptTokenCount > 0 and prefill only the new message.
   - Runs a tool-call chat smoke test.
3. Record the stable 87460a3f baseline from both scripts on this card.

## Rules
- Take bench.lock in the scratchpad before any real-model run and release it after.
- Coordinate with ACP/Router.

## Acceptance
- [ ] Both scripts run end to end; the baseline numbers and the golden file are recorded.
- [ ] The scripts are deterministic: two runs on the same build give identical ids. #qwen-perf