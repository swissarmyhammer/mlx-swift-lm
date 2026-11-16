---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jbkkdgph4h8mm2sk28de1h
  text: |-
    Research + design (scratchpad only, no library change; Libraries/ identical to 87460a3f):
    - qwencli/Package.swift already depends on the repo working tree by path (`.package(path: /Users/wballard/github/swissarmyhammer/mlx-swift-lm)`), own `.build`, Release product at qwencli/.build/out/Products/Release/qwencli.
    - Profiler prompt = `qids.txt` (248k ids, concatenated repo Swift sources, starts with ArraysCache). Bench uses it, as q_mtp/q_sweep did.
    - dfl4 "32k" agent prompts are really 28284/28818/29127 tokens; golden uses `agent_32000_2.txt` (29127 tok, closest to 32k). short = dfl4/short_prose.txt (64 tok), code = dfl4/short_code.txt (73 tok, Swift LCS request), toolchat = weather question + get_weather tool rendered by the chat template in qwencli.
    - New qwencli modes: `golden` (plain via TokenIterator(parameters:) temp 0 = production path, MTP via MTPSpeculativeTokenIterator bs4/pMin0.75), `mtpbench` (MTP only, no plain reference; shares runMTP with `mtp`), `session` (ChatSession, system prompt + 3 turns, thinking off, plain and MTP; follow-up must have cached > 0 and prefilled <= message tokens + 16 template tokens), `toolchat` (ChatSession with get_weather, plain and MTP, must call get_weather(Paris) once and answer with 18).
    - No top-2 logit gaps in the golden (flintcheck had them): taking them would need a custom sampler or logProbabilities, which changes the measured path (MTP checks `sampler is ArgMaxSampler`). The diff report gives the first divergent step and both ids.
    - Gotcha: zsh runs an EXIT trap set inside a function when the function returns, so the bench.lock release trap is set at script top level (`eval $QWEN_UNLOCK_TRAPS`).
    - Files: qwenenv.zsh (shared paths/build/lock), qwenbench.sh, qwencheck.sh, qwencli/Sources/qwencli/main.swift (backup of the profiler version: qwencli_main.swift.orig_8khnvfw).
  timestamp: 2026-11-16T14:19:04.496711+00:00
- actor: claude-code
  id: 01m7jcpmcn3tkzbza8mythev11
  text: |-
    ## Stable baseline (87460a3f library code, qwen-perf HEAD 6061fdd5 = 87460a3f + kanban), M5, Qwen3.8-27B-4bit + Qwen3.8-27B-MTP-4bit, MLXVLM load path, Release

    ### qwenbench.sh, median of 3 runs on qids.txt (profiler Swift-code prompt), 256 decode tokens. File: qwen_bench_base.txt (read-only)
    | metric | 2k | 16k | 32k | 64k |
    |---|---|---|---|---|
    | plain decode ms/tok | 18.26 | (19.66) | 20.60 | 23.44 |
    | MTP bs4 pMin0.75 ms/tok | 10.17 | – | 13.02 | 14.20 |
    | MTP tokens per verify | 3.29 | – | 2.40 | 2.65 |
    | prefill tok/s | 1334 | 1152 | (963) | 722 |
    The run-to-run spread is small: plain decode ±0.05 ms, MTP ±0.1 ms, prefill ±20 tok/s at 2k and ±1 tok/s elsewhere. These match the profiler (q_mtp/q_sweep).

    ### qwencheck.sh. Golden: qwen_golden_base.json (read-only; the script refuses to overwrite it, which was tested: exit 1)
    - The golden has 4 prompts × {plain, MTP} with 257 greedy ids each. The prompts: short (short_prose, 64 tok), code (short_code, 73 tok), 32k (agent_32000_2, 29127 tok), toolchat (345 tok).
    - A second run (qwen_golden_r2.json) gives MATCH on all 8: every 257 ids are identical. The harness is deterministic.
    - ChatSession, system prompt + 3 turns, thinking off, PASS in plain and MTP. Turn 1 cached 0 / prefilled 102. Turn 2 cached 139 / prefilled 32 (message 19 + 13 template). Turn 3 cached 203 / prefilled 29 (message 16 + 13). The answers are identical in plain and MTP. MTP accepted 22/28, 23/23 and 21/24 drafts.
    - Tool-call smoke PASS in plain and MTP: exactly one get_weather(city: Paris) call, then the answer states 18°C and light rain.
    - The run ends with CHECK PASS, exit 0.

    Logs: qwencheck_base.log, qwencheck_r2.log, qwenbench_base.log, qwen_bench_base.txt.raw.
    Use: `qwenbench.sh [out] [ref]` (default out qwen_bench_new.txt, ref qwen_bench_base.txt; prints a delta % with better/worse) and `qwencheck.sh [out.json]` (compares against qwen_golden_base.json by mode). Each script takes bench.lock (mkdir; it retries every 30 s) and releases it on exit.
  timestamp: 2026-11-16T14:38:12.373588+00:00
- actor: claude-code
  id: 01m7jcpyw3da905yw10mg8jj3j
  text: |-
    ### implement — changed
    - evidence: 4 scratchpad files (no repo change): qwencli/Sources/qwencli/main.swift (new modes golden/mtpbench/session/toolchat, shared runPlain/runMTP), qwenenv.zsh, qwenbench.sh, qwencheck.sh. Runs: qwencheck.sh qwen_golden_base.json → CHECK PASS; qwencheck.sh qwen_golden_r2.json → 8/8 MATCH, CHECK PASS; qwenbench.sh qwen_bench_base.txt → exit 0, baseline on card. The goldens are read-only. bench.lock was released after each run.
    - next: review. The next card (^8h1td1f) runs `qwenbench.sh` and `qwencheck.sh` with their defaults after its change.
  timestamp: 2026-11-16T14:38:23.107281+00:00
position_column: doing
position_ordinal: '80'
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
- [x] Both scripts run end to end; the baseline numbers and the golden file are recorded.
- [x] The scripts are deterministic: two runs on the same build give identical ids. #qwen-perf