---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7k0tqvqw4fkvnc38wv5mb3v
  text: 'Research: the library already supports an opt-in 8-bit affine KV cache through `GenerateParameters(kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)` (legacy adapter -> `KVCachePlan` -> `maybeAffineQuantizeKVCache`). The plan rewrites only `.simple` leaves (`KVCacheSimple` -> `QuantizedKVCache`), so on Qwen3.8 only the 16 full-attention layers convert and the 48 GDN `MambaCache` layers keep their state. Prefill stays bf16; conversion happens on the first decode step. `TokenIterator`, `MTPSpeculativeTokenIterator` (re-applies the plan after each round) and `ChatSession` all honour the plan. The VLM Qwen35 attention routes a `QuantizedKVCacheProtocol` cache through `MLXLMCommon.attentionWithCacheUpdate` -> `quantizedCacheAttention` (no custom decode/verify kernels), and the compiled-decode gate accepts quantized attention caches. `QuantizedKVCache.trim` only moves `offset`, so MTP rejection rewinds work. The drafter cache stays bf16 with a quantized main cache (documented in using.md / model-compatibility.md). Existing unit coverage: `Qwen35MTPCacheKindTests` (MLXLLM text model, kvBits 8 gs32: MTP tokens == plain tokens). Plan: no library change expected; add a `QWEN_KV_BITS` env option to the scratch qwencli so every mode (bench, mtpbench, golden, session, toolchat) can run with KV8, plus an `agree` mode (teacher-forced top-1 agreement against the bf16 golden ids) and a `restore` mode (save/load prompt-cache snapshot round trip under KV8 + MTP).'
  timestamp: 2026-11-16T20:29:58.519876+00:00
- actor: claude-code
  id: 01m7k1ndct0wgmdp2k87br1cmm
  text: |-
    KV8 correctness (QWEN_KV_BITS=8 -> GenerateParameters(kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0); log $S/kv8check.log, $S/kv8restore.log):
    - cacheStatus after turn 2: 64 layers, 16 compressed (the full-attention layers), GDN untouched.
    - Teacher-forced top-1 agreement vs bf16 golden ids (qwen_golden_mn44v3n.json, 257 steps each): bf16 1028/1028 (harness calibration); KV8 short 255/257 (gaps 0.000, 0.125), code 257/257, 32k 255/257 (gaps 0.125, 0.125), toolchat 256/257 (gap 0.000). Total 1023/1028 = 99.5%; every miss is a near-tie within one bf16 logit ulp (<= 0.125).
    - Free-running greedy KV8 vs bf16 golden: code plain+mtp identical; toolchat mtp identical, plain diverges at step 201; short plain+mtp diverge at 114; 32k plain diverges at step 1 (near-tie), 32k mtp at 189. KV8 plain vs KV8 mtp also differ at the same near-tie steps, so with a quantized cache MTP is not bit-identical to plain decode (verify attends L=4 rows over the quantized cache through quantized matmuls; decode attends one row).
    - ChatSession multi-turn (plain + MTP): SESSION PASS, each follow-up reuses the cache and prefills only its message (cached 140/204 tok). Tool-call chat: PASS both modes.
    - Prompt-cache save/restore (new `restore` mode): turns 1-2, saveCache, loadPromptCacheSnapshot into a new session, turn 3: restored answer == live answer, prefilled 70 of 233 tokens, both plain and MTP, KV8 and bf16 alike. A raw restored cache reports cachedPromptTokenCount 0 in both modes (no transcript -> fragment continuation; not KV8-specific).
    - Harness bug found and fixed: the teacher-forcing processor is called once past maxTokens (TokenIterator computes one step ahead), so the recorder ignores that step.
  timestamp: 2026-11-16T20:44:32.538879+00:00
- actor: claude-code
  id: 01m7k2ng5r8x3843m37pf7063f
  text: |-
    ## KV8 result table (Qwen3.8-27B-4bit, M5 Ultra, median of 3, qwenbench.sh; KV8 = kvBits 8, group 64, the 16 full-attention layers, GDN state unchanged)

    | metric | bf16 KV (qwen_bench_mn44v3n.txt) | KV8 (qwen_bench_kv8.txt) | delta |
    |---|---|---|---|
    | plain decode ms/tok 2k | 17.14 | 17.59 | +2.6% |
    | plain decode ms/tok 32k | 19.92 | 22.36 | +12.2% |
    | plain decode ms/tok 64k | 22.82 | 26.90 | +17.9% |
    | MTP decode ms/tok 2k | 9.46 | 11.41 | +20.6% |
    | MTP decode ms/tok 32k | 12.50 | 32.25 | +158% |
    | MTP decode ms/tok 64k | 13.59 | 47.90 | +252% |
    | MTP tokens/verify 2k / 32k / 64k | 3.29 / 2.40 / 2.65 | 3.06 / 2.18 / 2.50 | -6 to -9% |
    | prefill tok/s 2k / 16k / 64k | 1470 / 1269 / 797 | 1468 / 1265 / 794 | same (prefill stays bf16) |
    | teacher-forced top-1 agreement vs bf16 golden (4 prompts x 257) | 1028/1028 | 1023/1028 (99.5%) | 5 misses, all near-ties (gap <= 0.125) |
    | free-running greedy vs bf16 golden, first divergence (plain / MTP) | identical | short 114/114, code none/none, 32k 1/189, toolchat 201/none | |
    | ChatSession multi-turn reuse, plain + MTP | PASS | PASS (follow-ups prefill only their message) | |
    | tool-call chat, plain + MTP | PASS | PASS | |
    | prompt-cache save -> load -> next turn, plain + MTP | PASS | PASS (restored answer == live answer, 70 of 233 tok prefilled) | |
    | MTP verify trim/rewind | ok | ok (MTP output stays on plain's tokens except at near-ties; bf16 also splits plain/MTP at near-ties: short 114, 32k 82) | |

    Why it is slower: Qwen3.8 attention has head dim 256, so a quantized cache never takes the dequantize+fused-SDPA route (`dequantizedPrefillAttentionHeadDims` excludes 256) nor the bf16 `DecodeAttentionKernel`/`VerifyAttentionKernel`; it takes `quantizedScaledDotProductAttention` (quantized matmul -> softmax -> quantized matmul, unfused). For one decode row that costs more than the bytes saved; for the L=4 verify rows it is far worse, which is why MTP loses 2.6x-3.5x at 32k-64k. Memory: the 64k KV drops from ~4.3 GB to ~2.3 GB, but bench peak memory is unchanged (prefill builds the bf16 cache first and quantizes at the first decode step).

    **Recommendation: NO-GO.** KV8 is slower at every context in both plain and MTP decode, lowers MTP acceptance, and changes greedy output at near-ties. Nothing ships; the default stays bf16. The library already exposes KV8 opt-in (`GenerateParameters(kvBits: 8)`) and it is correct with MTP verify, multi-turn reuse and prompt-cache restore, so no library change was made. The only path to the card's ~1.6 ms/step saving would be a fused 8-bit-KV decode and verify attention kernel for head dim 256; that is new work, not part of this card.

    Default path unchanged: qwencheck.sh with default settings -> $S/qwen_golden_cden2rc.json, all 8 records identical to qwen_golden_mn44v3n.json; vs qwen_golden_base.json only the two documented 32k near-tie divergences (plain step 82, MTP step 212), every other record MATCH; SESSION PASS; TOOLCHAT PASS x2.
  timestamp: 2026-11-16T21:02:03.960988+00:00
- actor: claude-code
  id: 01m7k2p4trzbnxrepq3fwj8ng5
  text: |-
    ### implement — no-change
    - evidence: no library or test file changed (git status: only .kanban). The existing opt-in `GenerateParameters(kvBits: 8)` already works with MTP verify, ChatSession reuse and prompt-cache restore. Harness (scratchpad only): qwencli/Sources/qwencli/main.swift gains QWEN_KV_BITS (all modes), `agree` (teacher-forced top-1) and `restore` (save/load snapshot) modes, line-buffered stdout; new kv8check.sh, goldcmp.py. Measurements: qwen_bench_kv8.txt(.raw), qwenbench_kv8.log, kv8check.log, kv8restore.log, kv8_golden.json; default qwencheck -> qwencheck_cden2rc.log / qwen_golden_cden2rc.json (8/8 identical to qwen_golden_mn44v3n.json, SESSION PASS, TOOLCHAT PASS). Correction to the earlier comment: bf16 MTP also splits from plain at near-ties (short 114, 32k 82), so plain/MTP near-tie splits are not KV8-specific.
    - next: user decides on the NO-GO; follow-up candidate ^6xcb05k (fused 8-bit-KV attention kernel) needs sign-off.
  timestamp: 2026-11-16T21:02:25.112828+00:00
- actor: claude-code
  id: 01m7k2qc3tme0agdna04qefe3f
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (4015abdd) — 0 findings (0 confirmed, 0 refuted); 6 .kanban files changed, no validator matches them, no source file in the diff. Acceptance: both items were already checked, and the evidence supports them: the KV8 table and the NO-GO are in the comments, and the default qwencheck run gives 8/8 records identical to qwen_golden_mn44v3n.json with no library change.
    - next: done; the follow-up ^6xcb05k needs the user's sign-off.
  timestamp: 2026-11-16T21:03:05.338205+00:00
- actor: claude-code
  id: 01m7k2qnt64rs5ytpzq0rgyhe4
  text: |-
    ### finish iteration 1 — clean
    - implement: no-change — measured NO-GO (KV8 slower: plain +12/+18%, MTP 2.6–3.5x at 32k/64k)
    - test: n/a — no library change
    - commit: 4015abdd (kanban evidence)
    - review: clean → done
  timestamp: 2026-11-16T21:03:15.270935+00:00
depends_on:
- 01M7J84C1V3YZ5VGSXGMN44V3N
position_column: done
position_ordinal: ffffbb80
title: 'Qwen3.8: measure an 8-bit KV cache for the full-attention layers (opt-in go/no-go)'
---
## Why
At 64k context the KV cache adds 4.3 GB of reads per token. At 8-bit that would save about 1.6 ms per decode step, and more per verify. It changes the output, so it can never become the default without the user's sign-off.

## What
- Measure decode and MTP tok/s with an 8-bit quantized KV cache (group size 64) on the 16 full-attention layers only, at 32k and 64k.
- Measure top-1 agreement against the bf16 KV cache on the golden prompts.
- Check that a quantized cache trims, rewinds and restores correctly under MTP verify and multi-turn prompt-cache reuse.

## Acceptance
- [x] A table of speed and agreement on this card, with a GO/NO-GO recommendation.
- [x] Nothing changes by default. If it ships at all, it ships opt-in, and `qwencheck.sh` stays identical with the default settings. #qwen-perf

## Result: NO-GO
KV8 (kvBits 8, group 64, existing opt-in `GenerateParameters(kvBits: 8)`) is correct with MTP verify, multi-turn reuse and prompt-cache restore, but it is slower everywhere: plain 19.92 -> 22.36 ms/tok at 32k and 22.82 -> 26.90 at 64k; MTP 12.50 -> 32.25 at 32k and 13.59 -> 47.90 at 64k. Top-1 agreement vs bf16 is 1023/1028 (99.5%, near-ties only). Cause: head dim 256 sends a quantized cache through the unfused quantized-matmul attention route instead of the bf16 decode/verify kernels. No library change; the default stays bf16. Full table in the comments.