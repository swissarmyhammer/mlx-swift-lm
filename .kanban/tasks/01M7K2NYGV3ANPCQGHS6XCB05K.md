---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7mmnnj8c1g6vd6ea9hpq8tj
  text: |-
    ### sign-off — user approved (2026-11-17)
    - The user said "go ahead and code this" after being told: KV8 is opt-in and output-changing vs bf16 (99.5% top-1 agreement, near-ties only); the estimated gain is ~7% plain / ~4% MTP at 64k (the 1.6 ms/pass is a bandwidth-floor estimate, realistically ~1 ms), ~half at 32k, ~0 at 2k; KV memory halves. The first "What" bullet (decide with the user) is resolved: build it.
  timestamp: 2026-11-17T11:35:58.280715+00:00
- actor: claude-code
  id: 01m7mmxrdqddh4b3qx29vz4b01
  text: 'Research (implement pickup): both Qwen35 twins (MLXVLM `Attention.attend`, MLXLLM `Qwen35Attention.attend`) reach attention only through `PrefillAttentionKernel.attentionWithCacheUpdate` (MLXLMCommon), which today hands every `QuantizedKVCacheProtocol` cache to `MLXLMCommon.attentionWithCacheUpdate` -> `quantizedCacheAttention` -> `quantizedScaledDotProductAttention` (head dim 256 is outside `dequantizedPrefillAttentionHeadDims`). So one routing point in MLXLMCommon covers LLM+VLM; no model file changes. Quantized caches never take the compiled attention segment (VLM gate / LLM `usesPlainAttentionCacheRoute`), so the attention runs eagerly in `fullAttention` for decode (mask .none) and verify (mask .causal from BaseKVCache.makeMask). QuantizedKVCache layout: packed uint32 `(B, kvH, cap, 64)` (8-bit codes, element j in bits 8j), scales/biases `(B, kvH, cap, 4)` in the key dtype, returned as `[..., ..<offset, :]` slices. Plan: (1) decode: generalize `GroupSoftmax::add_stream` over its row reader and add a `QuantizedLaneRows` reader (lane = 8 numbers = 2 words + one group scale/bias) — the block/combine passes are otherwise shared; (2) verify: keep the PrefillAttentionKernel threadgroup layout/loop and give the loader SIMD groups a `load_tiles` overload that reads `QuantizedTileRows` (uint4 = 16 codes) and dequantizes into the same bf16 threadgroup tiles; (3) new `QuantizedKVAttentionKernel` (MLXLMCommon) with canRun (affine, bits 8, group 64, head 256, >= 1024 keys; decode 1 row/no mask, verify 2-8 rows causal/none) and a route that falls back to `quantizedCacheAttention`; `PrefillAttentionKernel.attentionWithCacheUpdate` routes quantized caches through it. The bf16 kernel sources are only split into named parts that concatenate to the same text.'
  timestamp: 2026-11-17T11:40:23.351948+00:00
- actor: claude-code
  id: 01m7mntmcg6h6dtyst4m6ctnhv
  text: |-
    Implementation landed (uncommitted): new `Libraries/MLXLMCommon/QuantizedKVAttentionKernel.swift` (decode kernel = DecodeAttentionKernel block pass with a `QuantizedLaneRows` reader; verify kernel = VerifyAttentionKernel block pass whose loader SIMD groups dequantize packed uint4 tiles into the bf16 threadgroup tiles via a `load_tiles` overload; shared combine pass). `PrefillAttentionKernel.attentionWithCacheUpdate` (the single Qwen35 LLM+VLM attention route) now sends `QuantizedKVCacheProtocol` caches to `QuantizedKVAttentionKernel.scaledDotProductAttention`, which falls back to `quantizedCacheAttention` for anything but affine/8-bit/group 64/head 256/>=1024 keys/(decode no mask | verify 2-8 rows causal or none). bf16 kernel sources were only split into named parts (same concatenated text); `add_stream` is generic over its row reader.

    Correctness:
    - Unit tests (new QuantizedKVAttentionKernelTests, 7 tests / 41 cases): kernel vs SDPA over dequantized K/V in float32 within the existing per-dtype tolerances (decode 4 shapes, verify 5 shapes x causal/none x f32/f16/bf16); kernel vs `quantizedScaledDotProductAttention` (bf16, decode + L=4 verify) within 2x bf16 tolerance; measured: the unfused path is 0.0099/0.0067 from the exact dequantized reference, the kernel 0.0034/0.0030 — the kernel is closer, and a test pins that. Route gives the kernel output bit for bit through a real QuantizedKVCache, and the quantized path bit for bit for 4-bit, group 32, mask array, short cache, prefill block.
    - Full MLXLMTests: 2379 Swift Testing + 1134 XCTest, 0 failures (3 known issues and 4 env-gated LFM2 skips are pre-existing).
    - Default bf16 path: qwencheck.sh -> qwen_golden_6xcb05k.json, 8/8 records identical to qwen_golden_mn44v3n.json, SESSION PASS, TOOLCHAT PASS x2.
    - KV8 real model (kv8check_6xcb05k.log): teacher-forced top-1 vs bf16 golden 1023/1028 (same as the old KV8 path; all misses near-ties gap <= 0.125); vs the OLD KV8 path's greedy ids (kv8_golden.json) 1024/1028 = 99.6% (4 misses, all in 32k, gaps 0.000/0.125). Free-running greedy vs old KV8: short/code/toolchat plain+mtp all 257 identical; 32k plain diverges at step 1 (near-tie, gap 0.000), 32k mtp at 189. Output coherent. ChatSession multi-turn KV8 SESSION PASS plain+MTP (follow-ups prefill only their message, cached 140/204 tok); TOOLCHAT PASS x2; prompt-cache save/restore KV8 RESTORE PASS (16 compressed layers, 70 of 233 tok prefilled).
  timestamp: 2026-11-17T11:56:09.488558+00:00
- actor: claude-code
  id: 01m7mqhehkxe3apga5m2akdhcv
  text: |-
    ## KV8 bench with the new kernels (Qwen3.8-27B-4bit, M5 Ultra, qwenbench.sh median of 3, same day)

    | metric | bf16 KV today (qwen_bench_bf16_6xcb05k.txt) | old KV8 (qwen_bench_kv8.txt) | new KV8 (qwen_bench_kv8_6xcb05k.txt) | new KV8 vs bf16 | new vs old KV8 |
    |---|---|---|---|---|---|
    | plain ms/tok 2k | 17.14 | 17.59 | 17.52 | +2.2% | -0.4% |
    | plain ms/tok 32k | 19.86 | 22.36 | 20.12 | +1.3% | -10.0% |
    | plain ms/tok 64k | 22.84 | 26.90 | 22.61 | -1.0% | -15.9% |
    | MTP ms/tok 2k | 9.51 | 11.41 | 10.17 | +6.9% | -10.9% |
    | MTP ms/tok 32k | 12.52 | 32.25 | 13.23 | +5.7% | -59.0% |
    | MTP ms/tok 64k | 13.61 | 47.90 | 14.43 | +6.0% | -69.9% |
    | MTP tok/verify 2k/32k/64k | 3.29/2.40/2.65 | 3.06/2.18/2.50 | 3.21/2.42/2.68 | ~same | +5..11% |
    | prefill tok/s 2k/16k/64k | 1472/1275/803 | 1468/1265/794 | 1479/1274/805 | same | same |

    bf16 today vs qwen_bench_mn44v3n.txt: every row within +-0.8% (no bf16 regression; stable-vs-qwen-perf reference MTP 9.48/12.47/13.55, plain 17.13/19.84/22.77).

    **Recommendation: measured NO-GO for KV8 as a speedup.** The fused kernels remove the unfused-path penalty (MTP 2.4-3.3x faster than the old KV8, plain 10-16% faster at 32k/64k), but KV8 is still not faster than bf16 KV at 32k (plain +1.3%, MTP +5.7%) or MTP at 64k (+6.0%); only plain 64k wins (-1.0%). Why: the attention slope from 2k to 64k drops only ~0.6 ms per plain step (bf16 +5.70 ms, KV8 +5.09 ms), well below the ~1.6 ms bandwidth estimate, and KV8 carries a context-independent cost of ~0.4-0.7 ms per step (visible at 2k: plain +0.38 ms, MTP +0.66 ms/tok) from `QuantizedKVCache.updateQuantized` (a `quantized()` of the new rows plus 6 slice writes and 6 slices per layer per step, x16 layers). The kernels are worth keeping only if opt-in KV8 (half the KV memory) should not cost 2.6-3.5x in MTP; the user decides whether the change ships. A follow-up candidate for the fixed update cost is filed separately.
  timestamp: 2026-11-17T12:26:05.747587+00:00
- actor: claude-code
  id: 01m7mqj004ffmk2r4q6venz9v9
  text: |-
    ### implement — changed
    - evidence: 7 files — new Libraries/MLXLMCommon/QuantizedKVAttentionKernel.swift, Tests/MLXLMTests/QuantizedKVAttentionKernelTests.swift; edited Libraries/MLXLMCommon/{DecodeAttentionKernel,VerifyAttentionKernel,PrefillAttentionKernel}.swift, doc comments in Libraries/MLXLLM/Models/Qwen35.swift and Libraries/MLXVLM/Models/Qwen35.swift. swift build --build-tests: 0 warnings. Tests: MLXLMTests 2379 ST + 1134 XCTest 0 failures; MLXGuidedGenerationTests 78, MLXFoundationModelsTests 496, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 — all pass. qwencheck 8/8 identical to mn44v3n; KV8 agree 99.6% vs old KV8 path, session/toolchat/restore PASS. Bench: KV8 plain 17.52/20.12/22.61, MTP 10.17/13.23/14.43 ms/tok vs bf16 17.14/19.86/22.84, 9.51/12.52/13.61 -> measured NO-GO (KV8 2.4-3.3x faster MTP than the old KV8 path). swift-format run. Not committed.
    - next: /review; the user decides whether the KV8 kernels ship given the NO-GO vs bf16; follow-up candidate ^2h5wtct (fixed per-step updateQuantized cost).
  timestamp: 2026-11-17T12:26:23.620568+00:00
- actor: claude-code
  id: 01m7mqkzrc161hd0v8abzyp350
  text: |-
    ### finish iteration 1 — changed (paused before review)
    - implement: changed — 7 files; measured NO-GO vs bf16 KV (KV8 MTP +5.7/+6.0% at 32k/64k), 2.4–3.3x faster than the old KV8 path
    - test: green — run inside the implement step: all 5 bundles, 0 failures, 0 warnings (4 pre-existing LFM2 skips, ^gcsvy3f)
    - commit: e6eb2737 (qwen-perf, local)
    - review: not yet — paused so ^4zv2rdx (Router chain) goes first; resume with /review 6xcb05k e6eb2737~1..e6eb2737
  timestamp: 2026-11-17T12:27:28.908542+00:00
depends_on:
- 01M7J84FZ3SJF8MB599CDEN2RC
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8: fused 8-bit-KV decode and verify attention kernel for head dim 256 (candidate, needs sign-off)'
---
## Why
^cden2rc measured the existing opt-in KV8 (`GenerateParameters(kvBits: 8)`) as NO-GO: with head dim 256 a `QuantizedKVCache` goes through `quantizedScaledDotProductAttention` (quantized matmul -> softmax -> quantized matmul, unfused) instead of the bf16 `DecodeAttentionKernel` / `VerifyAttentionKernel`. Plain decode lost 12-18% at 32k-64k and MTP lost 2.6x-3.5x, although the KV reads drop from ~4.3 GB to ~2.3 GB at 64k. The card's estimated saving (~1.6 ms per step at 64k) needs a kernel that reads 8-bit keys and values directly.

## What
- Decide with the user whether an output-changing opt-in is worth a custom kernel (top-1 agreement 99.5% vs bf16; near-ties only).
- If yes: extend the decode and verify attention kernels to read affine 8-bit K/V (group 64) with inline dequantization, and route `QuantizedKVCache` there for head dim 256.

## Acceptance
- [x] KV8 plain and MTP decode at 32k and 64k faster than bf16 KV (qwenbench.sh with QWEN_KV_BITS=8), or a recorded NO-GO. (Recorded NO-GO: KV8 is +1.3%/-1.0% plain and +5.7%/+6.0% MTP vs bf16 at 32k/64k; see the bench comment.)
- [x] Default path unchanged (`qwencheck.sh` identical). #qwen-perf-candidate