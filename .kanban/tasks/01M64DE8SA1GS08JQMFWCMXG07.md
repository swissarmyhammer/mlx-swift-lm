---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66d07ykfcmcy70a0sjv5xg1
  text: |-
    Research + profile (before), Release, M5 Ultra, uptime load ~9.7 (other work on the box). New profiling test MTPVerifyForwardProfileTests (text path) times every layer of a kind with its own weights/cache, inside the speculative-verify scope.

    Routes found:
    - Text path verify: Qwen35Attention.attend -> PrefillAttentionKernel route (needs >=64 queries) -> DecodeAttentionKernel route (needs q_len 1 and no mask) -> MLX fast SDPA with `.causal` (symbolic, no materialized mask). MLX picks sdpa_vector_2pass for q_len<=8 when q_len*GQA<=32 (Qwen3.8 GQA 6: q_len<=5 vector, q_len 6+ falls back to unfused SDPA). The 2-pass kernel runs one SIMD group per (head,row) and per-lane dot+simd_sum per key, so its cost grows linearly with rows x context.
    - Router path (MLXVLM Qwen35): attentionWithCacheUpdate with an ARRAY causal mask (createAttentionMask(returnArray: true)), never the decode/prefill kernels.

    Profile ms (verify / GDN [proj, recurrence] / attn proj / attention / feed-forward / rest):
    32k r1 21.84 / 4.32 [3.43, 0.89] / 1.48 / 2.49 / 9.99 / 3.58
    32k r2 26.79 / 6.81 / 1.60 / 5.09 / 12.62 / 0.67
    32k r3 35.10 / 8.63 / 2.21 / 6.87 / 15.18 / 2.21
    32k r4 41.91 / 9.58 / 2.43 / 8.92 / 17.38 / 3.61
    32k r5 49.86 / 12.96 [7.59, 5.37] / 2.86 / 9.87 / 20.55 / 3.62
    100k r1 27.15 / 5.26 [3.44, 1.82] / 1.45 / 7.16 / 9.94 / 3.34
    100k r2 36.24 / 6.79 / 1.65 / 14.26 / 13.44 / 0.10
    100k r3 49.04 / 8.44 / 2.16 / 19.91 / 15.31 / 3.23
    100k r4 57.72 / 10.78 / 2.47 / 25.51 / 17.82 / 1.15
    100k r5 71.02 / 10.91 [7.41, 3.50] / 2.92 / 31.76 / 20.87 / 4.55
    MLX SDPA with an explicit array causal mask (router-style microbench): 32k 2.5/34.6/36.9/39.7/43.7, 100k 7.8/107/110/117/122 ms for rows 1-5.

    Conclusion: at 100k the dominant per-row growth is the full-attention SDPA (+6.2 ms/row, 24.6 of the 43.9 ms from 1->5 rows); next is the feed-forward multi-row matvec (+2.7 ms/row at both contexts, context-independent), then GDN (+1.4-2.2 ms/row). Plan: a split-K MMA kernel for 2..8 query rows over a long cache (reads each KV tile once for all rows and 3 heads), reusing the prefill kernel's tile/MMA body and the decode kernel's combine pass.
  timestamp: 2026-10-30T12:36:52.307499+00:00
- actor: claude-code
  id: 01m66k8yknsydw0978mktxhyez
  text: |-
    ### After: verify attention kernel (Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit)
    The change is VerifyAttentionKernel, a split-K MMA attention for 3 to 8 query tokens over 1024 or more keys, with a head width of 256 and a causal mask or no mask. It uses 512 keys per block and 4 to 64 blocks, and 6 heads per threadgroup when the GQA group is a multiple of 6. The route is Prefill, then Verify, then Decode, then MLX SDPA. Two-row verifies stay on MLX SDPA, which is faster for 2 rows. The kernel is not bit-exact, but the unit tests check it against a tolerance, and the model-level test gives plain decode logits within bf16, with the greedy token equal up to a near tie.

    **Round-cost breakdown (ms, guard off; verify rows 1–5).** Before: 07:36–07:44, load 3.5–31. After: 08:26–08:36, load 4.4 → 2.3.
    | path, ctx | plain before → after | verify before | verify after | per extra row before → after |
    |---|---|---|---|---|
    | text 32k | 21.08 → 20.98 | 21.23/26.17/35.44/41.72/48.97 | 20.88/25.66/33.65/38.27/43.78 | 6.94 → 5.73 |
    | text 100k | 26.28 → 27.23 | 26.19/34.92/46.98/57.00/68.62 | 27.33/37.22/46.19/51.42/57.03 | 10.61 → 7.43 |
    | router 32k | 22.90 → 21.90 | 23.69/30.70/41.13/48.62/56.55 | 22.42/29.37/39.41/45.88/53.76 | 8.22 → 7.84 |
    | router 100k | 28.20 → 27.38 | 28.28/45.02/58.74/67.71/79.85 | 27.67/43.20/56.41/67.16/79.49 | 12.89 → 12.96 |
    The router path does not change because it passes the mask as an array (see the new task).

    **Layer-kind profile, text path, after (ms, rows 1–5)**
    - 32k: verify 20.94/25.39/33.73/38.51/43.93; attention 2.40/5.08/6.75/6.64/6.66 (before 2.49/5.09/6.87/8.92/9.87); FF 9.77/12.81/14.55/17.48/20.08; GDN 4.13/6.01/6.95/7.85/8.67; mask-array attention 2.40/34.27/35.55/36.63/38.83
    - 100k: verify 29.21/37.48/49.88/54.30/61.15 (before 27.15/36.24/49.04/57.72/71.02); attention 8.07/15.53/18.76/18.45/18.00 (before 7.16/14.26/19.91/25.51/31.76); FF 10.59..20.36; GDN 5.48/6.81/12.05/13.66/14.04; mask-array attention 8.17/108.01/109.27/120.89/126.69
    - Attention microbench (16 layers, route vs MLX SDPA): 32k [3.71, 7.45, 7.25, 7.41, 7.85] vs [4.48, 7.91, 9.99, 14.31, 18.75]; 100k [13.52, 15.09, 18.67, 18.96, 18.49] vs [10.01, 15.13, 23.87, 27.02, 30.70]. This run was noisier than the earlier tuning run, which measured 32k [2.45, 6.60, 6.54, 6.37, 6.49] vs [3.06, 4.82, 6.82, 7.97, 9.84] and 100k [7.08, 17.62, 17.93, 17.83, 17.97] vs [8.50, 13.52, 19.22, 25.45, 28.69].

    **Sweep (tok/s, MTP/plain ratio).** "Before" is ^j2tnzpn phase 2 (head fe3b25d8, load 8.6 → 5.4). Before the change this card's own sweep never ran: the earlier identifier without `(path:)` matched no test. After: 08:37–09:11, load about 3–5.
    | path, ctx | plain before/after | guard off before → after | guard on before → after |
    |---|---|---|---|
    | text 2k | 55.2 / 56.3 | 88.0 (1.59) → 83.2 (1.48) | 84.8 (1.53) → 90.4 (1.61) |
    | text 8k | 54.4 / 53.9 | 95.3 (1.75) → 87.0 (1.61) | 82.8 (1.52) → 88.9 (1.65) |
    | text 32k | 47.8 / 48.3 | 74.6 (1.56) → 88.1 (1.83) | 60.6 (1.27) → 88.2 (1.83) |
    | text 64k | 38.3 / 38.3 | 52.4 (1.37) → 62.2 (1.63) | 43.5 (1.14) → 59.0 (1.54) |
    | text 100k | 32.3 / 32.9 | 43.7 (1.35) → 46.9 (1.43) | 33.9 (1.05) → 39.8 (1.21) |
    | router 2k | 55.3 / 55.5 | 76.3 (1.38) → 81.5 (1.47) | 65.3 (1.18) → 81.2 (1.46) |
    | router 8k | 53.5 / 53.6 | 76.3 (1.43) → 88.2 (1.64) | 60.2 (1.13) → 82.1 (1.53) |
    | router 32k | 46.8 / 46.8 | 55.6 (1.19) → 55.9 (1.19) | 48.1 (1.03) → 53.8 (1.15) |
    | router 64k | 37.3 / 36.3 | 48.6 (1.30) → 48.6 (1.34) | 37.8 (1.01) → 38.3 (1.05) |
    | router 100k | 30.3 / 31.9 | 37.9 (1.25) → 38.5 (1.21) | 29.8 (0.99) → 30.6 (0.96) |
    Guard-off ms per round, text before → after: 33.2/39.4/45.4/57.5/74.7 → 30.4/41.2/47.0/56.2/64.9. Acceptance and tokens per verify move between runs, because the kernel is not bit-exact and the generated text changes. For example, text 2k gave 2.53 tokens per verify after vs 2.92 before, and text 32k gave 4.15 vs 3.39. So the 2k/8k guard-off rates are lower from acceptance, not from cost per round. The router path does not use the kernel, so its changes are noise and acceptance.
    The test recorded 1 known issue: router 100k guard on was 0.960 of plain, below the 0.97 floor. This is the router array-mask path, which this change does not touch. The guard behavior is ^tdj3waa.

    **Guards (09:12–09:25, load 2.7 → 3.2):** `qwen38FourBitMTPGeneration()` passed (bs5 pMin0.75 mean 90.2 tok/s; math 100.6 / code 87.2 / prose 82.7 vs plain ~59.5). `qwen38FourBitMTPSystemPromptCache(adaptivePolicy:)` passed in both cases. Edit turn 2 prefilled 123 tokens, the tokens matched plain exactly, and it ran at 1.76x.

    **Target not met.** The target was a per-extra-row cost at 100k "well under 5 ms". On the text path it is now 7.4 ms, down from 10.6. The per-row cost left is in the feed-forward and GDN projection multi-row matvec, about 2.6 ms per row at every context, and in the router array-mask path. Follow-up tasks are filed.
  timestamp: 2026-10-30T14:26:29.109022+00:00
- actor: claude-code
  id: 01m66kae5kqv1mtnc3tm1w4xyp
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLMCommon/VerifyAttentionKernel.swift (new), Libraries/MLXLMCommon/PrefillAttentionKernel.swift, Libraries/MLXLMCommon/DecodeAttentionKernel.swift, Tests/MLXLMTests/VerifyAttentionKernelTests.swift (new), Tests/MLXLMTests/Qwen35VerifyAttentionTests.swift (new), Tests/MLXLMTests/Qwen35AttentionLayerFixture.swift, Tests/MLXLMTests/Qwen35DecodeAttentionTests.swift, IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPVerifyForwardProfileTests.swift (new), IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Support/GPUWallTime.swift (new), IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPRoundCostBreakdownTests.swift. swift build --build-tests had 0 warnings from source; the only line is llbuild's "missing creator for mutated node" on mlx-swift_Cmlx.bundle. Release build-for-testing: TEST BUILD SUCCEEDED. All 5 bundles have 0 failures and 0 skipped: CXGrammarTests 7 tests; MLXFoundationModelsTests 457; MLXGuidedGenerationTests 72; MLXHuggingFaceMacrosTests 7 XCTest; MLXLMTests 920 XCTest plus 2080 Swift Testing. Measurements: after-profile (MTPVerifyForwardProfileTests + MTPRoundCostBreakdownTests), after-sweep (decodeRateFromTwoToOneHundredThousandContextTokens(path:), 1 known issue: router 100k guard on 0.96x), after-guards (qwen38FourBitMTPGeneration and qwen38FourBitMTPSystemPromptCache passed). The 100k per-row target was not met (7.4 ms). Follow-ups: ^vre4msr (FF/GDN matvec), ^8p0rfxx (router array mask), ^mm16fx2 (leaner verify kernel).
    - next: /review
  timestamp: 2026-10-30T14:27:17.811493+00:00
position_column: doing
position_ordinal: '8380'
title: 'MTP: cut the per-row cost of the multi-row verify at long context (5-row verify is 2.3–3.0x a plain forward at 32k–100k)'
---
Step-4 breakdown (^j2tnzpn, Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, guard off): a 1-row verify costs about a plain forward, but each extra row adds ~5–7 ms at 32k and ~10–13 ms at 100k context:
- text 32k: plain 21.44 ms; verify rows 1–5: 21.85 / 26.59 / 35.33 / 42.48 / 48.95
- text 100k: plain 26.42; verify 27.43 / 36.31 / 48.60 / 58.32 / 68.92
- router 100k: plain 27.59; verify 28.88 / 43.59 / 57.25 / 70.02 / 81.98
The weights are read once per forward (multi-row matvec), so the growth with context points at attention for q_len 2–6 (and/or the GDN T-token path). This is the main reason MTP's margin falls from ~1.6x at 2k to ~1.04x at 80–110k on the real SWE-bench instance.

- [x] Profile the verify forward at 32k/100k by layer kind: full attention (SDPA with q_len 2–6 + mask) vs GDN recurrence vs matmuls. Check whether MLX's SDPA uses the vector kernel for small q_len and whether the causal mask for S×L is materialized.
- [x] Fix the dominant cost (e.g. a small-q_len SDPA vector path, no materialized mask, fused GDN T-token step) with output identical to the current verify up to the bf16 near-tie rule.
- [x] Re-run MTPRoundCostBreakdownTests and the 2k–100k sweep (MTPSWEBenchScaleTests) before/after; target per-extra-row cost at 100k well under 5 ms. (Re-run done. Target NOT met: text 100k per extra row is 10.6 → 7.4 ms, and the router path stays at ~13 ms. Follow-ups are filed for the FF/GDN matvec, the router array mask and a leaner verify kernel.)

#mtp #perf