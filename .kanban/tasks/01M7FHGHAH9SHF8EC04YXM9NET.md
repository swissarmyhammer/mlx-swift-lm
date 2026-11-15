---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7gj84q3zn3x50cxth1f66w1
  text: |-
    Research / rollback (step 2), stable HEAD 2c68203d.

    Can the repo roll back rejected drafts on a wrapped RotatingKVCache? YES, three ways, so rollback is not a prerequisite cost:
    1. A multi-row write into a wrapped ring with no reserve goes through `RotatingKVCache.updateConcat` (Libraries/MLXLMCommon/RotatingKVCache.swift). It linearizes the ring and keeps window-1+L rows, so `maxTrimCount` = L-1 and `trimPromptCache(cache, L-1)` rewinds exactly. Measured: 0 trim failures in 2x(65 reps x 5 widths) verify+trim cycles at 2k and 16k on Flint oQ4e. The "returns 0" case is only after single-token writes, where live == window < offset. `canRewindSpeculativeVerify` / `rewindSpeculativePromptCache` (Libraries/MLXLMCommon/KVCache.swift) encode this (`case is RotatingKVCache: positions > 1`).
    2. Staged rounds: `KVCacheStorage.beginRound/commit/rewindLastRound` (KVCachePlan.swift) with `RotatingRoundStrategy` (KVCacheRound.swift). The generic `SpeculativeTokenIterator` (TokenIterators.swift) uses these when the main cache is not trimmable.
    3. In-place ring reserve: `RotatingKVCache(rewindCapacity:)`, set by `GenerateParameters(kvCache: KVCacheConfiguration(rewind: .init(maxTokens:)))` through `makeSlidingWindowKVCache`. Flint's `newCache(parameters:)` already passes the parameters on. Same verify cost (verify_r8.txt).
    Caveats: `MTPSpeculativeTokenIterator.verifiesInPlace` requires a MambaCache in the cache, so the MTP in-place path does not apply to Flint. The parked origin/dflash2 iterator refuses rotating caches (`dflash2SupportsCache` accepts KVCacheSimple/MambaCache only), so a Flint integration would need the generic trim/round path.

    Discovery (explains the verify cliff): MLX `ScaledDotProductAttention` (mlx-swift Cmlx scaled_dot_product_attention.cpp, has_fused_kernel) refuses the vector kernel when `query_len * gqa_factor > 32`. Flint has 32 q heads and 4 kv heads, so GQA is 8 and every verify with L >= 5 drops to the unfused fallback. At 16k the time goes from 14.4 ms at L=4 to 25.9 ms at L=5.

    Drafter capture without a repo change: a `CaptureNorm: RMSNorm` subclass is swapped in through `Module.update(modules:)` as `layers[id+1].input_layernorm`. It records the output of layer id. The drafter is ported eagerly in scratchpad flintcli/Sources/flintcli/DFlashDraft.swift from origin/dflash2 DFlash2.swift. It uses Flint's embed_tokens and an f32 lm_head. Sanity check: math acceptance 3.56-4.41 tok/verify, against the published GSM8K 4.23.
  timestamp: 2026-11-15T21:36:40.163774+00:00
- actor: claude-code
  id: 01m7gj8xa0b69aabt2h7a6aar5
  text: |-
    ## Measurement: Flint oQ4e on M5, stable 2c68203d, greedy

    ### 1. Verify cost
    Setup: a warm cache filled from prompt_706 ids. Each forward takes L real tokens and ends in an eval-synced argmax. After it, trimPromptCache(L-1) runs. The figure is the median of 60 reps (p10-p90 within ±2%). Scratchpad `flintcli verify`, raw output in verify_r0.txt and verify_r8.txt.
    Plain pipelined decode (TokenIterator, bench): 6.29 ms/tok at 2k and 7.01 ms/tok at 16k. A synced L=1 forward takes 7.70 / 8.31 ms. The compiled one-token MoE path applies only at L=1.

    | L | 2k ms | R=t/t(1) | R vs pipelined | 16k ms | R=t/t(1) | R vs pipelined |
    |---|---|---|---|---|---|---|
    | 1 | 7.70 | 1.00 | 1.22 | 8.31 | 1.00 | 1.19 |
    | 2 | 9.74 | 1.26 | 1.55 | 11.06 | 1.33 | 1.58 |
    | 3* | 11.33 | 1.47 | 1.80 | 13.19 | 1.59 | 1.88 |
    | 4 | 11.98 | 1.56 | 1.90 | 14.36 | 1.73 | 2.05 |
    | 5* | 15.3 | 1.99 | 2.43 | 25.2 | 3.03 | 3.60 |
    | 6 | 16.12 | 2.09 | 2.56 | 25.95 | 3.12 | 3.70 |
    | 8 | 17.49 | 2.27 | 2.78 | 27.86 | 3.35 | 3.97 |
    (*) L=3 and L=5 come from the reserve-8 run, scaled by its L=1. The cliff between L=4 and L=5 is the MLX SDPA vector-kernel limit (q_len x GQA 8 > 32, so the unfused fallback runs). The DFlash2 block of 8 costs 2.8 plain steps at 2k and 4.0 at 16k. That matches the Qwen3.8 DFlash2 failure (~2.9).

    ### 2. Rollback
    YES, see the previous comment. A wrapped ring trims L-1 rows after a multi-row write. Staged rounds and the rewind reserve also work. It is not a prerequisite cost.

    ### 3. Drafter cost (incoai/Flint-DFlash2, 0.81 GB bf16, block 8, eager port)
    - One proposal (5 layers, a 2047-row context window, f32 lm_head on 7 rows, top-16 selector walk), eval-synced: 2.92 ms at short context and 3.38-3.84 ms at 18-20k.
    - Projection of the 8 new context rows per round: 0.30-0.34 ms.
    - With 4-bit drafter Linears: 2.75 ms short and 3.18-3.27 ms at 18-20k, with the same acceptance. The drafter is op-bound, not bandwidth-bound.
    - So the drafter costs about 0.5 plain steps (0.49-0.53).

    ### 4. Acceptance, offline
    Setup: Flint greedy continuation (TokenIterator, 256 tokens). The drafter reads Flint's real layer 5/15/26/36/46 hidden states. The cell is mean tokens per verify (accepted+1); drafts are truncated to k-1 for a verify of width k.

    | workload | k2 | k3 | k4 | k5 | k6 | k8 |
    |---|---|---|---|---|---|---|
    | math (ctx 36) | 1.79 | 2.33 | 2.69 | 2.98 | 3.28 | 3.56 |
    | code (30) | 1.74 | 2.12 | 2.46 | 2.61 | 2.67 | 2.78 |
    | prose (18) | 1.70 | 2.08 | 2.42 | 2.56 | 2.64 | 2.72 |
    | agent cut 18107 | 1.66 | 2.00 | 2.17 | 2.23 | 2.23 | 2.23 |
    | agent cut 18673 | 1.77 | 2.19 | 2.51 | 2.64 | 2.69 | 2.75 |
    | agent full 19728 | 1.71 | 2.12 | 2.37 | 2.56 | 2.67 | 2.81 |

    Speedup = (accepted+1) / (R_pipelined(k) + drafter), as a geometric mean over the 3 agent cuts:
    - bf16 drafter: k2 0.81, k3 0.87, k4 0.91, k6 0.60, k8 0.57.
    - Drafter cost 0: k2 1.09, k3 1.12, k4 1.15, k8 0.65.
    - Short prompts, bf16 drafter: best k4, at 1.11 math, 1.02 code and 1.00 prose. With drafter cost 0: 1.41 / 1.29 / 1.27.
    - If the SDPA L>4 fallback were fixed (t(8) at 16k ≈ 17.4 ms, R ≈ 2.5): agent k8 is still 0.87, or 1.05 with drafter cost 0.

    ### 5. Prompt-lookup (n-gram), no drafter cost
    Setup: the same greedy sequences (scratchpad pyscripts/ngram_sim.py, dfl_bf16_seqs.txt). The lookup takes the most recent earlier match of the longest suffix n-gram and drafts the k tokens that follow it. A round with no match is a plain step.
    - Best setting: n in 4..3, k=3 drafts (L=4 verify, below the cliff).
      - Agent results 0.90 / 1.47 / 1.31, geomean 1.20, if a step with no match keeps the pipelined cost.
      - Geomean 1.10 (0.80 / 1.40 / 1.20) if every step must sync to look up the n-gram, which is the simple implementation.
      - Short prompts 1.01-1.09 (synced: 0.83-0.94).
    - k=7 drafts (L=8): agent geomean 1.02 (synced 0.95).
    - A low n (3..1) is worse: frequent wrong matches at about 2 steps per verify.
    - The gain is all in tool-call/arguments copying: cut 18673 echoes the task text. On reasoning text the n-gram drafts lose (0.80-0.90).

    ## Go / no-go: NO-GO
    - DFlash2 on the agent transcript is 0.57-0.91x at every verify width. Even with a drafter of zero cost it is at most 1.15x (k4), below the 1.2x bar.
    - With the SDPA cliff fixed, block 8 is still at most 1.05x.
    - The causes: an MoE verify costs 1.9x (L=4) to 4.0x (L=8) a pipelined step at 16k, the drafter costs about 0.5 step, and agent acceptance is only 2.2-2.8 tokens per verify.
    - N-gram lookup reaches 1.10-1.20x on average, but that depends on workload (0.80-1.47), so it is not clearly above 1.2x either.
    - No follow-up integration card is filed.
    - Prerequisites if this is ever revisited:
      - (a) A vector SDPA for q_len x GQA > 32, for example 4-query groups. This is an MLX kernel limit.
      - (b) A cheaper MoE multi-token verify; L=4 costs 1.9-2.05 steps.
  timestamp: 2026-11-15T21:37:05.344672+00:00
- actor: claude-code
  id: 01m7gj96t5xmr769v5rstjthzp
  text: |-
    ### implement — no-change
    - evidence: no repo file changed (git status: .kanban only). Measurement code is in the scratchpad: flintcli modes `verify` and `dflash`, flintcli/Sources/flintcli/DFlashDraft.swift, and pyscripts/ngram_sim.py. Raw output: verify_r0.txt, verify_r8.txt, dfl_bf16.txt, dfl_q4.txt, dfl_bf16_seqs.txt.
    - R vs pipelined step:
      - 2k: L2 1.55, L4 1.90, L8 2.78.
      - 16k: L2 1.58, L4 2.05, L8 3.97.
      - The cliff at L>=5 is the MLX SDPA q_len x GQA > 32 fallback.
    - Rollback on a wrapped RotatingKVCache: yes (multi-row write then trim L-1, staged rounds, or the rewind reserve).
    - Drafter cost: 2.9-3.8 ms bf16, 2.75-3.3 ms 4-bit, about 0.5 step.
    - Agent acceptance: 2.2-2.8 tok/verify at k8 and 2.2-2.5 at k4.
    - DFlash2 agent speedup is 0.57-0.91x, and at most 1.15x even with a drafter of zero cost.
    - N-gram agent speedup is 1.10-1.20x geomean (0.80-1.47).
    - Verdict: NO-GO, so no follow-up card is filed.
    - Caveat: the cut-19728 continuation runs past `<|user|>`, because TokenIterator does not stop at EOS. Its acceptance includes some text after the turn.
    - next: user decides (review, or close as measured no-go).
  timestamp: 2026-11-15T21:37:15.077103+00:00
- actor: claude-code
  id: 01m7gj9gfzjg24beayspmex8ze
  text: |-
    Break-even acceptance. Tokens per verify needed = 1.2 x (R_pipelined(k) + drafter 0.5 step) for 1.2x, or R_pipelined(k) + 0.5 for 1.0x:

    | | 1.0x | 1.2x | measured |
    |---|---|---|---|
    | 16k, k4 | 2.58 | 3.09 | agent 2.17-2.51 |
    | 16k, k8 | 4.50 | 5.4 | agent 2.23-2.81 |
    | 2k, k4 | 2.42 | 2.9 | math 2.69, code 2.46, prose 2.42 |
    | 2k, k8 | 3.30 | 3.96 | math 3.56 |

    For n-gram lookup (no drafter cost) at 16k, the 1.0x bar is 2.05 for k=3 drafts (L=4). The measured value per lookup hit, accepted+1, is 1.4-3.4.
  timestamp: 2026-11-15T21:37:24.991762+00:00
- actor: claude-code
  id: 01m7gjax54ste2fkfke5fg275q
  text: |-
    ### commit — changed
    - evidence: commit subject "chore(kanban): record the Flint DFlash2 and prompt-lookup go/no-go measurements (^yxm9net)"; files: .kanban/tasks/01M7FHGHAH9SHF8EC04YXM9NET.{jsonl,md}, .kanban/tasks/01M7GCNVYNJNS73KPQGTWC3ZH6.{jsonl,md}
    - next: none (local commit only, not pushed)
  timestamp: 2026-11-15T21:38:10.724567+00:00
depends_on:
- 01M7FHFQT76AN9817CJJ4PQTWR
- 01M7FHFWPMBDPGDSD9BG68BHJ1
- 01M7FHG1KPDNCWYR3DCPG2HSB8
- 01M7FHG7TA2MGDAS4Q5NANQJ4P
position_column: doing
position_ordinal: '80'
title: 'Flint: measure whether DFlash2 speculative decoding can pay off (8-token verify cost)'
---
## Why
Flint has no MTP head. `incoai/Flint-DFlash2` exists (block 8, target layers 5/15/26/36/46). On the M5, DFlash2 for Qwen3.8-27B lost because a round cost ~2.9 plain steps (see memory `dflash2-branch-status`; branch `origin/dflash2` is parked). An 8-token MoE verify touches up to 48 distinct experts per layer instead of 6, so it may cost far more than a 1-token step.

## What
- Do this after the fusion, cast, compile and sliding-attention cards land, so the baseline step is the fast one.
- Measure the wall time of an 8-token verify forward against a 1-token step on oQ4e, at 2k and 16k context.
- Measure the drafter forward cost.
- Estimate break-even acceptance. Build the integration only if the measured round cost leaves real headroom.

## Acceptance
- A written measurement on this card: verify/step ratio, drafter cost, and a go/no-go.
- If go: a follow-up card for the integration with ChatSession and the prompt cache, per memory `goal-speed-with-cache-reuse`. #flint-perf