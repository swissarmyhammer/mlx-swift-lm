---
assignees:
- claude-code
position_column: done
position_ordinal: '9880'
title: 'MTP: profile the verify block pass and explain why the 100k round barely moves'
---
Follows ^mm16fx2 (row packing in VerifyAttentionKernel, `fewestQueries` 2, 2048 keys per block, 4..16 blocks). The ABBA 8/8 router-path round cost B−A was 2k ~−1.1 ms (−3%), 32k ~−2.8 ms (−7%), 100k ~−0.6 ms (within noise). The microbench predicted ~8 ms less attention at 100k.

- [x] Profile the block pass (Metal capture or counters): occupancy, threadgroup memory, MMA tile loads vs the decode kernel's SIMD-per-key design.
- [x] Explain the gap at 100k: the microbench predicted ~8 ms less attention, the real round dropped ~0.6 ms.
- [x] Explain the acceptance shift between base and change: tokens per verify 2k 2.66 vs 2.87, 100k 3.38 vs 2.80, which confounds tok/s.

#mtp #perf

## Result 2026-11-03 (7fe13261, c4493248; worktree 0f2f15bf) — kernel unchanged
- Block pass profile, 16 layers, M5 Ultra (MLXLMTests.VerifyAttentionBlockPassProfileTests, opt-in MLX_VERIFY_ATTENTION_PROFILE=1). 100k, b16 (current): 7.58/10.33/10.98/12.58 ms for 2/3/4/5 rows vs one decode step 7.59 ms (864 GB/s over 6.55 GB K/V) and SDPA 13.3/19.1/25.7/29.7. 32k 2/4 rows: 3.12/4.04 vs SDPA 4.87/8.48. b32/b64/b128 equal or slower; combine pass +0.1–0.4 ms. 64 threadgroups (16 blocks x 4 KV heads), ~27–30 KB threadgroup memory each. 2-row verify is bandwidth bound; rows 3–5 cost 1.36–1.66x a decode step.
- 100k gap: the ~5–7 ms/verify saving is real (pre-packing kernel ~17.8 ms). The ABBA arms decoded different text: at 100k the change averaged 3.45 rows/verify vs 3.00, so the saving went to the extra draft steps and verify rows (52.92 vs 53.41 ms/round).
- Acceptance shift: text divergence after near ties caused by the verify numerics. An in-process rule switch reproduces both arms' tokens/verify to every digit (base 2.8683/3.0937/2.7978, current 2.6597/2.9510/3.3795 at 2k/32k/100k). Row packing is bit-identical per row; the block split and the 2-row route change the logits. First divergence vs plain at exact bf16 ties (2k both arms, 32k base) or a 0.125 near tie (32k current); 100k gaps not read (reference prefill hit the run timeout).
- Possible follow-up, not filed: route verifies under ~4096 keys to SDPA (≤0.5 ms per 2k round, inconsistent under load).
- review: worktree commit 2 findings (print in the profile) fixed in c4493248 with swiftlint disable comments + reason; re-review HEAD~2..HEAD 0 findings. Five bundles green in the worktree (MLXLMTests 930 + 2120, 1 known issue); merged diff adds only comments.
