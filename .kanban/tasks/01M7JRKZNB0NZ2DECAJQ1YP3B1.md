---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7k72s6zcsgxfkdr8hkecvww
  text: |-
    Research:
    - `fewestRows` is read by `canRun` only; `projection(_:by:)` takes the kernel inside `run(asSpeculativeVerify:)` scope, so 2-row calls outside a verify (prefill tail, plain decode, drafter) keep MLX. Callers: Qwen35 VLM/LLM attention/GDN/MLP/lm_head projections and `FusedQuantizedProjectionGroup` (vectorCalls < 6 rows) — no other row gate.
    - Tests derived from `fewestRows` (MultiRowProjectionFixture.rows / kernelVerifyTokenCounts, Qwen3NextMultiRowMLPTests.verifyRows, ExactProduct rowCounts, bitIdenticalRows) follow the constant automatically; text that says "3 to 8" lives in MultiRowQuantizedMatvecTests header + test name, Qwen35(VLM)MultiRowProjectionTests headers. IntegrationTesting benchmark's `qmvWideRows`/`qmvWideTolerance`/`relativeOutputError` become dead (every measured row count bit-exact).
    - `CompiledVerifyShape.fewestTokens` (= 2) names the shortest verify; the new RED test uses it.
    - `MTPChatTokenMatch` doc (BenchmarkHelpers) says the 4-bit matmul gives two rows different bits than one — no longer true inside a verify after this change.
    - mn44v3n golden: MTP ids already differ from plain ids at short@114 and 32k@82; code/toolchat MTP == plain. After this change MTP should move toward plain.
    - Baselines at HEAD 9ac70903 being measured: verify_before_q1yp3b1.txt, qwen_bench_head_q1yp3b1.txt (scratchpad).
  timestamp: 2026-11-16T22:19:13.503576+00:00
- actor: claude-code
  id: 01m7k92tn45ffr5hgxps9mn6an
  text: |-
    Landed (uncommitted): `MultiRowQuantizedMatvec.fewestRows = 2` with docs (2 to 8 rows everywhere), RED/GREEN test `testTheShortestVerifyRunsTheKernel` (canRun for `CompiledVerifyShape.fewestTokens` rows; failed before, passes after), renamed `testTheKernelRunsTwoToEightRowsAndLeavesOtherCountsToMLX`, header text in Qwen35(VLM)MultiRowProjectionTests, IntegrationTesting benchmark: dropped `qmvWideRows` / `qmvWideTolerance` / `relativeOutputError` (every measured row count now bit-exact). Fixture-derived tests follow the constant. `MTPChatTokenMatch` doc no longer claims the verify's 4-bit matmuls differ for two rows.

    **L=2 verify microbench (`qwencli verify`, mtp-verify-state, 15 reps median)** — before (HEAD 9ac70903) → after:
    - 2k: 21.10 → 21.77 ms (+0.67, +3.2%)
    - 32k: 23.87 → 24.18 ms (+0.31, +1.3%)
    - 64k: 27.28 → 27.77 ms (+0.49, +1.8%)
    L=1/3/4 unchanged within noise. So end to end a 2-row verify is SLOWER through the kernel than through `qmv_wide`, despite the 14.6 vs 17.6 ms step-matmul microbench of ^ets0cjp. (Files: scratchpad verify_before_q1yp3b1.txt / verify_after_q1yp3b1.txt.)

    **Golden decision (orchestrator policy: near-ties < 0.5 only)** — qwencheck vs qwen_golden_mn44v3n.json: plain all MATCH; code/toolchat MTP MATCH; short MTP DIVERGE at step 114 (12680 vs 947) — new MTP ids are now identical to plain over all 257 ids (moved toward plain greedy); 32k MTP DIVERGE at step 1 (1892 vs 11), which moves away from plain (11). `qwencli neartie` (plain TokenIterator, teacher-forced): 32k step 1 top1 11 = 26.75, top2 1892 = 26.75, gap 0.0 (exact bf16 tie); next divergence from plain at step 7: gap 0.125 (2887 vs 5236 on plain ids; 76951 vs 2887 on the new MTP ids). Every divergence is a near-tie or matches plain → new golden written to qwen_golden_q1yp3b1.json (read-only, + .session/.toolchat), `qwencheck.sh` default BASE now points at it. qwen_golden_base.json and qwen_golden_mn44v3n.json untouched. SESSION PASS (warm turns cached 139/203 tok), TOOLCHAT plain+MTP PASS.

    **qwenbench (medians of 3)** — HEAD = qwen_bench_head_q1yp3b1.txt (measured this session), stable = qwen_bench_base.txt:
    - MTP ms/tok 2k/32k/64k: 9.66 / 13.31 / 13.49. vs HEAD 9.48 / 12.47 / 13.55: +1.9% / +6.7% / -0.4%. vs stable 10.17 / 13.02 / 14.20: -5.0% / +2.2% / -5.0%.
    - tokens/verify: 3.21 / 2.22 / 2.73 (HEAD 3.29 / 2.40 / 2.65): the greedy trajectory changed, so acceptance moved; cost per round (ms/tok × tokens/verify) 31.19→31.01 / 29.93→29.55 / 35.91→36.83 ms.
    - plain decode 17.16 / 19.84 / 22.73 (HEAD 17.13 / 19.84 / 22.77, same); vs stable -6.0% / -3.7% / -3.0%.
    - prefill 1460 / 1272 / 804 tok/s (HEAD 1469 / 1272 / 802, same); vs stable +9.4% / +10.4% / +11.4%.
    Net: the change buys bit-exact 2-row verifies (MTP == plain on short) but is not a speed win; the L=2 verify microbench is 1.3–3.2% slower. Decision for the user/orchestrator whether to keep it.
  timestamp: 2026-11-16T22:54:12.132329+00:00
- actor: claude-code
  id: 01m7k974rag3gf99d27crynf2e
  text: |-
    **NO-GO (orchestrator decision), reverted.** The batch bar is "faster with output preserved"; this change misses both parts.
    - Speed: L=2 mtp-verify-state microbench 21.10→21.77 / 23.87→24.18 / 27.28→27.77 ms at 2k/32k/64k (+3.2% / +1.3% / +1.8%). qwenbench MTP ms/tok vs HEAD 9.48→9.66 / 12.47→13.31 / 13.55→13.49 (+1.9% / +6.7% / −0.4%), tokens/verify 3.29→3.21 / 2.40→2.22 / 2.65→2.73. Plain decode and prefill unchanged.
    - Output: the MTP golden changes. Short MTP became identical to plain. 32k MTP moved AWAY from plain at step 1 (1892 vs 11, an exact bf16 tie, gap 0.0).
    - The ^ets0cjp step-matmul microbench win (2 rows: 14.6 ms through the kernel vs 17.6 ms through `qmv_wide`) did NOT carry over end to end: the whole 2-row verify got slower.
    - Reverted: `git checkout -- Libraries Tests IntegrationTesting`. `git status` shows only .kanban. `fewestRows` stays 3.
    - Golden: scratchpad `qwencheck.sh` default BASE is back on `qwen_golden_mn44v3n.json`. `qwen_golden_q1yp3b1.json` (read-only, plus .session/.toolchat) stays as a record only. qwen_golden_base.json and qwen_golden_mn44v3n.json were never touched.
    - Test run stopped at the orchestrator's request before the five bundles finished. Not needed: the tree is back at HEAD.
    - Side finding, filed as ^6gkhfah: the IntegrationTesting scheme does not build on qwen-perf at HEAD (DeepseekV4AgenticPromptCacheAssessmentTests: switch must be exhaustive; Qwen38FourBitMTPModel: ContainerLoader conversion).
    - Records in the scratchpad: verify_before_q1yp3b1.txt, verify_after_q1yp3b1.txt, qwen_bench_head_q1yp3b1.txt, qwen_bench_q1yp3b1.txt, qwencheck_q1yp3b1.log, q1yp3b1_32k_{plain,mtp}_ids.txt.
  timestamp: 2026-11-16T22:56:33.546423+00:00
- actor: claude-code
  id: 01m7k97766zm1xfv0gcnvfskad
  text: |-
    ### implement — no-change
    - evidence: change reverted on orchestrator NO-GO; `git status` shows only .kanban. Measured before revert: L=2 verify +3.2/+1.3/+1.8%, MTP ms/tok vs HEAD +1.9/+6.7/−0.4%; 32k MTP golden moved off plain at step 1 (tie, gap 0.0). qwencheck BASE restored to qwen_golden_mn44v3n.json; qwen_golden_q1yp3b1.json kept as a record.
    - next: orchestrator/user decides the card's fate (close as no-go). Follow-up ^6gkhfah: IntegrationTesting build errors at HEAD.
  timestamp: 2026-11-16T22:56:36.038180+00:00
depends_on:
- 01M7J840QAGWRNMGV5HETS0CJP
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 MTP verify: run 2-row verifies through the bit-exact multi-row kernel'
---
## Why
^ets0cjp rebuilt `MultiRowQuantizedMatvec` (inputs staged in threadgroup memory across 8 SIMD groups, next-block weight prefetch, half2 nibble decode). In the step matmul microbench ($S/mrbench, Qwen3.8-27B 4-bit g64, M5 Ultra) the new kernel runs 2 rows in 14.6 ms, bit-exact, while MLX `qmv_wide` takes 17.6 ms and differs from a row alone in the last bits. `fewestRows` is still 3 because the old kernel made the 2-token MTP verify ~11% slower end to end.

## What
- Set `MultiRowQuantizedMatvec.fewestRows` to 2 and update its doc comment and the tests that assume 3 (`MultiRowQuantizedMatvecTests`, `MultiRowProjectionFixture`, `Qwen3NextMultiRowMLPTests`, `MultiRowQuantizedMatvecExactProductTests`, the IntegrationTesting benchmark's `qmvWideRows`).
- The MTP greedy ids of L=2 verifies will change (they become those of a plain decode). Regenerate the MTP golden on purpose and record why, or confirm the ids stay identical.

## Acceptance
- [ ] `qwencli verify` L=2 mtp-verify-state before/after at 2k/32k/64k.
- [ ] `qwencheck.sh` passes (golden policy decided and recorded); parity tests exact; all five bundles green.
- [ ] `qwenbench.sh` delta on the card. #qwen-perf