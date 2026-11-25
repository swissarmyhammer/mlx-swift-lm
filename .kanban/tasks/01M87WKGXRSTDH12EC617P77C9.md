---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m87y1a07fzqvr1q1nskhv8vs
  text: |-
    Input from the 32-step 2-Mac runs:
    - Qwen3.8-27B: KL < 0.002 at every step except step 31 (0.025).
    - Inkling Mini: KL ≤ 0.005, but the top-10 logprob diff grows steadily, ~0.07 → ~0.51 over 32 steps.
    - Both suggest slow drift. Check 256 and 1024 steps.
  timestamp: 2026-11-24T23:25:11.047074+00:00
- actor: claude-code
  id: 01m884k9k34pd2mxtwajnyysmp
  text: |-
    ### Long-run results (2-Mac JACCL, m5u2↔m5u1, cluster 59883cbe, teacher-forced, FAST_SYNCH=1) — no drift
    **Qwen3.8-27B-mxfp4**
    - KL max/mean by window: 0–32 0.0030/0.0006; 33–128 0.0170/0.0008; 129–256 0.0171/0.0006; 257–512 0.0267/0.0005; 513–1024 0.0177/0.0007. Flat.
    - Argmax 1023/1025, 1 tie skipped (d105, margin 0.125).
    - 1 mismatch at d140: margin 0.375 (3 bf16 ulps), KL 0.017, inside a degenerate run of special tokens 248045/248046.
    - Speed: 17.2 ms/token one Mac vs **13.1 two Macs**, holding over 1024 steps.

    **Inkling-2-Mini-oQ4e-mtp**
    - 256 steps: PASS, max KL 0.0298.
    - 1024 steps, KL max/mean by window: 0–32 0.0164/0.0024; 33–128 0.0298/0.0021; 129–256 0.0157/0.0012; 257–512 0.0764/0.0011; 513–1024 0.0244/0.0004. Means fall over time.
    - One isolated outlier at d369, KL 0.0764 (argmax agrees; neighbours ≤ 0.0016).
    - Argmax 1020/1025, 5 ties skipped.
    - Speed: 11.7 ms/token one Mac vs 15.5 two Macs (stays local).

    **Conclusion:** no accumulating drift. Sharding error is bf16 partial-sum noise with rare single-step outliers. The current per-step gate (every step ≤ 0.05, ties ≤ 2 ulps) is right for short checks but trips on 1-in-1000 outliers in long runs. Follow-up: a long-run gate mode on the comparator.
  timestamp: 2026-11-25T01:19:51.907526+00:00
depends_on:
- 01M87HW2SS24TQTQ9HFE8SMZKZ
position_column: done
position_ordinal: fffff480
title: 'Long-run accuracy check: 2-Mac JACCL, 256+ decode steps'
---
On a 32-step Qwen3.8-27B run over JACCL, KL vs the single-Mac reference was < 0.002 at every step except step 31 (0.025), and the argmax matched at every step. Check whether KL drifts upward over longer generations, e.g. if GatedDelta recurrent state accumulates sharding round-off.

- [ ] ShardCheck: a `--decode-steps 256` (and 1024) run of Qwen3.8-27B-4bit and Inkling-2-Mini-4bit on m5u2 ↔ m5u1 with MLX_METAL_FAST_SYNCH=1. The cluster session has the setup in /tmp/fmtest on m5u1.
- [ ] Plot or print KL per step. If it trends up, find which layer type drives it (shard only attention vs only GatedDelta) and whether float32 accumulation in the allSum fixes it.
- [ ] Record the numbers on this task and in cluster.md.

Depends on e8smzkz (Inkling sharding). #cluster