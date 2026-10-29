---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: 'MTP: cut the per-row cost of the multi-row verify at long context (5-row verify is 2.3–3.0x a plain forward at 32k–100k)'
---
Step-4 breakdown (^j2tnzpn, Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, guard off): a 1-row verify costs about a plain forward, but each extra row adds ~5–7 ms at 32k and ~10–13 ms at 100k context:
- text 32k: plain 21.44 ms; verify rows 1–5: 21.85 / 26.59 / 35.33 / 42.48 / 48.95
- text 100k: plain 26.42; verify 27.43 / 36.31 / 48.60 / 58.32 / 68.92
- router 100k: plain 27.59; verify 28.88 / 43.59 / 57.25 / 70.02 / 81.98
The weights are read once per forward (multi-row matvec), so the growth with context points at attention for q_len 2–6 (and/or the GDN T-token path). This is the main reason MTP's margin falls from ~1.6x at 2k to ~1.04x at 80–110k on the real SWE-bench instance.

- [ ] Profile the verify forward at 32k/100k by layer kind: full attention (SDPA with q_len 2–6 + mask) vs GDN recurrence vs matmuls. Check whether MLX's SDPA uses the vector kernel for small q_len and whether the causal mask for S×L is materialized.
- [ ] Fix the dominant cost (e.g. a small-q_len SDPA vector path, no materialized mask, fused GDN T-token step) with output identical to the current verify up to the bf16 near-tie rule.
- [ ] Re-run MTPRoundCostBreakdownTests and the 2k–100k sweep (MTPSWEBenchScaleTests) before/after; target per-extra-row cost at 100k well under 5 ms.

#mtp #perf