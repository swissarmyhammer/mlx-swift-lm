---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84drb71yg17sww2b64t8pcs
  text: |-
    ## Feasibility notes (mlx-swift 0.33.0)
    - `MLXFast.scaledDotProductAttention` takes `sinks`, but as one logit **per head** (`[H]`). Inkling's sink is per query per head (`[B, L, Hq]`, from `sinkProj`), and SDPA returns no LSE.
    - Far part: pass `keys[..<S-1024]` with `mask: .causal`. MLX aligns causal masks bottom-right, so query i sees keys ≤ S−L+i−1024, exactly the distance-≥1024 set. Rows with no far keys need a sink, or they produce NaN.
    - To recover the far denominator without an LSE: give SDPA a per-head sink c, and V an extra channel of ones (head dim padded 128→256). Then O_extra = D_far/(D_far+e^c), so D_far = e^c·O_extra/(1−O_extra). That cancels badly when D_far ≫ e^c, and the padding doubles the PV work and the V reads.
    - Cleaner options: (a) an upstream MLX change returning the LSE from the steel attention kernel; (b) port the steel/NAX tiling into `relativeAttentionSource`.
    - Either way, the summation order on long prompts changes. Any golden divergence must be shown to be a near tie.
    - Upside: SDPA is 22 ms against ~78 ms per 2048×64k chunk, about +40% prefill at 64k. Left open for the user to choose an approach.
  timestamp: 2026-11-23T14:42:56.865189+00:00
position_column: todo
position_ordinal: 8c80
title: 'Inkling 2 prefill: full-attention far keys through MLXFast SDPA, near band through the relative kernel'
---
## Why
^4nfm0y0: at long contexts, prefill time is the full-attention relative kernel, at ~21 TFLOP/s. `MLXFast.scaledDotProductAttention` runs the same L=2048 × S=64k shape in 22 ms (~74 TFLOP/s), against ~78 ms. Estimated gain: +40% prefill at 64k, ~+20% at 32k.

## What
In a full layer the relative bias is zero past distance R=1024 (`rel_extent`).
- Far part: keys at distance ≥ 1024 have no bias. Run them through SDPA (causal with an offset, or an explicit key range).
- Near band: d < 1024 plus the sink, through the existing kernel, changed to also return each row's LSE.
- Merge the two by their log-sum-exps.

Open problem: MLX's SDPA returns no LSE. Options:
- (a) An upstream MLX change that exposes the LSE.
- (b) The sinks argument plus an extra value channel of ones, to recover the denominator.
- (c) Port the steel attention tiling (NAX) into the relative kernel.

## Acceptance
- [ ] The output matches the current kernel within bf16 tolerance. Any golden divergence must be shown to be a near tie.
- [ ] inklingbench shows a prefill gain at 32k and 64k. #inkling-perf